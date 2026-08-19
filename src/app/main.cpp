// tglab — CV/CG algorithm research lab.
//
// M1 vertical slice: load an image, run a script that calls an algorithm with
// slider-driven parameters, display the result. Everything on the UI thread.
#include <windows.h>
#include <shellapi.h>   // DragAcceptFiles / DragQueryFile

#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "../core/algorithm.h"
#include "../core/image_io.h"
#include "../core/compare.h"
#include "../core/pipeline.h"
#include "../core/worker.h"
#include "../script/interp.h"
#include "../script/parser.h"
#include "../gpu/device.h"
#include "file_watch.h"
#include "image_view.h"

#include "imgui.h"
#include "imgui_internal.h"   // DockBuilder* for the default layout
#include "backends/imgui_impl_dx12.h"
#include "backends/imgui_impl_win32.h"

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND, UINT, WPARAM, LPARAM);

namespace tglab {
namespace {

// Mirrors Device's tracing; set TGLAB_VERBOSE=1.
void AppTrace(const char* stage) {
    static const bool on = GetEnvironmentVariableA("TGLAB_VERBOSE", nullptr, 0) > 0;
    if (!on) return;
    std::fprintf(stderr, "[tglab/app] %s\n", stage);
    std::fflush(stderr);
    OutputDebugStringA(stage);
}

struct PaletteEntry {
    std::string name;
    std::string path;
    Data        data;
};

class App {
public:
    bool Init(HWND hwnd);
    void Shutdown();
    void Frame();
    void OnResize(UINT w, UINT h) { if (m_dev.Ready()) m_dev.OnResize(w, h); }

    void LoadImageIntoPalette(const std::string& path);
    void SetScriptPath(const std::string& path);

private:
    void RunScript();
    void BuildDefaultLayout(ImGuiID dockspace);
    void DrawMenuBar();
    void DrawControlsPanel();
    void DrawPalettePanel();
    void DrawAlgorithmsPanel();
    void DrawScriptsPanel();
    void RescanScripts();
    void DrawComparePanel();
    void RequestCompare();
    void SyncViews();
    void DockLooseViewers();
    void ReportError(const std::string& prevError);
    void PollWorker();
    void UpdateWindowTitle();
    ImGuiID CentreDockNode() const;

    static bool        IsModified(const UiControl& c);
    static void        ResetControl(UiControl& c);
    static std::string DefaultText(const UiControl& c);

    HWND        m_hwnd = nullptr;
    Device      m_dev;
    FileWatch   m_watch;
    std::string m_scriptPath;
    std::string m_source;
    std::string m_shownTitle;   // avoids SetWindowText on every frame

    UiState        m_ui;
    PipelineWorker m_worker;
    std::vector<PaletteEntry> m_palette;

    // Viewer names come from phase 1 (immediately); images arrive later from
    // the worker. Keeping them separate lets panels appear and dock straight
    // away rather than popping in when the first result lands.
    std::vector<std::string> m_viewerNames;
    std::vector<ViewerImage> m_viewerImages;
    uint64_t                 m_pendingSeq = 0;
    uint64_t                 m_shownSeq   = 0;

    std::vector<std::unique_ptr<ImageViewPanel>> m_views;
    ViewCamera m_sharedCam;
    bool       m_syncCameras = true;

    std::string m_error;        // last script/run error, empty when fine
    bool        m_dirty = true; // re-run requested
    uint64_t    m_contentVersion = 1;
    bool        m_rebuildLayout = false;   // explicit "Reset layout" request
    int         m_layoutCountdown = 0;     // frames until the default layout is built
    bool        m_haveSavedLayout = false; // a tglab_layout.ini already existed
    ImGuiID     m_centreNode = 0;          // where new viewers get docked
    ImGuiID     m_dockspaceId = 0;         // fallback when a saved layout exists
    ImGuiID     m_compareNode = 0;         // right-hand column for compare
    int         m_spinner = 0;             // "working..." animation

    // Compare mode (M4). TGLAB_COMPARE=1 opens the panel at startup, which is
    // how the panel gets verified without driving the menus.
    std::shared_ptr<CompareResult> m_compare;
    GpuTexture m_diffTex;
    bool       m_compareOpen       = false;
    int        m_compareStage      = -1;   // -1 = last stage
    uint64_t   m_compareVersion    = 1;
    bool       m_compareRequested  = false;
    std::vector<std::string> m_stageNames;
    std::vector<bool>        m_stageGpuCapable;

    // Script browser: the .tgl files sitting beside the current script.
    std::string              m_scriptDir;
    std::string              m_scriptName;
    std::vector<std::string> m_scriptList;
    bool                     m_scriptListDirty = true;
};

App* g_app = nullptr;

bool App::Init(HWND hwnd) {
    m_hwnd = hwnd;
    if (!m_dev.Init(hwnd, /*enableDebugLayer=*/
#ifdef _DEBUG
                    true
#else
                    false
#endif
                    ))
        return false;

    // After the device exists: the worker builds its own compute context (and
    // its own compute queue) from it, on the worker thread.
    m_worker.Start(m_dev.Get());

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.IniFilename = "tglab_layout.ini";   // remembers panel layout across runs
    ImGui::StyleColorsDark();

    // If the user already has a layout, never overwrite it with the default.
    m_haveSavedLayout = (GetFileAttributesA(io.IniFilename) != INVALID_FILE_ATTRIBUTES);

    // Build the default arrangement unless the user has a saved one. Deferred
    // by one frame: DockBuilderDockWindow only binds windows ImGui has already
    // seen, and none of the panels exist until they have been drawn once.
    // Doing this here rather than when viewers appear means the panels are
    // docked even when the script fails before declaring any -- which is the
    // state you land in launching with no arguments.
    m_layoutCountdown = m_haveSavedLayout ? 0 : 2;

    AppTrace("imgui context created");
    ImGui_ImplWin32_Init(hwnd);
    AppTrace("win32 backend init");

    // The app owns SRV descriptor allocation; the backend calls these.
    ImGui_ImplDX12_InitInfo info = {};
    info.Device            = m_dev.Get();
    info.CommandQueue      = m_dev.Queue();
    info.NumFramesInFlight = kNumFramesInFlight;
    info.RTVFormat         = DXGI_FORMAT_R8G8B8A8_UNORM;
    info.DSVFormat         = DXGI_FORMAT_UNKNOWN;
    info.UserData          = &m_dev;
    info.SrvDescriptorHeap = m_dev.Srv().Heap();
    info.SrvDescriptorAllocFn = [](ImGui_ImplDX12_InitInfo* i,
                                   D3D12_CPU_DESCRIPTOR_HANDLE* cpu,
                                   D3D12_GPU_DESCRIPTOR_HANDLE* gpu) {
        static_cast<Device*>(i->UserData)->Srv().Alloc(cpu, gpu);
    };
    info.SrvDescriptorFreeFn = [](ImGui_ImplDX12_InitInfo* i,
                                  D3D12_CPU_DESCRIPTOR_HANDLE cpu,
                                  D3D12_GPU_DESCRIPTOR_HANDLE gpu) {
        static_cast<Device*>(i->UserData)->Srv().Free(cpu, gpu);
    };
    if (!ImGui_ImplDX12_Init(&info)) { AppTrace("dx12 backend init FAILED"); return false; }
    AppTrace("dx12 backend init ok");

    return true;
}

void App::Shutdown() {
    m_worker.Stop();   // before the device: results reference image memory
    m_dev.WaitForLastSubmittedFrame();

    // Every GpuTexture holds an SRV descriptor from the device's heap, so all
    // of them must be released BEFORE m_dev.Shutdown() destroys that heap.
    // Anything left to a member destructor would run afterwards and free a
    // descriptor against a dangling heap base.
    m_views.clear();
    m_diffTex.Release();

    ImGui_ImplDX12_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();
    m_dev.Shutdown();
}

void App::SetScriptPath(const std::string& path) {
    m_scriptPath = path;

    // Split for the script browser: it lists everything beside this file.
    {
        const size_t slash = path.find_last_of("/\\");
        if (slash == std::string::npos) { m_scriptDir = "."; m_scriptName = path; }
        else { m_scriptDir = path.substr(0, slash); m_scriptName = path.substr(slash + 1); }
        m_scriptListDirty = true;
    }
    m_watch.Watch(path);
    if (!ReadTextFile(path, &m_source))
        m_error = "could not read script '" + path + "'";
    m_dirty = true;
    UpdateWindowTitle();

    if (GetEnvironmentVariableA("TGLAB_COMPARE", nullptr, 0) > 0) m_compareOpen = true;
}

void App::LoadImageIntoPalette(const std::string& path) {
    Image img;
    std::string err;
    if (!LoadImageFile(path, &img, &err)) {
        m_error = err;
        return;
    }

    // Name is the filename without extension — what image("...") refers to.
    std::string name = path;
    if (auto slash = name.find_last_of("/\\"); slash != std::string::npos) name = name.substr(slash + 1);
    if (auto dot = name.find_last_of('.'); dot != std::string::npos) name = name.substr(0, dot);

    for (PaletteEntry& e : m_palette) {
        if (e.name == name) {           // replace in place on reload
            e.path = path;
            e.data = Data{std::move(img)};
            m_dirty = true;
            return;
        }
    }

    PaletteEntry e;
    e.name = std::move(name);
    e.path = path;
    e.data = Data{std::move(img)};
    m_palette.push_back(std::move(e));
    m_dirty = true;
}

// Phase 1 only: parse and interpret (microseconds), then hand the recorded
// pipeline to the worker. Interpret() is what declares sliders into m_ui, so
// keeping it here means UiState is never shared across threads.
void App::RunScript() {
    const std::string prevError = m_error;
    m_error.clear();

    // Sources are rebuilt each run so PortRef{-1, i} stays in step. The worker
    // gets its own copies, since it may still be reading them next frame.
    std::vector<Data> sources;
    std::vector<SourceImage> names;
    for (size_t i = 0; i < m_palette.size(); ++i) {
        if (std::holds_alternative<Image>(m_palette[i].data))
            sources.push_back(Data{std::get<Image>(m_palette[i].data).Clone()});
        else
            sources.push_back(Data{});
        names.push_back({m_palette[i].name, int(i)});
    }

    // Any failure here leaves the last good result on screen and reports the
    // reason; the viewers are only replaced when a run succeeds.
    Program prog;
    std::string err;
    if (!Parse(m_source, &prog, &err)) {
        m_error = err;
    } else {
        Pipeline built;
        InterpResult r = Interpret(prog, names, &m_ui, &built);
        if (!r.ok) {
            m_error = r.error;
        } else {
            // Viewer panels are created from the declarations now, so the
            // layout settles immediately rather than waiting for the result.
            m_viewerNames.clear();
            for (const ViewerDecl& d : built.Viewers()) m_viewerNames.push_back(d.name);
            SyncViews();

            // Stage list for the compare panel's picker.
            m_stageNames.clear();
            m_stageGpuCapable.clear();
            for (const Stage& s : built.Stages()) {
                m_stageNames.push_back(std::to_string(m_stageNames.size()) + ": " + s.algoName);
                m_stageGpuCapable.push_back(s.algo->HasGPU() && s.algo->GpuSource());
            }

            m_pendingSeq = m_worker.Submit(std::move(built), std::move(sources));
        }
    }

    ReportError(prevError);
    UpdateWindowTitle();
}

// Re-runs the script and asks the worker to compare CPU vs GPU on the final
// stage. Separate from the normal path because it is an explicit request, not
// something a slider change triggers.
void App::RequestCompare() {
    std::vector<Data> sources;
    std::vector<SourceImage> names;
    for (size_t i = 0; i < m_palette.size(); ++i) {
        if (std::holds_alternative<Image>(m_palette[i].data))
            sources.push_back(Data{std::get<Image>(m_palette[i].data).Clone()});
        else
            sources.push_back(Data{});
        names.push_back({m_palette[i].name, int(i)});
    }

    Program prog;
    std::string err;
    if (!Parse(m_source, &prog, &err)) { m_error = err; return; }

    Pipeline built;
    InterpResult r = Interpret(prog, names, &m_ui, &built);
    if (!r.ok) { m_error = r.error; return; }

    m_worker.SubmitCompare(std::move(built), std::move(sources), m_compareStage);
    m_compareOpen = true;
}

// Picks up a finished run. Called once per frame, before anything draws.
void App::PollWorker() {
    PipelineOutcome out;
    if (!m_worker.TryFetch(&out)) return;

    // A result older than the newest submitted job is stale: the user has
    // already moved the slider again, and a newer result is on its way.
    if (out.seq < m_shownSeq) return;
    m_shownSeq = out.seq;

    const std::string prevError = m_error;

    if (out.isCompare) {
        m_error = out.ok ? std::string() : out.error;
        m_compare = std::move(out.compare);
        ++m_compareVersion;
        ReportError(prevError);
        UpdateWindowTitle();
        return;
    }

    if (!out.ok) {
        m_error = out.error;          // keep the previous images on screen
    } else {
        m_error.clear();
        m_viewerImages = std::move(out.viewers);
        ++m_contentVersion;           // tells the views their pixels are new
    }
    ReportError(prevError);
    UpdateWindowTitle();
}

// "tglab — edges.tgl", with an [error] marker when the script is failing, so
// a broken hot reload is visible from the taskbar without focusing the window.
void App::UpdateWindowTitle() {
    if (!m_hwnd) return;

    std::string name = m_scriptPath;
    if (auto slash = name.find_last_of("/\\"); slash != std::string::npos)
        name = name.substr(slash + 1);

    // ASCII only: the title goes through SetWindowTextA, so a UTF-8 em-dash
    // would arrive mojibaked ("â€”") under a non-UTF-8 code page.
    std::string title = "tglab";
    if (!name.empty()) title += " - " + name;
    if (!m_error.empty()) title += "  [error]";

    if (title == m_shownTitle) return;   // SetWindowText every frame flickers
    m_shownTitle = title;
    SetWindowTextA(m_hwnd, title.c_str());
}

// Echo script errors to stderr as well as the Status panel. Running from a
// terminal is the natural way to debug a script, and errors that only appear
// in the UI are easy to miss.
void App::ReportError(const std::string& prevError) {
    if (m_error == prevError) return;   // only on change, not every frame
    if (!m_error.empty()) {
        std::fprintf(stderr, "[tglab] %s\n", m_error.c_str());
        std::fflush(stderr);
    } else if (!prevError.empty()) {
        std::fprintf(stderr, "[tglab] script ok\n");
        std::fflush(stderr);
    }
}

void App::SyncViews() {
    const size_t before = m_views.size();

    // Reuse panels by name so layout and camera survive re-runs.
    std::vector<std::unique_ptr<ImageViewPanel>> next;
    for (const std::string& name : m_viewerNames) {
        std::unique_ptr<ImageViewPanel> panel;
        for (auto& v : m_views) {
            if (v && v->Name() == name) { panel = std::move(v); break; }
        }
        if (!panel) panel = std::make_unique<ImageViewPanel>(name);
        next.push_back(std::move(panel));
    }
    m_views = std::move(next);
    for (auto& v : m_views) v->SetSharedCamera(m_syncCameras ? &m_sharedCam : nullptr);

    // The default layout is built once per session, not when viewers first
    // appear. Gating it on viewers meant that a script which fails before
    // declaring any (no image loaded, a syntax error) left every panel
    // floating in a stack — which is exactly the state you land in when
    // launching with no arguments at all.
    (void)before;
    DockLooseViewers();
}

// Any viewer without a dock node would open as a small floating window. This
// happens whenever a script declares viewer names the saved layout has never
// seen — e.g. switching from a script using "original" to one using "source".
//
// Regression to watch for (this shipped broken once): the bug only appears
// when tglab_layout.ini ALREADY EXISTS and lacks the current viewer names, so
// testing always from a deleted layout hides it. To reproduce by hand: run
// hello.tgl, quit, then run compare.tgl — its viewers must dock, not float.
void App::DockLooseViewers() {
    const ImGuiID target = CentreDockNode();
    if (target == 0) return;

    for (auto& v : m_views) {
        ImGuiWindow* w = ImGui::FindWindowByName(v->Name().c_str());
        // Never seen before, or known but not docked anywhere.
        if (w == nullptr || w->DockId == 0)
            ImGui::DockBuilderDockWindow(v->Name().c_str(), target);
    }

    // The compare panel is created on demand, so it misses the initial layout
    // pass and would otherwise open as a small floating window over the others.
    // m_compareNode is only set when the default layout was built, so fall back
    // to the viewers' node — same reasoning as CentreDockNode() above.
    if (m_compareOpen) {
        const ImGuiID node = m_compareNode != 0 ? m_compareNode : target;
        ImGuiWindow* w = ImGui::FindWindowByName("Compare CPU / GPU");
        if (node != 0 && (w == nullptr || w->DockId == 0))
            ImGui::DockBuilderDockWindow("Compare CPU / GPU", node);
    }
}

// Where new viewers should go, in order of preference:
//   1. beside a viewer that is already docked (so they stay together wherever
//      the user dragged them),
//   2. the node recorded when the default layout was built,
//   3. the dockspace's central node — the case that matters when a *saved*
//      layout exists but has never seen these viewer names.
ImGuiID App::CentreDockNode() const {
    for (const auto& v : m_views) {
        ImGuiWindow* w = ImGui::FindWindowByName(v->Name().c_str());
        if (w && w->DockId != 0) return w->DockId;
    }
    if (m_centreNode != 0) return m_centreNode;

    if (m_dockspaceId != 0) {
        if (ImGuiDockNode* node = ImGui::DockBuilderGetCentralNode(m_dockspaceId))
            return node->ID;
        return m_dockspaceId;
    }
    return 0;
}

// Docks the panels into a sensible arrangement the first time the app runs
// (or after Reset layout). Once tglab_layout.ini exists the user's own
// arrangement wins and this does nothing.
void App::BuildDefaultLayout(ImGuiID dockspace) {
    // Two ways in: a countdown that lets the panels be drawn once before we
    // dock them (DockBuilderDockWindow only binds windows ImGui has already
    // seen), or an explicit Reset layout, which can fire immediately because
    // by then every panel exists.
    bool build = m_rebuildLayout;
    if (m_layoutCountdown > 0) {
        --m_layoutCountdown;
        if (m_layoutCountdown == 0) build = true;
    }
    if (!build) return;
    m_rebuildLayout = false;

    ImGui::DockBuilderRemoveNode(dockspace);
    ImGui::DockBuilderAddNode(dockspace, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dockspace, ImGui::GetMainViewport()->Size);

    ImGuiID centre = dockspace;
    const ImGuiID left   = ImGui::DockBuilderSplitNode(centre, ImGuiDir_Left, 0.22f, nullptr, &centre);
    const ImGuiID right  = ImGui::DockBuilderSplitNode(centre, ImGuiDir_Right, 0.28f, nullptr, &centre);
    const ImGuiID bottom = ImGui::DockBuilderSplitNode(centre, ImGuiDir_Down, 0.18f, nullptr, &centre);
    ImGuiID leftBottom   = left;
    const ImGuiID leftTop = ImGui::DockBuilderSplitNode(leftBottom, ImGuiDir_Up, 0.5f, nullptr, &leftBottom);

    ImGui::DockBuilderDockWindow("Images",     leftTop);
    ImGui::DockBuilderDockWindow("Algorithms", leftTop);
    ImGui::DockBuilderDockWindow("Scripts",    leftTop);
    ImGui::DockBuilderDockWindow("Controls",   leftBottom);
    ImGui::DockBuilderDockWindow("Status",     bottom);
    // Compare gets its own column on the right: it is read side by side with
    // the viewers, not instead of them.
    ImGui::DockBuilderDockWindow("Compare CPU / GPU", right);

    // Viewers declared by the script share the centre node, so several
    // display() calls tab together rather than piling up as floating windows.
    for (const std::string& name : m_viewerNames)
        ImGui::DockBuilderDockWindow(name.c_str(), centre);

    ImGui::DockBuilderFinish(dockspace);
    m_centreNode  = centre;   // remember it for viewers added by later edits
    m_compareNode = right;    // ditto for the on-demand compare panel
}

void App::DrawMenuBar() {
    if (!ImGui::BeginMainMenuBar()) return;

    if (ImGui::BeginMenu("File")) {
        if (ImGui::MenuItem("Reload script", "F5")) m_dirty = true;
        if (ImGui::MenuItem("Reset all controls", nullptr, false, !m_ui.Controls().empty())) {
            for (UiControl& c : m_ui.Controls()) ResetControl(c);
            m_dirty = true;
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Exit")) PostQuitMessage(0);
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("View")) {
        if (ImGui::MenuItem("Sync pan/zoom", nullptr, &m_syncCameras)) SyncViews();
        if (ImGui::MenuItem("Reset layout")) m_rebuildLayout = true;
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Compute")) {
        const ExecMode mode = m_worker.GetExecMode();
        if (ImGui::MenuItem("Auto (GPU when available)", nullptr, mode == ExecMode::Auto)) {
            m_worker.SetExecMode(ExecMode::Auto);
            m_dirty = true;
        }
        if (ImGui::MenuItem("Force CPU", nullptr, mode == ExecMode::ForceCPU)) {
            m_worker.SetExecMode(ExecMode::ForceCPU);
            m_dirty = true;
        }
        if (ImGui::MenuItem("Force GPU", nullptr, mode == ExecMode::ForceGPU)) {
            m_worker.SetExecMode(ExecMode::ForceGPU);
            m_dirty = true;
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Compare CPU / GPU...", nullptr, m_compareOpen)) {
            m_compareOpen = !m_compareOpen;
            if (m_compareOpen && !m_compare) RequestCompare();
        }
        ImGui::EndMenu();
    }
    ImGui::EndMainMenuBar();
}

void App::DrawControlsPanel() {
    if (!ImGui::Begin("Controls")) { ImGui::End(); return; }

    if (m_ui.Controls().empty()) {
        ImGui::TextDisabled("No controls declared.");
        ImGui::TextWrapped("Use slider(\"name\", min, max, default) in the script.");
        ImGui::End();
        return;
    }

    // Enabled only when something actually differs from the script's defaults,
    // so the button doubles as an indicator that values have been changed.
    bool anyModified = false;
    for (const UiControl& c : m_ui.Controls())
        if (IsModified(c)) { anyModified = true; break; }

    ImGui::BeginDisabled(!anyModified);
    if (ImGui::Button("Reset all")) {
        for (UiControl& c : m_ui.Controls()) ResetControl(c);
        m_dirty = true;
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::TextDisabled("(double-click a control to reset it)");
    ImGui::Separator();

    for (UiControl& c : m_ui.Controls()) {
        switch (c.kind) {
            case UiControl::Kind::Slider: {
                float v = float(c.value);
                if (ImGui::SliderFloat(c.label.c_str(), &v, float(c.lo), float(c.hi))) {
                    c.value = v;
                    m_dirty = true;
                }
                break;
            }
            case UiControl::Kind::Check: {
                bool b = c.value != 0;
                if (ImGui::Checkbox(c.label.c_str(), &b)) {
                    c.value = b ? 1 : 0;
                    m_dirty = true;
                }
                break;
            }
            case UiControl::Kind::Choose: {
                if (c.options.empty()) break;
                const int sel = std::clamp(c.selected, 0, int(c.options.size()) - 1);
                if (ImGui::BeginCombo(c.label.c_str(), c.options[size_t(sel)].c_str())) {
                    for (int i = 0; i < int(c.options.size()); ++i) {
                        const bool chosen = (i == sel);
                        if (ImGui::Selectable(c.options[size_t(i)].c_str(), chosen)) {
                            if (i != c.selected) {
                                c.selected = i;
                                m_dirty = true;   // different algorithm -> re-run
                            }
                        }
                        if (chosen) ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }
                break;
            }
        }

        // Double-click the control just drawn to restore its scripted default.
        // Note ImGui reserves ctrl+click on a slider for typing an exact value,
        // so double-click is the free gesture here.
        if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
            if (IsModified(c)) {
                ResetControl(c);
                m_dirty = true;
            }
        }
        if (ImGui::IsItemHovered() && IsModified(c))
            ImGui::SetTooltip("default: %s\ndouble-click to reset", DefaultText(c).c_str());
    }
    ImGui::End();
}

bool App::IsModified(const UiControl& c) {
    if (c.kind == UiControl::Kind::Choose) return c.selected != 0;
    return c.value != c.def;
}

void App::ResetControl(UiControl& c) {
    if (c.kind == UiControl::Kind::Choose) c.selected = 0;   // first listed option
    else                                   c.value = c.def;
}

std::string App::DefaultText(const UiControl& c) {
    char buf[64];
    switch (c.kind) {
        case UiControl::Kind::Check:
            return c.def != 0 ? "on" : "off";
        case UiControl::Kind::Choose:
            return c.options.empty() ? std::string("-") : c.options[0];
        case UiControl::Kind::Slider:
        default:
            std::snprintf(buf, sizeof(buf), "%.3f", c.def);
            return buf;
    }
}

// CPU vs GPU for one algorithm: both results, their difference, and the
// numbers. A kernel that merely *looks* right is not verified.
void App::DrawComparePanel() {
    if (!m_compareOpen) return;

    // Big enough to show the numbers and the diff image without resizing.
    ImGui::SetNextWindowSize(ImVec2(460, 620), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Compare CPU / GPU", &m_compareOpen)) { ImGui::End(); return; }

    // Which stage to compare. Defaulting to the last one often lands on an
    // algorithm with no GPU kernel (a pipeline usually ends in something CPU
    // -only), so offer the comparable stages explicitly.
    const char* preview = m_compareStage < 0
                              ? "auto (first comparable)"
                              : (size_t(m_compareStage) < m_stageNames.size()
                                     ? m_stageNames[size_t(m_compareStage)].c_str()
                                     : "?");
    ImGui::SetNextItemWidth(220);
    if (ImGui::BeginCombo("stage", preview)) {
        if (ImGui::Selectable("auto (first comparable)", m_compareStage < 0)) {
            m_compareStage = -1;
            m_compare.reset();
            m_compareRequested = false;
        }
        for (int i = 0; i < int(m_stageNames.size()); ++i) {
            const bool gpuCapable = i < int(m_stageGpuCapable.size()) && m_stageGpuCapable[size_t(i)];
            // Non-comparable stages stay visible but unselectable, so it is
            // obvious *why* a given algorithm is not on offer.
            ImGui::BeginDisabled(!gpuCapable);
            const std::string label =
                m_stageNames[size_t(i)] + (gpuCapable ? "" : "  (CPU only)");
            if (ImGui::Selectable(label.c_str(), i == m_compareStage)) {
                m_compareStage = i;
                m_compare.reset();
                m_compareRequested = false;
            }
            ImGui::EndDisabled();
        }
        ImGui::EndCombo();
    }

    // Kick off the first comparison automatically when the panel opens, so it
    // shows numbers rather than an empty box.
    if (!m_compare && !m_worker.Busy() && !m_compareRequested) {
        m_compareRequested = true;
        RequestCompare();
    }

    if (ImGui::Button("Run comparison")) { m_compareRequested = true; RequestCompare(); }
    ImGui::SameLine();
    ImGui::TextDisabled("runs the pipeline twice, forced to each backend");

    if (m_worker.Busy()) {
        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f), "running...");
        ImGui::End();
        return;
    }

    if (!m_compare) {
        ImGui::Separator();
        ImGui::TextDisabled("No comparison yet.");
        ImGui::End();
        return;
    }

    const CompareResult& c = *m_compare;
    ImGui::Separator();

    if (!c.ok) {
        ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.45f, 1.0f), "%s", c.error.c_str());
        ImGui::End();
        return;
    }

    ImGui::Text("algorithm: %s", c.algorithm.c_str());

    if (ImGui::BeginTable("timing", 2, ImGuiTableFlags_SizingFixedFit)) {
        ImGui::TableNextRow(); ImGui::TableNextColumn();
        ImGui::Text("CPU"); ImGui::TableNextColumn(); ImGui::Text("%.1f ms", c.cpuMs);
        ImGui::TableNextRow(); ImGui::TableNextColumn();
        ImGui::Text("GPU"); ImGui::TableNextColumn(); ImGui::Text("%.1f ms", c.gpuMs);
        ImGui::TableNextRow(); ImGui::TableNextColumn();
        ImGui::Text("speedup"); ImGui::TableNextColumn();
        ImGui::TextColored(ImVec4(0.6f, 0.9f, 0.6f, 1.0f), "%.1fx", c.Speedup());
        ImGui::EndTable();
    }

    ImGui::Separator();

    // Agreement is the part that matters: a fast kernel that disagrees with
    // the reference is a bug, not an optimisation.
    const CompareStats& s = c.stats;
    const bool agrees = s.maxAbsDiff <= 4.0;
    ImGui::TextColored(agrees ? ImVec4(0.5f, 0.9f, 0.5f, 1.0f) : ImVec4(1.0f, 0.6f, 0.3f, 1.0f),
                       agrees ? "results agree" : "results DIFFER");
    ImGui::Text("max abs diff  %.4f", s.maxAbsDiff);
    ImGui::Text("mean abs diff %.4f", s.meanAbsDiff);
    ImGui::Text("rmse          %.4f", s.rmse);
    ImGui::Text("pixels beyond tolerance: %d / %d  (%.2f%%)",
                s.diffPixels, s.totalPixels, s.DiffFraction() * 100.0);

    ImGui::Separator();
    ImGui::TextDisabled("difference, amplified 16x");

    // The diff image is the fastest way to see *where* two results disagree,
    // which is usually more informative than the aggregate numbers.
    Image& diff = const_cast<Image&>(c.diffImage);
    if (diff.Valid() && m_diffTex.Update(m_dev, diff, m_compareVersion)) {
        const float avail = std::max(ImGui::GetContentRegionAvail().x, 64.0f);
        const float scale = std::min(1.0f, avail / float(m_diffTex.Width()));
        ImGui::Image(ImTextureRef(static_cast<ImTextureID>(m_diffTex.Handle().ptr)),
                     ImVec2(float(m_diffTex.Width()) * scale,
                            float(m_diffTex.Height()) * scale));
    }

    ImGui::End();
}

// Lists every .tgl beside the current script, so switching experiments is a
// click rather than a relaunch. Scanned on demand rather than watched: the
// directory changes far less often than the script itself.
void App::DrawScriptsPanel() {
    if (!ImGui::Begin("Scripts")) { ImGui::End(); return; }

    if (ImGui::Button("Refresh")) m_scriptListDirty = true;
    ImGui::SameLine();
    ImGui::TextDisabled("%s", m_scriptDir.empty() ? "(no directory)" : m_scriptDir.c_str());
    ImGui::Separator();

    if (m_scriptListDirty) {
        m_scriptListDirty = false;
        RescanScripts();
    }

    if (m_scriptList.empty()) {
        ImGui::TextDisabled("No .tgl files found.");
        ImGui::End();
        return;
    }

    for (const std::string& name : m_scriptList) {
        const bool current = (name == m_scriptName);
        if (ImGui::Selectable(name.c_str(), current) && !current) {
            SetScriptPath(m_scriptDir + "\\" + name);
        }
    }
    ImGui::End();
}

// Fills m_scriptList with the .tgl files in m_scriptDir, sorted.
void App::RescanScripts() {
    m_scriptList.clear();
    if (m_scriptDir.empty()) return;

    WIN32_FIND_DATAA fd{};
    const std::string pattern = m_scriptDir + "\\*.tgl";
    HANDLE h = FindFirstFileA(pattern.c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return;
    do {
        if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
            m_scriptList.push_back(fd.cFileName);
    } while (FindNextFileA(h, &fd));
    FindClose(h);

    std::sort(m_scriptList.begin(), m_scriptList.end());
}

void App::DrawPalettePanel() {
    if (!ImGui::Begin("Images")) { ImGui::End(); return; }

    if (m_palette.empty()) {
        ImGui::TextDisabled("Drag image files here.");
    }
    for (const PaletteEntry& e : m_palette) {
        ImGui::BulletText("%s", e.name.c_str());
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", e.path.c_str());
    }

    ImGui::Separator();
    ImGui::TextDisabled("Reference from script as image(\"name\")");
    ImGui::End();
}

void App::DrawAlgorithmsPanel() {
    if (!ImGui::Begin("Algorithms")) { ImGui::End(); return; }

    for (const std::string& name : Registry::Get().Names()) {
        auto a = Registry::Get().Create(name);
        if (!a) continue;
        if (ImGui::TreeNode(name.c_str())) {
            const char* cat = a->Category();
            if (cat && *cat) ImGui::TextDisabled("category: %s", cat);
            for (const Port& p : a->Inputs())  ImGui::BulletText("in  %s", p.name);
            for (const Port& p : a->Outputs()) ImGui::BulletText("out %s", p.name);
            for (ParamBase* p : a->Params())
                ImGui::BulletText("%s : %s", p->Name(), ParamTypeName(p->Type()));
            ImGui::TreePop();
        }
    }
    ImGui::End();
}

void App::Frame() {
    if (m_watch.Poll()) {
        if (ReadTextFile(m_watch.Path(), &m_source)) m_dirty = true;
    }

    // Re-run before NewFrame() so that viewers declared by the script exist
    // when the dock layout is built. This is only phase 1 (parse + interpret);
    // execution happens on the worker.
    if (m_dirty) {
        m_dirty = false;
        RunScript();
    }
    PollWorker();

    ImGui_ImplDX12_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();

    const ImGuiID dockspace = ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport());
    m_dockspaceId = dockspace;
    BuildDefaultLayout(dockspace);

    // SyncViews() may have run before the dockspace existed (first frame, or a
    // hot reload that introduced new viewer names), so catch any still-floating
    // viewer here now that a target node is resolvable.
    DockLooseViewers();

    if (ImGui::IsKeyPressed(ImGuiKey_F5, false)) m_dirty = true;

    DrawMenuBar();
    DrawPalettePanel();
    DrawScriptsPanel();
    DrawControlsPanel();
    DrawAlgorithmsPanel();
    // DrawComparePanel() uploads a texture, so it must run after BeginFrame()
    // has opened the command list — see below, with the image views.

    // Status strip: errors never blank the viewers, they just report.
    if (ImGui::Begin("Status")) {
        if (m_worker.Busy()) {
            // Visible feedback that a slow algorithm is still running — the
            // whole point of the worker is that the UI keeps drawing meanwhile.
            const char* spin = "|/-\\";
            m_spinner = (m_spinner + 1) % 4;
            ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f),
                               "%c working...", spin[m_spinner]);
        } else if (m_error.empty()) {
            ImGui::TextColored(ImVec4(0.5f, 0.9f, 0.5f, 1.0f), "OK");
        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.45f, 1.0f), "%s", m_error.c_str());
        }
        ImGui::TextDisabled("%s", m_scriptPath.empty() ? "(no script)" : m_scriptPath.c_str());

        // Timing plus where the work ran — the two numbers you want side by
        // side when deciding whether a GPU port was worth it.
        const char* modeName = "auto";
        switch (m_worker.GetExecMode()) {
            case ExecMode::ForceCPU: modeName = "CPU"; break;
            case ExecMode::ForceGPU: modeName = "GPU"; break;
            case ExecMode::Auto:     modeName = "auto"; break;
        }
        ImGui::TextDisabled("last run %.1f ms   %d stage(s) on GPU   [%s]",
                            m_worker.LastRunMs(), m_worker.LastGpuStages(), modeName);
    }
    ImGui::End();

    // A frame must be in flight before textures can be uploaded, so the
    // command list has to be open before views draw.
    ID3D12GraphicsCommandList* cl = m_dev.BeginFrame();

    for (auto& v : m_views) {
        // Images lag the declarations by one worker round-trip, so a freshly
        // declared viewer draws "computing..." until its first result lands.
        Image* img = nullptr;
        for (ViewerImage& vi : m_viewerImages)
            if (vi.name == v->Name()) { img = &vi.image; break; }

        v->SetContentVersion(m_contentVersion);
        v->Draw(m_dev, img);
    }

    // Also uploads a texture (the diff image), so it belongs here rather than
    // with the other panels above.
    DrawComparePanel();

    ImGui::Render();
    ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), cl);
    m_dev.EndFrame();
}

LRESULT WINAPI WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (ImGui_ImplWin32_WndProcHandler(hwnd, msg, wParam, lParam)) return true;

    switch (msg) {
        case WM_SIZE:
            if (g_app && wParam != SIZE_MINIMIZED)
                g_app->OnResize(UINT(LOWORD(lParam)), UINT(HIWORD(lParam)));
            return 0;
        case WM_DROPFILES: {
            HDROP drop = reinterpret_cast<HDROP>(wParam);
            const UINT n = DragQueryFileA(drop, 0xFFFFFFFF, nullptr, 0);
            for (UINT i = 0; i < n; ++i) {
                char path[MAX_PATH] = {};
                if (DragQueryFileA(drop, i, path, MAX_PATH) && g_app)
                    g_app->LoadImageIntoPalette(path);
            }
            DragFinish(drop);
            return 0;
        }
        case WM_SYSCOMMAND:
            if ((wParam & 0xfff0) == SC_KEYMENU) return 0;   // disable ALT menu
            break;
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
    }
    return DefWindowProcA(hwnd, msg, wParam, lParam);
}

} // namespace
} // namespace tglab

int main(int argc, char** argv) {
    using namespace tglab;

    WNDCLASSEXA wc = {sizeof(wc), CS_CLASSDC, WndProc, 0, 0,
                      GetModuleHandle(nullptr), nullptr, nullptr, nullptr, nullptr,
                      "tglab", nullptr};
    RegisterClassExA(&wc);

    // Size the window so the *client* area is the size we want; otherwise the
    // dockspace is laid out larger than the visible area and right-hand panels
    // run off the edge of the screen.
    RECT want = {0, 0, 1600, 1000};
    AdjustWindowRect(&want, WS_OVERLAPPEDWINDOW, TRUE);
    HWND hwnd = CreateWindowA(wc.lpszClassName, "tglab", WS_OVERLAPPEDWINDOW,
                              100, 100, want.right - want.left, want.bottom - want.top,
                              nullptr, nullptr, wc.hInstance, nullptr);

    App app;
    g_app = &app;
    if (!app.Init(hwnd)) {
        MessageBoxA(nullptr, "Failed to initialize D3D12 / ImGui.", "tglab", MB_ICONERROR);
        DestroyWindow(hwnd);
        UnregisterClassA(wc.lpszClassName, wc.hInstance);
        return 1;
    }

    DragAcceptFiles(hwnd, TRUE);
    ShowWindow(hwnd, SW_SHOWDEFAULT);
    UpdateWindow(hwnd);

    // Command line: [script.tgl] [image ...]
    std::string script = "scripts/hello.tgl";
    bool gotImage = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a.size() > 4 && a.substr(a.size() - 4) == ".tgl") {
            script = a;
        } else {
            app.LoadImageIntoPalette(a);
            gotImage = true;
        }
    }
    // Launching bare should show something working, not an error about a
    // missing palette entry, so load the sample the stock scripts refer to.
    if (!gotImage) app.LoadImageIntoPalette("assets/test.png");
    app.SetScriptPath(script);

    bool running = true;
    while (running) {
        MSG msg;
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
            if (msg.message == WM_QUIT) running = false;
        }
        if (!running) break;
        app.Frame();
    }

    app.Shutdown();
    g_app = nullptr;
    DestroyWindow(hwnd);
    UnregisterClassA(wc.lpszClassName, wc.hInstance);
    return 0;
}
