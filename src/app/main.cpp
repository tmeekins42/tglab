// tglab — CV/CG algorithm research lab.
//
// M1 vertical slice: load an image, run a script that calls an algorithm with
// slider-driven parameters, display the result. Everything on the UI thread.
#include <windows.h>
#include <shellapi.h>   // DragAcceptFiles / DragQueryFile

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "../core/algorithm.h"
#include "../core/image_io.h"
#include "../core/pipeline.h"
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
    void SyncViews();

    Device      m_dev;
    FileWatch   m_watch;
    std::string m_scriptPath;
    std::string m_source;

    UiState  m_ui;
    Pipeline m_pipe;
    Pipeline m_prev;
    std::vector<PaletteEntry> m_palette;
    std::vector<Data>         m_sources;   // parallel to m_palette

    std::vector<std::unique_ptr<ImageViewPanel>> m_views;
    ViewCamera m_sharedCam;
    bool       m_syncCameras = true;

    std::string m_error;        // last script/run error, empty when fine
    bool        m_dirty = true; // re-run requested
    bool        m_everRan = false;
    uint64_t    m_contentVersion = 1;
    bool        m_rebuildLayout = false;   // set once viewer names are known
    bool        m_haveSavedLayout = false; // a tglab_layout.ini already existed
    ImGuiID     m_centreNode = 0;          // where new viewers get docked
};

App* g_app = nullptr;

bool App::Init(HWND hwnd) {
    if (!m_dev.Init(hwnd, /*enableDebugLayer=*/
#ifdef _DEBUG
                    true
#else
                    false
#endif
                    ))
        return false;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.IniFilename = "tglab_layout.ini";   // remembers panel layout across runs
    ImGui::StyleColorsDark();

    // If the user already has a layout, never overwrite it with the default.
    m_haveSavedLayout = (GetFileAttributesA(io.IniFilename) != INVALID_FILE_ATTRIBUTES);

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
    m_dev.WaitForLastSubmittedFrame();
    m_views.clear();
    ImGui_ImplDX12_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();
    m_dev.Shutdown();
}

void App::SetScriptPath(const std::string& path) {
    m_scriptPath = path;
    m_watch.Watch(path);
    if (!ReadTextFile(path, &m_source))
        m_error = "could not read script '" + path + "'";
    m_dirty = true;
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

void App::RunScript() {
    m_error.clear();

    // Sources are rebuilt each run so PortRef{-1, i} stays in step.
    m_sources.clear();
    std::vector<SourceImage> names;
    for (size_t i = 0; i < m_palette.size(); ++i) {
        // Data is move-only; hand the pipeline a shallow clone of the pixels.
        if (std::holds_alternative<Image>(m_palette[i].data))
            m_sources.push_back(Data{std::get<Image>(m_palette[i].data).Clone()});
        else
            m_sources.push_back(Data{});
        names.push_back({m_palette[i].name, int(i)});
    }

    Program prog;
    std::string err;
    if (!Parse(m_source, &prog, &err)) {
        m_error = err;
        return;   // keep the last good result on screen
    }

    Pipeline built;
    InterpResult r = Interpret(prog, names, &m_ui, &built);
    if (!r.ok) {
        m_error = r.error;
        return;
    }

    if (!built.Execute(&m_sources, m_everRan ? &m_pipe : nullptr, &err)) {
        m_error = err;
        return;
    }

    m_pipe = std::move(built);
    m_everRan = true;
    ++m_contentVersion;   // tells the views their pixels are new
    SyncViews();
}

void App::SyncViews() {
    const size_t before = m_views.size();

    // Reuse panels by name so layout and camera survive re-runs.
    std::vector<std::unique_ptr<ImageViewPanel>> next;
    for (const ViewerDecl& d : m_pipe.Viewers()) {
        std::unique_ptr<ImageViewPanel> panel;
        for (auto& v : m_views) {
            if (v && v->Name() == d.name) { panel = std::move(v); break; }
        }
        if (!panel) panel = std::make_unique<ImageViewPanel>(d.name);
        next.push_back(std::move(panel));
    }
    m_views = std::move(next);
    for (auto& v : m_views) v->SetSharedCamera(m_syncCameras ? &m_sharedCam : nullptr);

    // First time viewers appear, dock them — but only when the user has no
    // saved layout of their own to preserve.
    if (before == 0 && !m_views.empty() && !m_haveSavedLayout) m_rebuildLayout = true;

    // A viewer added by a script edit has no dock node yet and would open as a
    // floating window. Park it with its siblings instead.
    if (m_centreNode != 0) {
        for (auto& v : m_views) {
            if (ImGui::FindWindowByName(v->Name().c_str()) == nullptr)
                ImGui::DockBuilderDockWindow(v->Name().c_str(), m_centreNode);
        }
    }
}

// Docks the panels into a sensible arrangement the first time the app runs
// (or after Reset layout). Once tglab_layout.ini exists the user's own
// arrangement wins and this does nothing.
void App::BuildDefaultLayout(ImGuiID dockspace) {
    if (!m_rebuildLayout) return;
    m_rebuildLayout = false;

    ImGui::DockBuilderRemoveNode(dockspace);
    ImGui::DockBuilderAddNode(dockspace, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dockspace, ImGui::GetMainViewport()->Size);

    ImGuiID centre = dockspace;
    const ImGuiID left   = ImGui::DockBuilderSplitNode(centre, ImGuiDir_Left, 0.22f, nullptr, &centre);
    const ImGuiID bottom = ImGui::DockBuilderSplitNode(centre, ImGuiDir_Down, 0.18f, nullptr, &centre);
    ImGuiID leftBottom   = left;
    const ImGuiID leftTop = ImGui::DockBuilderSplitNode(leftBottom, ImGuiDir_Up, 0.5f, nullptr, &leftBottom);

    ImGui::DockBuilderDockWindow("Images",     leftTop);
    ImGui::DockBuilderDockWindow("Algorithms", leftTop);
    ImGui::DockBuilderDockWindow("Controls",   leftBottom);
    ImGui::DockBuilderDockWindow("Status",     bottom);

    // Viewers declared by the script share the centre node, so several
    // display() calls tab together rather than piling up as floating windows.
    for (const ViewerDecl& d : m_pipe.Viewers())
        ImGui::DockBuilderDockWindow(d.name.c_str(), centre);

    ImGui::DockBuilderFinish(dockspace);
    m_centreNode = centre;   // remember it for viewers added by later edits
}

void App::DrawMenuBar() {
    if (!ImGui::BeginMainMenuBar()) return;

    if (ImGui::BeginMenu("File")) {
        if (ImGui::MenuItem("Reload script", "F5")) m_dirty = true;
        ImGui::Separator();
        if (ImGui::MenuItem("Exit")) PostQuitMessage(0);
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("View")) {
        if (ImGui::MenuItem("Sync pan/zoom", nullptr, &m_syncCameras)) SyncViews();
        if (ImGui::MenuItem("Reset layout")) m_rebuildLayout = true;
        ImGui::EndMenu();
    }
    ImGui::EndMainMenuBar();
}

void App::DrawControlsPanel() {
    if (!ImGui::Begin("Controls")) { ImGui::End(); return; }

    if (m_ui.Controls().empty()) {
        ImGui::TextDisabled("No controls declared.");
        ImGui::TextWrapped("Use slider(\"name\", min, max, default) in the script.");
    }

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
            case UiControl::Kind::Choose:
                // M2
                break;
        }
    }
    ImGui::End();
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
    // when the dock layout is built. This is pure CPU work — no ImGui or D3D12
    // state is touched, and uploads happen later inside the view Draw calls.
    if (m_dirty) {
        m_dirty = false;
        RunScript();
    }

    ImGui_ImplDX12_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();

    const ImGuiID dockspace = ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport());
    BuildDefaultLayout(dockspace);

    if (ImGui::IsKeyPressed(ImGuiKey_F5, false)) m_dirty = true;

    DrawMenuBar();
    DrawPalettePanel();
    DrawControlsPanel();
    DrawAlgorithmsPanel();

    // Status strip: errors never blank the viewers, they just report.
    if (ImGui::Begin("Status")) {
        if (m_error.empty()) {
            ImGui::TextColored(ImVec4(0.5f, 0.9f, 0.5f, 1.0f), "OK");
        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.45f, 1.0f), "%s", m_error.c_str());
        }
        ImGui::TextDisabled("%s", m_scriptPath.empty() ? "(no script)" : m_scriptPath.c_str());
    }
    ImGui::End();

    // A frame must be in flight before textures can be uploaded, so the
    // command list has to be open before views draw.
    ID3D12GraphicsCommandList* cl = m_dev.BeginFrame();

    for (auto& v : m_views) {
        const ViewerDecl* decl = nullptr;
        for (const ViewerDecl& d : m_pipe.Viewers())
            if (d.name == v->Name()) { decl = &d; break; }

        Data* data = nullptr;
        if (decl) data = const_cast<Data*>(m_pipe.Resolve(decl->source, &m_sources));
        v->SetContentVersion(m_contentVersion);
        v->Draw(m_dev, data);
    }

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
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a.size() > 4 && a.substr(a.size() - 4) == ".tgl") script = a;
        else app.LoadImageIntoPalette(a);
    }
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
