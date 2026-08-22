// tglab — CV/CG algorithm research lab.
//
// M1 vertical slice: load an image, run a script that calls an algorithm with
// slider-driven parameters, display the result. Everything on the UI thread.
#include <windows.h>
#include <shellapi.h>   // DragAcceptFiles / DragQueryFile

#include <algorithm>
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "../core/algorithm.h"
#include "../core/image_io.h"
#include "../core/compare.h"
#include "../core/pipeline.h"
#include "../algo_util/histogram.h"
#include "../core/exif.h"
#include "../core/image_loader.h"
#include "../core/image_stats.h"
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

// Resolves a data path (scripts/, assets/) whatever directory the app was
// launched from. Tried in order: as given, then beside the executable, then
// one/two/three levels up from it -- which is where the repo root sits when
// running straight out of build/Debug.
std::string ResolveDataPath(const std::string& rel) {
    if (GetFileAttributesA(rel.c_str()) != INVALID_FILE_ATTRIBUTES) return rel;

    char exe[MAX_PATH] = {};
    if (GetModuleFileNameA(nullptr, exe, MAX_PATH) == 0) return rel;

    std::string dir(exe);
    if (const size_t slash = dir.find_last_of("/\\"); slash != std::string::npos)
        dir = dir.substr(0, slash);

    for (int up = 0; up < 4; ++up) {
        const std::string candidate = dir + "\\" + rel;
        if (GetFileAttributesA(candidate.c_str()) != INVALID_FILE_ATTRIBUTES) return candidate;
        const size_t slash = dir.find_last_of("/\\");
        if (slash == std::string::npos) break;
        dir = dir.substr(0, slash);
    }
    return rel;   // let the caller report the failure
}

// Box-downsamples `src` so its longest side is at most `maxSide`. Cheap and
// done once per image, so the palette never uploads a full-resolution texture
// just to draw a small icon.
void MakeThumbnail(Image& src, int maxSide, Image* out) {
    ImageView v = src.MapCpuRead();
    if (!v.Valid()) { out->Reset(); return; }

    const int sw = v.desc.width;
    const int sh = v.desc.height;
    const int step = std::max(1, (std::max(sw, sh) + maxSide - 1) / maxSide);
    const int dw = std::max(1, sw / step);
    const int dh = std::max(1, sh / step);

    out->Alloc({dw, dh, Format::RGBA8});
    ImageView o = out->MapCpuWrite();

    for (int y = 0; y < dh; ++y) {
        for (int x = 0; x < dw; ++x) {
            // Average the step x step block, so downsampling does not alias a
            // scanned page into noise.
            int acc[4] = {0, 0, 0, 0};
            int n = 0;
            for (int sy = y * step; sy < std::min((y + 1) * step, sh); ++sy) {
                for (int sx = x * step; sx < std::min((x + 1) * step, sw); ++sx) {
                    if (v.desc.format == Format::R32F) {
                        // A mosaic holds raw sensor counts (2047..15488 on a
                        // Canon CR2), not 0..1. Clamping those to 1.0 made
                        // every raw thumbnail pure white, so scale by the
                        // sensor's own levels first. An ordinary R32F image
                        // has blackLevel 0 and whiteLevel 1, so this is a
                        // no-op for it.
                        const float black = v.desc.blackLevel;
                        const float range = std::max(v.desc.whiteLevel - black, 1e-6f);
                        const float f = (*v.At<float>(sx, sy) - black) / range;
                        const int g = int(std::clamp(f, 0.0f, 1.0f) * 255.0f);
                        acc[0] += g; acc[1] += g; acc[2] += g; acc[3] += 255;
                    } else if (v.desc.format == Format::RGBA32F) {
                        const float* p = v.At<float>(sx, sy);
                        for (int c = 0; c < 4; ++c)
                            acc[c] += int(std::clamp(p[c], 0.0f, 1.0f) * 255.0f);
                    } else if (v.desc.format == Format::RGBA16F) {
                        const uint16_t* p = v.At<uint16_t>(sx, sy);
                        for (int c = 0; c < 4; ++c)
                            acc[c] += int(std::clamp(HalfToFloat(p[c]), 0.0f, 1.0f) * 255.0f);
                    } else {
                        const uint8_t* p = v.At<uint8_t>(sx, sy);
                        for (int c = 0; c < 4; ++c) acc[c] += p[c];
                    }
                    ++n;
                }
            }
            uint8_t* d = o.At<uint8_t>(x, y);
            for (int c = 0; c < 4; ++c) d[c] = uint8_t(n ? acc[c] / n : 0);
        }
    }
}

// One image in the palette. The script-visible `name` is deliberately separate
// from the file it came from: dropping a new file onto an existing slot keeps
// the name, so scripts referring to image("test") keep working when the file
// behind it changes.
struct PaletteEntry {
    std::string name;      // what image("...") refers to
    std::string path;      // where it was loaded from
    Data        data;
    GpuTexture  thumb;       // small preview, built lazily
    Image       thumbImage;  // downsampled copy the texture is built from
    uint64_t    thumbVersion = 0;   // version thumbImage was built at
    uint64_t    version = 1;   // bumped on reload so the thumbnail refreshes

    // Screen rect of this row, recorded while drawing so a WM_DROPFILES at a
    // given point can find which slot it landed on.
    ImVec2      rowMin{}, rowMax{};
};

class App {
public:
    bool Init(HWND hwnd);
    void Shutdown();
    void Frame();
    void OnResize(UINT w, UINT h) { if (m_dev.Ready()) m_dev.OnResize(w, h); }

    void LoadImageIntoPalette(const std::string& path, const std::string& targetSlot = "");

    // Queues a load on the loader thread. Used by the drop handler, which runs
    // on the UI thread and must not block on file I/O.
    void RequestImageLoad(const std::string& path, const std::string& targetSlot = "");

    // Name of the palette slot at a screen point, or "" if none. Used by the
    // drop handler, which only has the App pointer.
    std::string SlotNameAt(int sx, int sy) const;
    void SetScriptPath(const std::string& path);

private:
    void RunScript();
    void BuildDefaultLayout(ImGuiID dockspace);
    void DrawMenuBar();
    void DrawControlsPanel();
    void DrawControl(UiControl& c);
    void DrawPalettePanel();
    void BeginRename(int i);
    void CommitRename(int i);
    int  SlotAtScreenPos(int sx, int sy) const;
    void DrawAlgorithmsPanel();
    void DrawScriptsPanel();
    void RescanScripts();
    void DrawComparePanel();
    void DrawInfoPanel();
    bool ViewerVisible(const std::string& name) const;
    void RequestCompare();
    void SyncViews();
    void DockLooseViewers();
    void DockOnDemandPanels();
    void ReleaseRetiredViews();
    void ReportError(const std::string& prevError);
    void PollWorker();
    void PollLoader();
    void PollStats();
    void InstallLoadedImage(LoadResult&& r);
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
    ImageLoader    m_loader;
    std::vector<PaletteEntry> m_palette;
    int  m_renamingSlot = -1;      // index being renamed inline, or -1
    bool m_renameFocused = false;
    char m_renameBuf[128] = {};

    // Viewer names come from phase 1 (immediately); images arrive later from
    // the worker. Keeping them separate lets panels appear and dock straight
    // away rather than popping in when the first result lands.
    std::vector<std::string> m_viewerNames;
    std::vector<ViewerImage> m_viewerImages;
    uint64_t                 m_pendingSeq = 0;
    uint64_t                 m_shownSeq   = 0;

    std::vector<std::unique_ptr<ImageViewPanel>> m_views;

    // Viewers a script switch removed. Held for a few frames so the GPU is
    // done with their textures before the destructor frees them.
    struct RetiredView {
        std::unique_ptr<ImageViewPanel> view;
        int framesLeft = 0;
    };
    std::vector<RetiredView> m_retiredViews;
    ViewCamera m_sharedCam;
    bool       m_syncCameras = true;

    std::string m_error;        // last script/run error, empty when fine
    std::string m_loadError;    // script file could not be read (survives a re-run)
    bool        m_dirty = true; // re-run requested
    bool        m_rebuildLayout = false;   // explicit "Reset layout" request
    int         m_layoutCountdown = 0;     // frames until the default layout is built
    int         m_switchCountdown = 0;     // self-test: frames until TGLAB_SWITCH fires
    int         m_dropTestCountdown = 0;   // self-test: frames until TGLAB_DROPTEST fires
    int         m_infoTestCountdown = 0;   // self-test: frames until TGLAB_INFOTEST fires
    bool        m_infoTestNudged    = false;
    bool        m_haveSavedLayout = false; // a tglab_layout.ini already existed
    ImGuiID     m_centreNode = 0;          // where new viewers get docked
    ImGuiID     m_dockspaceId = 0;         // fallback when a saved layout exists
    ImGuiID     m_compareNode = 0;         // right-hand column for compare
    ImGuiID     m_lastViewerNode = 0;      // where viewers were docked, for script switches
    int         m_spinner = 0;             // "working..." animation

    // Compare mode (M4). TGLAB_COMPARE=1 opens the panel at startup, which is
    // how the panel gets verified without driving the menus.
    std::shared_ptr<CompareResult> m_compare;
    GpuTexture m_diffTex;
    bool       m_compareOpen       = false;

    // Info panel: dimensions, per-channel histogram, and capture settings for
    // whichever image is being looked at.
    bool       m_infoOpen          = true;    // shown by default; it is where the histogram lives
    // Histograms are computed on their own thread: four 256-bin passes over an
    // 8 MP image is far too much to spend in a frame for a panel that is purely
    // informational, and it was noticeably slowing the app when done inline.
    // Which viewer the info panel describes: the last one selected, not the one
    // focused right now, since touching any control takes focus away from it.
    std::string m_infoViewer;

    ImageStats  m_statsWorker;
    StatsResult m_stats;                     // newest result, drawn as-is
    bool        m_statsRequested   = false;  // a request is outstanding
    std::string m_requestedSource;           // what that request was for
    uint64_t    m_requestedVersion = 0;

    ExifData   m_infoExif;
    std::string m_infoExifPath;              // avoids re-reading the same file
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
    m_loader.Start();
    m_statsWorker.Start();

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
    if (GetEnvironmentVariableA("TGLAB_SWITCH", nullptr, 0) > 0) m_switchCountdown = 120;
    if (GetEnvironmentVariableA("TGLAB_DROPTEST", nullptr, 0) > 0) m_dropTestCountdown = 120;
    if (GetEnvironmentVariableA("TGLAB_INFOTEST", nullptr, 0) > 0) m_infoTestCountdown = 120;

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
    m_loader.Stop();
    m_statsWorker.Stop();
    m_dev.WaitForLastSubmittedFrame();

    // Every GpuTexture holds an SRV descriptor from the device's heap, so all
    // of them must be released BEFORE m_dev.Shutdown() destroys that heap.
    // Anything left to a member destructor would run afterwards and free a
    // descriptor against a dangling heap base.
    m_views.clear();
    m_retiredViews.clear();
    m_diffTex.Release();
    for (PaletteEntry& e : m_palette) e.thumb.Release();

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
    if (!ReadTextFile(path, &m_source)) {
        // Keep the failure visible: an unread script leaves m_source empty,
        // and an empty script parses fine and declares nothing, so without
        // this the app would look like it started normally with no panels.
        m_source.clear();
        m_loadError = "could not read script '" + path + "'";
    } else {
        m_loadError.clear();
    }
    m_dirty = true;
    UpdateWindowTitle();

    if (GetEnvironmentVariableA("TGLAB_COMPARE", nullptr, 0) > 0) m_compareOpen = true;
    // Self-test hook: opens the info panel at startup so its cost can be
    // measured without driving the menus.
    if (GetEnvironmentVariableA("TGLAB_INFO", nullptr, 0) > 0) m_infoOpen = true;
}

// Queues a file for background decoding. The read happens on the loader thread
// because stbi_load() blocks on file I/O, and doing that inside WM_DROPFILES
// freezes the entire window until it finishes -- badly so for a large scan on
// a network drive.
void App::RequestImageLoad(const std::string& path, const std::string& targetSlot) {
    // Clear any previous error before the load starts.
    //
    // Otherwise a failed drop (an unsupported file, say) leaves its message on
    // screen while the *next* file decodes -- and a large raw takes seconds --
    // so the new load looks like it has already failed. The error belongs to
    // the file that produced it, and that file is no longer what the user is
    // waiting on.
    const std::string prevError = m_error;
    m_error.clear();
    ReportError(prevError);
    UpdateWindowTitle();

    m_loader.Request(path, targetSlot);
}

// Installs a decoded image. Called on the UI thread once the loader is done.
void App::LoadImageIntoPalette(const std::string& path, const std::string& targetSlot) {
    LoadResult r;
    r.path       = path;
    r.targetSlot = targetSlot;
    r.ok         = LoadImageFile(path, &r.image, &r.error);
    InstallLoadedImage(std::move(r));
}

void App::InstallLoadedImage(LoadResult&& r) {
    if (!r.ok) {
        m_error = r.error;
        ReportError("");
        return;
    }

    // Replace a specific slot, keeping its script-visible name.
    if (!r.targetSlot.empty()) {
        for (PaletteEntry& e : m_palette) {
            if (e.name == r.targetSlot) {
                e.path = r.path;
                e.data = Data{std::move(r.image)};
                ++e.version;            // makes the thumbnail rebuild
                m_dirty = true;
                return;
            }
        }
    }

    // Otherwise the name is the filename without extension.
    std::string name = r.path;
    if (auto slash = name.find_last_of("/\\"); slash != std::string::npos) name = name.substr(slash + 1);
    if (auto dot = name.find_last_of('.'); dot != std::string::npos) name = name.substr(0, dot);

    for (PaletteEntry& e : m_palette) {
        if (e.name == name) {           // reloading the same name replaces it
            e.path = r.path;
            e.data = Data{std::move(r.image)};
            ++e.version;
            m_dirty = true;
            return;
        }
    }

    PaletteEntry e;
    e.name = std::move(name);
    e.path = r.path;
    e.data = Data{std::move(r.image)};
    m_palette.push_back(std::move(e));
    m_dirty = true;
}

// Picks up finished loads. Called once per frame.
void App::PollLoader() {
    LoadResult r;
    while (m_loader.TryFetch(&r)) InstallLoadedImage(std::move(r));
}

// Picks up a finished histogram. Called once per frame.
void App::PollStats() {
    StatsResult s;
    if (!m_statsWorker.TryFetch(&s)) return;
    m_stats = std::move(s);
    // Only clear the outstanding flag when this is the result we asked for; a
    // late arrival for an older image must not stop the current request from
    // being re-issued.
    if (m_stats.source == m_requestedSource && m_stats.version == m_requestedVersion)
        m_statsRequested = false;
}

// Phase 1 only: parse and interpret (microseconds), then hand the recorded
// pipeline to the worker. Interpret() is what declares sliders into m_ui, so
// keeping it here means UiState is never shared across threads.
void App::RunScript() {
    const std::string prevError = m_error;
    m_error.clear();

    // A script that could not be read is not an empty script: running one
    // would silently succeed and declare no panels at all.
    if (!m_loadError.empty()) {
        m_error = m_loadError;
        ReportError(prevError);
        UpdateWindowTitle();
        return;
    }

    // Sources are rebuilt each run so PortRef{-1, i} stays in step. The worker
    // gets its own copies, since it may still be reading them next frame.
    std::vector<Data> sources;
    std::vector<SourceImage> names;
    std::vector<uint64_t> versions;
    for (size_t i = 0; i < m_palette.size(); ++i) {
        if (std::holds_alternative<Image>(m_palette[i].data))
            sources.push_back(Data{std::get<Image>(m_palette[i].data).Clone()});
        else
            sources.push_back(Data{});
        {
            SourceImage si;
            si.name  = m_palette[i].name;
            si.index = int(i);
            // Tells the interpreter to insert a demosaic for this source, so a
            // script never has to mention one.
            if (std::holds_alternative<Image>(m_palette[i].data))
                si.isMosaic = std::get<Image>(m_palette[i].data).Desc().IsMosaic();
            names.push_back(si);
        }
        // Bumped when a drop replaces the file behind this slot, which is what
        // lets the stage cache notice the pixels changed.
        versions.push_back(m_palette[i].version);
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

            m_pendingSeq = m_worker.Submit(std::move(built), std::move(sources),
                                           std::move(versions));
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
    std::vector<uint64_t> versions;
    for (size_t i = 0; i < m_palette.size(); ++i) {
        if (std::holds_alternative<Image>(m_palette[i].data))
            sources.push_back(Data{std::get<Image>(m_palette[i].data).Clone()});
        else
            sources.push_back(Data{});
        {
            SourceImage si;
            si.name  = m_palette[i].name;
            si.index = int(i);
            // Tells the interpreter to insert a demosaic for this source, so a
            // script never has to mention one.
            if (std::holds_alternative<Image>(m_palette[i].data))
                si.isMosaic = std::get<Image>(m_palette[i].data).Desc().IsMosaic();
            names.push_back(si);
        }
        // Bumped when a drop replaces the file behind this slot, which is what
        // lets the stage cache notice the pixels changed.
        versions.push_back(m_palette[i].version);
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

        // Merge by name rather than replace. The worker omits viewers whose
        // pixels did not change -- displaying a source while a slider drives a
        // later stage is the common case -- so a wholesale assignment would
        // blank them. Views look themselves up by name, and m_views is rebuilt
        // from the script, so an entry for a removed display() is simply never
        // consulted.
        for (ViewerImage& in : out.viewers) {
            auto it = std::find_if(m_viewerImages.begin(), m_viewerImages.end(),
                                   [&](const ViewerImage& v) { return v.name == in.name; });
            if (it == m_viewerImages.end())
                m_viewerImages.push_back(std::move(in));
            else
                *it = std::move(in);
        }
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
    // A script switch replaces every viewer, so a remembered selection may name
    // one that no longer exists. Drop it rather than leave the info panel
    // pointing at nothing.
    if (!m_infoViewer.empty() &&
        std::find(m_viewerNames.begin(), m_viewerNames.end(), m_infoViewer) ==
            m_viewerNames.end())
        m_infoViewer.clear();

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

    // Retire the panels this script no longer declares instead of destroying
    // them here. Their GpuTextures may still be referenced by frames the GPU
    // has not finished, and freeing those out from under it hangs the device
    // (TDR) rather than failing cleanly -- which looks exactly like the app
    // locking up. Switching scripts replaces every viewer at once, so this is
    // the path that hits it.
    for (auto& v : m_views)
        if (v) m_retiredViews.push_back({std::move(v), kNumFramesInFlight + 1});

    m_views = std::move(next);
    for (auto& v : m_views) v->SetSharedCamera(m_syncCameras ? &m_sharedCam : nullptr);

    DockLooseViewers();
}

// Frees retired viewers once the GPU can no longer be referencing their
// textures. Called once per frame.
void App::ReleaseRetiredViews() {
    for (auto& r : m_retiredViews) {
        if (r.framesLeft > 0) --r.framesLeft;
    }
    std::erase_if(m_retiredViews, [](const RetiredView& r) { return r.framesLeft == 0; });
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

    DockOnDemandPanels();
}

// Compare and Image Info are opened from a menu rather than declared by the
// script, so they are drawn near the end of the frame and never exist when the
// default layout is built. DockBuilderDockWindow only binds windows ImGui has
// already seen, so both used to open as floating windows over the viewers --
// and stayed that way, since a floating window with no DockId is exactly what
// this checks for.
//
// Called both from DockLooseViewers() and again after the panels draw, so the
// first appearance is caught on the following frame.
void App::DockOnDemandPanels() {
    // m_compareNode is only set when this session built the default layout, so
    // fall back to the viewers' node -- same reasoning as CentreDockNode().
    const ImGuiID fallback = CentreDockNode();
    const ImGuiID node = m_compareNode != 0 ? m_compareNode : fallback;
    if (node == 0) return;

    auto dock = [&](bool open, const char* name) {
        if (!open) return;
        ImGuiWindow* w = ImGui::FindWindowByName(name);
        // Only when ImGui has seen it and it is not docked: re-docking a window
        // the user has deliberately dragged elsewhere would fight them.
        if (w != nullptr && w->DockId == 0) ImGui::DockBuilderDockWindow(name, node);
    };

    dock(m_compareOpen, "Compare CPU / GPU");
    dock(m_infoOpen,    "Image Info");
}

// Where new viewers should go, in order of preference:
//   1. beside a viewer that is already docked (so they stay together wherever
//      the user dragged them),
//   2. the node recorded when the default layout was built,
//   3. the dockspace's central node — the case that matters when a *saved*
//      layout exists but has never seen these viewer names.
ImGuiID App::CentreDockNode() const {
    // 1. Beside a viewer that is already docked, so viewers stay together
    //    wherever the user dragged them.
    for (const auto& v : m_views) {
        ImGuiWindow* w = ImGui::FindWindowByName(v->Name().c_str());
        if (w && w->DockId != 0) {
            const_cast<App*>(this)->m_lastViewerNode = w->DockId;
            return w->DockId;
        }
    }
    // 2. The node recorded when this session built the default layout.
    if (m_centreNode != 0) return m_centreNode;

    // 3. Where the previous viewers were docked. Switching scripts destroys
    //    every viewer at once (the new script uses different names), so this
    //    is the branch that matters for a script switch, and with a saved
    //    layout m_centreNode was never set.
    if (m_lastViewerNode != 0) return m_lastViewerNode;

    // 4. The dockspace central node, but only if it is a usable leaf. Handing
    //    DockBuilder a non-leaf node wedges the whole UI rather than failing.
    if (m_dockspaceId != 0) {
        if (ImGuiDockNode* node = ImGui::DockBuilderGetCentralNode(m_dockspaceId))
            if (node->IsLeafNode()) return node->ID;
    }

    // 4. Nothing safe to dock into. Returning 0 leaves the viewer floating,
    //    which is recoverable; docking into a bad node is not.
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
    // Compare and Image Info share the right-hand column: both are read
    // alongside the viewers rather than instead of them, and tabbing them
    // together keeps the centre as wide as possible.
    ImGui::DockBuilderDockWindow("Compare CPU / GPU", right);
    ImGui::DockBuilderDockWindow("Image Info",        right);

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
        if (ImGui::MenuItem("Image Info", nullptr, m_infoOpen)) m_infoOpen = !m_infoOpen;
        ImGui::Separator();
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

    // Grouped controls are drawn inside a collapsing header, in first-seen
    // order. params() puts each algorithm instance in its own group, so a
    // script comparing two filters gets two independently foldable boxes
    // instead of one flat list of ambiguous names.
    std::vector<std::string> groups;          // "" first, then in declaration order
    for (const UiControl& c : m_ui.Controls())
        if (std::find(groups.begin(), groups.end(), c.group) == groups.end())
            groups.push_back(c.group);

    for (const std::string& group : groups) {
        bool open = true;
        if (!group.empty()) {
            ImGui::PushID(group.c_str());
            open = ImGui::CollapsingHeader(group.c_str(), ImGuiTreeNodeFlags_DefaultOpen);
        }

        if (open) {
            // Give the label column a fixed share so long names are readable
            // and the sliders still line up. Without this the widget takes the
            // full width and the text is pushed off the panel.
            const float avail = ImGui::GetContentRegionAvail().x;
            ImGui::PushItemWidth(std::max(80.0f, avail * 0.55f));
            for (UiControl& c : m_ui.Controls()) {
                if (c.group != group) continue;
                DrawControl(c);
            }
            ImGui::PopItemWidth();
        }

        if (!group.empty()) ImGui::PopID();
    }
    ImGui::End();
}

// One control row, plus the reset gestures shared by all of them.
void App::DrawControl(UiControl& c) {
    // The visible text; `label` stays unique but is often far too long to read.
    const std::string& shown = c.display.empty() ? c.label : c.display;
    // ImGui identifies widgets by label, so the unique one goes after "##".
    const std::string widgetId = shown + "##" + c.label;

    {
        switch (c.kind) {
            case UiControl::Kind::Slider: {
                float v = float(c.value);
                // Drag over the soft range when one is declared; the full range
                // stays reachable by ctrl+click, which AlwaysClamp bounds.
                const bool soft = c.softLo != c.softHi;
                const float slo = float(soft ? c.softLo : c.lo);
                const float shi = float(soft ? c.softHi : c.hi);
                const char* fmt = (c.step > 0.0 && c.step >= 0.01) ? "%.2f" : "%.3f";

                bool changed = false;
                if (ImGui::SliderFloat(widgetId.c_str(), &v, slo, shi, fmt,
                                       ImGuiSliderFlags_AlwaysClamp))
                    changed = true;

                // Same fine-tuning affordances as the inspector rows: arrows
                // step, right-click resets, ctrl+click types.
                if (ImGui::IsItemActive() || ImGui::IsItemFocused()) {
                    const float s = c.step > 0.0 ? float(c.step) : (shi - slo) * 0.01f;
                    if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow,  true)) { v -= s; changed = true; }
                    if (ImGui::IsKeyPressed(ImGuiKey_RightArrow, true)) { v += s; changed = true; }
                }
                if (ImGui::IsItemHovered()) {
                    if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
                        v = float(c.def);
                        changed = true;
                    }
                    // What it does comes first -- that is the question an
                    // unfamiliar algorithm raises. Then the full name (the row
                    // shows only the short one), then the range, then the
                    // gestures, which are the least interesting once known.
                    ImGui::BeginTooltip();
                    if (!c.help.empty()) {
                        ImGui::TextUnformatted(c.help.c_str());
                        ImGui::Separator();
                    }
                    ImGui::TextDisabled("%s", c.label.c_str());
                    ImGui::TextDisabled("range %g .. %g   default %g", c.lo, c.hi, c.def);
                    ImGui::TextDisabled(
                        "ctrl+click to type   arrows to step   right-click to reset");
                    ImGui::EndTooltip();
                }

                if (changed) {
                    // Snap to the step from the declared minimum, so a window
                    // of 3..201 step 2 lands on odd sizes.
                    if (c.step > 0.0) {
                        const double n = std::round((double(v) - c.lo) / c.step);
                        v = float(c.lo + n * c.step);
                    }
                    c.value = std::clamp(double(v), c.lo, c.hi);
                    m_dirty = true;
                }
                break;
            }
            case UiControl::Kind::Check: {
                bool b = c.value != 0;
                if (ImGui::Checkbox(widgetId.c_str(), &b)) {
                    c.value = b ? 1 : 0;
                    m_dirty = true;
                }
                break;
            }
            case UiControl::Kind::Choose: {
                if (c.options.empty()) break;
                const int sel = std::clamp(c.selected, 0, int(c.options.size()) - 1);
                if (ImGui::BeginCombo(widgetId.c_str(), c.options[size_t(sel)].c_str())) {
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
        // Sliders set their own richer tooltip above; this covers the rest.
        if (c.kind != UiControl::Kind::Slider && ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            if (!c.help.empty()) {
                ImGui::TextUnformatted(c.help.c_str());
                ImGui::Separator();
            }
            ImGui::TextDisabled("%s", c.label.c_str());
            if (IsModified(c))
                ImGui::TextDisabled("default: %s   double-click to reset",
                                    DefaultText(c).c_str());
            ImGui::EndTooltip();
        }
    }
}

bool App::IsModified(const UiControl& c) {
    if (c.kind == UiControl::Kind::Choose) return c.selected != c.defaultIndex;
    return c.value != c.def;
}

void App::ResetControl(UiControl& c) {
    // Back to the script-declared default, which is the first option only
    // when the script did not name one.
    if (c.kind == UiControl::Kind::Choose) c.selected = c.defaultIndex;
    else                                   c.value = c.def;
}

std::string App::DefaultText(const UiControl& c) {
    char buf[64];
    switch (c.kind) {
        case UiControl::Kind::Check:
            return c.def != 0 ? "on" : "off";
        case UiControl::Kind::Choose:
            return c.options.empty() ? std::string("-")
                                     : c.options[size_t(std::clamp(c.defaultIndex, 0,
                                           int(c.options.size()) - 1))];
        case UiControl::Kind::Slider:
        default:
            std::snprintf(buf, sizeof(buf), "%.3f", c.def);
            return buf;
    }
}

// CPU vs GPU for one algorithm: both results, their difference, and the
// numbers. A kernel that merely *looks* right is not verified.
// True when the named viewer was on screen at its last Draw().
//
// Used to decide whether a remembered selection is still worth following: when
// viewers are tabbed together, selecting a different tab hides the previous
// one, and the info panel should move with it rather than keep describing an
// image the user can no longer see.
bool App::ViewerVisible(const std::string& name) const {
    for (const auto& v : m_views)
        if (v->Name() == name) return v->Visible();
    return false;
}

// Image information: dimensions, an RGB histogram, and capture settings.
//
// The histogram is what a photo editor shows and for the same reason: it says
// at a glance whether an image is clipped, low-contrast, or badly exposed,
// which is exactly what decides how a filter or threshold will behave on it.
void App::DrawInfoPanel() {
    if (!m_infoOpen) return;
    if (!ImGui::Begin("Image Info", &m_infoOpen)) { ImGui::End(); return; }

    // Describes whichever viewer was selected last, falling back to the first
    // palette entry when none has been.
    //
    // Deliberately the *last* selection rather than what is focused right now:
    // clicking a slider moves focus to the Controls panel, so a live focus test
    // reports no viewer at all and the panel silently reverts to describing the
    // source image -- exactly when the user is watching the histogram to see
    // what their parameter change did.
    Image*      shown = nullptr;
    std::string shownName;
    std::string filePath;

    // Clicking a viewer selects it explicitly.
    for (auto& v : m_views)
        if (v->Focused()) { m_infoViewer = v->Name(); break; }

    // Otherwise follow whichever viewer is actually on screen. Without this the
    // panel described the source image at startup: nothing has been clicked
    // yet, so no viewer reports focus, even though a result tab is plainly the
    // one being looked at. It also keeps the panel in step when the user
    // switches tabs by any means other than a click.
    if (m_infoViewer.empty() || !ViewerVisible(m_infoViewer)) {
        for (auto& v : m_views)
            if (v->Visible()) { m_infoViewer = v->Name(); break; }
    }

    if (!m_infoViewer.empty()) {
        for (ViewerImage& vi : m_viewerImages)
            if (vi.name == m_infoViewer && vi.image.Valid()) {
                shown     = &vi.image;
                shownName = vi.name;
                break;
            }
    }
    if (!shown) {
        for (PaletteEntry& e : m_palette) {
            if (!std::holds_alternative<Image>(e.data)) continue;
            Image& img = std::get<Image>(e.data);
            if (!img.Valid()) continue;
            shown     = &img;
            shownName = e.name;
            filePath  = e.path;
            break;
        }
    } else {
        // A viewer shows a processed result, so its EXIF is that of whichever
        // palette image fed the pipeline -- there is only one source in
        // practice, and claiming otherwise would be worse than saying nothing.
        for (PaletteEntry& e : m_palette)
            if (std::holds_alternative<Image>(e.data)) { filePath = e.path; break; }
    }

    if (!shown) {
        ImGui::TextDisabled("No image.");
        ImGui::End();
        return;
    }

    const ImageDesc d = shown->Desc();
    ImGui::TextUnformatted(shownName.c_str());
    ImGui::Separator();

    const char* fmt = "?";
    switch (d.format) {
        case Format::RGBA8:   fmt = "RGBA8";   break;
        case Format::R32F:    fmt = "R32F";    break;
        case Format::RGBA32F: fmt = "RGBA32F"; break;
        case Format::RGBA16F: fmt = "RGBA16F"; break;
        default: break;
    }
    ImGui::Text("%d x %d  (%.1f MP)", d.width, d.height,
                double(d.width) * double(d.height) / 1e6);
    ImGui::Text("%s   %.1f MB", fmt, double(d.SizeInBytes()) / 1e6);
    if (!filePath.empty()) {
        ImGui::TextDisabled("%s", filePath.c_str());
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", filePath.c_str());
    }

    // --- histogram ----------------------------------------------------------
    ImGui::Spacing();
    ImGui::SeparatorText("Histogram");

    // Computed on a background thread. Four 256-bin passes over an 8 MP image
    // is tens of milliseconds -- far too much to spend in a frame for a panel
    // that is purely informational. A request goes out when the subject or the
    // content changes; until it returns, the previous bins stay on screen.
    // The version of the viewer this panel is showing, not a global one: with
    // per-viewer versions a global counter would either never change (and the
    // histogram would freeze) or change for every viewer (and it would
    // recompute needlessly).
    uint64_t shownVersion = 0;
    for (const ViewerImage& vi : m_viewerImages)
        if (vi.name == shownName) { shownVersion = vi.version; break; }

    if (m_stats.source != shownName || m_stats.version != shownVersion) {
        if (!m_statsRequested ||
            m_requestedSource != shownName || m_requestedVersion != shownVersion) {
            if (GetEnvironmentVariableA("TGLAB_INFODBG", nullptr, 0) > 0) {
                std::fprintf(stderr, "[info] subject=%s version=%llu\n",
                             shownName.c_str(), (unsigned long long)shownVersion);
                std::fflush(stderr);
            }
            m_statsWorker.Request(*shown, shownName, shownVersion);
            m_statsRequested   = true;
            m_requestedSource  = shownName;
            m_requestedVersion = shownVersion;
        }
    }

    // Drawn by hand rather than with PlotHistogram: three overlapping additive
    // curves is what makes an RGB histogram readable, and stacked separate
    // plots would not show where the channels diverge.
    {
        const ImVec2 size(ImGui::GetContentRegionAvail().x, 120.0f);
        const ImVec2 p0 = ImGui::GetCursorScreenPos();
        ImDrawList* dl = ImGui::GetWindowDrawList();

        dl->AddRectFilled(p0, ImVec2(p0.x + size.x, p0.y + size.y),
                          IM_COL32(20, 20, 24, 255));

        auto curve = [&](const std::vector<float>& bins, ImU32 colour) {
            if (bins.empty()) return;
            const float dx = size.x / float(255);
            for (int i = 0; i < 255; ++i) {
                const float h0 = std::min(bins[size_t(i)] / m_stats.peak, 1.0f);
                const float h1 = std::min(bins[size_t(i + 1)] / m_stats.peak, 1.0f);
                dl->AddLine(ImVec2(p0.x + dx * float(i),     p0.y + size.y * (1.0f - h0)),
                            ImVec2(p0.x + dx * float(i + 1), p0.y + size.y * (1.0f - h1)),
                            colour, 1.0f);
            }
        };

        if (m_stats.valid) {
            if (m_stats.r.empty()) {
                curve(m_stats.luma, IM_COL32(200, 200, 200, 255));
            } else {
                curve(m_stats.r, IM_COL32(255, 80, 80, 200));
                curve(m_stats.g, IM_COL32(80, 220, 80, 200));
                curve(m_stats.b, IM_COL32(90, 140, 255, 200));
            }
        }

        // Quarter marks, so a clipped highlight or crushed shadow is locatable
        // rather than merely visible.
        for (int i = 1; i < 4; ++i) {
            const float x = p0.x + size.x * float(i) / 4.0f;
            dl->AddLine(ImVec2(x, p0.y), ImVec2(x, p0.y + size.y),
                        IM_COL32(255, 255, 255, 24), 1.0f);
        }

        ImGui::Dummy(size);
    }

    if (!m_stats.valid) {
        ImGui::TextDisabled("computing...");
    } else {
        // Float images live in 0..1, so "%.1f" showed 0.0 for everything. Three
        // decimals there, one for the 0..255 case where they would be noise.
        if (m_stats.scale <= 1.5) {
            ImGui::Text("mean %.3f   median %.3f   stddev %.3f",
                        m_stats.mean, m_stats.median, m_stats.stddev);
            if (m_stats.hasHeadroom)
                ImGui::TextDisabled("headroom to %.2f (%.1f stops above white)",
                                    m_stats.maxValue, std::log2(m_stats.maxValue));
        } else
        ImGui::Text("mean %.1f   median %.1f   stddev %.1f",
                    m_stats.mean, m_stats.median, m_stats.stddev);
        // Clipping is the thing worth flagging: it is unrecoverable, and it
        // changes what a threshold or a normalising filter will do.
        if (m_stats.clipLow > 0.01 || m_stats.clipHigh > 0.01)
            ImGui::TextDisabled("clipped: %.1f%% black, %.1f%% white",
                                m_stats.clipLow * 100.0, m_stats.clipHigh * 100.0);
        // Says plainly that what is on screen is one image behind, rather than
        // letting a stale curve look current.
        if (m_stats.source != shownName || m_stats.version != shownVersion)
            ImGui::TextDisabled("(updating...)");
    }

    // --- capture settings ---------------------------------------------------
    if (!filePath.empty() && filePath != m_infoExifPath) {
        m_infoExifPath = filePath;
        m_infoExif     = ReadExif(filePath);
    }

    ImGui::Spacing();
    ImGui::SeparatorText("Capture");
    if (!m_infoExif.present) {
        // Said plainly: most scans and every PNG have none, and an empty
        // section would read as a bug.
        ImGui::TextDisabled("No EXIF metadata in this file.");
    } else {
        auto row = [](const char* label, const std::string& value) {
            if (value.empty()) return;
            ImGui::TextDisabled("%-9s", label);
            ImGui::SameLine();
            ImGui::TextUnformatted(value.c_str());
        };
        std::string camera = m_infoExif.cameraMake;
        if (!m_infoExif.cameraModel.empty()) {
            if (!camera.empty()) camera += " ";
            camera += m_infoExif.cameraModel;
        }
        row("camera", camera);
        row("lens", m_infoExif.lens);
        row("exposure", m_infoExif.exposureTime);
        row("aperture", m_infoExif.aperture);
        row("iso", m_infoExif.iso);
        row("focal", m_infoExif.focalLength);
        row("taken", m_infoExif.dateTaken);
    }

    ImGui::End();
}

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
        ImGui::Separator();
        ImGui::TextDisabled("Reference from a script as image(\"name\")");
        ImGui::End();
        return;
    }

    const float thumbSize = 48.0f;
    int removeIndex = -1;

    for (int i = 0; i < int(m_palette.size()); ++i) {
        PaletteEntry& e = m_palette[size_t(i)];
        ImGui::PushID(i);

        const ImVec2 rowStart = ImGui::GetCursorScreenPos();

        // Thumbnail. Downsampled ONCE into a small image rather than uploading
        // the full-resolution source: an 8 MP scan is a 33 MB texture, and
        // building one every frame to draw a 48 px icon locks the UI thread
        // solid -- which looks exactly like the whole app hanging.
        if (std::holds_alternative<Image>(e.data)) {
            Image& img = std::get<Image>(e.data);
            if (img.Valid() && e.thumbVersion != e.version) {
                MakeThumbnail(img, int(thumbSize) * 2, &e.thumbImage);
                e.thumbVersion = e.version;
            }
            if (e.thumbImage.Valid() && e.thumb.Update(m_dev, e.thumbImage, e.version)) {
                const float iw = float(e.thumb.Width());
                const float ih = float(e.thumb.Height());
                const float scale = (iw > 0 && ih > 0) ? std::min(thumbSize / iw, thumbSize / ih) : 1.0f;
                const ImVec2 size(iw * scale, ih * scale);
                // Centre inside a fixed box so rows line up whatever the aspect.
                const ImVec2 p = ImGui::GetCursorScreenPos();
                ImGui::GetWindowDrawList()->AddImage(
                    ImTextureRef(static_cast<ImTextureID>(e.thumb.Handle().ptr)),
                    ImVec2(p.x + (thumbSize - size.x) * 0.5f, p.y + (thumbSize - size.y) * 0.5f),
                    ImVec2(p.x + (thumbSize + size.x) * 0.5f, p.y + (thumbSize + size.y) * 0.5f));
            }
        }
        ImGui::Dummy(ImVec2(thumbSize, thumbSize));
        ImGui::SameLine();

        // Name, with the file it came from underneath. Showing both matters now
        // that they are independent.
        ImGui::BeginGroup();
        if (m_renamingSlot == i) {
            ImGui::SetNextItemWidth(-1.0f);
            if (ImGui::IsWindowAppearing() || ImGui::IsItemHovered()) {}
            if (!m_renameFocused) { ImGui::SetKeyboardFocusHere(); m_renameFocused = true; }
            if (ImGui::InputText("##rename", m_renameBuf, sizeof(m_renameBuf),
                                 ImGuiInputTextFlags_EnterReturnsTrue) ) {
                CommitRename(i);
            }
            // Clicking away or pressing Escape cancels rather than half-applying.
            if (ImGui::IsKeyPressed(ImGuiKey_Escape) ||
                (!ImGui::IsItemActive() && !ImGui::IsItemFocused() && m_renameFocused &&
                 !ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left))) {
                m_renamingSlot = -1;
                m_renameFocused = false;
            }
        } else {
            ImGui::TextUnformatted(e.name.c_str());
            if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
                BeginRename(i);
        }

        // Filename only, with the full path on hover.
        std::string file = e.path;
        if (auto slash = file.find_last_of("/\\"); slash != std::string::npos)
            file = file.substr(slash + 1);
        ImGui::TextDisabled("%s", file.c_str());

        if (std::holds_alternative<Image>(e.data)) {
            const ImageDesc& d = std::get<Image>(e.data).Desc();
            ImGui::TextDisabled("%d x %d", d.width, d.height);
        }
        ImGui::EndGroup();

        // Row rect, for drop targeting. Measured from the window's own content
        // edges rather than the cursor: after EndGroup() the cursor has moved,
        // so GetContentRegionAvail() there reports what is left below the row,
        // not the row's width.
        const float rowHeight = std::max(thumbSize, ImGui::GetCursorScreenPos().y - rowStart.y);
        const float left  = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMin().x;
        const float right = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;
        e.rowMin = ImVec2(left, rowStart.y);
        e.rowMax = ImVec2(right, rowStart.y + rowHeight);

        if (ImGui::BeginPopupContextItem("slot")) {
            if (ImGui::MenuItem("Rename...")) BeginRename(i);
            if (ImGui::MenuItem("Reload from disk")) {
                const std::string p = e.path, n = e.name;
                LoadImageIntoPalette(p, n);
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Remove from palette")) removeIndex = i;
            ImGui::EndPopup();
        }

        ImGui::PopID();
        ImGui::Separator();
    }

    if (removeIndex >= 0) {
        // Release the GPU texture before the entry goes, while the device lives.
        m_palette[size_t(removeIndex)].thumb.Release();
        m_palette.erase(m_palette.begin() + removeIndex);
        m_dirty = true;
    }

    ImGui::TextDisabled("Reference from a script as image(\"name\")");
    ImGui::TextDisabled("Drop a file on a row to swap it, keeping the name.");
    ImGui::End();
}


// Which palette row is under this screen point, or -1. Used to route a dropped
// file onto a specific slot rather than adding a new one.
int App::SlotAtScreenPos(int sx, int sy) const {
    for (int i = 0; i < int(m_palette.size()); ++i) {
        const PaletteEntry& e = m_palette[size_t(i)];
        if (float(sx) >= e.rowMin.x && float(sx) < e.rowMax.x &&
            float(sy) >= e.rowMin.y && float(sy) < e.rowMax.y)
            return i;
    }
    return -1;
}

std::string App::SlotNameAt(int sx, int sy) const {
    const int i = SlotAtScreenPos(sx, sy);
    return i >= 0 ? m_palette[size_t(i)].name : std::string();
}
// Starts an inline rename of slot `i`.
void App::BeginRename(int i) {
    if (i < 0 || size_t(i) >= m_palette.size()) return;
    m_renamingSlot = i;
    m_renameFocused = false;
    std::snprintf(m_renameBuf, sizeof(m_renameBuf), "%s", m_palette[size_t(i)].name.c_str());
}

// Applies the pending rename, rejecting names that would collide or be empty --
// two slots with the same name would make image("x") ambiguous.
void App::CommitRename(int i) {
    m_renamingSlot = -1;
    m_renameFocused = false;
    if (i < 0 || size_t(i) >= m_palette.size()) return;

    std::string want = m_renameBuf;
    // Trim, since a trailing space in a script name is invisible and baffling.
    while (!want.empty() && std::isspace(static_cast<unsigned char>(want.front()))) want.erase(want.begin());
    while (!want.empty() && std::isspace(static_cast<unsigned char>(want.back())))  want.pop_back();
    if (want.empty()) return;

    for (int j = 0; j < int(m_palette.size()); ++j)
        if (j != i && m_palette[size_t(j)].name == want) {
            m_error = "another image is already named '" + want + "'";
            return;
        }

    m_palette[size_t(i)].name = want;
    m_dirty = true;   // the script may now resolve differently
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

// Worst frame time since startup. A drop that makes the app feel frozen shows
// up here even with no console attached, which is the only way to see it on a
// machine that is not the developer's.
static double g_worstFrameMs = 0.0;

void App::Frame() {
    // TGLAB_FRAMEDBG=1 reports slow frames. A frame is ~16 ms; anything much
    // over that is UI-thread work that should not be there, and a run of them
    // is what "the whole app locked up" actually looks like.
    const auto frameStart = std::chrono::steady_clock::now();
    struct FrameTimer {
        std::chrono::steady_clock::time_point t0;
        ~FrameTimer() {
            // Recorded unconditionally: the status line shows it, so a hitch is
            // visible without setting an environment variable first.
            const double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
            if (ms > g_worstFrameMs) g_worstFrameMs = ms;

            static const bool on = GetEnvironmentVariableA("TGLAB_FRAMEDBG", nullptr, 0) > 0;
            if (on && ms > 50.0) {
                std::fprintf(stderr, "[frame] %.1f ms\n", ms);
                std::fflush(stderr);
            }
        }
    } frameTimer{frameStart};

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
    PollLoader();
    PollStats();
    PollWorker();
    ReleaseRetiredViews();

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

    // Self-test hook: TGLAB_DROPTEST="x,y,path" simulates dropping `path` at
    // client point (x,y), which is the one interaction a harness cannot drive
    // -- a synthetic WM_DROPFILES does not carry a usable drop point.
    // Several files may be listed after the point, separated by ';'. Each is
    // dropped in turn, a few frames apart, so the texture is resized while
    // earlier frames are still in flight -- the case that hung the device.
    // Self-test hook: TGLAB_INFOTEST="<viewer>" selects that viewer for the
    // info panel, then a few frames later nudges the first slider. Reproduces
    // the sequence that made the panel silently revert to the source image --
    // selecting a viewer and then touching a control cannot be driven any other
    // way, and a live focus test passes right up until the control is touched.
    if (m_infoTestCountdown > 0 && --m_infoTestCountdown == 0) {
        char buf[128] = {};
        if (GetEnvironmentVariableA("TGLAB_INFOTEST", buf, sizeof(buf)) > 0) {
            if (!m_infoTestNudged) {
                m_infoViewer = buf;          // as if the user clicked that viewer
                AppTrace(("self-test: info panel following '" + m_infoViewer + "'").c_str());
                m_infoTestNudged   = true;
                m_infoTestCountdown = 30;    // then change a parameter
            } else {
                for (UiControl& c : m_ui.Controls()) {
                    if (c.kind != UiControl::Kind::Slider) continue;
                    c.value = c.value + (c.hi - c.lo) * 0.1;
                    m_dirty = true;
                    AppTrace("self-test: nudged a slider");
                    break;
                }
            }
        }
    }

    if (m_dropTestCountdown > 0 && --m_dropTestCountdown == 0) {
        char buf[1024] = {};
        if (GetEnvironmentVariableA("TGLAB_DROPTEST", buf, sizeof(buf)) > 0) {
            std::string spec(buf);
            const size_t c1 = spec.find(',');
            const size_t c2 = spec.find(',', c1 == std::string::npos ? 0 : c1 + 1);
            if (c1 != std::string::npos && c2 != std::string::npos) {
                const int dx = std::atoi(spec.substr(0, c1).c_str());
                const int dy = std::atoi(spec.substr(c1 + 1, c2 - c1 - 1).c_str());

                std::string files = spec.substr(c2 + 1);
                std::string file  = files;
                if (const size_t semi = files.find(';'); semi != std::string::npos) {
                    file = files.substr(0, semi);
                    // Rewrite the variable so the next firing takes the next file.
                    SetEnvironmentVariableA(
                        "TGLAB_DROPTEST",
                        (spec.substr(0, c2 + 1) + files.substr(semi + 1)).c_str());
                    m_dropTestCountdown = 20;
                }

                const std::string slot = SlotNameAt(dx, dy);
                AppTrace(("self-test: drop '" + file + "' at (" + std::to_string(dx) + "," +
                          std::to_string(dy) + ") -> slot '" + slot + "'").c_str());
                // The real drop path: queue on the loader thread, as
                // WM_DROPFILES does, rather than decoding inline.
                RequestImageLoad(file, slot);
            }
        }
    }

    // Self-test hook: TGLAB_SWITCH=<path> switches to that script after a few
    // frames, driving exactly the path a Scripts-panel click takes. Clicking
    // cannot be driven reliably from a test harness, and this is the one
    // interaction that wedged the UI.
    if (m_switchCountdown > 0 && --m_switchCountdown == 0) {
        char buf[MAX_PATH] = {};
        if (GetEnvironmentVariableA("TGLAB_SWITCH", buf, MAX_PATH) > 0) {
            AppTrace(("self-test: switching to " + std::string(buf)).c_str());
            SetScriptPath(buf);
        }
    }

    if (ImGui::IsKeyPressed(ImGuiKey_F5, false)) m_dirty = true;

    DrawMenuBar();
    // DrawPalettePanel() uploads thumbnail textures, so it runs after
    // BeginFrame() opens the command list -- see below with the image views.
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

        // Timing plus where the work ran — the numbers you want side by side
        // when deciding whether a GPU port was worth it. Cached stages are
        // shown too, so the counts always account for every stage rather than
        // reading as "nothing ran" after a cache hit.
        const char* modeName = "auto";
        switch (m_worker.GetExecMode()) {
            case ExecMode::ForceCPU: modeName = "CPU"; break;
            case ExecMode::ForceGPU: modeName = "GPU"; break;
            case ExecMode::Auto:     modeName = "auto"; break;
        }
        // A large scan on a slow drive takes real time to decode. Saying so
        // beats an unexplained pause, which reads as a hang.
        if (const int pending = m_loader.Pending(); pending > 0) {
            ImGui::SameLine();
            ImGui::TextDisabled("| loading %d image%s...", pending, pending == 1 ? "" : "s");
        }
        if (g_worstFrameMs > 100.0) {
            ImGui::SameLine();
            ImGui::TextDisabled("| worst frame %.0f ms", g_worstFrameMs);
        }

        const int cached = m_worker.LastCachedStages();
        if (cached > 0) {
            ImGui::TextDisabled("last run %.1f ms   %d CPU / %d GPU / %d cached   [%s]",
                                m_worker.LastRunMs(), m_worker.LastCpuStages(),
                                m_worker.LastGpuStages(), cached, modeName);
        } else {
            ImGui::TextDisabled("last run %.1f ms   %d CPU / %d GPU stage(s)   [%s]",
                                m_worker.LastRunMs(), m_worker.LastCpuStages(),
                                m_worker.LastGpuStages(), modeName);
        }

        // Video memory. A 45 MP intermediate is ~340 MB in RGBA16F, so a
        // pipeline of several stages can approach the card's budget -- at which
        // point the driver starts paging and everything slows down for a reason
        // that is otherwise completely invisible.
        {
            uint64_t used = 0, budget = 0;
            m_dev.VideoMemory(&used, &budget);
            if (budget > 0) {
                const double pct = 100.0 * double(used) / double(budget);
                const ImVec4 colour =
                    pct > 90.0 ? ImVec4(1.0f, 0.4f, 0.4f, 1.0f)     // paging territory
                  : pct > 75.0 ? ImVec4(1.0f, 0.8f, 0.3f, 1.0f)
                               : ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
                ImGui::TextColored(colour, "vram %.0f / %.0f MB  (%.0f%%)",
                                   double(used) / (1024.0 * 1024.0),
                                   double(budget) / (1024.0 * 1024.0), pct);
            }
        }
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

        // Per-viewer, so an unchanged viewer skips the upload entirely rather
        // than re-converting the same pixels because some other viewer moved.
        uint64_t ver = 0;
        for (const ViewerImage& vi : m_viewerImages)
            if (vi.name == v->Name()) { ver = vi.version; break; }
        v->SetContentVersion(ver);
        v->Draw(m_dev, img);
    }

    // These upload textures, so they belong here rather than with the other
    // panels above: BeginFrame() has opened the command list by now.
    DrawPalettePanel();
    DrawComparePanel();
    DrawInfoPanel();

    // Compare and Image Info are drawn here, *after* the DockLooseViewers()
    // call earlier in the frame, so on the frame one of them first appears
    // that call had no window to dock -- and both opened floating. Docking
    // them again now that they exist takes effect on the next frame.
    DockOnDemandPanels();

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

            // Where the file was released, so it can replace the slot under the
            // cursor rather than always adding a new entry.
            //
            // ImGui reports window rects in CLIENT coordinates (multi-viewport
            // is off), so the drop point has to stay in client space too.
            // DragQueryPoint gives client coordinates already; when it reports
            // nothing useful, fall back to the actual cursor position converted
            // the same way.
            POINT pt{};
            if (!DragQueryPoint(drop, &pt) || (pt.x == 0 && pt.y == 0)) {
                GetCursorPos(&pt);
                ScreenToClient(hwnd, &pt);
            }

            std::string target;
            if (g_app) target = g_app->SlotNameAt(pt.x, pt.y);

            const UINT n = DragQueryFileA(drop, 0xFFFFFFFF, nullptr, 0);
            for (UINT i = 0; i < n; ++i) {
                char path[MAX_PATH] = {};
                if (DragQueryFileA(drop, i, path, MAX_PATH) && g_app) {
                    // Only the first file replaces the slot; any others are
                    // added normally, since one row cannot hold several images.
                    g_app->RequestImageLoad(path, i == 0 ? target : std::string());
                }
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
    std::string script = ResolveDataPath("scripts/hello.tgl");
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
    if (!gotImage) app.LoadImageIntoPalette(ResolveDataPath("assets/test.png"));
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
