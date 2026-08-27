// Reproduces a group of raws through the merge script, headless.
//
// The app hung on this with no error and a frozen spinner, which means the UI
// thread is blocked rather than the work merely being slow. Running the same
// pipeline without any UI separates the two: if this finishes, the fault is in
// the app's threading or display; if it hangs or dies here, it is in the
// pipeline itself.
//
//   group_merge <file1.CR3> <file2.CR3> ...
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include <d3d12.h>
#include <dxgi1_4.h>
#include <windows.h>

#include <algorithm>

#include "../src/algo_util/pixel_buffer.h"
#include "../src/core/image_group.h"
#include "../src/core/pipeline.h"
#include "../src/core/raw_io.h"
#include "../src/script/interp.h"
#include "../src/gpu/compute.h"
#include "../src/gpu/gpu_image.h"
#include "../src/script/parser.h"

using namespace tglab;

static double Now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// Stage to append after the merge, from TGLAB_POST.
static std::string PostAlgo() {
    char buf[64] = {};
    GetEnvironmentVariableA("TGLAB_POST", buf, sizeof buf);
    return buf[0] ? std::string(buf) : std::string("tonemap");
}

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: group_merge <raw> [raw...]\n"); return 2; }

    Data group{};
    const double t0 = Now();
    for (int i = 1; i < argc; ++i) {
        Image img;
        std::string err;
        if (!LoadRawMosaic(argv[i], &img, &err)) {
            std::printf("load %s: %s\n", argv[i], err.c_str());
            return 2;
        }
        const ImageDesc& d = img.Desc();
        // Exposure is printed because merge_hdr depends on it entirely: a
        // bracket whose frames all report the same relative exposure is not a
        // bracket as far as the merge is concerned, and that is worth seeing
        // rather than inferring from a flat-looking result.
        std::printf("  %s  %dx%d mosaic=%d  %.4fs f/%.1f ISO%.0f  rel=%.5f\n",
                    argv[i], d.width, d.height, int(d.IsMosaic()),
                    double(d.shutter), double(d.aperture), double(d.iso),
                    double(d.RelativeExposure()));
        AppendToGroup(&group, std::move(img), "frame");
    }
    std::printf("loaded %d frames in %.1fs, shape %s\n\n",
                argc - 1, Now() - t0, ShapeOf(group).ToString().c_str());

    // TGLAB_MERGE picks the reduction, so the same harness exercises either.
    char algo[64] = "merge_mean";
    GetEnvironmentVariableA("TGLAB_MERGE", algo, sizeof algo);
    const std::string srcStr =
        std::string("frames = image(\"group\")\nmerged = ") + algo +
        "(frames)\n" +
        // TGLAB_POST appends a stage after the merge, so the tone mapper can be
        // measured against the raw merge output in the same run.
        (GetEnvironmentVariableA("TGLAB_POST", nullptr, 0) > 0
             ? std::string("merged = ") + PostAlgo() + "(merged)\n"
             : std::string()) +
        "display(merged)\n";
    const char* src = srcStr.c_str();
    Program prog;
    std::string err;
    if (!Parse(src, &prog, &err)) { std::printf("parse: %s\n", err.c_str()); return 2; }

    SourceImage si;
    si.name     = "group";
    si.index    = 0;
    si.shape    = ShapeOf(group);
    si.isMosaic = true;
    std::vector<SourceImage> names{si};

    UiState ui;
    Pipeline pipe;
    const auto r = Interpret(prog, names, &ui, &pipe);
    if (!r.ok) { std::printf("interpret: %s\n", r.error.c_str()); return 2; }
    std::printf("interpreted: %zu stages\n", pipe.Stages().size());
    for (const auto& s : pipe.Stages()) std::printf("  %s\n", s.algoName.c_str());
    std::printf("\n");

    ComputeContext gpu;
    // Debug layer and DRED, so a device hang leaves breadcrumbs naming the op
    // the GPU stopped on. Opt-in: both cost per-op overhead.
    if (GetEnvironmentVariableA("TGLAB_DEBUG", nullptr, 0) > 0) {
        ID3D12Debug* dbg = nullptr;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dbg)))) {
            dbg->EnableDebugLayer();
            dbg->Release();
        }
        ID3D12DeviceRemovedExtendedDataSettings1* dred = nullptr;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dred)))) {
            dred->SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            dred->SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            dred->Release();
        }
        std::printf("debug layer + DRED on\n");
    }

    ID3D12Device* dev = nullptr;
    if (SUCCEEDED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&dev))))
        gpu.Init(dev);
    // Without this, Image::MapCpuRead cannot pull a GPU-resident image back:
    // the readback hook is null, the assert is compiled out in release, and the
    // caller silently gets an uninitialised buffer. Omitting it here cost an
    // afternoon -- it made every frame pile into one command list, which the
    // GPU watchdog then killed.
    InstallGpuResidencyHooks();
    std::printf("gpu: %s\n", gpu.Ready() ? "yes" : "no");

    // Per-frame VRAM, to see whether a fused reduction releases each frame or
    // accumulates them.
    IDXGIFactory4* factory = nullptr;
    IDXGIAdapter3* adapter = nullptr;
    if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
        IDXGIAdapter1* a1 = nullptr;
        if (SUCCEEDED(factory->EnumAdapters1(0, &a1))) a1->QueryInterface(&adapter);
        if (a1) a1->Release();
    }
    if (adapter) {
        g_frameTrace = [adapter](int f) {
            DXGI_QUERY_VIDEO_MEMORY_INFO info{};
            if (SUCCEEDED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info)))
                std::printf("  frame %d: vram %.0f MB used, %.0f MB budget\n", f,
                            double(info.CurrentUsage) / (1024.0 * 1024.0),
                            double(info.Budget) / (1024.0 * 1024.0));
        };
    }

    std::vector<Data> sources;
    sources.push_back(std::move(group));

    const double t1 = Now();
    std::string xerr;
    const bool ok = pipe.Execute(&sources, nullptr, &xerr, gpu.Ready() ? &gpu : nullptr,
                                 ExecMode::Auto);
    if (adapter) {
        DXGI_QUERY_VIDEO_MEMORY_INFO info{};
        if (SUCCEEDED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info)))
            std::printf("vram at end: %.0f MB\n", double(info.CurrentUsage) / (1024.0 * 1024.0));
    }
    std::printf("stages run: %d cpu, %d gpu\n", pipe.CpuStageCount(), pipe.GpuStageCount());

    // What the merge actually produced. For an HDR merge the headroom is the
    // whole point -- a maximum at or below 1.0 would mean the extra range the
    // bracket captured was thrown away somewhere.
    if (ok && !pipe.Viewers().empty()) {
        if (const Data* d = pipe.Resolve(pipe.Viewers().back().source, &sources)) {
            if (const auto* im = std::get_if<Image>(d)) {
                ImageView v = const_cast<Image&>(*im).MapCpuRead();
                if (v.data) {
                    PixelBuffer pb;
                    pb.Unpack(v);
                    const std::vector<float>& px = pb.Data();
                    float lo = 1e30f, hi = -1e30f;
                    double sum = 0.0;
                    size_t n = 0;
                    const int ch = pb.Channels();
                    for (size_t i = 0; i < px.size(); ++i) {
                        if (ch == 4 && (i % 4) == 3) continue;   // skip alpha
                        lo = std::min(lo, px[i]);
                        hi = std::max(hi, px[i]);
                        sum += double(px[i]);
                        ++n;
                    }
                    std::printf("result: %dx%d  min %.5f  max %.4f  mean %.5f\n",
                                im->Desc().width, im->Desc().height,
                                double(lo), double(hi), n ? sum / double(n) : 0.0);
                }
            }
        }
    }

    for (const std::string& r : pipe.GpuFallbacks())
        std::printf("gpu fallback: %s\n", r.c_str());
    for (const auto& st : pipe.Stages())
        if (st.algo) {
            const std::string rep = st.algo->RunReport();
            if (!rep.empty()) std::printf("report: %s\n", rep.c_str());
        }
    std::printf("execute: %s in %.1fs\n", ok ? "ok" : xerr.c_str(), Now() - t1);
    return ok ? 0 : 1;
}
