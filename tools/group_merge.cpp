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

#include "../src/core/image_group.h"
#include "../src/core/pipeline.h"
#include "../src/core/raw_io.h"
#include "../src/script/interp.h"
#include "../src/gpu/compute.h"
#include "../src/script/parser.h"

using namespace tglab;

static double Now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
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
        std::printf("  %s  %dx%d mosaic=%d\n", argv[i], d.width, d.height, int(d.IsMosaic()));
        AppendToGroup(&group, std::move(img), "frame");
    }
    std::printf("loaded %d frames in %.1fs, shape %s\n\n",
                argc - 1, Now() - t0, ShapeOf(group).ToString().c_str());

    const char* src = "frames = image(\"group\")\nmerged = merge_mean(frames)\ndisplay(merged)\n";
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
    ID3D12Device* dev = nullptr;
    if (SUCCEEDED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&dev))))
        gpu.Init(dev);
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
    std::printf("stages run: %d cpu, %d gpu\n", pipe.CpuStageCount(), pipe.GpuStageCount());
    std::printf("execute: %s in %.1fs\n", ok ? "ok" : xerr.c_str(), Now() - t1);
    return ok ? 0 : 1;
}
