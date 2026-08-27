// gpu_leak_scripts — VRAM growth when SWITCHING between scripts.
//
// Distinct from gpu_leak, which re-runs one pipeline many times and measures
// steady state. That case passes: a pipeline reuses its kernels, its scratch
// and its outputs across runs, so nothing accumulates.
//
// This is the case Tim reported -- clicking from script to script, watching
// VRAM climb. Each script builds a NEW pipeline with new stages, and every
// stage holds GPU resources: compiled kernels, a ping-pong scratch image, and
// for a multi-pass stage a pool of full-size scratch planes. If the old
// pipeline's resources are not released when it is replaced, switching scripts
// leaks a whole pipeline's worth each time.
//
// Cycles through the real scripts repeatedly and reports the trend, for the
// same reason gpu_leak does: the first pass legitimately allocates as each
// script is seen for the first time, and a two-point comparison reads that
// warm-up as a leak.
//
//   gpu_leak_scripts [cycles] [image]
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <d3d12.h>
#include <dxgi1_4.h>

#include "../src/core/image.h"
#include "../src/core/image_io.h"
#include "../src/core/pipeline.h"
#include "../src/core/raw_io.h"
#include "../src/gpu/compute.h"
#include "../src/script/interp.h"
#include "../src/script/parser.h"

using namespace tglab;

namespace {

// Local VRAM in use by this process, from DXGI rather than a driver total, so
// another application's allocations do not appear as our growth.
double UsedMb(IDXGIAdapter3* adapter) {
    DXGI_QUERY_VIDEO_MEMORY_INFO info{};
    if (FAILED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info)))
        return 0.0;
    return double(info.CurrentUsage) / (1024.0 * 1024.0);
}

std::string ReadFile(const std::string& path) {
    FILE* f = nullptr;
    if (fopen_s(&f, path.c_str(), "rb") != 0 || !f) return {};
    std::string s;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
    fclose(f);
    return s;
}

} // namespace

int main(int argc, char** argv) {
    const int cycles = (argc > 1) ? std::atoi(argv[1]) : 6;
    const char* imgPath = (argc > 2) ? argv[2] : "assets/test.png";

    // Every script that takes a plain image. demosaic.tgl and the raw-only ones
    // are excluded so this runs on the committed test asset rather than needing
    // a photograph on disk.
    const char* kScripts[] = {
        "scripts/hello.tgl",
        "scripts/edges.tgl",
        "scripts/filters.tgl",
        "scripts/thresholds.tgl",
        "scripts/gradient.tgl",
    };

    Image img;
    std::string err;
    if (IsRawExtension(imgPath)) {
        if (!LoadRawMosaic(imgPath, &img, &err)) {
            std::printf("could not load %s: %s\n", imgPath, err.c_str());
            return 2;
        }
    } else if (!LoadImageFile(imgPath, &img, &err)) {
        std::printf("could not load %s: %s\n", imgPath, err.c_str());
        return 2;
    }

    ID3D12Device* dev = nullptr;
    if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&dev)))) {
        std::printf("no D3D12 device; skipping\n");
        return 0;
    }
    IDXGIFactory4* factory = nullptr;
    IDXGIAdapter3* adapter = nullptr;
    if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
        IDXGIAdapter1* a1 = nullptr;
        if (SUCCEEDED(factory->EnumAdapters1(0, &a1))) a1->QueryInterface(&adapter);
        if (a1) a1->Release();
    }
    if (!adapter) { std::printf("no DXGI adapter; skipping\n"); return 0; }

    ComputeContext gpu;
    if (!gpu.Init(dev)) { std::printf("compute init failed; skipping\n"); return 0; }

    std::printf("Switching between %zu scripts, %d cycles, on %s.\n\n",
                sizeof(kScripts) / sizeof(kScripts[0]), cycles, imgPath);
    std::printf("  %-6s %-24s %10s\n", "cycle", "script", "vram MB");

    std::vector<double> perCycle;
    // Mirrors PipelineWorker::Run(): the worker keeps the PREVIOUS pipeline so
    // Execute() can move cached outputs out of it, then the freshly-built one
    // becomes the previous for the next job. Reproducing that structure is the
    // point -- a leak here is about what happens to the old pipeline's GPU
    // resources when it is replaced, and a test that held only one pipeline
    // would not exercise it.
    Pipeline prev;
    bool havePrev = false;

    for (int c = 0; c < cycles; ++c) {
        for (const char* s : kScripts) {
            const std::string src = ReadFile(s);
            if (src.empty()) { std::printf("  (missing %s)\n", s); continue; }

            Program prog;
            std::string perr;
            if (!Parse(src, &prog, &perr)) { std::printf("  parse %s: %s\n", s, perr.c_str()); continue; }

            SourceImage si;
            si.name = "test";
            si.index = 0;
            si.isMosaic = img.Desc().IsMosaic();
            std::vector<SourceImage> names{si};

            UiState ui;
            Pipeline built;
            const auto r = Interpret(prog, names, &ui, &built);
            if (!r.ok) { std::printf("  interp %s: %s\n", s, r.error.c_str()); continue; }

            std::vector<Data> sources;
            sources.push_back(Data{img.Clone()});
            std::string eerr;
            built.Execute(&sources, havePrev ? &prev : nullptr, &eerr,
                          &gpu, ExecMode::Auto);

            // This job's pipeline becomes the previous one, which releases
            // whatever the old `prev` still held. Exactly the worker's move.
            prev = std::move(built);
            havePrev = true;
        }
        const double mb = UsedMb(adapter);
        perCycle.push_back(mb);
        std::printf("  %-6d %-24s %10.1f\n", c, "(all)", mb);
    }

    // The trend over the BACK HALF, so first-pass warm-up is excluded.
    std::printf("\n");
    if (perCycle.size() >= 4) {
        const size_t half = perCycle.size() / 2;
        const double a = perCycle[half];
        const double b = perCycle.back();
        const double perCycleMb = (b - a) / double(perCycle.size() - half - 1);
        std::printf("  growth over the back half: %+.1f MB total, %+.2f MB per cycle\n",
                    b - a, perCycleMb);
        // A megabyte per cycle is noise -- descriptor heaps and command
        // allocators move around. A whole pipeline's worth is tens of MB.
        if (perCycleMb > 4.0) {
            std::printf("\n  LEAKING: switching scripts does not release the previous\n"
                        "  pipeline's GPU resources.\n");
            return 1;
        }
        std::printf("\n  steady state\n");
    }
    return 0;
}
