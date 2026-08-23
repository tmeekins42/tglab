// gpu_leak — re-runs a pipeline many times and reports VRAM growth.
//
// Dragging a slider re-runs the pipeline once per tick. Anything allocated per
// run and not freed shows up here as a straight line: Tim ran the machine out
// of VRAM by moving the Sauvola window slider.
//
// Reports the trend rather than a single before/after pair, because the first
// few runs legitimately allocate (kernels compile, outputs and scratch are
// created) and a naive two-point comparison reads that as a leak.
//
//   gpu_leak [algorithm] [runs]
//
// Exit code 1 if VRAM is still climbing over the back half of the run, which is
// what distinguishes a leak from steady-state warm-up.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <d3d12.h>
#include <dxgi1_4.h>

#include "../src/core/algorithm.h"
#include "../src/core/pipeline.h"
#include "../src/gpu/compute.h"
#include "../src/gpu/gpu_image.h"
#include "../src/script/interp.h"
#include "../src/script/parser.h"

using namespace tglab;

static IDXGIAdapter3* g_adapter = nullptr;

static uint64_t UsedVram() {
    if (!g_adapter) return 0;
    DXGI_QUERY_VIDEO_MEMORY_INFO info = {};
    if (FAILED(g_adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info)))
        return 0;
    return info.CurrentUsage;
}

// Finds a numeric parameter to vary, and a couple of values inside its range.
//
// Hardcoding "window" only worked for the thresholds; a blur has sigma and a
// brightness has gain. Asking the algorithm keeps this usable for every
// registered kernel, which is the point -- a leak in the shared iterative path
// would show up on whichever algorithm happens to exercise it.
static bool PickParam(const std::string& algo, std::string* name,
                      std::vector<double>* values) {
    auto a = Registry::Get().Create(algo);
    if (!a) return false;
    for (ParamBase* p : a->Params()) {
        if (p->Type() == ParamType::Bool) continue;
        UiControl c;
        if (!p->DescribeControl(&c)) continue;
        *name = p->Name();
        // Three values well inside the range, so clamping does not collapse
        // them to the same number and leave every run cache-clean.
        const double lo = c.lo, hi = c.hi;
        const double span = hi - lo;
        values->clear();
        values->push_back(lo + span * 0.25);
        values->push_back(lo + span * 0.35);
        values->push_back(lo + span * 0.45);
        if (p->Type() == ParamType::Int)
            for (double& v : *values) v = double(int(v));
        return true;
    }
    return false;
}

static Image MakeSource(int dim) {
    Image img;
    img.Alloc({dim, dim, Format::RGBA8});
    ImageView v = img.MapCpuWrite();
    for (int y = 0; y < dim; ++y) {
        for (int x = 0; x < dim; ++x) {
            uint8_t* p = v.At<uint8_t>(x, y);
            p[0] = uint8_t(x % 256);
            p[1] = uint8_t(y % 256);
            p[2] = uint8_t((x + y) % 256);
            p[3] = 255;
        }
    }
    return img;
}

int main(int argc, char** argv) {
    const std::string algo = (argc > 1) ? argv[1] : "threshold_sauvola";
    const int runs = (argc > 2) ? std::atoi(argv[2]) : 40;
    const int dim = 1024;

    ID3D12Device* dev = nullptr;
    if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&dev)))) {
        std::printf("no D3D12 device\n");
        return 0;
    }

    // The adapter is what reports video memory; the device cannot.
    IDXGIFactory4* factory = nullptr;
    if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
        IDXGIAdapter1* a1 = nullptr;
        if (SUCCEEDED(factory->EnumAdapters1(0, &a1))) {
            a1->QueryInterface(IID_PPV_ARGS(&g_adapter));
            a1->Release();
        }
        factory->Release();
    }

    ComputeContext gpu;
    if (!gpu.Init(dev)) { std::printf("compute init failed\n"); return 1; }
    InstallGpuResidencyHooks();

    std::printf("re-running %s %d times at %dx%d\n\n", algo.c_str(), runs, dim, dim);

    std::vector<uint64_t> used;
    used.reserve(size_t(runs));

    // A fresh Pipeline each iteration, matching what the app does: the script is
    // re-interpreted on every parameter change and the stage list rebuilt. The
    // PREVIOUS pipeline is passed in, which is how cached outputs and compiled
    // kernels survive -- and is exactly the path where a per-run allocation
    // would escape.
    std::string paramName;
    std::vector<double> paramValues;
    if (!PickParam(algo, &paramName, &paramValues)) {
        std::printf("%s has no numeric parameter to vary\n", algo.c_str());
        return 1;
    }
    std::printf("varying %s\n\n", paramName.c_str());

    Pipeline prev;
    bool havePrev = false;

    for (int i = 0; i < runs; ++i) {
        // Vary a parameter so the stage is dirty every run, as a slider drag
        // makes it. Without this the cache would satisfy every run after the
        // first and nothing would allocate at all.
        const std::string script =
            "src = image(\"test\")\n"
            "o = " + algo + "(src, " + paramName + " = " +
            std::to_string(paramValues[size_t(i) % paramValues.size()]) + ")\n"
            "display(o, \"out\")\n";

        std::vector<Data> sources;
        sources.push_back(Data{MakeSource(dim)});
        std::vector<SourceImage> names{{"test", 0}};

        Program prog;
        std::string err;
        if (!Parse(script.c_str(), &prog, &err)) {
            std::printf("parse failed: %s\n", err.c_str());
            return 1;
        }

        UiState ui;
        Pipeline pipe;
        auto ir = Interpret(prog, names, &ui, &pipe);
        if (!ir.ok) { std::printf("interpret failed: %s\n", ir.error.c_str()); return 1; }

        if (!pipe.Execute(&sources, havePrev ? &prev : nullptr, &err, &gpu,
                          ExecMode::ForceGPU)) {
            std::printf("run %d failed: %s\n", i, err.c_str());
            return 1;
        }

        // Force completion, so the measurement below is not taken while the
        // queue still holds work (and its resources).
        const Data* d = pipe.Resolve(pipe.Viewers()[0].source, &sources);
        if (d && std::holds_alternative<Image>(*d))
            const_cast<Image&>(std::get<Image>(*d)).MapCpuRead();

        prev = std::move(pipe);
        havePrev = true;

        const uint64_t u = UsedVram();
        used.push_back(u);
        if (i % 5 == 0 || i == runs - 1)
            std::printf("  run %2d   %6.1f MB\n", i, double(u) / (1024.0 * 1024.0));
    }

    // Compare the two halves rather than first-to-last. Warm-up allocation is
    // real and belongs to the first few runs; a leak keeps climbing after it.
    const size_t half = used.size() / 2;
    double firstHalf = 0, secondHalf = 0;
    for (size_t i = 0; i < half; ++i)             firstHalf  += double(used[i]);
    for (size_t i = half; i < used.size(); ++i)   secondHalf += double(used[i]);
    firstHalf  /= double(half);
    secondHalf /= double(used.size() - half);

    const double growthMb = (secondHalf - firstHalf) / (1024.0 * 1024.0);
    const double perRunMb = (double(used.back()) - double(used[half])) /
                            double(used.size() - half) / (1024.0 * 1024.0);

    std::printf("\nfirst half avg  %8.1f MB\n", firstHalf  / (1024.0 * 1024.0));
    std::printf("second half avg %8.1f MB\n", secondHalf / (1024.0 * 1024.0));
    std::printf("growth          %8.1f MB   (%.2f MB per run over the back half)\n",
                growthMb, perRunMb);

    gpu.Shutdown();
    if (g_adapter) g_adapter->Release();
    dev->Release();

    // 16 MB of drift over the back half is noise -- other processes share this
    // adapter and the driver caches. A real leak here is hundreds of MB.
    if (growthMb > 16.0) {
        std::printf("\nLEAKING: VRAM still climbing after warm-up\n");
        return 1;
    }
    std::printf("\nsteady state\n");
    return 0;
}
