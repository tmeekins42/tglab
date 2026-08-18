// Runtime tests: the worker thread and the GPU compute path.
//
// Separate from test_script.cpp because these need a D3D12 device and take
// seconds rather than milliseconds. Run both: test_script covers the language
// and pipeline semantics, this covers execution.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <d3d12.h>

#include "../src/core/algorithm.h"
#include "../src/core/pipeline.h"
#include "../src/core/worker.h"
#include "../src/gpu/compute.h"
#include "../src/script/interp.h"
#include "../src/script/parser.h"

using namespace tglab;
using clockt = std::chrono::steady_clock;

static int g_fail = 0;
static void Check(bool ok, const std::string& what) {
    std::printf("%s  %s\n", ok ? "[ ok ]" : "[FAIL]", what.c_str());
    if (!ok) ++g_fail;
}
static void Section(const char* name) { std::printf("\n--- %s ---\n", name); }

static Image MakeSource(int dim) {
    Image img;
    img.Alloc({dim, dim, Format::RGBA8});
    ImageView v = img.MapCpuWrite();
    for (int y = 0; y < dim; ++y)
        for (int x = 0; x < dim; ++x) {
            uint8_t* p = v.At<uint8_t>(x, y);
            const bool checker = ((x / 24) + (y / 24)) % 2 == 0;
            p[0] = checker ? 240 : 15;
            p[1] = uint8_t(x);
            p[2] = uint8_t(y);
            p[3] = 255;
        }
    return img;
}

static bool BuildPipeline(const char* script, int dim, UiState* ui,
                          Pipeline* pipe, std::vector<Data>* sources, std::string* err) {
    sources->clear();
    sources->push_back(Data{MakeSource(dim)});
    std::vector<SourceImage> names{{"test", 0}};
    Program prog;
    if (!Parse(script, &prog, err)) return false;
    auto r = Interpret(prog, names, ui, pipe);
    if (!r.ok) { *err = r.error; return false; }
    return true;
}

// ---------------------------------------------------------------------------

static void TestWorker() {
    Section("worker thread");

    const char* kSlow =
        "src = image(\"test\")\n"
        "b = gaussian_blur(src, sigma = 8)\n"
        "display(b, \"out\")\n";
    const int dim = 512;

    PipelineWorker worker;
    worker.Start(nullptr);                 // CPU-only, to time the slow path
    worker.SetExecMode(ExecMode::ForceCPU);

    // Submitting must not block on the work.
    {
        UiState ui; Pipeline pipe; std::vector<Data> src; std::string err;
        if (!BuildPipeline(kSlow, dim, &ui, &pipe, &src, &err)) {
            Check(false, "build pipeline: " + err);
            return;
        }
        const auto t0 = clockt::now();
        worker.Submit(std::move(pipe), std::move(src));
        const double submitMs =
            std::chrono::duration<double, std::milli>(clockt::now() - t0).count();

        PipelineOutcome out;
        while (!worker.TryFetch(&out)) std::this_thread::sleep_for(std::chrono::milliseconds(2));
        const double totalMs =
            std::chrono::duration<double, std::milli>(clockt::now() - t0).count();

        std::printf("       submit=%.2fms  work=%.1fms\n", submitMs, totalMs);
        Check(submitMs < 5.0, "Submit() does not block on the work");
        Check(totalMs > submitMs * 2, "the work really was the slow part");
        Check(out.ok && out.viewers.size() == 1 && out.viewers[0].image.Valid(),
              "result carries a valid viewer image");
    }

    // A burst coalesces to far fewer runs, and the newest wins.
    {
        uint64_t lastSeq = 0;
        for (int i = 0; i < 20; ++i) {      // simulates a slider drag
            UiState ui; Pipeline pipe; std::vector<Data> src; std::string err;
            if (!BuildPipeline(kSlow, dim, &ui, &pipe, &src, &err)) return;
            lastSeq = worker.Submit(std::move(pipe), std::move(src));
        }
        int delivered = 0;
        uint64_t newest = 0;
        const auto deadline = clockt::now() + std::chrono::seconds(30);
        while (clockt::now() < deadline) {
            PipelineOutcome out;
            if (worker.TryFetch(&out)) {
                ++delivered;
                newest = out.seq;
                if (newest == lastSeq) break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        std::printf("       submitted 20, delivered %d\n", delivered);
        Check(delivered < 20, "a burst does not run every request (coalesced)");
        Check(newest == lastSeq, "the newest request is the one that wins");
    }

    worker.Stop();
}

static void TestShaderCompiler() {
    Section("shader compilation");

    ShaderCompiler sc;
    Check(sc.Init(), "DXC initialises (dxcompiler.dll beside the exe)");
    if (!sc.Ready()) return;

    const char* kGood = R"(
RWTexture2D<float4> Dst : register(u0);
[numthreads(8,8,1)]
void main(uint3 tid : SV_DispatchThreadID) { Dst[tid.xy] = float4(1,0,0,1); }
)";
    ShaderBlob blob;
    std::string err;
    Check(sc.CompileCompute(kGood, "main", "ok.hlsl", &blob, &err),
          "valid shader compiles" + (err.empty() ? "" : ": " + err));

    // Unsigned DXIL is rejected by D3D12 with a misleading error, so check the
    // container really was signed by dxil.dll.
    if (blob.dxil.size() > 20) {
        bool signed_ = false;
        for (int i = 4; i < 20; ++i) if (blob.dxil[size_t(i)] != 0) signed_ = true;
        Check(signed_, "DXIL is signed (dxil.dll present)");
    }

    const char* kBad = R"(
RWTexture2D<float4> Dst : register(u0);
[numthreads(8,8,1)]
void main(uint3 tid : SV_DispatchThreadID) { Dst[tid.xy] = nope(tid); }
)";
    ShaderBlob bad;
    std::string badErr;
    Check(!sc.CompileCompute(kBad, "main", "bad.hlsl", &bad, &badErr),
          "invalid shader fails rather than silently succeeding");
    Check(badErr.find("nope") != std::string::npos,
          "error names the offending symbol");
    sc.Shutdown();
}

static void TestGpuPipeline(ID3D12Device* dev) {
    Section("GPU pipeline");

    ComputeContext gpu;
    Check(gpu.Init(dev), "compute context initialises (own compute queue)");
    if (!gpu.Ready()) return;

    const char* kScript =
        "src = image(\"test\")\n"
        "b = gaussian_blur(src, sigma = 3)\n"
        "display(b, \"out\")\n";
    const int dim = 384;

    auto run = [&](ExecMode mode, Image* out, double* ms, int* stages, std::string* err) {
        UiState ui; Pipeline pipe; std::vector<Data> src;
        if (!BuildPipeline(kScript, dim, &ui, &pipe, &src, err)) return false;
        const auto t0 = clockt::now();
        if (!pipe.Execute(&src, nullptr, err, &gpu, mode)) return false;
        *ms = std::chrono::duration<double, std::milli>(clockt::now() - t0).count();
        *stages = pipe.GpuStageCount();
        const Data* d = pipe.Resolve(pipe.Viewers()[0].source, &src);
        *out = const_cast<Image&>(std::get<Image>(*d)).Clone();
        return true;
    };

    Image cpuImg, gpuImg;
    double cpuMs = 0, gpuMs = 0;
    int cpuStages = -1, gpuStages = -1;
    std::string err;

    Check(run(ExecMode::ForceCPU, &cpuImg, &cpuMs, &cpuStages, &err),
          "ForceCPU runs" + (err.empty() ? "" : ": " + err));
    Check(cpuStages == 0, "ForceCPU used no GPU stages");

    Check(run(ExecMode::ForceGPU, &gpuImg, &gpuMs, &gpuStages, &err),
          "ForceGPU runs" + (err.empty() ? "" : ": " + err));
    Check(gpuStages == 1, "ForceGPU ran the stage on the GPU");

    std::printf("       CPU %.1f ms   GPU %.1f ms\n", cpuMs, gpuMs);
    Check(gpuMs < cpuMs, "the GPU path is faster end to end");

    // The whole point of having both: they must agree.
    {
        ImageView a = cpuImg.MapCpuRead();
        ImageView b = gpuImg.MapCpuRead();
        int maxDiff = 0;
        double sum = 0;
        int n = 0;
        for (int y = 0; y < dim; ++y)
            for (int x = 0; x < dim; ++x) {
                const uint8_t* p = a.At<uint8_t>(x, y);
                const uint8_t* q = b.At<uint8_t>(x, y);
                for (int c = 0; c < 3; ++c) {
                    const int diff = std::abs(int(p[c]) - int(q[c]));
                    if (diff > maxDiff) maxDiff = diff;
                    sum += diff;
                    ++n;
                }
            }
        std::printf("       CPU vs GPU: max=%d mean=%.3f (0-255)\n", maxDiff, sum / double(n));
        Check(maxDiff <= 4, "CPU and GPU agree within a few LSBs");
    }

    // Auto picks the GPU; a CPU-only algorithm still runs; mixed works.
    {
        Image a; double ms = 0; int st = -1; std::string e;
        Check(run(ExecMode::Auto, &a, &ms, &st, &e), "Auto runs");
        Check(st == 1, "Auto chose the GPU");
    }
    {
        UiState ui; Pipeline pipe; std::vector<Data> src; std::string e;
        BuildPipeline("src = image(\"test\")\ng = grayscale(src)\ndisplay(g)\n",
                      dim, &ui, &pipe, &src, &e);
        Check(pipe.Execute(&src, nullptr, &e, &gpu, ExecMode::Auto),
              "CPU-only algorithm runs under Auto");
        Check(pipe.GpuStageCount() == 0, "no GPU stage claimed for a CPU-only algorithm");
    }
    {
        UiState ui; Pipeline pipe; std::vector<Data> src; std::string e;
        BuildPipeline("src = image(\"test\")\nb = gaussian_blur(src, sigma = 2)\n"
                      "gx, gy, mag = sobel(b)\ndisplay(mag)\n",
                      dim, &ui, &pipe, &src, &e);
        Check(pipe.Execute(&src, nullptr, &e, &gpu, ExecMode::Auto),
              "mixed CPU/GPU pipeline runs" + (e.empty() ? "" : ": " + e));
        Check(pipe.GpuStageCount() == 1, "exactly the GPU-capable stage used the GPU");
    }

    gpu.Shutdown();
}

int main() {
    TestWorker();
    TestShaderCompiler();

    ID3D12Device* dev = nullptr;
    if (SUCCEEDED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&dev)))) {
        TestGpuPipeline(dev);
        dev->Release();
    } else {
        std::printf("\n(no D3D12 device — GPU tests skipped)\n");
    }

    std::printf("\n%s\n", g_fail == 0 ? "all runtime checks passed" : "FAILURES PRESENT");
    return g_fail == 0 ? 0 : 1;
}
