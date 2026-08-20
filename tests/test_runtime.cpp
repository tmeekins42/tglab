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
#include "../src/core/compare.h"
#include "../src/core/pipeline.h"
#include "../src/core/worker.h"
#include "../src/gpu/compute.h"
#include "../src/gpu/gpu_image.h"
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
    // Deliberately not asserting the GPU is faster. On a small image the fixed
    // dispatch cost (upload, descriptor setup, fence wait) dominates, and in a
    // Release build the CPU blur is quick enough to win outright -- this test
    // uses 384x384. What must hold is that both paths ran and agree; which is
    // faster at a given size is a measurement, not a correctness property.
    Check(cpuMs > 0.0 && gpuMs > 0.0, "both paths produced a timing");

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

static void TestResidency(ID3D12Device* dev) {
    Section("GPU residency");

    ComputeContext gpu;
    if (!gpu.Init(dev)) { Check(false, "compute context"); return; }

    // A GPU stage must leave its output GPU-resident, not read it back. That
    // is what lets a chain upload once and transfer nothing in the middle.
    {
        UiState ui; Pipeline pipe; std::vector<Data> src; std::string e;
        BuildPipeline("src = image(\"test\")\nb = gaussian_blur(src, sigma = 2)\ndisplay(b)\n",
                      256, &ui, &pipe, &src, &e);
        Check(pipe.Execute(&src, nullptr, &e, &gpu, ExecMode::ForceGPU),
              "GPU stage runs" + (e.empty() ? "" : ": " + e));

        const Data* d = pipe.Resolve(pipe.Viewers()[0].source, &src);
        Image& out = const_cast<Image&>(std::get<Image>(*d));
        Check(out.HasGpu(), "the output is GPU-resident after the dispatch");
        Check(!out.HasCpu(), "no readback happened (output is GPU-only)");

        // Asking for CPU pixels is what triggers the single transfer.
        ImageView v = out.MapCpuRead();
        Check(v.Valid() && out.HasCpu(), "MapCpuRead() pulls the pixels back on demand");
        Check(out.HasGpu(), "the GPU copy survives readback (now resident on both)");

        // And the pixels must be real, not an uninitialised buffer.
        int nonZero = 0;
        for (int i = 0; i < v.desc.width * v.desc.height * 4; ++i)
            if (v.data[i] != 0) ++nonZero;
        Check(nonZero > v.desc.width * v.desc.height,
              "read-back pixels are real data, not a blank buffer");
    }

    // Chained GPU stages: the intermediate must never touch the CPU.
    {
        UiState ui; Pipeline pipe; std::vector<Data> src; std::string e;
        BuildPipeline("src = image(\"test\")\n"
                      "a = gaussian_blur(src, sigma = 2)\n"
                      "b = gaussian_blur(a, sigma = 2)\n"
                      "display(b)\n",
                      256, &ui, &pipe, &src, &e);
        Check(pipe.Execute(&src, nullptr, &e, &gpu, ExecMode::ForceGPU),
              "chained GPU stages run" + (e.empty() ? "" : ": " + e));
        Check(pipe.GpuStageCount() == 2, "both stages ran on the GPU");

        // The intermediate (stage 0's output) is the one that would have cost
        // a readback + upload round trip before residency.
        Image& mid = std::get<Image>(pipe.Stages()[0].outputs[0]);
        Check(mid.HasGpu() && !mid.HasCpu(),
              "the intermediate stayed GPU-only — no round trip through the CPU");
    }

    // Writing on the CPU must invalidate the GPU copy, or a later GPU stage
    // would silently consume stale pixels.
    {
        Image img;
        img.Alloc({64, 64, Format::RGBA8});
        Check(img.HasCpu() && !img.HasGpu(), "a fresh image is CPU-only");
        Check(img.AcquireGpuRead(gpu) != nullptr, "AcquireGpuRead uploads");
        Check(img.HasGpu() && img.HasCpu(), "after upload it is resident on both");

        img.MapCpuWrite();
        Check(img.HasCpu() && !img.HasGpu(),
              "MapCpuWrite() invalidates the GPU copy");
    }

    gpu.Shutdown();
}

// Every GPU kernel must agree with its CPU implementation. A kernel that
// merely runs and produces a plausible image is not verified: the whole point
// of having both is that they compute the same thing, and compare mode is
// only trustworthy if that holds.
static void TestGpuAgreement(ID3D12Device* dev) {
    Section("CPU/GPU agreement");

    ComputeContext gpu;
    if (!gpu.Init(dev)) { Check(false, "compute context"); return; }

    struct Case {
        const char* name;
        const char* script;
        double      tolerance;      // max, in 0..255 units
        double      meanTolerance;  // average, which is what catches real bugs
    };

    // Tolerances reflect real numerical differences, not sloppiness: the GPU
    // works in normalised floats and the CPU in 0..255, so rounding differs by
    // roughly an LSB. Iterative schemes accumulate that over their passes.
    const Case cases[] = {
        {"box_blur",
         "src = image(\"test\")\n"
         "o = box_blur(src, radius = 3)\n"
         "display(o)\n", 2.0, 1.0},

        {"gaussian_blur",
         "src = image(\"test\")\n"
         "o = gaussian_blur(src, sigma = 2.0)\n"
         "display(o)\n", 2.0, 1.0},

        {"bilateral",
         "src = image(\"test\")\n"
         "o = bilateral(src, sigma_space = 3.0, sigma_range = 0.15)\n"
         "display(o)\n", 3.0, 1.0},

        {"kuwahara",
         "src = image(\"test\")\n"
         "o = kuwahara(src, radius = 3)\n"
         "display(o)\n", 3.0, 1.0},

        // Looser max than the rest, for a reason specific to this filter: it
        // picks one member of each symmetric pair by an exact comparison
        // (da <= db), evaluated at 0..255 on the CPU and normalised 0..1 on
        // the GPU. Where the two are exactly tied -- common on the flat regions
        // of the checkerboard fixture -- the sides can keep different pixels,
        // which is a genuine difference in value rather than rounding. The mean
        // check below is what confirms it stays isolated.
        {"symmetric_nearest",
         "src = image(\"test\")\n"
         "o = symmetric_nearest(src, radius = 3)\n"
         "display(o)\n", 5.0, 1.0},

        // Iterative: exercises the ping-pong path, where a parity mistake
        // would leave the result in the scratch image and show the input
        // unchanged -- which a "did it run" check would not catch.
        {"anisotropic_diffusion",
         "src = image(\"test\")\n"
         "o = anisotropic_diffusion(src, iterations = 8, lambda = 0.2, k = 0.1)\n"
         "display(o)\n", 6.0, 1.0},
    };

    for (const Case& c : cases) {
        UiState ui; Pipeline pipe; std::vector<Data> src; std::string e;
        if (!BuildPipeline(c.script, 1024, &ui, &pipe, &src, &e)) {
            Check(false, std::string(c.name) + " builds: " + e);
            continue;
        }

        const CompareResult r = CompareCpuGpu(pipe, &src, &gpu, -1);
        if (!r.ok) {
            Check(false, std::string(c.name) + " compares: " + r.error);
            continue;
        }

        Check(r.algorithm == c.name,
              std::string("compare selected ") + c.name + " (got " + r.algorithm + ")");
        Check(r.stats.maxAbsDiff <= c.tolerance,
              std::string(c.name) + " GPU matches CPU (max diff " +
                  std::to_string(r.stats.maxAbsDiff) + " <= " +
                  std::to_string(c.tolerance) + ")");

        // The mean is the real check. A wrong kernel is wrong more or less
        // everywhere, so it moves the average well beyond a rounding step;
        // per-pixel LSB differences cannot, however many pixels have them.
        //
        // The bound is one LSB: the two sides quantise differently (the CPU
        // rounds from 0..255 values, the GPU from normalised floats), so every
        // pixel may legitimately land one step apart, but no more. Anything
        // above that is a difference in the maths, not in the rounding.
        Check(r.stats.meanAbsDiff <= c.meanTolerance,
              std::string(c.name) + " differs only by quantisation (mean " +
                  std::to_string(r.stats.meanAbsDiff) + " <= " +
                  std::to_string(c.meanTolerance) + ")");

        // Reported, not asserted: at 128x128 the dispatch overhead dominates,
        // so a speedup here would be measuring the wrong thing. The point of
        // the GPU path is interactive sliders on megapixel scans.
        std::printf("       %-24s cpu %7.1f ms   gpu %7.1f ms   %.1fx\n",
                    c.name, r.cpuMs, r.gpuMs, r.Speedup());
    }

    // An iterative stage must actually change the image. Getting the ping-pong
    // parity wrong yields the *input* rather than the result, which still
    // "matches" nothing and would slip past a tolerance check alone.
    {
        UiState ui; Pipeline pipe; std::vector<Data> src; std::string e;
        const char* script =
            "src = image(\"test\")\n"
            "o = anisotropic_diffusion(src, iterations = 12, lambda = 0.25, k = 0.5)\n"
            "display(o)\n";
        if (BuildPipeline(script, 128, &ui, &pipe, &src, &e) &&
            pipe.Execute(&src, nullptr, &e, &gpu, ExecMode::ForceGPU)) {
            Check(pipe.GpuStageCount() == 1, "the iterative stage ran on the GPU");

            const Data* d = pipe.Resolve({0, 0}, &src);
            Image& out = const_cast<Image&>(std::get<Image>(*d));
            Image& in  = std::get<Image>(src[0]);
            const CompareStats s = CompareImages(in, out, 0.0);
            Check(s.maxAbsDiff > 1.0,
                  "the iterative GPU result differs from its input (ping-pong parity)");
        } else {
            Check(false, "iterative GPU stage ran: " + e);
        }
    }
}

static void TestCompare(ID3D12Device* dev) {
    Section("compare mode");

    ComputeContext gpu;
    if (!gpu.Init(dev)) { Check(false, "compute context"); return; }

    // Identical images must report zero difference — the baseline that makes
    // any non-zero result meaningful.
    {
        Image a, b;
        a.Alloc({32, 32, Format::RGBA8});
        b.Alloc({32, 32, Format::RGBA8});
        ImageView va = a.MapCpuWrite(), vb = b.MapCpuWrite();
        for (int i = 0; i < 32 * 32 * 4; ++i) { va.data[i] = uint8_t(i); vb.data[i] = uint8_t(i); }
        const CompareStats s = CompareImages(a, b, 1.0);
        Check(s.maxAbsDiff == 0 && s.meanAbsDiff == 0, "identical images differ by zero");
        Check(s.diffPixels == 0, "no pixels beyond tolerance");
    }

    // A known offset must be measured exactly.
    {
        Image a, b;
        a.Alloc({16, 16, Format::RGBA8});
        b.Alloc({16, 16, Format::RGBA8});
        ImageView va = a.MapCpuWrite(), vb = b.MapCpuWrite();
        for (int y = 0; y < 16; ++y)
            for (int x = 0; x < 16; ++x) {
                uint8_t* pa = va.At<uint8_t>(x, y);
                uint8_t* pb = vb.At<uint8_t>(x, y);
                pa[0] = 100; pa[1] = 100; pa[2] = 100; pa[3] = 255;
                pb[0] = 110; pb[1] = 100; pb[2] = 100; pb[3] = 255;   // +10 on red
            }
        const CompareStats s = CompareImages(a, b, 1.0);
        Check(s.maxAbsDiff == 10.0, "max difference is measured exactly");
        Check(s.diffPixels == 256, "every pixel counted as differing");
        // 10 on one of three compared channels.
        Check(std::fabs(s.meanAbsDiff - 10.0 / 3.0) < 0.01, "mean is over compared channels");
    }

    // Alpha is excluded: most algorithms pass it through, so counting it would
    // dilute a real difference in the colour channels.
    {
        Image a, b;
        a.Alloc({8, 8, Format::RGBA8});
        b.Alloc({8, 8, Format::RGBA8});
        ImageView va = a.MapCpuWrite(), vb = b.MapCpuWrite();
        for (int i = 0; i < 8 * 8; ++i) {
            va.data[i * 4 + 3] = 255;
            vb.data[i * 4 + 3] = 0;      // alpha differs wildly
        }
        const CompareStats s = CompareImages(a, b, 1.0);
        Check(s.maxAbsDiff == 0, "alpha differences are ignored");
    }

    // The real thing: run gaussian_blur both ways through the pipeline.
    {
        UiState ui; Pipeline pipe; std::vector<Data> src; std::string e;
        BuildPipeline("src = image(\"test\")\nb = gaussian_blur(src, sigma = 3)\ndisplay(b)\n",
                      256, &ui, &pipe, &src, &e);
        const CompareResult r = CompareCpuGpu(pipe, &src, &gpu, -1);
        Check(r.ok, "CompareCpuGpu runs" + (r.ok ? "" : ": " + r.error));
        Check(r.algorithm == "gaussian_blur", "reports which algorithm was compared");
        std::printf("       CPU %.1f ms   GPU %.1f ms   speedup %.1fx   maxdiff %.3f\n",
                    r.cpuMs, r.gpuMs, r.Speedup(), r.stats.maxAbsDiff);
        Check(r.stats.maxAbsDiff <= 4.0, "CPU and GPU agree");
        Check(r.cpuMs > 0 && r.gpuMs > 0, "both paths were timed");
        Check(r.cpuImage.Valid() && r.gpuImage.Valid(), "both result images captured");
        Check(r.diffImage.Valid(), "diff image produced");
    }

    // A pipeline with nothing comparable must say so, rather than silently
    // comparing something against itself.
    {
        UiState ui; Pipeline pipe; std::vector<Data> src; std::string e;
        BuildPipeline("src = image(\"test\")\ng = grayscale(src)\ndisplay(g)\n",
                      128, &ui, &pipe, &src, &e);
        const CompareResult r = CompareCpuGpu(pipe, &src, &gpu, -1);
        Check(!r.ok && r.error.find("no stage") != std::string::npos,
              "an all-CPU pipeline reports nothing to compare: " + r.error);
    }
    // Naming a specific CPU-only stage reports that stage by name.
    {
        UiState ui; Pipeline pipe; std::vector<Data> src; std::string e;
        BuildPipeline("src = image(\"test\")\ng = grayscale(src)\ndisplay(g)\n",
                      128, &ui, &pipe, &src, &e);
        const CompareResult r = CompareCpuGpu(pipe, &src, &gpu, 0);
        Check(!r.ok && r.error.find("grayscale") != std::string::npos,
              "an explicitly chosen CPU-only stage names itself: " + r.error);
    }
    // Auto-selection skips CPU-only stages to find a comparable one — a
    // pipeline usually ends in something without a kernel, so defaulting to
    // the last stage would almost always report "cannot compare".
    {
        UiState ui; Pipeline pipe; std::vector<Data> src; std::string e;
        BuildPipeline("src = image(\"test\")\n"
                      "b = gaussian_blur(src, sigma = 2)\n"
                      "gx, gy, mag = sobel(b)\n"
                      "display(mag)\n",
                      128, &ui, &pipe, &src, &e);
        const CompareResult r = CompareCpuGpu(pipe, &src, &gpu, -1);
        Check(r.ok && r.algorithm == "gaussian_blur",
              "auto-select picks the comparable stage, not the last one");
    }

    gpu.Shutdown();
}

int main() {
    TestWorker();
    TestShaderCompiler();

    ID3D12Device* dev = nullptr;
    if (SUCCEEDED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&dev)))) {
        TestGpuPipeline(dev);
        TestResidency(dev);
        TestCompare(dev);
        TestGpuAgreement(dev);
        dev->Release();
    } else {
        std::printf("\n(no D3D12 device — GPU tests skipped)\n");
    }

    std::printf("\n%s\n", g_fail == 0 ? "all runtime checks passed" : "FAILURES PRESENT");
    return g_fail == 0 ? 0 : 1;
}
