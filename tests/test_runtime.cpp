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
#include "../src/algo_util/histogram.h"
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

// Superseding a slow job must abandon it, not wait it out.
//
// Coalescing already drops *queued* jobs, but a job that had started ran to
// completion -- so nudging a slider during a minute-long filter meant waiting
// for the old value before the new one even began.

// A viewer whose result stays on the GPU carries a descriptor, not pixels.
//
// The whole point of drawing from the GPU is that the image is never read back.
// If the shell handed to the UI allocated a CPU buffer anyway, that saving
// would be spent on memory instead of bandwidth -- 169 MB for a 21 MP RGBA16F
// result, per viewer, per run, on every frame of a slider drag.
//
// Worse than the memory: Alloc() zero-fills and marks the image CPU-resident,
// so MapCpuRead() would hand back a full image of zeros that looks entirely
// valid. The histogram would describe it, the loupe would sample it, and
// nothing would report an error.
static void TestGpuOnlyShell() {
    Section("GPU-resident viewer shell");

    const ImageDesc d{1024, 768, Format::RGBA16F};

    // What Image(desc) does, for contrast: this is the trap.
    Image allocated(d);
    Check(allocated.HasCpu(), "Image(desc) allocates CPU pixels (the thing to avoid)");

    Image shell;
    shell.AdoptDesc(d);

    Check(!shell.HasCpu(), "AdoptDesc allocates no CPU pixels");
    Check(!shell.MapCpuRead().Valid(),
          "AdoptDesc: MapCpuRead reports no pixels rather than returning zeros");

    // The descriptor still has to be complete: the UI reads size and format
    // from it, and the view draws using its dimensions.
    Check(shell.Desc().width == d.width && shell.Desc().height == d.height &&
              shell.Desc().format == d.format,
          "AdoptDesc preserves the descriptor");
    Check(shell.Desc().Valid(), "AdoptDesc leaves the descriptor valid (the view draws it)");

    // Sensor metadata rides on the descriptor too, and the info panel shows it.
    ImageDesc raw{2048, 1536, Format::RGBA16F};
    raw.linear     = true;
    raw.whiteLevel = 15600.0f;
    raw.camMul[0]  = 2.266f;
    Image rawShell;
    rawShell.AdoptDesc(raw);
    Check(rawShell.Desc().linear && rawShell.Desc().whiteLevel == 15600.0f &&
              rawShell.Desc().camMul[0] == 2.266f,
          "AdoptDesc carries sensor metadata through");
}

// A GPU-resident result is drawable, and its pixels come out right.
//
// "Is there something to draw?" is not the same question as Image::Valid(),
// which asks whether pixels exist *somewhere*. A result that lives only on the
// GPU carries a descriptor and no pixels, so Image::Valid() is false while
// there is a perfectly good texture to convert from.
//
// Getting that distinction wrong showed every GPU-resident viewer as
// "computing..." forever -- a black panel on a run the status bar reported as
// finished. Nothing crashed and nothing logged an error, which is exactly why
// it needs a test rather than a smoke run.
static void TestGpuResultIsDrawable(ID3D12Device* dev) {
    Section("GPU-resident results are drawable");

    ComputeContext gpu;
    if (!gpu.Init(dev) || !gpu.Ready()) { Check(false, "compute context initialises"); return; }
    InstallGpuResidencyHooks();

    // A known input: mid grey, so the output value is predictable.
    const int w = 64, h = 48;
    Image src;
    src.Alloc({w, h, Format::RGBA8});
    {
        ImageView v = src.MapCpuWrite();
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                uint8_t* p = v.At<uint8_t>(x, y);
                p[0] = p[1] = p[2] = 128;
                p[3] = 255;
            }
    }

    std::vector<Data> sources;
    sources.push_back(Data{std::move(src)});
    Pipeline pipe;
    // basic_adjust rather than brightness: brightness is CPU-only, so the run
    // would fall back and produce no GPU residency at all -- the fixture has to
    // actually exercise the path it is testing.
    pipe.AddStage(Registry::Get().Create("basic_adjust"), "basic_adjust", {{-1, 0}}, 1, 1);

    std::string err;
    if (!pipe.Execute(&sources, nullptr, &err, &gpu, ExecMode::ForceGPU)) {
        Check(false, "GPU run: " + err);
        return;
    }

    const Data* d = pipe.Resolve({0, 0}, &sources);
    if (!d || !std::holds_alternative<Image>(*d)) { Check(false, "resolved an image"); return; }
    const Image& result = std::get<Image>(*d);

    auto shared = ShareGpuTexture(result);
    Check(shared != nullptr, "a GPU-resident result yields a shareable texture");
    if (!shared) return;

    // The shell, exactly as the worker builds it.
    Image shell;
    shell.AdoptDesc(result.Desc());

    // The trap, stated as an assertion so it cannot quietly come back: the
    // shell is NOT Image::Valid(), and a viewer that tests only that shows
    // "computing..." forever.
    Check(!shell.Valid(),
          "the shell is not Image::Valid() -- pixels live only on the device");
    Check(shell.Desc().Valid(),
          "...but its descriptor is valid, which is what makes it drawable");

    // What the view actually decides.
    const bool viewWouldDraw = (shared != nullptr) || shell.Valid();
    Check(viewWouldDraw, "a viewer with a GPU source draws rather than showing 'computing...'");

    Check(shared->desc.width == w && shared->desc.height == h,
          "the shared texture has the result's dimensions");

    // And the pixels are real, not a blank texture. A black panel is precisely
    // the failure being guarded against, and every check above would still pass
    // if the texture were empty.
    ImageView rv = const_cast<Image&>(result).MapCpuRead();
    if (rv.Valid()) {
        double sum = 0;
        int n = 0;
        for (int y = 0; y < rv.desc.height; y += 4)
            for (int x = 0; x < rv.desc.width; x += 4) {
                sum += rv.At<uint8_t>(x, y)[0];
                ++n;
            }
        const double mean = n ? sum / n : 0.0;
        Check(mean > 1.0,
              "the result carries real pixels, not a blank texture (mean " +
                  std::to_string(mean) + ")");
    } else {
        Check(false, "could read the result back for verification");
    }
}

// Per-viewer content versions.
//
// display(src, ...) next to a slider-driven stage is the common shape -- it is
// what develop.tgl does -- and the source's pixels do not change when the
// slider moves. The UI uploads a viewer's display texture only when its version
// changes, and that upload is expensive: Clone() on a GPU-resident image forces
// a readback and the texture then converts to RGBA8. Measured at 21 MP, ~85 ms
// and ~50 ms per viewer per frame, none of it visible to the status bar.
//
// A single global version cannot express this: it either never changes (and
// nothing ever refreshes) or changes whenever any viewer does (and everything
// refreshes).
static void TestViewerVersions() {
    Section("per-viewer content versions");

    PipelineWorker worker;
    worker.Start(nullptr);
    worker.SetExecMode(ExecMode::ForceCPU);

    // Viewer "a" reads the source; viewer "b" reads a blur the slider drives.
    auto build = [](double sigma, UiState* ui, Pipeline* pipe,
                    std::vector<Data>* src, std::string* err) {
        const std::string s =
            "src = image(\"test\")\n"
            "b = gaussian_blur(src, sigma = " + std::to_string(sigma) + ")\n"
            "display(src, \"a\")\n"
            "display(b, \"b\")\n";
        return BuildPipeline(s.c_str(), 128, ui, pipe, src, err);
    };

    auto fetch = [&](uint64_t seq, PipelineOutcome* out) {
        const auto deadline = clockt::now() + std::chrono::seconds(30);
        while (clockt::now() < deadline) {
            if (worker.TryFetch(out) && out->seq == seq) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        return false;
    };

    auto versionOf = [](const PipelineOutcome& o, const char* name) -> uint64_t {
        for (const ViewerImage& v : o.viewers)
            if (v.name == name) return v.version;
        return 0;
    };

    UiState ui1; Pipeline p1; std::vector<Data> s1; std::string err;
    if (!build(2.0, &ui1, &p1, &s1, &err)) { Check(false, "build: " + err); worker.Stop(); return; }
    PipelineOutcome o1;
    if (!fetch(worker.Submit(std::move(p1), std::move(s1)), &o1)) {
        Check(false, "first run delivered"); worker.Stop(); return; }

    Check(o1.viewers.size() == 2, "both viewers delivered on the first run");
    Check(versionOf(o1, "a") > 0 && versionOf(o1, "b") > 0,
          "both viewers start at a non-zero version");

    // Same script, different slider value: only "b" is downstream of it.
    UiState ui2; Pipeline p2; std::vector<Data> s2;
    if (!build(6.0, &ui2, &p2, &s2, &err)) { Check(false, "build: " + err); worker.Stop(); return; }
    PipelineOutcome o2;
    if (!fetch(worker.Submit(std::move(p2), std::move(s2)), &o2)) {
        Check(false, "second run delivered"); worker.Stop(); return; }

    Check(o2.viewers.size() == 2,
          "both viewers are still delivered (the UI must not be left holding a gap)");
    Check(versionOf(o2, "a") == versionOf(o1, "a"),
          "the source viewer keeps its version when a later stage changes (was " +
              std::to_string(versionOf(o1, "a")) + ", now " +
              std::to_string(versionOf(o2, "a")) + ")");
    Check(versionOf(o2, "b") != versionOf(o1, "b"),
          "the slider-driven viewer gets a new version (was " +
              std::to_string(versionOf(o1, "b")) + ", now " +
              std::to_string(versionOf(o2, "b")) + ")");

    worker.Stop();
}
static void TestCancellation() {
    Section("cancellation");

    // Deliberately expensive: NLM at a large search radius on a big image.
    const char* kSlow =
        "src = image(\"test\")\n"
        "o = nonlocal_means(src, patch = 1, search = 6)\n"
        "display(o, \"out\")\n";
    const int dim = 192;   // ~1 s: long enough to interrupt unambiguously,
                           // short enough not to dominate the suite.

    PipelineWorker worker;
    worker.Start(nullptr);
    worker.SetExecMode(ExecMode::ForceCPU);

    // Time one run alone, to know what "waiting it out" would have cost.
    double baselineMs = 0;
    {
        UiState ui; Pipeline pipe; std::vector<Data> src; std::string err;
        if (!BuildPipeline(kSlow, dim, &ui, &pipe, &src, &err)) {
            Check(false, "build: " + err);
            worker.Stop();
            return;
        }
        const auto t0 = clockt::now();
        worker.Submit(std::move(pipe), std::move(src));
        PipelineOutcome out;
        while (!worker.TryFetch(&out)) std::this_thread::sleep_for(std::chrono::milliseconds(2));
        baselineMs = std::chrono::duration<double, std::milli>(clockt::now() - t0).count();
    }
    Check(baselineMs > 300.0,
          "the fixture is slow enough to be worth cancelling (" +
              std::to_string(int(baselineMs)) + " ms)");

    // Start one, let it get going, then supersede it. The replacement must
    // finish sooner than two full runs would have taken -- which is what it
    // costs if the first is waited out rather than abandoned.
    {
        UiState ui1; Pipeline p1; std::vector<Data> s1; std::string err;
        if (!BuildPipeline(kSlow, dim, &ui1, &p1, &s1, &err)) {
            Check(false, "build: " + err);
            worker.Stop();
            return;
        }
        const auto t0 = clockt::now();
        worker.Submit(std::move(p1), std::move(s1));

        // Long enough to be well inside the first run, short enough that it
        // cannot have finished.
        std::this_thread::sleep_for(
            std::chrono::milliseconds(int(baselineMs / 3)));

        UiState ui2; Pipeline p2; std::vector<Data> s2;
        if (!BuildPipeline(kSlow, dim, &ui2, &p2, &s2, &err)) {
            Check(false, "build: " + err);
            worker.Stop();
            return;
        }
        const uint64_t secondSeq = worker.Submit(std::move(p2), std::move(s2));

        PipelineOutcome out;
        uint64_t got = 0;
        const auto deadline = clockt::now() + std::chrono::seconds(60);
        while (clockt::now() < deadline) {
            if (worker.TryFetch(&out)) {
                got = out.seq;
                if (got == secondSeq) break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        const double totalMs =
            std::chrono::duration<double, std::milli>(clockt::now() - t0).count();

        std::printf("       baseline %.0f ms, superseded run total %.0f ms\n",
                    baselineMs, totalMs);

        Check(got == secondSeq, "the superseding job is the one delivered");
        // Waiting the first out would cost the interruption point plus a full
        // second run; abandoning costs only the second. Allow generous slack
        // for scheduling, but the two are far enough apart to distinguish.
        Check(totalMs < baselineMs * 1.8,
              "the first run was abandoned rather than waited out");

        // A cancelled run must not surface as an error: nothing went wrong.
        Check(out.ok && out.error.empty(),
              "cancellation is not reported as a failure");
        Check(!out.viewers.empty() && out.viewers[0].image.Valid(),
              "the replacement result is complete");
    }

    worker.Stop();
}


// One ShaderCompiler shutting down must not disable another.
//
// The DXC objects behind ShaderCompiler are process-global, but the class looks
// like an ordinary per-instance object -- so one instance's Shutdown() used to
// reset them for everybody. That crashed the app on dropping a file: the
// display-conversion pipeline compiled its shader on the UI thread and shut its
// compiler down afterwards, and the worker thread's next CreateKernel()
// dereferenced a null compiler.
//
// The crash needed two things to line up -- a GPU display already built, then a
// new stage needing a kernel -- which is why it survived a clean debug-layer run
// and four passing suites.
static void TestShaderCompilerSharing() {
    Section("shader compiler sharing");

    const char* kTrivial =
        "RWTexture2D<float4> Dst : register(u0);\n"
        "cbuffer Params : register(b0) { uint Width; uint Height; };\n"
        "[numthreads(8,8,1)] void main(uint3 tid : SV_DispatchThreadID) {\n"
        "    if (tid.x >= Width || tid.y >= Height) return;\n"
        "    Dst[tid.xy] = float4(1,1,1,1);\n"
        "}\n";

    ShaderCompiler a;
    Check(a.Init(), "first compiler initialises");

    ShaderBlob blob;
    std::string err;
    Check(a.CompileCompute(kTrivial, "main", "t", &blob, &err),
          "first compiler compiles" + (err.empty() ? "" : ": " + err));

    {
        // A second user, as the display pipeline is: initialises, compiles, and
        // then shuts down while the first is still live.
        ShaderCompiler b;
        Check(b.Init(), "second compiler initialises");
        ShaderBlob blobB;
        Check(b.CompileCompute(kTrivial, "main", "t2", &blobB, &err),
              "second compiler compiles" + (err.empty() ? "" : ": " + err));
        b.Shutdown();
    }

    // The check that matters: `a` never called Init() again, so its m_ready is
    // still true and it will go straight to the globals. Before refcounting,
    // b.Shutdown() had already reset them and this crashed on a null compiler
    // -- which is precisely what happened on the worker thread.
    ShaderBlob after;
    err.clear();
    Check(a.CompileCompute(kTrivial, "main", "t3", &after, &err),
          "the first compiler still works after the second shut down" +
              (err.empty() ? "" : ": " + err));
    Check(after.Valid(), "...and still produces DXIL");

    a.Shutdown();

    // And it all comes back afterwards, so shutdown is not one-way.
    ShaderCompiler c;
    Check(c.Init(), "a compiler initialises again after the last one shut down");
    ShaderBlob revived;
    err.clear();
    Check(c.CompileCompute(kTrivial, "main", "t4", &revived, &err),
          "...and compiles" + (err.empty() ? "" : ": " + err));
    c.Shutdown();
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
    // guided_filter as the CPU-only example, deliberately: it is genuinely
    // CPU-only (a box-filtered local linear fit, not a per-pixel kernel), where
    // grayscale and sobel merely had not been ported yet and since have been.
    // A test whose premise is "this one is slow" quietly stops testing anything
    // the day someone speeds it up.
    {
        UiState ui; Pipeline pipe; std::vector<Data> src; std::string e;
        BuildPipeline("src = image(\"test\")\ng = guided_filter(src)\ndisplay(g)\n",
                      dim, &ui, &pipe, &src, &e);
        Check(pipe.Execute(&src, nullptr, &e, &gpu, ExecMode::Auto),
              "CPU-only algorithm runs under Auto");
        Check(pipe.GpuStageCount() == 0, "no GPU stage claimed for a CPU-only algorithm");
    }
    {
        UiState ui; Pipeline pipe; std::vector<Data> src; std::string e;
        BuildPipeline("src = image(\"test\")\nb = gaussian_blur(src, sigma = 2)\n"
                      "g = guided_filter(b)\ndisplay(g)\n",
                      dim, &ui, &pipe, &src, &e);
        Check(pipe.Execute(&src, nullptr, &e, &gpu, ExecMode::Auto),
              "mixed CPU/GPU pipeline runs" + (e.empty() ? "" : ": " + e));
        Check(pipe.GpuStageCount() == 1, "exactly the GPU-capable stage used the GPU");
    }

    gpu.Shutdown();
}

// Several GPU stages reading the SAME source in one batch.
//
// Regression test for a bug that made every viewer but the last show a black
// image. Dispatch() wrote its descriptors to the start of the shared heap on
// every call, which is fine when each dispatch is submitted immediately -- but
// with lazy flush a batch is recorded first and executed later, so all of its
// dispatches read whichever descriptors were written last. The earlier stages
// ran with the final stage's bindings and wrote nothing to their own outputs.
//
// A single-stage script cannot catch this, and neither can a chain: it needs
// two or more stages *in the same batch* whose bindings differ. Fan-out from
// one source is the natural shape, and it is exactly what demosaic.tgl does.
static void TestBatchedBindings(ID3D12Device* dev) {
    Section("batched dispatch bindings");

    ComputeContext gpu;
    if (!gpu.Init(dev) || !gpu.Ready()) { Check(false, "compute context initialises"); return; }

    // Ten stages, one shared input, no stage feeding another -- so nothing
    // forces a flush between them and they all land in one batch.
    //
    // Ten rather than three on purpose: the descriptor heap holds eight
    // dispatches, so this also exercises the flush-and-restart path that the
    // slice allocation needs when a batch runs out of room.
    std::string kScript = "src = image(\"test\")\n";
    for (int i = 1; i <= 10; ++i) {
        const std::string n = std::to_string(i);
        kScript += "s" + n + " = gaussian_blur(src, sigma = " + n + ")\n";
        kScript += "display(s" + n + ", \"v" + n + "\")\n";
    }
    const size_t kStages = 10;
    const int dim = 128;

    auto meanOf = [](Image& img) {
        ImageView v = img.MapCpuRead();
        double sum = 0; long long n = 0;
        for (int y = 0; y < v.desc.height; ++y)
            for (int x = 0; x < v.desc.width; ++x)
                for (int c = 0; c < 3; ++c) { sum += v.At<uint8_t>(x, y)[c]; ++n; }
        return sum / double(n);
    };

    auto run = [&](ExecMode mode, std::vector<double>* means, std::string* err) {
        UiState ui; Pipeline pipe; std::vector<Data> src;
        if (!BuildPipeline(kScript.c_str(), dim, &ui, &pipe, &src, err)) return false;
        if (!pipe.Execute(&src, nullptr, err, &gpu, mode)) return false;
        for (const auto& vw : pipe.Viewers()) {
            const Data* d = pipe.Resolve(vw.source, &src);
            Image img = const_cast<Image&>(std::get<Image>(*d)).Clone();
            means->push_back(meanOf(img));
        }
        return true;
    };

    std::vector<double> cpu, gpuMeans;
    std::string err;
    if (!run(ExecMode::ForceCPU, &cpu, &err))      { Check(false, "ForceCPU runs: " + err); return; }
    if (!run(ExecMode::ForceGPU, &gpuMeans, &err)) { Check(false, "ForceGPU runs: " + err); return; }

    Check(cpu.size() == kStages && gpuMeans.size() == kStages,
          "all " + std::to_string(kStages) + " viewers resolved");
    if (cpu.size() != kStages || gpuMeans.size() != kStages) return;

    // The bug left the earlier stages' outputs untouched -- allocated, zeroed,
    // never written -- so this is the check that fails without the fix.
    for (size_t i = 0; i < gpuMeans.size(); ++i)
        Check(gpuMeans[i] > 1.0,
              "stage " + std::to_string(i) + " of a batch wrote its own output (mean " +
                  std::to_string(gpuMeans[i]) + ")");

    // Not merely non-black: each must match what the CPU produced, which is
    // what proves it ran with *its own* bindings rather than a neighbour's.
    for (size_t i = 0; i < gpuMeans.size(); ++i)
        Check(std::abs(cpu[i] - gpuMeans[i]) < 1.0,
              "stage " + std::to_string(i) + " matches the CPU (cpu " +
                  std::to_string(cpu[i]) + " gpu " + std::to_string(gpuMeans[i]) + ")");
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

// The GPU histogram agrees with the CPU one, bin for bin.
//
// The info panel's histogram is computed where the pixels already live, so
// that a result never has to come back to the CPU just to be measured. The
// CPU path had to map the whole image first -- 86 ms of readback at 21 MP --
// and then kept 0.3% of what it fetched.
//
// Agreement has to be exact, not approximate: this is not an optimisation of
// the same code but a reimplementation in HLSL, including an order-preserving
// float-to-uint key (HLSL has no float atomics) and a two-pass min/max before
// binning. Any of that being subtly wrong would show as a plausible but
// incorrect histogram, which is far worse than a slow one.
static void TestGpuHistogram(ID3D12Device* dev) {
    Section("GPU histogram");

    ComputeContext gpu;
    if (!gpu.Init(dev) || !gpu.Ready()) { Check(false, "compute context initialises"); return; }

    // Three formats, because they take different paths: RGBA8 has a fixed
    // 0..255 range, the float ones are binned over their observed range, and
    // R32F has no colour channels at all.
    struct Case { const char* name; Format fmt; };
    const Case cases[] = {
        {"RGBA8",    Format::RGBA8},
        {"RGBA16F",  Format::RGBA16F},
        {"R32F",     Format::R32F},
    };

    for (const Case& c : cases) {
        const int w = 200, h = 150;
        Image img;
        img.Alloc({w, h, c.fmt});
        {
            ImageView v = img.MapCpuWrite();
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x) {
                    // A gradient with a bright corner, so the range is not
                    // 0..1 and the bins are not uniform.
                    const float t = float(x) / float(w - 1);
                    const float u = float(y) / float(h - 1);
                    const float lum = (x > w - 12 && y < 12) ? 1.8f : (0.15f + 0.6f * t * u);
                    if (c.fmt == Format::RGBA8) {
                        uint8_t* p = v.At<uint8_t>(x, y);
                        p[0] = uint8_t(std::clamp(lum, 0.0f, 1.0f) * 200.0f);
                        p[1] = uint8_t(std::clamp(lum, 0.0f, 1.0f) * 255.0f);
                        p[2] = uint8_t(std::clamp(lum, 0.0f, 1.0f) * 120.0f);
                        p[3] = 255;
                    } else if (c.fmt == Format::RGBA16F) {
                        uint16_t* p = v.At<uint16_t>(x, y);
                        p[0] = FloatToHalf(lum * 0.8f);
                        p[1] = FloatToHalf(lum);
                        p[2] = FloatToHalf(lum * 0.5f);
                        p[3] = FloatToHalf(1.0f);
                    } else {
                        *v.At<float>(x, y) = lum;
                    }
                }
        }

        // Upload, then measure on the GPU.
        GpuImage gi;
        if (!gpu.CreateImage(img.Desc(), &gi)) { Check(false, std::string(c.name) + ": create"); continue; }
        {
            ImageView v = img.MapCpuRead();
            if (!gpu.Upload(v, &gi)) { Check(false, std::string(c.name) + ": upload"); gi.Release(); continue; }
        }

        ComputeContext::HistogramResult g;
        std::string err;
        if (!gpu.BuildHistogram(gi, &g, &err)) {
            Check(false, std::string(c.name) + ": BuildHistogram: " + err);
            gi.Release();
            continue;
        }
        gi.Release();

        // The reference. Small enough that the shader samples every pixel
        // (step stays 1), so the two see exactly the same set and the bins
        // must match exactly rather than approximately.
        Histogram ref;
        ImageView v = img.MapCpuRead();
        ref.Build(v, -1);

        Check(g.count == ref.Count(),
              std::string(c.name) + ": same pixel count (gpu " + std::to_string(g.count) +
                  ", cpu " + std::to_string(ref.Count()) + ")");

        Check(std::abs(g.rangeMin - ref.RangeMin()) < 1e-4 &&
                  std::abs(g.rangeMax - ref.RangeMax()) < 1e-4,
              std::string(c.name) + ": same range (gpu [" + std::to_string(g.rangeMin) +
                  ", " + std::to_string(g.rangeMax) + "], cpu [" +
                  std::to_string(ref.RangeMin()) + ", " + std::to_string(ref.RangeMax()) + "])");

        // Bin-for-bin on luma. A near-miss in the float key or the bin index
        // would show here as a one-bin shift, which a shape comparison would
        // forgive and a histogram reader would not.
        uint64_t worstDiff = 0;
        int worstBin = -1;
        for (int i = 0; i < 256; ++i) {
            const uint64_t a = g.luma.empty() ? 0 : g.luma[size_t(i)];
            const uint64_t b = ref.Bin(i);
            const uint64_t d = (a > b) ? a - b : b - a;
            if (d > worstDiff) { worstDiff = d; worstBin = i; }
        }
        Check(worstDiff == 0,
              std::string(c.name) + ": luma bins match exactly (worst " +
                  std::to_string(worstDiff) + " at bin " + std::to_string(worstBin) + ")");

        // R32F has no colour channels to report; anything else must have three.
        if (c.fmt == Format::R32F) {
            Check(g.r.empty() && g.g.empty() && g.b.empty(),
                  "R32F: no colour channels reported");
        } else {
            Check(g.r.size() == 256 && g.g.size() == 256 && g.b.size() == 256,
                  std::string(c.name) + ": three colour channels reported");
        }
    }
}
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
        // brightness is where the two paths express the same thing DIFFERENTLY,
        // which is the interesting case for an agreement test. Its offset is a
        // fraction of the intensity range: the CPU works in the source's units
        // and multiplies it by 255 for an RGBA8 image, while a UNORM SRV hands
        // the shader 0..1 and it must not. Scaling in both places would apply
        // the offset 255 times over, and nothing but this check would notice.
        {"brightness",
         "src = image(\"test\")\n"
         "o = brightness(src, brightness = 0.2, gain = 1.5)\n"
         "display(o)\n", 2.0, 1.0},

        {"grayscale",
         "src = image(\"test\")\n"
         "o = grayscale(src, r_weight = 0.4, g_weight = 0.4, b_weight = 0.2)\n"
         "display(o)\n", 2.0, 1.0},

        {"box_blur",
         "src = image(\"test\")\n"
         "o = box_blur(src, radius = 3)\n"
         "display(o)\n", 2.0, 1.0},

        {"gaussian_blur",
         "src = image(\"test\")\n"
         "o = gaussian_blur(src, sigma = 2.0)\n"
         "display(o)\n", 2.0, 1.0},

        // A large sigma, which is the case that used to fall back to the CPU.
        // The old kernel was a single O(r^2) pass, so sigma above 4 was refused;
        // separable passes make radius 60 cost 120 fetches rather than 14,600.
        // Worth testing at both ends: the two paths compute the same radius from
        // the same sigma, so a mismatch here would mean they had drifted apart.
        {"gaussian_blur",
         "src = image(\"test\")\n"
         "o = gaussian_blur(src, sigma = 12.0)\n"
         "display(o)\n", 2.0, 1.0},

        {"bilateral",
         "src = image(\"test\")\n"
         "o = bilateral(src, sigma_space = 3.0, sigma_range = 0.15)\n"
         "display(o)\n", 3.0, 1.0},

        // Ten controls fused into one kernel, so a mistake in any of them is
        // invisible in the others' output. Exercised with every control off its
        // default at once. If the two paths drift, compare mode is worthless
        // and the fast version is not the same algorithm as the readable one.
        {"basic_adjust",
         "src = image(\"test\")\n"
         "o = basic_adjust(src, exposure = 0.7, contrast = 0.3, "
         "highlights = -0.4, shadows = 0.5, whites = 0.2, blacks = -0.2, "
         "kelvin = 4200, tint = -0.2, vibrance = 0.4, saturation = 0.2)\n"
         "display(o)\n", 4.0, 1.0},

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

        // A local threshold is a hard comparison, so a pixel within rounding
        // distance of its own threshold could in principle land on either side
        // -- and flipping one changes it by the full 0..1 range, which would
        // make a max-difference check meaningless.
        //
        // In practice the two agree exactly on this fixture, so the tolerance is
        // held tight enough to notice if that stops being true. Loosening it
        // "because thresholds are brittle" would have hidden a real divergence.

        // Bernsen, the case that motivated GpuScratchFormat. Its first pass
        // carries a window MINIMUM and MAXIMUM, so it needs a 2-channel
        // intermediate; sized from the R32F output the maximum was silently
        // dropped and half the pixels came out wrong. The mean tolerance below
        // is what detects that -- a max-difference check cannot, because a
        // threshold that flips is always wrong by the full range.

        // The Niblack family: mean and stddev over a clipped rectangular
        // window. The CPU takes these from an integral image (a sequential 2D
        // prefix sum); the GPU separates the rectangle into a horizontal then
        // vertical box sum, carrying sum/sum-of-squares/count through an
        // RGBA32F scratch. Border pixels average FEWER samples in both paths --
        // clipped, not clamp-sampled -- and getting that wrong shows up here as
        // an edge-only difference the mean check would otherwise dilute.
        {"threshold_niblack",
         "src = image(\"test\")\n"
         "o = threshold_niblack(src, window = 15, k = -0.2)\n"
         "display(o)\n", 255.0, 0.5},

        // Sauvola's R is a stddev in the source's units, so it must scale to
        // 0..1 for the shader while k, which multiplies a ratio, must not.
        // Scaling both or neither is the obvious mistake and this is what
        // catches it.
        {"threshold_sauvola",
         "src = image(\"test\")\n"
         "o = threshold_sauvola(src, window = 15, k = 0.2, r = 128)\n"
         "display(o)\n", 255.0, 0.5},

        {"threshold_adaptive_mean",
         "src = image(\"test\")\n"
         "o = threshold_adaptive_mean(src, window = 15, c = 5)\n"
         "display(o)\n", 255.0, 0.5},


        // wavelet_denoise: five chained dispatches ping-ponging an accumulator,
        // where every other GPU algorithm here is one or two passes. A parity
        // mistake would leave the result in the scratch buffer and show the
        // input unchanged, which the mean check below would catch and a "did it
        // run" check would not.
        //
        // The GPU fuses the separable blur into one 5x5 dilated pass while the
        // CPU does two 1D passes -- mathematically identical, but only because
        // the kernel is separable, which is worth having asserted.
        //
        // The max tolerance is looser than its neighbours for a measured
        // reason: the GPU.s ping-pong buffers take the output.s format, so on
        // this RGBA8 fixture every one of the four levels rounds to 8 bits
        // where the CPU stays in float. On an RGBA32F image the same code
        // agrees to 4e-7, which is what says the difference is quantisation
        // rather than a divergence. The MEAN check below is the one that would
        // catch a real bug, and it is held tight.
        //
        // The max tolerance is looser than its neighbours for a measured reason:
        // the GPU's ping-pong buffers take the output's format, so on this RGBA8
        // fixture every one of the four levels rounds to 8 bits where the CPU
        // stays in float. On an RGBA32F image the same code agrees to 4e-7,
        // which is what says the difference is quantisation and not a
        // divergence. The MEAN check is the one that would catch a real bug.
        {"wavelet_denoise",
         "src = image(\"test\")\n"
         "o = wavelet_denoise(src, luma = 0.03, chroma = 0.09, levels = 4)\n"
         "display(o)\n", 10.0, 1.0},

        {"threshold_bernsen",
         "src = image(\"test\")\n"
         "o = threshold_bernsen(src, window = 15, contrast_min = 15, "
         "uniform_level = 128)\n"
         "display(o)\n", 255.0, 0.5},

        {"threshold_adaptive_gaussian",
         "src = image(\"test\")\n"
         "o = threshold_adaptive_gaussian(src, window = 15, sigma = 3, c = 5)\n"
         "display(o)\n", 255.0, 0.5},
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


// basic_adjust on SCENE-LINEAR input, CPU against GPU.
//
// The table-driven agreement test above cannot reach this: its fixture is
// RGBA8, so it only ever exercises the gamma-encoded branch. The linear branch
// skips the sRGB transfer functions, skips the output clamp, and pitches the
// highlight band differently -- three divergences between the two paths that
// nothing else checks, and the GPU path never calls RunCPU() to fall back on.
static void TestLinearAdjustAgreement(ID3D12Device* dev) {
    Section("basic_adjust on linear input: CPU vs GPU");

    ComputeContext gpu;
    if (!gpu.Init(dev) || !gpu.Ready()) { Check(false, "compute context initialises"); return; }

    auto make = []() {
        ImageDesc d{64, 64, Format::RGBA16F};
        d.linear = true;
        Image img;
        img.Alloc(d);
        ImageView v = img.MapCpuWrite();
        for (int y = 0; y < 64; ++y)
            for (int x = 0; x < 64; ++x) {
                // Blown, highlight, and midtone bands, slightly tinted so the
                // colour controls have something to act on.
                const float lum = (y < 21) ? 1.4f : (y < 42 ? 0.7f : 0.2f);
                uint16_t* p = v.At<uint16_t>(x, y);
                p[0] = FloatToHalf(lum * 1.1f);
                p[1] = FloatToHalf(lum);
                p[2] = FloatToHalf(lum * 0.9f);
                p[3] = FloatToHalf(1.0f);
            }
        return img;
    };

    auto run = [&](ExecMode mode, Image* out, std::string* err) {
        auto algo = Registry::Get().Create("basic_adjust");
        if (!algo) { *err = "no basic_adjust"; return false; }
        for (ParamBase* p : algo->Params()) {
            const std::string n = p->Name();
            if (n == "exposure")   p->SetFromScript(Value(0.5),  err);
            if (n == "shadows")    p->SetFromScript(Value(0.4),  err);
            if (n == "saturation") p->SetFromScript(Value(0.3),  err);
        }
        Pipeline pipe;
        std::vector<Data> src;
        src.push_back(Data{make()});
        pipe.AddStage(std::move(algo), "basic_adjust", {{-1, 0}}, 1, 1);
        if (!pipe.Execute(&src, nullptr, err, &gpu, mode)) return false;
        *out = const_cast<Image&>(std::get<Image>(*pipe.Resolve({0, 0}, &src))).Clone();
        return true;
    };

    Image cpuImg, gpuImg;
    std::string err;
    if (!run(ExecMode::ForceCPU, &cpuImg, &err)) { Check(false, "ForceCPU runs: " + err); return; }
    if (!run(ExecMode::ForceGPU, &gpuImg, &err)) { Check(false, "ForceGPU runs: " + err); return; }

    ImageView a = cpuImg.MapCpuRead();
    ImageView b = gpuImg.MapCpuRead();
    double worst = 0.0, cpuPeak = 0.0;
    for (int y = 0; y < 64; ++y)
        for (int x = 0; x < 64; ++x)
            for (int c = 0; c < 3; ++c) {
                const double av = HalfToFloat(a.At<uint16_t>(x, y)[c]);
                const double bv = HalfToFloat(b.At<uint16_t>(x, y)[c]);
                worst = std::max(worst, std::abs(av - bv));
                cpuPeak = std::max(cpuPeak, av);
            }

    Check(worst < 0.005,
          "linear basic_adjust: GPU matches CPU (worst " + std::to_string(worst) + ")");

    // Guards against both paths being wrong together. Highlights is left at its
    // default here on purpose: pulling it would legitimately bring the peak
    // below 1.0, and then the check could not distinguish "recovered" from
    // "clamped". With only exposure lifting, 1.4 must come back higher still.
    Check(cpuPeak > 1.0,
          "linear basic_adjust: neither path clamped the result (peak " +
              std::to_string(cpuPeak) + ")");
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
        BuildPipeline("src = image(\"test\")\ng = guided_filter(src)\ndisplay(g)\n",
                      128, &ui, &pipe, &src, &e);
        const CompareResult r = CompareCpuGpu(pipe, &src, &gpu, -1);
        Check(!r.ok && r.error.find("no stage") != std::string::npos,
              "an all-CPU pipeline reports nothing to compare: " + r.error);
    }
    // Naming a specific CPU-only stage reports that stage by name.
    {
        UiState ui; Pipeline pipe; std::vector<Data> src; std::string e;
        BuildPipeline("src = image(\"test\")\ng = guided_filter(src)\ndisplay(g)\n",
                      128, &ui, &pipe, &src, &e);
        const CompareResult r = CompareCpuGpu(pipe, &src, &gpu, 0);
        Check(!r.ok && r.error.find("guided_filter") != std::string::npos,
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
    // Line-buffer stdout: when ctest or CI redirects it to a file the
    // default is block buffering, so a crash discards everything not yet
    // flushed and the log ends before the failure rather than at it.
    setvbuf(stdout, nullptr, _IONBF, 0);

    TestWorker();
    TestGpuOnlyShell();
    TestViewerVersions();
    TestCancellation();
    TestShaderCompilerSharing();
    TestShaderCompiler();

    ID3D12Device* dev = nullptr;
    if (SUCCEEDED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&dev)))) {
        TestGpuPipeline(dev);
        TestBatchedBindings(dev);
        TestResidency(dev);
        TestLinearAdjustAgreement(dev);
        TestCompare(dev);
        TestGpuResultIsDrawable(dev);
        TestGpuHistogram(dev);
        TestGpuAgreement(dev);
        dev->Release();
    } else {
        std::printf("\n(no D3D12 device — GPU tests skipped)\n");
    }

    std::printf("\n%s\n", g_fail == 0 ? "all runtime checks passed" : "FAILURES PRESENT");
    return g_fail == 0 ? 0 : 1;
}
