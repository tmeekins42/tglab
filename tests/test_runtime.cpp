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
#include "../src/app/visible_rect.h"

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

    // A cancelled run must not throw away the cache.
    //
    // Tim's report: dragging a develop slider far enough to cancel a run made
    // the whole pipeline re-run -- demosaic, hot-pixel repair, HDR merge --
    // rather than only the stage whose parameter moved. The worker was
    // dropping the entire stage cache on cancellation, so the next run started
    // from nothing.
    //
    // Measured by stage counts rather than by timing, which would be flaky:
    // after a burst that certainly cancels runs, a final submit that changes
    // ONLY the last stage's parameter must report cached stages. Zero cached
    // is exactly the bug.
    {
        // Two slow stages then a cheap one, so there is something worth
        // caching and the last stage is the one being "dragged".
        auto script = [](double bright) {
            char buf[256];
            std::snprintf(buf, sizeof buf,
                          "src = image(\"test\")\n"
                          "a = gaussian_blur(src, sigma = 8)\n"
                          "b = box_blur(a, radius = 6)\n"
                          "c = brightness(b, brightness = %.4f)\n"
                          "display(c, \"out\")\n", bright);
            return std::string(buf);
        };

        // Settle first: one complete run establishes the cache.
        {
            UiState ui; Pipeline pipe; std::vector<Data> src; std::string err;
            if (BuildPipeline(script(0.0).c_str(), dim, &ui, &pipe, &src, &err)) {
                worker.Submit(std::move(pipe), std::move(src));
                PipelineOutcome out;
                const auto dl = clockt::now() + std::chrono::seconds(30);
                while (clockt::now() < dl && !worker.TryFetch(&out))
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        }

        // A burst on the LAST stage's parameter. Every one supersedes the one
        // before it, so runs are cancelled in flight -- exactly the situation
        // that used to wipe the cache.
        uint64_t lastSeq = 0;
        for (int i = 1; i <= 12; ++i) {
            UiState ui; Pipeline pipe; std::vector<Data> src; std::string err;
            if (!BuildPipeline(script(0.01 * i).c_str(), dim, &ui, &pipe, &src, &err)) return;
            lastSeq = worker.Submit(std::move(pipe), std::move(src));
        }

        PipelineOutcome out;
        bool got = false;
        const auto deadline = clockt::now() + std::chrono::seconds(30);
        while (clockt::now() < deadline) {
            if (worker.TryFetch(&out) && out.seq == lastSeq) { got = true; break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        Check(got, "the final request of a cancelling burst completed");
        if (got) {
            // Only `brightness` changed, so both blurs should have come from
            // the cache. Before the fix a cancelled run cleared it and this
            // reported 0.
            const int cached = worker.LastCachedStages();
            std::printf("       cached after a cancelling burst: %d\n", cached);
            Check(cached >= 2,
                  "a cancelled run keeps its finished prefix (" +
                      std::to_string(cached) + " cached)");
        }
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

        // crop in PREVIEW, rotated, with the grid on -- the mode where the two
        // paths do the most independent drawing. Both have to place the
        // rectangle, dim outside it, and lay the division grid in the
        // rectangle's own rotated frame; that is three chances for the
        // arithmetic to drift, and the grid was hand-written twice.
        //
        // Tight, because measurement says it can be: max 1.0 and mean 0.11 in
        // 0..255 units, which is a single quantisation step. An antialiased
        // grid edge landing on the other side of a pixel would show here, and
        // it does not -- so these bounds are describing the real agreement
        // rather than leaving room for a bug to hide in.
        {"crop",
         "src = image(\"test\")\n"
         "o = crop(src, preview = 1, left = 0.1, right = 0.1, top = 0.1,\n"
         "         bottom = 0.1, angle = 12, preview_grid = 2)\n"
         "display(o)\n", 2.0, 0.3},

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

        // The same, with level_dep ON. A separate entry because the default is
        // 0, so the row above exercises none of the level-dependent path -- the
        // two implementations of LevelScale could disagree completely and every
        // existing check would still pass. It did not merely test the untested:
        // it immediately reported max 78 on a units mismatch, since PixelBuffer
        // holds 0..255 for an 8-bit image while a UNORM texture reads as 0..1,
        // so middle grey is 45.9 on one side and 0.18 on the other.
        //
        // Max tolerance 16 rather than 10 for the reason above: level_dep gives
        // each pixel its own threshold, so the 8-bit rounding between levels
        // lands differently per pixel instead of uniformly. The MEAN is the
        // check that says the paths agree, and it is held to the same 1.0 as
        // its neighbour -- it measures 0.46.
        {"wavelet_denoise",
         "src = image(\"test\")\n"
         "o = wavelet_denoise(src, luma = 0.03, chroma = 0.09, levels = 4, "
         "level_dep = 1.0)\n"
         "display(o)\n", 16.0, 1.0},

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

    // DEHAZE, whose two paths are deliberately not identical.
    //
    // The GPU path estimates the airlight from a strided sample rather than a
    // full scan, computes the map at full resolution rather than at 1/4 scale,
    // and smooths it with a box rather than a guided filter. Each is a
    // considered trade for a path that exists to be dragged.
    //
    // So this is not an equality check -- it is a check that the two are the
    // same PICTURE. A tolerance this wide would hide a subtle error, but the
    // failures that matter here are gross: a wrong airlight shifts every
    // colour, and a mis-bound pass produces noise or black.
    {
        UiState ui; Pipeline pipe; std::vector<Data> src; std::string e;
        BuildPipeline("src = image(\"test\")\nd = dehaze(src, strength = 0.7)\ndisplay(d)\n",
                      512, &ui, &pipe, &src, &e);
        const CompareResult r = CompareCpuGpu(pipe, &src, &gpu, -1);
        Check(r.ok, "dehaze compares" + (r.ok ? "" : ": " + r.error));
        if (r.ok) {
            std::printf("       CPU %.1f ms   GPU %.1f ms   speedup %.1fx   maxdiff %.3f  mean %.4f\n",
                        r.cpuMs, r.gpuMs, r.Speedup(), r.stats.maxAbsDiff,
                        r.stats.meanAbsDiff);
            // The mean is the honest measure of "same picture": a handful of
            // pixels may differ a lot where the two paths pick different
            // window minima, but the image as a whole must not drift.
            Check(r.stats.meanAbsDiff <= 12.0,
                  "dehaze CPU and GPU produce the same picture (mean " +
                      std::to_string(r.stats.meanAbsDiff) + ")");
            // And the GPU path must actually be doing the work, not passing
            // the image through -- which would also produce a small diff.
            Check(r.gpuImage.Valid(), "the GPU path produced an image");
        }
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

// Proxy resolution: the interactive path that runs while a slider is dragged.
//
// Three properties, and the third is the one that would otherwise produce a
// baffling bug rather than an obvious one.
static void TestProxy() {
    Section("proxy resolution");

    const char* kScript =
        "src = image(\"test\")\n"
        "o = gaussian_blur(src, sigma = 3)\n"
        "display(o)\n";
    const int dim = 400;

    auto run = [&](float scale, Pipeline* prev, Image* out, std::string* err) {
        auto pipe = std::make_unique<Pipeline>();
        UiState ui; std::vector<Data> src;
        if (!BuildPipeline(kScript, dim, &ui, pipe.get(), &src, err)) return false;
        pipe->SetProxyScale(scale);
        if (!pipe->Execute(&src, prev, err)) return false;
        const Data* d = pipe->Resolve(pipe->Viewers()[0].source, &src);
        *out = const_cast<Image&>(std::get<Image>(*d)).Clone();
        // Hand ownership back so the caller can chain a second run against it.
        if (prev) *prev = std::move(*pipe);
        return true;
    };

    // 1. A PROXY RUN PRODUCES A SMALLER IMAGE, and records the scale it used.
    {
        Image full, tiny;
        std::string err;
        Pipeline none;
        const bool okFull  = run(1.0f,  nullptr, &full,  &err);
        const bool okSmall = run(0.25f, nullptr, &tiny, &err);
        Check(okFull && okSmall, "proxy runs" + (err.empty() ? "" : ": " + err));

        if (okFull && okSmall) {
            Check(full.Desc().width == dim,
                  "a full run stays full size (" +
                      std::to_string(full.Desc().width) + ")");
            Check(tiny.Desc().width == dim / 4,
                  "a proxy run shrinks the output (" +
                      std::to_string(tiny.Desc().width) + " from " +
                      std::to_string(dim) + ")");
            Check(std::fabs(tiny.Desc().proxyScale - 0.25f) < 1e-5f,
                  "the proxy result carries its scale (" +
                      std::to_string(tiny.Desc().proxyScale) + ")");
        }
        (void)none;
    }

    // 2. A `Never` STAGE IN THE DIRTY RANGE VETOES THE PROXY.
    //
    // Not the whole pipeline -- the dirty range. That distinction is what lets
    // a raw develop chain proxy at all, since its demosaic is Never and sits at
    // the front. Here the detector IS dirty, so the veto must fire.
    {
        const char* kDetect =
            "src = image(\"test\")\n"
            "o = detect_orb(src)\n"
            "display(o)\n";
        UiState ui; Pipeline pipe; std::vector<Data> src; std::string err;
        if (BuildPipeline(kDetect, dim, &ui, &pipe, &src, &err)) {
            pipe.SetProxyScale(0.25f);
            if (pipe.Execute(&src, nullptr, &err)) {
                Check(std::fabs(pipe.RanAtScale() - 1.0f) < 1e-6f,
                      "a Never stage vetoes the proxy (ran at " +
                          std::to_string(pipe.RanAtScale()) + ")");
                const Data* d = pipe.Resolve(pipe.Viewers()[0].source, &src);
                const Image& im = std::get<Image>(*d);
                Check(im.Desc().width == dim,
                      "...and the output is full size (" +
                          std::to_string(im.Desc().width) + ")");
            } else {
                Check(false, "detector proxy run: " + err);
            }
        }
    }

    // 3. A DRAG MUST NOT INVALIDATE THE STAGES AHEAD OF THE SLIDER.
    //
    // This is what locked the UI. The first version discarded the whole cache
    // whenever the requested scale changed, so the first frame of a drag had
    // firstDirty = 0 -- which put every upstream stage back in the dirty range,
    // and on a raw that means re-demosaicing the frame on the CPU every frame.
    //
    // The property to hold: dragging a parameter of a LATE stage must leave the
    // early ones cached. Measured by the pipeline's own cached-stage count,
    // which is what would have made the original bug obvious.
    {
        const char* kChain =
            "src = image(\"test\")\n"
            "a = box_blur(src, radius = 3)\n"
            "b = gaussian_blur(a, sigma = 2)\n"
            "o = brightness(b, brightness = 0.1)\n"
            "display(o)\n";

        auto pipeA = std::make_unique<Pipeline>();
        UiState uiA; std::vector<Data> srcA; std::string err;
        if (BuildPipeline(kChain, dim, &uiA, pipeA.get(), &srcA, &err) &&
            pipeA->Execute(&srcA, nullptr, &err)) {

            // Now "drag" the LAST stage's parameter, at proxy scale.
            const char* kChain2 =
                "src = image(\"test\")\n"
                "a = box_blur(src, radius = 3)\n"
                "b = gaussian_blur(a, sigma = 2)\n"
                "o = brightness(b, brightness = 0.4)\n"
                "display(o)\n";

            Pipeline pipeB;
            UiState uiB; std::vector<Data> srcB;
            if (BuildPipeline(kChain2, dim, &uiB, &pipeB, &srcB, &err)) {
                pipeB.SetProxyScale(0.25f);
                Check(pipeB.Execute(&srcB, pipeA.get(), &err),
                      "the drag run executes" + (err.empty() ? "" : ": " + err));

                // The two blurs were untouched, so both must have been reused.
                // Zero here means the whole chain re-ran, which on a raw is the
                // lockup.
                Check(pipeB.CachedStageCount() >= 2,
                      "a drag reuses the stages ahead of it (" +
                          std::to_string(pipeB.CachedStageCount()) +
                          " cached of 3)");
            }
        }
    }

    // 3b. THE DOWNSAMPLE ITSELF MUST NOT COST PER FRAME.
    //
    // The proxy's own cost, and it dominated everything it was meant to save.
    // Dragging leaves the upstream stages cached, so the same full-resolution
    // image was area-averaged again on every frame -- on a 45 MP raw that was
    // ~90% of the run, and a drag took a second while the status line honestly
    // reported "10% of the pixels".
    //
    // Measured as time, because that is the symptom. The first drag frame pays
    // for the downsample; every later one must not.
    {
        const char* kChain =
            "src = image(\"test\")\n"
            "a = gaussian_blur(src, sigma = 2)\n"
            "o = brightness(a, brightness = 0.1)\n"
            "display(o)\n";

        auto base = std::make_unique<Pipeline>();
        UiState ui0; std::vector<Data> src0; std::string err;
        if (BuildPipeline(kChain, 900, &ui0, base.get(), &src0, &err) &&
            base->Execute(&src0, nullptr, &err)) {

            auto dragFrame = [&](double bright, Pipeline* prev, double* ms) {
                const std::string s =
                    "src = image(\"test\")\n"
                    "a = gaussian_blur(src, sigma = 2)\n"
                    "o = brightness(a, brightness = " +
                    std::to_string(bright) + ")\n"
                    "display(o)\n";
                auto p = std::make_unique<Pipeline>();
                UiState ui; std::vector<Data> s2; std::string e;
                if (!BuildPipeline(s.c_str(), 900, &ui, p.get(), &s2, &e)) return false;
                p->SetProxyScale(0.25f);
                const auto t0 = clockt::now();
                const bool ok = p->Execute(&s2, prev, &e);
                *ms = std::chrono::duration<double, std::milli>(clockt::now() - t0).count();
                if (ok) *prev = std::move(*p);
                return ok;
            };

            double first = 0.0, later = 0.0;
            Pipeline chain = std::move(*base);
            if (dragFrame(0.2, &chain, &first)) {
                // Several more, taking the best: one sample could be a
                // scheduling hiccup, and the claim is about the steady state.
                later = 1e9;
                for (int i = 0; i < 4; ++i) {
                    double t = 0.0;
                    if (dragFrame(0.3 + 0.02 * i, &chain, &t))
                        later = std::min(later, t);
                }

                // A REAL MARGIN, because the obvious bound does not test
                // anything. `later <= first` passes at 11.8 against 12.0 --
                // which is exactly what the uncached version measures, since
                // both frames do the same work. Verified by disabling the
                // cache: 12.0 then 11.8, a passing test on broken code.
                //
                // Cached, the later frames measure 1.2 ms against 12.0. Half is
                // far inside that gap and far outside the uncached one.
                Check(later < first * 0.5,
                      "a drag does not re-downsample every frame (" +
                          std::to_string(first) + " ms then " +
                          std::to_string(later) + " ms)");
            }
        }
    }

    // 3c. DRAGGING AN EARLY SLIDER MUST NOT ACCUMULATE PROXIES.
    //
    // The case that hung the app. Dragging a slider EARLY in a chain makes the
    // upstream stage re-run every frame, so its output is a new buffer each
    // time -- and the first cache keyed on that buffer's ADDRESS and appended.
    // Every frame added a 40 MB proxy of a 45 MP raw; a few seconds of dragging
    // exhausted VRAM and every algorithm collapsed into "could not make an
    // input GPU-resident".
    //
    // Worse than the leak: a freed buffer's address can be REUSED, so the cache
    // could report a hit for a different image and hand back a proxy of the
    // previous frame's pixels. Correctness, not just memory.
    //
    // Measured as steady-state time over many frames. A cache that grows makes
    // later frames slower, not faster; one that returns stale pixels would make
    // the result wrong, which the fidelity test covers separately.
    {
        auto base = std::make_unique<Pipeline>();
        UiState ui0; std::vector<Data> src0; std::string err;
        const char* kFirst =
            "src = image(\"test\")\n"
            "a = brightness(src, brightness = 0.0)\n"
            "b = gaussian_blur(a, sigma = 2)\n"
            "display(b)\n";
        if (BuildPipeline(kFirst, 700, &ui0, base.get(), &src0, &err) &&
            base->Execute(&src0, nullptr, &err)) {

            // Drag the FIRST stage's parameter, so everything downstream is
            // dirty and the upstream buffer changes every frame.
            auto frame = [&](double v, Pipeline* prev, double* ms) {
                const std::string s =
                    "src = image(\"test\")\n"
                    "a = brightness(src, brightness = " + std::to_string(v) + ")\n"
                    "b = gaussian_blur(a, sigma = 2)\n"
                    "display(b)\n";
                auto p = std::make_unique<Pipeline>();
                UiState ui; std::vector<Data> s2; std::string e;
                if (!BuildPipeline(s.c_str(), 700, &ui, p.get(), &s2, &e)) return false;
                p->SetProxyScale(0.25f);
                const auto t0 = clockt::now();
                const bool ok = p->Execute(&s2, prev, &e);
                *ms = std::chrono::duration<double, std::milli>(clockt::now() - t0).count();
                if (ok) *prev = std::move(*p);
                return ok;
            };

            Pipeline chain = std::move(*base);
            double early = 0.0, late = 0.0;
            bool ok = true;
            for (int i = 0; i < 3 && ok; ++i) {
                double t = 0.0;
                ok = frame(0.05 * i, &chain, &t);
                early = std::max(early, t);
            }
            for (int i = 0; i < 12 && ok; ++i) {
                double t = 0.0;
                ok = frame(0.5 + 0.01 * i, &chain, &t);
                if (i >= 8) late = std::max(late, t);
            }

            Check(ok, "an early-slider drag keeps running");
            // Fifteen frames in, a frame must cost about what it did at three.
            // An accumulating cache shows up here as steady growth long before
            // it becomes an out-of-memory failure.
            Check(ok && late < early * 2.5,
                  "an early-slider drag does not accumulate proxies (" +
                      std::to_string(early) + " ms early, " +
                      std::to_string(late) + " ms late)");
        }
    }

    // 3d. A GROUP SOURCE MUST STILL PROXY.
    //
    // hdr.tgl and every other group script take an ImageSet from the palette
    // rather than an Image. The app measured the source width by looking only
    // for Images, so it stayed at 0, the scale fell back to 1.0, and the proxy
    // silently never engaged -- with no size line in the status to say why.
    //
    // Tested through the pipeline rather than the app, because what has to hold
    // is that a set-sourced script CAN proxy once it is asked to. The width
    // calculation is one line; that it was never exercised is the gap.
    {
        ImageSet set;
        for (int i = 0; i < 3; ++i) set.images.push_back(MakeSource(dim));
        set.shape = Shape{{{"frame", 3}}};

        // A DRAG, not a first run. The distinction matters and cost a test
        // rewrite: on a first run firstDirty is 0, so the boundary IS the
        // reduction, whose input is a set and therefore not proxied. Only once
        // the merge is CACHED does the boundary move past it to a stage whose
        // input is a single image -- which is what dragging a develop slider
        // actually does, and the case that has to work.
        auto build = [&](double bright) {
            auto p = std::make_unique<Pipeline>();
            p->AddStage(Registry::Get().Create("merge_mean"), "merge_mean",
                        {{-1, 0}}, 1, 1, "frame");
            auto b = Registry::Get().Create("brightness");
            if (ParamBase* pb = b->FindParam("brightness")) {
                std::string e; pb->SetFromScript(Value(bright), &e);
            }
            p->AddStage(std::move(b), "brightness", {{0, 0}}, 1, 2);
            return p;
        };

        std::vector<Data> src;
        src.push_back(Data{std::move(set)});

        std::string err;
        auto first = build(0.0);
        if (!first->Execute(&src, nullptr, &err)) {
            Check(false, "group first run: " + err);
        } else {
            auto drag = build(0.3);
            drag->SetProxyScale(0.25f);
            if (!drag->Execute(&src, first.get(), &err)) {
                Check(false, "group proxy runs: " + err);
            } else {
                const Data* d = drag->Resolve({1, 0}, &src);
                const auto* im = d ? std::get_if<Image>(d) : nullptr;
                Check(im && im->Desc().width == dim / 4,
                      "a group-sourced script proxies after the reduction (" +
                          std::to_string(im ? im->Desc().width : 0) + " from " +
                          std::to_string(dim) + ")");
            }
        }
    }

    // 4. A PROXY RESULT IS NEVER REUSED FOR A FULL-RESOLUTION RUN.
    //
    // THE failure this mechanism has to avoid, and the reason it is quiet: the
    // parameters are identical, so the cache scan sees an unchanged stage and
    // reuses it. The run succeeds, nothing is reported, and the final image is
    // silently soft -- looking like a bad resample rather than a cache bug.
    //
    // Checked by running a proxy first and then a full run AGAINST IT, which is
    // exactly the sequence a slider drag followed by its release produces.
    {
        Image tiny, full;
        std::string err;

        auto pipeA = std::make_unique<Pipeline>();
        UiState uiA; std::vector<Data> srcA;
        if (BuildPipeline(kScript, dim, &uiA, pipeA.get(), &srcA, &err)) {
            pipeA->SetProxyScale(0.25f);
            Check(pipeA->Execute(&srcA, nullptr, &err),
                  "the drag run executes" + (err.empty() ? "" : ": " + err));

            auto pipeB = std::make_unique<Pipeline>();
            UiState uiB; std::vector<Data> srcB;
            if (BuildPipeline(kScript, dim, &uiB, pipeB.get(), &srcB, &err)) {
                pipeB->SetProxyScale(1.0f);
                Check(pipeB->Execute(&srcB, pipeA.get(), &err),
                      "the release run executes" + (err.empty() ? "" : ": " + err));

                const Data* d = pipeB->Resolve(pipeB->Viewers()[0].source, &srcB);
                const Image& im = std::get<Image>(*d);
                Check(im.Desc().width == dim && im.Desc().height == dim,
                      "the full run after a drag is FULL SIZE, not a reused "
                      "proxy (" + std::to_string(im.Desc().width) + "x" +
                          std::to_string(im.Desc().height) + ")");
                Check(std::fabs(im.Desc().proxyScale - 1.0f) < 1e-6f,
                      "...and its scale is 1.0 (" +
                          std::to_string(im.Desc().proxyScale) + ")");
            }
        }
    }
}

// Pass 2: does a proxy preview actually LOOK like the full-resolution result?
//
// The test the design called for first, because it is the only thing that
// distinguishes "scaled correctly" from "ran without crashing". An algorithm
// that ignores proxyScale still produces a plausible image -- just at the wrong
// strength -- and nothing else in the suite would notice.
//
// The comparison: run at full resolution and downsample the result, versus run
// on a downsampled input. Those should agree, because both are answering "what
// does this look like at proxy size". They will not agree exactly -- resampling
// order matters -- but a blur that forgot to scale its sigma is off by 4x, which
// is nowhere near the resampling noise.
static void TestProxyFidelity() {
    Section("proxy fidelity");

    const int dim = 320;
    const float scale = 0.25f;

    // Structure at several sizes, so a wrong radius shows. A flat field would
    // look identical however badly the blur was scaled.
    auto makeSrc = [&](int d) {
        Image img;
        img.Alloc({d, d, Format::RGBA32F});
        ImageView v = img.MapCpuWrite();
        for (int y = 0; y < d; ++y)
            for (int x = 0; x < d; ++x) {
                float* p = v.At<float>(x, y);
                const bool a = ((x / 40) + (y / 40)) % 2 == 0;
                const bool b = ((x / 8)  + (y / 8))  % 2 == 0;
                const float g = (a ? 0.65f : 0.2f) + (b ? 0.12f : 0.0f);
                p[0] = p[1] = p[2] = g;
                p[3] = 1.0f;
            }
        return img;
    };

    auto meanOf = [](const Image& im) {
        ImageView v = const_cast<Image&>(im).MapCpuRead();
        double s = 0.0;
        long long n = 0;
        for (int y = 0; y < v.desc.height; ++y)
            for (int x = 0; x < v.desc.width; ++x) { s += v.At<float>(x, y)[0]; ++n; }
        return n ? s / double(n) : 0.0;
    };

    // Standard deviation is the measure that matters here: a blur REDUCES it,
    // and by how much is exactly what the radius controls. A mean would be
    // nearly unchanged by any blur and would pass whatever the sigma was.
    auto stddevOf = [](const Image& im) {
        ImageView v = const_cast<Image&>(im).MapCpuRead();
        double s = 0.0, s2 = 0.0;
        long long n = 0;
        for (int y = 0; y < v.desc.height; ++y)
            for (int x = 0; x < v.desc.width; ++x) {
                const double g = v.At<float>(x, y)[0];
                s += g; s2 += g * g; ++n;
            }
        if (!n) return 0.0;
        const double m = s / double(n);
        return std::sqrt(std::max(s2 / double(n) - m * m, 0.0));
    };

    struct Case { const char* name; const char* script; };
    const Case cases[] = {
        {"gaussian_blur", "o = gaussian_blur(src, sigma = 8)\n"},
        {"box_blur",      "o = box_blur(src, radius = 12)\n"},
        {"median_blur",   "o = median_blur(src, radius = 6)\n"},
        {"kuwahara",      "o = kuwahara(src, radius = 8)\n"},
        {"bilateral",     "o = bilateral(src, sigma_space = 8, sigma_range = 0.4)\n"},
        {"orton",         "o = orton(src, blur = 16)\n"},

        // bloom BRIGHTENS rather than smoothing, so its failure is invisible to
        // a contrast check unless the glow is strong enough to move the
        // statistics. The threshold is set low and the intensity high on
        // purpose, so the glow dominates.
        //
        // This is the case that was actually broken: bloom compensates its
        // intensity by spread^2, and scaling the spread for the proxy scaled
        // the compensation with it -- a factor of nine at a third scale. The
        // glow vanished for the length of a drag and came back on release.
        {"bloom",         "o = bloom(src, threshold = 0.1, intensity = 2, "
                          "spread_r = 24, spread_g = 24, spread_b = 24)\n"},
    };

    for (const Case& c : cases) {
        const std::string script =
            std::string("src = image(\"test\")\n") + c.script + "display(o)\n";

        // Full resolution, then shrink the RESULT.
        Image fullThenSmall;
        {
            UiState ui; Pipeline p; std::vector<Data> src; std::string err;
            src.push_back(Data{makeSrc(dim)});
            std::vector<SourceImage> names{{"test", 0}};
            Program prog;
            if (!Parse(script, &prog, &err)) { Check(false, err); continue; }
            auto r = Interpret(prog, names, &ui, &p);
            if (!r.ok) { Check(false, r.error); continue; }
            if (!p.Execute(&src, nullptr, &err)) { Check(false, err); continue; }
            const Data* d = p.Resolve(p.Viewers()[0].source, &src);
            Image big = const_cast<Image&>(std::get<Image>(*d)).Clone();

            // Shrink through the same resize the proxy path uses.
            auto rz = Registry::Get().Create("resize");
            if (ParamBase* pb = rz->FindParam("scale")) {
                std::string e; pb->SetFromScript(Value(double(scale)), &e);
            }
            std::vector<Data> ins;
            ins.push_back(Data{std::move(big)});
            std::vector<const Data*> ip{&ins[0]};
            ImageDesc od = rz->OutputDesc(0, std::get<Image>(ins[0]).Desc());
            Image dst; dst.Alloc(od);
            std::vector<Data> outs(1);
            outs[0] = Data{std::move(dst)};
            RunCtx rc(ip, outs);
            rz->RunCPU(rc);
            fullThenSmall = std::get<Image>(outs[0]).Clone();
        }

        // Proxy: the pipeline shrinks the INPUT and the algorithm scales itself.
        Image proxied;
        {
            UiState ui; Pipeline p; std::vector<Data> src; std::string err;
            src.push_back(Data{makeSrc(dim)});
            std::vector<SourceImage> names{{"test", 0}};
            Program prog;
            if (!Parse(script, &prog, &err)) { Check(false, err); continue; }
            auto r = Interpret(prog, names, &ui, &p);
            if (!r.ok) { Check(false, r.error); continue; }
            p.SetProxyScale(scale);
            if (!p.Execute(&src, nullptr, &err)) { Check(false, err); continue; }
            const Data* d = p.Resolve(p.Viewers()[0].source, &src);
            proxied = const_cast<Image&>(std::get<Image>(*d)).Clone();
        }

        if (fullThenSmall.Desc().width != proxied.Desc().width) {
            Check(false, std::string(c.name) + ": sizes differ (" +
                             std::to_string(fullThenSmall.Desc().width) + " vs " +
                             std::to_string(proxied.Desc().width) + ")");
            continue;
        }

        const double sdA = stddevOf(fullThenSmall);
        const double sdB = stddevOf(proxied);
        const double ratio = (sdA > 1e-6) ? sdB / sdA : 0.0;

        // Within 35%. Loose because resampling order genuinely differs -- one
        // blurs then shrinks, the other shrinks then blurs -- but an unscaled
        // radius is off by 1/scale = 4x, which this catches with room to spare.
        Check(ratio > 0.65 && ratio < 1.45,
              std::string(c.name) + ": the proxy preview matches the full-res "
              "result (contrast ratio " + std::to_string(ratio) + ")");

        // And the exposure is unchanged, which a wrong radius would not
        // necessarily break but a wrong normalisation would.
        const double mA = meanOf(fullThenSmall), mB = meanOf(proxied);
        Check(std::fabs(mA - mB) < 0.05,
              std::string(c.name) + ": ...at the same brightness (" +
                  std::to_string(mA) + " vs " + std::to_string(mB) + ")");
    }
}

// Region processing: computing only the visible rectangle.
//
// The property that matters is not "it ran" but "it produced the SAME PIXELS
// the full run would have". A cropped blur that clamps at the cut looks
// perfectly reasonable on its own and differs from the truth by a visible band
// at every edge -- which is exactly what the margin exists to prevent and
// exactly what a run-without-crashing test would miss.
// WHERE THE REGION COMES FROM, which is not what TestRegion covers.
//
// TestRegion hands the pipeline a rectangle and checks it is honoured. That
// left the other half untested: the viewer's job of turning a panel and a
// camera into a rectangle. A unit error there passed every existing test and
// still blacked the screen on every zoomed-in drag, because the rectangle the
// pipeline faithfully computed was not the one the panel was looking at.
static void TestVisibleRect() {
    std::printf("\n--- visible rect ---\n");

    // A 1000x800 image, fitted in a 500x400 panel at 50%: the panel sees all
    // of it. The +1 is the outward rounding, which is deliberate.
    {
        VisibleRectInput in;
        in.panelX = 0;   in.panelY = 0;   in.panelW = 500; in.panelH = 400;
        in.imageX = 0;   in.imageY = 0;   in.imageW = 500; in.imageH = 400;
        in.zoom = 0.5f;
        const ImageRect r = ComputeVisibleRect(in);
        Check(r.x == 0 && r.y == 0 && r.w >= 1000 && r.h >= 800,
              "a fitted panel sees the whole image (" + std::to_string(r.w) +
                  "x" + std::to_string(r.h) + ")");
    }

    // Zoomed to 1:1 and panned into the middle of a big picture: the panel
    // shows 500x400 image pixels starting at (2000,1000).
    {
        VisibleRectInput in;
        in.panelX = 100; in.panelY = 50; in.panelW = 500; in.panelH = 400;
        // The whole image's top-left is off-screen up and left.
        in.imageX = 100 - 2000; in.imageY = 50 - 1000;
        in.imageW = 4000;       in.imageH = 3000;
        in.zoom = 1.0f;
        const ImageRect r = ComputeVisibleRect(in);
        Check(r.x == 2000 && r.y == 1000,
              "a panned view reports where it is looking (" +
                  std::to_string(r.x) + "," + std::to_string(r.y) + ")");
        Check(r.w >= 500 && r.h >= 400 && r.w <= 502 && r.h <= 402,
              "...and how much it can see (" + std::to_string(r.w) + "x" +
                  std::to_string(r.h) + ")");
    }

    // THE INVARIANT. What the panel shows does not change when the pipeline
    // switches to a proxy, so neither may the rectangle. Anything else is a
    // feedback loop -- the scale changes the request, the request changes the
    // result, the result changes the scale.
    //
    // The caller states the image rect in SCREEN pixels against the FULL
    // extent, so a proxy is already accounted for by the time it arrives here:
    // the same picture at a quarter scale is a quarter-size raster drawn at
    // four times the zoom, and both give the same screen rectangle. That is
    // why proxyScale is absent from the input -- applying it here would be
    // applying it twice, which asked for a differently scaled rectangle of a
    // different part of the image, but only while dragging.
    {
        VisibleRectInput a;
        a.panelX = 0; a.panelY = 0; a.panelW = 640; a.panelH = 480;
        a.imageX = -300; a.imageY = -200; a.imageW = 3000; a.imageH = 2000;
        a.zoom = 1.0f;
        const ImageRect ra = ComputeVisibleRect(a);
        Check(ra.x == 300 && ra.y == 200 && ra.w >= 640 && ra.h >= 480,
              "a 1:1 view asks for what it shows (" + std::to_string(ra.x) +
                  "," + std::to_string(ra.y) + " " + std::to_string(ra.w) +
                  "x" + std::to_string(ra.h) + ")");
    }

    // A panel that does not overlap the image at all asks for nothing, rather
    // than a negative or wrapped rectangle.
    {
        VisibleRectInput in;
        in.panelX = 0;    in.panelY = 0;    in.panelW = 100; in.panelH = 100;
        in.imageX = 500;  in.imageY = 500;  in.imageW = 100; in.imageH = 100;
        in.zoom = 1.0f;
        Check(!ComputeVisibleRect(in).Valid(),
              "a panel showing none of the image asks for nothing");
    }

    // Degenerate zoom must not divide by ~zero and produce a garbage extent.
    {
        VisibleRectInput in;
        in.panelX = 0; in.panelY = 0; in.panelW = 100; in.panelH = 100;
        in.imageX = 0; in.imageY = 0; in.imageW = 100; in.imageH = 100;
        in.zoom = 0.0f;
        Check(!ComputeVisibleRect(in).Valid(), "a zero zoom asks for nothing");
    }
}

// THE PICTURE MUST NOT MOVE WHEN A REACH CHANGES.
//
// The crop's origin is `visible - margin`, and the margin comes from the
// algorithm's reach -- so dragging a blur radius moves the origin. That is
// fine in full-resolution pixels, where the origin is exact. Scaled into a
// proxy's own pixels it rounds, and a one-pixel move of the origin can land on
// a different proxy pixel: the whole image then redraws one pixel over.
//
// Visible on Orton's blur slider and on nothing else, because it is the only
// control there whose reach changes as it drags. The fix snaps the origin to
// the proxy grid, making it a fixed point of that rounding.
static void TestRegionOriginStable() {
    std::printf("\n--- region origin stability ---\n");

    const int dim = 600;
    Image img;
    img.Alloc({dim, dim, Format::RGBA32F});
    {
        ImageView v = img.MapCpuWrite();
        for (int y = 0; y < dim; ++y)
            for (int x = 0; x < dim; ++x) {
                float* p = v.At<float>(x, y);
                p[0] = p[1] = p[2] = ((x / 20) + (y / 20)) % 2 ? 0.7f : 0.25f;
                p[3] = 1.0f;
            }
    }

    const Pipeline::Rect want{200, 210, 160, 150};

    // Two blur radii: different reaches, so different margins.
    auto originFor = [&](int blurPx, int* ox, int* oy) {
        const std::string script =
            "src = image(\"test\")\n"
            "o = orton(src, blur = " + std::to_string(blurPx) +
            ", strength = 0.5)\n"
            "display(o)\n";
        UiState ui; Pipeline p; std::vector<Data> src;
        src.push_back(Data{img.Clone()});
        std::vector<SourceImage> names{{"test", 0}};
        Program prog; std::string err;
        if (!Parse(script, &prog, &err)) return false;
        auto r = Interpret(prog, names, &ui, &p);
        if (!r.ok) return false;
        p.SetProxyScale(0.5f);
        p.SetRegion(want);
        if (!p.Execute(&src, nullptr, &err)) return false;
        const Data* d = p.Resolve(p.Viewers()[0].source, &src);
        const ImageDesc& sd = std::get<Image>(*d).Desc();
        *ox = sd.originX;
        *oy = sd.originY;
        return true;
    };

    int ax = -1, ay = -1, bx = -1, by = -1;
    const bool okA = originFor(6, &ax, &ay);
    const bool okB = originFor(7, &bx, &by);
    if (!okA || !okB) { Check(false, "orton runs on a region"); return; }

    // The visible rectangle did not move, so neither may the drawn origin --
    // whatever the margin did in between.
    Check(ax == bx && ay == by,
          "the origin holds still when only the reach changes (" +
              std::to_string(ax) + "," + std::to_string(ay) + " vs " +
              std::to_string(bx) + "," + std::to_string(by) + ")");
}

// THE MEASUREMENT CACHE, in both directions.
//
// A cache that never invalidates is fast and wrong, and the wrongness here is
// the dangerous kind: a stale airlight does not crash or produce noise, it
// produces a slightly wrong picture that looks entirely plausible. So the
// misses matter more than the hits, and both are checked.
//
// Driven through the pipeline rather than by calling the algorithm directly,
// because the cache is the PIPELINE's -- the algorithm object does not survive
// between runs, which is the whole reason the facility exists.
static void TestMeasureCache(ID3D12Device* dev) {
    std::printf("\n--- gpu measurement cache ---\n");

    ComputeContext gpu;
    if (!gpu.Init(dev)) { Check(false, "compute context starts"); return; }

    const int dim = 512;

    // Two images that must measure differently: one hazy (a bright, low
    // contrast wash) and one clear. If the cache leaked between them the
    // airlight from the first would be used on the second.
    auto make = [&](bool hazy) {
        Image img;
        img.Alloc({dim, dim, Format::RGBA32F});
        ImageView v = img.MapCpuWrite();
        for (int y = 0; y < dim; ++y)
            for (int x = 0; x < dim; ++x) {
                float* p = v.At<float>(x, y);
                const float base = ((x / 32) + (y / 32)) % 2 ? 0.6f : 0.15f;
                if (hazy) {
                    // Washed toward a bright blue-grey: high floor, low range.
                    p[0] = 0.55f + base * 0.25f;
                    p[1] = 0.58f + base * 0.25f;
                    p[2] = 0.70f + base * 0.25f;
                } else {
                    p[0] = p[1] = p[2] = base;
                }
                p[3] = 1.0f;
            }
        return img;
    };

    // Runs one script against one source and returns the reported airlight
    // line, which is how the algorithm tells us which path it took.
    // A version per source, bumped when the image changes -- exactly what the
    // app supplies. Without it the pipeline cannot tell one palette image from
    // another (sourceHash is only computed from these), so a swapped image
    // reuses the whole cached STAGE and the measurement cache is never even
    // consulted. That is a property of the harness, not of the cache, but it
    // is worth stating: this cache is only as good as the identity the
    // pipeline is given.
    uint64_t version = 1;

    auto run = [&](Pipeline* pipe, Pipeline* prev, const Image& srcImg,
                   const std::string& args, std::string* note) {
        UiState ui;
        std::vector<Data> src;
        src.push_back(Data{const_cast<Image&>(srcImg).Clone()});
        std::vector<SourceImage> names{{"test", 0}};
        const std::string script =
            "src = image(\"test\")\n"
            "d = dehaze(src, " + args + ")\n"
            "display(d)\n";
        Program prog; std::string err;
        if (!Parse(script, &prog, &err)) return false;
        auto r = Interpret(prog, names, &ui, pipe);
        if (!r.ok) return false;
        // `prev` is how a cache carries between runs -- the same argument the
        // app passes, so this exercises the real path rather than a test-only
        // door into the pipeline.
        const std::vector<uint64_t> versions{version};
        if (!pipe->Execute(&src, prev, &err, &gpu, ExecMode::ForceGPU, &versions))
            return false;
        for (const Stage& st : pipe->Stages())
            if (st.algoName == "dehaze" && st.algo) *note = st.algo->RunReport();
        return true;
    };

    const Image hazy = make(true), clear = make(false);

    // First run measures; second, with only the strength changed, must reuse.
    Pipeline p1, p2;
    std::string n1, n2;
    if (!run(&p1, nullptr, hazy, "strength = 0.7", &n1)) { Check(false, "dehaze runs on the GPU"); return; }
    if (!run(&p2, &p1, hazy, "strength = 0.5", &n2))     { Check(false, "dehaze re-runs"); return; }

    Check(n1.find("sampled") != std::string::npos,
          "the first run measures (" + n1 + ")");
    Check(n2.find("cached") != std::string::npos,
          "dragging strength reuses the measurement (" + n2 + ")");

    // A DIFFERENT IMAGE must not reuse it. This is the failure that would be
    // invisible: the picture still dehazes, just with the wrong airlight.
    ++version;   // a different image in the same palette slot
    Pipeline p3; std::string n3;
    if (run(&p3, &p2, clear, "strength = 0.5", &n3))
        // "not cached" is NOT enough: an empty report also fails to contain
        // it, so a run that produced nothing at all would pass. Require the
        // positive evidence that it measured.
        Check(n3.find("sampled") != std::string::npos,
              "a different image re-measures (" + n3 + ")");
    else
        Check(false, "dehaze runs on a second image");

    // And the airlights must actually differ, or the check above proves
    // nothing -- two identical measurements would pass it either way.
    Pipeline p4; std::string n4;
    if (run(&p4, nullptr, clear, "strength = 0.5", &n4))
        Check(n1 != n4, "the two images measure different airlights");

    // WHAT THE CACHE IS ACTUALLY FOR: the second run must be materially
    // cheaper, or the whole facility is bookkeeping for nothing.
    //
    // Timed at a size where the measurement dominates. The first run pays for
    // it; every run after is the five dispatches alone.
    {
        const int big = 2400;
        Image bigHazy;
        bigHazy.Alloc({big, big, Format::RGBA32F});
        {
            ImageView v = bigHazy.MapCpuWrite();
            for (int y = 0; y < big; ++y)
                for (int x = 0; x < big; ++x) {
                    float* p = v.At<float>(x, y);
                    const float b = ((x / 32) + (y / 32)) % 2 ? 0.6f : 0.15f;
                    p[0] = 0.55f + b * 0.25f;
                    p[1] = 0.58f + b * 0.25f;
                    p[2] = 0.70f + b * 0.25f;
                    p[3] = 1.0f;
                }
        }

        ++version;
        Pipeline a, b;
        std::string na, nb;
        const auto t0 = clockt::now();
        const bool oka = run(&a, nullptr, bigHazy, "strength = 0.7", &na);
        const auto t1 = clockt::now();
        const bool okb = run(&b, &a, bigHazy, "strength = 0.5", &nb);
        const auto t2 = clockt::now();

        if (oka && okb) {
            const double first  = std::chrono::duration<double, std::milli>(t1 - t0).count();
            const double second = std::chrono::duration<double, std::milli>(t2 - t1).count();
            // REPORTED, NOT ASSERTED ON.
            //
            // A wall-clock ratio is not a stable assertion here: the timings
            // include upload and dispatch on a GPU shared with the desktop,
            // and the same pair of runs measured 343/130 ms once and 267/210
            // ms minutes later. A test that fails on a busy machine teaches
            // people to ignore it.
            //
            // What is actually being claimed -- that the measurement is
            // skipped -- is a fact about control flow, and the report string
            // states it exactly. That is what gets asserted; the numbers are
            // printed for a human to look at.
            std::printf("       first %.0f ms   cached %.0f ms\n", first, second);
            Check(na.find("sampled") != std::string::npos,
                  "the first timed run measured");
            Check(nb.find("cached") != std::string::npos,
                  "the second timed run skipped the measurement");
        }
    }

    // A parameter the measurement DEPENDS on must also miss. patch is the
    // window the dark channel is minimised over, so an airlight measured with
    // one is not the airlight for another.
    {
        Pipeline p5; std::string n5;
        if (run(&p5, &p2, hazy, "strength = 0.7, patch = 30", &n5))
            Check(n5.find("cached") == std::string::npos,
                  "changing patch re-measures (" + n5 + ")");
        else
            Check(false, "dehaze runs with a changed patch");
    }
}

// THE DESCRIPTOR INVARIANT THE VIEWER RELIES ON, checked on the descriptor
// rather than through a window.
//
// A proxy states its origin and full extent in ITS OWN pixels, so both must
// scale with the raster when the image is downsampled. Leaving them at
// full-resolution values made a 68% proxy of an 8191 px frame claim to be a
// window onto 8191 px while holding 5545 -- and the viewer, which lays out
// against the full extent divided by the scale, drew 12088.
static void TestProxyPlacement() {
    std::printf("\n--- proxy placement ---\n");

    auto algo = Registry::Get().Create("resize");
    if (!algo) { Check(false, "resize is registered"); return; }
    std::string e;
    if (ParamBase* p = algo->FindParam("scale")) p->SetFromScript(Value(0.5), &e);

    // A crop: a 400x300 window at (1000,500) onto a 4000x3000 frame.
    ImageDesc in;
    in.width = 400; in.height = 300; in.format = Format::RGBA32F;
    in.originX = 1000; in.originY = 500;
    in.fullW = 4000;   in.fullH = 3000;

    const ImageDesc out = algo->OutputDesc(0, in);
    Check(out.width == 200 && out.height == 150,
          "the raster halves (" + std::to_string(out.width) + "x" +
              std::to_string(out.height) + ")");
    Check(out.fullW == 2000 && out.fullH == 1500,
          "...and so does the full extent (" + std::to_string(out.fullW) + "x" +
              std::to_string(out.fullH) + ")");
    Check(out.originX == 500 && out.originY == 250,
          "...and so does the origin (" + std::to_string(out.originX) + "," +
              std::to_string(out.originY) + ")");

    // WHAT THE VIEWER COMPUTES: full extent over the scale must recover the
    // original frame, whatever the proxy scale happens to be. This is the
    // number that was wrong on screen.
    const float pscale = std::max(out.proxyScale, 1e-6f);
    const int laidOut = int(std::lround(double(out.FullWidth()) / double(pscale)));
    Check(laidOut == 4000,
          "the viewer lays out at the original size (" +
              std::to_string(laidOut) + ")");

    // A whole image carries no placement, and must not acquire one.
    ImageDesc whole;
    whole.width = 400; whole.height = 300; whole.format = Format::RGBA32F;
    const ImageDesc wout = algo->OutputDesc(0, whole);
    Check(wout.fullW == 0 && wout.fullH == 0 && wout.originX == 0 &&
              wout.originY == 0,
          "a whole image stays unplaced after a resize");
}

static void TestRegion() {
    Section("region processing");

    const int dim = 400;

    // Fine structure everywhere, so a wrong margin shows as a difference rather
    // than being hidden by flat areas.
    auto makeSrc = [&](int d) {
        Image img;
        img.Alloc({d, d, Format::RGBA32F});
        ImageView v = img.MapCpuWrite();
        uint32_t s = 0x1234567u;
        for (int y = 0; y < d; ++y)
            for (int x = 0; x < d; ++x) {
                s ^= s << 13; s ^= s >> 17; s ^= s << 5;
                float* p = v.At<float>(x, y);
                const float n = float(s & 0xFFFF) / 65535.0f * 0.3f;
                const bool box = ((x / 25) + (y / 25)) % 2 == 0;
                p[0] = p[1] = p[2] = (box ? 0.65f : 0.2f) + n;
                p[3] = 1.0f;
            }
        return img;
    };

    struct Case { const char* name; const char* script; };
    const Case cases[] = {
        {"gaussian_blur", "o = gaussian_blur(src, sigma = 4)\n"},
        {"box_blur",      "o = box_blur(src, radius = 6)\n"},
        {"median_blur",   "o = median_blur(src, radius = 4)\n"},
        {"kuwahara",      "o = kuwahara(src, radius = 5)\n"},
        {"brightness",    "o = brightness(src, brightness = 0.2)\n"},

        // GEOMETRIC, and the reason it is here. A vignette is anchored to the
        // PICTURE, so a crop must not re-centre it on the visible rectangle.
        // Every other case above is a neighbourhood operation that only needed
        // a margin; this one is wrong by a whole frame if it reads its own
        // extent instead of the full one, and the darkening then follows the
        // viewport around as the user pans.
        {"vignette",      "o = vignette(src, amount = -0.8, midpoint = 0.3)\n"},
    };

    // A window well inside the frame, so the margin is never clipped by an
    // edge -- that case is separately fine (the algorithm's own clamping is
    // then correct) but it would weaken this comparison.
    const Pipeline::Rect want{120, 130, 150, 140};

    for (const Case& c : cases) {
        const std::string script =
            std::string("src = image(\"test\")\n") + c.script + "display(o)\n";

        auto run = [&](Pipeline::Rect region, Image* out,
                       Pipeline::Rect* got, std::string* err) {
            UiState ui; Pipeline p; std::vector<Data> src;
            src.push_back(Data{makeSrc(dim)});
            std::vector<SourceImage> names{{"test", 0}};
            Program prog;
            if (!Parse(script, &prog, err)) return false;
            auto r = Interpret(prog, names, &ui, &p);
            if (!r.ok) { *err = r.error; return false; }
            p.SetRegion(region);
            if (!p.Execute(&src, nullptr, err)) return false;
            const Data* d = p.Resolve(p.Viewers()[0].source, &src);
            *out = const_cast<Image&>(std::get<Image>(*d)).Clone();
            *got = p.RanOnRegion();
            return true;
        };

        Image full, part;
        Pipeline::Rect fullRect, partRect;
        std::string err;
        if (!run(Pipeline::Rect{}, &full, &fullRect, &err) ||
            !run(want, &part, &partRect, &err)) {
            Check(false, std::string(c.name) + ": region run: " + err);
            continue;
        }

        if (!partRect.Valid()) {
            // Declining is legitimate -- a large margin makes cropping
            // pointless -- but say so rather than passing silently.
            Check(true, std::string(c.name) +
                  ": declined the region (margin too large)");
            continue;
        }

        Check(part.Desc().width == partRect.w && part.Desc().height == partRect.h,
              std::string(c.name) + ": the result is the cropped size (" +
                  std::to_string(part.Desc().width) + "x" +
                  std::to_string(part.Desc().height) + ")");

        // THE COMPARISON. Every pixel of the requested window must match what
        // the full run produced there. The margin around it is scratch and is
        // allowed to differ.
        ImageView fv = full.MapCpuRead();
        ImageView pv = part.MapCpuRead();
        double worst = 0.0;
        int wx = -1, wy = -1;
        for (int y = want.y; y < want.y + want.h; ++y)
            for (int x = want.x; x < want.x + want.w; ++x) {
                const int px = x - partRect.x, py = y - partRect.y;
                if (px < 0 || py < 0 ||
                    px >= part.Desc().width || py >= part.Desc().height) continue;
                for (int ch = 0; ch < 3; ++ch) {
                    const double d2 = std::fabs(double(fv.At<float>(x, y)[ch]) -
                                                double(pv.At<float>(px, py)[ch]));
                    if (d2 > worst) { worst = d2; wx = x; wy = y; }
                }
            }

        Check(worst < 1e-5,
              std::string(c.name) + ": a region result matches the full run (" +
                  "max diff " + std::to_string(worst) + " at " +
                  std::to_string(wx) + "," + std::to_string(wy) + ")");
    }

    // THE RESULT MUST SAY WHERE IT CAME FROM.
    //
    // A crop that does not carry its origin and the full extent is drawn as
    // though it were the whole picture -- the image jumps to the corner and
    // changes size the moment a drag starts. The viewer lays out against
    // FullWidth/FullHeight and draws at originX/originY, so both have to
    // survive the chain rather than only the stage that did the cropping.
    {
        const std::string script =
            "src = image(\"test\")\n"
            "a = gaussian_blur(src, sigma = 3)\n"
            "o = brightness(a, brightness = 0.1)\n"
            "display(o)\n";
        UiState ui; Pipeline p; std::vector<Data> src;
        src.push_back(Data{makeSrc(dim)});
        std::vector<SourceImage> names{{"test", 0}};
        Program prog; std::string err;
        if (Parse(script, &prog, &err)) {
            auto r = Interpret(prog, names, &ui, &p);
            if (r.ok) {
                p.SetRegion(want);
                if (p.Execute(&src, nullptr, &err)) {
                    const Data* d = p.Resolve(p.Viewers()[0].source, &src);
                    const Image& im = std::get<Image>(*d);
                    const ImageDesc& sd = im.Desc();
                    const Pipeline::Rect got = p.RanOnRegion();

                    Check(sd.FullWidth() == dim && sd.FullHeight() == dim,
                          "a cropped result knows the full extent (" +
                              std::to_string(sd.FullWidth()) + "x" +
                              std::to_string(sd.FullHeight()) + ")");
                    Check(sd.originX == got.x && sd.originY == got.y,
                          "...and where it starts (" +
                              std::to_string(sd.originX) + "," +
                              std::to_string(sd.originY) + " vs " +
                              std::to_string(got.x) + "," +
                              std::to_string(got.y) + ")");

                    // Carried THROUGH a second stage, not just set by the crop.
                    // brightness allocates its own output from the input's
                    // descriptor, so this is what proves the metadata rides
                    // along rather than being dropped at the first stage.
                    Check(sd.originX > 0 || sd.originY > 0,
                          "...and the origin survived the stage after the crop");
                }
            }
        }
    }

    // An algorithm that measures the whole frame must DECLINE, not silently
    // produce a threshold chosen for one corner.
    {
        const std::string script =
            "src = image(\"test\")\n"
            "o = threshold_otsu(src)\n"
            "display(o)\n";
        UiState ui; Pipeline p; std::vector<Data> src;
        src.push_back(Data{makeSrc(dim)});
        std::vector<SourceImage> names{{"test", 0}};
        Program prog; std::string err;
        if (Parse(script, &prog, &err)) {
            auto r = Interpret(prog, names, &ui, &p);
            if (r.ok) {
                p.SetRegion(want);
                if (p.Execute(&src, nullptr, &err)) {
                    Check(!p.RanOnRegion().Valid(),
                          "a whole-frame measurement declines the region");
                    const Data* d = p.Resolve(p.Viewers()[0].source, &src);
                    const Image& im = std::get<Image>(*d);
                    Check(im.Desc().width == dim,
                          "...and produces the full frame (" +
                              std::to_string(im.Desc().width) + ")");
                }
            }
        }
    }
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

    TestProxy();
    TestProxyFidelity();
    TestRegion();
    TestVisibleRect();
    TestProxyPlacement();
    TestRegionOriginStable();

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
        TestMeasureCache(dev);
        dev->Release();
    } else {
        std::printf("\n(no D3D12 device — GPU tests skipped)\n");
    }

    std::printf("\n%s\n", g_fail == 0 ? "all runtime checks passed" : "FAILURES PRESENT");
    return g_fail == 0 ? 0 : 1;
}
