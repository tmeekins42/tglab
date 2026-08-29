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
#include "../src/algo_util/white_balance.h"
#include "../src/core/image_group.h"
#include "../src/core/image_io.h"
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

// Stage to insert before the merge, from TGLAB_PRE.
static std::string PreAlgo() {
    char buf[64] = {};
    GetEnvironmentVariableA("TGLAB_PRE", buf, sizeof buf);
    return buf[0] ? std::string(buf) : std::string("align");
}

// A second group-level stage, from TGLAB_PRE2.
static std::string Pre2Algo() {
    char buf[64] = {};
    GetEnvironmentVariableA("TGLAB_PRE2", buf, sizeof buf);
    return buf;
}

// A third group-level stage, from TGLAB_PRE3.
static std::string Pre3Algo() {
    char buf[64] = {};
    GetEnvironmentVariableA("TGLAB_PRE3", buf, sizeof buf);
    return buf;
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
        // White balance, because a merged bracket reaching develop with kelvin
        // at 0 is a real symptom -- the control then opens at "no idea" and a
        // cast cannot be corrected from either end.
        float kk = 0.0f, tt = 0.0f;
        AsShotWhiteBalance(d, &kk, &tt);
        std::printf("      daylightWb=%d  kelvin=%.0f tint=%.2f\n",
                    int(d.hasDaylightWb), double(kk), double(tt));
        AppendToGroup(&group, std::move(img), "frame");
    }
    std::printf("loaded %d frames in %.1fs, shape %s\n\n",
                argc - 1, Now() - t0, ShapeOf(group).ToString().c_str());

    // TGLAB_MERGE picks the reduction, so the same harness exercises either.
    char algo[64] = "merge_mean";
    GetEnvironmentVariableA("TGLAB_MERGE", algo, sizeof algo);
    const std::string srcStr =
        std::string("frames = image(\"group\")\n") +
        // TGLAB_PRE inserts a stage BEFORE the merge -- align, in practice --
        // so a solve can be measured against the same bracket in one run.
        (GetEnvironmentVariableA("TGLAB_PRE", nullptr, 0) > 0
             ? std::string("frames = ") + PreAlgo() + "(frames" +
               (GetEnvironmentVariableA("TGLAB_NORM", nullptr, 0) > 0
                    ? ", normalize = 1" : "") + ")\n"
             : std::string()) +
        // TGLAB_PRE2 chains a SECOND group-level stage, so detect-then-match
        // can be exercised on real raws in one run.
        (GetEnvironmentVariableA("TGLAB_PRE2", nullptr, 0) > 0
             ? std::string("frames = ") + Pre2Algo() + "(frames)\n"
             : std::string()) +
        // A third, so detect -> match -> align can run end to end.
        (GetEnvironmentVariableA("TGLAB_PRE3", nullptr, 0) > 0
             ? std::string("frames = ") + Pre3Algo() + "(frames)\n"
             : std::string()) +
        "merged = " + algo + "(frames)\n" +
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

    // TGLAB_CPU=1 forces the CPU path, for comparing it against the GPU one and
    // for profiling CPU algorithms without the GPU quietly taking the work.
    const bool forceCpu = GetEnvironmentVariableA("TGLAB_CPU", nullptr, 0) > 0;

    const double t1 = Now();
    std::string xerr;
    const bool ok = pipe.Execute(&sources, nullptr, &xerr,
                                 (gpu.Ready() && !forceCpu) ? &gpu : nullptr,
                                 forceCpu ? ExecMode::ForceCPU : ExecMode::Auto);
    // TGLAB_SAVE=<path> writes the result, exercising the save path on real
    // pixels rather than only on test fixtures.
    {
        char sbuf[512] = {};
        if (GetEnvironmentVariableA("TGLAB_SAVE", sbuf, sizeof sbuf) > 0 && ok) {
            const Data* d = pipe.Resolve(pipe.Viewers().back().source, &sources);
            if (const auto* im = d ? std::get_if<Image>(d) : nullptr) {
                std::string serr;
                if (SaveImage(sbuf, const_cast<Image&>(*im), &serr))
                    std::printf("saved: %s\n", sbuf);
                else
                    std::printf("save failed: %s\n", serr.c_str());
            }
        }
    }

    if (adapter) {
        DXGI_QUERY_VIDEO_MEMORY_INFO info{};
        if (SUCCEEDED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info)))
            std::printf("vram at end: %.0f MB\n", double(info.CurrentUsage) / (1024.0 * 1024.0));
    }
    std::printf("stages run: %d cpu, %d gpu\n", pipe.CpuStageCount(), pipe.GpuStageCount());

    // Where the time went. GpuMs is submit-and-wait, the only point at which
    // the device does anything: a dispatch merely records into a batch. The
    // remainder is CPU-side, and includes the recording itself.
    if (gpu.Ready()) {
        const double total = (Now() - t1) * 1000.0;   // Now() is in seconds
        const double g     = std::min(gpu.GpuMs(), total);
        std::printf("time: %.0f ms total, %.0f ms cpu, %.0f ms gpu (%.0f%% gpu)\n",
                    total, total - g, g, total > 0.0 ? 100.0 * g / total : 0.0);
    }

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
                    // Percentiles say what a min/max cannot: WHERE the picture
                    // sits, and how many stops separate its shadows from its
                    // highlights. That is the number a tone mapper has to fit.
                    {
                        std::vector<float> s;
                        s.reserve(200000);
                        const int st = std::max(1, im->Desc().width / 700);
                        PixelBuffer sb;
                        sb.Unpack(v);
                        for (int y = 0; y < sb.Height(); y += st)
                            for (int x = 0; x < sb.Width(); x += st) {
                                const float* q = sb.At(x, y);
                                const float l = 0.2126f * q[0] + 0.7152f * q[1] + 0.0722f * q[2];
                                if (l > 0.0f) s.push_back(l);
                            }
                        if (s.size() > 100) {
                            std::sort(s.begin(), s.end());
                            auto P = [&](double f) {
                                return s[std::min(s.size() - 1,
                                                  size_t(f * double(s.size() - 1)))];
                            };
                            std::printf("percentiles: p1 %.4g  p10 %.4g  p50 %.4g  "
                                        "p90 %.4g  p99 %.4g  p99.9 %.4g\n",
                                        double(P(0.01)), double(P(0.10)), double(P(0.50)),
                                        double(P(0.90)), double(P(0.99)), double(P(0.999)));
                            std::printf("dynamic range p1..p99: %.1f stops\n",
                                        std::log2(double(P(0.99)) / std::max(double(P(0.01)), 1e-9)));
                        }
                    }
                    // Gradient energy: the sharpness measure. Higher is
                    // sharper, and comparing two runs of the same bracket is
                    // what says whether an alignment choice actually helped
                    // rather than merely changed the numbers.
                    {
                        PixelBuffer sb;
                        sb.Unpack(v);
                        double e = 0.0;
                        for (int y = 8; y < sb.Height() - 8; y += 3)
                            for (int x = 8; x < sb.Width() - 8; x += 3) {
                                const float gx = sb.At(x + 1, y)[1] - sb.At(x - 1, y)[1];
                                const float gy = sb.At(x, y + 1)[1] - sb.At(x, y - 1)[1];
                                e += double(gx) * double(gx) + double(gy) * double(gy);
                            }
                        std::printf("sharpness: %.4g\n", e);

                        // Scale-invariant local contrast: |gradient| relative
                        // to the local value, averaged.
                        //
                        // Raw gradient energy is NOT comparable between two
                        // renderings at different brightness -- scaling an
                        // image by 2 quadruples it while changing no contrast
                        // at all. That matters precisely here, because a tone
                        // mapper's whole job is to change brightness, so the
                        // plain figure would credit any operator that merely
                        // brightened the frame. This ratio is what "local
                        // contrast" actually means and it survives a rescale.
                        double rel = 0.0;
                        size_t relN = 0;
                        for (int y = 8; y < sb.Height() - 8; y += 3)
                            for (int x = 8; x < sb.Width() - 8; x += 3) {
                                const float c = sb.At(x, y)[1];
                                if (c < 1e-4f) continue;   // noise, not contrast
                                const float gx = sb.At(x + 1, y)[1] - sb.At(x - 1, y)[1];
                                const float gy = sb.At(x, y + 1)[1] - sb.At(x, y - 1)[1];
                                rel += std::sqrt(double(gx) * gx + double(gy) * gy) / double(c);
                                ++relN;
                            }
                        if (relN) std::printf("local contrast: %.4f\n", rel / double(relN));
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
