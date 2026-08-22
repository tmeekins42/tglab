// Behavioural checks for the demosaic family.
//
// The fixture is built by *mosaicing* a known RGB image -- throwing away two
// channels per pixel exactly as a sensor does -- so the original is available
// as ground truth. That is the only honest way to test an interpolator: "it
// produced an image" says nothing about whether the image is right.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <d3d12.h>

#include "../src/core/algorithm.h"
#include "../src/core/pipeline.h"
#include "../src/gpu/compute.h"
#include "../src/script/value.h"

using namespace tglab;

static int g_fail = 0;
static void Check(bool cond, const std::string& what) {
    std::printf("%s  %s\n", cond ? "[ ok ]" : "[FAIL]", what.c_str());
    if (!cond) ++g_fail;
}

namespace {

constexpr int kW = 64;
constexpr int kH = 64;

// The scene, before the sensor sees it. Smooth gradients rather than hard
// edges: every demosaic reconstructs a smooth region almost exactly, so this
// isolates "is the interpolation correct" from "how does it handle edges",
// which is a separate question.
void TruthAt(int x, int y, float* rgb) {
    rgb[0] = 0.2f + 0.6f * float(x) / float(kW - 1);
    rgb[1] = 0.3f + 0.4f * float(y) / float(kH - 1);
    rgb[2] = 0.5f + 0.3f * float(x + y) / float(kW + kH - 2);
}

// What the sensor records: one channel per pixel, chosen by the CFA.
Image MakeMosaic(CfaPattern cfa) {
    Image img;
    ImageDesc d{kW, kH, Format::R32F};
    d.cfa        = cfa;
    d.blackLevel = 0.0f;
    d.whiteLevel = 1.0f;
    img.Alloc(d);

    ImageView v = img.MapCpuWrite();
    for (int y = 0; y < kH; ++y)
        for (int x = 0; x < kW; ++x) {
            float rgb[3];
            TruthAt(x, y, rgb);
            *v.At<float>(x, y) = rgb[CfaColorAt(cfa, x, y)];
        }
    return img;
}

// Runs one demosaic, optionally forced to a specific backend. The GPU variant
// is what checks that PrepareGpu() delivered the CFA and sensor levels.
bool RunDemosaicMode(const std::string& name, Image&& input, Image* out,
                     std::string* err, ComputeContext* gpu, ExecMode mode) {
    auto algo = Registry::Get().Create(name);
    if (!algo) { *err = "no such algorithm: " + name; return false; }

    Pipeline pipe;
    std::vector<Data> sources;
    sources.push_back(Data{std::move(input)});
    pipe.AddStage(std::move(algo), name, {{-1, 0}}, 1, 1);

    if (!pipe.Execute(&sources, nullptr, err, gpu, mode)) return false;

    const Data* d = pipe.Resolve({0, 0}, &sources);
    if (!d || !std::holds_alternative<Image>(*d)) { *err = "no output"; return false; }
    *out = const_cast<Image&>(std::get<Image>(*d)).Clone();
    return true;
}

bool RunDemosaic(const std::string& name, Image&& input,
                 const std::vector<std::pair<std::string, double>>& params,
                 Image* out, std::string* err) {
    auto algo = Registry::Get().Create(name);
    if (!algo) { *err = "no such algorithm: " + name; return false; }

    for (const auto& [key, value] : params) {
        bool found = false;
        for (ParamBase* p : algo->Params())
            if (key == p->Name()) { found = true; if (!p->SetFromScript(Value(value), err)) return false; }
        if (!found) { *err = name + " has no parameter '" + key + "'"; return false; }
    }

    Pipeline pipe;
    std::vector<Data> sources;
    sources.push_back(Data{std::move(input)});
    pipe.AddStage(std::move(algo), name, {{-1, 0}}, 1, 1);

    if (!pipe.Execute(&sources, nullptr, err)) return false;

    const Data* d = pipe.Resolve({0, 0}, &sources);
    if (!d || !std::holds_alternative<Image>(*d)) { *err = "no output"; return false; }
    *out = const_cast<Image&>(std::get<Image>(*d)).Clone();
    return true;
}

// Mean absolute error against the truth, over the interior. The border is
// excluded because every method clamps there and the error says more about the
// edge policy than the interpolation.
double MeanError(Image& img, int channel) {
    ImageView v = img.MapCpuRead();
    double sum = 0.0;
    int    n   = 0;
    for (int y = 2; y < kH - 2; ++y)
        for (int x = 2; x < kW - 2; ++x) {
            float truth[3];
            TruthAt(x, y, truth);
            const float got = HalfToFloat(v.At<uint16_t>(x, y)[channel]);
            sum += std::abs(double(got) - double(truth[channel]));
            ++n;
        }
    return n ? sum / n : 0.0;
}

} // namespace

int main() {
    std::string err;

    // --- the fixture is a real mosaic ---------------------------------------
    {
        Image m = MakeMosaic(CfaPattern::RGGB);
        Check(m.Desc().IsMosaic(), "the fixture reports itself as a mosaic");
        Check(m.Desc().format == Format::R32F, "a mosaic is single-channel");

        ImageView v = m.MapCpuRead();
        float truth[3];
        TruthAt(0, 0, truth);
        // (0,0) in RGGB is a red site, so the stored sample must be the red
        // component and nothing else.
        Check(std::abs(*v.At<float>(0, 0) - truth[0]) < 1e-5f,
              "an RGGB red site stores the red component");
        TruthAt(1, 1, truth);
        Check(std::abs(*v.At<float>(1, 1) - truth[2]) < 1e-5f,
              "an RGGB blue site stores the blue component");
    }

    // --- passthrough does not interpolate -----------------------------------
    {
        Image out;
        if (RunDemosaic("demosaic_passthrough", MakeMosaic(CfaPattern::RGGB),
                        {{"colour", 1}}, &out, &err)) {
            ImageView v = out.MapCpuRead();
            // At a red site in colour mode, green and blue must be zero: the
            // sensor never measured them, and passthrough must not invent them.
            const uint16_t* p = v.At<uint16_t>(2, 2);   // (even,even) = red in RGGB
            Check(HalfToFloat(p[1]) == 0.0f && HalfToFloat(p[2]) == 0.0f,
                  "passthrough leaves unmeasured channels empty");
            Check(HalfToFloat(p[0]) > 0.0f, "...and keeps the measured one");
        } else {
            Check(false, "passthrough runs: " + err);
        }
    }

    // --- bilinear reconstructs the scene ------------------------------------
    //
    // The check that matters. On a smooth gradient the interpolation should be
    // near-exact, so a large error here means the CFA phase is wrong -- the
    // classic failure, where red and blue end up swapped and the image still
    // "looks like an image".
    {
        for (CfaPattern cfa : {CfaPattern::RGGB, CfaPattern::BGGR,
                               CfaPattern::GRBG, CfaPattern::GBRG}) {
            Image out;
            if (!RunDemosaic("demosaic_bilinear", MakeMosaic(cfa), {}, &out, &err)) {
                Check(false, std::string(CfaPatternName(cfa)) + " runs: " + err);
                continue;
            }
            const double er = MeanError(out, 0);
            const double eg = MeanError(out, 1);
            const double eb = MeanError(out, 2);
            const double worst = std::max(er, std::max(eg, eb));

            // 0.01 on a 0..1 scale. A phase error would put this near 0.3,
            // since red and blue differ by roughly that much in the fixture.
            Check(worst < 0.01,
                  std::string(CfaPatternName(cfa)) + " reconstructs the scene (worst channel " +
                      std::to_string(worst) + ")");
        }
    }

    // --- the output is no longer a mosaic -----------------------------------
    //
    // If the descriptor still claimed a CFA, a second demosaic in a chain would
    // treat finished RGB as sensor data and mangle it.
    {
        Image out;
        if (RunDemosaic("demosaic_bilinear", MakeMosaic(CfaPattern::RGGB), {}, &out, &err)) {
            Check(!out.Desc().IsMosaic(), "a demosaiced image is no longer a mosaic");
            Check(out.Desc().format == Format::RGBA16F, "and is RGBA16F");
        } else {
            Check(false, "bilinear runs: " + err);
        }
    }

    // --- an ordinary image passes through unharmed --------------------------
    //
    // A script may apply a demosaic unconditionally; doing so to a PNG must be
    // a no-op rather than garbage.
    {
        Image plain;
        plain.Alloc({kW, kH, Format::RGBA8});
        ImageView v = plain.MapCpuWrite();
        for (int y = 0; y < kH; ++y)
            for (int x = 0; x < kW; ++x) {
                uint8_t* p = v.At<uint8_t>(x, y);
                p[0] = 200; p[1] = 100; p[2] = 50; p[3] = 255;
            }

        Image out;
        if (RunDemosaic("demosaic_bilinear", std::move(plain), {}, &out, &err)) {
            ImageView o = out.MapCpuRead();
            const uint16_t* p = o.At<uint16_t>(10, 10);
            Check(std::abs(HalfToFloat(p[0]) - 200.0f / 255.0f) < 0.01f &&
                  std::abs(HalfToFloat(p[1]) - 100.0f / 255.0f) < 0.01f,
                  "a non-mosaic image passes through unchanged");
        } else {
            Check(false, "bilinear on a plain image: " + err);
        }
    }

    // --- black and white levels are applied ---------------------------------
    //
    // Raw sensors do not start at zero. Ignoring the black level lifts the
    // whole image; ignoring the white level scales it wrongly.
    {
        Image m;
        ImageDesc d{kW, kH, Format::R32F};
        d.cfa        = CfaPattern::RGGB;
        d.blackLevel = 0.25f;
        d.whiteLevel = 0.75f;
        m.Alloc(d);
        ImageView v = m.MapCpuWrite();
        // Every sample at the black level: the result must be black, not grey.
        for (int y = 0; y < kH; ++y)
            for (int x = 0; x < kW; ++x) *v.At<float>(x, y) = 0.25f;

        Image out;
        if (RunDemosaic("demosaic_bilinear", std::move(m), {}, &out, &err)) {
            ImageView o = out.MapCpuRead();
            const uint16_t* p = o.At<uint16_t>(10, 10);
            Check(HalfToFloat(p[0]) < 0.01f && HalfToFloat(p[1]) < 0.01f,
                  "a sample at the black level becomes black");
        } else {
            Check(false, "black level: " + err);
        }
    }

    // --- white balance and the colour matrix are applied --------------------
    //
    // A sensor's green photosites are roughly twice as sensitive as its red and
    // blue ones, so raw data without the camera's own multipliers is heavily
    // green. Measured on real files, a Canon 5D3 needs R x1.45 / B x2.37 -- so
    // omitting them is not a subtle error, it is a visibly green photograph.
    {
        // A flat grey scene: after white balance the three channels must come
        // back equal, which is exactly what "balanced" means.
        ImageDesc d{kW, kH, Format::R32F};
        d.cfa        = CfaPattern::RGGB;
        d.blackLevel = 0.0f;
        d.whiteLevel = 1.0f;
        // A sensor that measures green twice as strongly as red and blue.
        d.camMul[0] = 2.0f; d.camMul[1] = 1.0f; d.camMul[2] = 2.0f;

        Image m;
        m.Alloc(d);
        ImageView v = m.MapCpuWrite();
        for (int y = 0; y < kH; ++y)
            for (int x = 0; x < kW; ++x) {
                // Red and blue sites read half what green does, as a real
                // sensor would on a neutral subject.
                const int c = CfaColorAt(CfaPattern::RGGB, x, y);
                *v.At<float>(x, y) = (c == 1) ? 0.5f : 0.25f;
            }

        Image out;
        if (RunDemosaic("demosaic_bilinear", std::move(m), {}, &out, &err)) {
            ImageView o = out.MapCpuRead();
            const uint16_t* p = o.At<uint16_t>(20, 20);
            const float r = HalfToFloat(p[0]), g = HalfToFloat(p[1]), b = HalfToFloat(p[2]);
            Check(std::abs(r - g) < 0.02f && std::abs(b - g) < 0.02f,
                  "white balance neutralises a grey scene (R=" + std::to_string(r) +
                      " G=" + std::to_string(g) + " B=" + std::to_string(b) + ")");
        } else {
            Check(false, "white balance: " + err);
        }
    }
    {
        // The colour matrix must actually be applied, not silently ignored.
        ImageDesc d{kW, kH, Format::R32F};
        d.cfa = CfaPattern::RGGB;
        d.whiteLevel = 1.0f;
        // Swap red and blue: an obviously wrong matrix, chosen so that failing
        // to apply it is unmistakable rather than a subtle hue shift.
        const float swap[9] = {0, 0, 1,  0, 1, 0,  1, 0, 0};
        for (int i = 0; i < 9; ++i) d.rgbCam[i] = swap[i];

        Image m;
        m.Alloc(d);
        ImageView v = m.MapCpuWrite();
        for (int y = 0; y < kH; ++y)
            for (int x = 0; x < kW; ++x) {
                const int c = CfaColorAt(CfaPattern::RGGB, x, y);
                *v.At<float>(x, y) = (c == 0) ? 0.8f : 0.1f;   // a red scene
            }

        Image out;
        if (RunDemosaic("demosaic_bilinear", std::move(m), {}, &out, &err)) {
            ImageView o = out.MapCpuRead();
            const uint16_t* p = o.At<uint16_t>(20, 20);
            Check(HalfToFloat(p[2]) > HalfToFloat(p[0]),
                  "the colour matrix is applied (a red scene came out blue)");
        } else {
            Check(false, "colour matrix: " + err);
        }
    }

    // --- the GPU kernels agree with the CPU ---------------------------------
    //
    // A demosaic reads the CFA pattern and sensor levels from the *image*, not
    // from a parameter, and the GPU path never calls RunCPU(). If PrepareGpu()
    // does not deliver them the shader runs on stale values -- and the result
    // is a plausible-looking image with the wrong colours, which no "did it
    // run" check would catch.
    {
        ID3D12Device* dev = nullptr;
        if (SUCCEEDED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&dev)))) {
            ComputeContext gpu;
            if (gpu.Init(dev)) {
                for (const char* name : {"demosaic_passthrough", "demosaic_bilinear"}) {
                    for (CfaPattern cfa : {CfaPattern::RGGB, CfaPattern::BGGR,
                                           CfaPattern::GRBG, CfaPattern::GBRG}) {
                        Image cpuOut, gpuOut;
                        std::string e1, e2;
                        const bool okCpu = RunDemosaicMode(name, MakeMosaic(cfa), &cpuOut,
                                                           &e1, nullptr, ExecMode::ForceCPU);
                        const bool okGpu = RunDemosaicMode(name, MakeMosaic(cfa), &gpuOut,
                                                           &e2, &gpu, ExecMode::ForceGPU);
                        if (!okCpu || !okGpu) {
                            Check(false, std::string(name) + " ran both ways: " +
                                             (okCpu ? e2 : e1));
                            continue;
                        }

                        // Compared on the reconstructed values, not raw bits:
                        // half rounding differs slightly between the two paths.
                        ImageView a = cpuOut.MapCpuRead();
                        ImageView b = gpuOut.MapCpuRead();
                        double worst = 0.0;
                        for (int y = 1; y < kH - 1; ++y)
                            for (int x = 1; x < kW - 1; ++x)
                                for (int c = 0; c < 3; ++c)
                                    worst = std::max(worst,
                                        std::abs(double(HalfToFloat(a.At<uint16_t>(x, y)[c])) -
                                                 double(HalfToFloat(b.At<uint16_t>(x, y)[c]))));

                        // A stale CFA would put this near 0.3 -- the distance
                        // between the fixture's red and blue.
                        Check(worst < 0.005,
                              std::string(name) + " " + CfaPatternName(cfa) +
                                  " GPU matches CPU (worst " + std::to_string(worst) + ")");
                    }
                }
                gpu.Shutdown();
            }
            dev->Release();
        } else {
            std::printf("(no D3D12 device - GPU agreement skipped)\n");
        }
    }

    std::printf("\n%s\n", g_fail == 0 ? "all demosaic checks passed" : "FAILURES PRESENT");
    return g_fail == 0 ? 0 : 1;
}
