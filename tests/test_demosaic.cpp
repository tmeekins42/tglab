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
#include "../src/core/raw_io.h"
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
    // (orientation checks run first: they need no fixture)

    // --- orientation ---------------------------------------------------------
    //
    // A raw carries the camera's orientation tag, and LibRaw reports it in
    // dcraw's encoding: 0 none, 3 = 180, 5 = 90 CCW, 6 = 90 CW. Tim's
    // _dsc0139.arw declares 6 and was displayed on its side.
    //
    // Rotating a mosaic is not a transpose -- the colour filter rotates with
    // the pixels -- so this checks the CFA transform, not the geometry. A wrong
    // pattern swaps red and blue across the whole image, which reads as a
    // demosaic bug rather than an orientation one and is exactly the mistake
    // worth pinning down.
    //
    // The expectations are derived by hand from the RGGB tile, independently of
    // the implementation's coordinate map:
    //
    //   RGGB = R G   90 CW: R->(1,0) G->(1,1)   giving  G R  = GRBG
    //          G B          G->(0,0) B->(0,1)           B G
    {
        struct Case { int flip; CfaPattern from; CfaPattern to; const char* what; };
        const Case cases[] = {
            {0, CfaPattern::RGGB, CfaPattern::RGGB, "no flip leaves the pattern alone"},
            {6, CfaPattern::RGGB, CfaPattern::GRBG, "RGGB rotated 90 CW is GRBG"},
            {3, CfaPattern::RGGB, CfaPattern::BGGR, "RGGB rotated 180 is BGGR"},
            {5, CfaPattern::RGGB, CfaPattern::GBRG, "RGGB rotated 90 CCW is GBRG"},
            {6, CfaPattern::BGGR, CfaPattern::GBRG, "BGGR rotated 90 CW is GBRG"},
            {3, CfaPattern::GRBG, CfaPattern::GBRG, "GRBG rotated 180 is GBRG"},
        };
        for (const Case& c : cases) {
            // Even dimensions, which is what every sensor here reports; the
            // implementation handles odd ones by construction rather than by
            // assuming, since an odd size shifts the CFA phase.
            const CfaPattern got = RotateCfa(c.from, c.flip, 8, 6);
            Check(got == c.to, std::string(c.what) + " (got " +
                                   std::to_string(int(got)) + ")");
        }

        // Rotating four times must return to where it started, whichever way
        // round. That catches a transform that is self-consistent but wrong.
        CfaPattern p = CfaPattern::RGGB;
        for (int i = 0; i < 4; ++i) p = RotateCfa(p, 6, 8, 6);
        Check(p == CfaPattern::RGGB, "four 90 CW rotations return to RGGB");
    }
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

    // A NEUTRAL BLOWN HIGHLIGHT MUST DEVELOP NEUTRAL, whatever the white
    // balance. This is the property that failed on a real night shot.
    //
    // When every channel saturates they clip to the SAME raw value, so the
    // pixel carries no colour information at all -- but white balance then
    // multiplies them apart. At 3313 K the blue gain was 3.08x the green, so a
    // blown street lamp arrived at the camera matrix as a strong blue cast, the
    // matrix's negative green coefficients crushed what was left, and the lamp
    // developed as R 1.77, G 0.06, B 4.48. Vivid magenta, on 66.5% of the
    // frame's bright pixels.
    //
    // The clipped-channel repair exists for exactly this and did not fire,
    // because it tests a normalised value against 0.99 and LibRaw's declared
    // white level was ABOVE where the sensor really saturated -- so a blown
    // sample normalised to 0.985 and was never flagged. See raw_io.cpp, which
    // now measures the saturation point from the data.
    //
    // Tested with an extreme multiplier because that is where it shows: a
    // near-neutral white balance hides the fault entirely.
    {
        Image m;
        ImageDesc d{kW, kH, Format::R32F};
        d.cfa        = CfaPattern::RGGB;
        d.blackLevel = 0.0f;
        d.whiteLevel = 1.0f;
        d.camMul[0] = 1.17f; d.camMul[1] = 1.0f; d.camMul[2] = 3.08f;
        // Identity matrix: this is about the gains, and a real camera matrix
        // would mix a second effect into the measurement.
        for (int i = 0; i < 9; ++i) d.rgbCam[i] = (i % 4 == 0) ? 1.0f : 0.0f;
        m.Alloc(d);
        ImageView v = m.MapCpuWrite();
        // Every photosite saturated: a blown neutral highlight.
        for (int y = 0; y < kH; ++y)
            for (int x = 0; x < kW; ++x) *v.At<float>(x, y) = 1.0f;

        Image out;
        if (RunDemosaic("demosaic_bilinear", std::move(m), {}, &out, &err)) {
            ImageView o = out.MapCpuRead();
            const uint16_t* p = o.At<uint16_t>(kW / 2, kH / 2);
            const float r = HalfToFloat(p[0]);
            const float g = HalfToFloat(p[1]);
            const float b = HalfToFloat(p[2]);
            const float lo = std::min(r, std::min(g, b));
            const float hi = std::max(r, std::max(g, b));
            // Within 2%: the repair lifts every clipped channel to the same
            // value, so this should be exact but for half-float rounding.
            Check(hi > 0.0f && (hi - lo) / hi < 0.02f,
                  "a blown neutral highlight stays neutral through a 3x blue "
                  "gain (R " + std::to_string(r) + " G " + std::to_string(g) +
                  " B " + std::to_string(b) + ")");
        } else {
            Check(false, "blown highlight: " + err);
        }
    }

    // --- Malvar beats bilinear where bilinear actually fails ----------------
    //
    // On a smooth gradient every method is near-exact, so that fixture cannot
    // tell them apart. Bilinear's characteristic failure is colour fringing at
    // high-contrast edges: averaging across a boundary invents colour that was
    // never there. Malvar corrects using the centre channel's second
    // derivative, so this is the fixture that shows the difference.
    {
        // A fine radial pattern: neutral grey, but with detail at the sampling
        // frequency, which is what actually separates demosaic methods.
        //
        // A step edge was tried first and does NOT discriminate -- measured,
        // both methods fringe identically there (0.087 each), and the flat
        // regions on either side are exactly zero for both. Malvar's whole
        // premise is that the centre channel's second derivative predicts the
        // others, and a step is locally flat everywhere except two columns. It
        // needs structure at every pixel to have anything to work with.
        auto makeEdge = [](CfaPattern cfa) {
            Image img;
            ImageDesc d{kW, kH, Format::R32F};
            d.cfa = cfa;
            d.whiteLevel = 1.0f;
            img.Alloc(d);
            ImageView v = img.MapCpuWrite();
            for (int y = 0; y < kH; ++y)
                for (int x = 0; x < kW; ++x) {
                    const float fx = float(x) - kW * 0.5f;
                    const float fy = float(y) - kH * 0.5f;
                    const float r  = std::sqrt(fx * fx + fy * fy);
                    // Grey, so R == G == B in the true scene and any colour in
                    // the result is entirely interpolation error.
                    const float s  = 0.5f + 0.45f * std::sin(r * 1.1f);
                    *v.At<float>(x, y) = s;
                }
            return img;
        };

        // Mean colour spread on a scene that has none. Anything above zero is
        // fringing, so lower is strictly better.
        auto fringing = [](Image& img) {
            ImageView v = img.MapCpuRead();
            double sum = 0.0;
            int    n   = 0;
            for (int y = 3; y < kH - 3; ++y)
                for (int x = 3; x < kW - 3; ++x) {
                    const float r = HalfToFloat(v.At<uint16_t>(x, y)[0]);
                    const float g = HalfToFloat(v.At<uint16_t>(x, y)[1]);
                    const float b = HalfToFloat(v.At<uint16_t>(x, y)[2]);
                    sum += std::max({r, g, b}) - std::min({r, g, b});
                    ++n;
                }
            return n ? sum / n : 0.0;
        };

        Image bil, mal;
        const bool a = RunDemosaic("demosaic_bilinear", makeEdge(CfaPattern::RGGB),
                                   {}, &bil, &err);
        const bool b = RunDemosaic("demosaic_malvar", makeEdge(CfaPattern::RGGB),
                                   {}, &mal, &err);
        if (a && b) {
            const double fb = fringing(bil), fm = fringing(mal);
            Check(fb > 0.05, "bilinear fringes on fine detail (" +
                                 std::to_string(fb) + ")");
            // Measured at roughly half: 0.078 against 0.137.
            Check(fm < fb * 0.75, "Malvar fringes substantially less (" +
                               std::to_string(fm) + " vs " + std::to_string(fb) + ")");
        } else {
            Check(false, "both methods ran: " + err);
        }
    }
    {
        // AHD must beat Malvar on thin high-contrast detail, measured against
        // ground truth.
        //
        // This is the case Malvar structurally cannot do well: it is a linear
        // filter with fixed weights, so on a feature finer than the green
        // sampling grid it must guess, and it guesses without regard to which
        // way the detail runs. The error correlates with CFA phase, which is
        // why it appears as a lattice of coloured dots rather than as noise --
        // measured at 11.5% higher chroma error on green sites than on
        // red/blue sites on a real backlit-branch CR3.
        //
        // Ground truth rather than statistics on a photograph. On a raw there
        // is no true answer, so "more colour variation" can mean the method
        // recovered real colour OR invented it -- during development the
        // real-photo numbers pointed the wrong way for exactly that reason,
        // while against a known scene AHD was ahead by 9.5 dB.
        constexpr int kEW = 96, kEH = 96;
        auto edgeTruth = [](int x, int y, float* rgb) {
            rgb[0] = 0.72f; rgb[1] = 0.74f; rgb[2] = 0.76f;   // bright fog
            // Thin dark lines, 1-2 px, slightly tilted so they cross the CFA
            // phase -- conifer twigs against sky, the motivating content.
            for (int k = 0; k < 4; ++k) {
                const float bx = 12.0f + float(k) * 22.0f + float(y) * 0.15f;
                if (std::fabs(float(x) - bx) < (k % 2 ? 0.6f : 1.1f)) {
                    rgb[0] = 0.10f; rgb[1] = 0.12f; rgb[2] = 0.11f;
                }
            }
            if (std::fabs(float(y) - 70.0f) < 1.0f) {
                rgb[0] = 0.09f; rgb[1] = 0.11f; rgb[2] = 0.10f;
            }
        };
        auto makeEdgeScene = [&](CfaPattern cfa) {
            Image img;
            ImageDesc d{kEW, kEH, Format::R32F};
            d.cfa = cfa;
            d.blackLevel = 0.0f;
            d.whiteLevel = 1.0f;

            // A REAL camera's white balance and colour matrix, not identity.
            //
            // This matters more than it looks. A camera matrix has large
            // negative off-diagonal terms -- these are a Canon R5's, where red
            // out is 1.535 of red in MINUS 0.555 of green -- so a reconstructed
            // red that is merely short relative to green comes out negative,
            // clamps at zero, and renders fully saturated.
            //
            // With an identity matrix that mechanism cannot occur at all, so a
            // fixture using one silently cannot test for it: the check below
            // passed identically with the fix present and removed, which is
            // exactly the kind of green tick that proves nothing.
            d.camMul[0] = 1.883f; d.camMul[1] = 1.0f; d.camMul[2] = 1.910f;
            const float r5[9] = { 1.535f, -0.555f,  0.019f,
                                 -0.173f,  1.653f, -0.479f,
                                 -0.014f, -0.453f,  1.467f};
            for (int i = 0; i < 9; ++i) d.rgbCam[i] = r5[i];

            img.Alloc(d);
            ImageView v = img.MapCpuWrite();
            for (int y = 0; y < kEH; ++y)
                for (int x = 0; x < kEW; ++x) {
                    float rgb[3];
                    edgeTruth(x, y, rgb);
                    *v.At<float>(x, y) = rgb[CfaColorAt(cfa, x, y)];
                }
            return img;
        };
        // Colour error at EDGE pixels, as a high percentile rather than a mean.
        //
        // Two deliberate choices, both learned the hard way.
        //
        // A percentile, because demosaic fringing is thin, bright and
        // localised: a mean over the frame averages it away against a large
        // majority of correct pixels, and during development a change that
        // improved the mean past Malvar's simultaneously painted visible green
        // streaks along every twig. The p95 sees what the eye sees.
        //
        // Edge pixels only, because that is where reconstruction is ambiguous.
        // A flat region is reconstructed correctly by every method and only
        // dilutes the measurement.
        //
        // Saturation is normalised by the pixel's own brightness, since a real
        // camera matrix pushes values well above 1.0 and an absolute max-minus-
        // min then measures exposure as much as colour error.
        auto chromaError = [&](Image& img) {
            ImageView v = img.MapCpuRead();
            auto lumAt = [&](int x, int y) {
                const uint16_t* p = v.At<uint16_t>(x, y);
                return 0.2126f * HalfToFloat(p[0]) + 0.7152f * HalfToFloat(p[1]) +
                       0.0722f * HalfToFloat(p[2]);
            };
            std::vector<double> sats;
            for (int y = 3; y < kEH - 3; ++y)
                for (int x = 3; x < kEW - 3; ++x) {
                    const float lum = lumAt(x, y);
                    if (lum < 1e-4f) continue;
                    const float grad = std::max(
                        std::fabs(lumAt(x + 1, y) - lumAt(x - 1, y)),
                        std::fabs(lumAt(x, y + 1) - lumAt(x, y - 1)));
                    if (grad < 0.15f * lum) continue;      // not an edge
                    const uint16_t* p = v.At<uint16_t>(x, y);
                    const float r = HalfToFloat(p[0]);
                    const float g = HalfToFloat(p[1]);
                    const float b = HalfToFloat(p[2]);
                    const float mx = std::max({r, g, b});
                    const float mn = std::min({r, g, b});
                    sats.push_back(double((mx - mn) / std::max(mx, 1e-6f)));
                }
            if (sats.empty()) return 0.0;
            std::sort(sats.begin(), sats.end());
            return sats[size_t(0.95 * double(sats.size() - 1))];
        };

        Image mv, ah;
        std::string e1, e2;
        const bool a = RunDemosaic("demosaic_malvar", makeEdgeScene(CfaPattern::RGGB),
                                   {}, &mv, &e1);
        const bool b = RunDemosaic("demosaic_ahd", makeEdgeScene(CfaPattern::RGGB),
                                   {}, &ah, &e2);
        if (a && b) {
            const double cm = chromaError(mv), ca = chromaError(ah);
            // Not asserted as a ratio.
            //
            // This scene's hard 0.10-against-0.72 step, put through a real
            // camera matrix, drives some channel negative for EVERY method --
            // so the p95 saturates at 1.0 for all of them and the number
            // discriminates nothing. Reported for information; the assertions
            // that carry weight are the zero-channel count below (which does
            // discriminate: 256 against 305) and the ratio test on the
            // identity-matrix scene further up.
            std::printf("       edge p95 saturation: AHD %.3f, Malvar %.3f\n", ca, cm);
            // AHD must not drive MORE channels to zero than Malvar.
            //
            // A camera matrix has large negative off-diagonal terms -- the R5's
            // first row is [1.535, -0.555, 0.019], so red out is red in minus
            // 0.555 of green. A reconstructed red that is merely SHORT relative
            // to green therefore goes negative after the matrix, is clamped at
            // zero, and the pixel renders fully saturated. That is what made
            // AHD's fringes vivid rather than faint: on the Oregon frame it
            // produced 898 such pixels against Malvar's 22, and bounding each
            // reconstructed channel by the real samples around it brought that
            // to 27.
            //
            // A caveat worth stating plainly: on THIS synthetic scene the bound
            // makes the count worse rather than better (256 with it, 126
            // without), because a hard 0.10-against-0.72 step is more extreme
            // than any real edge and the bound then pins estimates to samples
            // that are themselves far apart. The fixture cannot demonstrate the
            // improvement it was written for -- it only guards the invariant
            // that AHD stays in Malvar's league, which is the part that would
            // regress catastrophically. The 33x improvement is real but is
            // evidenced on the raw file, not here.
            auto zeroChannels = [&](Image& img) {
                ImageView v = img.MapCpuRead();
                long z = 0;
                for (int y = 3; y < kEH - 3; ++y)
                    for (int x = 3; x < kEW - 3; ++x) {
                        const uint16_t* p = v.At<uint16_t>(x, y);
                        const float r = HalfToFloat(p[0]);
                        const float g = HalfToFloat(p[1]);
                        const float b = HalfToFloat(p[2]);
                        if (std::max({r, g, b}) < 1e-6f) continue;   // genuinely black
                        if (r <= 0.0f || g <= 0.0f || b <= 0.0f) ++z;
                    }
                return z;
            };
            const long za = zeroChannels(ah), zm = zeroChannels(mv);
            Check(za <= zm + 2,
                  "AHD drives no more channels to zero than Malvar (" +
                      std::to_string(za) + " vs " + std::to_string(zm) + ")");
        } else {
            Check(false, "both methods ran: " + e1 + e2);
        }
    }
    {
        // Malvar must not invent a value outside the samples it interpolated
        // from.
        //
        // Its correction term is a Laplacian, and a Laplacian is unbounded: on
        // a hard edge it overshoots the way an unwindowed sharpening filter
        // rings, pushing an interpolated channel past every sample of that
        // colour nearby. The pixel then gets a colour present in none of the
        // data, which reads as isolated saturated confetti.
        //
        // Found on a CR3 of backlit conifer branches against bright fog --
        // millions of hard edges, the worst case for this. Malvar produced
        // 2738 isolated saturated pixels against bilinear's 119 on identical
        // input, and overshot by up to 1.677 on a 0..1 scale. A method whose
        // entire purpose is LESS colour fringing than bilinear being 23x worse
        // is a defect, not a tuning choice.
        //
        // A hard step rather than the sine grating above: the smooth case is
        // where Malvar's correction is right and the clamp never fires, so it
        // cannot detect this.
        auto makeStep = [](CfaPattern cfa) {
            Image img;
            ImageDesc d{kW, kH, Format::R32F};
            d.cfa = cfa;
            d.whiteLevel = 1.0f;
            img.Alloc(d);
            ImageView v = img.MapCpuWrite();
            for (int y = 0; y < kH; ++y)
                for (int x = 0; x < kW; ++x) {
                    // Grey again, so any colour out is interpolation error.
                    // Several stripes, so every CFA phase meets an edge.
                    const bool bright = ((x / 3) & 1) != 0;
                    *v.At<float>(x, y) = bright ? 0.95f : 0.05f;
                }
            return img;
        };

        Image mal;
        std::string e;
        if (RunDemosaic("demosaic_malvar", makeStep(CfaPattern::RGGB), {}, &mal, &e)) {
            ImageView v = mal.MapCpuRead();
            double worst = 0.0;
            for (int y = 3; y < kH - 3; ++y)
                for (int x = 3; x < kW - 3; ++x)
                    for (int c = 0; c < 3; ++c)
                        worst = std::max(worst, double(HalfToFloat(v.At<uint16_t>(x, y)[c])));

            // The scene tops out at 0.95. White balance gains are applied after
            // interpolation and legitimately push a channel above that, so this
            // allows generous room -- it is checking for the 1.6x overshoot,
            // not for exact bounds.
            Check(worst < 1.4,
                  "Malvar does not overshoot a hard edge (peak " +
                      std::to_string(worst) + ", scene peak 0.95)");
        } else {
            Check(false, "step-edge demosaic ran: " + e);
        }
    }
    {
        // With its gains at zero Malvar reduces to exactly bilinear. That is
        // what makes the correction term testable rather than taken on trust.
        Image bil, mal;
        std::string e1, e2;
        const bool a = RunDemosaic("demosaic_bilinear", MakeMosaic(CfaPattern::RGGB),
                                   {}, &bil, &e1);
        const bool b = RunDemosaic("demosaic_malvar", MakeMosaic(CfaPattern::RGGB),
                                   {{"alpha", 0.0}, {"beta", 0.0}, {"gamma", 0.0}},
                                   &mal, &e2);
        if (a && b) {
            ImageView va = bil.MapCpuRead();
            ImageView vb = mal.MapCpuRead();
            double worst = 0.0;
            for (int y = 3; y < kH - 3; ++y)
                for (int x = 3; x < kW - 3; ++x)
                    for (int c = 0; c < 3; ++c)
                        worst = std::max(worst,
                            std::abs(double(HalfToFloat(va.At<uint16_t>(x, y)[c])) -
                                     double(HalfToFloat(vb.At<uint16_t>(x, y)[c]))));
            Check(worst < 0.002,
                  "Malvar with zero gains is exactly bilinear (worst " +
                      std::to_string(worst) + ")");
        } else {
            Check(false, "both methods ran: " + (a ? e2 : e1));
        }
    }
    {
        // demosaic_consistent at strength 0 must be exactly bilinear.
        //
        // That reduction is what makes the algorithm's measurements
        // trustworthy: at zero the correction is off, so any divergence is a
        // bug in the correction rather than a property of it. The numbers in
        // that file's header -- reliably better than bilinear, reliably behind
        // AHD -- only mean something while this holds.
        // A scene with EDGES, not the smooth gradient MakeMosaic() produces.
        //
        // That matters, and it was checked: bilinear reconstructs a linear
        // gradient exactly, so the consistency residual there is genuinely
        // zero, the correction has nothing to apply, and the test passes no
        // matter how badly `strength` is mishandled. Verified by breaking the
        // strength handling deliberately -- the smooth fixture still reported
        // "worst 0.000000". A fixture that cannot express the failure cannot
        // test for it.
        auto edged = [](CfaPattern cfa) {
            Image img;
            ImageDesc d{kW, kH, Format::R32F};
            d.cfa = cfa;
            d.blackLevel = 0.0f;
            d.whiteLevel = 1.0f;
            img.Alloc(d);
            ImageView v = img.MapCpuWrite();
            for (int y = 0; y < kH; ++y)
                for (int x = 0; x < kW; ++x) {
                    // Stripes at several widths, so some are finer than the
                    // green sampling grid -- which is where bilinear loses
                    // detail and the residual becomes non-zero.
                    const bool on = ((x / 3) & 1) != 0 || ((y / 5) & 1) != 0;
                    const float rgb[3] = {on ? 0.80f : 0.15f,
                                          on ? 0.75f : 0.12f,
                                          on ? 0.70f : 0.18f};
                    *v.At<float>(x, y) = rgb[CfaColorAt(cfa, x, y)];
                }
            return img;
        };

        Image bil, con;
        std::string e1, e2;
        const bool a = RunDemosaic("demosaic_bilinear", edged(CfaPattern::RGGB),
                                   {}, &bil, &e1);
        // Both the correction AND the chroma median off. The median is a
        // separate stage with its own control, so "reduces to bilinear" means
        // "with everything this algorithm adds turned off" -- turning off only
        // the correction leaves the median running, which legitimately differs
        // from bilinear.
        const bool b = RunDemosaic("demosaic_consistent", edged(CfaPattern::RGGB),
                                   {{"strength", 0.0}, {"chroma_median", 0.0}},
                                   &con, &e2);
        if (a && b) {
            ImageView va = bil.MapCpuRead();
            ImageView vb = con.MapCpuRead();
            double worst = 0.0;
            for (int y = 3; y < kH - 3; ++y)
                for (int x = 3; x < kW - 3; ++x)
                    for (int c = 0; c < 3; ++c)
                        worst = std::max(worst,
                            std::abs(double(HalfToFloat(va.At<uint16_t>(x, y)[c])) -
                                     double(HalfToFloat(vb.At<uint16_t>(x, y)[c]))));
            Check(worst < 0.002,
                  "consistent at strength 0 is exactly bilinear (worst " +
                      std::to_string(worst) + ")");
        } else {
            Check(false, "both methods ran: " + (a ? e2 : e1));
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
                // demosaic_ahd is the reason the multi-pass GPU path exists --
                // four different kernels over shared scratch planes rather than
                // one kernel run repeatedly. This check is what proves the
                // buffer bindings are right: a pass reading the wrong plane
                // still produces a plausible-looking image, so nothing but a
                // direct comparison against the CPU catches it.
                for (const char* name : {"demosaic_passthrough", "demosaic_bilinear",
                                         "demosaic_malvar", "demosaic_ahd",
                                         "demosaic_consistent"}) {
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

                // The chroma median, on a fixture with EDGES.
                //
                // MakeMosaic() above is a smooth gradient, where the median's
                // edge gate never rejects a neighbour: every pixel takes all
                // nine and the branch that decides WHICH survive is never
                // exercised. So that comparison would pass even if the gate
                // were wrong.
                //
                // This matters because the CPU implementation filters both
                // colour-difference planes in one traversal against a cached
                // luma plane, while the GPU filters them separately. They must
                // still agree exactly, and only a fixture with real edges can
                // say so.
                {
                    auto striped = [](CfaPattern cfa) {
                        Image img;
                        ImageDesc d{kW, kH, Format::R32F};
                        d.cfa = cfa;
                        d.blackLevel = 0.0f;
                        d.whiteLevel = 1.0f;
                        img.Alloc(d);
                        ImageView v = img.MapCpuWrite();
                        for (int y = 0; y < kH; ++y)
                            for (int x = 0; x < kW; ++x) {
                                // Strong colour edges at several widths, so the
                                // gate rejects some neighbours and keeps others.
                                const bool on = ((x / 3) & 1) != 0 || ((y / 5) & 1) != 0;
                                const float rgb[3] = {on ? 0.80f : 0.15f,
                                                      on ? 0.20f : 0.75f,
                                                      on ? 0.70f : 0.10f};
                                *v.At<float>(x, y) = rgb[CfaColorAt(cfa, x, y)];
                            }
                        return img;
                    };

                    Image cpuOut, gpuOut;
                    std::string e1, e2;
                    const bool okCpu = RunDemosaicMode("demosaic_consistent",
                                                       striped(CfaPattern::RGGB),
                                                       &cpuOut, &e1, nullptr,
                                                       ExecMode::ForceCPU);
                    const bool okGpu = RunDemosaicMode("demosaic_consistent",
                                                       striped(CfaPattern::RGGB),
                                                       &gpuOut, &e2, &gpu,
                                                       ExecMode::ForceGPU);
                    if (okCpu && okGpu) {
                        ImageView a = cpuOut.MapCpuRead();
                        ImageView b = gpuOut.MapCpuRead();
                        double worst = 0.0;
                        for (int y = 1; y < kH - 1; ++y)
                            for (int x = 1; x < kW - 1; ++x)
                                for (int c = 0; c < 3; ++c)
                                    worst = std::max(worst,
                                        std::abs(double(HalfToFloat(a.At<uint16_t>(x, y)[c])) -
                                                 double(HalfToFloat(b.At<uint16_t>(x, y)[c]))));
                        Check(worst < 0.005,
                              "chroma median agrees CPU vs GPU across edges (worst " +
                                  std::to_string(worst) + ")");
                    } else {
                        Check(false, std::string("striped fixture ran both ways: ") +
                                         (okCpu ? e2 : e1));
                    }
                }
                // tonemap_local, CPU against GPU.
                //
                // Here rather than in test_script because this file already
                // stands up a device, and the check is the same kind: seven
                // passes over four shared planes, where a wrong binding -- a
                // pass reading the plane it writes, or a and b swapped --
                // still produces a plausible image with plausible statistics.
                // Only a per-pixel comparison against the CPU says the wiring
                // is right.
                {
                    constexpr int kTW = 96, kTH = 72;
                    auto hdrScene = [&] {
                        Image img;
                        ImageDesc d{kTW, kTH, Format::RGBA32F};
                        d.linear = true;
                        img.Alloc(d);
                        ImageView v = img.MapCpuWrite();
                        for (int y = 0; y < kTH; ++y)
                            for (int x = 0; x < kTW; ++x) {
                                // A wide illumination step with texture on both
                                // sides: the structure the operator is for, and
                                // enough range that a mis-bound plane is off by
                                // whole units rather than thousandths.
                                const float illum = (x < kTW / 2) ? 0.25f : 6.0f;
                                const float tex = 1.0f + 0.3f * float(((x / 2 + y / 3) & 1) ? 1 : -1);
                                float* p = v.At<float>(x, y);
                                p[0] = illum * tex * 0.9f;
                                p[1] = illum * tex;
                                p[2] = illum * tex * 1.1f;
                                p[3] = 1.0f;
                            }
                        return img;
                    };

                    auto runTl = [&](ExecMode mode, ComputeContext* g, Image* out) {
                        auto algo = Registry::Get().Create("tonemap_local");
                        if (!algo) return false;
                        Pipeline p;
                        std::vector<Data> srcs;
                        srcs.push_back(Data{hdrScene()});
                        p.AddStage(std::move(algo), "tonemap_local", {{-1, 0}}, 1, 1);
                        std::string e;
                        if (!p.Execute(&srcs, nullptr, &e, g, mode)) return false;
                        const Data* d = p.Resolve({0, 0}, &srcs);
                        const auto* im = d ? std::get_if<Image>(d) : nullptr;
                        if (!im) return false;
                        *out = const_cast<Image&>(*im).Clone();
                        return true;
                    };

                    Image cpuOut, gpuOut;
                    const bool okCpu = runTl(ExecMode::ForceCPU, nullptr, &cpuOut);
                    const bool okGpu = runTl(ExecMode::ForceGPU, &gpu, &gpuOut);
                    if (okCpu && okGpu) {
                        ImageView a = cpuOut.MapCpuRead();
                        ImageView b = gpuOut.MapCpuRead();
                        double worst = 0.0;
                        for (int y = 1; y < kTH - 1; ++y)
                            for (int x = 1; x < kTW - 1; ++x)
                                for (int c = 0; c < 3; ++c)
                                    worst = std::max(worst,
                                        std::abs(double(a.At<float>(x, y)[c]) -
                                                 double(b.At<float>(x, y)[c])));
                        // Loose enough for float order-of-operations -- the CPU
                        // slides a running sum along a row while the shader
                        // gathers 2r+1 samples, so the two accumulate in
                        // different orders -- and far tighter than any wiring
                        // mistake could survive.
                        Check(worst < 0.02,
                              "tonemap_local GPU matches CPU (worst " +
                                  std::to_string(worst) + ")");
                    } else {
                        Check(false, "tonemap_local ran both ways");
                    }
                }

                // crop, CPU against GPU, in BOTH modes.
                //
                // Both, because they are different code paths that happen to
                // share a kernel: applied resamples through a rotation, preview
                // draws a polygon. A wiring mistake in one would not show in
                // the other -- and the applied path is also the only algorithm
                // here whose OUTPUT IS A DIFFERENT SIZE than its input, so this
                // doubles as the check that OutputDesc reaches the GPU
                // dispatch. It does: the shader's Width/Height and the thread
                // grid both come from outputs[0], which is already the cropped
                // raster by the time it is bound.
                for (int preview = 0; preview <= 1; ++preview) {
                    auto runCrop = [&](ExecMode mode, ComputeContext* g, Image* out) {
                        auto algo = Registry::Get().Create("crop");
                        if (!algo) return false;
                        auto set = [&](const char* n, double v) {
                            if (ParamBase* pb = algo->FindParam(n)) {
                                std::string e; pb->SetFromScript(Value(v), &e);
                            }
                        };
                        set("preview", double(preview));
                        set("left", 0.15); set("right", 0.10);
                        set("top", 0.20);  set("bottom", 0.05);
                        // Rotated: an unrotated crop is a pure copy and would
                        // agree even if the sampling were wrong.
                        set("angle", 5.0);

                        // A gradient with fine checker detail, so bilinear
                        // sampling under rotation actually has something to
                        // disagree about. A smooth ramp would agree between
                        // any two interpolators.
                        constexpr int kCW = 96, kCH = 72;
                        Image src;
                        ImageDesc cd{kCW, kCH, Format::RGBA32F};
                        src.Alloc(cd);
                        {
                            ImageView v = src.MapCpuWrite();
                            for (int y = 0; y < kCH; ++y)
                                for (int x = 0; x < kCW; ++x) {
                                    float* p = v.At<float>(x, y);
                                    const float check =
                                        (((x / 3) + (y / 3)) & 1) ? 0.7f : 0.2f;
                                    p[0] = float(x) / float(kCW) * check;
                                    p[1] = float(y) / float(kCH) * check;
                                    p[2] = check;
                                    p[3] = 1.0f;
                                }
                        }

                        std::vector<Data> srcs;
                        srcs.push_back(Data{std::move(src)});
                        Pipeline p;
                        p.AddStage(std::move(algo), "crop", {{-1, 0}}, 1, 1);
                        std::string e;
                        if (!p.Execute(&srcs, nullptr, &e, g, mode)) return false;
                        const Data* d = p.Resolve({0, 0}, &srcs);
                        const auto* im = d ? std::get_if<Image>(d) : nullptr;
                        if (!im) return false;
                        *out = const_cast<Image&>(*im).Clone();
                        return true;
                    };

                    Image cpuOut, gpuOut;
                    const bool okCpu = runCrop(ExecMode::ForceCPU, nullptr, &cpuOut);
                    const bool okGpu = runCrop(ExecMode::ForceGPU, &gpu, &gpuOut);
                    const char* what = preview ? "preview" : "applied";

                    if (okCpu && okGpu) {
                        Check(cpuOut.Desc().width == gpuOut.Desc().width &&
                              cpuOut.Desc().height == gpuOut.Desc().height,
                              std::string("crop ") + what +
                                  ": both paths agree on the output size (" +
                                  std::to_string(cpuOut.Desc().width) + "x" +
                                  std::to_string(cpuOut.Desc().height) + " vs " +
                                  std::to_string(gpuOut.Desc().width) + "x" +
                                  std::to_string(gpuOut.Desc().height) + ")");

                        ImageView a = cpuOut.MapCpuRead();
                        ImageView b = gpuOut.MapCpuRead();
                        if (a.desc.width == b.desc.width &&
                            a.desc.height == b.desc.height) {
                            double worst = 0.0;
                            for (int y = 0; y < a.desc.height; ++y)
                                for (int x = 0; x < a.desc.width; ++x)
                                    for (int c = 0; c < 3; ++c)
                                        worst = std::max(worst,
                                            std::abs(double(a.At<float>(x, y)[c]) -
                                                     double(b.At<float>(x, y)[c])));
                            // Tight: both paths do the same four loads and
                            // three lerps, so only float rounding separates
                            // them. The preview's line COLOUR legitimately
                            // differs between paths on a float image -- see
                            // GpuConstants -- but the fixture's rectangle sits
                            // inside the frame and the line is a few pixels of
                            // it, so it does not dominate this.
                            Check(worst < 0.05,
                                  std::string("crop ") + what +
                                      " GPU matches CPU (worst " +
                                      std::to_string(worst) + ")");
                        }
                    } else {
                        Check(false, std::string("crop ") + what + " ran both ways");
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
