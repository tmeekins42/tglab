// The base tone curve, and the camera preview extraction.
//
// The curve's important properties are all checkable without a photograph, so
// this half runs in ctest. The preview half needs a raw file and is skipped
// without one.
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

#include "../src/algo_util/tone_curve.h"
#include "../src/algo_util/auto_develop.h"
#include "../src/core/raw_io.h"
#include "../src/core/image.h"

using namespace tglab;

static int g_fail = 0;
static void Check(bool ok, const std::string& what) {
    std::printf("%s  %s\n", ok ? "[ ok ]" : "[FAIL]", what.c_str());
    if (!ok) ++g_fail;
}

int main(int argc, char** argv) {
    std::printf("=== tone curve ===\n");

    // Black must be black. This is the property the first design got wrong:
    // compressing both ends asymptotically meant the toe could never reach
    // zero, and 0.002 linear rendered as 0.226 -- a milky grey where black
    // belonged. Worth an explicit test because the failure looked like a
    // washed-out picture rather than an obviously broken curve.
    Check(ToneCurve(0.0f) == 0.0f, "zero maps to zero");
    Check(ToneCurve(-1.0f) == 0.0f, "negative maps to zero");
    Check(ToneCurve(0.002f) < 0.06f, "near-black stays near black");

    // The midtone anchor: 0.18 linear is middle grey and must land where every
    // other developer puts it, or our render is not comparable to theirs.
    Check(std::fabs(ToneCurve(0.18f) - 0.46f) < 0.005f,
          "middle grey 0.18 -> 0.46");

    // The shoulder: nothing, however bright, may reach 1.0. This is the whole
    // reason the curve exists -- the alley's sky sits at 4.0 linear after the
    // exposure push and used to clip flat.
    Check(ToneCurve(1.0f) < 1.0f, "1.0 linear does not clip");
    Check(ToneCurve(4.0f) < 1.0f, "4.0 linear does not clip");
    Check(ToneCurve(8.0f) < 1.0f, "8.0 linear does not clip");
    Check(ToneCurve(4.0f) > 0.85f, "4.0 linear still renders bright");

    // The shoulder covers the range a photograph reaches, not every possible
    // float. Its asymptote sits at 1.39, so inputs above about 19 linear do
    // still clamp -- see kShoulder for why that trade is deliberate rather
    // than a gap. Asserted so a future change to the constants has to
    // acknowledge it.
    Check(ToneCurve(19.0f) < 1.0f, "the top of the real range is safe");
    Check(ToneCurve(64.0f) >= 1.0f,
          "beyond the real range it does clamp, as documented");

    // Monotonic: a curve that inverted any ordering would make a brighter
    // subject render darker than a dimmer one.
    {
        bool mono = true;
        float prev = -1.0f;
        for (double x = 0.0; x <= 8.0; x += 0.001) {
            const float y = ToneCurve(float(x));
            if (y < prev - 1e-6f) { mono = false; break; }
            prev = y;
        }
        Check(mono, "monotonic across 0..8");
    }

    // NaN must not propagate into the display texture as a garbage byte.
    {
        const float nan = std::nanf("");
        Check(ToneCurve(nan) == 0.0f, "NaN maps to zero");
    }

    // The HLSL copy must exist and carry the same constants, since compare
    // mode checks the CPU result against the GPU one and a divergence here
    // would look like a shader bug.
    {
        const std::string hlsl = kToneCurveHlsl;
        Check(hlsl.find("0.18") != std::string::npos &&
              hlsl.find("0.46") != std::string::npos &&
              hlsl.find("3.2")  != std::string::npos,
              "HLSL copy carries the same anchors");
    }

    if (argc < 2) {
        std::printf("\n(no raw file given; skipping preview and measurement checks)\n");
        std::printf("\n%s\n", g_fail ? "FAILED" : "all passed");
        return g_fail ? 1 : 0;
    }

    // --- the camera's embedded preview -------------------------------------
    std::printf("\n=== embedded preview: %s ===\n", argv[1]);
    {
        Image thumb;
        std::string err;
        const bool ok = LoadRawPreview(argv[1], 96, &thumb, &err);
        Check(ok, "extracted the camera's JPEG preview");
        if (ok) {
            const ImageDesc& d = thumb.Desc();
            std::printf("       %dx%d  %s\n", d.width, d.height,
                        d.linear ? "linear" : "display-encoded");
            Check(d.width > 0 && d.height > 0, "preview has real dimensions");
            Check(d.width <= 96 * 2 && d.height <= 96 * 2,
                  "preview was downsampled to about the requested size");
            Check(!d.linear, "preview is display-encoded (a JPEG already is)");

            // The point of the exercise: the camera's own render should be a
            // legible picture, not the near-black a naive raw decode gives.
            // The undeveloped mosaic's median measured 0.0219 linear, which is
            // 0.160 display -- so anything at all bright proves we are looking
            // at the camera's rendering rather than the sensor data.
            ImageView v = thumb.MapCpuRead();
            double sum = 0; long n = 0;
            for (int y = 0; y < d.height; ++y) {
                const uint8_t* row = v.At<uint8_t>(0, y);
                for (int x = 0; x < d.width; ++x) {
                    sum += (row[x * 4 + 0] + row[x * 4 + 1] + row[x * 4 + 2]) / 3.0;
                    ++n;
                }
            }
            const double mean = n ? sum / double(n) / 255.0 : 0.0;
            std::printf("       mean brightness %.3f\n", mean);
            Check(mean > 0.20, "preview is a legible picture, not near-black");
        }
    }

    // --- what the measurement now suggests ----------------------------------
    std::printf("\n=== auto-develop measurement ===\n");
    {
        Image m; std::string err;
        if (!LoadRawMosaic(argv[1], &m, &err)) {
            std::printf("load: %s\n", err.c_str());
            return 1;
        }
        const AutoDevelopSuggestion s = SuggestExposure(m);
        std::printf("  exposure %+.2f  highlights %+.2f  shadows %+.2f  blacks %+.2f\n",
                    s.exposure, s.highlights, s.shadows, s.blacks);
        std::printf("  median %.4f  dynamic range %.2f stops  clipped %.2f%% -> %.2f%%\n",
                    s.midtone, s.dynamicRange,
                    100.0 * s.clippedFrac, 100.0 * s.clippedAfter);

        Check(s.valid, "measurement is valid");
        Check(s.dynamicRange > 0.0f, "dynamic range was measured");

        // A wide scene must get proactive highlight recovery even with nothing
        // clipped. That is the case clipping-driven recovery could never see,
        // and it is why the alley rendered flat while Lightroom's Auto chose
        // highlights -69 on the same frame.
        if (s.dynamicRange > 5.0f) {
            Check(s.highlights < 0.0f,
                  "a wide scene gets highlight recovery without clipping");
            Check(s.shadows > 0.0f, "a wide scene gets a shadow lift");
        }

        // The push, through the curve, must land the midtones in a sensible
        // place and must NOT drive the top to white. This is the end-to-end
        // claim: the same exposure that used to blow the sky now does not.
        const float lifted = s.midtone * std::exp2(s.exposure);
        const float shown  = ToneCurve(lifted);
        std::printf("  median through the curve: %.3f\n", shown);
        Check(shown > 0.25f && shown < 0.60f,
              "median lands in the usable midtones after the curve");

        const float sky = ToneCurve(s.highlight * std::exp2(s.exposure));
        std::printf("  brightest detail through the curve: %.3f\n", sky);
        Check(sky < 1.0f, "the brightest detail does not clip");
    }

    std::printf("\n%s\n", g_fail ? "FAILED" : "all passed");
    return g_fail ? 1 : 0;
}
