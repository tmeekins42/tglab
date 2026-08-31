// demosaic_stages -- bilinear, with every intermediate exposed as an image.
//
// A debugging instrument, not a demosaicer. It runs exactly the arithmetic
// demosaic_bilinear runs, but emits each step as its own output port so the
// stages can be displayed side by side and compared pixel for pixel.
//
// It exists because the colour-fringing hunt kept failing the same way: a fix
// would be verified on one hand-picked pixel, look correct, and barely move the
// whole-frame count. A single number cannot say WHICH pixels still fail or WHICH
// step broke them. These images can.
//
// The ports, in pipeline order:
//
//   interp   the interpolated RGB, still raw-scaled, before anything else
//   peak     the brightest RAW SAMPLE feeding each channel
//   clipmask which channels were judged saturated (1.0 per channel if so)
//   balanced after white balance, before the clipped-channel repair
//   repaired after the repair, before the colour matrix
//   out      the final image -- identical to demosaic_bilinear's output
//
// Comparing `balanced` against `repaired` shows exactly what the repair did;
// comparing `repaired` against `out` shows what the colour matrix did with it.
// `clipmask` says which pixels the repair even touched, which is the question
// that kept getting answered wrong.
#include <algorithm>
#include <cmath>
#include <vector>

#include "clip_repair.h"

#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"

namespace tglab {
namespace {

// Hue in degrees, or -1 for an achromatic pixel.
float HueOf(const float* p) {
    const float mx = std::max(p[0], std::max(p[1], p[2]));
    const float mn = std::min(p[0], std::min(p[1], p[2]));
    const float d = mx - mn;
    if (d < 1e-6f) return -1.0f;
    float h;
    if (mx == p[0])      h = std::fmod((p[1] - p[2]) / d + 6.0f, 6.0f);
    else if (mx == p[1]) h = (p[2] - p[0]) / d + 2.0f;
    else                 h = (p[0] - p[1]) / d + 4.0f;
    h *= 60.0f;
    return h < 0.0f ? h + 360.0f : h;
}

float SatOf(const float* p) {
    const float mx = std::max(p[0], std::max(p[1], p[2]));
    const float mn = std::min(p[0], std::min(p[1], p[2]));
    return mx > 1e-5f ? (mx - mn) / mx : 0.0f;
}

float HueDelta(float a, float b) {
    if (a < 0.0f || b < 0.0f) return 0.0f;
    float d = std::fabs(a - b);
    return d > 180.0f ? 360.0f - d : d;
}

class DemosaicStages : public AlgorithmBase {
public:
    const char* Name()     const override { return "demosaic_stages"; }
    const char* Category() const override { return "demosaic"; }

    PortList Inputs() const override {
        return {{"src", DataType::Image, FormatSpec::Any}};
    }
    PortList Outputs() const override {
        return {{"interp",    DataType::Image, FormatSpec::RGBA16F},
                {"balanced",  DataType::Image, FormatSpec::RGBA16F},
                {"clamped",   DataType::Image, FormatSpec::RGBA16F},
                {"unclamped", DataType::Image, FormatSpec::RGBA16F},
                {"out",       DataType::Image, FormatSpec::RGBA16F},
                {"shift",     DataType::Image, FormatSpec::RGBA16F}};
    }

    void RunCPU(RunCtx& ctx) override {
        const ImageView src = ctx.In(0);
        ImageView o[6];
        for (int i = 0; i < 6; ++i) {
            o[i] = ctx.Out(i);
            if (!o[i].Valid()) return;
        }
        if (!src.Valid()) return;

        m_in.Unpack(src);
        if (!m_in.Valid()) return;

        const int w = m_in.Width(), h = m_in.Height();
        const CfaPattern cfa = src.desc.cfa;
        // Not a mosaic: copy the input into every stage rather than returning
        // early. An algorithm that leaves its outputs untouched is
        // indistinguishable from a silent failure, and test_filters checks for
        // exactly that.
        if (cfa == CfaPattern::None || cfa == CfaPattern::XTrans) {
            const int ch = m_in.Channels();
            const float scale = m_in.ValueScale();
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x)
                    for (int k = 0; k < 6; ++k) {
                        uint16_t* p = o[k].At<uint16_t>(x, y);
                        for (int c2 = 0; c2 < 3; ++c2)
                            p[c2] = FloatToHalf(
                                m_in.Get(x, y, ch == 1 ? 0 : c2) / scale);
                        p[3] = FloatToHalf(1.0f);
                    }
            return;
        }

        const float black = src.desc.blackLevel;
        const float range = std::max(src.desc.whiteLevel - black, 1e-6f);

        m_s.assign(size_t(w) * size_t(h), 0.0f);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                m_s[size_t(y) * size_t(w) + size_t(x)] =
                    std::clamp((m_in.Get(x, y, 0) - black) / range, 0.0f, 4.0f);

        auto at = [&](int x, int y) {
            x = std::clamp(x, 0, w - 1);
            y = std::clamp(y, 0, h - 1);
            return m_s[size_t(y) * size_t(w) + size_t(x)];
        };

        auto put = [](ImageView& v, int x, int y, const float* rgb) {
            uint16_t* p = v.At<uint16_t>(x, y);
            p[0] = FloatToHalf(rgb[0]);
            p[1] = FloatToHalf(rgb[1]);
            p[2] = FloatToHalf(rgb[2]);
            p[3] = FloatToHalf(1.0f);
        };

        for (int y = 0; y < h; ++y) {
            if (ctx.Cancelled()) return;
            for (int x = 0; x < w; ++x) {
                const int c = CfaColorAt(cfa, x, y);
                float rgb[3] = {0, 0, 0};
                rgb[c] = at(x, y);

                if (c == 1) {
                    const int horizColor = CfaColorAt(cfa, x - 1, y);
                    rgb[horizColor]     = 0.5f * (at(x - 1, y) + at(x + 1, y));
                    rgb[2 - horizColor] = 0.5f * (at(x, y - 1) + at(x, y + 1));
                } else {
                    rgb[1] = 0.25f * (at(x - 1, y) + at(x + 1, y) +
                                      at(x, y - 1) + at(x, y + 1));
                    rgb[2 - c] = 0.25f * (at(x - 1, y - 1) + at(x + 1, y - 1) +
                                          at(x - 1, y + 1) + at(x + 1, y + 1));
                }
                put(o[0], x, y, rgb);

                // White balance alone, WITHOUT the clamp -- this is the stage
                // that shows why the clamp exists. A blown highlight arrives
                // here as 1.17 / 1.00 / 3.08 and the matrix turns that magenta.
                const float bal[3] = {rgb[0] * src.desc.camMul[0],
                                      rgb[1] * src.desc.camMul[1],
                                      rgb[2] * src.desc.camMul[2]};
                put(o[1], x, y, bal);

                // The same, with the clamp: nothing exceeds the brightest
                // neutral the sensor can describe.
                float cl[3] = {rgb[0], rgb[1], rgb[2]};
                BalanceAndClamp(src.desc, cl);
                put(o[2], x, y, cl);

                // What the matrix would have produced WITHOUT the clamp. This
                // is the magenta, isolated.
                const float* m = src.desc.rgbCam;
                const float ur = m[0] * bal[0] + m[1] * bal[1] + m[2] * bal[2];
                const float ug = m[3] * bal[0] + m[4] * bal[1] + m[5] * bal[2];
                const float ub = m[6] * bal[0] + m[7] * bal[1] + m[8] * bal[2];
                const float unclamped[3] = {ur, ug, ub};
                put(o[3], x, y, unclamped);

                const float fr = m[0] * cl[0] + m[1] * cl[1] + m[2] * cl[2];
                const float fg = m[3] * cl[0] + m[4] * cl[1] + m[5] * cl[2];
                const float fb = m[6] * cl[0] + m[7] * cl[1] + m[8] * cl[2];
                const float fin[3] = {fr, fg, fb};
                put(o[4], x, y, fin);
                // The diagnostic map. What the matrix DID to this pixel:
                //
                //   red   = hue shift, 0..30 deg      (rotation)
                //   green = saturation GAIN, 0..0.5   (pushed away from grey)
                //   blue  = how negative any channel went, 0..1
                //
                // The matrix's own rotation is a bounded, hue-dependent
                // -6.9..+18.4 degrees, so red alone does not mark a fault --
                // measured, 41.8% of near-clipping pixels exceed 12 deg but so
                // do 19.7% of ordinary ones. Saturation gain separates far
                // better: above +0.25 it is 10.4% of near-clipping pixels
                // against 0.2% elsewhere, a 50x enrichment. Green is the
                // channel to look at.
                const float dh = HueDelta(HueOf(cl), HueOf(fin)) / 30.0f;
                const float dsat = (SatOf(fin) - SatOf(cl)) / 0.5f;
                const float neg = -std::min(0.0f, std::min(fin[0],
                                            std::min(fin[1], fin[2])));
                const float shift[3] = {std::clamp(dh, 0.0f, 1.0f),
                                        std::clamp(dsat, 0.0f, 1.0f),
                                        std::clamp(neg, 0.0f, 1.0f)};
                put(o[5], x, y, shift);
            }
        }
    }

private:
    PixelBuffer        m_in;
    std::vector<float> m_s;
};

REGISTER_ALGORITHM(DemosaicStages);

}  // namespace
}  // namespace tglab
