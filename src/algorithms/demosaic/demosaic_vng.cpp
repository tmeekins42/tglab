// demosaic_vng — Variable Number of Gradients (Chang, Cheng and Acharya, 1999).
//
// The idea in one line: measure the gradient in all EIGHT directions, keep only
// the directions that are smooth, and average over exactly those. The number
// kept varies per pixel -- hence the name -- so a pixel in flat sky averages
// over all eight while one on a hard edge may use two.
//
// WHY THAT IS DIFFERENT FROM AHD, which is the reason both are worth having.
// AHD makes a BINARY choice: interpolate green horizontally or vertically, pick
// whichever homogeneity map prefers, and commit. That is exactly right on a
// feature aligned with an axis and has nowhere to go on one that is not -- a
// 45-degree branch is equally bad both ways, so AHD picks the lesser evil and
// still smears it.
//
// VNG never chooses a single direction. It takes a threshold over the gradient
// spread and averages every direction below it, so a diagonal edge keeps its
// two diagonal neighbours and drops the six that cross it. The cost is that it
// is softer than AHD on clean horizontal and vertical detail, because averaging
// several directions is a low-pass however carefully the set is chosen.
//
// VERIFIED AGAINST THE REFERENCE. This matches LibRaw's own vng_interpolate()
// on a real frame, which is the only claim that matters for a method whose
// purpose is to be compared against others: an implementation that is merely
// "VNG-like" measures its own bugs rather than the algorithm.
//
//     LibRaw vs ours, _L0A0738.CR2, 5796x3870, identity white balance and
//     identity colour matrix on both sides so only the interpolation differs:
//
//       mean absolute difference   0.26 of 255
//       beyond 1/255                2.93% of samples
//       beyond 4/255                1.28%
//
// The residual is the pipeline around the interpolation, not the interpolation:
// this stage applies BalanceAndClamp and CameraMatrixInGamut, which dcraw has
// no equivalent of, and 39% of the pixels beyond 4/255 sit within 3px of
// clipping -- exactly where that handling acts. Sample pixels agree to three
// decimal places.
//
// GETTING THERE TOOK THREE WRONG IMPLEMENTATIONS, and the reason is worth
// recording because it is not a detail:
//
//   1. VNG IS A REFINEMENT PASS, NOT A FROM-MOSAIC METHOD. dcraw's
//      vng_interpolate() opens with lin_interpolate() -- plain bilinear filling
//      all three channels everywhere -- and every subsequent read, both the
//      gradients and the neighbour averaging, is of a filled RGB pixel. Reading
//      the raw mosaic instead produces an asymmetry that cannot be patched: an
//      axis direction reaches green and the centre colour but never the
//      opposite one, a diagonal reaches the opposite colour but never green, so
//      whichever directions survive, the channels end up averaged over
//      different neighbourhoods and subtracting those means manufactures
//      colour.
//
//   2. THE GRADIENT IS A TABLE, NOT A FORMULA. Three attempts at deriving an
//      equivalent all failed differently. dcraw uses 64 curated candidate
//      sample pairs, each tagged with a weight and a BITMASK of which of the
//      eight directions it contributes to -- a pair feeds several directions at
//      once, with differing weights, and rows are filtered per CFA phase so
//      only same-colour pairs count. Nothing about that falls out of "measure
//      roughness along a direction". It is transcribed below, verified
//      byte-for-byte against LibRaw's copy.
//
// The symptom of getting this wrong, for anyone debugging a similar method: the
// output looked "like bilinear but more pronounced", which was literally the
// arithmetic -- 71% of pixels kept 6 or more of the 8 directions, median 7, and
// averaging almost every direction IS bilinear with extra reach.
//
// HOW IT MEASURES, and the honest reading is that VNG is not the one to reach
// for on a modern sensor.
//
//     REAL FRAME (_L0A0738.CR2). "false colour" counts pixels where green is
//     crushed although the green photosites were the brightest of the three --
//     colour the sensor never recorded:
//
//       method       false colour   detail    ms
//       malvar              9282   0.00325   1675
//       ahd                 9790   0.00314  11564
//       ppg                10993   0.00319   3203
//       bilinear           16876   0.00259    935
//       consistent         17415   0.00407   7677
//       vng                22045   0.00525  14087
//
//     SYNTHETIC, features 8px wide -- well inside what a Bayer array can
//     resolve:
//
//       method       vertical  diagonal   texture   overall
//       consistent    0.01526   0.01438   0.11703   0.04861
//       ppg           0.01495   0.01955   0.11378   0.04921
//       ahd           0.00764   0.02082   0.12303   0.05031
//       bilinear      0.01845   0.02440   0.10904   0.05044
//       malvar        0.02065   0.03326   0.11979   0.05776
//       vng           0.02470   0.03507   0.11859   0.05930
//
// A WARNING ABOUT THE SYNTHETIC FIXTURE, because it misled this work badly.
// An earlier version used 3-pixel bars and a 2-pixel checkerboard -- at or past
// the Nyquist limit for red and blue, which a Bayer array samples every other
// pixel. On that fixture VNG scored BEST of the six and the broken,
// non-conformant version scored best of all. Widening the features to 8px
// reverses the ranking and collapses the spread. The fixture was measuring
// aliasing behaviour, not reconstruction quality, and it ranked a known-broken
// implementation first -- which should have been the warning it was not.
//
// So VNG earns its place here as a faithful reference point, not as a
// recommendation. It carries the most luminance detail of the six and the most
// false colour, which is the trade the paper's design implies: deciding from
// gradients is exactly what noise and clipping corrupt, and a method that
// commits to one direction (AHD) or rides colour differences (PPG) degrades
// more gracefully on a real photograph.
//
// THE THRESHOLD IS THE WHOLE ALGORITHM. Chang's paper sets it from the spread
// of the gradients themselves:
//
//     T = k1 * min(gradients) + k2 * (max - min)
//
// with k1 = 1.5 and k2 = 0.5. Proportional rather than absolute, so it means
// the same thing in shadow as in highlight -- a fixed threshold either keeps
// everything in shadow, where all gradients are small, or nothing in highlight.
// Both constants are exposed, because the paper's values are a starting point
// and this is a lab.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "clip_repair.h"

#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"

namespace tglab {

class DemosaicVng : public AlgorithmBase {
public:
    const char* Name()     const override { return "demosaic_vng"; }
    const char* Category() const override { return "demosaic"; }

    PortList Inputs()  const override { return {{"src", DataType::Image, FormatSpec::Any}}; }
    PortList Outputs() const override { return {{"out", DataType::Image, FormatSpec::RGBA16F}}; }

    void RunCPU(RunCtx& ctx) override {
        const ImageView src = ctx.In(0);
        ImageView       dst = ctx.Out(0);
        if (!src.Valid() || !dst.Valid()) return;

        m_in.Unpack(src);
        if (!m_in.Valid()) return;

        const int w = m_in.Width(), h = m_in.Height();
        const CfaPattern cfa = src.desc.cfa;

        // Not a mosaic: pass it through rather than inventing a pattern, so an
        // unconditional demosaic in a script is harmless on an ordinary image.
        if (cfa == CfaPattern::None || cfa == CfaPattern::XTrans) {
            PassThrough(dst, w, h);
            return;
        }

        const float black = src.desc.blackLevel;
        const float range = std::max(src.desc.whiteLevel - black, 1e-6f);
        const size_t n = size_t(w) * size_t(h);

        m_s.assign(n, 0.0f);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                m_s[size_t(y) * size_t(w) + size_t(x)] =
                    std::clamp((m_in.Get(x, y, 0) - black) / range, 0.0f, 4.0f);

        auto S = [&](int x, int y) -> float {
            x = std::clamp(x, 0, w - 1);
            y = std::clamp(y, 0, h - 1);
            return m_s[size_t(y) * size_t(w) + size_t(x)];
        };

        // --- pass 1: bilinear, filling all three channels everywhere --------
        //
        // VNG IS A REFINEMENT PASS, NOT A FROM-MOSAIC METHOD, and missing that
        // is what made the first two attempts fail. dcraw's vng_interpolate()
        // opens with lin_interpolate(), and every subsequent read -- both the
        // gradient terms and the neighbour averaging -- is of a fully populated
        // RGB pixel, never of a raw mosaic sample.
        //
        // That single fact dissolves the problem the earlier versions kept
        // hitting. Sampling the mosaic directly, an axis direction reaches
        // green and the centre colour but never the opposite one, while a
        // diagonal reaches the opposite colour but never green -- so whichever
        // directions survive, the three channels end up averaged over different
        // neighbourhoods, and subtracting those means manufactures colour. On a
        // filled image every direction offers all three channels and the
        // asymmetry does not exist.
        m_rgb.assign(n * 3, 0.0f);
        for (int y = 0; y < h; ++y) {
            if (ctx.Cancelled()) return;
            for (int x = 0; x < w; ++x) {
                const int c = CfaColorAt(cfa, x, y);
                float* p = &m_rgb[(size_t(y) * size_t(w) + size_t(x)) * 3];
                p[c] = S(x, y);
                if (c == 1) {
                    const int hc = CfaColorAt(cfa, x - 1, y);
                    p[hc]     = 0.5f * (S(x - 1, y) + S(x + 1, y));
                    p[2 - hc] = 0.5f * (S(x, y - 1) + S(x, y + 1));
                } else {
                    p[1] = 0.25f * (S(x - 1, y) + S(x + 1, y) +
                                    S(x, y - 1) + S(x, y + 1));
                    p[2 - c] = 0.25f * (S(x - 1, y - 1) + S(x + 1, y - 1) +
                                        S(x - 1, y + 1) + S(x + 1, y + 1));
                }
            }
        }

        auto P = [&](int x, int y) -> const float* {
            x = std::clamp(x, 0, w - 1);
            y = std::clamp(y, 0, h - 1);
            return &m_rgb[(size_t(y) * size_t(w) + size_t(x)) * 3];
        };

        const float k1 = float(m_k1);
        const float k2 = float(m_k2);
        const float floorT = float(m_floor);

        // --- pass 2: variable-number-of-gradients refinement ----------------
        for (int y = 0; y < h; ++y) {
            if (ctx.Cancelled()) return;
            for (int x = 0; x < w; ++x) {
                const int c = CfaColorAt(cfa, x, y);
                float rgb[3];

                if (x < 2 || y < 2 || x + 2 >= w || y + 2 >= h) {
                    // No 5x5 neighbourhood at the border. dcraw simply does not
                    // write those rows; keeping bilinear there is the same
                    // decision without leaving the edge undefined.
                    const float* p = P(x, y);
                    rgb[0] = p[0]; rgb[1] = p[1]; rgb[2] = p[2];
                } else {
                    Refine(cfa, P, S, x, y, c, k1, k2, floorT, rgb);
                }

                ApplyColour(src.desc, rgb);

                uint16_t* q = dst.At<uint16_t>(x, y);
                q[0] = FloatToHalf(rgb[0]);
                q[1] = FloatToHalf(rgb[1]);
                q[2] = FloatToHalf(rgb[2]);
                q[3] = FloatToHalf(1.0f);
            }
        }
    }

private:
    // The eight neighbour directions, in dcraw's `chood` order.
    static constexpr int kDx[8] = {-1,  0, +1, +1, +1,  0, -1, -1};
    static constexpr int kDy[8] = {-1, -1, -1,  0, +1, +1, +1,  0};

    // dcraw's `terms` table, transcribed rather than re-derived.
    //
    // WHY TRANSCRIBED. Three attempts at deriving an equivalent gradient all
    // failed, each in a different way, and the reason is that this table is not
    // a formula with a closed form -- it is a curated list of 64 candidate
    // sample pairs, each tagged with a weight and a BITMASK saying which of the
    // eight directions it contributes to. A pair contributes to several
    // directions at once, and the per-term weights differ. Nothing about that
    // falls out of "measure roughness along a direction".
    //
    // Each row is: y1, x1, y2, x2, weight, grads
    //   (y1,x1) and (y2,x2) are the two samples to difference
    //   weight is a left-shift, so 0 = x1, 1 = x2, 2 = x4
    //   grads is a bitmask over the eight directions
    //
    // Rows are filtered at setup: a row is used only where both samples land on
    // the SAME colour for this CFA phase, and rows whose two samples are one
    // diagonal step apart in a diagonal-symmetric phase are dropped. That
    // filtering is why the effective term set differs per phase, and why a
    // single generic formula cannot reproduce it.
    //
    // A weight of -128 or -120 in the table is dcraw's marker for a row that
    // contributes to every direction (0x80 | 0x40 | ...); it is stored as a
    // signed char and read as a bitmask, so it must be taken unsigned.
    static const signed char* Terms() {
        static const signed char t[] = {
            -2,-2,+0,-1, 0,0x01, -2,-2,+0,+0, 1,0x01, -2,-1,-1,+0, 0,0x01,
            -2,-1,+0,-1, 0,0x02, -2,-1,+0,+0, 0,0x03, -2,-1,+0,+1, 1,0x01,
            -2,+0,+0,-1, 0,0x06, -2,+0,+0,+0, 1,0x02, -2,+0,+0,+1, 0,0x03,
            -2,+1,-1,+0, 0,0x04, -2,+1,+0,-1, 1,0x04, -2,+1,+0,+0, 0,0x06,
            -2,+1,+0,+1, 0,0x02, -2,+2,+0,+0, 1,0x04, -2,+2,+0,+1, 0,0x04,
            -1,-2,-1,+0, 0,-128, -1,-2,+0,-1, 0,0x01, -1,-2,+1,-1, 0,0x01,
            -1,-2,+1,+0, 1,0x01, -1,-1,-1,+1, 0,-120, -1,-1,+1,-2, 0,0x40,
            -1,-1,+1,-1, 0,0x22, -1,-1,+1,+0, 0,0x33, -1,-1,+1,+1, 1,0x11,
            -1,+0,-1,+2, 0,0x08, -1,+0,+0,-1, 0,0x44, -1,+0,+0,+1, 0,0x11,
            -1,+0,+1,-2, 1,0x40, -1,+0,+1,-1, 0,0x66, -1,+0,+1,+0, 1,0x22,
            -1,+0,+1,+1, 0,0x33, -1,+0,+1,+2, 1,0x10, -1,+1,+1,-1, 1,0x44,
            -1,+1,+1,+0, 0,0x66, -1,+1,+1,+1, 0,0x22, -1,+1,+1,+2, 0,0x10,
            -1,+2,+0,+1, 0,0x04, -1,+2,+1,+0, 1,0x04, -1,+2,+1,+1, 0,0x04,
            +0,-2,+0,+0, 1,-128, +0,-1,+0,+1, 1,-120, +0,-1,+1,-2, 0,0x40,
            +0,-1,+1,+0, 0,0x11, +0,-1,+2,-2, 0,0x40, +0,-1,+2,-1, 0,0x20,
            +0,-1,+2,+0, 0,0x30, +0,-1,+2,+1, 1,0x10, +0,+0,+0,+2, 1,0x08,
            +0,+0,+2,-2, 1,0x40, +0,+0,+2,-1, 0,0x60, +0,+0,+2,+0, 1,0x20,
            +0,+0,+2,+1, 0,0x30, +0,+0,+2,+2, 1,0x10, +0,+1,+1,+0, 0,0x44,
            +0,+1,+1,+2, 0,0x10, +0,+1,+2,-1, 1,0x40, +0,+1,+2,+0, 0,0x60,
            +0,+1,+2,+1, 0,0x20, +0,+1,+2,+2, 0,0x10, +1,-2,+1,+0, 0,-128,
            +1,-1,+1,+1, 0,-120, +1,+0,+1,+2, 0,0x08, +1,+0,+2,-1, 0,0x40,
            +1,+0,+2,+1, 0,0x10,
        };
        return t;
    }

    // The eight gradients at one pixel, built from the table above.
    //
    // Only same-colour pairs count, exactly as dcraw's setup filters them --
    // that is what makes the eight numbers commensurate, and getting it wrong
    // by comparing a red sample against a green one is what made the earlier
    // attempts rank channels rather than directions.
    template <class Sample>
    static void Gradients(CfaPattern cfa, Sample S, int x, int y, float* gval) {
        for (int d = 0; d < 8; ++d) gval[d] = 0.0f;

        // dcraw drops rows whose two samples sit one diagonal step apart when
        // the phase is diagonally symmetric (a green site on a Bayer array).
        const int diag =
            (CfaColorAt(cfa, x, y + 1) == CfaColorAt(cfa, x + 1, y)) ? 2 : 1;

        const signed char* cp = Terms();
        for (int i = 0; i < 64; ++i) {
            const int y1 = *cp++, x1 = *cp++;
            const int y2 = *cp++, x2 = *cp++;
            const int weight = *cp++;
            const int grads  = int(static_cast<unsigned char>(*cp++));

            const int c1 = CfaColorAt(cfa, x + x1, y + y1);
            if (CfaColorAt(cfa, x + x2, y + y2) != c1) continue;
            if (std::abs(y1 - y2) == diag && std::abs(x1 - x2) == diag) continue;

            const float diff =
                std::fabs(S(x + x1, y + y1) - S(x + x2, y + y2)) *
                float(1 << weight);
            for (int d = 0; d < 8; ++d)
                if (grads & (1 << d)) gval[d] += diff;
        }
    }


    // Average the neighbours in the surviving directions, and carry the result
    // across as a COLOUR DIFFERENCE from the centre.
    //
    // dcraw's own arithmetic:
    //     thold = gmin + (gmax >> 1);
    //     ... sum[c] over directions with gval[g] <= thold ...
    //     t = pix[color];  if (c != color) t += (sum[c] - sum[color]) / num;
    //
    // sum[c] and sum[color] accumulate over the SAME set of directions, and
    // every direction offers all three channels because the image has already
    // been bilinear-filled. That is what makes subtracting the two means sound
    // -- an earlier version summed over the raw mosaic, where an axis direction
    // never reaches the opposite colour and a diagonal never reaches green, so
    // the two means came from different neighbourhoods and manufactured colour.
    template <class Pix, class Sample>
    static void Refine(CfaPattern cfa, Pix P, Sample S, int x, int y, int c,
                       float k1, float k2, float floorT, float* rgb) {
        float gval[8];
        Gradients(cfa, S, x, y, gval);

        float lo = gval[0], hi = gval[0];
        for (int d = 1; d < 8; ++d) {
            lo = std::min(lo, gval[d]);
            hi = std::max(hi, gval[d]);
        }

        const float* centre = P(x, y);
        if (hi <= 0.0f) {
            // Perfectly flat: dcraw copies the pixel through untouched rather
            // than dividing by a zero threshold.
            rgb[0] = centre[0]; rgb[1] = centre[1]; rgb[2] = centre[2];
            return;
        }

        // dcraw's threshold is gmin + gmax/2, which is algebraically identical
        // to Chang's k1*lo + k2*(hi-lo) at the paper's k1 = 1.5, k2 = 0.5:
        //
        //     1.5*lo + 0.5*(hi - lo) = lo + hi/2
        //
        // The parameterised form is kept so the constants stay sweepable; the
        // defaults reproduce dcraw exactly.
        const float t = std::max(k1 * lo + k2 * (hi - lo), floorT);

        float sum[3] = {0, 0, 0};
        int   num    = 0;
        for (int d = 0; d < 8; ++d) {
            if (gval[d] > t) continue;
            const float* p = P(x + kDx[d], y + kDy[d]);
            for (int k = 0; k < 3; ++k) sum[k] += p[k];
            ++num;
        }

        if (num == 0) {
            rgb[0] = centre[0]; rgb[1] = centre[1]; rgb[2] = centre[2];
            return;
        }

        // The centre's MEASURED channel is kept exactly; the other two are
        // carried across by the local colour difference, which is what
        // preserves the sharpness the sensor actually recorded.
        for (int k = 0; k < 3; ++k)
            rgb[k] = (k == c) ? centre[c]
                              : centre[c] + (sum[k] - sum[c]) / float(num);
    }

    // Not a mosaic: copy through unchanged.
    void PassThrough(ImageView& dst, int w, int h) {
        const int ch = m_in.Channels();
        const float scale = m_in.ValueScale();
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                uint16_t* p = dst.At<uint16_t>(x, y);
                for (int c = 0; c < 3; ++c)
                    p[c] = FloatToHalf(m_in.Get(x, y, ch == 1 ? 0 : c) / scale);
                p[3] = FloatToHalf(1.0f);
            }
    }

    Param<float> m_k1{
        this, "k1", 1.5f, 0.0f, 4.0f,
        {.help = "Weight on the SMALLEST gradient when setting the threshold. "
                 "Chang's paper uses 1.5. Higher keeps more directions, so the "
                 "result is smoother and less prone to speckle.",
         .step = 0.05}};

    Param<float> m_k2{
        this, "k2", 0.5f, 0.0f, 2.0f,
        {.help = "Weight on the SPREAD between the largest and smallest "
                 "gradient. Chang's paper uses 0.5. Higher keeps more "
                 "directions across an edge, trading sharpness for smoothness.",
         .step = 0.05}};

    // NOT part of the algorithm, and off by default so the stage stays
    // conformant. It was added on the theory that shadow gradients are noise,
    // so the threshold collapses there and a dark pixel averages a few
    // randomly-chosen directions. Swept against a broken build and found to do
    // almost nothing -- 21637 -> 20180 false-colour pixels at a floor ten times
    // the shadow signal -- so the theory is unsupported. Kept because it costs
    // nothing and a lab should be able to sweep it; anything other than 0
    // departs from dcraw.
    Param<float> m_floor{
        this, "threshold_floor", 0.0f, 0.0f, 0.2f,
        {.help = "Smallest gradient threshold, as a floor under the paper's "
                 "formula. 0 is dcraw's behaviour and the only conformant "
                 "setting; raising it keeps more directions where the "
                 "gradients are small, which is an experiment rather than a "
                 "fix.",
         .step = 0.002, .softMin = 0.0, .softMax = 0.06}};

    PixelBuffer        m_in;
    std::vector<float> m_s, m_rgb;
};

REGISTER_ALGORITHM(DemosaicVng);

}  // namespace tglab
