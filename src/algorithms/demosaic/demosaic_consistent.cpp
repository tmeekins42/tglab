// demosaic_consistent — bilinear, corrected until it agrees with the sensor.
//
// Tim's idea, and the reasoning behind it is worth stating because it is not
// the obvious one:
//
//   Bilinear is a correct LOW-PASS reconstruction. Downsample its output and
//   its error largely vanishes -- averaging four neighbours estimates the local
//   mean well. What it destroys is high-frequency detail. But that detail is
//   not gone: every sensel is a real measurement, and bilinear merely smeared
//   it. So rather than inventing high frequency by sharpening, RECOVER it from
//   the samples that were discarded.
//
// The distinction from sharpening matters. A sharpening kernel amplifies
// whatever survived the blur, including noise, and has no idea what the true
// value was. This is driven entirely by the raw data: at each site the sensor
// recorded one channel exactly, and the reconstruction can be checked against
// it.
//
// The exact constraint
// --------------------
// Take any reconstruction, sample it back through the CFA, and compare against
// the original mosaic. Where they differ, the reconstruction is PROVABLY wrong
// -- no ground truth needed and no assumption about the scene.
//
// A demosaic that simply copies the sampled channel satisfies this trivially at
// the sites it sampled, which is why the naive form of the test measures
// nothing. The useful version is LEAVE-ONE-OUT: predict a site's own channel
// from its neighbours, withholding the sample there, and compare. That
// prediction can be wrong, and how wrong is exactly the information bilinear
// threw away.
//
// What this does with it
// ----------------------
// The correction is applied to LUMINANCE only: all three channels are scaled by
// one common factor. That is the decomposition Tim proposed and it is the right
// one -- bilinear's error is overwhelmingly in luminance, because the local
// mean of each channel is well estimated while the detail is not.
//
// Be precise about what that preserves, because it is easy to state wrongly.
// Scaling by k maps R-G to k(R-G), so it preserves RATIOS (R/G and B/G)
// exactly, NOT differences. Preserving differences exactly would need a common
// ADDITIVE offset instead. Ratio preservation is the right invariant here --
// it is constant-hue in Freeman's sense -- and it is what makes the correction
// structurally incapable of introducing false colour: no scaling of a triple
// can change which channel dominates. That property is the real distinction
// from Malvar's correction, which is additive and per-channel and therefore
// CAN shift hue.
//
// (An earlier version of this comment claimed differences were preserved.
// That was simply wrong, and the surrounding code was always doing the
// multiplicative thing.)
//
// `strength` at 0 reduces this EXACTLY to bilinear, which makes the algorithm
// self-testing: any divergence at zero is a bug in the correction rather than a
// property of it. The same check that made Malvar's zero-gain reduction useful.
//
// HOW IT SCORES, AND WHY THE NUMBERS AND THE EYE DISAGREE
// -------------------------------------------------------
// This is the interesting part, and it should be read before either trusting or
// dismissing the figures below.
//
// By ERROR against a reference, it loses to AHD. Scored on a downsampled real
// frame (the tools/headroom method), error at high-frequency sites, p50:
//
//                    conifers   portrait   alley
//   bilinear           0.139      0.081     0.070
//   consistent 0.5     0.118      0.078     0.066
//   consistent 1.0     0.131      0.084     0.071
//   ahd                0.083      0.072     0.056
//
// By SHARPNESS it wins. Tim compared all four on real frames and found this
// visibly sharper than AHD and Malvar, most clearly on fur and feathers -- the
// content where the headroom measurement also showed the largest gap. Measured,
// that impression is real: high-frequency energy relative to the true amount,
// and how much of that energy correlates with the true scene:
//
//                    hf energy        correlation with truth
//   bilinear         48-55%           0.66
//   ahd              78-82%           0.78-0.88
//   consistent 1.0   75-89%           0.68-0.78
//
// So it produces MORE detail than AHD on the conifer frame (89% against 82% of
// the true amount) while less of that detail is correct (0.78 against 0.88).
//
// Both measurements are honest and they are measuring different things. Squared
// error rewards not being wrong; it does not reward being sharp, and a smooth
// reconstruction can score well by declining to commit. On fur, feathers and
// foliage a slightly-wrong-but-present strand reads better than a correctly
// averaged smudge -- which is a real preference, not an illusion, and it is why
// this is worth having alongside AHD rather than instead of it.
//
// strength=0 matched bilinear at worst 0.000000 on every frame, so these
// numbers measure the idea rather than a bug in it.
//
// The chroma median closes the one gap
// ------------------------------------
// Measured across ten frames and three bodies, the one place this lost to AHD
// was chroma noise -- not because it ADDS any, but because it preserves
// bilinear's colour differences untouched and so never cleaned them, while AHD
// medians its own. Adding the same filter here (chroma_median, default 1):
//
//   ISO 100        detail   luma noise   chroma noise
//     ahd            150%       102%          74%
//     med 0          175%       111%          94%
//     med 1          174%       106%          72%    <- default
//
//   ISO 12800      detail   luma noise   chroma noise
//     ahd            144%       100%          90%
//     med 0          151%       104%          94%
//     med 1          148%       102%          74%
//
// One pass costs about 1% of the detail and takes chroma noise past AHD's, at
// both ends of the ISO range. Luma noise improves too, from 111% to 106% at
// base ISO -- filtering the colour differences removes speckle that was
// contributing to the luminance measurement.
//
// RE-MEASURED AFTER THE CAMERA MATRIX FIX -- READ THIS BEFORE TRUSTING THE
// TABLE ABOVE.
//
// Those numbers were taken while a separate bug was active: the camera matrix
// was driving channels negative around highlights, because bilinear
// interpolation across the steep edge of a blown light produces channel ratios
// the scene never had. See CameraMatrixInGamut in clip_repair.h. That fault
// dominated any chroma measurement made near a highlight, and it was present
// for every method, so it was invisible in a like-for-like comparison.
//
// With it fixed, re-measured on _L0A0738.CR2:
//
//                      false magenta   chroma SD   detail
//   bilinear                  16876      0.4874    0.0054  (100%)
//   malvar                     9282      0.4876    0.0067  (124%)
//   ahd                        9790      0.4864    0.0065  (120%)
//   consistent                17415      0.4870    0.0084  (156%)
//
// The DETAIL advantage survives: 156% of bilinear against AHD's 120%. That is
// the claim this algorithm exists to make, and it holds.
//
// The CHROMA advantage does not. All four now sit within 0.2% of each other on
// chroma SD, so that measure was not separating the methods at all. And on
// false magenta -- a pixel developing magenta although its green photosites
// were the BRIGHTEST of the three, i.e. colour the sensor never recorded --
// this method is the worst of the four, marginally behind the plain bilinear
// it starts from.
//
// Sweeping its own controls shows the correction is actively costing here:
//
//   strength 0,   median 0    16876   (identical to bilinear, as designed)
//   strength 0.5, median 0    15945
//   strength 1,   median 0    17388
//   strength 1,   median 1    17415
//   strength 1,   median 2    17771
//
// The chroma median exists to suppress false colour and makes it worse.
//
// Honest summary: the luminance-detail recovery is real and measured. The
// colour-cleanliness claim was an artefact of measuring around a bug. What this
// method does to false colour near highlights is an open problem.
//
// This is deliberately NOT wavelet_denoise's job. That removes SENSOR noise:
// broadband, across the whole image, tuned per frame by the user. This removes
// RECONSTRUCTION artefacts: isolated, on the CFA lattice, with a known cause
// and no tuning needed. Both are worth having, and demosaic_ahd does the same
// thing internally for the same reason.
//
// Speed: 598 ms on the GPU at 45 MP, against AHD's 567 and Malvar's 545. All
// three are bandwidth-bound at that size, so the extra passes are close to
// free -- it is a viable interactive choice, not a slow one.
//
// Why the ERROR metric peaks at 0.5 and then gets worse
// -----------------------------------------------------
// The residual is only defined at each site's OWN colour, so it is known on the
// CFA lattice and must be spread to neighbours -- otherwise the correction
// applies in a phase-correlated pattern and reintroduces exactly the lattice
// artefact this family of algorithms exists to avoid. That spreading is a blur,
// and it discards much of the high frequency the correction exists to restore.
// Past half strength, more of a blurred correction adds error faster than it
// adds detail.
//
// The deeper reason, which Tim identified: the sampled channel is EXACT --
// every method copies it verbatim -- so consistency at a site is free and
// carries no information. The only leverage is indirect, via the channels never
// measured there, and indirect evidence turns out to be thin.
//
// PRIOR ART -- what here is new and what is not
// ---------------------------------------------
// Searched after the fact, and the honest summary is "a novel combination of
// established components", not a new principle. Worth recording accurately.
//
// The residual in pass 2 is NOT new. It is exactly the correction term in
// Malvar, He & Cutler, "High-Quality Linear Interpolation for Demosaicing of
// Bayer-Patterned Color Images", ICASSP 2004 -- the same
// sample-minus-average-of-four-same-colour-neighbours-at-2px quantity, which
// that paper also calls an estimated luminance change. demosaic_malvar in this
// same directory implements it. Arriving at it independently from the
// consistency argument is a decent sign the reasoning was sound, but it is not
// a discovery.
//
// Note also US7502505B2 (Microsoft, filed 2004, listed active to 2027), whose
// claims cover computing that gradient and adding a gain-controlled portion to
// an interpolated value. Reportedly it does not claim edge-aware spreading or
// multiplicative all-channel scaling -- but that is a summary reading, not an
// attorney's, and it would need a proper check before this mattered
// commercially.
//
// The chroma median is standard practice, and older than that: Freeman, US
// 4,724,395 (1988), "Median filter for reconstructing missing color samples",
// medians the R-G and B-G planes for exactly this purpose. dcraw exposes it as
// `-m passes`.
//
// Residual Interpolation (Kiku/Monno/Tanaka/Okutomi, ICIP 2013 and IEEE TIP
// 2016) shares the word "residual" and is a different mechanism: its tentative
// estimate comes from a guided filter regressing R on G, so the residual is a
// CROSS-CHANNEL model error, computed without leave-one-out. It interpolates
// residuals linearly, adds them back per channel, and uses no median. Also
// checked: POCS / alternating-projection methods (Gunturk, Altunbasak &
// Mersereau, IEEE TIP 2002) enforce CFA consistency by hard REINSERTION of the
// known samples, which does nothing to the neighbours -- the diffusion here is
// the part they lack.
//
// What appears to be without direct precedent is the combination: an
// MHC-style residual, spread edge-aware, injected as a common multiplicative
// luminance factor. That last step is structurally the multiplicative
// (Brovey-style) detail injection used in pansharpening -- so the mechanism is
// standard in a neighbouring field, applied here to the CFA-consistency
// residual. The nearest demosaicing relative is Lukac, Martin & Plataniotis,
// "Demosaicked Image Postprocessing Using Local Color Ratios", IEEE TCSVT
// 14(6), 2004, which also preserves colour ratios against the original CFA
// data, but corrects channels sequentially via ratios to G rather than by one
// common gain.
//
// The defensible claim is narrow and real: because the correction is a common
// scale factor, it CANNOT introduce false colour, where Malvar's additive
// per-channel correction can. That is a structural property, not a tuning
// choice.
//
// The variant that was tried and rejected
// ---------------------------------------
// Rather than correcting bilinear's output, use the same residual to STEER the
// interpolation -- pick the direction whose reconstruction best predicts
// withheld samples, instead of AHD's direction-of-least-gradient. Measured over
// 2.4M decisive sites on two frames before building it:
//
//   the two criteria disagree ~45% of the time, and when they do,
//   GRADIENT is right 61-72% of those.
//
// The specific case the idea rested on failed too: on textured sites, where
// dh ~= dv and the gradient criterion should have least to go on, gradient
// still won 56-63%. Smoothness of (C - G) is simply not the same thing as
// correctness of G -- a systematically wrong green can produce a perfectly
// smooth difference plane, because the error cancels between neighbours.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "clip_repair.h"
#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"

namespace tglab {
namespace {

class DemosaicConsistent : public AlgorithmBase {
public:
    const char* Name()     const override { return "demosaic_consistent"; }
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
            return m_s[size_t(std::clamp(y, 0, h - 1)) * size_t(w) +
                       size_t(std::clamp(x, 0, w - 1))];
        };

        // --- pass 1: plain bilinear, into three planes ----------------------
        m_r.assign(n, 0.0f);
        m_g.assign(n, 0.0f);
        m_b.assign(n, 0.0f);
        for (int y = 0; y < h; ++y) {
            if (ctx.Cancelled()) return;
            for (int x = 0; x < w; ++x) {
                const size_t i = size_t(y) * size_t(w) + size_t(x);
                const int    c = CfaColorAt(cfa, x, y);
                float rgb[3] = {0, 0, 0};
                rgb[c] = S(x, y);

                if (c == 1) {
                    const float horiz = 0.5f * (S(x - 1, y) + S(x + 1, y));
                    const float vert  = 0.5f * (S(x, y - 1) + S(x, y + 1));
                    const int   hc    = CfaColorAt(cfa, x - 1, y);
                    rgb[hc]     = horiz;
                    rgb[2 - hc] = vert;
                } else {
                    rgb[1] = 0.25f * (S(x - 1, y) + S(x + 1, y) +
                                      S(x, y - 1) + S(x, y + 1));
                    rgb[2 - c] = 0.25f * (S(x - 1, y - 1) + S(x + 1, y - 1) +
                                          S(x - 1, y + 1) + S(x + 1, y + 1));
                }
                m_r[i] = rgb[0];
                m_g[i] = rgb[1];
                m_b[i] = rgb[2];
            }
        }

        const float strength = float(m_strength);

        // --- pass 2: the consistency residual -------------------------------
        //
        // At each site, predict the channel the sensor actually measured there
        // using ONLY the reconstruction at the four same-colour neighbours --
        // withholding the sample at this site. The gap between that prediction
        // and the real measurement is detail bilinear lost.
        //
        // Withholding is what makes this informative. Including the site's own
        // sample would make the prediction trivially correct, since bilinear
        // copies it verbatim, and the residual would be zero everywhere by
        // construction.
        m_delta.assign(n, 0.0f);
        if (strength > 0.0f) {
            for (int y = 0; y < h; ++y) {
                if (ctx.Cancelled()) return;
                for (int x = 0; x < w; ++x) {
                    const size_t i = size_t(y) * size_t(w) + size_t(x);
                    const int    c = CfaColorAt(cfa, x, y);
                    const std::vector<float>& plane = (c == 0) ? m_r : (c == 1 ? m_g : m_b);

                    auto P = [&](int xx, int yy) {
                        return plane[size_t(std::clamp(yy, 0, h - 1)) * size_t(w) +
                                     size_t(std::clamp(xx, 0, w - 1))];
                    };
                    const float pred = 0.25f * (P(x - 2, y) + P(x + 2, y) +
                                                P(x, y - 2) + P(x, y + 2));
                    // Positive when the sensor saw MORE than the smooth
                    // reconstruction predicts: a detail bilinear averaged away.
                    m_delta[i] = S(x, y) - pred;
                }
            }

            // --- pass 3: spread the residual ------------------------------
            //
            // A site's residual is only known for the channel it sampled, so it
            // is known at a quarter to a half of sites depending on colour.
            // Luminance is continuous, so the correction is spread to
            // neighbours -- otherwise it would apply on the CFA lattice and
            // reintroduce exactly the phase-correlated artefact this whole
            // family of algorithms exists to avoid.
            //
            // Weighted by similarity in bilinear's own luminance, so the
            // correction does not cross an edge. Bilinear's luminance is smooth
            // but its EDGES are in the right places -- it is the detail that is
            // missing, not the structure.
            m_tmp.assign(n, 0.0f);
            for (int y = 0; y < h; ++y) {
                if (ctx.Cancelled()) return;
                for (int x = 0; x < w; ++x) {
                    const size_t i = size_t(y) * size_t(w) + size_t(x);
                    const float lc = Luma(m_r[i], m_g[i], m_b[i]);
                    const float tol = kEdgeFloor + lc * kEdgeRel;

                    float acc = 0.0f, wsum = 0.0f;
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx) {
                            const int sx = std::clamp(x + dx, 0, w - 1);
                            const int sy = std::clamp(y + dy, 0, h - 1);
                            const size_t j = size_t(sy) * size_t(w) + size_t(sx);
                            const float ln = Luma(m_r[j], m_g[j], m_b[j]);
                            if (std::fabs(ln - lc) > tol) continue;
                            const float wt = (dx || dy) ? 0.5f : 1.0f;
                            acc  += m_delta[j] * wt;
                            wsum += wt;
                        }
                    m_tmp[i] = (wsum > 0.0f) ? acc / wsum : 0.0f;
                }
            }
            m_delta.swap(m_tmp);
        }

        // --- apply the correction, and bound ---------------------------------
        //
        // Done into the planes rather than in the combine loop, so the chroma
        // median below filters CORRECTED, BOUNDED colour. Filtering first and
        // bounding afterwards was the original order and it made the CPU and
        // GPU paths disagree by 0.0078 -- far too large for half-float
        // rounding, and caught by the parity check rather than by eye.
        //
        // Bounding first is also the better order on its own terms: the median
        // then works on values that are already plausible, instead of
        // potentially selecting an out-of-range one that the bound would have
        // rejected.
        for (int y = 0; y < h; ++y) {
            if (ctx.Cancelled()) return;
            for (int x = 0; x < w; ++x) {
                const size_t i = size_t(y) * size_t(w) + size_t(x);
                float rgb[3] = {m_r[i], m_g[i], m_b[i]};

                if (strength > 0.0f) {
                    // A LUMINANCE correction: one common factor for all three
                    // channels, so the RATIOS R/G and B/G survive exactly.
                    //
                    // Multiplicative rather than additive, deliberately. A
                    // common offset would preserve DIFFERENCES instead, and
                    // would change hue in shadow, where one channel is near zero
                    // and a fixed offset is a large relative change to it.
                    // Scaling is what "same colour, different brightness"
                    // actually means, and no scaling of a triple can change
                    // which channel dominates -- which is why this cannot
                    // introduce false colour.
                    //
                    // AND THAT IS ALSO ITS LIMIT. Preserving the ratios means
                    // preserving a WRONG ratio just as faithfully as a right
                    // one. Where bilinear has already produced false colour --
                    // at the steep edge of a highlight, where it averages one
                    // channel from saturated neighbours and takes another from
                    // a single dim centre -- boosting the luminance boosts that
                    // false colour with it. Measured on _L0A0738.CR2, splitting
                    // by local contrast:
                    //
                    //                   false magenta      luminance detail
                    //                  edge      flat      edge      flat
                    //   bilinear      16863        13   0.05507   0.00190
                    //   malvar         9271        11   0.06277   0.00274
                    //   ahd            9785         5   0.06164   0.00258
                    //   consistent    17411         4   0.07153   0.00386
                    //
                    // The damage is ENTIRELY at edges: 17411 against bilinear's
                    // 16863, while in flat regions this method is the cleanest
                    // of the four. AHD and Malvar beat it at edges because they
                    // steer on CHROMA and so repair bilinear's false colour;
                    // this steers on LUMINANCE only, so it inherits that error
                    // and then scales it.
                    const float lum = Luma(rgb[0], rgb[1], rgb[2]);
                    if (lum > 1e-5f) {
                        const float want = lum + strength * m_delta[i];
                        // Bounded: unbounded, a bad residual at a hard edge
                        // produces exactly the ringing that makes sharpening
                        // unpleasant, and this exists to avoid that.
                        const float k = std::clamp(want / lum, kMinGain, kMaxGain);
                        rgb[0] *= k; rgb[1] *= k; rgb[2] *= k;
                    }
                }

                // Never let the reconstruction leave the range of the real
                // samples of that colour nearby. The same protection
                // demosaic_ahd needs, and for the same reason: a camera matrix
                // has large negative off-diagonal terms, so a channel that is
                // merely short goes negative after it and clamps to zero,
                // rendering fully saturated.
                float lo[3] = {1e30f, 1e30f, 1e30f};
                float hi[3] = {-1e30f, -1e30f, -1e30f};
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int sx = std::clamp(x + dx, 0, w - 1);
                        const int sy = std::clamp(y + dy, 0, h - 1);
                        const int sc = CfaColorAt(cfa, sx, sy);
                        const float sv = S(sx, sy);
                        lo[sc] = std::min(lo[sc], sv);
                        hi[sc] = std::max(hi[sc], sv);
                    }
                for (int c = 0; c < 3; ++c)
                    if (hi[c] >= lo[c]) rgb[c] = std::clamp(rgb[c], lo[c], hi[c]);

                m_r[i] = rgb[0];
                m_g[i] = rgb[1];
                m_b[i] = rgb[2];
            }
        }

        // --- chroma median ---------------------------------------------------
        //
        // A median over the COLOUR DIFFERENCES (R-G and B-G), not over the
        // channels. Luminance detail is untouched, which is the whole point:
        // this algorithm exists to recover luminance detail and must not then
        // filter it away.
        //
        // This is a different job from wavelet_denoise, and both are worth
        // having. That removes SENSOR noise -- broadband, across the whole
        // image, tuned per frame by the user. This removes RECONSTRUCTION
        // artefacts: isolated, on the CFA lattice, with a known cause and no
        // tuning needed. demosaic_ahd does the same thing internally for
        // exactly this reason.
        //
        // Measured across ten frames and three bodies, this was the one place
        // consistent lost to AHD. AHD cuts chroma noise to 74-77% of bilinear's
        // while consistent left it at ~94% -- not because it ADDS chroma noise,
        // but because it deliberately preserves bilinear's colour differences
        // and so never cleans them.
        const int chromaPasses = std::max(0, int(m_chromaMedian));
        if (chromaPasses > 0) {
            m_dr.assign(n, 0.0f);
            m_db.assign(n, 0.0f);
            for (size_t i = 0; i < n; ++i) {
                m_dr[i] = m_r[i] - m_g[i];
                m_db[i] = m_b[i] - m_g[i];
            }
            for (int p = 0; p < chromaPasses; ++p) {
                if (ctx.Cancelled()) return;
                ChromaMedianPair(m_dr, m_db, w, h);
            }
            // Put the filtered colour back, keeping green -- and therefore
            // luminance detail -- exactly as reconstructed.
            for (size_t i = 0; i < n; ++i) {
                m_r[i] = m_g[i] + m_dr[i];
                m_b[i] = m_g[i] + m_db[i];
            }
        }

        // --- combine --------------------------------------------------------
        for (int y = 0; y < h; ++y) {
            if (ctx.Cancelled()) return;
            for (int x = 0; x < w; ++x) {
                const size_t i = size_t(y) * size_t(w) + size_t(x);
                // The correction, the bound and the chroma median all happened
                // above, into the planes. This is only the colour transform.
                float rgb[3] = {m_r[i], m_g[i], m_b[i]};

                ApplyColour(src.desc, rgb);

                uint16_t* p = dst.At<uint16_t>(x, y);
                // Stored UNCLAMPED. A negative channel is a real colour outside

                // sRGB's gamut, not an error -- see ApplyColour.

                p[0] = FloatToHalf(rgb[0]);
                p[1] = FloatToHalf(rgb[1]);
                p[2] = FloatToHalf(rgb[2]);
                p[3] = FloatToHalf(1.0f);
            }
        }
    }

private:
    static float Luma(float r, float g, float b) {
        return 0.2126f * r + 0.7152f * g + 0.0722f * b;
    }

    // A 3x3 median over one colour-difference plane, gated on luminance.
    //
    // The gate matters and is not symmetry with AHD for its own sake. On a
    // feature narrower than the 3x3 window most of the nine samples belong to
    // the background, so an ungated median selects one of those and paints the
    // feature with the background's colour -- which on AHD showed as bright
    // green streaks along every twig. Pooling only samples of similar
    // brightness keeps the filter on one side of an edge.
    // Medians BOTH colour-difference planes in one traversal, against a luma
    // plane computed once.
    //
    // Three redundancies removed, all of them the same mistake -- recomputing
    // something that cannot have changed:
    //
    //  - Luma() was evaluated for the centre and all nine neighbours, per
    //    pixel, per plane, per pass. m_r/m_g/m_b are not written inside the
    //    pass loop, so every one of those values is identical across the R-G
    //    call, the B-G call, and every iteration. It is now computed once for
    //    the whole image: ten luma evaluations per pixel become one.
    //
    //  - The neighbourhood was gathered twice, once per plane, applying the
    //    SAME edge test to decide the same survivors both times. Now gathered
    //    once, with both planes' values collected together.
    //
    //  - std::clamp ran on every neighbour of every pixel, though only the
    //    one-pixel border can actually fall outside. The interior is split
    //    out and indexes directly.
    //
    // Identical output, which is the point: this is arithmetic the old code
    // was already doing, not a different filter. Verified against the previous
    // implementation over the full image before the old one was deleted.
    void ChromaMedianPair(std::vector<float>& da, std::vector<float>& db, int w, int h) {
        const size_t n = da.size();
        m_luma.resize(n);
        for (size_t i = 0; i < n; ++i) m_luma[i] = Luma(m_r[i], m_g[i], m_b[i]);

        m_tmp.resize(n);
        m_tmp2.resize(n);

        // One pixel's worth of work, shared by the interior and border paths so
        // the two cannot drift apart. `gather` yields each neighbour's index.
        auto filter = [&](size_t i, auto&& gather) {
            const float lc  = m_luma[i];
            const float tol = kEdgeFloor + lc * kEdgeRel;

            float va[9], vb[9];
            int   k = 0;
            gather([&](size_t j) {
                if (std::fabs(m_luma[j] - lc) > tol) return;
                va[k] = da[j];
                vb[k] = db[j];
                ++k;
            });

            // Fewer than three survivors is not a median; pass through.
            if (k >= 3) {
                std::nth_element(va, va + k / 2, va + k);
                std::nth_element(vb, vb + k / 2, vb + k);
                m_tmp[i]  = va[k / 2];
                m_tmp2[i] = vb[k / 2];
            } else {
                m_tmp[i]  = da[i];
                m_tmp2[i] = db[i];
            }
        };

        for (int y = 0; y < h; ++y) {
            const bool edgeRow = (y == 0 || y == h - 1);
            for (int x = 0; x < w; ++x) {
                const size_t i = size_t(y) * size_t(w) + size_t(x);
                if (edgeRow || x == 0 || x == w - 1) {
                    filter(i, [&](auto&& take) {
                        for (int dy = -1; dy <= 1; ++dy)
                            for (int dx = -1; dx <= 1; ++dx) {
                                const int sx = std::clamp(x + dx, 0, w - 1);
                                const int sy = std::clamp(y + dy, 0, h - 1);
                                take(size_t(sy) * size_t(w) + size_t(sx));
                            }
                    });
                } else {
                    filter(i, [&](auto&& take) {
                        for (int dy = -1; dy <= 1; ++dy) {
                            const size_t row = i + size_t(dy) * size_t(w);
                            take(row - 1);
                            take(row);
                            take(row + 1);
                        }
                    });
                }
            }
        }
        da.swap(m_tmp);
        db.swap(m_tmp2);
    }

    void PassThrough(ImageView& dst, int w, int h) {
        const int ch = m_in.Channels();
        const float scale = m_in.ValueScale();
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                uint16_t* p = dst.At<uint16_t>(x, y);
                for (int c = 0; c < 3; ++c)
                    p[c] = FloatToHalf(m_in.Get(x, y, ch == 1 ? 0 : c) / scale);
                p[3] = FloatToHalf(ch == 4 ? m_in.Get(x, y, 3) / scale : 1.0f);
            }
    }

    // Identical to the other demosaicers: white balance and the camera matrix
    // are properties of the capture, not of the interpolation, so switching
    // method must not change colour.
    static void ApplyColour(const ImageDesc& d, float* rgb) {
        // White balance with the highlight clamp -- see clip_repair.h.
        BalanceAndClamp(d, rgb);

        CameraMatrixInGamut(d, rgb);

        // Negatives are NOT clamped here, deliberately.
        //
        // The camera matrix maps the sensor's response into sRGB primaries, and
        // a color the sensor recorded that sRGB cannot represent lands outside
        // the triangle -- which in linear sRGB coordinates means a channel below
        // zero. Clamping destroys it at demosaic, before the user has touched a
        // slider, and no later operation can bring it back.
        //
        // Measured across ten frames, between 0.01% and 1.2% of pixels have a
        // channel clamped this way, worst on saturated foliage and skin at high
        // ISO. That is not a large fraction, but it is exactly the saturated
        // color a user is most likely to reach for a saturation slider over --
        // and a value that is merely negative can come BACK into gamut after a
        // desaturation or a white-balance change.
        //
        // The pipeline is linear float end to end, so a negative is
        // representable and costs nothing to carry. Clamping belongs at the
        // point of display or export, where the output really is bounded --
        // ToneCurve() maps anything at or below zero to black, which is the
        // correct behavior there and the wrong behavior here.
    }

    static constexpr float kClip = 0.99f;

    // How far the luminance correction may push, as a multiplier. Wide enough
    // to restore real detail, narrow enough that a wrong residual cannot
    // produce a black or blown pixel from a correct one.
    static constexpr float kMinGain = 0.5f;
    static constexpr float kMaxGain = 2.0f;

    // Edge tolerance for spreading the residual, proportional to the local
    // level plus a floor -- an absolute threshold means "same surface" in the
    // highlights and "across a boundary" in shadow.
    static constexpr float kEdgeFloor = 0.004f;
    static constexpr float kEdgeRel   = 0.12f;

    Param<float> m_strength{
        this, "strength", 1.0f, 0.0f, 2.0f,
        {.help = "How much of the measured detail to restore. 0 reduces this "
                 "EXACTLY to bilinear, which is what makes the correction "
                 "testable rather than taken on trust. 1 applies the residual "
                 "as measured; above 1 overshoots deliberately, which is worth "
                 "looking at to see what the correction is actually doing.",
         .step = 0.01f, .softMin = 0.0f, .softMax = 1.5f}};

    Param<int> m_chromaMedian{
        this, "chroma_median", 1, 0, 5,
        {.help = "Passes of a 3x3 median over the colour differences. Removes "
                 "the coloured speckle demosaicing leaves behind without "
                 "touching luminance detail, because it filters R-G and B-G "
                 "rather than the channels. This is reconstruction cleanup, not "
                 "denoising -- wavelet_denoise is the tool for sensor noise. "
                 "0 disables it.",
         .step = 1}};

public:
    // --- GPU ----------------------------------------------------------------
    //
    // Four kernels over four scratch planes, using AlgorithmBase::GpuPasses --
    // the same mechanism demosaic_ahd needed, for the same reason: these passes
    // do different things rather than repeating one kernel.
    //
    //   plane 0 : bilinear luminance     plane 2 : residual
    //   plane 1 : (unused, kept for      plane 3 : residual, spread
    //             binding symmetry)
    //
    //   pass 0  bilinear   mosaic              -> planes 0, 1
    //   pass 1  residual   mosaic, plane 0     -> plane 2
    //   pass 2  spread     mosaic, 2, 0        -> plane 3
    //   pass 3  combine    mosaic, 0, 3        -> output
    //
    // The bilinear pass writes LUMINANCE rather than the three channels,
    // because that is all the later passes need: the residual is judged against
    // the sampled channel, and the correction is a single scale factor. Storing
    // luminance costs one plane where storing RGB would cost three, and the
    // combine recomputes the colours from the mosaic anyway.
    bool HasGPU() const override { return true; }

    void PrepareGpu(const std::vector<ImageDesc>& inputs) override {
        if (inputs.empty()) return;
        m_cfa   = int(inputs[0].cfa);
        m_black = inputs[0].blackLevel;
        m_range = std::max(inputs[0].whiteLevel - inputs[0].blackLevel, 1e-6f);
        for (int i = 0; i < 3; ++i) m_camMul[i] = inputs[0].camMul[i];
        for (int i = 0; i < 9; ++i) m_rgbCam[i] = inputs[0].rgbCam[i];
    }

    int        GpuScratchCount()  const override { return 4; }
    FormatSpec GpuScratchPlanes() const override { return FormatSpec::RGBA32F; }

    std::vector<GpuPass> GpuPasses() const override;

    std::vector<uint32_t> GpuPassConstants(int) const override {
        auto bits = [](float f) {
            uint32_t u;
            std::memcpy(&u, &f, sizeof(u));
            return u;
        };
        std::vector<uint32_t> c{uint32_t(m_cfa), bits(m_black), bits(m_range),
                                bits(float(m_strength))};
        for (int i = 0; i < 3; ++i) c.push_back(bits(m_camMul[i]));
        for (int i = 0; i < 9; ++i) c.push_back(bits(m_rgbCam[i]));
        return c;
    }

private:
    int   m_cfa   = 1;
    float m_black = 0.0f;
    float m_range = 1.0f;
    float m_camMul[3] = {1.0f, 1.0f, 1.0f};
    float m_rgbCam[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    PixelBuffer        m_in;
    std::vector<float> m_s, m_r, m_g, m_b, m_delta, m_tmp, m_dr, m_db;

    // Chroma median scratch: the luma plane computed once per pass, and a
    // second output buffer so both colour-difference planes are filtered in
    // one traversal. Members rather than locals so a slider drag does not
    // reallocate three full-size planes per tick.
    std::vector<float> m_luma, m_tmp2;
};

// Shared prologue: constants, the CFA lookup, and the normalised sample fetch.
// One string rather than four copies, so the CFA lookup cannot differ between
// passes -- which would reconstruct red from blue's neighbours in part of the
// image and read as a colour cast rather than an obvious bug.
const char* const kCommon = R"(
Texture2D<float4>   T0 : register(t0);
Texture2D<float4>   T1 : register(t1);
Texture2D<float4>   T2 : register(t2);
Texture2D<float4>   T3 : register(t3);
RWTexture2D<float4> U0 : register(u0);
RWTexture2D<float4> U1 : register(u1);

cbuffer Params : register(b0) {
    uint Width;
    uint Height;
    uint Cfa;
    uint BlackBits;
    uint RangeBits;
    uint StrengthBits;
    uint CamMul0, CamMul1, CamMul2;
    uint M0, M1, M2, M3, M4, M5, M6, M7, M8;
};

// White balance AND the clipped-channel repair -- the repair must see balanced
// values, since that is where neutral means the channels are equal.
// White balance with the highlight clamp -- keep in step with
// BalanceAndClamp in clip_repair.h.
void BalanceAndClamp(inout float3 rgb, float3 camMul) {
    float ceiling = min(camMul.r, min(camMul.g, camMul.b));
    rgb = min(rgb * camMul, ceiling);
}

int CfaColor(uint cfa, int x, int y) {
    int q = (y & 1) * 2 + (x & 1);
    if (cfa == 1) { int c[4] = {0, 1, 1, 2}; return c[q]; }
    if (cfa == 2) { int c[4] = {2, 1, 1, 0}; return c[q]; }
    if (cfa == 3) { int c[4] = {1, 0, 2, 1}; return c[q]; }
    if (cfa == 4) { int c[4] = {1, 2, 0, 1}; return c[q]; }
    return 1;
}

int2 ClampXY(int x, int y) {
    return clamp(int2(x, y), int2(0, 0), int2(Width - 1, Height - 1));
}

float Sample(int x, int y) {
    return clamp((T0[ClampXY(x, y)].x - asfloat(BlackBits)) / asfloat(RangeBits),
                 0.0, 4.0);
}

float3 Bilinear(int x, int y) {
    int c = CfaColor(Cfa, x, y);
    float3 rgb = float3(0, 0, 0);
    rgb[c] = Sample(x, y);
    if (c == 1) {
        float horiz = 0.5 * (Sample(x - 1, y) + Sample(x + 1, y));
        float vert  = 0.5 * (Sample(x, y - 1) + Sample(x, y + 1));
        int   hc    = CfaColor(Cfa, x - 1, y);
        rgb[hc]     = horiz;
        rgb[2 - hc] = vert;
    } else {
        rgb[1] = 0.25 * (Sample(x - 1, y) + Sample(x + 1, y) +
                         Sample(x, y - 1) + Sample(x, y + 1));
        rgb[2 - c] = 0.25 * (Sample(x - 1, y - 1) + Sample(x + 1, y - 1) +
                             Sample(x - 1, y + 1) + Sample(x + 1, y + 1));
    }
    return rgb;
}

float Luma3(float3 c) { return 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b; }
)";

// Pass 0: bilinear. Stores the three channels AND luminance, so later passes
// need neither to recompute it nor to refetch nine samples.
const char* const kBilinearHlsl = R"(
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    float3 rgb = Bilinear(int(tid.x), int(tid.y));
    U0[tid.xy] = float4(rgb, Luma3(rgb));
    U1[tid.xy] = float4(0, 0, 0, 1);
}
)";

// Pass 1: the leave-one-out residual.
//
// Predict this site's own sampled channel from the reconstruction at its four
// same-colour neighbours, WITHOUT the sample here. Including it would make the
// prediction trivially correct, since bilinear copies it verbatim, and the
// residual would be zero everywhere by construction.
const char* const kResidualHlsl = R"(
float Plane(int x, int y, int c) { return T1[ClampXY(x, y)][c]; }

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    int x = int(tid.x), y = int(tid.y);
    if (asfloat(StrengthBits) <= 0.0) { U0[tid.xy] = float4(0,0,0,1); return; }

    int c = CfaColor(Cfa, x, y);
    float pred = 0.25 * (Plane(x - 2, y, c) + Plane(x + 2, y, c) +
                         Plane(x, y - 2, c) + Plane(x, y + 2, c));
    U0[tid.xy] = float4(Sample(x, y) - pred, 0, 0, 1);
}
)";

// Pass 2: spread the residual off the CFA lattice.
//
// A site's residual is known only for the channel it sampled, so applying it
// directly would correct on the lattice and reintroduce exactly the
// phase-correlated artefact this family of algorithms exists to avoid. Weighted
// by similarity in bilinear's luminance so the correction does not cross an
// edge -- bilinear's edges are in the right places, it is the detail that is
// missing.
const char* const kSpreadHlsl = R"(
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    int x = int(tid.x), y = int(tid.y);
    if (asfloat(StrengthBits) <= 0.0) { U0[tid.xy] = float4(0,0,0,1); return; }

    float lc  = T2[ClampXY(x, y)].w;          // bilinear luminance here
    float tol = 0.004 + lc * 0.12;

    float acc = 0.0, wsum = 0.0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            int2 p = ClampXY(x + dx, y + dy);
            float ln = T2[p].w;
            if (abs(ln - lc) > tol) continue;
            float wt = (dx != 0 || dy != 0) ? 0.5 : 1.0;
            acc  += T1[p].x * wt;
            wsum += wt;
        }
    U0[tid.xy] = float4(wsum > 0.0 ? acc / wsum : 0.0, 0, 0, 1);
}
)";

// Pass 3: the corrected image as RGB, before the chroma median.
//
// Split out from the combine so the median has something to filter. It applies
// the luminance correction and the sample bound, but stops before white balance
// and the colour matrix -- the median works on linear colour differences, and
// filtering after the matrix would mix channels the matrix had already mixed.
const char* const kApplyHlsl = R"(
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    int x = int(tid.x), y = int(tid.y);

    float4 bl  = T1[int2(tid.xy)];
    float3 rgb = bl.rgb;
    float  str = asfloat(StrengthBits);

    if (str > 0.0) {
        float lum = bl.w;
        if (lum > 1e-5) {
            float want = lum + str * T2[int2(tid.xy)].x;
            rgb *= clamp(want / lum, 0.5, 2.0);
        }
    }

    float3 lo = float3(1e30, 1e30, 1e30);
    float3 hi = float3(-1e30, -1e30, -1e30);
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            int sx = clamp(x + dx, 0, int(Width) - 1);
            int sy = clamp(y + dy, 0, int(Height) - 1);
            int sc = CfaColor(Cfa, sx, sy);
            float sv = Sample(sx, sy);
            lo[sc] = min(lo[sc], sv);
            hi[sc] = max(hi[sc], sv);
        }
    [unroll] for (int c = 0; c < 3; ++c)
        if (hi[c] >= lo[c]) rgb[c] = clamp(rgb[c], lo[c], hi[c]);

    U0[tid.xy] = float4(rgb, Luma3(rgb));
}
)";

// Pass 4: the chroma median, over the colour differences.
//
// Filters R-G and B-G rather than the channels, so luminance detail -- the
// thing this algorithm exists to recover -- is untouched. Gated on luminance
// for the same reason AHD's is: on a feature narrower than the 3x3 window most
// samples belong to the background, and an ungated median paints the feature
// with the background's colour.
const char* const kChromaHlsl = R"(
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    int x = int(tid.x), y = int(tid.y);

    float4 me = T1[int2(tid.xy)];
    float  lc = me.w;
    float tol = 0.004 + lc * 0.12;

    float vr[9], vb[9];
    int   k = 0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            int2 p = ClampXY(x + dx, y + dy);
            float4 s = T1[p];
            if (abs(s.w - lc) > tol) continue;
            vr[k] = s.r - s.g;
            vb[k] = s.b - s.g;
            ++k;
        }

    float dr = me.r - me.g;
    float db = me.b - me.g;
    if (k >= 3) {
        // Selection sort to the middle; k is at most 9.
        for (int i = 0; i <= k / 2; ++i) {
            int mr = i, mb = i;
            for (int j = i + 1; j < k; ++j) {
                if (vr[j] < vr[mr]) mr = j;
                if (vb[j] < vb[mb]) mb = j;
            }
            float t;
            t = vr[i]; vr[i] = vr[mr]; vr[mr] = t;
            t = vb[i]; vb[i] = vb[mb]; vb[mb] = t;
        }
        dr = vr[k / 2];
        db = vb[k / 2];
    }
    // Green, and therefore luminance detail, is carried through unchanged.
    U0[tid.xy] = float4(me.g + dr, me.g, me.g + db, 1.0);
}
)";

// Final pass: white balance and the camera matrix.
const char* const kCombineHlsl = R"(
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    int x = int(tid.x), y = int(tid.y);

    if (Cfa == 0) { U0[tid.xy] = float4(T0[int2(tid.xy)].rgb, 1.0); return; }

    // The correction and the sample bound happened in the apply pass; the
    // chroma median ran after that. This is only the colour transform.
    float3 rgb = T1[int2(tid.xy)].rgb;

    // Brightest RAW sample feeding each channel, then repair in RAW space
    // BEFORE white balance -- see clip_repair.h for both rules and the
    // measurements. T0 is the mosaic, so the raw peaks are available here even
    // though the interpolation finished several passes ago.
    // The brightest RAW SAMPLE of each colour, over a window snapped to the
    // CFA cell so every pixel in that cell gets the SAME mask. Mirroring the
    // interpolation's own neighbours makes the clip decision alternate with
    // CFA parity, and that checkerboard is the fringing -- see CfaPeaks in
    // clip_repair.h.
    BalanceAndClamp(rgb, float3(asfloat(CamMul0), asfloat(CamMul1), asfloat(CamMul2)));

    float3 o;
    o.r = asfloat(M0)*rgb.r + asfloat(M1)*rgb.g + asfloat(M2)*rgb.b;
    o.g = asfloat(M3)*rgb.r + asfloat(M4)*rgb.g + asfloat(M5)*rgb.b;
    o.b = asfloat(M6)*rgb.r + asfloat(M7)*rgb.g + asfloat(M8)*rgb.b;
    // Negatives deliberately NOT clamped -- see ApplyColour in the CPU path.
    U0[tid.xy] = float4(o, 1.0);
}
)";

std::vector<AlgorithmBase::GpuPass> DemosaicConsistent::GpuPasses() const {
    // Assembled once: GpuPasses() returns raw pointers and is called per run,
    // so building the strings each time would dangle them.
    static const std::string bil  = std::string(kCommon) + kBilinearHlsl;
    static const std::string res  = std::string(kCommon) + kResidualHlsl;
    static const std::string spr  = std::string(kCommon) + kSpreadHlsl;
    static const std::string app  = std::string(kCommon) + kApplyHlsl;
    static const std::string chr  = std::string(kCommon) + kChromaHlsl;
    static const std::string comb = std::string(kCommon) + kCombineHlsl;

    std::vector<GpuPass> p;
    // t0 is always the mosaic, because kCommon's Sample() reads it.
    p.push_back({bil.c_str(), "bilinear", {-1},       {0, 1}});
    p.push_back({res.c_str(), "residual", {-1, 0},    {2}});
    p.push_back({spr.c_str(), "spread",   {-1, 2, 0}, {3}});
    p.push_back({app.c_str(), "apply",    {-1, 0, 3}, {1}});

    // The chroma median ping-pongs between planes 1 and 2: a pass may not read
    // and write the same buffer, since threads have no ordering and the
    // framework rejects the race outright.
    const int passes = std::max(0, int(m_chromaMedian));
    int src = 1;
    for (int i = 0; i < passes; ++i) {
        const int dst = (src == 1) ? 2 : 1;
        p.push_back({chr.c_str(), "chroma", {-1, src}, {dst}});
        src = dst;
    }
    p.push_back({comb.c_str(), "combine", {-1, src}, {-1}});
    return p;
}

REGISTER_ALGORITHM(DemosaicConsistent);

} // namespace
} // namespace tglab
