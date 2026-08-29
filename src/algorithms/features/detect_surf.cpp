// detect_surf — Bay, Ess, Tuytelaars & Van Gool (2006/2008).
//
// PATENT NOTICE. SURF is covered by patents held by ETH Zurich (US 2009/0238460
// and family). SIFT's expired in 2020; SURF's have not. OpenCV ships it in the
// "non-free" xfeatures2d module for this reason. Tim's call to include it here
// anyway, for a research lab that is not being sold -- but anyone shipping a
// product built on this file should check their position first.
//
// HOW IT DIFFERS FROM SIFT, which is the reason to have both:
//
//   SIFT blurs the image repeatedly and downsamples between octaves. SURF never
//   downsamples and never blurs: it computes an INTEGRAL IMAGE once, and then
//   approximates the Gaussian second derivatives with BOX FILTERS, which cost
//   the same at any size. So instead of shrinking the image to find larger
//   features, it grows the filter -- and every filter size costs the same three
//   or four array lookups per pixel.
//
//   The determinant of the approximated Hessian is the response, rather than a
//   difference of Gaussians. It detects blobs like SIFT does but with a
//   different bias: SURF prefers rounder, higher-contrast blobs and finds fewer
//   of them, which on the same image usually means fewer features that are
//   individually a little more reliable.
//
//   The descriptor is Haar wavelet responses in a 4x4 grid: 64 floats as
//   [sum dx, sum |dx|, sum dy, sum |dy|] per cell, or 128 when `extended`
//   splits those by the sign of the other axis. That is the one place a
//   detector here genuinely offers a descriptor CHOICE, and it is exposed as an
//   ordinary parameter -- see features.h for why nothing else has to.
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "../../algo_util/features.h"
#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"

namespace tglab {
namespace {

// Summed-area table: Sum(x0,y0,x1,y1) in four lookups regardless of area.
//
// This is what makes SURF's constant-time box filters possible, and therefore
// the whole reason the algorithm is shaped the way it is. Double rather than
// float: the accumulated sum over a 45 MP image reaches values where float's
// 24-bit mantissa loses the low bits, and a box filter is a DIFFERENCE of two
// large sums -- exactly where that loss shows up as noise in the result.
struct Integral {
    std::vector<double> v;
    int w = 0, h = 0;

    void Build(const std::vector<float>& g, int width, int height) {
        w = width + 1;
        h = height + 1;
        v.assign(size_t(w) * size_t(h), 0.0);
        for (int y = 0; y < height; ++y) {
            double row = 0.0;
            for (int x = 0; x < width; ++x) {
                row += double(g[size_t(y) * size_t(width) + size_t(x)]);
                v[size_t(y + 1) * size_t(w) + size_t(x + 1)] =
                    v[size_t(y) * size_t(w) + size_t(x + 1)] + row;
            }
        }
    }

    // Sum over [x, x+bw) x [y, y+bh), clamped to the image.
    double Box(int x, int y, int bw, int bh) const {
        const int x0 = std::clamp(x,      0, w - 1);
        const int y0 = std::clamp(y,      0, h - 1);
        const int x1 = std::clamp(x + bw, 0, w - 1);
        const int y1 = std::clamp(y + bh, 0, h - 1);
        return v[size_t(y1) * size_t(w) + size_t(x1)] -
               v[size_t(y0) * size_t(w) + size_t(x1)] -
               v[size_t(y1) * size_t(w) + size_t(x0)] +
               v[size_t(y0) * size_t(w) + size_t(x0)];
    }
};

// One response layer: the Hessian determinant at one filter size.
struct Layer {
    std::vector<float> det;
    std::vector<int>   lap;      // sign of the trace, for the matching shortcut
    int w = 0, h = 0, size = 0, step = 0;

    float Det(int x, int y) const {
        return det[size_t(std::clamp(y, 0, h - 1)) * size_t(w) +
                   size_t(std::clamp(x, 0, w - 1))];
    }
};

class DetectSurf : public AlgorithmBase {
public:
    const char* Name()     const override { return "detect_surf"; }
    const char* Category() const override { return "features"; }

    PortList Inputs()  const override { return {{"src", DataType::Image, FormatSpec::Any}}; }
    PortList Outputs() const override {
        return {{"out", DataType::Image, FormatSpec::SameAsInput}};
    }

    void RunCPU(RunCtx& ctx) override {
        const ImageView src = ctx.In(0);
        ImageView       dst = ctx.Out(0);
        if (!src.Valid() || !dst.Valid()) return;

        PixelBuffer in;
        in.Unpack(src);
        if (!in.Valid()) return;

        const int w = in.Width(), h = in.Height(), ch = in.Channels();
        const float scale = in.ValueScale();

        PixelBuffer out;
        out.Unpack(dst);
        if (out.Valid()) {
            out.Data() = in.Data();
            out.PackInto(dst);
        }

        m_found = 0;
        if (w < 32 || h < 32) return;

        std::vector<float> grey(size_t(w) * size_t(h));
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const float* p = in.At(x, y);
                grey[size_t(y) * size_t(w) + size_t(x)] = (ch >= 3)
                    ? (0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2]) / scale
                    : p[0] / scale;
            }

        Integral ii;
        ii.Build(grey, w, h);

        auto sidecar = std::make_shared<FeatureSidecar>();
        sidecar->detector = "surf";
        sidecar->descriptors.kind = DescriptorKind::Float;
        sidecar->descriptors.dim  = bool(m_extended) ? 128 : 64;

        Detect(ii, w, h, ctx, sidecar.get());
        m_found = int(sidecar->keypoints.size());

        if (Image* im = ctx.OutImage(0)) im->Sidecars().Set(kFeatureSidecar, sidecar);
    }

    std::string RunReport() const override {
        if (m_found <= 0) return {};
        return std::to_string(m_found) + " SURF features (" +
               std::to_string(bool(m_extended) ? 128 : 64) + "f)";
    }

    bool HasGPU() const override { return false; }

private:
    // Approximated Gaussian second derivatives, as box filters.
    //
    // The weights and proportions are Bay's: a 9x9 filter approximates sigma
    // 1.2, and the 0.9 on the cross term compensates for the box filter being a
    // poor approximation to the Gaussian -- without it the determinant is
    // biased and the detector prefers the wrong blobs.
    static float Response(const Integral& ii, int x, int y, int size, int* laplacian) {
        const int lobe = size / 3;             // 3 for a 9x9 filter
        const int b    = size / 2;
        const float norm = 1.0f / float(size * size);   // scale invariance

        // Dxx: three horizontal lobes, the middle one negative and doubled.
        const double dxxAll = ii.Box(x - b, y - lobe / 2 - lobe / 2, size, 2 * lobe - 1);
        const double dxxMid = ii.Box(x - lobe / 2, y - lobe + 1, lobe, 2 * lobe - 1);
        const float dxx = float(dxxAll - 3.0 * dxxMid) * norm;

        // Dyy: the same rotated.
        const double dyyAll = ii.Box(x - lobe / 2 - lobe / 2, y - b, 2 * lobe - 1, size);
        const double dyyMid = ii.Box(x - lobe + 1, y - lobe / 2, 2 * lobe - 1, lobe);
        const float dyy = float(dyyAll - 3.0 * dyyMid) * norm;

        // Dxy: four square lobes in a pinwheel.
        const double a1 = ii.Box(x - lobe,     y - lobe,     lobe, lobe);
        const double a2 = ii.Box(x + 1,        y - lobe,     lobe, lobe);
        const double a3 = ii.Box(x - lobe,     y + 1,        lobe, lobe);
        const double a4 = ii.Box(x + 1,        y + 1,        lobe, lobe);
        const float dxy = float(a1 - a2 - a3 + a4) * norm;

        if (laplacian) *laplacian = (dxx + dyy >= 0.0f) ? 1 : -1;
        return dxx * dyy - 0.81f * dxy * dxy;
    }

    void Detect(const Integral& ii, int w, int h, RunCtx& ctx, FeatureSidecar* out) {
        const int octaves = std::clamp(int(m_octaves), 1, 4);
        const int perOct  = std::clamp(int(m_scales), 1, 4);
        const int maxFeat = std::max(1, int(m_maxFeatures));

        // Filter sizes. SURF grows the FILTER rather than shrinking the image,
        // so an octave is a doubling of the step between sizes rather than a
        // halving of resolution -- which is why nothing here downsamples.
        std::vector<Layer> layers;
        for (int o = 0; o < octaves; ++o) {
            const int step = 1 << o;
            for (int i = 0; i < perOct + 2; ++i) {
                const int size = 9 + 6 * step * i;
                if (size > std::min(w, h) / 2) break;

                Layer L;
                L.w = w / step;
                L.h = h / step;
                L.size = size;
                L.step = step;
                if (L.w < 8 || L.h < 8) break;

                L.det.assign(size_t(L.w) * size_t(L.h), 0.0f);
                L.lap.assign(size_t(L.w) * size_t(L.h), 0);
                for (int y = 0; y < L.h; ++y) {
                    if (ctx.Cancelled()) return;
                    for (int x = 0; x < L.w; ++x) {
                        int lap = 0;
                        L.det[size_t(y) * size_t(L.w) + size_t(x)] =
                            Response(ii, x * step, y * step, size, &lap);
                        L.lap[size_t(y) * size_t(L.w) + size_t(x)] = lap;
                    }
                }
                layers.push_back(std::move(L));
            }
        }

        // Extrema across three consecutive layers of the SAME step, which is
        // what makes the comparison meaningful: layers at different steps have
        // different sampling and their responses are not directly comparable.
        const float thresh = float(m_threshold);

        for (size_t i = 1; i + 1 < layers.size(); ++i) {
            const Layer& below = layers[i - 1];
            const Layer& here  = layers[i];
            const Layer& above = layers[i + 1];
            if (below.step != here.step || above.step != here.step) continue;
            if (ctx.Cancelled()) return;

            for (int y = 1; y < here.h - 1; ++y)
                for (int x = 1; x < here.w - 1; ++x) {
                    const float v = here.Det(x, y);
                    if (v < thresh) continue;

                    bool isMax = true;
                    for (int dy = -1; dy <= 1 && isMax; ++dy)
                        for (int dx = -1; dx <= 1; ++dx) {
                            if (below.Det(x + dx, y + dy) >= v) { isMax = false; break; }
                            if (above.Det(x + dx, y + dy) >= v) { isMax = false; break; }
                            if ((dx || dy) && here.Det(x + dx, y + dy) >= v) {
                                isMax = false; break;
                            }
                        }
                    if (!isMax) continue;

                    // Sub-pixel by the same quadratic fit SIFT uses. Without it
                    // positions quantise to the layer's step, which at step 4
                    // is a 4-pixel grid -- far too coarse for alignment.
                    float ox = 0.0f, oy = 0.0f, os = 0.0f;
                    if (!Interpolate(below, here, above, x, y, &ox, &oy, &os)) continue;

                    if (int(out->keypoints.size()) >= maxFeat) return;

                    Keypoint kp;
                    kp.x = (float(x) + ox) * float(here.step);
                    kp.y = (float(y) + oy) * float(here.step);
                    // Filter size to sigma: Bay's 9x9 filter approximates
                    // sigma 1.2, and the relationship is linear in size.
                    const float filt = float(here.size) +
                                       os * float(here.size - below.size);
                    kp.scale = 1.2f * filt / 9.0f;
                    kp.response = v;
                    kp.octave = int(std::log2(double(here.step)));

                    kp.angle = m_upright ? 0.0f : Orientation(ii, kp);
                    out->keypoints.push_back(kp);

                    const int dim = bool(m_extended) ? 128 : 64;
                    std::vector<float> desc(size_t(dim), 0.0f);
                    Describe(ii, kp, desc.data());
                    out->descriptors.f.insert(out->descriptors.f.end(),
                                              desc.begin(), desc.end());
                }
        }
    }

    static bool Interpolate(const Layer& below, const Layer& here, const Layer& above,
                            int x, int y, float* ox, float* oy, float* os) {
        const float v2 = 2.0f * here.Det(x, y);
        const float dxx = here.Det(x + 1, y) + here.Det(x - 1, y) - v2;
        const float dyy = here.Det(x, y + 1) + here.Det(x, y - 1) - v2;
        const float dss = above.Det(x, y) + below.Det(x, y) - v2;
        const float dx = 0.5f * (here.Det(x + 1, y) - here.Det(x - 1, y));
        const float dy = 0.5f * (here.Det(x, y + 1) - here.Det(x, y - 1));
        const float ds = 0.5f * (above.Det(x, y) - below.Det(x, y));

        // Diagonal only. The full 3x3 solve is what SIFT does; here the cross
        // terms are small because the layers are coarsely spaced, and a
        // diagonal fit is both stable and enough to get sub-pixel position.
        if (std::fabs(dxx) < 1e-12f || std::fabs(dyy) < 1e-12f ||
            std::fabs(dss) < 1e-12f) return false;

        *ox = -dx / dxx;
        *oy = -dy / dyy;
        *os = -ds / dss;
        return std::fabs(*ox) < 0.6f && std::fabs(*oy) < 0.6f && std::fabs(*os) < 0.6f;
    }

    // Haar wavelet response over a box, in constant time.
    static float HaarX(const Integral& ii, int x, int y, int s) {
        return float(ii.Box(x, y - s / 2, s / 2, s) - ii.Box(x - s / 2, y - s / 2, s / 2, s));
    }
    static float HaarY(const Integral& ii, int x, int y, int s) {
        return float(ii.Box(x - s / 2, y, s, s / 2) - ii.Box(x - s / 2, y - s / 2, s, s / 2));
    }

    // Dominant orientation from Haar responses in a circular neighbourhood.
    //
    // A sliding window over ORIENTATION rather than a histogram, which is
    // Bay's method and differs from SIFT's deliberately: it sums the vector
    // responses in a 60-degree wedge and takes the wedge with the longest sum,
    // so the answer is a direction rather than a bin.
    static float Orientation(const Integral& ii, const Keypoint& kp) {
        const int s = std::max(1, int(std::round(kp.scale)));
        const int r = 6 * s;

        std::vector<float> rx, ry, ang;
        for (int dy = -r; dy <= r; dy += s)
            for (int dx = -r; dx <= r; dx += s) {
                if (dx * dx + dy * dy > r * r) continue;
                const int px = int(kp.x) + dx, py = int(kp.y) + dy;
                const float gx = HaarX(ii, px, py, 4 * s);
                const float gy = HaarY(ii, px, py, 4 * s);
                // Gaussian weighted at 2.5s, Bay's value.
                const float wgt = std::exp(-float(dx * dx + dy * dy) /
                                           (2.0f * (2.5f * float(s)) * (2.5f * float(s))));
                rx.push_back(gx * wgt);
                ry.push_back(gy * wgt);
                ang.push_back(std::atan2(gy, gx));
            }
        if (rx.empty()) return 0.0f;

        float best = 0.0f, bestLen = -1.0f;
        constexpr float kWedge = 1.0471976f;   // 60 degrees
        for (float a = 0.0f; a < 6.2831853f; a += 0.15f) {
            float sx = 0.0f, sy = 0.0f;
            for (size_t i = 0; i < rx.size(); ++i) {
                float d = ang[i] - a;
                while (d < -3.14159265f) d += 6.2831853f;
                while (d >  3.14159265f) d -= 6.2831853f;
                if (std::fabs(d) > kWedge * 0.5f) continue;
                sx += rx[i];
                sy += ry[i];
            }
            const float len = sx * sx + sy * sy;
            if (len > bestLen) { bestLen = len; best = std::atan2(sy, sx); }
        }
        return best;
    }

    // The descriptor: Haar responses in a 4x4 grid, in the keypoint's frame.
    void Describe(const Integral& ii, const Keypoint& kp, float* desc) const {
        const int s = std::max(1, int(std::round(kp.scale)));
        const float cosA = std::cos(kp.angle), sinA = std::sin(kp.angle);
        const bool ext = bool(m_extended);
        const int per = ext ? 8 : 4;

        int at = 0;
        for (int cy = -2; cy < 2; ++cy)
            for (int cx = -2; cx < 2; ++cx) {
                float sdx = 0, sdy = 0, sadx = 0, sady = 0;
                // Extended: the same sums split by the sign of the OTHER axis,
                // which distinguishes patterns that the plain version confuses
                // -- at the cost of doubling the descriptor.
                float sdxP = 0, sdxN = 0, sdyP = 0, sdyN = 0;

                for (int j = 0; j < 5; ++j)
                    for (int i = 0; i < 5; ++i) {
                        // Sample position in the ROTATED frame, which is what
                        // makes the descriptor rotation invariant.
                        const float lx = float(cx * 5 + i) * float(s);
                        const float ly = float(cy * 5 + j) * float(s);
                        const int px = int(kp.x + lx * cosA - ly * sinA);
                        const int py = int(kp.y + lx * sinA + ly * cosA);

                        // Responses measured along the frame's axes, not the
                        // image's -- so they rotate with the keypoint.
                        const float hx = HaarX(ii, px, py, 2 * s);
                        const float hy = HaarY(ii, px, py, 2 * s);
                        const float dx =  hx * cosA + hy * sinA;
                        const float dy = -hx * sinA + hy * cosA;

                        const float wgt = std::exp(-(lx * lx + ly * ly) /
                                                   (2.0f * (3.3f * float(s)) *
                                                          (3.3f * float(s))));
                        const float wx = dx * wgt, wy = dy * wgt;

                        sdx  += wx;  sdy  += wy;
                        sadx += std::fabs(wx);
                        sady += std::fabs(wy);
                        if (dy >= 0.0f) { sdxP += wx; } else { sdxN += wx; }
                        if (dx >= 0.0f) { sdyP += wy; } else { sdyN += wy; }
                    }

                if (!ext) {
                    desc[at++] = sdx;  desc[at++] = sadx;
                    desc[at++] = sdy;  desc[at++] = sady;
                } else {
                    desc[at++] = sdxP; desc[at++] = std::fabs(sdxP);
                    desc[at++] = sdxN; desc[at++] = std::fabs(sdxN);
                    desc[at++] = sdyP; desc[at++] = std::fabs(sdyP);
                    desc[at++] = sdyN; desc[at++] = std::fabs(sdyN);
                }
                (void)per;
            }

        // Unit length, so the descriptor is invariant to contrast -- the same
        // role normalisation plays in SIFT, and what lets one distance
        // threshold work across images.
        float n = 0.0f;
        const int dim = ext ? 128 : 64;
        for (int i = 0; i < dim; ++i) n += desc[i] * desc[i];
        n = std::sqrt(n);
        if (n > 1e-9f) for (int i = 0; i < dim; ++i) desc[i] /= n;
    }

    Param<int> m_octaves{this, "octaves", 3, 1, 4,
        {.help = "How many filter-size doublings to search. SURF grows the "
                 "filter rather than shrinking the image, so this costs less "
                 "than it does in SIFT."}};

    Param<int> m_scales{this, "scales_per_octave", 2, 1, 4,
        {.help = "Filter sizes within each octave."}};

    Param<float> m_threshold{this, "threshold", 0.0004f, 0.0f, 0.02f,
        {.help = "Minimum Hessian determinant. Raise it to keep only strong, "
                 "round blobs; SURF finds fewer features than SIFT at "
                 "comparable settings and they are individually more reliable.",
         .step = 0.0001, .softMax = 0.004}};

    Param<bool> m_extended{this, "extended", false,
        "128-float descriptor instead of 64, splitting each cell's sums by the "
        "sign of the other axis. More discriminating and twice the matching "
        "cost -- this is the one place a detector here genuinely offers a "
        "descriptor choice."};

    Param<bool> m_upright{this, "upright", false,
        "Skip orientation and assume the camera is level (U-SURF). Much faster "
        "and more repeatable when the images really are not rotated, and wrong "
        "the moment they are."};

    Param<int> m_maxFeatures{this, "max_features", 5000, 10, 50000,
        {.help = "Stop after this many. A bound on time rather than a quality "
                 "control -- the strongest are not found first."}};

    int m_found = 0;
};

} // namespace

REGISTER_ALGORITHM(DetectSurf);

} // namespace tglab
