// detect_brisk — Leutenegger, Chli & Siegwart (2011).
//
// Free of patents, like ORB and unlike SURF.
//
// WHY BOTH THIS AND ORB, since they occupy the same niche: fast binary
// detectors for work that does not need SIFT's descriptor. They differ in the
// two places that matter, and the differences point in opposite directions.
//
//   SCALE. ORB takes its scale from a pyramid and reports whichever level a
//   corner was found at -- a discrete answer. BRISK builds the same kind of
//   pyramid but adds INTRA-octave layers between the octaves, then fits a
//   parabola through the response across scale to refine the answer
//   continuously. That is SIFT's trick applied to a FAST detector, and it is
//   the reason BRISK survives a change of scale where ORB starts to drift.
//
//   THE SAMPLING PATTERN. BRIEF (in ORB) compares randomly placed pairs inside
//   a square patch. BRISK samples CONCENTRIC RINGS -- 60 points on four rings
//   plus the centre -- and each sample is Gaussian-smoothed by an amount
//   proportional to its distance from the centre. That smoothing is the part
//   worth understanding: a point far from the centre moves further for a given
//   rotation error, so blurring it more is what keeps the descriptor stable
//   when the orientation estimate is slightly off.
//
//   The pattern also splits naturally by distance. SHORT pairs (close
//   together) carry the intensity comparisons that become descriptor bits;
//   LONG pairs (far apart) are used only to estimate orientation, because a
//   long baseline gives a better gradient direction. ORB gets its orientation
//   from an intensity centroid instead -- cheaper, and noisier.
//
// SO: ORB is the faster of the two and the better choice when every frame is
// at the same scale, which is the panorama case. BRISK costs more and holds up
// better across scale. Both are far cheaper than the scale-space detectors --
// measured on a 45 MP frame, SIFT costs 8.2 s of detection where ORB costs 3.1.
//
// The descriptor is 512 bits, twice ORB's 256, which makes it more
// discriminating and its Hamming distance twice the work. That is the same
// trade AKAZE makes at 486 bits.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "../../algo_util/features.h"
#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"
#include "gpu_pyramid.h"

namespace tglab {
namespace {

struct Layer {
    std::vector<float> v;
    int   w = 0, h = 0;
    float scale = 1.0f;      // input pixels per pixel here

    float At(int x, int y) const {
        return v[size_t(std::clamp(y, 0, h - 1)) * size_t(w) +
                 size_t(std::clamp(x, 0, w - 1))];
    }

    // Bilinear, for the descriptor's sub-pixel ring samples.
    float Sample(float x, float y) const {
        const float cx = std::clamp(x, 0.0f, float(w - 1));
        const float cy = std::clamp(y, 0.0f, float(h - 1));
        const int   x0 = int(cx), y0 = int(cy);
        const int   x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
        const float fx = cx - float(x0), fy = cy - float(y0);
        const float a = At(x0, y0) + (At(x1, y0) - At(x0, y0)) * fx;
        const float b = At(x0, y1) + (At(x1, y1) - At(x0, y1)) * fx;
        return a + (b - a) * fy;
    }
};

constexpr int kCircleX[16] = { 0,  1,  2,  3,  3,  3,  2,  1,
                               0, -1, -2, -3, -3, -3, -2, -1};
constexpr int kCircleY[16] = {-3, -3, -2, -1,  0,  1,  2,  3,
                              3,  3,  2,  1,  0, -1, -2, -3};

// AGAST/FAST-9 corner test, with the same four-point early rejection as ORB's.
bool FastCorner(const Layer& L, int x, int y, float t) {
    const float c = L.At(x, y);
    const float hi = c + t, lo = c - t;

    int brightAxis = 0, darkAxis = 0;
    for (int i = 0; i < 16; i += 4) {
        const float p = L.At(x + kCircleX[i], y + kCircleY[i]);
        if (p > hi) ++brightAxis;
        else if (p < lo) ++darkAxis;
    }
    if (brightAxis < 3 && darkAxis < 3) return false;

    int runBright = 0, runDark = 0;
    for (int i = 0; i < 32; ++i) {
        const int k = i & 15;
        const float p = L.At(x + kCircleX[k], y + kCircleY[k]);
        if (p > hi) { runDark = 0; if (++runBright >= 9) return true; }
        else if (p < lo) { runBright = 0; if (++runDark >= 9) return true; }
        else { runBright = runDark = 0; }
    }
    return false;
}

// FAST score: the largest threshold at which this pixel is still a corner.
//
// BRISK uses this rather than Harris, and for a specific reason -- the score
// has to be comparable ACROSS SCALE LAYERS so the parabola fit through scale
// means something. A Harris response depends on the gradient magnitude at that
// layer, which changes as the image is downsampled; the FAST score is a
// threshold in the same normalised intensity units at every layer.
//
// Found by bisection rather than by a closed form: the corner test is not
// monotonic in any algebraic way, but it IS monotonic in the threshold, which
// is all bisection needs.
float FastScore(const Layer& L, int x, int y, float lo, float hi) {
    for (int i = 0; i < 8; ++i) {
        const float mid = 0.5f * (lo + hi);
        if (FastCorner(L, x, y, mid)) lo = mid;
        else                          hi = mid;
    }
    return lo;
}

// The BRISK sampling pattern: concentric rings, with a smoothing sigma per
// point that grows with its radius.
struct Pattern {
    struct Point { float x, y, sigma; };
    std::vector<Point> pts;

    // Pairs, split by the distance between their points.
    std::vector<std::pair<int, int>> shortPairs;   // descriptor bits
    std::vector<std::pair<int, int>> longPairs;    // orientation only

    explicit Pattern(float scale) {
        // Leutenegger's layout: 1 centre + 4 rings of 10, 14, 15 and 20 points.
        const int   counts[4] = {10, 14, 15, 20};
        const float radii[4]  = {0.17f, 0.31f, 0.46f, 0.62f};

        pts.push_back({0.0f, 0.0f, 0.0f});
        for (int r = 0; r < 4; ++r) {
            const float rad = radii[r] * scale;
            for (int i = 0; i < counts[r]; ++i) {
                const float a = 6.2831853f * float(i) / float(counts[r]) +
                                (r & 1 ? 3.14159265f / float(counts[r]) : 0.0f);
                // Sigma proportional to the ring spacing: a point on an outer
                // ring covers more ground for the same angular error, so it is
                // averaged over a wider area to stay stable.
                pts.push_back({rad * std::cos(a), rad * std::sin(a),
                               0.6f * rad * 6.2831853f / float(counts[r])});
            }
        }

        // Split every pair by distance. The thresholds are Leutenegger's,
        // expressed relative to the pattern scale.
        const float dMax = 0.27f * scale;   // short: descriptor bits
        const float dMin = 0.55f * scale;   // long: orientation
        for (size_t i = 0; i < pts.size(); ++i)
            for (size_t j = i + 1; j < pts.size(); ++j) {
                const float dx = pts[i].x - pts[j].x;
                const float dy = pts[i].y - pts[j].y;
                const float d  = std::sqrt(dx * dx + dy * dy);
                if (d < dMax)      shortPairs.emplace_back(int(i), int(j));
                else if (d > dMin) longPairs.emplace_back(int(i), int(j));
            }
    }
};

Layer Downsample(const Layer& src, float factor) {
    Layer out;
    out.w = std::max(1, int(float(src.w) / factor));
    out.h = std::max(1, int(float(src.h) / factor));
    out.scale = src.scale * factor;
    out.v.assign(size_t(out.w) * size_t(out.h), 0.0f);
    // Blur to the target scale, then sample BILINEARLY at the fractional step.
    //
    // The obvious version -- an integer box average at a truncated source
    // position -- is wrong for BRISK specifically, because BRISK's whole point
    // is the INTRA-OCTAVE layers at a factor of 1.5. An integer box cannot
    // represent a 1.5x reduction: it averages a 2x2 block while stepping 1.5
    // pixels, so consecutive output pixels overlap unevenly and the result
    // aliases. Aliasing does not stay still when the image shifts, so the
    // features found on those layers do not repeat.
    //
    // Measured, before this: BRISK's octave 0 repeated at 90% under an 11 px
    // shift and its octave 1 -- the first one reached through a 1.5x step --
    // collapsed to 35%. ORB, whose pyramid uses only integer-friendly steps,
    // held 100% through octave 4.
    //
    // Gaussian sigma of 0.4 * factor is the usual band-limiting rule for a
    // resample: enough to remove what the new sampling rate cannot represent,
    // not so much that the layer is needlessly soft.
    const float sigma = 0.4f * factor;
    const int   r     = std::max(1, int(std::ceil(sigma * 2.5f)));

    std::vector<float> k(size_t(r) * 2 + 1);
    float ksum = 0.0f;
    for (int i = -r; i <= r; ++i) {
        const float e = std::exp(-float(i * i) / (2.0f * sigma * sigma));
        k[size_t(i + r)] = e;
        ksum += e;
    }
    for (float& v : k) v /= ksum;

    // Separable: blur the source into a scratch plane, then sample it.
    Layer blur;
    blur.w = src.w; blur.h = src.h; blur.scale = src.scale;
    blur.v.assign(size_t(src.w) * size_t(src.h), 0.0f);
    std::vector<float> tmp(size_t(src.w) * size_t(src.h), 0.0f);
    for (int y = 0; y < src.h; ++y)
        for (int x = 0; x < src.w; ++x) {
            float a = 0.0f;
            for (int i = -r; i <= r; ++i) a += k[size_t(i + r)] * src.At(x + i, y);
            tmp[size_t(y) * size_t(src.w) + size_t(x)] = a;
        }
    {
        Layer mid;
        mid.w = src.w; mid.h = src.h; mid.v = std::move(tmp);
        for (int y = 0; y < src.h; ++y)
            for (int x = 0; x < src.w; ++x) {
                float a = 0.0f;
                for (int i = -r; i <= r; ++i) a += k[size_t(i + r)] * mid.At(x, y + i);
                blur.v[size_t(y) * size_t(src.w) + size_t(x)] = a;
            }
    }

    for (int y = 0; y < out.h; ++y)
        for (int x = 0; x < out.w; ++x)
            out.v[size_t(y) * size_t(out.w) + size_t(x)] =
                blur.Sample(float(x) * factor, float(y) * factor);
    return out;
}

class DetectBrisk : public AlgorithmBase {
public:
    const char* Name()     const override { return "detect_brisk"; }
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
        if (w < 64 || h < 64) return;

        Layer base;
        base.w = w; base.h = h;
        base.v.assign(size_t(w) * size_t(h), 0.0f);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const float* p = in.At(x, y);
                base.v[size_t(y) * size_t(w) + size_t(x)] = (ch >= 3)
                    ? (0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2]) / scale
                    : p[0] / scale;
            }

        // Normalised by the image's own level -- see Percentile99 in
        // features.h. FAST's threshold is an absolute intensity difference, so
        // without this it means nothing on a scene-referred raw.
        m_level = Percentile99(base.v);
        if (m_level > 1e-6f) {
            const float inv = 1.0f / m_level;
            for (float& v : base.v) v *= inv;
        }

        auto sidecar = std::make_shared<FeatureSidecar>();
        sidecar->detector = "brisk";
        sidecar->descriptors.kind = DescriptorKind::Binary;
        sidecar->descriptors.dim  = kDescBits;

        Detect(base, ctx, sidecar.get());
        m_found = int(sidecar->keypoints.size());

        if (Image* im = ctx.OutImage(0)) im->Sidecars().Set(kFeatureSidecar, sidecar);
    }

    std::string RunReport() const override {
        if (m_found <= 0) return {};
        char buf[80];
        std::snprintf(buf, sizeof buf, "%d BRISK features (%d bits, level %.3f)",
                      m_found, kDescBits, m_level);
        return buf;
    }

    bool HasGPU() const override { return false; }

private:
    static constexpr int kDescBits    = 512;
    static constexpr int kPatternSize = 40;   // pattern diameter at scale 1

    void Detect(const Layer& base, RunCtx& ctx, FeatureSidecar* out) {
        const int   octaves = std::clamp(int(m_octaves), 1, 6);
        const float thresh  = std::max(0.001f, float(m_threshold));
        const int   border  = kPatternSize;

        // The pyramid, with an INTRA-octave layer between each pair.
        //
        // That interleaving is what distinguishes BRISK's scale handling from
        // ORB's: octave o sits at 2^o and intra-octave o at 1.5 * 2^o, so
        // consecutive layers are about 1.33x apart rather than 2x. A parabola
        // through three of them gives a continuous scale, where a plain
        // octave pyramid can only name the layer.
        std::vector<Layer> layers;
        layers.push_back(base);                          // octave 0
        {
            Layer intra = Downsample(base, 1.5f);        // intra-octave 0
            if (intra.w >= 2 * border && intra.h >= 2 * border)
                layers.push_back(std::move(intra));
        }
        for (int o = 1; o < octaves; ++o) {
            Layer oct = Downsample(layers[size_t(2 * (o - 1))], 2.0f);
            if (oct.w < 2 * border || oct.h < 2 * border) break;
            layers.push_back(std::move(oct));
            Layer intra = Downsample(layers.back(), 1.5f);
            if (intra.w < 2 * border || intra.h < 2 * border) break;
            layers.push_back(std::move(intra));
        }

        struct Cand {
            float x, y, scale, response;
            int   layer, lx, ly;
        };
        std::vector<Cand> cands;

        // The cross-layer score lookups stay on the CPU, though precomputing a
        // whole-image score map per layer looks like the obvious win.
        //
        // It is not. IsScaleMax and RefineScale ask neighbouring layers for a
        // score at a handful of positions each, and they run per CANDIDATE --
        // and after suppression there are thousands of candidates against
        // millions of pixels. Measured on 1200x800: building the maps for all
        // seven layers cost 51 ms and computed roughly 75x more than was ever
        // read (about 4M evaluations to serve ~53k lookups), taking BRISK from
        // 1.0x to 0.9x. Sparse demand does not justify a dense dispatch.
        //
        // The per-pixel scan below is the part worth offloading, because there
        // every pixel really is visited.
        const std::function<float(size_t, int, int)> scoreAt =
            [&](size_t li, int x, int y) -> float {
                return FastScore(layers[li], x, y, 0.0f, 1.0f);
            };

        for (size_t li = 0; li < layers.size(); ++li) {
            if (ctx.Cancelled()) return;
            const Layer& L = layers[li];

            // FAST plus its bisected score: up to nine ring walks per pixel,
            // which is where BRISK's time goes. Dense map, so the scan below
            // keeps its order and the candidate list is identical either way.
            //
            // Only this part offloads. The scale-space suppression that follows
            // reads NEIGHBOURING layers, so it is not a per-pixel test over one
            // plane and stays on the CPU. See gpu_pyramid.h.
            std::vector<float> score;
            bool haveMap = false;
            if (ComputeContext* dev = ctx.Gpu()) {
                GpuPlane gsrc;
                gsrc.v = L.v; gsrc.w = L.w; gsrc.h = L.h;
                GpuPlane map;
                std::string gerr;
                if (GpuFastScore(dev, gsrc, &map, thresh, border, &gerr)) {
                    score = std::move(map.v);
                    haveMap = true;
                }
            }

            for (int y = border; y < L.h - border; ++y)
                for (int x = border; x < L.w - border; ++x) {
                    float s;
                    if (haveMap) {
                        s = score[size_t(y) * size_t(L.w) + size_t(x)];
                        if (s <= 0.0f) continue;   // FAST rejected it
                    } else {
                        if (!FastCorner(L, x, y, thresh)) continue;
                        s = FastScore(L, x, y, thresh, 1.0f);
                    }

                    // Non-maximum suppression IN SCALE as well as space.
                    //
                    // Without the scale part the same corner is reported once
                    // per layer it survives on, which is most of them -- and a
                    // matcher then sees several near-identical descriptors at
                    // one position and the ratio test throws all of them away.
                    // That failure is quiet: the detector reports plenty of
                    // features and almost nothing matches.
                    if (!IsScaleMax(layers, li, x, y, L.scale, s, scoreAt)) continue;

                    Cand c;
                    c.lx = x;
                    c.ly = y;
                    c.x  = float(x) * L.scale;
                    c.y  = float(y) * L.scale;
                    c.scale    = RefineScale(layers, li, x, y, L.scale, s, scoreAt);
                    c.response = s;
                    c.layer    = int(li);
                    cands.push_back(c);
                }
        }

        const int cap = std::max(1, int(m_maxFeatures));
        if (int(cands.size()) > cap) {
            std::nth_element(cands.begin(), cands.begin() + cap, cands.end(),
                             [](const Cand& a, const Cand& b) {
                                 return a.response > b.response;
                             });
            cands.resize(size_t(cap));
        }

        const bool upright = bool(m_upright);
        out->keypoints.reserve(cands.size());
        out->descriptors.b.assign(cands.size() * size_t(kDescBits / 8), 0);

        for (size_t i = 0; i < cands.size(); ++i) {
            if (ctx.Cancelled()) return;
            const Cand&  c = cands[i];
            const Layer& L = layers[size_t(c.layer)];

            // The pattern is sized by the feature's own refined scale,
            // expressed in THIS LAYER's pixels -- which is what makes the
            // descriptor scale invariant. c.scale is in input pixels per
            // pattern unit and L.scale is input pixels per layer pixel, so
            // their ratio is the pattern size in layer pixels.
            const Pattern pat(float(kPatternSize) * c.scale / L.scale);

            const float angle = upright ? 0.0f : Orientation(L, c.lx, c.ly, pat);

            Keypoint k;
            k.x        = c.x;
            k.y        = c.y;
            k.scale    = c.scale * float(kPatternSize) * 0.5f;
            k.angle    = angle;
            k.response = c.response;
            k.octave   = c.layer / 2;
            out->keypoints.push_back(k);

            Describe(L, c.lx, c.ly, angle, pat,
                     &out->descriptors.b[i * size_t(kDescBits / 8)]);
        }
    }

    // Is this the strongest response at this position across neighbouring
    // scale layers? Compared at the SAME image position, which means
    // converting coordinates between layers of different sizes.
    // How to get a layer's FAST score at a point: from a precomputed map when
    // there is one, or by computing it. Passed in rather than looked up so
    // these stay static and testable.
    using ScoreFn = const std::function<float(size_t, int, int)>&;

    static bool IsScaleMax(const std::vector<Layer>& layers, size_t li,
                           int x, int y, float scale, float s, ScoreFn scoreAt) {
        const float ix = float(x) * scale, iy = float(y) * scale;
        for (int d = -1; d <= 1; d += 2) {
            const long long n = (long long)li + d;
            if (n < 0 || n >= (long long)layers.size()) continue;
            const Layer& N = layers[size_t(n)];
            const int nx = int(std::lround(ix / N.scale));
            const int ny = int(std::lround(iy / N.scale));
            // A 3x3 window, because the same feature lands at a slightly
            // different rounded position on a differently sized layer.
            //
            // COMPARED BY SCORE, not by "is it also a corner" -- which is what
            // the first version asked, and it made every feature that survived
            // on two adjacent layers suppress ITSELF. A real corner is a corner
            // across several scales by definition, so that rejected most of
            // them: repeatability came out at 56% against ORB's 86%, on 142
            // features where ORB found 575.
            //
            // The strict > is what breaks the tie: with >= two layers scoring
            // exactly equal would each suppress the other and the feature would
            // vanish from both.
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    if (!FastCorner(N, nx + dx, ny + dy, s)) continue;
                    if (scoreAt(n, nx + dx, ny + dy) > s) return false;
                }
        }
        return true;
    }

    // Sub-layer scale by fitting a parabola through the response at this layer
    // and its two neighbours. Falls back to the layer's own scale when there
    // are not three samples or the fit is degenerate.
    static float RefineScale(const std::vector<Layer>& layers, size_t li,
                             int x, int y, float scale, float s,
                             ScoreFn scoreAt) {
        if (li == 0 || li + 1 >= layers.size()) return scale;

        const float ix = float(x) * scale, iy = float(y) * scale;
        auto at = [&](size_t n) {
            const Layer& N = layers[n];
            return scoreAt(n, int(std::lround(ix / N.scale)),
                           int(std::lround(iy / N.scale)));
        };
        const float sm = at(li - 1), sp = at(li + 1);
        const float denom = sm - 2.0f * s + sp;
        if (std::abs(denom) < 1e-9f) return scale;

        const float off = std::clamp(0.5f * (sm - sp) / denom, -0.5f, 0.5f);
        // Layers are a fixed ratio apart, so an offset in layers is a
        // multiplicative step in scale.
        const float ratio = layers[li + 1].scale / layers[li].scale;
        return scale * std::pow(ratio, off);
    }

    // Orientation from the LONG pairs: the average intensity gradient
    // direction over widely separated sample points.
    //
    // Long pairs specifically, because a gradient estimated over a long
    // baseline is less affected by noise at either end -- and orientation is
    // the one quantity where a small error costs the whole descriptor.
    static float Orientation(const Layer& L, int x, int y, const Pattern& p) {
        float gx = 0.0f, gy = 0.0f;
        for (const auto& pr : p.longPairs) {
            const auto& a = p.pts[size_t(pr.first)];
            const auto& b = p.pts[size_t(pr.second)];
            const float va = L.Sample(float(x) + a.x, float(y) + a.y);
            const float vb = L.Sample(float(x) + b.x, float(y) + b.y);
            const float dx = b.x - a.x, dy = b.y - a.y;
            const float d2 = dx * dx + dy * dy;
            if (d2 < 1e-6f) continue;
            const float w = (va - vb) / d2;
            gx += w * dx;
            gy += w * dy;
        }
        return std::atan2(gy, gx);
    }

    // The descriptor: one bit per SHORT pair, rotated by the orientation.
    static void Describe(const Layer& L, int x, int y, float angle,
                         const Pattern& p, uint8_t* desc) {
        const float c = std::cos(angle), s = std::sin(angle);
        const int n = std::min(kDescBits, int(p.shortPairs.size()));
        for (int i = 0; i < n; ++i) {
            const auto& a = p.pts[size_t(p.shortPairs[size_t(i)].first)];
            const auto& b = p.pts[size_t(p.shortPairs[size_t(i)].second)];

            const float axr = c * a.x - s * a.y, ayr = s * a.x + c * a.y;
            const float bxr = c * b.x - s * b.y, byr = s * b.x + c * b.y;

            if (L.Sample(float(x) + axr, float(y) + ayr) <
                L.Sample(float(x) + bxr, float(y) + byr))
                desc[i >> 3] |= uint8_t(1u << (i & 7));
        }
    }

    Param<int> m_octaves{this, "octaves", 4, 1, 6,
        {.help = "Pyramid octaves. Each has an intra-octave layer between it "
                 "and the next, so the scale sampling is about 1.33x per layer "
                 "rather than 2x -- which is what lets the scale be refined "
                 "continuously rather than only named."}};

    Param<float> m_threshold{this, "threshold", 0.06f, 0.001f, 0.5f,
        {.help = "How much brighter or darker the FAST ring must be than the "
                 "centre, relative to the image's own 99th percentile. Raise "
                 "it to keep only high-contrast corners.",
         .step = 0.005, .softMax = 0.2}};

    Param<bool> m_upright{this, "upright", false,
        "Skip the orientation estimate and describe every feature axis-aligned. "
        "Faster, and correct when the camera never rolls."};

    Param<int> m_maxFeatures{this, "max_features", 5000, 10, 50000,
        {.help = "Keep this many, strongest first by FAST score."}};

    int   m_found = 0;
    float m_level = 0.0f;
};

REGISTER_ALGORITHM(DetectBrisk);

} // namespace
} // namespace tglab
