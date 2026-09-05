// detect_akaze — Alcantarilla, Nuevo & Bartoli (2013). BSD, no patent.
//
// THE IDEA THAT MAKES IT DIFFERENT, and the reason it is worth having beside
// SIFT and SURF rather than being a third variation on the same theme:
//
//   SIFT and SURF both build a GAUSSIAN scale space. Gaussian blur is
//   isotropic -- it smooths across edges exactly as readily as along them --
//   so as the scale grows, object boundaries blur into their surroundings and
//   the features on them drift. That drift is a real cost: it is why a
//   coarse-scale SIFT feature localises less precisely than a fine one.
//
//   AKAZE builds a NONLINEAR diffusion scale space instead. Diffusion is
//   locally slowed where the gradient is strong, so smoothing proceeds inside
//   regions and stops at their boundaries. Edges stay sharp at every scale, and
//   a feature detected coarsely sits where the fine-scale one does.
//
// The price is that there is no closed form: the diffusion equation has to be
// integrated numerically. This uses the explicit scheme (AOS in the paper's
// terms is more stable but far more code), which is stable for a step below
// 0.25 and is what the `steps` parameter subdivides.
//
// The descriptor is M-LDB: a BINARY string, compared by Hamming distance rather
// than L2. That is the point of carrying DescriptorKind on the set -- a matcher
// given L2 over these bits returns matches and they are meaningless. See
// features.h.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "../../algo_util/features.h"
#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"
#include "gpu_pyramid.h"

namespace tglab {
namespace {

struct Plane {
    std::vector<float> v;
    int w = 0, h = 0;

    float At(int x, int y) const {
        return v[size_t(std::clamp(y, 0, h - 1)) * size_t(w) +
                 size_t(std::clamp(x, 0, w - 1))];
    }
    float& Ref(int x, int y) {
        return v[size_t(std::clamp(y, 0, h - 1)) * size_t(w) +
                 size_t(std::clamp(x, 0, w - 1))];
    }
};

// Perona-Malik conductivity, g2 in the paper.
//
// Near 1 where the gradient is small (diffuse freely) and near 0 where it is
// large (do not diffuse across this edge). `k` is the contrast parameter that
// decides what counts as "large", and it is measured from the image rather than
// fixed -- see ContrastFactor.
inline float Conductivity(float gradSq, float k) {
    return 1.0f / (1.0f + gradSq / (k * k));
}

// The contrast parameter, as the 70th percentile of the gradient magnitude.
//
// Measured rather than fixed, and that matters: k decides which gradients count
// as edges, so a value tuned on a high-contrast image stops all diffusion on a
// low-contrast one -- the scale space then does nothing and every feature is
// found at the base scale.
float ContrastFactor(const Plane& p) {
    std::vector<float> mags;
    mags.reserve(size_t(p.w) * size_t(p.h) / 4);
    for (int y = 1; y < p.h - 1; y += 2)
        for (int x = 1; x < p.w - 1; x += 2) {
            const float gx = p.At(x + 1, y) - p.At(x - 1, y);
            const float gy = p.At(x, y + 1) - p.At(x, y - 1);
            mags.push_back(std::sqrt(gx * gx + gy * gy));
        }
    if (mags.empty()) return 0.1f;
    const size_t i = size_t(0.7 * double(mags.size() - 1));
    std::nth_element(mags.begin(), mags.begin() + i, mags.end());
    return std::max(mags[i], 1e-4f);
}

// One explicit diffusion step.
//
// The five-point Laplacian, weighted per-edge by the conductivity between the
// two pixels. That per-edge weighting is the whole mechanism: a uniform weight
// would be an ordinary blur.
void DiffuseStep(Plane& p, const Plane& cond, float dt, std::vector<float>& tmp) {
    tmp = p.v;
    for (int y = 1; y < p.h - 1; ++y)
        for (int x = 1; x < p.w - 1; ++x) {
            const float c = cond.At(x, y);
            const float here = p.At(x, y);
            // Conductivity between two pixels as the average of theirs, which
            // keeps the scheme symmetric -- an asymmetric weight would let
            // brightness leak in one direction and drift the whole image.
            const float fl = 0.5f * (c + cond.At(x - 1, y)) * (p.At(x - 1, y) - here);
            const float fr = 0.5f * (c + cond.At(x + 1, y)) * (p.At(x + 1, y) - here);
            const float fu = 0.5f * (c + cond.At(x, y - 1)) * (p.At(x, y - 1) - here);
            const float fd = 0.5f * (c + cond.At(x, y + 1)) * (p.At(x, y + 1) - here);
            tmp[size_t(y) * size_t(p.w) + size_t(x)] = here + dt * (fl + fr + fu + fd);
        }
    p.v.swap(tmp);
}

void GaussianBlur3(Plane& p, std::vector<float>& tmp) {
    // A small fixed blur, used only to condition the gradient estimate that
    // feeds the conductivity. Not part of the scale space.
    tmp.assign(p.v.size(), 0.0f);
    for (int y = 0; y < p.h; ++y)
        for (int x = 0; x < p.w; ++x)
            tmp[size_t(y) * size_t(p.w) + size_t(x)] =
                0.25f * p.At(x, y) +
                0.125f * (p.At(x - 1, y) + p.At(x + 1, y) +
                          p.At(x, y - 1) + p.At(x, y + 1)) +
                0.0625f * (p.At(x - 1, y - 1) + p.At(x + 1, y - 1) +
                           p.At(x - 1, y + 1) + p.At(x + 1, y + 1));
    p.v.swap(tmp);
}

class DetectAkaze : public AlgorithmBase {
public:
    const char* Name()     const override { return "detect_akaze"; }
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

        Plane base;
        base.w = w; base.h = h;
        base.v.assign(size_t(w) * size_t(h), 0.0f);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const float* p = in.At(x, y);
                base.v[size_t(y) * size_t(w) + size_t(x)] = (ch >= 3)
                    ? (0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2]) / scale
                    : p[0] / scale;
            }

        auto sidecar = std::make_shared<FeatureSidecar>();
        sidecar->detector = "akaze";
        sidecar->descriptors.kind = DescriptorKind::Binary;
        sidecar->descriptors.dim  = kDescBits;

        Detect(base, ctx, sidecar.get());
        m_found = int(sidecar->keypoints.size());

        if (Image* im = ctx.OutImage(0)) im->Sidecars().Set(kFeatureSidecar, sidecar);
    }

    std::string RunReport() const override {
        if (m_found <= 0) return {};
        return std::to_string(m_found) + " AKAZE features (" +
               std::to_string(kDescBits) + " bits)";
    }

    // Sidecar coordinates are in IMAGE PIXELS and nothing rescales them, so a
    // proxy run would hand every downstream stage positions that are wrong by
    // the scale factor -- silently, since the sidecar is still present and
    // still looks valid. See AlgorithmBase::Proxy.
    ProxyBehaviour Proxy() const override { return ProxyBehaviour::Never; }

    bool HasGPU() const override { return false; }

private:
    // 3 grids (2x2, 3x3, 4x4) x 3 channels (intensity, dx, dy) x pairs.
    // 2x2 gives 6 pairs, 3x3 gives 36, 4x4 gives 120: (6+36+120)*3 = 486.
    static constexpr int kDescBits = 486;

    struct Level {
        Plane img;        // the diffused image at this scale
        Plane det;        // Hessian determinant response
        float sigma = 0.0f;
        int   octave = 0;
        int   step = 1;      // pixels in the input per pixel here
    };

    void Detect(const Plane& base, RunCtx& ctx, FeatureSidecar* out) {
        const int octaves = std::clamp(int(m_octaves), 1, 4);
        const int perOct  = std::clamp(int(m_scales), 1, 4);
        const float sigma0 = 1.6f;

        std::vector<float> tmp;

        // The conductivity is fixed from the BASE image and reused, which is
        // the paper's choice: recomputing it per level would let it drift as
        // the image smooths, and the diffusion would slowly become linear.
        Plane smoothed = base;
        GaussianBlur3(smoothed, tmp);
        const float k = ContrastFactor(smoothed) * float(m_contrastPercentile);

        Plane cond;
        cond.w = base.w; cond.h = base.h;
        cond.v.assign(base.v.size(), 1.0f);
        for (int y = 1; y < base.h - 1; ++y)
            for (int x = 1; x < base.w - 1; ++x) {
                const float gx = smoothed.At(x + 1, y) - smoothed.At(x - 1, y);
                const float gy = smoothed.At(x, y + 1) - smoothed.At(x, y - 1);
                cond.Ref(x, y) = Conductivity(gx * gx + gy * gy, k);
            }

        std::vector<Level> levels;
        Plane cur = base;
        Plane curCond = cond;
        float sigma = sigma0;

        // DOWNSAMPLED between octaves, which is not an optimisation but a
        // correctness requirement, and getting it wrong is what made the first
        // version find one feature in 1.3 seconds.
        //
        // Diffusion time grows as sigma^2, so reaching sigma 12.8 at full
        // resolution takes ~400 explicit steps over every pixel: slow, and by
        // then the image is so smoothed that nothing is extremal. Halving the
        // resolution each octave halves sigma in the new pixels, so the time
        // needed resets and every octave costs a similar handful of steps --
        // which is what the paper does and why it is fast.
        for (int o = 0; o < octaves; ++o) {
            if (o > 0) {
                cur     = Halve(cur);
                curCond = Halve(curCond);
                // Sigma is measured in THIS octave's pixels, so halving the
                // resolution halves it. Without this the time computed below
                // is for a scale the image has already passed.
                sigma *= 0.5f;
                if (cur.w < 16 || cur.h < 16) break;
            }

            for (int s = 0; s < perOct; ++s) {
                if (ctx.Cancelled()) return;

                // Target within this octave, in this octave's pixels: always
                // between sigma0 and 2*sigma0, so the diffusion is short.
                const float target = sigma0 * std::pow(2.0f, float(s) / float(perOct));
                if (target > sigma) {
                    // Evolution time for a given sigma: t = sigma^2 / 2.
                    const float dtTotal = 0.5f * (target * target - sigma * sigma);
                    // Subdivided so each step stays under the explicit scheme's
                    // stability limit of 0.25. Exceeding it does not blur more
                    // -- it oscillates and diverges.
                    const int steps = std::max(1, int(std::ceil(dtTotal / 0.2f)));
                    const float dt = dtTotal / float(steps);

                    // The whole chain in one call when there is a device: each
                    // step is a 5-point stencil, so it parallelises even though
                    // the sequence does not, and keeping the intermediate
                    // resident turns N round trips into one. See gpu_pyramid.h.
                    bool onGpu = false;
                    if (ComputeContext* dev = ctx.Gpu()) {
                        GpuPlane gsrc, gcond, gout;
                        gsrc.v  = cur.v;     gsrc.w  = cur.w;     gsrc.h  = cur.h;
                        gcond.v = curCond.v; gcond.w = curCond.w; gcond.h = curCond.h;
                        std::string gerr;
                        if (GpuDiffuse(dev, gsrc, gcond, steps, dt, &gout, &gerr)) {
                            cur.v = std::move(gout.v);
                            onGpu = true;
                        }
                    }
                    if (!onGpu)
                        for (int i = 0; i < steps; ++i)
                            DiffuseStep(cur, curCond, dt, tmp);
                    sigma = target;
                }

                Level L;
                L.img = cur;
                L.sigma = sigma;
                L.octave = o;
                L.step = 1 << o;
                // Normalised by the sigma IN THIS OCTAVE, which stays near
                // sigma0 -- so responses are comparable across octaves. The
                // first version used the absolute sigma, which spans 1.6 to
                // 12.8 and makes sigma^4 span 7 to 10,653: no single threshold
                // can work across that, which is the other half of why it
                // found one feature.
                L.det = Hessian(cur, sigma);
                levels.push_back(std::move(L));
            }
        }

        FindExtrema(levels, ctx, out);
    }

    // Halves a plane by point sampling. The source has already been diffused to
    // the octave's base scale, so it is band-limited and decimating is correct.
    static Plane Halve(const Plane& src) {
        Plane out;
        out.w = std::max(1, src.w / 2);
        out.h = std::max(1, src.h / 2);
        out.v.assign(size_t(out.w) * size_t(out.h), 0.0f);
        for (int y = 0; y < out.h; ++y)
            for (int x = 0; x < out.w; ++x)
                out.v[size_t(y) * size_t(out.w) + size_t(x)] = src.At(x * 2, y * 2);
        return out;
    }

    // Scale-normalised determinant of the Hessian.
    //
    // Multiplied by sigma^4 rather than left raw: without the normalisation the
    // response falls with scale and every feature is found at the finest level,
    // which defeats the point of building a scale space at all.
    static Plane Hessian(const Plane& p, float sigma) {
        Plane d;
        d.w = p.w; d.h = p.h;
        d.v.assign(p.v.size(), 0.0f);
        const float s4 = sigma * sigma * sigma * sigma;
        for (int y = 1; y < p.h - 1; ++y)
            for (int x = 1; x < p.w - 1; ++x) {
                const float c2 = 2.0f * p.At(x, y);
                const float dxx = p.At(x + 1, y) + p.At(x - 1, y) - c2;
                const float dyy = p.At(x, y + 1) + p.At(x, y - 1) - c2;
                const float dxy = 0.25f * (p.At(x + 1, y + 1) - p.At(x - 1, y + 1) -
                                           p.At(x + 1, y - 1) + p.At(x - 1, y - 1));
                d.v[size_t(y) * size_t(p.w) + size_t(x)] = s4 * (dxx * dyy - dxy * dxy);
            }
        return d;
    }

    // A candidate, held until the whole scan is done. `level` is which scale
    // space level it came from, needed to describe it afterwards.
    struct Cand {
        Keypoint kp;
        int      level;
    };

    void FindExtrema(const std::vector<Level>& levels, RunCtx& ctx,
                     FeatureSidecar* out) {
        const float thresh = float(m_threshold);
        std::vector<Cand> cands;

        for (size_t i = 1; i + 1 < levels.size(); ++i) {
            if (ctx.Cancelled()) return;
            const Plane& below = levels[i - 1].det;
            const Plane& here  = levels[i].det;
            const Plane& above = levels[i + 1].det;

            for (int y = 2; y < here.h - 2; ++y)
                for (int x = 2; x < here.w - 2; ++x) {
                    const float v = here.At(x, y);
                    if (v < thresh) continue;

                    bool isMax = true;
                    for (int dy = -1; dy <= 1 && isMax; ++dy)
                        for (int dx = -1; dx <= 1; ++dx) {
                            if (below.At(x + dx, y + dy) >= v) { isMax = false; break; }
                            if (above.At(x + dx, y + dy) >= v) { isMax = false; break; }
                            if ((dx || dy) && here.At(x + dx, y + dy) >= v) {
                                isMax = false; break;
                            }
                        }
                    if (!isMax) continue;

                    // Sub-pixel by a diagonal quadratic fit, as in SURF.
                    const float dxx = here.At(x + 1, y) + here.At(x - 1, y) - 2.0f * v;
                    const float dyy = here.At(x, y + 1) + here.At(x, y - 1) - 2.0f * v;
                    const float dx = 0.5f * (here.At(x + 1, y) - here.At(x - 1, y));
                    const float dy = 0.5f * (here.At(x, y + 1) - here.At(x, y - 1));
                    const float ox = (std::fabs(dxx) > 1e-12f) ? -dx / dxx : 0.0f;
                    const float oy = (std::fabs(dyy) > 1e-12f) ? -dy / dyy : 0.0f;
                    if (std::fabs(ox) > 0.6f || std::fabs(oy) > 0.6f) continue;

                    const float st = float(levels[i].step);
                    Keypoint kp;
                    // Back to INPUT coordinates: everything above works in
                    // this octave's pixels, and the sidecar is read by
                    // algorithms that know only the original image.
                    kp.x = (float(x) + ox) * st;
                    kp.y = (float(y) + oy) * st;
                    kp.scale = levels[i].sigma * st;
                    kp.response = v;
                    kp.octave = levels[i].octave;

                    // Collected, not described. The orientation and the
                    // descriptor are the expensive part and most of these will
                    // not survive the cap -- see the ranking below.
                    cands.push_back({kp, int(i)});
                }
        }

        // RANK, THEN CAP. Never stop the scan.
        //
        // This was a `return` the moment the count hit max_features, in the
        // middle of a raster scan -- so a frame with more than 5000 extrema
        // kept everything above some scanline and NOTHING below it. Tim found
        // it exactly as one would: the right-hand end of a panorama failed to
        // align, and the feature count sat pinned at the cap.
        //
        // The failure is worse than "fewer features", because the loss is
        // SPATIAL. Features are found top-to-bottom, so the discarded region is
        // a contiguous band at the bottom of the frame -- and if that band is
        // where the next frame overlaps, the pair has nothing to match on. The
        // measured effect on the pair that failed: 2% of candidates kept and 0%
        // inliers at the 5000 cap, against 5% and 49% at 15000. The features
        // existed; the scan simply never reached them.
        //
        // ORB already did this correctly, which is why it was the one detector
        // that did not fail. SURF and SIFT had the same bug.
        const int cap = std::max(1, int(m_maxFeatures));
        if (int(cands.size()) > cap) {
            std::nth_element(cands.begin(), cands.begin() + cap, cands.end(),
                             [](const Cand& a, const Cand& b) {
                                 return a.kp.response > b.kp.response;
                             });
            cands.resize(size_t(cap));
        }

        // Describe only the survivors.
        out->keypoints.reserve(cands.size());
        for (const Cand& c : cands) {
            if (ctx.Cancelled()) return;
            Keypoint kp = c.kp;
            kp.angle = m_upright ? 0.0f : Orientation(levels[size_t(c.level)].img, kp);
            out->keypoints.push_back(kp);

            std::vector<uint8_t> desc(size_t((kDescBits + 7) / 8), 0);
            Describe(levels[size_t(c.level)].img, kp, desc.data());
            out->descriptors.b.insert(out->descriptors.b.end(),
                                      desc.begin(), desc.end());
        }
    }

    // Dominant orientation, by the same wedge method SURF uses.
    static float Orientation(const Plane& p, const Keypoint& kp) {
        const int s = std::max(1, int(std::round(kp.scale)));
        const int r = 6 * s;

        std::vector<float> rx, ry, ang;
        for (int dy = -r; dy <= r; dy += s)
            for (int dx = -r; dx <= r; dx += s) {
                if (dx * dx + dy * dy > r * r) continue;
                const int px = int(kp.x) + dx, py = int(kp.y) + dy;
                const float gx = p.At(px + 1, py) - p.At(px - 1, py);
                const float gy = p.At(px, py + 1) - p.At(px, py - 1);
                const float wgt = std::exp(-float(dx * dx + dy * dy) /
                                           (2.0f * (2.5f * float(s)) * (2.5f * float(s))));
                rx.push_back(gx * wgt);
                ry.push_back(gy * wgt);
                ang.push_back(std::atan2(gy, gx));
            }
        if (rx.empty()) return 0.0f;

        float best = 0.0f, bestLen = -1.0f;
        for (float a = 0.0f; a < 6.2831853f; a += 0.15f) {
            float sx = 0.0f, sy = 0.0f;
            for (size_t i = 0; i < rx.size(); ++i) {
                float d = ang[i] - a;
                while (d < -3.14159265f) d += 6.2831853f;
                while (d >  3.14159265f) d -= 6.2831853f;
                if (std::fabs(d) > 0.5236f) continue;    // 30 degrees either side
                sx += rx[i];
                sy += ry[i];
            }
            const float len = sx * sx + sy * sy;
            if (len > bestLen) { bestLen = len; best = std::atan2(sy, sx); }
        }
        return best;
    }

    // M-LDB: Modified Local Difference Binary.
    //
    // Divide the oriented patch into a grid, average intensity and both
    // gradients over each cell, and set one bit per CELL PAIR per channel
    // according to which cell is larger. Three grids at 2x2, 3x3 and 4x4 give
    // coarse-to-fine structure in one string.
    //
    // Binary because comparison is then a XOR and a popcount rather than 128
    // multiply-adds -- which is most of why AKAZE matches far faster than SIFT,
    // and the reason DescriptorKind exists.
    void Describe(const Plane& p, const Keypoint& kp, uint8_t* desc) const {
        const float cosA = std::cos(kp.angle), sinA = std::sin(kp.angle);
        const float s = std::max(1.0f, kp.scale);
        const float patch = 12.0f * s;      // the region the descriptor covers

        int bit = 0;
        for (int grid : {2, 3, 4}) {
            const int cells = grid * grid;
            std::vector<float> mi(size_t(cells), 0.0f);
            std::vector<float> mx(size_t(cells), 0.0f);
            std::vector<float> my(size_t(cells), 0.0f);
            std::vector<int>   n(size_t(cells), 0);

            // Sample the patch, accumulating into whichever cell each sample
            // falls in. Sampling the patch once and binning is cheaper than
            // walking each cell separately, and it guarantees the cells tile.
            const int span = std::max(2, int(patch * 0.5f));
            for (int dy = -span; dy <= span; ++dy)
                for (int dx = -span; dx <= span; ++dx) {
                    // Rotate into the keypoint's frame.
                    const float rx = float(dx) * cosA + float(dy) * sinA;
                    const float ry = -float(dx) * sinA + float(dy) * cosA;
                    const float u = (rx / patch + 0.5f) * float(grid);
                    const float v = (ry / patch + 0.5f) * float(grid);
                    if (u < 0.0f || v < 0.0f || u >= float(grid) || v >= float(grid))
                        continue;

                    const int cx = std::clamp(int(u), 0, grid - 1);
                    const int cy = std::clamp(int(v), 0, grid - 1);
                    const int ci = cy * grid + cx;

                    const int px = int(kp.x) + dx, py = int(kp.y) + dy;
                    mi[size_t(ci)] += p.At(px, py);
                    mx[size_t(ci)] += p.At(px + 1, py) - p.At(px - 1, py);
                    my[size_t(ci)] += p.At(px, py + 1) - p.At(px, py - 1);
                    ++n[size_t(ci)];
                }

            for (int i = 0; i < cells; ++i)
                if (n[size_t(i)] > 0) {
                    mi[size_t(i)] /= float(n[size_t(i)]);
                    mx[size_t(i)] /= float(n[size_t(i)]);
                    my[size_t(i)] /= float(n[size_t(i)]);
                }

            // One bit per pair per channel.
            for (int i = 0; i < cells; ++i)
                for (int j = i + 1; j < cells; ++j) {
                    SetBit(desc, bit++, mi[size_t(i)] > mi[size_t(j)]);
                    SetBit(desc, bit++, mx[size_t(i)] > mx[size_t(j)]);
                    SetBit(desc, bit++, my[size_t(i)] > my[size_t(j)]);
                }
        }
    }

    static void SetBit(uint8_t* d, int bit, bool on) {
        if (bit < 0 || bit >= kDescBits) return;
        if (on) d[bit / 8] |= uint8_t(1u << (bit % 8));
    }

    Param<int> m_octaves{this, "octaves", 3, 1, 4,
        {.help = "How many scale doublings to search."}};

    Param<int> m_scales{this, "scales_per_octave", 3, 1, 4,
        {.help = "Diffusion levels within each octave."}};

    Param<float> m_threshold{this, "threshold", 0.0008f, 0.0f, 0.02f,
        {.help = "Minimum scale-normalised Hessian determinant. AKAZE's edges "
                 "stay sharp at every scale, so a feature found coarsely sits "
                 "where the fine-scale one does -- which is the reason to use "
                 "it over SIFT.",
         .step = 0.0002, .softMax = 0.005}};

    Param<float> m_contrastPercentile{this, "contrast_k", 1.0f, 0.2f, 3.0f,
        {.help = "Scales the measured contrast factor, which decides which "
                 "gradients count as edges to stop diffusing at. Measured from "
                 "the image rather than fixed: a value tuned on a contrasty "
                 "frame stops all diffusion on a flat one.",
         .step = 0.05}};

    Param<bool> m_upright{this, "upright", false,
        "Skip orientation and assume the camera is level. Faster and more "
        "repeatable when the images really are not rotated."};

    Param<int> m_maxFeatures{this, "max_features", 5000, 10, 50000,
        {.help = "Stop after this many."}};

    int m_found = 0;
};

} // namespace

REGISTER_ALGORITHM(DetectAkaze);

} // namespace tglab
