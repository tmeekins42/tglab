// detect_sift — Lowe's Scale-Invariant Feature Transform (2004).
//
// The patent expired in March 2020, so this is free to implement and ship.
//
// WHAT IT IS, in the order the code does it:
//
//   1. A Gaussian scale space: `octaves` resolutions, each holding
//      `scalesPerOctave + 3` blurs. Difference-of-Gaussian between adjacent
//      blurs approximates the Laplacian, whose extrema are blob centres.
//   2. Extrema of the DoG in 3x3x3 neighbourhoods -- across x, y AND scale,
//      which is what makes the result scale-invariant rather than merely
//      multi-resolution.
//   3. Sub-pixel refinement by fitting a quadratic, then two rejections: low
//      contrast, and edge responses (a ridge is extremal along one direction
//      only, and its position along the ridge is not repeatable).
//   4. A dominant orientation from a gradient histogram, so the descriptor can
//      be measured in a rotated frame.
//   5. The descriptor: 4x4 spatial cells, 8 orientation bins, 128 floats.
//
// WHY EACH REJECTION MATTERS, since without them the detector still "works"
// and produces features that will not match:
//
//   - low contrast: a DoG extremum in a flat region is noise, and its
//     descriptor is noise. These are the majority of raw extrema.
//   - edge response: the Hessian's eigenvalue RATIO distinguishes a corner
//     (both large) from an edge (one large). An edge point slides along its
//     own edge between frames, so it matches the wrong place.
//
// The threshold is on the eigenvalue ratio rather than the eigenvalues, using
// trace^2/det -- which needs no square roots and is the standard formulation.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "../../algo_util/features.h"
#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"
#include "gpu_pyramid.h"

namespace tglab {
namespace {

// One level of the scale space: a blurred greyscale plane.
struct Plane {
    std::vector<float> v;
    int w = 0, h = 0;

    float At(int x, int y) const {
        return v[size_t(std::clamp(y, 0, h - 1)) * size_t(w) +
                 size_t(std::clamp(x, 0, w - 1))];
    }
};

// Separable Gaussian blur. Radius from sigma at 3 sigma, which captures 99.7%
// of the kernel -- the same rule gaussian_blur uses, and worth matching so two
// blurs in this codebase do not mean subtly different things.
void Blur(const Plane& src, Plane& dst, float sigma, std::vector<float>& tmp) {
    const int w = src.w, h = src.h;
    const int r = std::max(1, int(std::ceil(sigma * 3.0f)));

    std::vector<float> k(size_t(r) * 2 + 1);
    float sum = 0.0f;
    for (int i = -r; i <= r; ++i) {
        const float e = std::exp(-float(i * i) / (2.0f * sigma * sigma));
        k[size_t(i + r)] = e;
        sum += e;
    }
    for (float& x : k) x /= sum;

    tmp.assign(size_t(w) * size_t(h), 0.0f);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float a = 0.0f;
            for (int i = -r; i <= r; ++i) a += k[size_t(i + r)] * src.At(x + i, y);
            tmp[size_t(y) * size_t(w) + size_t(x)] = a;
        }

    dst.w = w; dst.h = h;
    dst.v.assign(size_t(w) * size_t(h), 0.0f);
    const Plane mid{tmp, w, h};
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float a = 0.0f;
            for (int i = -r; i <= r; ++i) a += k[size_t(i + r)] * mid.At(x, y + i);
            dst.v[size_t(y) * size_t(w) + size_t(x)] = a;
        }
}

// Halves a plane by taking every second sample.
//
// Point sampling rather than averaging, deliberately: the source has already
// been blurred to the octave's base sigma, so it is band-limited and decimating
// it is correct. Averaging again would over-smooth and shift the scale space.
Plane Halve(const Plane& src) {
    Plane out;
    out.w = std::max(1, src.w / 2);
    out.h = std::max(1, src.h / 2);
    out.v.assign(size_t(out.w) * size_t(out.h), 0.0f);
    for (int y = 0; y < out.h; ++y)
        for (int x = 0; x < out.w; ++x)
            out.v[size_t(y) * size_t(out.w) + size_t(x)] = src.At(x * 2, y * 2);
    return out;
}

class DetectSift : public AlgorithmBase {
public:
    const char* Name()     const override { return "detect_sift"; }
    const char* Category() const override { return "features"; }

    PortList Inputs()  const override { return {{"src", DataType::Image, FormatSpec::Any}}; }

    // Passes the image through and attaches the features.
    //
    // Same shape as align: the output IS the input, with a sidecar added.
    // A detector that produced a different image would force every script to
    // choose between "the picture" and "the picture with features", when what
    // it actually wants is one image that now knows about its own features.
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

        // Copy the pixels through unchanged first: the features are the
        // product, and the image is passed along so the chain continues.
        PixelBuffer out;
        out.Unpack(dst);
        if (out.Valid()) {
            out.Data() = in.Data();
            out.PackInto(dst);
        }

        m_found = 0;
        if (w < 16 || h < 16) return;    // nothing meaningful to find

        // Greyscale, normalised to 0..1 so the contrast threshold means the
        // same thing on an 8-bit scan and a float raw.
        Plane base;
        base.w = w; base.h = h;
        base.v.assign(size_t(w) * size_t(h), 0.0f);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const float* p = in.At(x, y);
                const float g = (ch >= 3)
                    ? (0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2]) / scale
                    : p[0] / scale;
                base.v[size_t(y) * size_t(w) + size_t(x)] = g;
            }

        // ...and then normalised again, by the image's OWN level.
        //
        // ValueScale() handles the FORMAT -- 8-bit to 0..1 -- and that is all it
        // can handle. A scene-referred float raw is already "in 0..1" and still
        // occupies the bottom sixth of it: measured on a 15-frame panorama, the
        // median pixel was 0.058 and the 99th percentile 0.16. Lowe's contrast
        // default of 0.02 was calibrated against display-referred images whose
        // median sits near 0.5, so against a raw it rejects nearly everything
        // real. It is not a threshold that is slightly too high; it is a
        // threshold measured in the wrong units.
        //
        // The symptom was invisible in the ordinary way. Feature counts fell
        // along the pan -- 354, 63, 6, 2 -- which reads exactly like a scene
        // running out of texture, and the frames that failed were the flatter
        // ones. Dropping the threshold to 0.005 took the worst frame from 2
        // features to 1562, which is what proved it was never the scene.
        //
        // AKAZE already did this right, measuring its contrast factor as a
        // gradient percentile with the comment that a value tuned on a
        // high-contrast image misbehaves on a low-contrast one. Same disease,
        // and it had the cure first.
        //
        // The 99th percentile rather than the max, because the max is one hot
        // pixel or one specular highlight; and the same statistic is used by
        // SURF, so a threshold means the same thing in both detectors.
        m_level = Percentile99(base.v);
        if (m_level > 1e-6f) {
            const float inv = 1.0f / m_level;
            for (float& v : base.v) v *= inv;
        }

        auto sidecar = std::make_shared<FeatureSidecar>();
        sidecar->detector = "sift";
        sidecar->descriptors.kind = DescriptorKind::Float;
        sidecar->descriptors.dim  = kDescDim;

        Detect(base, ctx, sidecar.get());
        m_found = int(sidecar->keypoints.size());

        // Attach to the OUTPUT, which is what downstream stages resolve.
        if (Image* im = ctx.OutImage(0)) im->Sidecars().Set(kFeatureSidecar, sidecar);
    }

    std::string RunReport() const override {
        if (m_found <= 0) return {};
        char buf[64];
        std::snprintf(buf, sizeof buf, "%d SIFT features (level %.3f)",
                      m_found, m_level);
        return buf;
    }

    // Sidecar coordinates are in IMAGE PIXELS and nothing rescales them, so a
    // proxy run would hand every downstream stage positions that are wrong by
    // the scale factor -- silently, since the sidecar is still present and
    // still looks valid. See AlgorithmBase::Proxy.
    ProxyBehaviour Proxy() const override { return ProxyBehaviour::Never; }

    bool HasGPU() const override { return false; }

private:
    static constexpr int kDescDim   = 128;   // 4x4 cells x 8 orientations
    static constexpr int kOriBins   = 36;    // orientation histogram resolution

    void Detect(const Plane& base, RunCtx& ctx, FeatureSidecar* out) {
        const int octaves = std::clamp(int(m_octaves), 1, 8);
        const int perOct  = std::clamp(int(m_scales), 1, 8);
        const float sigma0 = std::max(0.6f, float(m_sigma));

        // Blurs per octave: perOct + 3 planes give perOct + 2 DoG levels, of
        // which perOct are searchable (an extremum needs a neighbour above and
        // below in scale). That +3 is not arbitrary -- it is what makes the
        // octaves overlap so no scale falls between them.
        const int planes = perOct + 3;
        const float k = std::pow(2.0f, 1.0f / float(perOct));

        std::vector<float> tmp;
        Plane octaveBase = base;

        for (int o = 0; o < octaves && octaveBase.w >= 16 && octaveBase.h >= 16; ++o) {
            if (ctx.Cancelled()) return;

            // The blur stack for this octave.
            //
            // Incremental: blurring an already-blurred plane by the DIFFERENCE
            // in sigma composes to the total, and is far cheaper than
            // re-blurring the base by a growing kernel.
            std::vector<float> adds;
            adds.reserve(size_t(planes - 1));
            for (int i = 1; i < planes; ++i) {
                const float prev = sigma0 * std::pow(k, float(i - 1));
                const float cur  = sigma0 * std::pow(k, float(i));
                adds.push_back(std::sqrt(std::max(cur * cur - prev * prev, 0.01f)));
            }

            // resize() rather than a sized constructor: `vector<Plane> g(size_t(n))`
            // parses as a function declaration (the most vexing parse) and the
            // errors land on the USES rather than here.
            std::vector<Plane> gauss;

            // The GPU builds the whole stack in one call, keeping the chain
            // resident between levels; a per-level helper would read each plane
            // back and upload it again, which at 22 MP costs more than the blur.
            //
            // Failure falls back rather than failing the run: a missing device
            // or a dispatch error should make this slow, not broken. The two
            // paths agree because gpu_pyramid reproduces Blur()'s radius rule,
            // normalisation, addressing, and pass order -- see its header.
            bool onGpu = false;
            if (ComputeContext* dev = ctx.Gpu()) {
                GpuPlane gbase;
                gbase.v = octaveBase.v;
                gbase.w = octaveBase.w;
                gbase.h = octaveBase.h;

                std::vector<GpuPlane> stack;
                std::string gerr;
                if (GpuBlurStack(dev, gbase, adds, &stack, &gerr) &&
                    stack.size() == size_t(planes)) {
                    gauss.resize(size_t(planes));
                    for (int i = 0; i < planes; ++i) {
                        gauss[size_t(i)].v = std::move(stack[size_t(i)].v);
                        gauss[size_t(i)].w = stack[size_t(i)].w;
                        gauss[size_t(i)].h = stack[size_t(i)].h;
                    }
                    onGpu = true;
                }
            }

            if (!onGpu) {
                gauss.resize(size_t(planes));
                gauss[0] = octaveBase;
                for (int i = 1; i < planes; ++i)
                    Blur(gauss[size_t(i - 1)], gauss[size_t(i)],
                         adds[size_t(i - 1)], tmp);
            }

            // Difference of Gaussian.
            std::vector<Plane> dog;
            dog.resize(size_t(planes - 1));
            for (int i = 0; i + 1 < planes; ++i) {
                dog[size_t(i)].w = octaveBase.w;
                dog[size_t(i)].h = octaveBase.h;
                dog[size_t(i)].v.assign(gauss[0].v.size(), 0.0f);
                for (size_t j = 0; j < dog[size_t(i)].v.size(); ++j)
                    dog[size_t(i)].v[j] = gauss[size_t(i + 1)].v[j] - gauss[size_t(i)].v[j];
            }

            FindExtrema(gauss, dog, o, perOct, sigma0, k, ctx, out);
            if (ctx.Cancelled()) return;

            // The next octave starts from the plane whose blur is exactly twice
            // the base -- index perOct -- so halving it lands on the right sigma
            // rather than needing a fresh blur.
            octaveBase = Halve(gauss[size_t(perOct)]);
        }

        // RANK, THEN CAP, across every octave.
        //
        // This was a `return` the moment the count hit max_features, in the
        // middle of a raster scan -- so a frame with more extrema than the cap
        // kept everything above some scanline and NOTHING below it. The loss is
        // SPATIAL, not merely numerical: features are found top-to-bottom, so
        // what is discarded is a contiguous band at the bottom of the frame,
        // and if that band is where the next frame overlaps, the pair has
        // nothing to match on.
        //
        // Tim found it on AKAZE, which had the same bug: the right-hand end of
        // a panorama failed to align while the feature count sat pinned at the
        // cap. Measured there, on the pair that failed: 2% of candidates kept
        // and 0% inliers before, 11% and 64% after -- at the SAME cap. The
        // features existed; the scan never reached them.
        //
        // Done here rather than inside FindExtrema because SIFT's descriptor
        // needs that octave's Gaussian plane, which is gone by the time the
        // next octave runs. So candidates are described as they are found and
        // the cap sorts the finished arrays -- which costs describing some that
        // are then dropped, and is still far cheaper than being wrong.
        const int cap = std::max(1, int(m_maxFeatures));
        if (int(out->keypoints.size()) > cap) {
            std::vector<int> order;
            order.resize(out->keypoints.size());
            for (size_t i = 0; i < order.size(); ++i) order[i] = int(i);
            std::nth_element(order.begin(), order.begin() + cap, order.end(),
                             [&](int a, int b) {
                                 return std::abs(out->keypoints[size_t(a)].response) >
                                        std::abs(out->keypoints[size_t(b)].response);
                             });
            order.resize(size_t(cap));

            std::vector<Keypoint> keep;
            std::vector<float>    kd;
            keep.reserve(order.size());
            kd.reserve(order.size() * size_t(kDescDim));
            for (int idx : order) {
                keep.push_back(out->keypoints[size_t(idx)]);
                const float* d = &out->descriptors.f[size_t(idx) * size_t(kDescDim)];
                kd.insert(kd.end(), d, d + kDescDim);
            }
            out->keypoints.swap(keep);
            out->descriptors.f.swap(kd);
        }
    }

    void FindExtrema(const std::vector<Plane>& gauss, const std::vector<Plane>& dog,
                     int octave, int perOct, float sigma0, float k,
                     RunCtx& ctx, FeatureSidecar* out) {
        const int w = dog[0].w, h = dog[0].h;
        const float contrast = float(m_contrast);

        // The scale factor from this octave's coordinates back to the input's.
        const float octScale = float(1 << octave);

        for (size_t s = 1; s + 1 < dog.size(); ++s) {
            if (ctx.Cancelled()) return;
            const Plane& below = dog[s - 1];
            const Plane& here  = dog[s];
            const Plane& above = dog[s + 1];

            for (int y = 1; y < h - 1; ++y) {
                for (int x = 1; x < w - 1; ++x) {
                    const float v = here.At(x, y);

                    // Cheap rejection before the 26-neighbour scan: most points
                    // are nowhere near extremal, and this is the inner loop.
                    if (std::fabs(v) < contrast * 0.5f) continue;

                    bool isMax = true, isMin = true;
                    for (int dy = -1; dy <= 1 && (isMax || isMin); ++dy)
                        for (int dx = -1; dx <= 1; ++dx) {
                            const float a = below.At(x + dx, y + dy);
                            const float b = here.At(x + dx, y + dy);
                            const float c = above.At(x + dx, y + dy);
                            const float m = std::max({a, b, c});
                            const float n = std::min({a, b, c});
                            // `b` at the centre is v itself, so a strict test
                            // would always fail; compare against the others.
                            if (dx == 0 && dy == 0) {
                                if (a >= v || c >= v) isMax = false;
                                if (a <= v || c <= v) isMin = false;
                            } else {
                                if (m >= v) isMax = false;
                                if (n <= v) isMin = false;
                            }
                            if (!isMax && !isMin) break;
                        }
                    if (!isMax && !isMin) continue;

                    Keypoint kp;
                    if (!Refine(dog, s, x, y, octave, perOct, sigma0, k, &kp)) continue;

                    // Orientation, then the descriptor, both from the GAUSSIAN
                    // plane at this scale rather than the DoG: the DoG is a
                    // band-pass and its gradients are not the image's.
                    const Plane& g = gauss[s];
                    std::vector<float> angles;
                    Orientations(g, kp, &angles);

                    for (float ang : angles) {
                        Keypoint k2 = kp;
                        k2.angle = ang;
                        // Back to input coordinates. Everything above works in
                        // octave space; the sidecar is read by algorithms that
                        // know only the original image.
                        Keypoint k3 = k2;
                        k3.x     = k2.x * octScale;
                        k3.y     = k2.y * octScale;
                        k3.scale = k2.scale * octScale;

                        std::vector<float> desc(kDescDim, 0.0f);
                        Describe(g, k2, ang, desc.data());

                        out->keypoints.push_back(k3);
                        out->descriptors.f.insert(out->descriptors.f.end(),
                                                  desc.begin(), desc.end());
                    }
                }
            }
        }
    }

    // Sub-pixel position by fitting a quadratic to the DoG, plus the two
    // rejections that make the result matchable.
    bool Refine(const std::vector<Plane>& dog, size_t s, int x, int y,
                int octave, int perOct, float sigma0, float k, Keypoint* out) const {
        const Plane& c = dog[s];
        const Plane& below = dog[s - 1];
        const Plane& above = dog[s + 1];

        // First derivatives, central differences.
        const float dx = 0.5f * (c.At(x + 1, y) - c.At(x - 1, y));
        const float dy = 0.5f * (c.At(x, y + 1) - c.At(x, y - 1));
        const float ds = 0.5f * (above.At(x, y) - below.At(x, y));

        // Second derivatives, for the 3x3 Hessian.
        const float v2 = 2.0f * c.At(x, y);
        const float dxx = c.At(x + 1, y) + c.At(x - 1, y) - v2;
        const float dyy = c.At(x, y + 1) + c.At(x, y - 1) - v2;
        const float dss = above.At(x, y) + below.At(x, y) - v2;
        const float dxy = 0.25f * (c.At(x + 1, y + 1) - c.At(x - 1, y + 1) -
                                   c.At(x + 1, y - 1) + c.At(x - 1, y - 1));
        const float dxs = 0.25f * (above.At(x + 1, y) - above.At(x - 1, y) -
                                   below.At(x + 1, y) + below.At(x - 1, y));
        const float dys = 0.25f * (above.At(x, y + 1) - above.At(x, y - 1) -
                                   below.At(x, y + 1) + below.At(x, y - 1));

        // Solve H * offset = -gradient by Cramer's rule. 3x3, so an explicit
        // determinant is clearer and faster than a general solver.
        const float det =
            dxx * (dyy * dss - dys * dys) -
            dxy * (dxy * dss - dys * dxs) +
            dxs * (dxy * dys - dyy * dxs);
        if (std::fabs(det) < 1e-12f) return false;

        const float inv = 1.0f / det;
        const float ox = -inv * ( dx * (dyy * dss - dys * dys) -
                                  dy * (dxy * dss - dys * dxs) +
                                  ds * (dxy * dys - dyy * dxs));
        const float oy = -inv * (dxx * ( dy * dss - ds * dys) -
                                 dxy * ( dx * dss - ds * dxs) +
                                 dxs * ( dx * dys - dy * dxs));
        const float os = -inv * (dxx * (dyy * ds - dys * dy) -
                                 dxy * (dxy * ds - dys * dx) +
                                 dxs * (dxy * dy - dyy * dx));

        // A fit that lands more than half a sample away is describing a
        // different pixel than the one tested, so it is discarded rather than
        // iterated -- the neighbouring sample will be tested on its own.
        if (std::fabs(ox) > 0.5f || std::fabs(oy) > 0.5f || std::fabs(os) > 0.5f)
            return false;

        // Contrast AT THE REFINED POSITION, which is the value the quadratic
        // predicts rather than the sampled one.
        const float peak = c.At(x, y) + 0.5f * (dx * ox + dy * oy + ds * os);
        if (std::fabs(peak) < float(m_contrast)) return false;

        // Edge rejection by the eigenvalue ratio of the 2x2 spatial Hessian.
        //
        // trace^2/det = (r+1)^2/r for eigenvalue ratio r, which is monotonic in
        // r and needs no square roots. A corner has both eigenvalues large and
        // a small ratio; an edge has one large and one near zero.
        const float tr  = dxx + dyy;
        const float dt  = dxx * dyy - dxy * dxy;
        if (dt <= 0.0f) return false;      // saddle, not an extremum in 2D
        const float r = float(m_edgeRatio);
        if (tr * tr / dt >= (r + 1.0f) * (r + 1.0f) / r) return false;

        out->x = float(x) + ox;
        out->y = float(y) + oy;
        out->response = std::fabs(peak);
        out->octave = octave;
        // The scale this was found at, in this octave's pixels.
        out->scale = sigma0 * std::pow(k, float(s) + os);
        return true;
    }

    // Dominant orientations from a gradient histogram over the neighbourhood.
    //
    // Returns MORE THAN ONE when the histogram has several strong peaks, which
    // is Lowe's rule and is worth keeping: a corner where two edges meet has
    // two equally good frames, and committing to one arbitrarily makes the
    // descriptor unrepeatable. Duplicating the keypoint at each is what makes
    // such points matchable at all.
    void Orientations(const Plane& g, const Keypoint& kp,
                      std::vector<float>* out) const {
        float hist[kOriBins] = {};
        const float sigma = 1.5f * kp.scale;
        const int   rad   = std::max(1, int(std::round(3.0f * sigma)));
        const int   cx = int(std::round(kp.x)), cy = int(std::round(kp.y));

        for (int dy = -rad; dy <= rad; ++dy)
            for (int dx = -rad; dx <= rad; ++dx) {
                const int x = cx + dx, y = cy + dy;
                if (x < 1 || y < 1 || x >= g.w - 1 || y >= g.h - 1) continue;

                const float gx = g.At(x + 1, y) - g.At(x - 1, y);
                const float gy = g.At(x, y + 1) - g.At(x, y - 1);
                const float mag = std::sqrt(gx * gx + gy * gy);
                const float ang = std::atan2(gy, gx);

                // Gaussian weighted, so a sample at the edge of the window does
                // not count as much as one at the centre.
                const float wgt = std::exp(-float(dx * dx + dy * dy) /
                                           (2.0f * sigma * sigma));
                int bin = int(std::floor((ang + 3.14159265f) /
                                         (2.0f * 3.14159265f) * kOriBins));
                bin = std::clamp(bin, 0, kOriBins - 1);
                hist[bin] += mag * wgt;
            }

        // Smooth the histogram before looking for peaks.
        //
        // Not cosmetic. 36 raw bins from a few hundred weighted samples are
        // noisy, so several ADJACENT bins clear the 80% test and each is
        // reported as its own peak -- which duplicates the keypoint two or
        // three times over at essentially the same angle. Measured before this
        // was added: 63% of keypoints shared a position with another, and the
        // ratio test then rejected nearly all of them, since a keypoint's
        // near-identical twin is its own second-best match. Matching kept 13
        // of 275 features.
        //
        // Lowe smooths six times; three passes of the same [1 4 6 4 1]/16
        // kernel gets the duplicates down without flattening real double
        // peaks, which are what the rule exists to keep.
        for (int pass = 0; pass < 3; ++pass) {
            float sm[kOriBins];
            for (int i = 0; i < kOriBins; ++i) {
                const float m2 = hist[(i + kOriBins - 2) % kOriBins];
                const float m1 = hist[(i + kOriBins - 1) % kOriBins];
                const float c0 = hist[i];
                const float p1 = hist[(i + 1) % kOriBins];
                const float p2 = hist[(i + 2) % kOriBins];
                sm[i] = (m2 + 4.0f * m1 + 6.0f * c0 + 4.0f * p1 + p2) / 16.0f;
            }
            for (int i = 0; i < kOriBins; ++i) hist[i] = sm[i];
        }

        float peak = 0.0f;
        for (float v : hist) peak = std::max(peak, v);
        if (peak <= 0.0f) { out->push_back(0.0f); return; }

        // Every peak within 80% of the strongest, Lowe's threshold.
        for (int i = 0; i < kOriBins; ++i) {
            const float l = hist[(i + kOriBins - 1) % kOriBins];
            const float c = hist[i];
            const float r = hist[(i + 1) % kOriBins];
            if (c < peak * 0.8f || c < l || c < r) continue;

            // Parabolic interpolation between the neighbours, so the angle is
            // finer than the bin width.
            const float denom = l - 2.0f * c + r;
            const float off = (std::fabs(denom) > 1e-9f) ? 0.5f * (l - r) / denom : 0.0f;
            const float b = float(i) + off;
            out->push_back(b / kOriBins * 2.0f * 3.14159265f - 3.14159265f);
        }
        if (out->empty()) out->push_back(0.0f);
    }

    // The 128-float descriptor: 4x4 spatial cells, 8 orientation bins each.
    void Describe(const Plane& g, const Keypoint& kp, float angle, float* desc) const {
        constexpr int kCells = 4, kBins = 8;
        const float cosA = std::cos(-angle), sinA = std::sin(-angle);

        // Window size: 3 sigma per cell is Lowe's choice, and the sqrt(2)
        // covers the corners of the rotated square.
        const float cellSize = 3.0f * kp.scale;
        const float rad = cellSize * (kCells + 1) * 0.5f * 1.4142f;
        const int   irad = std::max(1, int(std::round(rad)));
        const int   cx = int(std::round(kp.x)), cy = int(std::round(kp.y));

        for (int dy = -irad; dy <= irad; ++dy)
            for (int dx = -irad; dx <= irad; ++dx) {
                const int x = cx + dx, y = cy + dy;
                if (x < 1 || y < 1 || x >= g.w - 1 || y >= g.h - 1) continue;

                // Into the keypoint's rotated frame, which is what makes the
                // descriptor rotation invariant.
                const float rx = (float(dx) * cosA - float(dy) * sinA) / cellSize;
                const float ry = (float(dx) * sinA + float(dy) * cosA) / cellSize;

                // Cell coordinates, centred so the 4x4 grid spans -2..2.
                const float bx = rx + kCells * 0.5f - 0.5f;
                const float by = ry + kCells * 0.5f - 0.5f;
                if (bx <= -1.0f || bx >= kCells || by <= -1.0f || by >= kCells) continue;

                const float gx = g.At(x + 1, y) - g.At(x - 1, y);
                const float gy = g.At(x, y + 1) - g.At(x, y - 1);
                float mag = std::sqrt(gx * gx + gy * gy);
                float ang = std::atan2(gy, gx) + angle;    // relative to the frame
                while (ang < 0.0f) ang += 2.0f * 3.14159265f;
                while (ang >= 2.0f * 3.14159265f) ang -= 2.0f * 3.14159265f;

                mag *= std::exp(-(rx * rx + ry * ry) / (0.5f * kCells * kCells));

                // Trilinear accumulation across the two spatial axes and the
                // orientation axis. Without it the descriptor jumps as a
                // gradient crosses a cell boundary, which is exactly the
                // instability that makes matching fail.
                const float ob = ang / (2.0f * 3.14159265f) * kBins;
                const int   x0 = int(std::floor(bx)), y0 = int(std::floor(by));
                const int   o0 = int(std::floor(ob));
                const float fx = bx - float(x0), fy = by - float(y0), fo = ob - float(o0);

                for (int iy = 0; iy <= 1; ++iy)
                    for (int ix = 0; ix <= 1; ++ix)
                        for (int io = 0; io <= 1; ++io) {
                            const int cxi = x0 + ix, cyi = y0 + iy;
                            if (cxi < 0 || cxi >= kCells || cyi < 0 || cyi >= kCells) continue;
                            const int oi = (o0 + io) % kBins;
                            const float wx = ix ? fx : 1.0f - fx;
                            const float wy = iy ? fy : 1.0f - fy;
                            const float wo = io ? fo : 1.0f - fo;
                            desc[(cyi * kCells + cxi) * kBins + oi] += mag * wx * wy * wo;
                        }
            }

        // Normalise, clip, renormalise -- Lowe's recipe.
        //
        // The clip is the part worth explaining: a large gradient from a
        // specular highlight or a strong edge would otherwise dominate the
        // whole vector, so no single element may exceed 0.2 of the norm. That
        // makes the descriptor robust to illumination change, which is most of
        // what it is for.
        auto normalise = [&] {
            float n = 0.0f;
            for (int i = 0; i < kDescDim; ++i) n += desc[i] * desc[i];
            n = std::sqrt(n);
            if (n > 1e-9f) for (int i = 0; i < kDescDim; ++i) desc[i] /= n;
        };
        normalise();
        for (int i = 0; i < kDescDim; ++i) desc[i] = std::min(desc[i], 0.2f);
        normalise();
    }

    Param<int> m_octaves{this, "octaves", 4, 1, 8,
        {.help = "How many resolutions to search. Each is half the last, so "
                 "more octaves find larger blobs at the cost of time."}};

    Param<int> m_scales{this, "scales_per_octave", 3, 1, 8,
        {.help = "Scale samples within each octave. Lowe found 3 best; more "
                 "finds features at intermediate scales but repeats them."}};

    Param<float> m_sigma{this, "sigma", 1.6f, 0.6f, 3.0f,
        {.help = "Blur of the first scale-space level. 1.6 is Lowe's value, "
                 "chosen to leave the input's own sampling blur intact.",
         .step = 0.05}};

    // 0.04 rather than Lowe's 0.02, and the difference is the normalisation
    // above rather than a disagreement with Lowe.
    //
    // Once the input is divided by its own p99, this threshold finally means
    // what it says -- and 0.02 turns out to be too permissive when it is
    // actually enforced. MEASURED, on the synthetic fixture (translate the
    // scene and count how many features come back):
    //
    //   0.02:  46 found, 22% on the blobs, 20% repeat
    //   0.03:  11 found, 91% on the blobs, 82% repeat
    //   0.04:  10 found, 100% on the blobs, 90% repeat
    //
    // The 36 extra features at 0.02 are not weaker features, they are noise:
    // the same 9 stable ones survive at every setting. And the real data
    // agrees, which is what settled it -- on a 45 MP panorama pair the RANSAC
    // inlier rate rose 52% -> 64% -> 72% across 0.02 / 0.04 / 0.06, while 0.04
    // still left 400-1300 matches, far more than a homography needs.
    //
    // 0.06 is better still on inlier rate alone. 0.04 is the knee: it takes
    // nearly all of the gain while keeping the match count high enough that a
    // low-overlap pair does not starve.
    Param<float> m_contrast{this, "contrast", 0.04f, 0.0f, 0.2f,
        {.help = "Minimum DoG response, relative to the image's own 99th "
                 "percentile. Raise it to keep only strong features; too low "
                 "and most of what is found is noise in flat regions.",
         .step = 0.002, .softMax = 0.08}};

    Param<float> m_edgeRatio{this, "edge_ratio", 10.0f, 2.0f, 50.0f,
        {.help = "Largest allowed ratio between the Hessian's eigenvalues. "
                 "Rejects points on edges, which slide along their own edge "
                 "between frames and so match the wrong place.",
         .step = 0.5}};

    Param<int> m_maxFeatures{this, "max_features", 5000, 10, 50000,
        {.help = "Stop after this many. A bound on time and memory rather "
                 "than a quality control -- the strongest are not found first."}};

    int   m_found = 0;
    // Reported, because a detector that silently rescales its input is a
    // detector whose threshold no longer means what the slider says. Seeing
    // "level 0.16" next to a raw is what makes the normalisation checkable.
    float m_level = 0.0f;
};

} // namespace

REGISTER_ALGORITHM(DetectSift);

} // namespace tglab
