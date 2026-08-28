// align: solves a sub-pixel affine warp per frame and attaches it as a sidecar.
//
// The fine end of alignment. It assumes the frames are already close -- a
// tripod that shifted a pixel or two between exposures, which is what Tim
// measured on hdrtest2 -- and refines that to a fraction of a pixel. Large
// displacement is the feature-matching solver's job, and this is designed to
// run AFTER it: the solve starts from whatever transform is already attached,
// so a coarse estimate from features gets refined here rather than discarded.
//
// It attaches rather than warping, so nothing is resampled here at all. A merge
// downstream samples through the transform, which means ONE interpolation
// instead of two and no second copy of the group.
//
// Method: inverse-compositional Lucas-Kanade on sampled points.
//
//   - DIRECT, not correspondence-based. On a photograph the nearest pixel by
//     value is ambiguous -- a sky pixel matches a million others -- which is
//     the aperture problem, and it is why ICP is the wrong tool here. Comparing
//     intensities under a warp has no such ambiguity.
//
//   - INVERSE-COMPOSITIONAL, so the Jacobian and its Hessian are computed once
//     on the reference rather than per iteration. At 22 MP that is the
//     difference between a solve that takes a second and one that takes a
//     minute.
//
//   - GRADIENT-WEIGHTED SAMPLING, not uniform random. A pixel in flat sky
//     contributes nothing to a displacement estimate: its residual is the same
//     whichever way the frame moved. An edge pixel constrains it strongly. So
//     the sample set is drawn from the strongest gradients, which is why a few
//     tens of thousands of points can beat hundreds of thousands taken at
//     random.
//
//   - COARSE TO FINE, so a displacement larger than the gradient's own scale
//     can still be found. Lucas-Kanade linearises around the current estimate
//     and only converges within roughly a pixel per level; the pyramid is what
//     extends that.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../../algo_util/pixel_buffer.h"
#include "../../algo_util/transform.h"
#include "../../core/algorithm.h"

namespace tglab {
namespace {

// Per-frame solve detail, off unless TGLAB_ALIGN is set. A maximum hides the
// shape of the answer, and the shape is what says whether a solve worked.
static bool AlignVerbose() {
    static const bool on = [] {
        size_t n = 0;
        return getenv_s(&n, nullptr, 0, "TGLAB_ALIGN") == 0 && n > 0;
    }();
    return on;
}

// A luminance plane, which is all the solver reads.
struct Plane {
    std::vector<float>   v;
    // Per-pixel validity: 0 where the sample is clipped or crushed and cannot
    // be compared across exposures. See MakeLuma.
    std::vector<uint8_t> ok;
    int w = 0, h = 0;

    // The pivot for rotation and scale, and the origin the solve measures from.
    // See PickPoints: anchoring at the corner makes any scale error walk the
    // image centre, which is a visible global shift.
    float cx = 0.0f, cy = 0.0f;

    bool Ok(int x, int y) const {
        if (ok.empty()) return true;
        x = std::clamp(x, 0, w - 1);
        y = std::clamp(y, 0, h - 1);
        return ok[size_t(y) * size_t(w) + size_t(x)] != 0;
    }

    float At(int x, int y) const {
        x = std::clamp(x, 0, w - 1);
        y = std::clamp(y, 0, h - 1);
        return v[size_t(y) * size_t(w) + size_t(x)];
    }

    float Sample(float fx, float fy) const {
        const float cx = std::clamp(fx, 0.0f, float(w - 1));
        const float cy = std::clamp(fy, 0.0f, float(h - 1));
        const int x0 = int(cx), y0 = int(cy);
        const float tx = cx - float(x0), ty = cy - float(y0);
        if (tx == 0.0f && ty == 0.0f) return At(x0, y0);
        const float a = At(x0, y0), b = At(x0 + 1, y0);
        const float c = At(x0, y0 + 1), d = At(x0 + 1, y0 + 1);
        return (a + (b - a) * tx) + ((c + (d - c) * tx) - (a + (b - a) * tx)) * ty;
    }
};

// Luminance, scaled by the frame's own exposure so two frames of a bracket are
// directly comparable.
//
// Dividing the exposure out is free and correct -- EXIF already carries it --
// and it is much better than making brightness a solver unknown, which would
// trade off against genuine brightness changes in the scene. Frames with no
// exposure metadata simply compare as-is.
Plane MakeLuma(const PixelBuffer& pb, float exposure) {
    Plane p;
    p.w = pb.Width();
    p.h = pb.Height();
    p.v.resize(size_t(p.w) * size_t(p.h));
    p.ok.resize(size_t(p.w) * size_t(p.h), 1);
    p.cx = 0.5f * float(p.w - 1);
    p.cy = 0.5f * float(p.h - 1);
    const int ch = pb.Channels();
    const float inv = (exposure > 0.0f) ? 1.0f / exposure : 1.0f;
    for (int y = 0; y < p.h; ++y)
        for (int x = 0; x < p.w; ++x) {
            const float* q = pb.At(x, y);
            const float l = (ch >= 3)
                ? 0.2126f * q[0] + 0.7152f * q[1] + 0.0722f * q[2]
                : q[0];
            const size_t i = size_t(y) * size_t(p.w) + size_t(x);
            p.v[i] = l * inv;

            // CLIPPED AND CRUSHED PIXELS ARE UNUSABLE, and this matters more
            // than it sounds.
            //
            // Dividing by exposure puts two frames on a common scale only where
            // both actually recorded the light. A clipped pixel did not: it
            // saturated, so its value is the sensor's ceiling rather than the
            // scene's brightness, and dividing by a LONG exposure then makes it
            // read far darker than the same point in a short one. The solver
            // sees that as a huge residual and chases it.
            //
            // Measured on hdrtest1, before this existed: the four darkest
            // frames solved to 0.04-0.72 px -- exactly what a tripod does --
            // while the two brightest, whose skies are blown, ran away to
            // +2.81 and +21.25 px. The failure tracked EXPOSURE, not time, which
            // is what identified it.
            //
            // Same reasoning as merge_hdr's weight function, which already
            // refuses to trust these pixels. The thresholds are deliberately
            // generous: a pixel near the ceiling is already non-linear.
            const float raw = l;
            if (raw >= 0.95f || raw <= 0.0f) p.ok[i] = 0;
        }
    return p;
}

// Half-resolution, box-filtered. Averaging rather than dropping pixels: a
// decimated plane aliases, and the solver would then chase the aliasing rather
// than the displacement.
Plane Downsample(const Plane& in) {
    Plane o;
    o.w = std::max(1, in.w / 2);
    o.h = std::max(1, in.h / 2);
    o.v.resize(size_t(o.w) * size_t(o.h));
    o.ok.resize(size_t(o.w) * size_t(o.h), 1);
    o.cx = 0.5f * float(o.w - 1);
    o.cy = 0.5f * float(o.h - 1);
    for (int y = 0; y < o.h; ++y)
        for (int x = 0; x < o.w; ++x) {
            const float s = in.At(x * 2, y * 2) + in.At(x * 2 + 1, y * 2) +
                            in.At(x * 2, y * 2 + 1) + in.At(x * 2 + 1, y * 2 + 1);
            const size_t i = size_t(y) * size_t(o.w) + size_t(x);
            o.v[i] = s * 0.25f;

            // A coarse pixel is valid only if ALL four of its sources were.
            // Averaging a clipped sample with three good ones produces a value
            // that looks usable and is not, and at level 3 one bad pixel would
            // have poisoned a 8x8 block of the original.
            o.ok[i] = (in.Ok(x * 2, y * 2) && in.Ok(x * 2 + 1, y * 2) &&
                       in.Ok(x * 2, y * 2 + 1) && in.Ok(x * 2 + 1, y * 2 + 1))
                          ? 1 : 0;
        }
    return o;
}

// One sampled point, with everything the solve needs precomputed.
//
// This is the inverse-compositional trick: gx, gy and the Jacobian row depend
// only on the REFERENCE, so they are computed once and reused across every
// iteration and every frame.
struct Point {
    float x = 0, y = 0;      // location in the reference
    float ref = 0;           // reference luminance there
    float j[6] = {0,0,0,0,0,0};   // d(residual)/d(param)
};

// Picks sample points, weighted by gradient magnitude.
//
// A threshold on a sorted sample of gradients rather than a full sort of the
// image: the goal is "the strongest N", and an approximate cut-off is enough.
// Points are spread by a stride so the set covers the frame -- rotation and
// shear are only observable from points far apart, so a cluster on one strong
// edge would leave those parameters unconstrained.
std::vector<Point> PickPoints(const Plane& ref, int want) {
    std::vector<Point> pts;
    if (ref.w < 8 || ref.h < 8) return pts;

    // Gradient magnitude on a grid, to find a threshold.
    const int stride = std::max(1, int(std::sqrt(double(ref.w) * double(ref.h) /
                                                 std::max(1.0, double(want) * 4.0))));
    std::vector<float> mags;
    mags.reserve(size_t((ref.w / stride + 1)) * size_t((ref.h / stride + 1)));
    for (int y = 1; y < ref.h - 1; y += stride)
        for (int x = 1; x < ref.w - 1; x += stride) {
            const float gx = 0.5f * (ref.At(x + 1, y) - ref.At(x - 1, y));
            const float gy = 0.5f * (ref.At(x, y + 1) - ref.At(x, y - 1));
            mags.push_back(std::abs(gx) + std::abs(gy));
        }
    if (mags.empty()) return pts;

    std::sort(mags.begin(), mags.end());
    // Keep roughly the top quarter of candidates, which after the stride gives
    // about `want` points.
    const float cut = mags[size_t(double(mags.size()) * 0.75)];

    for (int y = 1; y < ref.h - 1; y += stride)
        for (int x = 1; x < ref.w - 1; x += stride) {
            const float gx = 0.5f * (ref.At(x + 1, y) - ref.At(x - 1, y));
            const float gy = 0.5f * (ref.At(x, y + 1) - ref.At(x, y - 1));
            if (std::abs(gx) + std::abs(gy) < cut) continue;

            // A point whose neighbourhood contains a clipped sample is
            // unusable, and worse than merely unusable: the clipping
            // MANUFACTURES the gradient, so a blown edge looks like a strong
            // feature and the selection above actively prefers it.
            if (!ref.Ok(x, y) || !ref.Ok(x - 1, y) || !ref.Ok(x + 1, y) ||
                !ref.Ok(x, y - 1) || !ref.Ok(x, y + 1)) continue;

            Point p;
            p.x = float(x);
            p.y = float(y);
            p.ref = ref.At(x, y);

            // Jacobian of the warped position with respect to the six affine
            // parameters, times the image gradient. The warp is
            //   x' = (1+p0)x + p1 y + p2
            //   y' = p3 x + (1+p4) y + p5
            // so d(x')/dp = [x, y, 1, 0, 0, 0] and d(y')/dp = [0, 0, 0, x, y, 1].
            //
            // COORDINATES ARE MEASURED FROM THE IMAGE CENTRE, not the corner,
            // and the reason is numerical rather than aesthetic.
            //
            // With raw pixel coordinates the linear columns reach 5796 while
            // the translation columns are 1, so the normal equations carry a
            // 5796:1 conditioning ratio and the linear terms dominate the fit.
            // The solve then prefers to explain a displacement with a tiny
            // scale or rotation rather than a translation -- and since those
            // are anchored at the ORIGIN, a scale of 1.0003 walks the image
            // centre by 0.78 px right and 0.52 px down.
            //
            // That is exactly what Tim saw: hdrtest2's merge shifted down and
            // right by about a pixel even though the reference frame has the
            // identity transform. The seven solves all reported the same sign
            // of rotation and the same above-unity scale, which is the
            // signature of a bias rather than seven independent answers.
            //
            // Centring makes the columns comparable in magnitude and puts the
            // rotation and scale pivot where a camera actually rotates about --
            // the middle of the frame -- so neither displaces the centre.
            const float cx = float(x) - ref.cx;
            const float cy = float(y) - ref.cy;
            p.j[0] = gx * cx;
            p.j[1] = gx * cy;
            p.j[2] = gx;
            p.j[3] = gy * cx;
            p.j[4] = gy * cy;
            p.j[5] = gy;
            pts.push_back(p);
        }
    return pts;
}

// Solves a 6x6 symmetric system by Gaussian elimination with partial pivoting.
// Six unknowns does not justify pulling in a linear algebra library.
bool Solve6(double A[6][6], double b[6], double x[6]) {
    for (int i = 0; i < 6; ++i) {
        int piv = i;
        for (int r = i + 1; r < 6; ++r)
            if (std::abs(A[r][i]) > std::abs(A[piv][i])) piv = r;
        if (std::abs(A[piv][i]) < 1e-12) return false;   // singular
        if (piv != i) { std::swap(A[piv], A[i]); std::swap(b[piv], b[i]); }
        for (int r = i + 1; r < 6; ++r) {
            const double f = A[r][i] / A[i][i];
            for (int c = i; c < 6; ++c) A[r][c] -= f * A[i][c];
            b[r] -= f * b[i];
        }
    }
    for (int i = 5; i >= 0; --i) {
        double s = b[i];
        for (int c = i + 1; c < 6; ++c) s -= A[i][c] * x[c];
        x[i] = s / A[i][i];
    }
    return true;
}

class Align : public AlgorithmBase {
public:
    const char* Name()     const override { return "align"; }
    const char* Category() const override { return "merge"; }

    PortList Inputs() const override {
        return {{"src", DataType::ImageSet, FormatSpec::Any, ShapeSpec::Any}};
    }
    PortList Outputs() const override {
        // Same shape in, same shape out: this is not a reduction in the usual
        // sense, it annotates a group and hands it back.
        return {{"out", DataType::ImageSet, FormatSpec::SameAsInput, ShapeSpec::SameAsInput}};
    }

    void RunCPU(RunCtx&) override {}

    // Handled directly by the pipeline, like reshape: it needs the whole set at
    // once because every frame is solved against a common reference.
    bool IsAligner() const override { return true; }

    // Solves every frame against the first and attaches the results.
    //
    // The first frame is the reference and gets identity. Which frame is chosen
    // barely matters for a bracket -- they are all within a pixel or two -- and
    // taking the first keeps the choice predictable.
    bool RunAlign(std::vector<Image>* images, std::string* err) override {
        if (!images || images->size() < 2) { m_report.clear(); return true; }

        // The reference plane, at the exposure of frame 0.
        ImageView rv = (*images)[0].MapCpuRead();
        if (!rv.data) { *err = "align: the reference frame has no pixels"; return false; }
        PixelBuffer rb;
        rb.Unpack(rv);
        if (!rb.Valid()) { *err = "align: unsupported format"; return false; }
        const float refExp = (*images)[0].Desc().RelativeExposure();
        const Plane ref0 = MakeLuma(rb, refExp);

        // Pyramid, coarsest last.
        std::vector<Plane> refPyr{ref0};
        while (refPyr.back().w > 256 && refPyr.back().h > 256 &&
               int(refPyr.size()) < MaxLevels())
            refPyr.push_back(Downsample(refPyr.back()));

        // Sample points per level, from the reference. Computed once and reused
        // for every frame -- that is the inverse-compositional saving.
        std::vector<std::vector<Point>> ptsPyr;
        ptsPyr.reserve(refPyr.size());
        for (const Plane& p : refPyr) ptsPyr.push_back(PickPoints(p, m_samples));

        AttachTransform(&(*images)[0], Affine{});

        float worst = 0.0f;
        int   solved = 0;
        for (size_t i = 1; i < images->size(); ++i) {
            ImageView mv = (*images)[i].MapCpuRead();
            if (!mv.data) continue;
            PixelBuffer mb;
            mb.Unpack(mv);
            if (!mb.Valid()) continue;

            const float exp = (*images)[i].Desc().RelativeExposure();
            const Plane mov0 = MakeLuma(mb, exp);
            std::vector<Plane> movPyr{mov0};
            while (movPyr.size() < refPyr.size())
                movPyr.push_back(Downsample(movPyr.back()));

            // START FROM WHAT IS ALREADY THERE.
            //
            // Tim's requirement, and it is what makes this composable with a
            // feature-matching stage: a coarse transform from features is
            // refined here rather than thrown away. With nothing attached this
            // is identity, which is the same as starting from scratch.
            Affine t = TransformOf((*images)[i]);

            // Coarse to fine. A transform solved at half resolution has its
            // translation in half-resolution pixels, so it is scaled on the way
            // down; the linear part is scale-free and carries over unchanged.
            for (int lv = int(refPyr.size()) - 1; lv >= 0; --lv) {
                const float s = float(1 << lv);
                Affine lt = t;
                lt.m[2] /= s;
                lt.m[5] /= s;

                lt = SolveLevel(refPyr[size_t(lv)], movPyr[size_t(lv)],
                                ptsPyr[size_t(lv)], lt);

                t = lt;
                t.m[2] *= s;
                t.m[5] *= s;
            }

            AttachTransform(&(*images)[i], t);
            ++solved;
            worst = std::max(worst, std::hypot(t.Dx(), t.Dy()));

            // Per-frame, because a maximum hides the shape of the answer. A
            // tripod bracket should show a few frames of similar small shifts;
            // one wild value among sane ones means a failed solve rather than a
            // gust of wind, and only the individual numbers tell them apart.
            if (AlignVerbose()) {
                std::fprintf(stderr,
                             "[align] frame %d: dx %+.2f dy %+.2f  rot %+.3f deg  "
                             "scale %.5f\n",
                             int(i), double(t.Dx()), double(t.Dy()),
                             double(t.RotationDeg()),
                             double(std::hypot(t.m[0], t.m[3])));
            }
        }

        char buf[128];
        std::snprintf(buf, sizeof buf,
                      "aligned %d frames, largest shift %.2f px", solved, double(worst));
        m_report = buf;
        return true;
    }

    std::string RunReport() const override { return m_report; }

private:
    // Pyramid levels. ONE by default, which means no pyramid at all.
    //
    // The coarse levels were actively harmful for the job this algorithm does.
    // Measured on hdrtest2 with four levels, every frame came back with about
    // 1 degree of rotation, 1% scale and tens of pixels of translation -- a
    // consistent, systematic wrongness that no tripod could produce. With one
    // level the same bracket solves to 0.14-2.13 px and 0.05 degrees, which
    // matches the blur Tim measured by eye.
    //
    // The reason is scale. At level 3 the image is one eighth size, so a
    // one-pixel displacement appears as 0.125 px -- below what a solve can
    // resolve -- and whatever noise it fits instead is multiplied by 8 on the
    // way back up. The pyramid buys reach for LARGE displacement, and this
    // algorithm exists for the small case; buying reach it does not need costs
    // it the precision it does.
    //
    // Large motion belongs to the feature-matching aligner, which will attach
    // a coarse transform this one then refines -- the two-class split Tim
    // described. Raising this is how to experiment with that boundary before
    // that aligner exists.
    static int MaxLevels() {
        static const int n = [] {
            char buf[16] = {};
            size_t got = 0;
            if (getenv_s(&got, buf, sizeof buf, "TGLAB_ALIGNLEVELS") == 0 && got > 1)
                return std::max(1, atoi(buf));
            return 1;
        }();
        return n;
    }

    // One Lucas-Kanade solve at one pyramid level.
    // The transform is STORED in corner coordinates -- that is what a consumer
    // walking the image grid wants -- but SOLVED in centred ones, because the
    // conditioning and the pivot both demand it. The conversion happens here,
    // at the boundary, so neither the caller nor the merge has to know.
    //
    //   centred:  p_c = p - c
    //   corner:   T(p) = c + Tc(p - c)
    static Affine ToCentred(const Affine& t, float cx, float cy) {
        Affine o = t;
        // Translation in centred space is where the centre goes, relative to
        // itself: T(c) - c.
        float mx, my;
        t.MapPoint(cx, cy, &mx, &my);
        o.m[2] = mx - cx;
        o.m[5] = my - cy;
        return o;
    }

    static Affine ToCorner(const Affine& t, float cx, float cy) {
        Affine o = t;
        // Undo the centring: T(p) = c + Tc(p - c) expands to a linear part
        // unchanged and a translation of c + tc - Tc*c.
        o.m[2] = cx + t.m[2] - (t.m[0] * cx + t.m[1] * cy);
        o.m[5] = cy + t.m[5] - (t.m[3] * cx + t.m[4] * cy);
        return o;
    }

    Affine SolveLevel(const Plane& ref, const Plane& mov,
                      const std::vector<Point>& pts, Affine tCorner) const {
        if (pts.size() < 12) return tCorner;   // too few to constrain six params

        Affine t = ToCentred(tCorner, ref.cx, ref.cy);

        for (int iter = 0; iter < kIters; ++iter) {
            double A[6][6] = {};
            double b[6] = {};
            int used = 0;

            for (const Point& p : pts) {
                float wx, wy;
                // Points carry CORNER coordinates; the solve works in centred
                // space, so map the centred point and add the centre back to
                // get a sampling position.
                t.MapPoint(p.x - ref.cx, p.y - ref.cy, &wx, &wy);
                wx += mov.cx;
                wy += mov.cy;
                // Points that warp outside the moving frame carry no
                // information and would pull the fit toward the edge clamp.
                if (wx < 1.0f || wy < 1.0f ||
                    wx > float(mov.w - 2) || wy > float(mov.h - 2)) continue;

                // And the sample it lands on must be usable in the MOVING frame
                // too. A point can be perfectly exposed in the reference and
                // blown in a longer exposure, which is the common case in a
                // bracket -- the reference check alone catches only half of it.
                if (!mov.Ok(int(wx), int(wy)) || !mov.Ok(int(wx) + 1, int(wy) + 1))
                    continue;

                const float r = mov.Sample(wx, wy) - p.ref;
                for (int a = 0; a < 6; ++a) {
                    b[a] -= double(p.j[a]) * double(r);
                    for (int c = a; c < 6; ++c) A[a][c] += double(p.j[a]) * double(p.j[c]);
                }
                ++used;
            }
            if (used < 12) break;
            for (int a = 0; a < 6; ++a)
                for (int c = 0; c < a; ++c) A[a][c] = A[c][a];

            // Levenberg-style damping. Without it a frame with gradients along
            // one direction only -- a horizon, say -- gives a near-singular
            // system and the solve jumps somewhere absurd.
            for (int a = 0; a < 6; ++a) A[a][a] *= 1.0 + kDamping;

            double d[6];
            if (!Solve6(A, b, d)) break;

            // ADD the increment to the current parameters.
            //
            // Forward-additive, not inverse-compositional, and the two must not
            // be mixed. The Jacobian here is built from the REFERENCE gradient
            // -- which is what makes it precomputable -- but the residual is
            // measured in the MOVING frame, so this is Lucas-Kanade's forward
            // form with a constant-gradient approximation. Its update is a
            // straight addition.
            //
            // The first version composed the increment's INVERSE instead, which
            // is the inverse-compositional rule, and mixing the two made the
            // solve diverge: it reported -6.66 for a shift of 2. Cheap to get
            // wrong and hard to see, since both forms are plausible in isolation
            // -- which is why the test checks a KNOWN shift rather than merely
            // that a solve happened.
            //
            // The approximation costs a little convergence speed on a large
            // displacement and nothing on the sub-pixel case this exists for;
            // the pyramid handles the large ones.
            t.m[0] += float(d[0]);
            t.m[1] += float(d[1]);
            t.m[2] += float(d[2]);
            t.m[3] += float(d[3]);
            t.m[4] += float(d[4]);
            t.m[5] += float(d[5]);

            // Converged once the step is far below a pixel.
            if (std::abs(d[2]) < 1e-3 && std::abs(d[5]) < 1e-3) break;
        }
        return ToCorner(t, ref.cx, ref.cy);
    }

    static constexpr int    kIters   = 12;
    static constexpr double kDamping = 1e-3;

    // How many points to sample per level. Enough to over-determine six
    // parameters many times over, few enough that a solve is milliseconds
    // rather than seconds -- which is the whole reason for sampling.
    Param<int> m_samples{this, "samples", 20000, 500, 200000, {.step = 500}};

    std::string m_report;
};

} // namespace

REGISTER_ALGORITHM(Align);

} // namespace tglab
