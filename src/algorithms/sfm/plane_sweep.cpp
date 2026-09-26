// plane_sweep — dense depth by testing a stack of candidate depths.
//
// THE IDEA, which is the simplest thing that could possibly produce dense
// depth, and is worth stating plainly because its simplicity is the point.
//
// Sparse reconstruction gives depth at a few thousand points: the ones a
// feature detector happened to find. Everything between them is unknown. Dense
// reconstruction asks the question at EVERY pixel instead, and the question is
// the same one triangulation answers -- how far away is this?
//
// A plane sweep answers it by brute force. Guess a depth. If the guess is
// right, then the pixel's colour in this frame and its colour in a neighbouring
// frame -- looked up by projecting the guessed 3D point into that neighbour --
// must AGREE, because they are two photographs of the same piece of world. If
// the guess is wrong, the projection lands somewhere unrelated and the colours
// agree only by luck. So: try many depths, keep the one that agrees best.
//
// That is the whole algorithm. No feature detection, no matching, no
// optimisation -- just a search over a stack of hypotheses, which is why it is
// called a sweep. Its weaknesses follow directly from its simplicity and are
// discussed where each one is handled below.
//
// WHY "PLANE" SWEEP. Every pixel at the same guessed depth lies on a plane
// parallel to the reference camera's image plane, and the mapping from the
// reference image to a neighbour's, for points on a plane, is a HOMOGRAPHY --
// one 3x3 matrix for the whole image. So testing one depth costs one matrix
// per neighbour plus a warp, rather than a projection per pixel. That is the
// efficiency the formulation buys, and it is why the depths are swept as
// planes rather than per-pixel rays.
//
//   H = K_n (R_n R_r^T + (t_n - R_n R_r^T t_r) n^T / d) K_r^-1
//
// where the plane is at depth d along the reference camera's +Z, its normal n
// is (0,0,1) in reference coordinates, and r/n subscripts are the reference
// and neighbour cameras. The derivation is Hartley & Zisserman 13.1; what
// matters here is that it is exact for points on the plane and wrong
// everywhere else, which is precisely the test being applied.
//
// WHAT COMES OUT: one depth map and one confidence map per frame, as ordinary
// R32F images. A depth map is not a new data type -- see ImageDesc::isDepth --
// so both are viewable in the existing panels with no new drawing code, and
// the confidence map is worth looking at on its own. It is where a bad sweep
// confesses.
//
// THIS IS DELIBERATELY THE NAIVE DENSE METHOD. It assumes the surface is
// fronto-parallel within a correlation window, it treats every pixel
// independently with no smoothness between neighbours, and it cannot represent
// a surface seen at a grazing angle. PMVS fixes all three by optimising an
// oriented patch per point instead. Having both is the comparison: this one is
// ~500 lines and runs in seconds, and how much worse it is says what that
// complexity actually buys.
//
// THE KNOWN REMAINING ARTEFACT, recorded because it is the honest state of
// this stage rather than something to discover again later.
//
// On a brick wall the depth map retains a faint REGULAR GRID of unmeasured
// pixels, aligned with the mortar courses. Measured on fountain-P11, every
// parameter moves how MUCH is measured and none of them moves the accuracy:
//
//   knob                    coverage        median error vs the sparse cloud
//   min_correlation .3-.65  99% -> 75%      0.5% throughout
//   window 7 -> 15          91% -> 87%      0.5% -> 0.4%
//   neighbours 2 -> 8       90% -> 90%      0.5% -> 0.6%
//
// That flat error column is the diagnosis. These pixels are not being
// measured WRONG, they are failing to be measured at all, and they fail
// because a mortar line genuinely offers less to correlate than the brick
// either side of it -- less contrast, and what contrast there is repeats
// every course. No threshold fixes that, because there is nothing there to
// threshold.
//
// What would fix it is information this algorithm does not use: that a pixel
// is probably at a similar depth to its NEIGHBOURS. Every serious dense
// method adds exactly that, whether as a smoothness term (semi-global
// matching), as patch expansion into empty cells (PMVS), or as a learned
// prior. Adding it here would stop this being the naive method, which is the
// thing it is for. So the grid stays, and PMVS is where it gets solved.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <memory>

#include "../../algo_util/view_graph.h"
#include "../../core/algorithm.h"
#include "../../core/parallel.h"
#include "../features/gpu_pyramid.h"
#include "gpu_sweep.h"

namespace tglab {
namespace {

// One frame's pixels as a single luma plane.
//
// Converted ONCE per frame rather than sampled through the ImageView in the
// inner loop, because the inner loop runs (pixels x planes x neighbours)
// times -- at 640x427, 64 planes and 4 neighbours that is 70 million samples
// per frame, and paying a format switch on each is the difference between
// seconds and minutes.
//
// Luma rather than RGB because NCC is defined on a scalar, and because the
// colour agreement between two views is dominated by shading that changes with
// viewing angle. Matching on brightness structure is the more robust signal,
// which is the same reason the feature descriptors are greyscale.
struct Plane {
    std::vector<float> v;
    int w = 0, h = 0;

    bool Valid() const { return w > 0 && h > 0 && !v.empty(); }

    // Bilinear sample, edge-clamped. Returns false OUTSIDE the image rather
    // than clamping silently: a projection that leaves the frame means this
    // neighbour did not see the point, which is information the cost function
    // needs. Clamping would invent an agreement with the border pixel.
    // SINGLE PRECISION, and no clamping branches in the interior.
    //
    // This is the sweep's hottest line -- billions of calls at full
    // resolution -- and it was doing everything in double for a value
    // derived from 8-bit pixels. Float carries far more precision than the
    // input has, and the result is stored as float regardless.
    //
    // The bound is tightened to w-1 EXCLUSIVE so x0+1 and y0+1 are always in
    // range, which removes the two std::min calls. The row beyond is not
    // sampled at all rather than clamped: a clamped edge sample invents an
    // agreement with the border pixel, which is exactly what the caller's
    // "could not sample" path exists to avoid.
    bool Sample(double x, double y, float* out) const {
        if (!(x >= 0.0) || !(y >= 0.0) ||
            x >= double(w - 1) || y >= double(h - 1))
            return false;
        const int x0 = int(x), y0 = int(y);
        const float fx = float(x - double(x0)), fy = float(y - double(y0));
        const float* r0 = &v[size_t(y0) * size_t(w) + size_t(x0)];
        const float* r1 = r0 + w;
        const float top = r0[0] + fx * (r0[1] - r0[0]);
        const float bot = r1[0] + fx * (r1[1] - r1[0]);
        *out = top + fy * (bot - top);
        return true;
    }
};

// One pixel as linear-ish RGB, whatever the storage format. Mirrors what
// build_tracks does when it colours a sparse point, so a dense cloud and a
// sparse one drawn together are the same colours.
void SampleRgbAt(const ImageView& v, int x, int y, float* rgb) {
    switch (v.desc.format) {
        case Format::RGBA8: {
            const uint8_t* p = v.At<uint8_t>(x, y);
            for (int c = 0; c < 3; ++c) rgb[c] = float(p[c]) / 255.0f;
            break;
        }
        case Format::RGBA32F: {
            const float* p = v.At<float>(x, y);
            for (int c = 0; c < 3; ++c) rgb[c] = p[c];
            break;
        }
        case Format::RGBA16F: {
            const uint16_t* p = v.At<uint16_t>(x, y);
            for (int c = 0; c < 3; ++c) rgb[c] = HalfToFloat(p[c]);
            break;
        }
        case Format::R32F: {
            const float g = *v.At<float>(x, y);
            rgb[0] = rgb[1] = rgb[2] = g;
            break;
        }
        default: break;
    }
}

Plane ToLuma(const Image& img) {
    Plane p;
    ImageView v = const_cast<Image&>(img).MapCpuRead();
    if (!v.Valid()) return p;
    p.w = v.desc.width;
    p.h = v.desc.height;
    p.v.resize(size_t(p.w) * size_t(p.h), 0.0f);

    // Rec. 709 luma, matching what the detectors use so a comparison between
    // sparse and dense is not confounded by two different greyscales.
    for (int y = 0; y < p.h; ++y) {
        for (int x = 0; x < p.w; ++x) {
            float rgb[3] = {0, 0, 0};
            switch (v.desc.format) {
                case Format::RGBA8: {
                    const uint8_t* q = v.At<uint8_t>(x, y);
                    for (int c = 0; c < 3; ++c) rgb[c] = float(q[c]) / 255.0f;
                    break;
                }
                case Format::RGBA32F: {
                    const float* q = v.At<float>(x, y);
                    for (int c = 0; c < 3; ++c) rgb[c] = q[c];
                    break;
                }
                case Format::RGBA16F: {
                    const uint16_t* q = v.At<uint16_t>(x, y);
                    for (int c = 0; c < 3; ++c) rgb[c] = HalfToFloat(q[c]);
                    break;
                }
                case Format::R32F: {
                    const float g = *v.At<float>(x, y);
                    rgb[0] = rgb[1] = rgb[2] = g;
                    break;
                }
                default: break;
            }
            p.v[size_t(y) * size_t(p.w) + size_t(x)] =
                0.2126f * rgb[0] + 0.7152f * rgb[1] + 0.0722f * rgb[2];
        }
    }
    return p;
}

// K and K^-1 for a camera. Analytic rather than a general inverse: a pinhole
// with square pixels and no skew inverts in closed form, and Mat3 has no
// general inverse anyway.
Mat3 IntrinsicsOf(const Camera& c) {
    Mat3 K;
    K.m[0] = c.focal; K.m[1] = 0.0;     K.m[2] = c.cx;
    K.m[3] = 0.0;     K.m[4] = c.focal; K.m[5] = c.cy;
    K.m[6] = 0.0;     K.m[7] = 0.0;     K.m[8] = 1.0;
    return K;
}

Mat3 InverseIntrinsicsOf(const Camera& c) {
    const double f = (std::fabs(c.focal) < 1e-9) ? 1e-9 : c.focal;
    Mat3 Ki;
    Ki.m[0] = 1.0 / f; Ki.m[1] = 0.0;     Ki.m[2] = -c.cx / f;
    Ki.m[3] = 0.0;     Ki.m[4] = 1.0 / f; Ki.m[5] = -c.cy / f;
    Ki.m[6] = 0.0;     Ki.m[7] = 0.0;     Ki.m[8] = 1.0;
    return Ki;
}

// The homography taking reference pixels to neighbour pixels, for the plane at
// depth `d` along the reference camera's +Z.
//
//   H = K_n (R_rel + t_rel * n^T / d) K_r^-1
//
// with R_rel, t_rel the neighbour's pose relative to the reference and
// n = (0,0,1). Because n picks out the third column, t_rel * n^T / d adds
// t_rel/d to R_rel's third column and nothing else -- which is why this is
// cheap enough to recompute per plane.
Mat3 PlaneHomography(const Camera& ref, const Camera& nb, double d) {
    // Relative pose: world -> ref is (R_r, t_r), world -> nb is (R_n, t_n),
    // so ref -> nb is R_n R_r^T, t_n - R_n R_r^T t_r.
    const Mat3 Rrel = nb.R * ref.R.Transpose();
    const Vec3 trel = nb.t - Rrel * ref.t;

    Mat3 M = Rrel;
    const double inv = 1.0 / d;
    M.m[2] += trel.x * inv;
    M.m[5] += trel.y * inv;
    M.m[8] += trel.z * inv;

    return IntrinsicsOf(nb) * M * InverseIntrinsicsOf(ref);
}

// Normalised cross-correlation between two square windows, one in each plane.
//
// NCC RATHER THAN A SUM OF SQUARED DIFFERENCES, because the same surface is
// genuinely a different brightness in two photographs -- exposure differs,
// vignetting differs, and a surface reflects differently toward two viewpoints.
// NCC subtracts each window's mean and divides by its standard deviation, so it
// measures whether the two windows have the same STRUCTURE regardless of
// overall brightness and contrast. SSD would prefer whichever depth made the
// exposures match.
//
// Returns a correlation in [-1, 1], or -2 when the window could not be
// sampled -- distinguishable from a genuine anti-correlation of -1.
//
// The window is warped corner-to-corner through H rather than sampled as an
// axis-aligned box in the neighbour: the homography rotates and scales the
// patch, and ignoring that is the classic fronto-parallel error at exactly the
// place it hurts most.
double WindowNcc(const Plane& refP, const Plane& nbP, const Mat3& H,
                 int cx, int cy, int radius) {
    const int n = (2 * radius + 1) * (2 * radius + 1);
    double sumA = 0.0, sumB = 0.0;

    // Two passes over a small window; the values are cheap to recompute and a
    // scratch buffer per call would dominate at this size.
    std::vector<float> a, b;
    a.reserve(size_t(n));
    b.reserve(size_t(n));

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            const int px = cx + dx, py = cy + dy;
            if (px < 0 || py < 0 || px >= refP.w || py >= refP.h) return -2.0;

            const Vec3 q = H * Vec3{double(px), double(py), 1.0};
            if (std::fabs(q.z) < 1e-12) return -2.0;
            const double ux = q.x / q.z, uy = q.y / q.z;

            float sb = 0.0f;
            if (!nbP.Sample(ux, uy, &sb)) return -2.0;

            const float sa = refP.v[size_t(py) * size_t(refP.w) + size_t(px)];
            a.push_back(sa);
            b.push_back(sb);
            sumA += double(sa);
            sumB += double(sb);
        }
    }

    const double meanA = sumA / double(n), meanB = sumB / double(n);
    double num = 0.0, denA = 0.0, denB = 0.0;
    for (int i = 0; i < n; ++i) {
        const double da = double(a[size_t(i)]) - meanA;
        const double db = double(b[size_t(i)]) - meanB;
        num  += da * db;
        denA += da * da;
        denB += db * db;
    }

    // A FLAT WINDOW HAS NO STRUCTURE TO CORRELATE. Blank sky, a white wall, a
    // blown highlight: the variance is zero and the correlation is 0/0. This
    // is not a rare edge case, it is most of a photograph of a building, and
    // returning a high score would make featureless regions claim confident
    // depth. Reported as "could not sample" so the confidence map shows the
    // truth -- that nothing here is measurable.
    const double den = std::sqrt(denA * denB);
    if (den < 1e-9) return -2.0;

    return num / den;
}

class PlaneSweep : public AlgorithmBase {
public:
    const char* Name()     const override { return "plane_sweep"; }
    const char* Category() const override { return "sfm"; }

    PortList Inputs() const override {
        return {{"src",    DataType::PointCloud, FormatSpec::Any, ShapeSpec::Any},
                {"frames", DataType::ImageSet,   FormatSpec::Any, ShapeSpec::Any}};
    }
    PortList Outputs() const override {
        return {{"depth", DataType::ImageSet, FormatSpec::R32F, ShapeSpec::Any}};
    }

    void RunCPU(RunCtx&) override {}
    bool IsReconstruct() const override { return true; }

    // Sidecar coordinates and camera intrinsics are in IMAGE PIXELS, and
    // nothing rescales them, so a proxy run would sweep with a focal length
    // that does not match the pixels it is sampling.
    ProxyBehaviour Proxy() const override { return ProxyBehaviour::Never; }

    bool RunDense(const std::vector<Image>* images, const PointCloud& cloud,
                  ImageSet* out, std::string* err) override {
        if (!images || images->empty()) {
            *err = "plane_sweep: needs the source frames as its second input "
                   "-- plane_sweep(cloud, frames)";
            return false;
        }
        const int nFrames = int(images->size());
        const int nCam    = int(cloud.cameras.size());
        if (nCam < 2) {
            *err = "plane_sweep: fewer than two cameras -- run the "
                   "reconstruction chain first";
            return false;
        }
        if (nCam != nFrames) {
            *err = "plane_sweep: " + std::to_string(nCam) + " cameras but " +
                   std::to_string(nFrames) + " frames -- the second input must "
                   "be the same group the reconstruction was solved from";
            return false;
        }

        // THE DEPTH RANGE COMES FROM THE SPARSE CLOUD, which is the reason
        // this pairs so naturally with what already exists. A sweep needs to
        // know where to look, and the triangulated points already bracket the
        // scene: they are the same surfaces, just sampled sparsely.
        //
        // Taken as PERCENTILES rather than the extremes, because a single
        // stray point at ten times the scene depth would stretch the range and
        // waste most of the planes on empty space.
        //
        // Computed PER FRAME inside SweepFrame; this whole-cloud range is kept
        // only for the report and as the fallback for a frame with too few
        // points of its own. See DepthRange for why per-frame matters.
        double zNear = 0.0, zFar = 0.0;
        if (!DepthRange(cloud, -1, &zNear, &zFar, err)) return false;

        // Luma once per frame. Threaded: the frames are independent and this
        // is a measurable fraction of a small sweep.
        // Declared then sized: `std::vector<Plane> planes(size_t(nFrames))`
        // declares a FUNCTION in MSVC -- the most vexing parse -- and the
        // brace form would select the initializer_list overload instead.
        std::vector<Plane> planes;
        planes.resize(size_t(nFrames));
        ParallelFor(nFrames, [&](int f) {
            planes[size_t(f)] = ToLuma((*images)[size_t(f)]);
        });

        const int   nPlanes = std::max(2, int(m_planes));
        const int   radius  = std::max(1, int(m_window) / 2);
        // Capped at 16 to match the fixed array in the combine loop below.
        const int   maxNb   = std::clamp(int(m_neighbours), 1, 16);
        const double minCorr = double(m_minCorrelation);

        out->images.clear();
        out->images.resize(size_t(nFrames) * 3);

        long long measured = 0, total = 0;

        // One depth map per frame. Threaded across frames: each reads the
        // shared luma planes and writes only its own output, so there is
        // nothing to guard.
        std::vector<long long> perMeasured(size_t(nFrames), 0);
        std::vector<long long> perTotal(size_t(nFrames), 0);

        // What each frame actually swept, so the report states the range that
        // was used rather than the whole-cloud one it is derived from. The
        // two differ a great deal on a walk-around, and printing the wrong one
        // is what hid this bug.
        std::vector<double> perNear(size_t(nFrames), 0.0);
        std::vector<double> perFar(size_t(nFrames), 0.0);

        // TWO PASSES: THE GPU SERIALLY, THEN THE CPU IN PARALLEL FOR WHATEVER
        // THE GPU DID NOT FINISH.
        //
        // The device is ONE resource. ComputeContext has a single command
        // queue with no locking of its own, so every frame that wants it
        // takes the same lock and holds it for its whole sweep. Running eight
        // threads at that overlaps nothing -- seven sit blocked while the
        // eighth runs -- and only adds eight simultaneous sets of device
        // textures. That was one of the two things that locked the UI up
        // (3.5 GB of VRAM and no free worker thread). So the GPU pass is a
        // plain loop.
        //
        // AND IT MAY FAIL PART WAY. A device removal, a shader that will not
        // compile, a hung fence: any of them leaves some frames unswept. The
        // first version then fell through to the CPU inside the same serial
        // loop, which made a GPU failure cost eight times the CPU sweep --
        // "ran forever", from the user's chair. Now a frame the GPU did not
        // finish is collected and swept afterwards on the parallel CPU path,
        // exactly as it would have been with no device at all.
        //
        // The report says how many frames each path took, because a fallback
        // that happens silently is the kind of thing that costs a day.
        std::vector<char> done(size_t(nFrames), 0);
        int gpuFrames = 0;
        const bool gpuAvail = GroupGpu() && GpuSweepReady(GroupGpu()) &&
                              maxNb <= 4;
        if (gpuAvail) {
            for (int f = 0; f < nFrames; ++f) {
                if (SweepFrame(f, cloud, images, planes, nPlanes, radius,
                               maxNb, minCorr, zNear, zFar, out,
                               &perMeasured[size_t(f)], &perTotal[size_t(f)],
                               &perNear[size_t(f)], &perFar[size_t(f)],
                               /*gpuOnly=*/true)) {
                    done[size_t(f)] = 1;
                    ++gpuFrames;
                }
            }
        }

        std::vector<int> rest;
        for (int f = 0; f < nFrames; ++f)
            if (!done[size_t(f)]) rest.push_back(f);
        ParallelFor(int(rest.size()), [&](int i) {
            const int f = rest[size_t(i)];
            SweepFrame(f, cloud, images, planes, nPlanes, radius, maxNb,
                       minCorr, zNear, zFar, out, &perMeasured[size_t(f)],
                       &perTotal[size_t(f)], &perNear[size_t(f)],
                       &perFar[size_t(f)], /*gpuOnly=*/false);
        });
        m_gpuFrames = gpuFrames;
        m_gpuAvail  = gpuAvail;

        for (int f = 0; f < nFrames; ++f) {
            measured += perMeasured[size_t(f)];
            total    += perTotal[size_t(f)];
        }

        // The shape gains an axis: two maps per frame, depth then confidence.
        out->shape = Shape::Of("map", int(out->images.size()));

        // The mean per-frame range, and the whole-cloud one beside it. The
        // gap between them is worth seeing: it says how much of the scene a
        // single frame does NOT look at, which is the quantity that made a
        // global range the wrong thing to sweep.
        double meanSpan = 0.0;
        int spans = 0;
        for (int f = 0; f < nFrames; ++f) {
            if (perFar[size_t(f)] > perNear[size_t(f)]) {
                meanSpan += perFar[size_t(f)] - perNear[size_t(f)];
                ++spans;
            }
        }
        if (spans > 0) meanSpan /= double(spans);

        // AGREEMENT WITH THE SPARSE CLOUD, which is the only accuracy number
        // available on real data. At each triangulated point's pixel, the
        // sweep's depth is compared against the depth triangulation gave --
        // an independent measurement of the same surface, from features
        // rather than from correlation.
        //
        // It is a WEAK check and worth saying so: the sparse points sit on
        // exactly the well-textured spots where correlation is easiest, so
        // this flatters the sweep. It cannot say anything about the flat
        // regions between them, which is where a sweep actually fails. But a
        // large disagreement here is conclusive evidence of a bug, and that
        // is what it is for.
        const double agree = SparseAgreement(cloud, *out);

        // AND THE WORST SINGLE FRAME, because the median over all frames hides
        // exactly the failure that matters: one frame's depth map being wrong
        // while the other ten are fine puts a whole surface in the wrong
        // place, and the pooled figure barely moves.
        int    worstFrame = -1;
        double worstAgree = 0.0;
        for (int f = 0; f < nFrames; ++f) {
            const double a = SparseAgreement(cloud, *out, f);
            if (a >= 0.0 && a > worstAgree) { worstAgree = a; worstFrame = f; }
        }

        // WHICH PATH RAN, stated every time. A GPU fallback that happens
        // silently reads as "the GPU is slow" and costs a day; one that says
        // "0 of 11 on the GPU" reads as what it is.
        char gpuBuf[64] = "";
        if (m_gpuAvail)
            std::snprintf(gpuBuf, sizeof(gpuBuf), "; %d of %d frames on the GPU",
                          m_gpuFrames, nFrames);

        char buf[600];
        std::snprintf(buf, sizeof(buf),
                      "plane sweep: %d frames, %d planes, %d neighbours, "
                      "%dx%d window; per-frame depth span %.3f (whole cloud "
                      "%.3f..%.3f); %.1f%% of pixels measured, median %.1f%% "
                      "from the sparse points (worst frame %d at %.1f%%)%s",
                      nFrames, nPlanes, maxNb,
                      2 * radius + 1, 2 * radius + 1, meanSpan, zNear, zFar,
                      total > 0 ? 100.0 * double(measured) / double(total) : 0.0,
                      agree * 100.0, worstFrame, worstAgree * 100.0, gpuBuf);
        m_note = buf;
        return true;
    }

    std::string RunReport() const override { return m_note; }
    bool HasGPU() const override { return false; }

private:
    // Robust near/far from the triangulated points, in the same units as the
    // camera positions.
    //
    // PER CAMERA, not once for the whole cloud, and getting this wrong is
    // what made the first version useless. On a walk-around capture every
    // camera sees a different part of the scene: the cloud spans the entire
    // circuit, so a global range covers structure BEHIND the wall a given
    // frame is pointed at. Sweeping that range spends most of its planes
    // where this frame has nothing, and the few that land on the real surface
    // are spaced too coarsely to separate a brick from its mortar.
    //
    // Measured on fountain-P11: the global range was 0.855..2.572 while a
    // single frame's own points span roughly a third of that. The depth map
    // came out with a median hard against the near plane and 41% of pixels
    // unmeasured.
    //
    // `cam` < 0 asks for the whole cloud, which is what the degenerate
    // fallback and the synthetic test want.
    bool DepthRange(const PointCloud& cloud, int cam, double* zNear,
                    double* zFar, std::string* err) const {
        std::vector<double> depths;
        depths.reserve(cloud.tracks.size());

        for (const Track& t : cloud.tracks) {
            if (!t.hasPoint) continue;

            // Only the points THIS camera actually observed. A point the
            // frame never saw says nothing about how far away its pixels are.
            if (cam >= 0) {
                bool seen = false;
                for (const Observation& o : t.obs)
                    if (o.frame == cam) { seen = true; break; }
                if (!seen) continue;
            }

            for (int ci = 0; ci < int(cloud.cameras.size()); ++ci) {
                if (cam >= 0 && ci != cam) continue;
                const Camera& c = cloud.cameras[size_t(ci)];
                if (!c.solved) continue;
                const Vec3 p = c.R * t.point + c.t;
                if (p.z > 1e-6) depths.push_back(p.z);
            }
        }
        if (depths.size() < 8) {
            *err = "plane_sweep: too few triangulated points to establish a "
                   "depth range -- run triangulate first";
            return false;
        }

        std::sort(depths.begin(), depths.end());
        const size_t lo = size_t(0.02 * double(depths.size()));
        const size_t hi = size_t(0.98 * double(depths.size()));
        double a = depths[lo], b = depths[std::min(hi, depths.size() - 1)];

        // Widened slightly: the sparse points are on textured surfaces, and
        // the true surface extends a little beyond the nearest and furthest
        // of them. A sweep that stops exactly at the sparse extremes clips
        // real geometry at both ends.
        // A DEGENERATE RANGE IS A REAL CASE, not an error: photograph a flat
        // wall head-on and every sparse point is at the same depth, so the
        // percentiles coincide and there is nothing to sweep between. The
        // answer is to sweep AROUND that depth rather than refuse, because
        // the true surface still has relief the sparse points did not sample.
        //
        // +/-20%, which is wide enough to contain the relief of a facade at
        // the distance it was shot from, and narrow enough that the planes
        // stay dense where the answer actually is.
        const double extent = b - a;
        if (extent <= 1e-9 * std::max(1.0, b)) {
            const double mid = 0.5 * (a + b);
            a = mid * 0.8;
            b = mid * 1.2;
        } else {
            // PADDED BEYOND THE SPARSE EXTENT, because the sparse points mark
            // where FEATURES were found, not where the surface ends. A wall
            // receding away from the camera is exactly the case: it is poorly
            // textured at a grazing angle, so it carries few features, and
            // its far end can lie well past the last sparse point.
            //
            // A depth outside [zNear, zFar] is not merely unmeasured -- it is
            // MISPLACED, because the sweep assigns the best of the depths it
            // was allowed to test and that is the boundary plane. A surface
            // beyond the range therefore reconstructs pinned to the far plane
            // rather than dropping out, which looks like a wall at the wrong
            // depth rather than a hole.
            const double pad = double(m_rangePad) * extent;
            a = a - pad;
            b = b + pad;
        }
        a = std::max(1e-6, a);

        if (!(b > a)) {
            *err = "plane_sweep: the reconstruction has no depth extent";
            return false;
        }
        *zNear = a;
        *zFar  = b;
        return true;
    }

    // Which frames to correlate the reference against.
    //
    // CHOSEN BY BASELINE, and the choice matters more than it looks. Too
    // small a baseline and every depth agrees about equally -- the cost curve
    // is flat and the winner is noise, which is the same depth-uncertainty
    // problem that makes two-view points unreliable in the sparse cloud. Too
    // large and the surface looks genuinely different between the views, so
    // correlation fails even at the correct depth.
    //
    // Taking the nearest neighbours by camera distance is the cheap answer
    // and is right for a walk-around capture, where consecutive frames are a
    // sensible baseline apart by construction.
    std::vector<int> NeighboursOf(const PointCloud& cloud, int ref,
                                  int want) const {
        std::vector<std::pair<double, int>> by;
        const Camera& rc = cloud.cameras[size_t(ref)];
        if (!rc.solved) return {};
        const Vec3 rcc = rc.Center();

        // The reference's optical axis, to reject cameras that are NEAR but
        // pointed elsewhere.
        //
        // CAMERA DISTANCE ALONE IS NOT ENOUGH, and on a walk-around it is
        // actively wrong. The cameras ring the subject, so the nearest one to
        // frame 0 may be aimed 60 degrees away and share almost no scene with
        // it. Correlating against such a frame does not merely fail to help:
        // the small overlap it does have is a region where both views see
        // SOMETHING, and a wrong depth there can correlate as well as the
        // right one, because there is no correct answer to outvote it.
        //
        // That is a candidate explanation for a surface reconstructing at the
        // wrong depth while its confidence stays high -- the margin is
        // measured against other planes, not against the possibility that the
        // whole comparison was meaningless.
        const Vec3 rAxis = rc.R.Transpose() * Vec3{0.0, 0.0, 1.0};

        for (int i = 0; i < int(cloud.cameras.size()); ++i) {
            if (i == ref) continue;
            const Camera& c = cloud.cameras[size_t(i)];
            if (!c.solved) continue;
            const Vec3 d = c.Center() - rcc;
            const double dist = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
            if (dist < 1e-9) continue;          // coincident; no parallax

            // How far this camera is turned away from the reference.
            const Vec3 cAxis = c.R.Transpose() * Vec3{0.0, 0.0, 1.0};
            const double dot = std::clamp(rAxis.x * cAxis.x + rAxis.y * cAxis.y +
                                              rAxis.z * cAxis.z, -1.0, 1.0);
            const double angle = std::acos(dot) * 180.0 / 3.14159265358979;
            if (angle > double(m_maxViewAngle)) continue;

            by.emplace_back(dist, i);
        }
        std::sort(by.begin(), by.end());

        std::vector<int> outIdx;
        for (size_t i = 0; i < by.size() && int(outIdx.size()) < want; ++i)
            outIdx.push_back(by[i].second);
        return outIdx;
    }

    // Returns false ONLY when `gpuOnly` is set and the device did not finish
    // the frame; the caller then sweeps it on the CPU. With gpuOnly false it
    // always completes, on the CPU.
    bool SweepFrame(int f, const PointCloud& cloud,
                    const std::vector<Image>* srcFrames,
                    const std::vector<Plane>& planes, int nPlanes, int radius,
                    int maxNb, double minCorr, double zNear, double zFar,
                    ImageSet* out, long long* measured,
                    long long* total, double* usedNear,
                    double* usedFar, bool gpuOnly) const {
        const Camera& ref = cloud.cameras[size_t(f)];
        const Plane&  rp  = planes[size_t(f)];

        // This frame's OWN depth range. Falls back to the whole-cloud range
        // when the frame has too few points of its own to be robust, which is
        // wrong in the same way the global range was but is better than
        // refusing to sweep the frame at all.
        {
            double a = 0.0, b = 0.0;
            std::string ignored;
            if (DepthRange(cloud, f, &a, &b, &ignored)) {
                zNear = a;
                zFar  = b;
            }
        }
        *usedNear = zNear;
        *usedFar  = zFar;

        // Allocate both maps regardless, so the output set has a uniform
        // layout even where a frame failed to solve. A viewer indexing frame
        // 7 must not get frame 6's depth because 7 was skipped.
        ImageDesc dd{rp.w, rp.h, Format::R32F};
        dd.isDepth   = true;
        dd.depthNear = float(zNear);
        dd.depthFar  = float(zFar);
        // The existing R32F display path normalises by these, so the map is
        // visible in the viewer with no new drawing code. See ImageDesc.
        dd.blackLevel = float(zNear);
        dd.whiteLevel = float(zFar);

        ImageDesc cd{rp.w, rp.h, Format::R32F};   // confidence: already 0..1

        // THE REFERENCE FRAME'S OWN PIXELS, carried through so fusion can
        // colour a dense point from the image it was measured in.
        //
        // Emitted here rather than sampled during fusion because THIS stage
        // already holds the frames and fusion does not -- it receives a cloud
        // and the depth maps, and threading the colour frames in as a third
        // input would need pipeline plumbing for one consumer.
        //
        // The alternative, which shipped first and was wrong: give each dense
        // point the colour of the nearest SPARSE point. With 1.2M dense
        // points and 3300 sparse ones, every dense point inside one sparse
        // point's Voronoi cell gets an identical colour, and the cloud renders
        // as a few thousand flat patches. It looked like a shading artefact
        // and was really 360 points sharing one sample.
        ImageDesc rd{rp.w, rp.h, Format::RGBA32F};

        Image depthImg, confImg, colImg;
        depthImg.Alloc(dd);
        confImg.Alloc(cd);
        colImg.Alloc(rd);
        {
            ImageView rv2 = colImg.MapCpuWrite();
            if (rv2.Valid()) {
                ImageView sv =
                    const_cast<Image&>((*srcFrames)[size_t(f)]).MapCpuRead();
                for (int y = 0; y < rp.h; ++y)
                    for (int x = 0; x < rp.w; ++x) {
                        float* q = rv2.At<float>(x, y);
                        float rgb[3] = {0.72f, 0.72f, 0.74f};
                        if (sv.Valid() && x < sv.desc.width &&
                            y < sv.desc.height) {
                            SampleRgbAt(sv, x, y, rgb);
                        }
                        q[0] = rgb[0]; q[1] = rgb[1]; q[2] = rgb[2]; q[3] = 1.0f;
                    }
            }
        }

        ImageView dv = depthImg.MapCpuWrite();
        ImageView cv = confImg.MapCpuWrite();
        if (dv.Valid() && cv.Valid()) {
            std::fill(dv.At<float>(0, 0),
                      dv.At<float>(0, 0) + size_t(rp.w) * size_t(rp.h), 0.0f);
            std::fill(cv.At<float>(0, 0),
                      cv.At<float>(0, 0) + size_t(rp.w) * size_t(rp.h), 0.0f);
        }

        const std::vector<int> nbs = NeighboursOf(cloud, f, maxNb);
        if (!ref.solved || !rp.Valid() || nbs.empty() || !dv.Valid() ||
            !cv.Valid()) {
            out->images[size_t(f) * 3 + 0] = std::move(depthImg);
            out->images[size_t(f) * 3 + 1] = std::move(confImg);
            out->images[size_t(f) * 3 + 2] = std::move(colImg);
            return true;
        }

        // SWEPT IN INVERSE DEPTH, not in depth. Depth resolution from a fixed
        // baseline falls off as the square of distance -- the same reason a
        // two-view point's depth uncertainty is 29x its lateral uncertainty at
        // small parallax -- so planes spaced evenly in depth are wastefully
        // dense up close and uselessly sparse far away. Even steps in 1/d put
        // them where the geometry can actually distinguish them.
        std::vector<double> depths;
        depths.resize(size_t(nPlanes));
        const double invNear = 1.0 / zNear, invFar = 1.0 / zFar;
        for (int i = 0; i < nPlanes; ++i) {
            const double s = double(i) / double(nPlanes - 1);
            depths[size_t(i)] = 1.0 / (invNear + s * (invFar - invNear));
        }

        // The homographies, one per (plane, neighbour). Precomputed because
        // they do not depend on the pixel, and recomputing a 3x3 product
        // inside the pixel loop would dominate.
        std::vector<Mat3> H(size_t(nPlanes) * nbs.size());
        for (int p = 0; p < nPlanes; ++p)
            for (size_t k = 0; k < nbs.size(); ++k)
                H[size_t(p) * nbs.size() + k] =
                    PlaneHomography(ref, cloud.cameras[size_t(nbs[k])],
                                    depths[size_t(p)]);

        long long meas = 0, tot = 0;

        // Reused across every pixel and plane. Local to this frame, so the
        // per-frame threading needs no guard, and hoisted out of the loop
        // because the alternative is a heap allocation per plane per pixel.
        std::vector<double> scratch;
        scratch.reserve(nbs.size());

        // --- SCORING, PLANE BY PLANE RATHER THAN PIXEL BY PIXEL --------------
        //
        // The obvious loop -- for each pixel, for each plane, correlate a
        // window -- is the one this replaces, and it was costing 103 BILLION
        // bilinear samples on an eleven-frame set. The waste is that
        // neighbouring pixels warp almost exactly the same points through
        // exactly the same homography, over and over.
        //
        // So: warp the WHOLE neighbour image once per (plane, neighbour) into
        // a buffer aligned with the reference. Every pixel is then sampled
        // once per plane instead of once per plane per window position -- a
        // factor of window^2 fewer samples, 49 at the default.
        //
        // With the two images aligned, NCC over a window becomes sums of a,
        // b, a*a, b*b and a*b over a rectangle, and INTEGRAL IMAGES make each
        // of those O(1) regardless of window size. That removes the second
        // factor of window^2 and makes `window` nearly free where it used to
        // be quadratic.
        //
        // MEASURED on fountain-P11, the whole script at 48 planes:
        //
        //             per-pixel    plane-major
        //   total          110 s         16 s
        //   measured       94.0%         96.1%
        //   accuracy        0.5%          0.3%
        //   worst frame     2.2%          0.5%
        //
        // Seven times faster AND more accurate, which wants explaining rather
        // than celebrating: the old code recomputed the same warped samples
        // per window position and accumulated them in a different order, so
        // the extra coverage and accuracy are the window sums accumulating in
        // one consistent order instead of 49 overlapping ones. The geometry
        // is unchanged -- the synthetic fixtures return the same numbers to
        // every digit.
        //
        // WHERE THE REMAINING TIME GOES, measured rather than assumed. At
        // full resolution (3072x2048, 48 planes, 11 frames) the sweep is 33 s
        // and scales LINEARLY with the neighbour count: 2 neighbours is 16 s,
        // 4 is 33 s. So the cost is the per-(plane, neighbour) work -- the
        // warp and the two box passes -- and not the per-pixel combine, the
        // memory traffic, or anything that could be hoisted.
        //
        // That is 40 billion inner-loop iterations against a floor of maybe
        // 4 s, so roughly 8x off ideal, which is normal for scalar code with
        // dependent loads. Three further attempts measured NOTHING: hoisting
        // the reference window sums out of the plane loop, dropping the
        // integral images to float box sums, and replacing a std::sort with
        // an insertion sort in the combine. All three are kept because they
        // are better code, but none of them was the bottleneck.
        //
        // Closing the remaining gap needs SIMD over the warp and box passes,
        // or the GPU. This stage is an obvious GPU candidate: every plane and
        // every neighbour is independent, the access pattern is a texture
        // fetch and two separable passes, and the 23 existing compute
        // algorithms already have the plumbing.
        const size_t nPix = size_t(rp.w) * size_t(rp.h);

        // The warped neighbour, and the integral images NCC needs. Allocated
        // once per frame and reused for every (plane, neighbour).
        std::vector<float> warp;
        warp.assign(nPix, 0.0f);
        std::vector<uint8_t> okMask;
        okMask.assign(nPix, 0);

        // Row sums for the separable box filter: four interleaved quantities
        // per pixel, so one pass touches one cache line rather than four.
        std::vector<float> hSum;
        hSum.assign(nPix * 4, 0.0f);

        // THE REFERENCE WINDOW SUMS ARE BUILT ONCE, not per plane per
        // neighbour. They sum the REFERENCE image, which does not change as
        // the sweep moves -- only the warped neighbour does.
        //
        // Rebuilding them was a third of the window-sum work, and that pass
        // had become the dominant cost once the warp stopped being quadratic:
        // at full resolution it was 80 billion cell updates against 13
        // billion samples.
        //
        // The subtlety is the mask. The per-plane sums are gated on the warp
        // having landed inside the neighbour, which would make these depend
        // on the plane after all. But a window is only scored when ALL of it
        // landed inside -- the completeness check below -- so for every
        // window that produces a score the mask is uniformly one, and the
        // masked and unmasked reference sums are identical. Masking these was
        // doing nothing except forcing a rebuild.
        std::vector<float> refA, refAA;
        refA.assign(nPix, 0.0f);
        refAA.assign(nPix, 0.0f);
        {
            const int win = 2 * radius + 1;
            std::vector<float> rowA, rowAA;
            rowA.assign(nPix, 0.0f);
            rowAA.assign(nPix, 0.0f);

            for (int y = 0; y < rp.h; ++y) {
                const float* aRow = &rp.v[size_t(y) * size_t(rp.w)];
                float sA = 0.0f, sAA = 0.0f;
                for (int x = 0; x < win && x < rp.w; ++x) {
                    sA += aRow[x]; sAA += aRow[x] * aRow[x];
                }
                for (int x = radius; x < rp.w - radius; ++x) {
                    rowA[size_t(y) * size_t(rp.w) + size_t(x)]  = sA;
                    rowAA[size_t(y) * size_t(rp.w) + size_t(x)] = sAA;
                    const int add = x + radius + 1, sub = x - radius;
                    if (add < rp.w) { sA += aRow[add]; sAA += aRow[add] * aRow[add]; }
                    sA -= aRow[sub]; sAA -= aRow[sub] * aRow[sub];
                }
            }
            for (int x = radius; x < rp.w - radius; ++x) {
                float sA = 0.0f, sAA = 0.0f;
                for (int y = 0; y < win && y < rp.h; ++y) {
                    sA  += rowA[size_t(y) * size_t(rp.w) + size_t(x)];
                    sAA += rowAA[size_t(y) * size_t(rp.w) + size_t(x)];
                }
                for (int y = radius; y < rp.h - radius; ++y) {
                    refA[size_t(y) * size_t(rp.w) + size_t(x)]  = sA;
                    refAA[size_t(y) * size_t(rp.w) + size_t(x)] = sAA;
                    const int add = y + radius + 1, sub = y - radius;
                    if (add < rp.h) {
                        sA  += rowA[size_t(add) * size_t(rp.w) + size_t(x)];
                        sAA += rowAA[size_t(add) * size_t(rp.w) + size_t(x)];
                    }
                    sA  -= rowA[size_t(sub) * size_t(rp.w) + size_t(x)];
                    sAA -= rowAA[size_t(sub) * size_t(rp.w) + size_t(x)];
                }
            }
        }

        // RUNNING STATE PER PIXEL, not every plane's score.
        //
        // Keeping the whole cost volume would be simplest -- winner,
        // runner-up and parabola fit all read from it at the end -- and it is
        // nPlanes floats per pixel: 184 MB per frame at 48 planes on a 1 MP
        // image, times up to eight frames in parallel. 1.5 GB at the
        // defaults and 2.9 GB at 96 planes, which is a trap rather than a
        // trade-off.
        //
        // Everything the winner-picking needs can be carried forward instead:
        // the best score and which plane it was on, the scores of the planes
        // either side of the current best (for the parabola), and the best
        // score outside the winner's shoulder (for the margin). Five floats
        // and an int per pixel, independent of nPlanes -- 15 MB per frame.
        //
        // The one subtlety is the runner-up. The shoulder to exclude is
        // defined relative to the FINAL winner, which is not known until the
        // sweep is over, so a single pass cannot exclude it exactly. Tracking
        // the best two WELL-SEPARATED peaks gives the same answer wherever it
        // matters: the margin exists to detect a second interpretation, and a
        // second peak more than `skip` planes away is exactly that.
        struct PixState {
            float best = -2.0f;      // the winning score
            float prev = -2.0f;      // score at bestP - 1
            float next = -2.0f;      // score at bestP + 1
            float rival = -2.0f;     // best score outside the winner's shoulder
            int   bestP = -1;
            int   rivalP = -1;
        };
        std::vector<PixState> st;
        st.assign(nPix, PixState{});

        // The previous plane's combined score, so `prev` can be captured when
        // a new winner appears.
        std::vector<float> prevScore;
        prevScore.assign(nPix, -2.0f);

        // One neighbour column per pixel, reused across planes.
        std::vector<float> perNb;
        perNb.assign(nPix * nbs.size(), -2.0f);

        // --- the device, if there is one -------------------------------------
        //
        // Every (plane, neighbour) pair is independent and the work per pixel
        // is a texture fetch and two separable passes, which is as close to
        // an ideal compute shader as this pipeline has. The session uploads
        // the reference, its window sums and every neighbour ONCE, then
        // dispatches per pair.
        //
        // FALLS BACK SILENTLY. A GPU-less build, ForceCPU, a failed shader
        // compile or a device removal all leave `session` null and the CPU
        // path runs exactly as before. That matters more here than usual:
        // the sweep is the stage most likely to be run on a machine without
        // a usable device, and a hard failure would make the whole pipeline
        // unusable rather than slow.
        const int skipNowShared = std::max(2, nPlanes / 24);
        bool onGpu = false;
        if (gpuOnly) {
            ComputeContext* dev = GroupGpu();
            if (dev && GpuSweepReady(dev) && nbs.size() <= 4) {
                SweepPlane sref{rp.v.data(), rp.w, rp.h};
                std::vector<SweepPlane> snb;
                snb.reserve(nbs.size());
                for (int nbi : nbs) {
                    const Plane& q = planes[size_t(nbi)];
                    snb.push_back(SweepPlane{q.v.data(), q.w, q.h});
                }

                // The whole homography table, flattened for one upload. The
                // session holds it on the device so a plane costs a dispatch
                // and nothing else.
                std::vector<double> Hflat(size_t(nPlanes) * nbs.size() * 9);
                for (int p = 0; p < nPlanes; ++p)
                    for (size_t k = 0; k < nbs.size(); ++k)
                        for (int e = 0; e < 9; ++e)
                            Hflat[(size_t(p) * nbs.size() + k) * 9 + size_t(e)] =
                                H[size_t(p) * nbs.size() + k].m[e];

                // ONE LOCK FOR THE WHOLE FRAME. ComputeContext has a single
                // command queue with no locking of its own. Held across every
                // dispatch rather than inside each, because upload, dispatch
                // and readback are one logical operation.
                GpuLock lock(dev);
                GpuSweepSession s;
                std::string gerr;
                if (s.Begin(dev, sref, snb, radius, nPlanes, skipNowShared,
                            Hflat.data(), &gerr)) {
                    bool ok = true;
                    for (int p = 0; p < nPlanes && ok; ++p)
                        ok = s.Plane(p, &gerr);

                    std::vector<float> gb, gp, gn, gr;
                    std::vector<int>   gi;
                    if (ok && s.Finish(&gb, &gp, &gn, &gr, &gi, &gerr)) {
                        for (size_t i = 0; i < nPix; ++i) {
                            PixState& q = st[i];
                            q.best   = gb[i];
                            q.prev   = gp[i];
                            q.next   = gn[i];
                            q.rival  = gr[i];
                            q.bestP  = gi[i];
                        }
                        onGpu = true;
                    }
                }
            }
            // The device did not finish this frame. Hand it back untouched so
            // the caller sweeps it on the parallel CPU path, rather than
            // doing that work here, serially, one frame at a time.
            if (!onGpu) return false;
        }

        for (int p = 0; !onGpu && p < nPlanes; ++p) {
            std::fill(perNb.begin(), perNb.end(), -2.0f);

            for (size_t k = 0; k < nbs.size(); ++k) {
                const Plane& nbP = planes[size_t(nbs[k])];
                const Mat3&  Hp  = H[size_t(p) * nbs.size() + k];

                // Warp the neighbour into the reference frame. A homography
                // is linear in homogeneous coordinates, so the row start and
                // the per-pixel increment are both constant -- no matrix
                // multiply in the inner loop.
                for (int y = 0; y < rp.h; ++y) {
                    double qx = Hp.m[1] * double(y) + Hp.m[2];
                    double qy = Hp.m[4] * double(y) + Hp.m[5];
                    double qz = Hp.m[7] * double(y) + Hp.m[8];
                    const double ax = Hp.m[0], ay = Hp.m[3], az = Hp.m[6];
                    float*   wrow = &warp[size_t(y) * size_t(rp.w)];
                    uint8_t* mrow = &okMask[size_t(y) * size_t(rp.w)];
                    for (int x = 0; x < rp.w; ++x) {
                        float s = 0.0f;
                        bool ok = false;
                        if (std::fabs(qz) > 1e-12)
                            ok = nbP.Sample(qx / qz, qy / qz, &s);
                        wrow[x] = ok ? s : 0.0f;
                        mrow[x] = ok ? uint8_t(1) : uint8_t(0);
                        qx += ax; qy += ay; qz += az;
                    }
                }
                // --- BOX SUMS, NOT INTEGRAL IMAGES -------------------------
                //
                // Same O(1)-per-pixel window sums, in HALF the bytes and
                // without the precision problem that forced doubles.
                //
                // An integral image holds a running total over the whole
                // frame, so at 3072x2048 the sum of a*a reaches 6.3 million
                // while the 7x7 window difference is about 49 -- six orders
                // of magnitude of cancellation, which needs double. A box sum
                // holds only the window, so it never exceeds 49 and float
                // carries far more precision than 8-bit pixels have.
                //
                // That matters because this pass is MEMORY BOUND, not
                // arithmetic bound. At full resolution the integral version
                // moved 792 GB across the eleven frames; halving the element
                // size halves that directly.
                //
                // Separable: a horizontal sliding sum into a scratch row
                // buffer, then a vertical sliding sum down the columns. Each
                // step adds the entering sample and subtracts the leaving
                // one, so the cost per pixel is constant in `window`.
                const int win = 2 * radius + 1;
                const float full = float(win * win);

                // Horizontal pass: for every row, the sum over [x-r, x+r].
                // Four quantities, interleaved so one pass over the row
                // touches one cache line per pixel rather than four.
                for (int y = 0; y < rp.h; ++y) {
                    const float*   aRow = &rp.v[size_t(y) * size_t(rp.w)];
                    const float*   bRow = &warp[size_t(y) * size_t(rp.w)];
                    const uint8_t* mRow = &okMask[size_t(y) * size_t(rp.w)];
                    float* out = &hSum[size_t(y) * size_t(rp.w) * 4];

                    float sB = 0, sBB = 0, sAB = 0, sN = 0;
                    for (int x = 0; x < win && x < rp.w; ++x) {
                        if (mRow[x]) {
                            const float a = aRow[x], b = bRow[x];
                            sB += b; sBB += b * b; sAB += a * b; sN += 1.0f;
                        }
                    }
                    for (int x = radius; x < rp.w - radius; ++x) {
                        float* o = out + size_t(x) * 4;
                        o[0] = sB; o[1] = sBB; o[2] = sAB; o[3] = sN;

                        const int add = x + radius + 1;
                        const int sub = x - radius;
                        if (add < rp.w && mRow[add]) {
                            const float a = aRow[add], b = bRow[add];
                            sB += b; sBB += b * b; sAB += a * b; sN += 1.0f;
                        }
                        if (mRow[sub]) {
                            const float a = aRow[sub], b = bRow[sub];
                            sB -= b; sBB -= b * b; sAB -= a * b; sN -= 1.0f;
                        }
                    }
                }

                // Vertical pass, and the NCC in the same sweep: once the
                // column sum is complete there is no reason to store it.
                for (int x = radius; x < rp.w - radius; ++x) {
                    float sB = 0, sBB = 0, sAB = 0, sN = 0;
                    for (int y = 0; y < win && y < rp.h; ++y) {
                        const float* o = &hSum[(size_t(y) * size_t(rp.w) +
                                                size_t(x)) * 4];
                        sB += o[0]; sBB += o[1]; sAB += o[2]; sN += o[3];
                    }
                    for (int y = radius; y < rp.h - radius; ++y) {
                        // EVERY pixel of the window must have landed inside
                        // the neighbour: a partially sampled window compares
                        // different amounts of image at different depths, and
                        // that biases one plane against another.
                        if (sN >= full - 0.5f) {
                            const float sa  = refA[size_t(y) * size_t(rp.w) +
                                                   size_t(x)];
                            const float saa = refAA[size_t(y) * size_t(rp.w) +
                                                    size_t(x)];

                            const float num = sAB - sa * sB / full;
                            const float da  = saa - sa * sa / full;
                            const float db  = sBB - sB * sB / full;
                            const float den = std::sqrt(da * db);

                            // A FLAT WINDOW HAS NO STRUCTURE TO CORRELATE:
                            // blank sky, a white wall, a blown highlight. The
                            // variance is zero and the correlation is 0/0,
                            // which is most of a photograph of a building, so
                            // it is left unmeasured rather than scored.
                            if (den > 1e-6f)
                                perNb[(size_t(y) * size_t(rp.w) + size_t(x)) *
                                      nbs.size() + k] = num / den;
                        }

                        const int add = y + radius + 1;
                        const int sub = y - radius;
                        if (add < rp.h) {
                            const float* o = &hSum[(size_t(add) * size_t(rp.w) +
                                                    size_t(x)) * 4];
                            sB += o[0]; sBB += o[1]; sAB += o[2]; sN += o[3];
                        }
                        {
                            const float* o = &hSum[(size_t(sub) * size_t(rp.w) +
                                                    size_t(x)) * 4];
                            sB -= o[0]; sBB -= o[1]; sAB -= o[2]; sN -= o[3];
                        }
                    }
                }
            }

            // Combine the neighbours: the best half, for the reason given
            // where that choice is explained.
            const int skipNow = std::max(2, nPlanes / 24);
            for (size_t i = 0; i < nPix; ++i) {
                // THE BEST HALF, ON A FIXED STACK ARRAY.
                //
                // This runs once per pixel per plane -- 302 million times at
                // full resolution -- and it used to clear a std::vector,
                // push_back into it and call std::sort. For at most sixteen
                // values, that is a heap-backed container and a general sort
                // where an insertion into a small array will do, and it cost
                // more than the window sums it was consuming.
                //
                // Insertion sort is the right algorithm at this size: the
                // array is tiny, it is already nearly sorted in the common
                // case, and it has no call overhead at all.
                float top[16];
                int n = 0;
                const size_t base = i * nbs.size();
                for (size_t k = 0; k < nbs.size(); ++k) {
                    const float c = perNb[base + k];
                    if (c <= -1.5f) continue;
                    int j = n++;
                    while (j > 0 && top[j - 1] < c) { top[j] = top[j - 1]; --j; }
                    top[j] = c;
                }

                float combined = -2.0f;
                if (n > 0) {
                    int take = (n + 1) / 2;
                    if (n >= 2 && take < 2) take = 2;
                    float acc = 0.0f;
                    for (int j = 0; j < take; ++j) acc += top[j];
                    combined = acc / float(take);
                }

                PixState& s2 = st[i];

                // This plane is the one AFTER the current best, so it is the
                // best's right shoulder.
                if (s2.bestP == p - 1) s2.next = combined;

                if (combined > s2.best) {
                    // A new winner. The old best becomes a rival only if it
                    // is far enough away to be a separate peak; otherwise it
                    // was the same peak's shoulder.
                    if (s2.bestP >= 0 && std::abs(s2.bestP - p) > skipNow &&
                        s2.best > s2.rival) {
                        s2.rival = s2.best;
                        s2.rivalP = s2.bestP;
                    }
                    s2.best = combined;
                    s2.bestP = p;
                    s2.prev = prevScore[i];
                    s2.next = -2.0f;            // filled on the next plane
                } else if (std::abs(p - s2.bestP) > skipNow &&
                           combined > s2.rival) {
                    s2.rival = combined;
                    s2.rivalP = p;
                }

                prevScore[i] = combined;
            }
        }

        // --- the winner per pixel -------------------------------------------
        //
        // Everything needed was carried forward by the sweep above, so this
        // is a plain pass over the running state rather than a search through
        // a cost volume.
        for (int y = radius; y < rp.h - radius; ++y) {
            for (int x = radius; x < rp.w - radius; ++x) {
                ++tot;
                const PixState& ps = st[size_t(y) * size_t(rp.w) + size_t(x)];

                const double best  = double(ps.best);
                const int    bestP = ps.bestP;
                if (bestP < 0 || best < minCorr) continue;

                // SUBPIXEL REFINEMENT by fitting a parabola through the
                // winning plane and its two neighbours, in inverse depth
                // where the sampling is uniform. Without it a flat wall comes
                // out as visible terraces at the plane spacing.
                double invD = invNear + (double(bestP) / double(nPlanes - 1)) *
                                            (invFar - invNear);
                if (bestP > 0 && bestP < nPlanes - 1) {
                    const double step = (invFar - invNear) / double(nPlanes - 1);
                    const double c0 = double(ps.prev);
                    const double c1 = best;
                    const double c2 = double(ps.next);

                    // CONCAVE DOWN ONLY: the vertex is a MAXIMUM only when
                    // c0 - 2c1 + c2 is negative, and fitting without that
                    // check refines toward a minimum wherever the samples
                    // happen to curve the other way.
                    const double den = c0 - 2.0 * c1 + c2;
                    if (c0 > -1.5 && c2 > -1.5 && den < -1e-12) {
                        double off = 0.5 * (c0 - c2) / den;
                        off = std::clamp(off, -0.5, 0.5);
                        invD += off * step;
                    }
                }
                if (invD < 1e-12) continue;

                // The runner-up, EXCLUDING THE WINNER'S OWN SHOULDER: the
                // planes either side of the peak always score nearly as high,
                // so a plain second-best measures the width of the curve
                // rather than the ambiguity of the match.
                const double second = double(ps.rival);
                const double margin = (second <= -1.5) ? 1.0
                                                       : std::max(0.0, best - second);

                *dv.At<float>(x, y) = float(1.0 / invD);
                *cv.At<float>(x, y) = float(std::clamp(margin, 0.0, 1.0));
                ++meas;
            }
        }

        *measured = meas;
        *total    = tot;

        out->images[size_t(f) * 3 + 0] = std::move(depthImg);
        out->images[size_t(f) * 3 + 1] = std::move(confImg);
        out->images[size_t(f) * 3 + 2] = std::move(colImg);
        return true;
    }

    // Median relative disagreement between the sweep and triangulation, at
    // the pixels where a sparse point was observed. See the call site for
    // what this can and cannot tell us.
    double SparseAgreement(const PointCloud& cloud, const ImageSet& maps,
                           int onlyFrame = -1) const {
        std::vector<double> rel;

        for (const Track& t : cloud.tracks) {
            if (!t.hasPoint) continue;
            for (const Observation& o : t.obs) {
                if (o.frame < 0) continue;
                if (onlyFrame >= 0 && o.frame != onlyFrame) continue;
                const size_t di = size_t(o.frame) * 3;
                if (di >= maps.images.size()) continue;

                const Camera& c = cloud.cameras[size_t(o.frame)];
                if (!c.solved) continue;

                const Vec3 p = c.R * t.point + c.t;
                if (p.z <= 1e-6) continue;

                ImageView dv =
                    const_cast<Image&>(maps.images[di]).MapCpuRead();
                if (!dv.Valid()) continue;
                const int px = int(o.x + 0.5), py = int(o.y + 0.5);
                if (px < 0 || py < 0 || px >= dv.desc.width ||
                    py >= dv.desc.height) continue;

                const float z = *dv.At<float>(px, py);
                if (z <= 0.0f) continue;          // unmeasured here

                rel.push_back(std::fabs(double(z) - p.z) / p.z);
            }
        }
        if (rel.empty()) return -1.0;
        std::sort(rel.begin(), rel.end());
        return rel[rel.size() / 2];
    }

    // One plane's score, combining the neighbours that could see this pixel.
    //
    // THE BEST HALF, NOT THE MEAN, and this is the single most consequential
    // choice in the cost function.
    //
    // Averaging looks like the fair thing and is wrong twice over. On a
    // repeating texture -- a brick wall, a row of windows -- a WRONG depth
    // that shifts the pattern by exactly one period correlates well in every
    // neighbour at once, so the mean does not penalise it at all: the views
    // agree, they are simply all agreeing about the wrong alignment. And a
    // pixel occluded in one view scores badly there however right the depth
    // is, which the mean charges against the correct answer.
    //
    // Taking the best half asks a better question: is there a SUBSTANTIAL
    // SUBSET of views that agree strongly? That tolerates occlusion in a
    // minority of views, and it stops one over-eager view from carrying a
    // depth on its own, because "best half" of four views is still two.
    //
    // This is the same reasoning behind PMVS's visibility set and COLMAP's
    // best-K photometric cost; the difference between methods is largely how
    // carefully this subset is chosen.
    //
    // Recomputed rather than cached: caching every plane's score for every
    // pixel is nPlanes floats per pixel, which at 64 planes is larger than
    // the image itself and is read at most twice.
    double AggregateScore(const Plane& rp, const std::vector<Plane>& planes,
                          const std::vector<int>& nbs,
                          const std::vector<Mat3>& H, int p, int x, int y,
                          int radius, std::vector<double>& scratch) const {
        scratch.clear();
        for (size_t k = 0; k < nbs.size(); ++k) {
            const double c = WindowNcc(rp, planes[size_t(nbs[k])],
                                       H[size_t(p) * nbs.size() + k],
                                       x, y, radius);
            if (c <= -1.5) continue;            // could not sample
            scratch.push_back(c);
        }
        if (scratch.empty()) return -2.0;

        // Descending, then average the leading half (at least one, and at
        // least two whenever two views were sampled -- a single view can
        // always be satisfied by some depth, so letting one carry the
        // decision would be no better than not checking).
        std::sort(scratch.begin(), scratch.end(), std::greater<double>());
        size_t take = (scratch.size() + 1) / 2;
        if (scratch.size() >= 2) take = std::max<size_t>(take, 2);

        double acc = 0.0;
        for (size_t i = 0; i < take; ++i) acc += scratch[i];
        return acc / double(take);
    }

    // HOW MANY DEPTHS TO TEST. The cost is linear in this and so is the depth
    // resolution before subpixel refinement, so it is the main speed/quality
    // dial -- and measured on fountain-P11 it is the one that matters most:
    //
    //   planes   measured   worst frame   agreement   points    time
    //       48      94.0%          2.2%       64.1%   441626      88s
    //       96      96.0%          0.9%       69.1%   486281     199s
    //
    // EVERY AXIS IMPROVES, which is unusual and worth noting. The worst
    // frame's error more than halves, which is the number to watch when a
    // surface comes out in the wrong place: a coarse plane spacing is the
    // likeliest reason a depth is misassigned, since the sweep can only
    // return a depth it actually tested.
    //
    // Left at 64 because the cost is real and linear. Raise it when the
    // geometry matters more than the wait.
    Param<int> m_planes{this, "planes", 64, 8, 512,
        {.help = "How many candidate depths to test between the near and far "
                 "bounds, spaced evenly in INVERSE depth. More is finer but "
                 "costs proportionally; subpixel refinement means the map is "
                 "not simply quantised to this.",
         .softMax = 128.0}};

    // THE CORRELATION WINDOW, and the central trade-off of the whole method.
    //
    // Larger is more robust -- more structure to correlate, less chance a
    // patch matches by accident -- and blurs depth edges, because a window
    // straddling a discontinuity sees two surfaces and can match neither. It
    // also assumes the surface is flat across the window, which is what makes
    // this method fail on anything seen at a grazing angle.
    Param<int> m_window{this, "window", 7, 3, 31,
        {.help = "Side of the square correlation window, in pixels. Larger is "
                 "more robust on weak texture and blurs depth edges; smaller "
                 "keeps edges and is noisier. Odd values only.",
         .step = 2.0}};

    // HOW FAR A NEIGHBOUR MAY BE TURNED AWAY from the reference, in degrees
    // between the two optical axes.
    //
    // A camera that is close but pointed elsewhere shares little of the scene,
    // and what it does share it sees at a very different angle -- so the
    // surface genuinely looks different and correlation is comparing two
    // things that are not the same patch. Excluding those is cheaper and more
    // honest than hoping the cost function notices.
    //
    // 90 degrees is permissive by design: it removes cameras facing away
    // while keeping the wide baselines that give a sweep its depth
    // resolution. Lower it on a capture that turns quickly.
    // HOW FAR BEYOND THE SPARSE POINTS TO SWEEP, as a fraction of their own
    // extent. See the padding in DepthRange for why a surface outside the
    // range is misplaced rather than merely missing.
    // MEASURED on fountain-P11, and the result argues AGAINST raising it:
    //
    //   range_pad   measured   worst frame   agreement   dense points
    //        0.10      94.0%          2.2%       64.1%         441626
    //        0.50      70.8%        434.6%       54.1%         280564
    //        1.00      66.8%        547.0%       23.8%         116542
    //
    // Widening the range with a FIXED plane count spreads the same 48 planes
    // over more depth, so every plane is coarser and more pixels land on the
    // wrong one. The extra reach costs far more than it recovers.
    //
    // This was tried on the theory that a surface beyond the swept range
    // reconstructs pinned to the boundary plane, which is true, but is
    // evidently not what ails this scene. Raising `planes` alongside this
    // would be the honest way to extend reach, at proportional cost.
    Param<float> m_rangePad{this, "range_pad", 0.1f, 0.0f, 2.0f,
        {.help = "How far past the sparse points' depth range to sweep, as a "
                 "fraction of that range. Raising this WITHOUT raising "
                 "`planes` measured worse on fountain-P11: the same planes "
                 "spread over more depth are individually coarser.",
         .step = 0.05, .softMax = 1.0}};

    Param<float> m_maxViewAngle{this, "max_view_angle", 90.0f, 5.0f, 180.0f,
        {.help = "Largest angle, in degrees, between the reference camera's "
                 "optical axis and a neighbour's, for that neighbour to be "
                 "correlated against. Excludes cameras that are nearby but "
                 "aimed elsewhere, which share too little of the scene to "
                 "vote usefully on a depth.",
         .step = 5.0}};

    Param<int> m_neighbours{this, "neighbours", 4, 1, 16,
        {.help = "How many nearby frames each frame is correlated against, "
                 "chosen by camera distance. More views make a wrong depth "
                 "much harder to sustain, at proportional cost."}};

    // THE ACCEPTANCE THRESHOLD. Below this the pixel is left unmeasured
    // rather than guessed, which is why the report states what fraction was
    // measured -- a sweep that fills every pixel on a photograph of a blank
    // wall is lying.
    // MEASURED on fountain-P11, at 48 planes and a 7x7 window, against the
    // sparse cloud:
    //
    //   min_correlation   measured   median error vs triangulation
    //              0.30      99.1%                           0.5%
    //              0.35      98.2%                           0.5%
    //              0.45      94.0%                           0.5%
    //              0.50      90.6%                           0.5%
    //              0.65      75.1%                           0.5%
    //
    // THE ERROR COLUMN DOES NOT MOVE, which is the finding. Over this range
    // the threshold is not buying accuracy, it is only deciding how much of
    // the map gets filled -- so the strictness that felt prudent was
    // discarding a quarter of the image for nothing.
    //
    // Read that column carefully though, because it is measured only where
    // sparse points exist, and those sit on exactly the well-textured spots
    // where correlation is easy. It says the threshold is not rejecting good
    // measurements on textured surface; it cannot say the extra pixels
    // admitted at 0.35 are right, because there is no ground truth there.
    //
    // 0.45 rather than 0.35: the visible black speckle on a brick wall is
    // mortar lines narrowly failing this test, and lowering it fills most of
    // them, but the pixels between 0.35 and 0.45 are unverifiable and a dense
    // map that is confidently wrong is worse than one with holes. This is the
    // honest middle, and the fusion stage will weight by confidence anyway.
    Param<float> m_minCorrelation{this, "min_correlation", 0.45f, -1.0f, 1.0f,
        {.help = "Lowest correlation accepted as a measurement, over the best "
                 "half of the views. Pixels below it get no depth, which is "
                 "the honest answer for untextured regions. Measured on "
                 "fountain-P11, agreement with the sparse points is unchanged "
                 "from 0.30 to 0.65 while coverage moves 99% to 75% -- so "
                 "lower this to fill holes, not to improve accuracy.",
         .step = 0.05}};

    std::string m_note;

    // How the last run split its frames between the device and the CPU.
    // Diagnostics for the report, set by RunDense; nothing reads them back.
    int  m_gpuFrames = 0;
    bool m_gpuAvail  = false;
};

REGISTER_ALGORITHM(PlaneSweep);

}  // namespace
}  // namespace tglab
