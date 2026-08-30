// bundle_adjust — refine every frame's rotation against every match at once.
//
// WHAT IT FIXES, and it is the last error left in the panorama chain.
//
// align_features solves each link independently and composes them in sequence.
// Every link carries a small residual, and composition accumulates it: measured
// on a 15-frame sweep with the stitcher's own overlap-disagreement metric,
//
//     2 frames  11.4%      8 frames  19.0%
//     4 frames  15.3%     15 frames  23.8%
//
// The growth is entirely chain length -- every individual link solves at 90-94%
// inliers. By the middle of the panorama that is tens of pixels, which is the
// doubled ridgeline Tim saw when he zoomed in.
//
// A sequential chain cannot fix this from the inside, because it contains no
// statement about where frame 8 sits relative to frame 5. Bundle adjustment
// supplies exactly that: match each frame against a WINDOW of neighbours and
// solve for all rotations SIMULTANEOUSLY, so an error is distributed across
// every constraint that disagrees with it rather than pushed down the chain.
//
// WHY THIS IS NOT A JOB FOR CERES, which is the natural thing to reach for.
//
// Ceres solves the general problem: autodiff, many solver types, and a sparse
// Schur complement for structure-from-motion with thousands of 3D points. A
// panorama has none of that structure. The unknowns here are three angles per
// frame plus one shared focal length -- 46 numbers for a 15-frame panorama --
// so the normal equations are a DENSE 46x46 system, 16 KB, solvable by the
// Gaussian elimination this codebase already has. The sparse machinery that
// makes Ceres valuable has nothing to be sparse about.
//
// Numeric differentiation is affordable for the same reason: 46 parameters x 2
// evaluations x ~20000 matches is about 39 Mflop per iteration, roughly 20 ms,
// against the 120 s the detectors cost. Autodiff, the other reason to reach for
// Ceres, buys nothing at this size.
//
// THE PARAMETERISATION IS THREE NUMBERS PER FRAME, not nine. An unconstrained
// 3x3 has six degrees of freedom it must not have, and a solver handed them
// will use them -- that is exactly the drift being removed. Angle-axis: the
// direction is the rotation axis, the magnitude the angle in radians. Three
// numbers, every value of which is a valid rotation.
//
// GAUGE FREEDOM: frame 0 is held fixed. Rotating every frame by the same amount
// changes no residual, so without that the system is singular and the solve is
// free to wander. Fixing one frame costs nothing -- the panorama is only ever
// defined up to where it is pointed.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "../../algo_util/features.h"
#include "../../algo_util/transform.h"
#include "../../core/algorithm.h"

namespace tglab {
namespace {

// One correspondence between two frames, in centred pixel coordinates.
struct Obs {
    int   i, j;            // frame indices
    float xi, yi;          // the point in frame i
    float xj, yj;          // the same point in frame j
};

// Angle-axis to a 3x3 rotation, by Rodrigues' formula.
void AxisAngleToR(const double* w, double R[9]) {
    const double t2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
    const double t  = std::sqrt(t2);
    if (t < 1e-12) {
        // The small-angle limit is the identity plus the skew matrix. Taking it
        // explicitly rather than dividing by t avoids a 0/0 at exactly zero,
        // which is where the solve STARTS on a frame with no prior estimate.
        R[0] = 1.0;   R[1] = -w[2]; R[2] =  w[1];
        R[3] =  w[2]; R[4] = 1.0;   R[5] = -w[0];
        R[6] = -w[1]; R[7] =  w[0]; R[8] = 1.0;
        return;
    }
    const double c = std::cos(t), s = std::sin(t);
    const double x = w[0] / t, y = w[1] / t, z = w[2] / t;
    const double C = 1.0 - c;
    R[0] = c + x * x * C;     R[1] = x * y * C - z * s; R[2] = x * z * C + y * s;
    R[3] = y * x * C + z * s; R[4] = c + y * y * C;     R[5] = y * z * C - x * s;
    R[6] = z * x * C - y * s; R[7] = z * y * C + x * s; R[8] = c + z * z * C;
}

// ...and back, for seeding the solve from the transforms align_features found.
void RToAxisAngle(const double R[9], double* w) {
    const double tr = R[0] + R[4] + R[8];
    const double c  = std::clamp((tr - 1.0) * 0.5, -1.0, 1.0);
    const double t  = std::acos(c);
    if (t < 1e-9) { w[0] = w[1] = w[2] = 0.0; return; }
    const double k = t / (2.0 * std::sin(t));
    w[0] = k * (R[7] - R[5]);
    w[1] = k * (R[2] - R[6]);
    w[2] = k * (R[3] - R[1]);
}

// Solves a dense symmetric positive-definite system by Cholesky.
//
// Cholesky rather than the Gaussian elimination in align_features, for two
// reasons that both matter at this size. Normal equations are symmetric
// positive-definite by construction, so half the factorisation is redundant
// work; and Cholesky is backward stable on such a matrix without any pivoting,
// where Gaussian elimination needs pivoting to be. At n=46 the speed is
// irrelevant and the stability is not.
//
// Returns false when the matrix is not positive definite -- which, with
// Levenberg damping applied, means the problem is genuinely degenerate rather
// than merely hard.
bool SolveSpd(std::vector<double>& a, std::vector<double>& b, int n) {
    // In-place Cholesky: a = L L^T, lower triangle.
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = a[size_t(i) * size_t(n) + size_t(j)];
            for (int k = 0; k < j; ++k)
                s -= a[size_t(i) * size_t(n) + size_t(k)] *
                     a[size_t(j) * size_t(n) + size_t(k)];
            if (i == j) {
                if (s <= 0.0) return false;
                a[size_t(i) * size_t(n) + size_t(i)] = std::sqrt(s);
            } else {
                a[size_t(i) * size_t(n) + size_t(j)] =
                    s / a[size_t(j) * size_t(n) + size_t(j)];
            }
        }
    }
    // Forward substitution, then back.
    for (int i = 0; i < n; ++i) {
        double s = b[size_t(i)];
        for (int k = 0; k < i; ++k) s -= a[size_t(i) * size_t(n) + size_t(k)] * b[size_t(k)];
        b[size_t(i)] = s / a[size_t(i) * size_t(n) + size_t(i)];
    }
    for (int i = n - 1; i >= 0; --i) {
        double s = b[size_t(i)];
        for (int k = i + 1; k < n; ++k) s -= a[size_t(k) * size_t(n) + size_t(i)] * b[size_t(k)];
        b[size_t(i)] = s / a[size_t(i) * size_t(n) + size_t(i)];
    }
    return true;
}

class BundleAdjust : public AlgorithmBase {
public:
    const char* Name()     const override { return "bundle_adjust"; }
    const char* Category() const override { return "merge"; }

    PortList Inputs() const override {
        return {{"src", DataType::ImageSet, FormatSpec::Any, ShapeSpec::Any}};
    }
    PortList Outputs() const override {
        return {{"out", DataType::ImageSet, FormatSpec::SameAsInput, ShapeSpec::SameAsInput}};
    }

    void RunCPU(RunCtx&) override {}
    bool IsAligner() const override { return true; }

    bool RunAlign(std::vector<Image>* images, std::string* err) override;

    std::string RunReport() const override {
        if (!m_note.empty()) return m_note;
        if (m_frames == 0) return {};
        char buf[224];
        std::snprintf(buf, sizeof buf,
                      "bundle: %d frames, %d observations, rms %.2f -> %.2f px "
                      "in %d iterations, focal %.0f px",
                      m_frames, m_obs, double(m_rms0), double(m_rms1),
                      m_iters, double(m_focal));
        return buf;
    }

    bool HasGPU() const override { return false; }

private:
    // The reprojection residual for one observation, given the current state.
    //
    // A point in frame i is a ray; rotate it into the reference frame with
    // R_i^T, then into frame j with R_j, and project. Where it lands should be
    // where the match says it is, and the difference is what the solve drives
    // to zero.
    //
    // In PIXELS rather than angles, deliberately: pixels are what the user
    // sees, what the stitcher's disagreement metric reports, and what a
    // threshold can be stated in. An angular residual would need a focal length
    // to interpret and would weight wide-angle frames differently.
    static void Residual(const Obs& o, const std::vector<double>& R,
                         double f, double cx, double cy,
                         double* rx, double* ry) {
        const double* Ri = &R[size_t(o.i) * 9];
        const double* Rj = &R[size_t(o.j) * 9];

        // Frame i pixel -> ray in i -> ray in the reference (R_i transpose).
        const double a = double(o.xi) - cx, b = double(o.yi) - cy, c = f;
        const double wx = Ri[0] * a + Ri[3] * b + Ri[6] * c;
        const double wy = Ri[1] * a + Ri[4] * b + Ri[7] * c;
        const double wz = Ri[2] * a + Ri[5] * b + Ri[8] * c;

        // ...then into frame j, and project.
        const double px = Rj[0] * wx + Rj[1] * wy + Rj[2] * wz;
        const double py = Rj[3] * wx + Rj[4] * wy + Rj[5] * wz;
        const double pz = Rj[6] * wx + Rj[7] * wy + Rj[8] * wz;

        if (pz <= 1e-9) {
            // Behind the camera. A large residual rather than a NaN, so the
            // solve steps away from it instead of dying on it.
            *rx = *ry = 1e4;
            return;
        }
        *rx = (f * px / pz + cx) - double(o.xj);
        *ry = (f * py / pz + cy) - double(o.yj);
    }

    Param<int> m_iterations{this, "iterations", 30, 1, 200,
        {.help = "Gauss-Newton steps. Converges in well under this on real "
                 "data; the cap is a bound on time, not a quality control."}};

    Param<float> m_huber{this, "huber", 3.0f, 0.0f, 50.0f,
        {.help = "Residuals beyond this many pixels are down-weighted rather "
                 "than trusted in full. RANSAC removed the gross outliers "
                 "already, but a few survive every filter and least squares "
                 "gives them the most influence of all. 0 disables it.",
         .step = 0.5}};

    Param<bool> m_refineFocal{this, "refine_focal", true,
        "Solve for the focal length as well as the rotations. It is one extra "
        "unknown against tens of thousands of observations, and a wrong focal "
        "bends every ray -- which the rotations then absorb as a fake tilt."};

    int         m_frames = 0;
    int         m_obs    = 0;
    int         m_iters  = 0;
    float       m_rms0 = 0.0f, m_rms1 = 0.0f;
    float       m_focal = 0.0f;
    std::string m_note;
};

bool BundleAdjust::RunAlign(std::vector<Image>* images, std::string* err) {
    m_frames = 0;
    m_obs    = 0;
    m_iters  = 0;
    m_rms0 = m_rms1 = 0.0f;
    m_note.clear();

    if (!images || images->size() < 2) {
        *err = "bundle_adjust needs a group of at least two images";
        return false;
    }

    const int n = int(images->size());
    const ImageDesc& d0 = (*images)[0].Desc();
    const double cx = 0.5 * double(d0.width);
    const double cy = 0.5 * double(d0.height);

    // --- gather every observation, from every match set ----------------------
    std::vector<Obs> obs;
    for (int j = 0; j < n; ++j) {
        const MatchSidecar* ms = MatchesOf((*images)[size_t(j)]);
        if (!ms) continue;
        const FeatureSidecar* fj = FeaturesOf((*images)[size_t(j)]);
        if (!fj) continue;

        for (const MatchSet& set : ms->sets) {
            if (set.reference < 0 || set.reference >= n) continue;
            const FeatureSidecar* fi = FeaturesOf((*images)[size_t(set.reference)]);
            if (!fi) continue;

            for (const Match& m : set.matches) {
                if (m.a < 0 || size_t(m.a) >= fi->keypoints.size()) continue;
                if (m.b < 0 || size_t(m.b) >= fj->keypoints.size()) continue;
                Obs o;
                o.i  = set.reference;
                o.j  = j;
                o.xi = fi->keypoints[size_t(m.a)].x;
                o.yi = fi->keypoints[size_t(m.a)].y;
                o.xj = fj->keypoints[size_t(m.b)].x;
                o.yj = fj->keypoints[size_t(m.b)].y;
                obs.push_back(o);
            }
        }
    }

    if (obs.empty()) {
        *err = "bundle_adjust: no matches attached -- run a detector and a "
               "matcher before adjusting";
        return false;
    }
    m_frames = n;
    m_obs    = int(obs.size());

    // --- seed from whatever alignment already exists -------------------------
    //
    // Gauss-Newton needs a starting point in the basin of the right minimum,
    // and align_features has already provided one: its chained transforms are
    // right to within the drift being removed. Starting from identity instead
    // would ask the solve to find a 60-degree pan from nothing, which it cannot
    // do -- the linearisation is only valid nearby.
    //
    // The focal seed comes from the frame width unless a transform carries a
    // better one; refine_focal then improves it.
    double focal = double(d0.width);
    std::vector<double> w(size_t(n) * 3, 0.0);
    std::vector<double> R(size_t(n) * 9, 0.0);

    // THE FOCAL SEED HAS TO COME FIRST, because every rotation is read through
    // it: K^-1 H K with the wrong K yields the wrong R.
    //
    // Measured when this was the frame width (5796) instead of the truth
    // (~4830): the solve started at an RMS of 117 px on transforms that
    // produce a visibly clean stitch. That number is the tell -- the seed
    // transforms are good, so a large initial residual can only mean the
    // conversion is misreading them.
    //
    // Estimated the same way stitch_panorama does: a pure rotation gives
    // h02 = f sin(a) and h20 = -sin(a)/f, so f = sqrt(-h02/h20), taken as the
    // MEDIAN over the pairwise links -- a link with little rotation makes both
    // terms nearly zero and their ratio is noise.
    {
        std::vector<double> est;
        for (int i = 1; i < n; ++i) {
            bool ok = false;
            const Affine prevInv = TransformOf((*images)[size_t(i - 1)]).Inverse(&ok);
            if (!ok) continue;
            Affine toC;   toC.m[2] = float(-cx);  toC.m[5] = float(-cy);
            Affine fromC; fromC.m[2] = float(cx); fromC.m[5] = float(cy);
            const Affine L =
                toC.Then(TransformOf((*images)[size_t(i)]).Then(prevInv)).Then(fromC);
            if (std::abs(L.m[6]) > 1e-5f) {
                const double f2 = -double(L.m[2]) / double(L.m[6]);
                if (f2 > 1.0 && std::isfinite(f2)) est.push_back(std::sqrt(f2));
            }
            if (std::abs(L.m[7]) > 1e-5f) {
                const double f2 = -double(L.m[5]) / double(L.m[7]);
                if (f2 > 1.0 && std::isfinite(f2)) est.push_back(std::sqrt(f2));
            }
        }
        if (!est.empty()) {
            std::sort(est.begin(), est.end());
            focal = est[est.size() / 2];
        }

        // CLAMPED HERE TOO, and this is where it actually mattered.
        //
        // The estimate is a median of sqrt(-h02/h20) over the links, and h20 is
        // sin(a)/f -- around 1e-4, small enough that a noisy solve moves it a
        // long way. With AKAZE's 55%-inlier matches the median came back 28022
        // px against a true ~4800, and every rotation was then read through the
        // wrong K: the solve stalled at 25 px RMS instead of converging to 5.
        // ORB's 90%-inlier matches on the same frames estimated correctly.
        //
        // The bounds are the field of view. Half a frame width is about 90
        // degrees horizontally and TWICE it is about 28 -- and 28 degrees is
        // already a long lens for something being stitched. The first attempt
        // used 8x, which is 7 degrees, and did nothing at all: the bad estimate
        // was 28022 px, a 12-degree field, comfortably INSIDE that bound.
        // "Physically possible" was the wrong test; "possible for frames that
        // overlap enough to stitch" is the right one.
        //
        // A clamp on the ITERATION alone was not enough, which is worth
        // recording: it left this untouched, and a bad seed is a bad seed
        // however well the steps behave afterwards.
        focal = std::clamp(focal, 0.5 * double(d0.width), 2.0 * double(d0.width));
    }

    for (int i = 0; i < n; ++i) {
        const Affine t = TransformOf((*images)[size_t(i)]);
        float H[9];
        t.To3x3(H);

        // R = K^-1 H K.
        double M[9];
        for (int c = 0; c < 3; ++c) {
            M[0 * 3 + c] = (double(H[0 * 3 + c]) - cx * double(H[2 * 3 + c])) / focal;
            M[1 * 3 + c] = (double(H[1 * 3 + c]) - cy * double(H[2 * 3 + c])) / focal;
            M[2 * 3 + c] =  double(H[2 * 3 + c]);
        }
        double Rm[9];
        for (int r = 0; r < 3; ++r) {
            Rm[r * 3 + 0] = M[r * 3 + 0] * focal;
            Rm[r * 3 + 1] = M[r * 3 + 1] * focal;
            Rm[r * 3 + 2] = M[r * 3 + 0] * cx + M[r * 3 + 1] * cy + M[r * 3 + 2];
        }

        // ORTHONORMALISE BEFORE CONVERTING. RToAxisAngle reads the angle from
        // the trace and the axis from the antisymmetric part, and BOTH are
        // wrong for a matrix carrying scale -- the trace of 1.4 R is not the
        // trace of R, so the recovered angle is not the rotation's angle. The
        // stored Affine always carries some scale, because its h22 is pinned to
        // 1 and a rotation-induced homography's is not.
        {
            auto dot = [&](int a, int b) {
                return Rm[a * 3] * Rm[b * 3] + Rm[a * 3 + 1] * Rm[b * 3 + 1] +
                       Rm[a * 3 + 2] * Rm[b * 3 + 2];
            };
            const double n0 = std::sqrt(dot(0, 0));
            if (n0 > 1e-12) {
                for (int c = 0; c < 3; ++c) Rm[c] /= n0;
                const double d10 = dot(1, 0);
                for (int c = 0; c < 3; ++c) Rm[3 + c] -= d10 * Rm[c];
                const double n1 = std::sqrt(dot(1, 1));
                if (n1 > 1e-12) {
                    for (int c = 0; c < 3; ++c) Rm[3 + c] /= n1;
                    Rm[6] = Rm[1] * Rm[5] - Rm[2] * Rm[4];
                    Rm[7] = Rm[2] * Rm[3] - Rm[0] * Rm[5];
                    Rm[8] = Rm[0] * Rm[4] - Rm[1] * Rm[3];
                }
            }
        }
        RToAxisAngle(Rm, &w[size_t(i) * 3]);
    }

    auto rebuild = [&]() {
        for (int i = 0; i < n; ++i) AxisAngleToR(&w[size_t(i) * 3], &R[size_t(i) * 9]);
    };
    rebuild();

    const double huber = double(std::max(0.0f, float(m_huber)));
    auto weight = [&](double r2) {
        // Huber: quadratic near zero, linear beyond. The weight applied to the
        // squared residual is what turns least squares into something an
        // outlier cannot dominate.
        if (huber <= 0.0) return 1.0;
        const double r = std::sqrt(r2);
        return (r <= huber) ? 1.0 : huber / r;
    };

    auto totalCost = [&]() {
        double sum = 0.0;
        for (const Obs& o : obs) {
            double rx, ry;
            Residual(o, R, focal, cx, cy, &rx, &ry);
            const double r2 = rx * rx + ry * ry;
            sum += weight(r2) * r2;
        }
        return sum;
    };

    m_rms0 = float(std::sqrt(totalCost() / double(obs.size())));

    // --- Levenberg-Marquardt -------------------------------------------------
    //
    // The parameter vector is [w_1 .. w_{n-1}, focal]: frame 0 is HELD FIXED,
    // because rotating every frame together changes no residual and the system
    // would otherwise be singular. That is the gauge freedom, and fixing it
    // costs nothing -- a panorama is only defined up to where it points.
    const bool doFocal = bool(m_refineFocal);
    const int  nP = (n - 1) * 3 + (doFocal ? 1 : 0);
    if (nP <= 0) {
        m_note = "nothing to adjust: a single frame has no free parameters";
        return true;
    }

    // assign() rather than sized constructors: `vector<double> A(size_t(n))`
    // parses as a function declaration -- the most vexing parse -- and the
    // errors land on every USE rather than here.
    std::vector<double> A, B, JA, JB, wSave;
    A.assign(size_t(nP) * size_t(nP), 0.0);
    B.assign(size_t(nP), 0.0);
    JA.assign(size_t(nP) * size_t(nP), 0.0);
    JB.assign(size_t(nP), 0.0);
    wSave = w;
    double lambda = 1e-3;
    double cost = totalCost();

    const int maxIter = std::max(1, int(m_iterations));
    for (int iter = 0; iter < maxIter; ++iter) {
        std::fill(A.begin(), A.end(), 0.0);
        std::fill(B.begin(), B.end(), 0.0);

        // Numeric Jacobian, one parameter at a time.
        //
        // Central differences would be more accurate and twice the cost; a
        // forward difference is enough here because Gauss-Newton only needs a
        // descent direction, and the step size below is chosen relative to the
        // parameter's own scale rather than being a fixed epsilon.
        for (const Obs& o : obs) {
            double rx, ry;
            Residual(o, R, focal, cx, cy, &rx, &ry);
            const double r2 = rx * rx + ry * ry;
            const double wt = weight(r2);

            // ONLY THE COLUMNS THIS OBSERVATION TOUCHES, which is what keeps
            // the whole thing affordable: two frames of 3 plus the focal is 7
            // non-zero columns out of 46, and the accumulation below skips the
            // rest. Collecting them into a short index list rather than
            // sweeping a 46-wide zero vector per observation is the difference
            // between 20 ms an iteration and 130.
            int  cols[7];
            double djx[7], djy[7];
            int nc = 0;

            auto derivative = [&](int frame) {
                if (frame == 0) return;   // gauge-fixed: frame 0 has no columns
                const int base = (frame - 1) * 3;
                for (int k = 0; k < 3; ++k) {
                    double* wk = &w[size_t(frame) * 3 + size_t(k)];
                    const double save = *wk;
                    // Relative to the parameter's own scale, with a floor: a
                    // fixed epsilon is either lost in rounding for a large
                    // angle or a huge step for one near zero, and the seed
                    // puts several frames near zero exactly.
                    const double h = std::max(1e-7, std::abs(save) * 1e-6);
                    *wk = save + h;
                    AxisAngleToR(&w[size_t(frame) * 3], &R[size_t(frame) * 9]);
                    double px, py;
                    Residual(o, R, focal, cx, cy, &px, &py);
                    *wk = save;
                    AxisAngleToR(&w[size_t(frame) * 3], &R[size_t(frame) * 9]);

                    cols[nc] = base + k;
                    djx[nc]  = (px - rx) / h;
                    djy[nc]  = (py - ry) / h;
                    ++nc;
                }
            };

            derivative(o.i);
            derivative(o.j);

            if (doFocal) {
                const double h = std::max(1e-3, focal * 1e-6);
                const double save = focal;
                focal = save + h;
                double px, py;
                Residual(o, R, focal, cx, cy, &px, &py);
                focal = save;
                cols[nc] = nP - 1;
                djx[nc]  = (px - rx) / h;
                djy[nc]  = (py - ry) / h;
                ++nc;
            }

            // J^T W J and J^T W r, over both residual components at once. Upper
            // triangle only; the mirror happens after the loop.
            for (int p = 0; p < nc; ++p) {
                const int cp = cols[p];
                for (int q = 0; q < nc; ++q) {
                    const int cq = cols[q];
                    if (cq < cp) continue;
                    A[size_t(cp) * size_t(nP) + size_t(cq)] +=
                        wt * (djx[p] * djx[q] + djy[p] * djy[q]);
                }
                B[size_t(cp)] -= wt * (djx[p] * rx + djy[p] * ry);
            }
        }

        // Mirror the upper triangle into the lower: only half was accumulated.
        for (int p = 0; p < nP; ++p)
            for (int q = 0; q < p; ++q)
                A[size_t(p) * size_t(nP) + size_t(q)] =
                    A[size_t(q) * size_t(nP) + size_t(p)];

        // Levenberg damping, then solve; on failure or a worse cost, damp
        // harder and retry. That is the whole of LM: a dial between
        // Gauss-Newton (fast, can overshoot) and gradient descent (safe, slow).
        bool stepped = false;
        for (int attempt = 0; attempt < 8 && !stepped; ++attempt) {
            JA = A;
            JB = B;
            for (int p = 0; p < nP; ++p)
                JA[size_t(p) * size_t(nP) + size_t(p)] *= (1.0 + lambda);

            if (!SolveSpd(JA, JB, nP)) { lambda *= 10.0; continue; }

            wSave = w;
            const double focalSave = focal;
            for (int i = 1; i < n; ++i)
                for (int k = 0; k < 3; ++k)
                    w[size_t(i) * 3 + size_t(k)] += JB[size_t((i - 1) * 3 + k)];
            if (doFocal) {
                // CLAMPED to a physically sensible range, which is not a
                // cosmetic guard.
                //
                // The focal length is one parameter against tens of thousands
                // of observations, so it is normally well determined -- but it
                // trades off against the rotations, and with enough surviving
                // outliers the solve can buy a lower cost by inflating it. On a
                // 15-frame panorama matched with AKAZE at 55% inliers it ran to
                // 28022 px against a true ~4800, and the RMS stalled at 25 px
                // instead of converging to 5. ORB's 90%-inlier matches on the
                // same frames converged fine.
                //
                // Same bounds as the seed clamp above -- see there for why 2x
                // rather than 8x.
                const double lo = 0.5 * double(d0.width);
                const double hi = 2.0 * double(d0.width);
                focal = std::clamp(focal + JB[size_t(nP - 1)], lo, hi);
            }
            rebuild();

            const double next = totalCost();
            if (next < cost) {
                cost = next;
                lambda = std::max(1e-9, lambda * 0.3);
                stepped = true;
            } else {
                w = wSave;
                focal = focalSave;
                rebuild();
                lambda *= 10.0;
            }
        }

        ++m_iters;
        if (!stepped) break;   // damping could not find an improvement
    }

    m_rms1 = float(std::sqrt(cost / double(obs.size())));
    m_focal = float(focal);

    // --- write the refined rotations back as transforms ----------------------
    for (int i = 0; i < n; ++i) {
        const double* Ri = &R[size_t(i) * 9];
        // H = K R K^-1.
        double RK[9], out[9];
        for (int r = 0; r < 3; ++r) {
            RK[r * 3 + 0] = Ri[r * 3 + 0] / focal;
            RK[r * 3 + 1] = Ri[r * 3 + 1] / focal;
            RK[r * 3 + 2] = Ri[r * 3 + 2] - (cx / focal) * Ri[r * 3 + 0]
                                          - (cy / focal) * Ri[r * 3 + 1];
        }
        for (int c = 0; c < 3; ++c) {
            out[0 * 3 + c] = focal * RK[0 * 3 + c] + cx * RK[2 * 3 + c];
            out[1 * 3 + c] = focal * RK[1 * 3 + c] + cy * RK[2 * 3 + c];
            out[2 * 3 + c] =                              RK[2 * 3 + c];
        }
        float h[9];
        for (int k = 0; k < 9; ++k) h[k] = float(out[k]);
        AttachTransform(&(*images)[size_t(i)], Affine::From3x3(h));
    }
    return true;
}

REGISTER_ALGORITHM(BundleAdjust);

} // namespace
} // namespace tglab
