// bundle_adjust_sfm — the joint refinement that ends a reconstruction.
//
// Everything before this stage solved one thing at a time: rotations from the
// view graph, then positions from the rays, then points from the positions.
// Each step took the previous one as fixed and correct, and none of them is.
// Bundle adjustment drops that pretence and moves EVERYTHING at once --
// cameras, points, and optionally focal length -- to minimise the one quantity
// that matters:
//
//     the distance, in pixels, between where a point actually appeared in an
//     image and where the current estimate says it should have appeared.
//
// That is the reprojection error, and it is what "correct" means for a
// reconstruction. It is also why this cannot be skipped: the upstream stages
// each minimise a PROXY (an angle between rotations, an angle between rays),
// and a solution optimal in those terms can still be a pixel or two off in the
// only measure anyone checks.
//
// ---------------------------------------------------------------------------
// THE SCHUR COMPLEMENT, which is not an optimisation but the thing that makes
// this possible at all.
//
// The normal equations have one block per camera (6 parameters) and one per
// point (3). Twenty cameras and twenty thousand points is 60,120 parameters,
// and the dense Hessian for that is
//
//     60120^2 * 8 bytes  =  27 GB.
//
// At a hundred thousand points it is 687 GB. Forming it is not slow, it is
// impossible.
//
// But the Hessian has structure. Writing cameras first and points second:
//
//     H = [ B    E  ]        B: 6x6 blocks, one per camera
//         [ E^T  C  ]        C: 3x3 blocks, one per point
//
// C is BLOCK DIAGONAL, because no two points share a parameter -- a point's
// only coupling to anything is through the cameras that saw it. So C inverts
// in 3x3 pieces, and the points can be eliminated algebraically:
//
//     S = B - E C^-1 E^T          (the reduced camera system, 6n x 6n)
//
// Solve S for the camera update, then back-substitute for the points one at a
// time. For twenty cameras S is 120x120 -- 0.11 MB, a rounding error -- and it
// does not grow with the point count at all. That is the entire trick, and it
// is why every bundle adjuster ever written is built around it.
//
// ---------------------------------------------------------------------------
// GAUGE FREEDOM, again. Reprojection error is unchanged by rotating,
// translating or scaling the whole reconstruction, so the Hessian is singular
// in seven directions and a plain Gauss-Newton step is not unique. Levenberg
// damping handles this incidentally -- adding lambda to the diagonal makes the
// system positive definite whatever the gauge -- which is one reason LM is
// universal here and pure Gauss-Newton is not.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "../../core/algorithm.h"

namespace tglab {
namespace {

// Dense Cholesky for the reduced camera system.
//
// Symmetric positive-definite by construction once damped, so Cholesky rather
// than Gaussian elimination: half the work, and backward stable with no
// pivoting. Returns false when the matrix is not positive definite, which after
// damping means the problem is genuinely degenerate rather than merely hard.
bool SolveSpd(std::vector<double>& a, std::vector<double>& b, int n) {
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
    for (int i = 0; i < n; ++i) {
        double s = b[size_t(i)];
        for (int k = 0; k < i; ++k)
            s -= a[size_t(i) * size_t(n) + size_t(k)] * b[size_t(k)];
        b[size_t(i)] = s / a[size_t(i) * size_t(n) + size_t(i)];
    }
    for (int i = n - 1; i >= 0; --i) {
        double s = b[size_t(i)];
        for (int k = i + 1; k < n; ++k)
            s -= a[size_t(k) * size_t(n) + size_t(i)] * b[size_t(k)];
        b[size_t(i)] = s / a[size_t(i) * size_t(n) + size_t(i)];
    }
    return true;
}

// Inverts a 3x3 in place. Used on each point block of C, which is where the
// Schur complement's whole advantage comes from.
bool Invert3(const double m[9], double out[9]) {
    const double det = m[0] * (m[4] * m[8] - m[5] * m[7]) -
                       m[1] * (m[3] * m[8] - m[5] * m[6]) +
                       m[2] * (m[3] * m[7] - m[4] * m[6]);
    if (std::fabs(det) < 1e-14) return false;
    out[0] = (m[4] * m[8] - m[5] * m[7]) / det;
    out[1] = (m[2] * m[7] - m[1] * m[8]) / det;
    out[2] = (m[1] * m[5] - m[2] * m[4]) / det;
    out[3] = (m[5] * m[6] - m[3] * m[8]) / det;
    out[4] = (m[0] * m[8] - m[2] * m[6]) / det;
    out[5] = (m[2] * m[3] - m[0] * m[5]) / det;
    out[6] = (m[3] * m[7] - m[4] * m[6]) / det;
    out[7] = (m[1] * m[6] - m[0] * m[7]) / det;
    out[8] = (m[0] * m[4] - m[1] * m[3]) / det;
    return true;
}

// One observation, flattened for the solver.
struct Obs {
    int    cam = -1;
    int    pt = -1;
    double u = 0.0, v = 0.0;   // measured pixel position
};

// The state being optimised. Cameras are angle-axis plus centre rather than
// matrix plus translation: an unconstrained optimiser handed nine matrix
// entries walks off the rotation manifold within one step, where three
// angle-axis numbers cannot.
struct State {
    std::vector<double> camRot;    // 3 per camera, angle-axis
    std::vector<double> camPos;    // 3 per camera, world centre
    std::vector<double> point;     // 3 per point
    std::vector<double> focal;     // 1 per camera
};

class BundleAdjustSfm : public AlgorithmBase {
public:
    const char* Name()     const override { return "bundle_adjust_sfm"; }
    const char* Category() const override { return "sfm"; }

    PortList Inputs() const override {
        return {{"src", DataType::PointCloud, FormatSpec::Any, ShapeSpec::Any}};
    }
    PortList Outputs() const override {
        return {{"out", DataType::PointCloud, FormatSpec::Any, ShapeSpec::Any}};
    }

    void RunCPU(RunCtx&) override {}
    bool IsReconstruct() const override { return true; }

    bool RunReconstruct(const std::vector<Image>*, PointCloud* cloud,
                        std::string* err) override;

    std::string RunReport() const override { return m_note; }
    ProxyBehaviour Proxy() const override { return ProxyBehaviour::Never; }
    bool HasGPU() const override { return false; }

private:
    // Reprojection residual and its derivatives, for one observation.
    //
    // Hand-derived rather than by automatic differentiation, which is what
    // Ceres would have supplied. The derivation is mechanical -- chain rule
    // through the projection -- and having it written out is the point: this
    // is a lab, and a bundle adjuster whose Jacobian cannot be read is a black
    // box at the most interesting place in the pipeline.
    //
    // Returns false when the point is behind the camera, where the projection
    // has no derivative worth taking and the observation must be skipped.
    static bool Residual(const State& s, const Obs& o, double cx, double cy,
                         bool doFocal, double* rx, double* ry,
                         double* Jcam, double* Jpt, double* Jf) {
        const double* w = &s.camRot[size_t(o.cam) * 3];
        const double* C = &s.camPos[size_t(o.cam) * 3];
        const double* P = &s.point[size_t(o.pt) * 3];
        const double f = s.focal[size_t(o.cam)];

        // World offset from the camera centre, then into camera coordinates.
        const Vec3 d{P[0] - C[0], P[1] - C[1], P[2] - C[2]};
        const Mat3 R = AxisAngleToMat(Vec3{w[0], w[1], w[2]});
        const Vec3 p = R * d;
        if (p.z <= 1e-6) return false;

        const double invz = 1.0 / p.z;
        const double xn = p.x * invz, yn = p.y * invz;
        *rx = f * xn + cx - o.u;
        *ry = f * yn + cy - o.v;

        // d(proj)/d(camera-space point): the projection Jacobian.
        const double dpx[3] = {f * invz, 0.0, -f * p.x * invz * invz};
        const double dpy[3] = {0.0, f * invz, -f * p.y * invz * invz};

        // --- with respect to the POINT: p = R d, so dp/dP = R.
        for (int k = 0; k < 3; ++k) {
            Jpt[k]     = dpx[0] * R.At(0, k) + dpx[1] * R.At(1, k) + dpx[2] * R.At(2, k);
            Jpt[3 + k] = dpy[0] * R.At(0, k) + dpy[1] * R.At(1, k) + dpy[2] * R.At(2, k);
        }

        // --- with respect to the camera CENTRE: dp/dC = -R.
        for (int k = 0; k < 3; ++k) {
            Jcam[k]     = -Jpt[k];
            Jcam[6 + k] = -Jpt[3 + k];
        }

        // --- with respect to the ROTATION.
        //
        // Numerically, and deliberately. The analytic derivative of Rodrigues'
        // formula with respect to its angle-axis vector has a removable
        // singularity at zero rotation and several pages of algebra around it;
        // a central difference on three parameters costs six matrix builds per
        // observation and is exact to eight digits. This is the one place where
        // the arithmetic is genuinely worse to write than to approximate, and
        // paying for it in the Jacobian rather than in a page of code nobody
        // can check is the right trade in a lab.
        const double eps = 1e-7;
        for (int k = 0; k < 3; ++k) {
            double wp[3] = {w[0], w[1], w[2]};
            wp[k] += eps;
            const Vec3 pPlus = AxisAngleToMat(Vec3{wp[0], wp[1], wp[2]}) * d;
            wp[k] -= 2.0 * eps;
            const Vec3 pMinus = AxisAngleToMat(Vec3{wp[0], wp[1], wp[2]}) * d;

            const double dz = (pPlus.z - pMinus.z) / (2.0 * eps);
            const double dx = (pPlus.x - pMinus.x) / (2.0 * eps);
            const double dy = (pPlus.y - pMinus.y) / (2.0 * eps);
            Jcam[3 + k]     = dpx[0] * dx + dpx[1] * dy + dpx[2] * dz;
            Jcam[6 + 3 + k] = dpy[0] * dx + dpy[1] * dy + dpy[2] * dz;
        }

        if (doFocal) { Jf[0] = xn; Jf[1] = yn; }
        else         { Jf[0] = Jf[1] = 0.0; }
        return true;
    }

    // Huber weight for a residual of magnitude r.
    //
    // WHY A ROBUST LOSS IS NOT OPTIONAL HERE. Squared error weights an
    // observation by the square of how wrong it is, so a single mismatched
    // track -- one that survived RANSAC and track conflict filtering -- pulls
    // the entire solve toward itself. Measured on real data, a few percent of
    // outliers is enough to double the final RMS. Huber is quadratic within
    // the threshold and LINEAR outside it, so a gross outlier contributes a
    // bounded gradient instead of an unbounded one.
    static double HuberWeight(double r, double delta, int kind) {
        if (kind == 0) return 1.0;                       // plain squared error
        const double a = std::fabs(r);
        if (a <= delta) return 1.0;
        if (kind == 1) return delta / a;                 // Huber
        return delta * delta / (delta * delta + a * a);  // Cauchy
    }

    static constexpr const char* kLossNames[] = {"squared", "huber", "cauchy"};

    Param<int> m_loss{this, "loss", 1, 0, 2,
        {.help = "Squared error weights an observation by the SQUARE of how "
                 "wrong it is, so one bad track pulls the whole solve. Huber "
                 "is quadratic near zero and linear beyond the threshold, "
                 "bounding an outlier's influence. Cauchy falls off faster "
                 "still and will discard a genuinely hard but real "
                 "observation, so it is the aggressive choice rather than the "
                 "safe one.",
         .choices = kLossNames, .choiceCount = 3}};

    Param<float> m_lossScale{this, "loss_px", 2.0f, 0.1f, 50.0f,
        {.help = "Where the robust loss starts discounting, in pixels. Set it "
                 "near the reprojection error you expect from good "
                 "observations: too low and real data is treated as outliers, "
                 "too high and outliers are treated as real."}};

    Param<int> m_iterations{this, "iterations", 40, 1, 500,
        {.help = "Levenberg-Marquardt steps. Convergence is usually well "
                 "inside twenty from a good global initialisation; a solve "
                 "still moving at forty is a sign the input geometry is "
                 "wrong rather than merely imprecise."}};

    Param<bool> m_refineFocal{this, "refine_focal", true,
        "Solve for focal length as well as pose. The upstream stages work from "
        "a guessed focal, and a wrong one trades against depth almost exactly "
        "-- so leaving it fixed bakes that error into the geometry. Turn it "
        "off when the calibration is genuinely known."};

    Param<float> m_maxDistance{this, "max_distance", 20.0f, 1.0f, 1000.0f,
        {.help = "How far a point may end up from the cameras before it is "
                 "dropped, as a MULTIPLE of the camera cluster's radius. "
                 "Minimising reprojection error alone lets a point on nearly "
                 "parallel rays slide far outward while the error FALLS, "
                 "because its depth is barely observable -- so the check has "
                 "to happen after the solve, not only before it. Matches "
                 "triangulate's parameter of the same name."}};

    std::string m_note;
};

bool BundleAdjustSfm::RunReconstruct(const std::vector<Image>*, PointCloud* cloud,
                                     std::string* err) {
    const int nCam = int(cloud->cameras.size());
    const int nPt = int(cloud->tracks.size());
    if (nCam < 2) { *err = "bundle_adjust_sfm needs at least two cameras"; return false; }

    // Only triangulated tracks and solved cameras take part. A track with no
    // 3D position has nothing to refine, and an unsolved camera has no pose to
    // start from -- including either would add parameters with no constraints,
    // which is exactly how a bundle adjuster becomes singular.
    std::vector<int> ptIndex(size_t(nPt), -1);
    int nActivePt = 0;
    for (int t = 0; t < nPt; ++t)
        if (cloud->tracks[size_t(t)].hasPoint) ptIndex[size_t(t)] = nActivePt++;

    std::vector<int> camIndex(size_t(nCam), -1);
    int nActiveCam = 0;
    for (int c = 0; c < nCam; ++c)
        if (cloud->cameras[size_t(c)].solved) camIndex[size_t(c)] = nActiveCam++;

    if (nActiveCam < 2 || nActivePt < 1) {
        *err = "bundle_adjust_sfm: nothing to refine -- run rotation_average, "
               "global_position and triangulate first";
        return false;
    }

    std::vector<Obs> obs;
    for (int t = 0; t < nPt; ++t) {
        if (ptIndex[size_t(t)] < 0) continue;
        for (const Observation& o : cloud->tracks[size_t(t)].obs) {
            if (o.frame < 0 || o.frame >= nCam) continue;
            if (camIndex[size_t(o.frame)] < 0) continue;
            obs.push_back(Obs{camIndex[size_t(o.frame)], ptIndex[size_t(t)],
                              double(o.x), double(o.y)});
        }
    }
    if (obs.size() < size_t(nActiveCam) * 6) {
        *err = "bundle_adjust_sfm: too few observations to constrain the cameras";
        return false;
    }

    // Principal points stay fixed: they are far less identifiable than focal
    // length and trade against camera position almost exactly, so refining
    // them on a small problem moves the solution without improving it.
    std::vector<double> cx, cy;
    cx.resize(size_t(nActiveCam));
    cy.resize(size_t(nActiveCam));

    State s;
    s.camRot.resize(size_t(nActiveCam) * 3);
    s.camPos.resize(size_t(nActiveCam) * 3);
    s.focal.resize(size_t(nActiveCam));
    s.point.resize(size_t(nActivePt) * 3);

    for (int c = 0; c < nCam; ++c) {
        const int i = camIndex[size_t(c)];
        if (i < 0) continue;
        const Camera& cam = cloud->cameras[size_t(c)];
        const Vec3 w = MatToAxisAngle(cam.R);
        s.camRot[size_t(i) * 3 + 0] = w.x;
        s.camRot[size_t(i) * 3 + 1] = w.y;
        s.camRot[size_t(i) * 3 + 2] = w.z;
        const Vec3 ctr = cam.Center();
        s.camPos[size_t(i) * 3 + 0] = ctr.x;
        s.camPos[size_t(i) * 3 + 1] = ctr.y;
        s.camPos[size_t(i) * 3 + 2] = ctr.z;
        s.focal[size_t(i)] = cam.focal;
        cx[size_t(i)] = cam.cx;
        cy[size_t(i)] = cam.cy;
    }
    for (int t = 0; t < nPt; ++t) {
        const int i = ptIndex[size_t(t)];
        if (i < 0) continue;
        const Vec3& p = cloud->tracks[size_t(t)].point;
        s.point[size_t(i) * 3 + 0] = p.x;
        s.point[size_t(i) * 3 + 1] = p.y;
        s.point[size_t(i) * 3 + 2] = p.z;
    }

    const bool doFocal = bool(m_refineFocal);
    const int camParams = doFocal ? 7 : 6;   // 3 rotation + 3 centre + focal
    const int nS = nActiveCam * camParams;
    const double delta = double(m_lossScale);
    const int lossKind = int(m_loss);

    // Total squared reprojection error, and the RMS in pixels.
    // AN OBSERVATION BEHIND THE CAMERA COSTS SOMETHING, and that is what makes
    // the cost comparable between iterations.
    //
    // Residual() refuses a point with non-positive depth: there is no
    // projection and no derivative to take. Skipping it in the cost as well --
    // which the first version did -- means the SET of observations being summed
    // changes from step to step. A step that shoves points behind cameras then
    // removes their error from the total and looks like an improvement, and one
    // that brings points back in front adds error and looks like a regression.
    // The comparison `trialCost < cost` is then meaningless, and Levenberg
    // responds by raising lambda until no step is accepted at all.
    //
    // Measured on castle-P19, where 453 of 2421 tracks triangulate behind a
    // camera: bundle adjustment accepted ZERO steps over forty iterations and
    // reported the error unchanged to three decimals.
    //
    // A fixed penalty rather than something derived from the geometry: the
    // quantity has no meaningful magnitude -- the point is on the wrong side of
    // the camera -- and what matters is only that it is large, constant, and
    // counted. The solver then sees a real cost for putting a point behind a
    // camera and a real reward for pulling it back.
    const double kBehindPenalty = 1e4;   // pixels squared, per observation

    auto evaluate = [&](const State& st, double* rms) {
        double sum = 0.0;
        for (const Obs& o : obs) {
            double rx, ry, Jc[14], Jp[6], Jf[2];
            if (!Residual(st, o, cx[size_t(o.cam)], cy[size_t(o.cam)], doFocal,
                          &rx, &ry, Jc, Jp, Jf)) {
                sum += kBehindPenalty;
                continue;
            }
            sum += rx * rx + ry * ry;
        }
        // Divided by EVERY observation, not the ones that happened to project,
        // so the reported RMS is comparable across runs too.
        *rms = obs.empty() ? 0.0 : std::sqrt(sum / double(obs.size()));
        return sum;
    };

    double rms0 = 0.0;
    double cost = evaluate(s, &rms0);

    double lambda = 1e-4;
    int taken = 0;

    for (int iter = 0; iter < int(m_iterations); ++iter) {
        // --- accumulate the blocks ------------------------------------------
        //
        // B: camera-camera, dense per camera but block diagonal across them.
        // C: point-point, 3x3 per point and block diagonal -- the property the
        //    Schur complement exploits.
        // E: camera-point coupling, one block per observation.
        std::vector<double> B(size_t(nS) * size_t(nS), 0.0);
        std::vector<double> Cblk(size_t(nActivePt) * 9, 0.0);
        std::vector<double> gCam(size_t(nS), 0.0);
        std::vector<double> gPt(size_t(nActivePt) * 3, 0.0);

        // E is stored per observation rather than as a matrix: it is the only
        // genuinely sparse part, and materialising it would cost the memory
        // the Schur complement exists to save.
        struct EBlock { int cam, pt; double m[7 * 3]; };
        std::vector<EBlock> E;
        E.reserve(obs.size());

        for (const Obs& o : obs) {
            double rx, ry, Jc[14], Jp[6], Jf[2];
            if (!Residual(s, o, cx[size_t(o.cam)], cy[size_t(o.cam)], doFocal,
                          &rx, &ry, Jc, Jp, Jf))
                continue;

            const double wgt = HuberWeight(std::sqrt(rx * rx + ry * ry),
                                           delta, lossKind);

            // Camera Jacobian rows, with focal appended when refined.
            double A[2][7] = {};
            for (int k = 0; k < 6; ++k) { A[0][k] = Jc[k]; A[1][k] = Jc[6 + k]; }
            if (doFocal) { A[0][6] = Jf[0]; A[1][6] = Jf[1]; }

            const int cbase = o.cam * camParams;
            for (int a = 0; a < camParams; ++a) {
                for (int b = 0; b < camParams; ++b)
                    B[size_t(cbase + a) * size_t(nS) + size_t(cbase + b)] +=
                        wgt * (A[0][a] * A[0][b] + A[1][a] * A[1][b]);
                gCam[size_t(cbase + a)] -= wgt * (A[0][a] * rx + A[1][a] * ry);
            }

            double* Cp = &Cblk[size_t(o.pt) * 9];
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b)
                    Cp[a * 3 + b] += wgt * (Jp[a] * Jp[b] + Jp[3 + a] * Jp[3 + b]);
                gPt[size_t(o.pt) * 3 + size_t(a)] -=
                    wgt * (Jp[a] * rx + Jp[3 + a] * ry);
            }

            EBlock eb;
            eb.cam = o.cam;
            eb.pt = o.pt;
            for (int a = 0; a < camParams; ++a)
                for (int b = 0; b < 3; ++b)
                    eb.m[a * 3 + b] = wgt * (A[0][a] * Jp[b] + A[1][a] * Jp[3 + b]);
            E.push_back(eb);
        }

        // --- Schur complement -----------------------------------------------
        //
        // S = B - E C^-1 E^T, and g_S = g_cam - E C^-1 g_pt.
        //
        // Damping goes on BOTH blocks before the elimination, or the point
        // blocks can be singular -- a point seen by two cameras on a line
        // through it has no determined depth, and its 3x3 is rank deficient.
        std::vector<double> S = B;
        std::vector<double> gS = gCam;

        // ADDITIVE as well as multiplicative, and the additive part is what
        // makes this work on real data.
        //
        // Reprojection error is unchanged by rotating, translating or scaling
        // the whole reconstruction, so the Hessian is SINGULAR in seven
        // directions however good the data is. Scaling the diagonal by
        // (1 + lambda) damps a direction in proportion to what is already
        // there -- which is nothing at all along a gauge direction, so the
        // system stays singular and Cholesky fails at every lambda.
        //
        // Measured on castle-P19: the solve failed on every iteration from
        // lambda 1e-4 upward and bundle adjustment accepted zero steps. The
        // synthetic tests did not catch it because a small, clean, well-spread
        // fixture has enough curvature on the diagonal for the multiplicative
        // term alone to carry it.
        //
        // Scaled by the mean diagonal so the additive term means the same
        // thing whatever units the scene is in -- a reconstruction's scale is
        // itself a gauge freedom, so an absolute epsilon would be arbitrary.
        double diagMean = 0.0;
        for (int a = 0; a < nS; ++a) diagMean += S[size_t(a) * size_t(nS) + size_t(a)];
        diagMean = (nS > 0) ? diagMean / double(nS) : 1.0;
        if (diagMean <= 0.0) diagMean = 1.0;

        for (int a = 0; a < nS; ++a) {
            double& d = S[size_t(a) * size_t(nS) + size_t(a)];
            d = d * (1.0 + lambda) + lambda * diagMean;
        }

        std::vector<double> Cinv(size_t(nActivePt) * 9, 0.0);
        std::vector<bool> ptOk(size_t(nActivePt), false);
        for (int p = 0; p < nActivePt; ++p) {
            double m[9];
            std::copy(&Cblk[size_t(p) * 9], &Cblk[size_t(p) * 9] + 9, m);
            for (int k = 0; k < 3; ++k) m[k * 3 + k] *= (1.0 + lambda);
            ptOk[size_t(p)] = Invert3(m, &Cinv[size_t(p) * 9]);
        }

        // Group the E blocks by point, so each point's contribution to S is
        // accumulated once over the cameras that saw it.
        std::vector<std::vector<int>> byPoint;
        byPoint.resize(size_t(nActivePt));
        for (size_t i = 0; i < E.size(); ++i)
            byPoint[size_t(E[i].pt)].push_back(int(i));

        for (int p = 0; p < nActivePt; ++p) {
            if (!ptOk[size_t(p)]) continue;
            const double* Ci = &Cinv[size_t(p) * 9];
            const double* gp = &gPt[size_t(p) * 3];

            // C^-1 g_p, used by both the gradient and the back-substitution.
            double Cg[3] = {};
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b) Cg[a] += Ci[a * 3 + b] * gp[b];

            for (int ii : byPoint[size_t(p)]) {
                const EBlock& ea = E[size_t(ii)];
                const int abase = ea.cam * camParams;

                for (int a = 0; a < camParams; ++a) {
                    double acc = 0.0;
                    for (int b = 0; b < 3; ++b) acc += ea.m[a * 3 + b] * Cg[b];
                    gS[size_t(abase + a)] -= acc;
                }

                for (int jj : byPoint[size_t(p)]) {
                    const EBlock& eb = E[size_t(jj)];
                    const int bbase = eb.cam * camParams;
                    for (int a = 0; a < camParams; ++a) {
                        // (E C^-1)_a, one row at a time.
                        double row[3] = {};
                        for (int k = 0; k < 3; ++k)
                            for (int b = 0; b < 3; ++b)
                                row[k] += ea.m[a * 3 + b] * Ci[b * 3 + k];
                        for (int b = 0; b < camParams; ++b) {
                            double acc = 0.0;
                            for (int k = 0; k < 3; ++k) acc += row[k] * eb.m[b * 3 + k];
                            S[size_t(abase + a) * size_t(nS) + size_t(bbase + b)] -= acc;
                        }
                    }
                }
            }
        }

        // --- solve and step ---------------------------------------------------
        std::vector<double> dCam = gS;
        std::vector<double> Scopy = S;
        if (!SolveSpd(Scopy, dCam, nS)) { lambda *= 10.0; continue; }

        // Back-substitution: dP = C^-1 (g_p - E^T dCam).
        std::vector<double> dPt(size_t(nActivePt) * 3, 0.0);
        for (int p = 0; p < nActivePt; ++p) {
            if (!ptOk[size_t(p)]) continue;
            double rhs[3] = {gPt[size_t(p) * 3], gPt[size_t(p) * 3 + 1],
                             gPt[size_t(p) * 3 + 2]};
            for (int ii : byPoint[size_t(p)]) {
                const EBlock& e = E[size_t(ii)];
                const int base = e.cam * camParams;
                for (int b = 0; b < 3; ++b)
                    for (int a = 0; a < camParams; ++a)
                        rhs[b] -= e.m[a * 3 + b] * dCam[size_t(base + a)];
            }
            const double* Ci = &Cinv[size_t(p) * 9];
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    dPt[size_t(p) * 3 + size_t(a)] += Ci[a * 3 + b] * rhs[b];
        }

        State trial = s;
        for (int c = 0; c < nActiveCam; ++c) {
            const int base = c * camParams;
            for (int k = 0; k < 3; ++k) {
                // THE ORDER HERE MUST MATCH Residual()'s Jacobian layout,
                // which writes the CENTRE derivative into Jcam[0..2] and the
                // ROTATION derivative into Jcam[3..5].
                //
                // These were swapped, and the symptom was not a wrong answer
                // but a plausible one: every step applied the position
                // correction to the rotation and the rotation correction to
                // the position. Since the two are nearly interchangeable for a
                // distant scene, the solve still made progress and still
                // converged -- to a firm, repeatable, WRONG minimum, identical
                // under every loss function and iteration count. A solver that
                // fails loudly is easier to debug than one that converges
                // confidently to the wrong place.
                trial.camPos[size_t(c) * 3 + size_t(k)] += dCam[size_t(base + k)];
                trial.camRot[size_t(c) * 3 + size_t(k)] += dCam[size_t(base + 3 + k)];
            }
            if (doFocal) {
                // Never let a focal go non-positive: the projection divides by
                // it, and a solve that overshoots into negative focal produces
                // a mirrored reconstruction it can never climb back out of.
                const double f = trial.focal[size_t(c)] + dCam[size_t(base + 6)];
                trial.focal[size_t(c)] = std::max(1.0, f);
            }
        }
        for (int p = 0; p < nActivePt * 3; ++p) trial.point[size_t(p)] += dPt[size_t(p)];

        double trialRms = 0.0;
        const double trialCost = evaluate(trial, &trialRms);

        if (trialCost < cost) {
            // CONVERGED WHEN THE COST STOPS MOVING MEANINGFULLY, not when the
            // iteration cap is reached.
            //
            // Without this the only exit was the cap, because ANY improvement
            // -- however infinitesimal -- is an accepted step. Measured on
            // fountain-P11: every single step was accepted, 40 of 40 and then
            // 100 of 100, and the refined focal came out wherever the cap
            // happened to land. Four starting guesses gave 51.1, 54.7, 54.3
            // and 50.0 degrees, which reads as a solve with several minima and
            // is really one solve stopped at four arbitrary points.
            //
            // RELATIVE rather than absolute, because the cost is a sum of
            // squared pixel errors over every observation: its scale depends
            // on how many there are, so any fixed threshold would mean
            // something different on each reconstruction.
            // NO EARLY-EXIT ON A SMALL IMPROVEMENT, and this was tried and
            // measured. A relative-improvement test -- break once the cost
            // stops moving, with patience over five consecutive steps -- looks
            // like the standard Levenberg termination and made every result
            // WORSE here: runs that had been reaching rms 0.44 in 110-245
            // steps stopped at 4-9 steps and 0.46-0.78, because LM takes tiny
            // steps early while lambda is still large and that phase is
            // indistinguishable from convergence by cost alone.
            //
            // So the iteration cap remains the terminator, with the lambda
            // bail-out below for a solve that genuinely cannot improve. A
            // proper test would be on the gradient or the step NORM rather
            // than on the cost delta; until one is measured to help, running
            // to the cap is the honest default.
            s = std::move(trial);
            cost = trialCost;
            lambda = std::max(1e-10, lambda * 0.3);
            ++taken;
        } else {
            lambda *= 10.0;
            if (lambda > 1e12) break;   // no step helps; the solve is done
        }
    }

    double rms1 = 0.0;
    evaluate(s, &rms1);

    // --- write back ---------------------------------------------------------
    for (int c = 0; c < nCam; ++c) {
        const int i = camIndex[size_t(c)];
        if (i < 0) continue;
        Camera& cam = cloud->cameras[size_t(c)];
        cam.R = AxisAngleToMat(Vec3{s.camRot[size_t(i) * 3],
                                    s.camRot[size_t(i) * 3 + 1],
                                    s.camRot[size_t(i) * 3 + 2]});
        const Vec3 ctr{s.camPos[size_t(i) * 3], s.camPos[size_t(i) * 3 + 1],
                       s.camPos[size_t(i) * 3 + 2]};
        cam.t = cam.R * ctr * -1.0;
        if (doFocal) cam.focal = s.focal[size_t(i)];
    }
    // --- POINTS BUNDLE ADJUSTMENT PUSHED OUT OF THE SCENE -------------------
    //
    // Triangulation rejects a point too far from the cameras, but it can only
    // speak for the cloud IT produced. Measured on castle-P19: triangulate
    // handed over a cloud 13.3 units across with nothing beyond its distance
    // limit, and bundle adjustment returned one 65.9 units across -- a 5x
    // inflation, created here, from roughly 45 points of 1129.
    //
    // It is not a malfunction. BA minimises reprojection error and nothing
    // else, and a point on a nearly-parallel pair of rays can slide a long way
    // outward while REDUCING that error: the same run improved rms from 2.155
    // to 0.435 px while flinging those points out. Depth is barely observable
    // in that configuration, so the optimiser is free to trade it for a
    // fraction of a pixel.
    //
    // Rejected rather than clamped. A clamp would place the point at an
    // arbitrary distance and keep it in the cloud as though it were measured,
    // and this point is not badly scaled -- its depth was never determined.
    // Dropping it says so honestly, and the track survives for a later
    // retriangulation once the cameras are better.
    //
    // The same camera-cluster reference as triangulate's test, for the same
    // reason: a reconstruction has no metric scale, so the cameras are the
    // only length available.
    Vec3 camMean{0, 0, 0};
    int camCount = 0;
    for (const Camera& c : cloud->cameras) {
        if (!c.solved) continue;
        camMean = camMean + c.Center();
        ++camCount;
    }
    double camRadius = 0.0;
    if (camCount > 0) {
        camMean = camMean * (1.0 / double(camCount));
        for (const Camera& c : cloud->cameras) {
            if (!c.solved) continue;
            camRadius = std::max(camRadius, (c.Center() - camMean).Norm());
        }
    }
    const double maxDist =
        (camRadius > 1e-9) ? camRadius * double(m_maxDistance) : 0.0;

    int dropped = 0;
    for (int t = 0; t < nPt; ++t) {
        const int i = ptIndex[size_t(t)];
        if (i < 0) continue;
        Track& tr = cloud->tracks[size_t(t)];
        const Vec3 p{s.point[size_t(i) * 3], s.point[size_t(i) * 3 + 1],
                     s.point[size_t(i) * 3 + 2]};
        if (maxDist > 0.0 && (p - camMean).Norm() > maxDist) {
            tr.hasPoint = false;
            ++dropped;
            continue;
        }
        tr.point = p;
    }

    // The mean refined focal, expressed as a horizontal field of view so it is
    // comparable with relative_pose's fov_deg parameter directly.
    double meanFovDeg = 0.0;
    {
        int nf = 0;
        for (const Camera& c : cloud->cameras) {
            if (!c.solved || c.focal <= 0.0) continue;
            meanFovDeg += 2.0 * std::atan2(0.5 * double(c.width), c.focal) *
                          180.0 / 3.14159265358979;
            ++nf;
        }
        if (nf > 0) meanFovDeg /= double(nf);
    }
    char fovBuf[64] = "";

    char buf[400];
    std::snprintf(buf, sizeof buf,
                  "bundle: %d cameras, %d points, %d observations; rms %.3f -> "
                  "%.3f px in %d accepted steps (%s loss%s); %d pushed past "
                  "%.1fx camera radius and dropped",
                  nActiveCam, nActivePt, int(obs.size()), rms0, rms1, taken,
                  kLossNames[lossKind],
                  // THE REFINED FOCAL, AS A FIELD OF VIEW, because it is the
                  // one place the pipeline answers back about its own most
                  // consequential input. fov_deg upstream is a guess; bundle
                  // adjustment solves for the focal against reprojection error
                  // in pixels, so where it settles is a measurement of what
                  // the lens actually was. A guess far from it is worth fixing
                  // at the source rather than letting BA absorb.
                  doFocal ? (std::snprintf(fovBuf, sizeof fovBuf,
                                           ", focal refined to fov %.1f deg",
                                           meanFovDeg),
                             fovBuf)
                          : "",
                  dropped, double(m_maxDistance));
    m_note = buf;
    return true;
}

REGISTER_ALGORITHM(BundleAdjustSfm);

}  // namespace
}  // namespace tglab
