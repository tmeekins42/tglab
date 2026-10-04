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

#include "../../algo_util/least_squares.h"
#include "../../algo_util/linalg.h"
#include "../../algo_util/sparse_cholesky.h"
#include "../../core/algorithm.h"
#include "../../core/parallel.h"

namespace tglab {
namespace {

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

    // The robust loss: see least_squares.h for why it is not optional. The
    // parameter's 0, 1, 2 are RobustLoss's Squared, Huber, Cauchy.
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

    // ONE LENS TOOK EVERY FRAME, so there is ONE focal to solve for.
    //
    // A FOCAL PER CAMERA LETS EACH ONE FIND ITS OWN LOCAL MINIMUM. That is
    // the real cost, and it is worse than the wasted parameters suggest. A
    // camera with a slightly wrong pose can reduce its own reprojection error
    // by bending its focal to suit, which makes the wrong pose FIT -- so the
    // solve settles there instead of correcting the pose. Every camera can do
    // this independently, and the result is a set of frames each internally
    // consistent and mutually disagreeing.
    //
    // Measured on castle-P19 before this: focals spread from 48.9 to 81.9
    // degrees across nineteen frames shot on one camera. Not noise around a
    // true value -- a 33-degree disagreement is the solver having found
    // nineteen different answers, each locally optimal.
    //
    // Sharing removes those degrees of freedom, so the only way to reduce
    // reprojection error is to fix the geometry. Measured on fountain-P11,
    // the focal then converges to 57.2-57.8 degrees from starting guesses of
    // 40, 50 and 58: three independent starts agreeing, which is what makes
    // it a measurement rather than an echo of the input.
    //
    // Shared is the right default because every capture this pipeline is
    // aimed at -- a walk-around, a bracket, a panorama -- is one camera. A
    // mixed set is the exception and turns this off.
    Param<bool> m_sharedFocal{this, "shared_focal", true,
        "Solve ONE focal length for the whole group rather than one per "
        "camera. Correct whenever a single camera took every frame, which is "
        "the usual case: separate focals let the solver absorb geometric "
        "error into intrinsics that cannot physically differ. Turn it off "
        "only for a set genuinely shot on different cameras or at different "
        "zooms."};

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

    // With a shared focal the steps are averaged, which keeps the focals
    // equal only if they START equal. Upstream gives every camera the same
    // guess so they normally are, but a second bundle pass inherits whatever
    // the first produced -- so make the invariant hold rather than assume it.
    if (bool(m_refineFocal) && bool(m_sharedFocal) && nActiveCam > 0) {
        double mean = 0.0;
        for (int c = 0; c < nActiveCam; ++c) mean += s.focal[size_t(c)];
        mean /= double(nActiveCam);
        for (int c = 0; c < nActiveCam; ++c) s.focal[size_t(c)] = mean;
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
    const bool shareFocal = doFocal && bool(m_sharedFocal) && nActiveCam > 1;

    // PARAMETER LAYOUT. Six per camera always -- centre then rotation -- plus
    // the focal.
    //
    // With a shared focal the focal is ONE column at the end of the system
    // rather than one per camera, because that is what "every frame was shot
    // on the same lens" means as a constraint. Every observation's focal
    // derivative then accumulates into that single column, so all the
    // evidence in the group bears on one number.
    //
    // An earlier attempt imposed this by solving per-camera focals and
    // AVERAGING the steps. That is the projection of the step onto the shared
    // subspace and it does converge to the right place eventually, but the
    // averaged step is a fraction of the true one -- measured on
    // fountain-P11, the focal moved from 50.0 to 49.9 in 24 iterations and
    // from 40.0 to 40.0, which reads exactly like the damping bug it was
    // meant to fix. One shared column moves it properly in one step.
    const int perCam = doFocal && !shareFocal ? 7 : 6;
    const int camParams = perCam;             // 3 centre + 3 rotation [+ focal]
    const int nS = nActiveCam * camParams + (shareFocal ? 1 : 0);
    const int focalCol = shareFocal ? nActiveCam * camParams : -1;
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

    // Observations in fixed chunks, each summed on its own thread and the
    // partial sums added in chunk order -- the same total whatever the
    // thread count, which matters because accepting a step compares costs.
    constexpr size_t kObsChunk = 4096;
    const size_t nChunks = (obs.size() + kObsChunk - 1) / kObsChunk;

    auto evaluate = [&](const State& st, double* rms) {
        std::vector<double> part(nChunks, 0.0);
        ParallelFor(nChunks, [&](size_t ci) {
            double sum = 0.0;
            const size_t end = std::min(obs.size(), (ci + 1) * kObsChunk);
            for (size_t i = ci * kObsChunk; i < end; ++i) {
                const Obs& o = obs[i];
                double rx, ry, Jc[14], Jp[6], Jf[2];
                if (!Residual(st, o, cx[size_t(o.cam)], cy[size_t(o.cam)], doFocal,
                              &rx, &ry, Jc, Jp, Jf)) {
                    sum += kBehindPenalty;
                    continue;
                }
                sum += rx * rx + ry * ry;
            }
            part[ci] = sum;
        });
        double sum = 0.0;
        for (double p : part) sum += p;
        // Divided by EVERY observation, not the ones that happened to project,
        // so the reported RMS is comparable across runs too.
        *rms = obs.empty() ? 0.0 : std::sqrt(sum / double(obs.size()));
        return sum;
    };

    double rms0 = 0.0;
    double cost = evaluate(s, &rms0);

    // THE REDUCED CAMERA SYSTEM IS SPARSE, and is stored and factored so.
    //
    // A node per camera, plus one for the shared focal, which couples to
    // every camera. Two cameras' block is non-zero only when some point is
    // seen by both, so the pattern comes straight from the observations and
    // does not change from one iteration to the next.
    //
    // It was dense: S, B and a copy per elimination worker, each nS x nS.
    // Fine at 179 cameras (9 MB a copy); at a room scan's 1577 that was
    // 720 MB a copy and a 9500-unknown Cholesky on every step -- 887 s of
    // bundle adjustment, for a matrix that is nearly all zeros. Sparse, the
    // same solve took 78 s to the same answer, step for step; at 179
    // cameras, where the system is nearly dense anyway, it is level with the
    // dense code (the factor is the same kernel, run on supernode panels).
    const int focalNode = shareFocal ? nActiveCam : -1;
    linalg::BlockSparseCholesky sys;
    {
        std::vector<int> sizes(static_cast<size_t>(nActiveCam), camParams);
        if (shareFocal) sizes.push_back(1);
        std::vector<std::vector<int>> camsOf(static_cast<size_t>(nActivePt));
        for (const Obs& o : obs) camsOf[size_t(o.pt)].push_back(o.cam);
        std::vector<std::pair<int, int>> pairs;
        for (auto& cs : camsOf) {
            std::sort(cs.begin(), cs.end());
            cs.erase(std::unique(cs.begin(), cs.end()), cs.end());
            for (size_t a = 0; a < cs.size(); ++a)
                for (size_t b = a + 1; b < cs.size(); ++b) pairs.push_back({cs[b], cs[a]});
        }
        if (shareFocal)
            for (int c = 0; c < nActiveCam; ++c) pairs.push_back({focalNode, c});
        if (!sys.Analyse(sizes, pairs)) { *err = "bundle_adjust_sfm: bad system pattern"; return false; }
    }
    // A scalar's diagonal entry in the sparse values, for the damping.
    auto diagAt = [&](int a) -> size_t {
        const int node = (shareFocal && a == focalCol) ? focalNode : a / camParams;
        const int within = a - sys.Scalar(node);
        return size_t(sys.Offset(node, node)) + size_t(within * sys.Size(node) + within);
    };

    // Each elimination worker's own share of S and g_S (see the elimination
    // below), allocated once for the whole solve.
    std::vector<std::vector<double>> workerS(ParallelThreads(size_t(nActivePt)));
    std::vector<std::vector<double>> workerG(workerS.size());
    for (size_t w = 0; w < workerS.size(); ++w) {
        workerS[w].assign(sys.NumValues(), 0.0);
        workerG[w].assign(size_t(nS), 0.0);
    }

    // The accumulated blocks, kept across iterations: see "only when the state
    // has moved" below.
    std::vector<double> B(sys.NumValues(), 0.0);
    std::vector<double> Cblk(size_t(nActivePt) * 9, 0.0);
    std::vector<double> gCam(size_t(nS), 0.0);
    std::vector<double> gPt(size_t(nActivePt) * 3, 0.0);
    // E is stored per observation rather than as a matrix: it is the only
    // genuinely sparse part, and materialising it would cost the memory the
    // Schur complement exists to save.
    struct EBlock { int cam, pt; double m[7 * 3]; };
    std::vector<EBlock> E;
    E.reserve(obs.size());
    std::vector<std::vector<int>> byPoint;
    byPoint.resize(size_t(nActivePt));
    struct ObsJ { bool ok; double rx, ry, Jc[14], Jp[6], Jf[2]; };
    std::vector<ObsJ> jac(obs.size());
    bool blocksValid = false;

    LmDamping damp{1e-4, 10.0, 0.3, 1e-10, 1e12};
    int taken = 0;

    for (int iter = 0; iter < int(m_iterations); ++iter) {
        if (GroupCancelled()) { *err = "cancelled"; return false; }   // see SetGroupCancel
        // --- accumulate the blocks ------------------------------------------
        //
        // B: camera-camera, dense per camera but block diagonal across them.
        // C: point-point, 3x3 per point and block diagonal -- the property the
        //    Schur complement exploits.
        // E: camera-point coupling, one block per observation.
        //
        // ONLY WHEN THE STATE HAS MOVED. A rejected step leaves the cameras
        // and points where they were, and these depend on nothing else -- the
        // damping enters only below -- so they are kept. That is most
        // iterations: Levenberg-Marquardt ends by raising the damping until no
        // step helps, and measured on a 179-camera video a solve of 21
        // iterations accepted 3.
        if (!blocksValid) {
        std::fill(B.begin(), B.end(), 0.0);
        std::fill(Cblk.begin(), Cblk.end(), 0.0);
        std::fill(gCam.begin(), gCam.end(), 0.0);
        std::fill(gPt.begin(), gPt.end(), 0.0);
        E.clear();

        // Residuals and Jacobians in parallel, the summing below serially.
        ParallelFor(nChunks, [&](size_t ci) {
            const size_t end = std::min(obs.size(), (ci + 1) * kObsChunk);
            for (size_t i = ci * kObsChunk; i < end; ++i) {
                const Obs& o = obs[i];
                ObsJ& j = jac[i];
                j.ok = Residual(s, o, cx[size_t(o.cam)], cy[size_t(o.cam)], doFocal, &j.rx,
                                &j.ry, j.Jc, j.Jp, j.Jf);
            }
        });

        for (size_t oi = 0; oi < obs.size(); ++oi) {
            const Obs& o = obs[oi];
            const ObsJ& j = jac[oi];
            if (!j.ok) continue;
            const double rx = j.rx, ry = j.ry;
            const double* Jc = j.Jc;
            const double* Jp = j.Jp;
            const double* Jf = j.Jf;

            const double wgt = RobustWeight(RobustLoss(lossKind),
                                            std::sqrt(rx * rx + ry * ry), delta);

            // Camera Jacobian rows, with focal appended when refined.
            double A[2][7] = {};
            for (int k = 0; k < 6; ++k) { A[0][k] = Jc[k]; A[1][k] = Jc[6 + k]; }
            if (doFocal) { A[0][6] = Jf[0]; A[1][6] = Jf[1]; }

            // Where each of this observation's camera parameters lands in the
            // reduced system. The six pose parameters go in this camera's own
            // block; the focal goes either alongside them (per-camera) or into
            // the one shared column at the end.
            const int cbase = o.cam * camParams;
            int idx[7];
            for (int a = 0; a < camParams; ++a) idx[a] = cbase + a;
            const int nA = shareFocal ? camParams + 1 : camParams;
            if (shareFocal) {
                idx[camParams] = focalCol;
                A[0][camParams] = Jf[0];
                A[1][camParams] = Jf[1];
            }

            // B is this camera's own block, plus -- when the focal is shared --
            // its coupling to the focal node and the focal's own entry.
            double M[8 * 8];
            for (int a = 0; a < nA; ++a) {
                for (int b = 0; b < nA; ++b)
                    M[a * nA + b] = wgt * (A[0][a] * A[0][b] + A[1][a] * A[1][b]);
                gCam[size_t(idx[a])] -= wgt * (A[0][a] * rx + A[1][a] * ry);
            }
            double own[7 * 7];
            for (int a = 0; a < camParams; ++a)
                for (int b = 0; b < camParams; ++b) own[a * camParams + b] = M[a * nA + b];
            sys.Add(B.data(), o.cam, o.cam, own);
            if (shareFocal) {
                sys.Add(B.data(), focalNode, o.cam, &M[camParams * nA]);
                sys.Add(B.data(), focalNode, focalNode, &M[camParams * nA + camParams]);
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
            for (int a = 0; a < nA; ++a)
                for (int b = 0; b < 3; ++b)
                    eb.m[a * 3 + b] = wgt * (A[0][a] * Jp[b] + A[1][a] * Jp[3 + b]);
            E.push_back(eb);
        }

        // Group the E blocks by point, so each point's contribution to S is
        // accumulated once over the cameras that saw it.
        for (auto& v : byPoint) v.clear();
        for (size_t i = 0; i < E.size(); ++i)
            byPoint[size_t(E[i].pt)].push_back(int(i));
        blocksValid = true;
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
        // PER PARAMETER KIND, not one mean over all of them, and this is what
        // makes refining the focal actually work.
        //
        // The three parameter kinds have derivatives in different units:
        // rotation is pixels per radian, position is pixels per scene unit,
        // and the focal is pixels per pixel -- dx/df is just the normalised
        // coordinate, order 1. Their diagonal entries therefore differ by
        // orders of magnitude, and a single mean is dominated by whichever is
        // largest.
        //
        // The additive term lambda * diagMean was then far bigger than the
        // focal's own curvature, so the focal direction was damped to a
        // standstill: measured on fountain-P11, every camera's focal stayed
        // at its starting value to four significant figures, from any start.
        // "focal refined to fov 50.0" from an input of 50.0 -- and equally
        // 40.0 from 40.0, and 60.0 from 60.0. It looked like a measurement
        // and was an echo.
        //
        // Scaling each kind by its OWN mean keeps what the additive term is
        // for -- the gauge directions have no curvature at all, so a purely
        // multiplicative damping leaves the system singular -- while letting
        // each kind move at its own scale.
        double diagSum[3] = {0.0, 0.0, 0.0};
        int    diagN[3]   = {0, 0, 0};
        auto kindOf = [&](int a) {
            if (a >= nActiveCam * camParams) return 2;   // the shared focal
            const int within = a % camParams;
            if (within < 3) return 0;              // centre
            if (within < 6) return 1;              // rotation
            return 2;                              // per-camera focal
        };
        for (int a = 0; a < nS; ++a) {
            const int k = kindOf(a);
            diagSum[k] += S[diagAt(a)];
            ++diagN[k];
        }
        double diagMean[3];
        for (int k = 0; k < 3; ++k) {
            diagMean[k] = (diagN[k] > 0) ? diagSum[k] / double(diagN[k]) : 1.0;
            if (diagMean[k] <= 0.0) diagMean[k] = 1.0;
        }

        for (int a = 0; a < nS; ++a) {
            double& d = S[diagAt(a)];
            d = d * (1.0 + damp.lambda) + damp.lambda * diagMean[kindOf(a)];
        }

        std::vector<double> Cinv(size_t(nActivePt) * 9, 0.0);
        std::vector<bool> ptOk(size_t(nActivePt), false);
        // Each point's own 3x3 block inverted, which is where the Schur
        // complement's whole advantage comes from.
        for (int p = 0; p < nActivePt; ++p) {
            Mat3 m, inv;
            std::copy(&Cblk[size_t(p) * 9], &Cblk[size_t(p) * 9] + 9, m.m);
            for (int k = 0; k < 3; ++k) m.m[k * 3 + k] *= (1.0 + damp.lambda);
            ptOk[size_t(p)] = m.Inverse(&inv);
            if (ptOk[size_t(p)]) std::copy(inv.m, inv.m + 9, &Cinv[size_t(p) * 9]);
        }

        // THE ELIMINATION, S -= E C^-1 E^T and g_S -= E C^-1 g_p, point by
        // point. A point seen by L cameras couples every pair of them, so
        // this is L^2 small products per point, and it was most of a bundle
        // adjustment's time -- 480 ms of every iteration on a 179-camera
        // video (36k points, tracks up to 100 frames long). Three things
        // cut it down, none of which changes the arithmetic beyond the
        // order of summation:
        //
        //   E C^-1 is formed ONCE per observation, rather than again for
        //   every pair the observation is in;
        //   only pairs ii <= jj are formed, the other half by symmetry;
        //   points are dealt round-robin to a fixed set of workers, each
        //   summing into its own copy of S and g_S, added together at the
        //   end -- S is shared by every point, so threads cannot write it
        //   directly.
        //
        // Slot `a` of camera `c`'s block, as an index into the reduced
        // system. Slots 0..camParams-1 are that camera's own; the extra slot,
        // present only when the focal is shared, is the one column every
        // camera contributes to.
        const int nE = shareFocal ? camParams + 1 : camParams;
        auto slot = [&](int cam, int a) {
            return (shareFocal && a == camParams) ? focalCol : cam * camParams + a;
        };
        ParallelFor(workerS.size(), [&](size_t w) {
            std::vector<double>& Sl = workerS[w];
            std::vector<double>& gl = workerG[w];
            std::fill(Sl.begin(), Sl.end(), 0.0);
            std::fill(gl.begin(), gl.end(), 0.0);
            std::vector<double> EC;   // this point's E C^-1, nE x 3 per observation
            for (size_t p = w; p < size_t(nActivePt); p += workerS.size()) {
                if (!ptOk[p]) continue;
                const double* Ci = &Cinv[p * 9];
                const double* gp = &gPt[p * 3];
                const std::vector<int>& seen = byPoint[p];
                const size_t L = seen.size();

                // C^-1 g_p, for the gradient.
                double Cg[3] = {};
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b) Cg[a] += Ci[a * 3 + b] * gp[b];

                EC.resize(L * size_t(nE) * 3);
                for (size_t q = 0; q < L; ++q) {
                    const EBlock& e = E[size_t(seen[q])];
                    double* ec = &EC[q * size_t(nE) * 3];
                    for (int a = 0; a < nE; ++a) {
                        for (int k = 0; k < 3; ++k) {
                            double v = 0.0;
                            for (int b = 0; b < 3; ++b) v += e.m[a * 3 + b] * Ci[b * 3 + k];
                            ec[a * 3 + k] = v;
                        }
                        double acc = 0.0;
                        for (int b = 0; b < 3; ++b) acc += e.m[a * 3 + b] * Cg[b];
                        gl[size_t(slot(e.cam, a))] -= acc;
                    }
                }

                // Each pair's nE x nE product, -E1 C^-1 E2^T, goes into the
                // sparse blocks it belongs to. The matrix is symmetric and
                // only its lower blocks are stored (diagonal blocks in full),
                // so of the dense version's two writes per pair -- the block
                // and its mirror -- whichever lands in stored space is kept:
                // the camera-camera block once, in either orientation (Add
                // transposes as needed), a camera's diagonal block both ways,
                // and the focal row, which is below every camera.
                for (size_t q1 = 0; q1 < L; ++q1) {
                    const EBlock& e1 = E[size_t(seen[q1])];
                    const double* ec = &EC[q1 * size_t(nE) * 3];
                    for (size_t q2 = q1; q2 < L; ++q2) {
                        const EBlock& e2 = E[size_t(seen[q2])];
                        double V[8 * 8];
                        for (int a = 0; a < nE; ++a)
                            for (int b = 0; b < nE; ++b)
                                V[a * nE + b] = -(ec[a * 3] * e2.m[b * 3] +
                                                  ec[a * 3 + 1] * e2.m[b * 3 + 1] +
                                                  ec[a * 3 + 2] * e2.m[b * 3 + 2]);
                        double cc[7 * 7];
                        for (int a = 0; a < camParams; ++a)
                            for (int b = 0; b < camParams; ++b) cc[a * camParams + b] = V[a * nE + b];
                        sys.Add(Sl.data(), e1.cam, e2.cam, cc);
                        if (e1.cam == e2.cam && q1 != q2) {
                            double ct[7 * 7];
                            for (int a = 0; a < camParams; ++a)
                                for (int b = 0; b < camParams; ++b)
                                    ct[a * camParams + b] = cc[b * camParams + a];
                            sys.Add(Sl.data(), e1.cam, e1.cam, ct);
                        }
                        if (shareFocal) {
                            const int F = camParams;   // the focal slot
                            sys.Add(Sl.data(), focalNode, e2.cam, &V[F * nE]);
                            double ff = V[F * nE + F];
                            if (q1 != q2) {
                                double col[7];
                                for (int a = 0; a < camParams; ++a) col[a] = V[a * nE + F];
                                sys.Add(Sl.data(), focalNode, e1.cam, col);
                                ff += V[F * nE + F];
                            }
                            sys.Add(Sl.data(), focalNode, focalNode, &ff);
                        }
                    }
                }
            }
        });
        // The workers' shares added in, in fixed slices of the values so the
        // sum is the same whatever the thread count.
        {
            constexpr size_t kSlice = 1 << 16;
            const size_t nv = S.size();
            ParallelFor((nv + kSlice - 1) / kSlice, [&](size_t si) {
                const size_t b0 = si * kSlice, b1 = std::min(nv, b0 + kSlice);
                for (size_t w = 0; w < workerS.size(); ++w) {
                    const double* Sl = workerS[w].data();
                    for (size_t i = b0; i < b1; ++i) S[i] += Sl[i];
                }
            });
            for (size_t w = 0; w < workerS.size(); ++w)
                for (size_t i = 0; i < gS.size(); ++i) gS[i] += workerG[w][i];
        }

        // --- solve and step ---------------------------------------------------
        std::vector<double> dCam = gS;
        // The reduced camera system is SPD once damped: Cholesky, sparse.
        if (!sys.Factor(S.data())) { damp.FailSingular(); continue; }
        sys.Solve(dCam.data());

        // Back-substitution: dP = C^-1 (g_p - E^T dCam).
        std::vector<double> dPt(size_t(nActivePt) * 3, 0.0);
        for (int p = 0; p < nActivePt; ++p) {
            if (!ptOk[size_t(p)]) continue;
            double rhs[3] = {gPt[size_t(p) * 3], gPt[size_t(p) * 3 + 1],
                             gPt[size_t(p) * 3 + 2]};
            for (int ii : byPoint[size_t(p)]) {
                const EBlock& e = E[size_t(ii)];
                const int nE = shareFocal ? camParams + 1 : camParams;
                for (int b = 0; b < 3; ++b)
                    for (int a = 0; a < nE; ++a) {
                        const int gi = (shareFocal && a == camParams)
                                           ? focalCol : e.cam * camParams + a;
                        rhs[b] -= e.m[a * 3 + b] * dCam[size_t(gi)];
                    }
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
                //
                // One step for every camera when the focal is shared: they
                // start equal and move together, which is the constraint.
                const int fi = shareFocal ? focalCol : base + 6;
                const double f = trial.focal[size_t(c)] + dCam[size_t(fi)];
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
            blocksValid = false;
            cost = trialCost;
            damp.Succeed();
            ++taken;
        } else if (!damp.Fail()) {
            break;   // no step helps; the solve is done
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
    double minFovDeg = 1e9, maxFovDeg = -1e9;
    {
        int nf = 0;
        for (const Camera& c : cloud->cameras) {
            if (!c.solved || c.focal <= 0.0) continue;
            const double fv = 2.0 * std::atan2(0.5 * double(c.width), c.focal) *
                              180.0 / 3.14159265358979;
            meanFovDeg += fv;
            minFovDeg = std::min(minFovDeg, fv);
            maxFovDeg = std::max(maxFovDeg, fv);
            ++nf;
        }
        if (nf > 0) meanFovDeg /= double(nf);
        else { minFovDeg = maxFovDeg = 0.0; }
    }
    char fovBuf[96] = "";

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
                  // The SPREAD as well as the mean. Per-camera focals that
                  // disagree are the solver absorbing geometric error into
                  // intrinsics it should not have: one lens took every frame,
                  // so a spread of several degrees is a symptom, not a
                  // measurement.
                  doFocal ? (std::snprintf(fovBuf, sizeof fovBuf,
                                           ", focal refined to fov %.1f deg "
                                           "(%.1f..%.1f across cameras)",
                                           meanFovDeg, minFovDeg, maxFovDeg),
                             fovBuf)
                          : "",
                  dropped, double(m_maxDistance));
    m_note = buf;
    return true;
}

REGISTER_ALGORITHM(BundleAdjustSfm);

}  // namespace
}  // namespace tglab
