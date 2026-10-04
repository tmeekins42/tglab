// global_position — where the cameras are, given where they are pointing.
//
// The last unknown in a global reconstruction, and by far the hardest. Rotation
// averaging is well conditioned because two views determine their relative
// rotation completely. Two views determine their relative translation only up
// to SCALE -- the epipolar constraint cannot distinguish a short baseline with
// near content from a long one with far content -- so every edge contributes a
// direction and withholds a magnitude, and the consistent set of magnitudes has
// to be recovered from the graph as a whole.
//
// TWO METHODS, and the difference between them is the point of this stage.
//
// TRANSLATION AVERAGING (1DSfM, LUD, and the classic global pipelines) uses
// only the edges: find camera positions whose pairwise directions best match
// the measured ones. Compact and fast, and it has a structural weakness that
// shows up constantly in practice -- it is ILL-POSED FOR COLLINEAR CAMERAS.
// When three cameras lie on a line, the direction from the first to the second
// and from the first to the third are identical, so the measurements say
// nothing about the spacing. A camera dollying down a corridor or a drone
// flying a straight transect is exactly this case, and the solution collapses:
// positions slide along the line, or pile up at a point.
//
// GLOBAL POSITIONING (Pan et al., "Global Structure-from-Motion Revisited",
// ECCV 2024 -- the GLOMAP paper) deletes the problem by refusing to separate
// the two halves. It solves for camera positions AND 3D point positions at the
// same time, against the rays from cameras to points rather than the directions
// between cameras. Three collinear cameras looking at a point off the line have
// three distinct rays, and the spacing is determined again.
//
// HOW IT IS SOLVED HERE, in two parts -- and the second is what makes it work.
//
//   1. A START, from uniform random positions: alternating least squares,
//      each point placed where its rays pass closest, then each camera where
//      its rays best reach its points. Cheap and globally convergent enough to
//      find the right basin.
//   2. THE ANSWER: Levenberg-Marquardt on the ANGLE between each ray and the
//      direction to its point -- the paper's normalised direction error, the
//      chord between two unit vectors, bounded -- with points eliminated per
//      track as bundle adjustment does. See RefineAngular.
//
// Part 1 used to be all of it, and it was the reason reconstructions failed.
// Perpendicular distance is not the paper's error: it rewards shrinking the
// scene, and alternating passes converge one link of the chain at a time.
// Measured on castle-P19: mean ray residual 4.97 degrees, 424 of 2803 tracks
// triangulated at a 32 px median -- a walk-around that did not reconstruct,
// which this README had put down to the imagery. With part 2: 0.81 degrees,
// 2166 tracks at 0.9 px, and the focal refined to 58.4 degrees against the
// dataset's calibrated 58.2. fountain-P11, which had always "worked", had been
// biased too -- its focal came back as 50 degrees; it now refines to 57.4.
//
// SCALE IS NOT RECOVERABLE. The whole reconstruction can be scaled freely and
// every measurement stays satisfied, exactly as the whole thing can be rotated
// freely (see rotation_average). Both are gauge freedoms, both are fixed by
// convention here, and both have to be accounted for in any test.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "../../algo_util/least_squares.h"
#include "../../algo_util/linalg.h"
#include "../../algo_util/sparse_cholesky.h"
#include "../../core/algorithm.h"
#include "../../core/parallel.h"

namespace tglab {
namespace {

// One camera-to-point observation, resolved into a world-space ray.
struct Ray {
    int  camera = -1;
    int  track = -1;
    Vec3 dir;       // unit, world space: which way the camera saw the point
    double weight = 1.0;
};

// THE REFINEMENT: Levenberg-Marquardt on the ANGLE each ray makes with the
// direction to its point, cameras and points together.
//
// Why it exists. The alternating solve above minimises each point's
// PERPENDICULAR DISTANCE from its rays. That objective has a trivial
// optimum -- shrink everything toward one spot and every distance goes to
// zero -- and alternating passes creep toward it while moving information
// only one link along the chain per pass. Measured on castle-P19: after the
// default 150 passes even the best-constrained cameras sat at a 1 degree
// median ray residual, and 2000 passes brought cameras 0-3 to 0.16 --
// nowhere near converged, and the answer depended on the pass count. What
// the formulation this stage follows actually bounds is the ANGLE, which is
// scale-free: no collapse to reward.
//
// The residual is the chord r = u - v between the observed unit ray v and
// the unit direction u = (X - c)/|X - c| to the point: |r| = 2 sin(theta/2),
// bounded by 2, smooth, and large for a point BEHIND the camera -- so
// cheirality needs no special case. Huber-weighted, so a mismatched track
// cannot drag a camera.
//
// Solved as bundle adjustment is: each point's 3x3 block eliminated (the
// Schur complement), leaving a dense 3C x 3C system in the camera positions
// -- 300 x 300 for a hundred cameras, trivial. NO camera is held fixed:
// translation and scale are left free and the damping keeps steps along them
// small. Holding the first camera fixed was tried and failed exactly when it
// mattered -- a start that placed that camera badly left it stranded, its
// rays written off as outliers by the robust loss while the rest solved
// around it (frame 0 of a video walk-around, 1.5 radii from frame 1).
double RefineAngular(const std::vector<Ray>& rays, const std::vector<bool>& solved,
                   int iterations, double huber, std::vector<Vec3>* camPos,
                   std::vector<Vec3>* pts,
                   const std::function<void(double)>& onStep = {}) {
    const int nCam = int(camPos->size()), nTrk = int(pts->size());
    std::vector<int> var(size_t(nCam), -1);
    int nv = 0;
    for (int c = 0; c < nCam; ++c) {
        if (!solved[size_t(c)]) continue;
        var[size_t(c)] = nv++;
    }
    if (nv == 0) return 0.0;
    const int n3 = nv * 3;

    // Rays grouped by track, for the per-point elimination.
    std::vector<std::vector<int>> byTrack;
    byTrack.resize(size_t(nTrk));
    for (size_t r = 0; r < rays.size(); ++r)
        if (solved[size_t(rays[r].camera)]) byTrack[size_t(rays[r].track)].push_back(int(r));

    auto cost = [&](const std::vector<Vec3>& cp, const std::vector<Vec3>& pp) {
        double s = 0.0;
        for (const Ray& r : rays) {
            if (!solved[size_t(r.camera)]) continue;
            const Vec3 d = pp[size_t(r.track)] - cp[size_t(r.camera)];
            const double L = d.Norm();
            if (L < 1e-12) continue;
            const double e = (d * (1.0 / L) - r.dir).Norm();
            s += e <= huber ? 0.5 * e * e : huber * (e - 0.5 * huber);
        }
        return s;
    };

    LmDamping damp{1e-3, 5.0, 1.0 / 3.0, 1e-6, 1e8};
    double cur = cost(*camPos, *pts);

    // THE CAMERA SYSTEM IS SPARSE, as bundle adjustment's is: two cameras
    // couple only through a point both saw. Stored and factored densely it
    // was 365 s of a 1577-camera room scan, and 113 s sparse, to the same
    // answer; see sparse_cholesky.h.
    linalg::BlockSparseCholesky sys;
    {
        std::vector<std::pair<int, int>> pairs;
        std::vector<int> cs;
        for (const auto& rs : byTrack) {
            cs.clear();
            for (int ri : rs) cs.push_back(var[size_t(rays[size_t(ri)].camera)]);
            std::sort(cs.begin(), cs.end());
            cs.erase(std::unique(cs.begin(), cs.end()), cs.end());
            for (size_t a = 0; a < cs.size(); ++a)
                for (size_t b = a + 1; b < cs.size(); ++b) pairs.push_back({cs[b], cs[a]});
        }
        if (!sys.Analyse(std::vector<int>(static_cast<size_t>(nv), 3), pairs)) return cur;
    }
    auto diagAt = [&](int a) {
        const int c = a / 3, k = a % 3;
        return size_t(sys.Offset(c, c)) + size_t(k * 3 + k);
    };
    std::vector<double> S, rhs;
    S.resize(sys.NumValues());
    rhs.resize(size_t(n3));

    // Per point: its block, its gradient, and each ray's camera coupling --
    // and the camera diagonal blocks and gradient before damping and
    // elimination. Kept across a REJECTED step, which leaves the state where
    // it was: the damping enters only after them.
    struct PointSys { double H[9]; Vec3 g; };
    std::vector<PointSys> psys(static_cast<size_t>(nTrk));
    std::vector<double> rayA(rays.size() * 9, 0.0);   // w A A per ray
    std::vector<Vec3>   rayG(rays.size());            // w A r per ray
    std::vector<double> Sbase(sys.NumValues()), rhsBase(static_cast<size_t>(n3));
    bool baseValid = false;

    // The elimination's workers, each with its own S and rhs: every point
    // updates the camera blocks of every pair of cameras that saw it, so
    // threads cannot share one. Allocated once.
    std::vector<std::vector<double>> workerS(ParallelThreads(size_t(nTrk)));
    std::vector<std::vector<double>> workerR(workerS.size());
    for (size_t w = 0; w < workerS.size(); ++w) {
        workerS[w].assign(sys.NumValues(), 0.0);
        workerR[w].assign(size_t(n3), 0.0);
    }
    std::vector<double> Hinv(size_t(nTrk) * 9, 0.0);
    for (int it = 0; it < iterations; ++it) {
        if (onStep) onStep(double(it) / double(std::max(1, iterations)));
        if (!baseValid) {
        // Each track's own block and couplings, in parallel: every write here
        // is to this track's slots.
        ParallelFor(size_t(nTrk), [&](size_t ti) {
            const int t = int(ti);
            PointSys& P = psys[size_t(t)];
            std::fill(P.H, P.H + 9, 0.0);
            P.g = Vec3{0, 0, 0};
            for (int ri : byTrack[size_t(t)]) {
                const Ray& r = rays[size_t(ri)];
                double* AA = &rayA[size_t(ri) * 9];
                std::fill(AA, AA + 9, 0.0);
                rayG[size_t(ri)] = Vec3{0, 0, 0};
                const Vec3 d = (*pts)[size_t(t)] - (*camPos)[size_t(r.camera)];
                const double L = d.Norm();
                if (L < 1e-12) continue;
                const Vec3 u = d * (1.0 / L);
                const Vec3 res = u - r.dir;
                const double e = res.Norm();
                const double w = RobustWeight(RobustLoss::Huber, e, huber);
                // du/dX = A = (I - u u^T) / L, symmetric; du/dc = -A.
                double A[9];
                const double uu[3] = {u.x, u.y, u.z};
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        A[a * 3 + b] = ((a == b ? 1.0 : 0.0) - uu[a] * uu[b]) / L;
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b) {
                        double v = 0.0;
                        for (int k = 0; k < 3; ++k) v += A[a * 3 + k] * A[k * 3 + b];
                        AA[a * 3 + b] = w * v;
                        P.H[a * 3 + b] += w * v;
                    }
                const double rr[3] = {res.x, res.y, res.z};
                double Ar[3];
                for (int a = 0; a < 3; ++a)
                    Ar[a] = w * (A[a * 3 + 0] * rr[0] + A[a * 3 + 1] * rr[1] + A[a * 3 + 2] * rr[2]);
                rayG[size_t(ri)] = Vec3{Ar[0], Ar[1], Ar[2]};
                P.g = P.g + Vec3{Ar[0], Ar[1], Ar[2]};
            }
        });
        // Camera diagonal blocks and gradient (d/dc = -A), serially and in
        // the same order as ever. A skipped ray contributes zeros.
        std::fill(Sbase.begin(), Sbase.end(), 0.0);
        std::fill(rhsBase.begin(), rhsBase.end(), 0.0);
        for (int t = 0; t < nTrk; ++t)
            for (int ri : byTrack[size_t(t)]) {
                const int vc = var[size_t(rays[size_t(ri)].camera)];
                if (vc < 0) continue;
                const double* AA = &rayA[size_t(ri) * 9];
                sys.Add(Sbase.data(), vc, vc, AA);
                const Vec3& Ar = rayG[size_t(ri)];
                rhsBase[size_t(vc * 3 + 0)] += Ar.x;   // -g_c = +A r
                rhsBase[size_t(vc * 3 + 1)] += Ar.y;
                rhsBase[size_t(vc * 3 + 2)] += Ar.z;
            }
        baseValid = true;
        }
        S = Sbase;
        rhs = rhsBase;

        // Damped point blocks, inverted; Schur-eliminated into S and rhs.
        auto inv3 = [](const double M[9], double* out) {
            const double det = M[0] * (M[4] * M[8] - M[5] * M[7]) -
                               M[1] * (M[3] * M[8] - M[5] * M[6]) +
                               M[2] * (M[3] * M[7] - M[4] * M[6]);
            if (std::fabs(det) < 1e-300) return false;
            out[0] = (M[4] * M[8] - M[5] * M[7]) / det; out[1] = (M[2] * M[7] - M[1] * M[8]) / det;
            out[2] = (M[1] * M[5] - M[2] * M[4]) / det; out[3] = (M[5] * M[6] - M[3] * M[8]) / det;
            out[4] = (M[0] * M[8] - M[2] * M[6]) / det; out[5] = (M[2] * M[3] - M[0] * M[5]) / det;
            out[6] = (M[3] * M[7] - M[4] * M[6]) / det; out[7] = (M[1] * M[6] - M[0] * M[7]) / det;
            out[8] = (M[0] * M[4] - M[1] * M[3]) / det;
            return true;
        };
        for (int a = 0; a < n3; ++a) S[diagAt(a)] *= (1.0 + damp.lambda);
        for (int a = 0; a < n3; ++a) S[diagAt(a)] += 1e-12;
        // Tracks dealt round-robin to the workers; see workerS.
        ParallelFor(workerS.size(), [&](size_t wk) {
        std::vector<double>& Sw = workerS[wk];   // this worker's share of S and rhs,
        std::vector<double>& rw = workerR[wk];   // added in below
        std::fill(Sw.begin(), Sw.end(), 0.0);
        std::fill(rw.begin(), rw.end(), 0.0);
        std::vector<std::pair<int, int>> sorted;   // (camera var, ray), per track
        std::vector<int> vs;
        std::vector<double> AH;
        std::vector<long long> offs;
        for (int t = int(wk); t < nTrk; t += int(workerS.size())) {
            PointSys& P = psys[size_t(t)];
            double Hd[9];
            std::copy(P.H, P.H + 9, Hd);
            for (int a = 0; a < 3; ++a) Hd[a * 4] = Hd[a * 4] * (1.0 + damp.lambda) + 1e-12;
            double* Hi = &Hinv[size_t(t) * 9];
            if (!inv3(Hd, Hi)) { std::fill(Hi, Hi + 9, 0.0); continue; }
            // H_ct = -AA per ray (d/dc = -A, d/dX = A). With -g_X = -P.g:
            // S -= H_ct Hi H_tc = AA_i Hi AA_j; rhs -= H_ct Hi (-g_X) = AA_i Hi (-P.g)... signs below.
            const std::vector<int>& rs = byTrack[size_t(t)];
            // Hi * (-g_X)
            const double gx[3] = {-P.g.x, -P.g.y, -P.g.z};
            double hg[3];
            for (int a = 0; a < 3; ++a)
                hg[a] = Hi[a * 3 + 0] * gx[0] + Hi[a * 3 + 1] * gx[1] + Hi[a * 3 + 2] * gx[2];
            // This track's rays by camera, ascending: the order the sparse
            // system's columns hold their rows in.
            sorted.clear();
            for (int ri : rs) {
                const int vi = var[size_t(rays[size_t(ri)].camera)];
                if (vi >= 0) sorted.push_back({vi, ri});
            }
            std::sort(sorted.begin(), sorted.end());
            const size_t L = sorted.size();
            AH.resize(L * 9);
            vs.resize(L);
            for (size_t q = 0; q < L; ++q) {
                const int vi = sorted[q].first;
                vs[q] = vi;
                const double* Ai = &rayA[size_t(sorted[q].second) * 9];   // H_{c_i t} = -Ai
                // rhs_i -= H_ct Hi (-g_X) = -(-Ai) hg = +Ai hg
                for (int a = 0; a < 3; ++a)
                    rw[size_t(vi * 3 + a)] +=
                        Ai[a * 3 + 0] * hg[0] + Ai[a * 3 + 1] * hg[1] + Ai[a * 3 + 2] * hg[2];
                // (Ai Hi) once, then times each Aj: S_ij -= (-Ai) Hi (-Aj) = Ai Hi Aj
                double* AiHi = &AH[q * 9];
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b) {
                        double v = 0.0;
                        for (int k = 0; k < 3; ++k) v += Ai[a * 3 + k] * Hi[k * 3 + b];
                        AiHi[a * 3 + b] = v;
                    }
            }
            // Only the blocks the sparse system stores, its lower half:
            // column vj takes rows vi >= vj, all found in one merge walk.
            // Two rays from the same camera give a diagonal block, which is
            // stored in full, so it takes both orders of the pair.
            offs.resize(L);
            for (size_t q2 = 0; q2 < L; ++q2) {
                const int vj = vs[q2];
                sys.Offsets(vj, &vs[q2], int(L - q2), offs.data());
                const double* Aj = &rayA[size_t(sorted[q2].second) * 9];
                for (size_t q1 = q2; q1 < L; ++q1) {
                    const long long off = offs[q1 - q2];
                    if (off < 0) continue;
                    const double* AiHi = &AH[q1 * 9];
                    double M[9];
                    for (int a = 0; a < 3; ++a)
                        for (int b = 0; b < 3; ++b) {
                            double v = 0.0;
                            for (int k = 0; k < 3; ++k) v += AiHi[a * 3 + k] * Aj[k * 3 + b];
                            M[a * 3 + b] = v;
                        }
                    double* dst = Sw.data() + off;
                    for (int e = 0; e < 9; ++e) dst[e] -= M[e];
                    if (vs[q1] == vj && q1 != q2)
                        for (int a = 0; a < 3; ++a)
                            for (int b = 0; b < 3; ++b) dst[a * 3 + b] -= M[b * 3 + a];
                }
            }
        }
        });
        {
            constexpr size_t kSlice = 1 << 16;
            const size_t nvals = S.size();
            ParallelFor((nvals + kSlice - 1) / kSlice, [&](size_t si) {
                const size_t b0 = si * kSlice, b1 = std::min(nvals, b0 + kSlice);
                for (size_t wk = 0; wk < workerS.size(); ++wk) {
                    const double* Sw = workerS[wk].data();
                    for (size_t i = b0; i < b1; ++i) S[i] += Sw[i];
                }
            });
            for (size_t wk = 0; wk < workerS.size(); ++wk)
                for (size_t i = 0; i < rhs.size(); ++i) rhs[i] += workerR[wk][i];
        }

        // The reduced camera system is SPD once damped: Cholesky, sparse.
        std::vector<double> dc = rhs;
        if (!sys.Factor(S.data())) { damp.FailSingular(); continue; }
        sys.Solve(dc.data());

        // Candidate: cameras moved, then each point from its own block:
        // dX = Hi (-g_X - H_tc dc) = Hi (-g_X + sum Ai dc_i).
        std::vector<Vec3> nc = *camPos, np = *pts;
        for (int c = 0; c < nCam; ++c) {
            const int v = var[size_t(c)];
            if (v >= 0) nc[size_t(c)] = nc[size_t(c)] + Vec3{dc[size_t(v * 3)], dc[size_t(v * 3 + 1)],
                                                            dc[size_t(v * 3 + 2)]};
        }
        for (int t = 0; t < nTrk; ++t) {
            const double* Hi = &Hinv[size_t(t) * 9];
            double q[3] = {-psys[size_t(t)].g.x, -psys[size_t(t)].g.y, -psys[size_t(t)].g.z};
            for (int ri : byTrack[size_t(t)]) {
                const int v = var[size_t(rays[size_t(ri)].camera)];
                if (v < 0) continue;
                const double* Ai = &rayA[size_t(ri) * 9];
                for (int a = 0; a < 3; ++a)
                    q[a] += Ai[a * 3 + 0] * dc[size_t(v * 3)] + Ai[a * 3 + 1] * dc[size_t(v * 3 + 1)] +
                            Ai[a * 3 + 2] * dc[size_t(v * 3 + 2)];
            }
            np[size_t(t)] = np[size_t(t)] + Vec3{Hi[0] * q[0] + Hi[1] * q[1] + Hi[2] * q[2],
                                                Hi[3] * q[0] + Hi[4] * q[1] + Hi[5] * q[2],
                                                Hi[6] * q[0] + Hi[7] * q[1] + Hi[8] * q[2]};
        }
        const double next = cost(nc, np);
        if (next < cur) {
            *camPos = std::move(nc);
            *pts = std::move(np);
            baseValid = false;
            const bool tiny = cur - next < 1e-9 * cur;
            cur = next;
            damp.Succeed();
            if (tiny) break;
        } else if (!damp.Fail()) {
            break;
        }
    }
    return cur;
}

class GlobalPosition : public AlgorithmBase {
public:
    const char* Name()     const override { return "global_position"; }
    const char* Category() const override { return "sfm"; }

    PortList Inputs() const override {
        return {{"src", DataType::PointCloud, FormatSpec::Any, ShapeSpec::Any}};
    }
    PortList Outputs() const override {
        return {{"out", DataType::PointCloud, FormatSpec::Any, ShapeSpec::Any}};
    }

    void RunCPU(RunCtx&) override {}
    bool IsReconstruct() const override { return true; }

    bool RunReconstruct(const std::vector<Image>* images, PointCloud* cloud,
                        std::string* err) override;

    std::string RunReport() const override { return m_note; }
    ProxyBehaviour Proxy() const override { return ProxyBehaviour::Never; }
    bool HasGPU() const override { return false; }

private:
    bool SolveJoint(PointCloud* cloud, std::string* err);
    bool SolveAveraging(PointCloud* cloud, std::string* err);

    static constexpr const char* kMethodNames[] = {
        "joint camera+point (GLOMAP)", "translation averaging"};

    Param<int> m_method{this, "method", 0, 0, 1,
        {.help = "Joint solves for camera and point positions together against "
                 "camera-to-point rays, which stays determined when the "
                 "cameras are collinear -- a dolly or a straight transect. "
                 "Translation averaging uses only the directions between "
                 "cameras: compact and fast, and structurally ill-posed in "
                 "exactly that case, because collinear cameras give identical "
                 "directions whatever the spacing.",
         .choices = kMethodNames, .choiceCount = 2}};

    Param<int> m_iterations{this, "iterations", 150, 10, 2000,
        {.help = "Optimisation passes. The joint method alternates between "
                 "updating points and updating cameras, so each pass is one of "
                 "each; convergence is fast at first and then slow, which is "
                 "normal for an alternating scheme."}};

    Param<int> m_refine{this, "refine", 100, 0, 1000,
        {.help = "Levenberg-Marquardt steps on the angle between each ray and "
                 "its point, after the alternating passes. The passes only "
                 "reach a starting point: they minimise distance, which rewards "
                 "shrinking the scene, and converge one link of the chain at a "
                 "time. 0 skips it, for comparison."}};

    Param<int> m_seed{this, "seed", 1, 0, 100000,
        {.help = "Random seed for the initial positions. The joint method "
                 "starts from uniform random -- the GLOMAP formulation is "
                 "bounded so it does not need a good starting point -- and "
                 "exposing the seed makes that claim testable rather than "
                 "merely asserted."}};

    std::string m_note;
    std::string m_start;   // which starting point the refinement kept
};

// --- the joint method --------------------------------------------------------
//
// Alternates two closed-form steps, which is what makes it tractable without a
// general solver:
//
//   1. Given camera positions, each point moves to the least-squares
//      intersection of the rays that see it.
//   2. Given point positions, each camera moves to the least-squares point
//      consistent with its own rays.
//
// Both are the same computation -- the point minimising squared perpendicular
// distance to a bundle of lines -- and both have an exact solution. A full
// Levenberg-Marquardt over everything at once (what GLOMAP actually does, via
// Ceres) converges in fewer iterations and needs the machinery of Phase 4;
// alternation gets to the same place with arithmetic that can be read.
//
// The perpendicular-distance formulation is what carries the [0,1] bound from
// the paper: a ray's contribution is the component of the offset ACROSS the
// ray, which saturates rather than growing without limit as an observation gets
// worse.
bool GlobalPosition::SolveJoint(PointCloud* cloud, std::string* err) {
    const int nCam = int(cloud->cameras.size());
    const int nTrk = int(cloud->tracks.size());
    m_start.clear();
    if (nTrk == 0) {
        *err = "global_position: no tracks -- run build_tracks first";
        return false;
    }

    // Rays need a solved rotation: the direction a camera saw a point is a
    // property of its orientation, which rotation averaging supplies.
    int solvedRot = 0;
    for (const Camera& c : cloud->cameras) if (c.solved) ++solvedRot;
    if (solvedRot < 2) {
        *err = "global_position: fewer than two cameras have orientations -- "
               "run rotation_average first";
        return false;
    }

    // HOW WELL SUPPORTED EACH CAMERA'S GEOMETRY IS, from the view graph.
    //
    // Not every edge is worth the same. Measured on castle-P19: the healthy
    // pairs carry 558-1060 inliers at 59-78%, while the pairs that span the
    // turns of a walk around a courtyard carry 130-286 at 28-48% -- the views
    // barely overlap there. And the three worst-supported links line up
    // exactly with the three largest jumps in camera spacing: pair 11->12 has
    // 130 inliers and the chain separates precisely there, into the two
    // clusters the viewport showed.
    //
    // Unweighted, a camera pinned by 130 weak correspondences speaks as loudly
    // as one pinned by 1060 good ones, so a starved link drags its whole side
    // of the graph with it. Weighting by support lets the well-observed
    // cameras hold their positions and makes the starved link yield instead.
    //
    // sqrt rather than the raw count, as in rotation averaging: the count is
    // roughly how much evidence there is, and the STANDARD ERROR of an
    // estimate from n samples falls as sqrt(n). Using n itself would let one
    // rich pair dominate the entire reconstruction.
    //
    // These weights apply to the alternating START only. RefineAngular, which
    // produces the answer, relies on its robust loss instead -- and the "two
    // clusters" measured above were mostly that start failing to converge,
    // not the weak link alone: with the refinement, castle-P19 is one
    // connected reconstruction at a 0.8 degree mean ray residual.
    std::vector<double> camSupport(size_t(nCam), 0.0);
    for (const PointCloud::ViewEdge& e : cloud->edges) {
        if (e.i < 0 || e.i >= nCam || e.j < 0 || e.j >= nCam) continue;
        const double w = std::sqrt(double(std::max(1, e.inliers)));
        camSupport[size_t(e.i)] = std::max(camSupport[size_t(e.i)], w);
        camSupport[size_t(e.j)] = std::max(camSupport[size_t(e.j)], w);
    }
    // Normalised so the weights are relative rather than absolute: the solve
    // below is scale-free in the weights, but keeping them near 1 leaves the
    // conditioning where it was when they were all exactly 1.
    double maxSupport = 0.0;
    for (double w : camSupport) maxSupport = std::max(maxSupport, w);
    if (maxSupport > 0.0)
        for (double& w : camSupport) w = (w > 0.0) ? (w / maxSupport) : 1.0;
    else
        for (double& w : camSupport) w = 1.0;

    std::vector<Ray> rays;
    rays.reserve(size_t(nTrk) * 3);
    for (int t = 0; t < nTrk; ++t) {
        const Track& tr = cloud->tracks[size_t(t)];
        for (const Observation& o : tr.obs) {
            if (o.frame < 0 || o.frame >= nCam) continue;
            const Camera& c = cloud->cameras[size_t(o.frame)];
            if (!c.solved) continue;
            Ray r;
            r.camera = o.frame;
            r.track = t;
            r.dir = c.RayThrough(double(o.x), double(o.y));
            if (r.dir.Norm() < 0.5) continue;   // degenerate
            r.weight = camSupport[size_t(o.frame)];
            rays.push_back(r);
        }
    }

    if (rays.size() < size_t(nCam) * 2) {
        *err = "global_position: too few observations to position the cameras";
        return false;
    }

    // UNIFORM RANDOM INITIALISATION, as the paper specifies. Not an oversight
    // and not laziness: the bounded error means there is no basin of attraction
    // to find, so a good starting point buys nothing and a bad one costs
    // nothing. Being able to start from noise is the formulation's headline
    // property, and starting from something better would hide whether it holds.
    std::mt19937 rng(uint32_t(int(m_seed)) * 2654435761u + 1u);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);

    std::vector<Vec3> camPos, pt;
    camPos.resize(size_t(nCam));
    pt.resize(size_t(nTrk));
    for (Vec3& p : camPos) p = Vec3{uni(rng), uni(rng), uni(rng)};
    for (Vec3& p : pt)     p = Vec3{uni(rng), uni(rng), uni(rng)};

    // The least-squares closest point to a bundle of lines, accumulated as
    // sum((I - d d^T)) x = sum((I - d d^T) o) for rays with origin o and unit
    // direction d. Each (I - d d^T) projects onto the plane perpendicular to
    // the ray, which is exactly "how far off the ray is this point".
    auto solve3 = [](const double A[9], const Vec3& b, Vec3* x) {
        const double det =
            A[0] * (A[4] * A[8] - A[5] * A[7]) -
            A[1] * (A[3] * A[8] - A[5] * A[6]) +
            A[2] * (A[3] * A[7] - A[4] * A[6]);
        // Singular means the rays are parallel or there is only one: the point
        // is undetermined along that direction, and leaving it where it is
        // beats moving it somewhere arbitrary.
        if (std::fabs(det) < 1e-12) return false;
        const double inv[9] = {
            (A[4] * A[8] - A[5] * A[7]) / det, (A[2] * A[7] - A[1] * A[8]) / det,
            (A[1] * A[5] - A[2] * A[4]) / det, (A[5] * A[6] - A[3] * A[8]) / det,
            (A[0] * A[8] - A[2] * A[6]) / det, (A[2] * A[3] - A[0] * A[5]) / det,
            (A[3] * A[7] - A[4] * A[6]) / det, (A[1] * A[6] - A[0] * A[7]) / det,
            (A[0] * A[4] - A[1] * A[3]) / det};
        *x = Vec3{inv[0] * b.x + inv[1] * b.y + inv[2] * b.z,
                  inv[3] * b.x + inv[4] * b.y + inv[5] * b.z,
                  inv[6] * b.x + inv[7] * b.y + inv[8] * b.z};
        return true;
    };

    const int iters = int(m_iterations);
    for (int it = 0; it < iters; ++it) {
        GroupProgress(0.1 * double(it) / double(std::max(1, iters)), "alternating start");
        // --- points, given cameras ---
        std::vector<double> A(size_t(nTrk) * 9, 0.0);
        std::vector<Vec3>   b;
        b.resize(size_t(nTrk));
        for (const Ray& r : rays) {
            const Vec3& o = camPos[size_t(r.camera)];
            const Vec3& d = r.dir;
            const double w = r.weight;
            double* a = &A[size_t(r.track) * 9];
            a[0] += w * (1.0 - d.x * d.x); a[1] += w * (-d.x * d.y);
            a[2] += w * (-d.x * d.z);
            a[3] += w * (-d.y * d.x);      a[4] += w * (1.0 - d.y * d.y);
            a[5] += w * (-d.y * d.z);
            a[6] += w * (-d.z * d.x);      a[7] += w * (-d.z * d.y);
            a[8] += w * (1.0 - d.z * d.z);
            const Vec3 po{o.x - d.x * d.Dot(o), o.y - d.y * d.Dot(o),
                          o.z - d.z * d.Dot(o)};
            b[size_t(r.track)] = b[size_t(r.track)] + po * w;
        }
        for (int t = 0; t < nTrk; ++t) {
            Vec3 x;
            if (solve3(&A[size_t(t) * 9], b[size_t(t)], &x)) pt[size_t(t)] = x;
        }

        // --- CHEIRALITY: a ray is a HALF-line -------------------------------
        //
        // (I - dd^T) measures perpendicular distance to the infinite LINE
        // through the camera, so a point BEHIND the camera fits it perfectly:
        // p = c - 5d is exactly on the line. The least-squares solve above has
        // no reason to prefer the front, and on real data it routinely picks
        // the back.
        //
        // Measured on castle-P19: 3891 of 16237 tracks landed behind a camera,
        // and since such a point is roughly 180 degrees from where its camera
        // saw it, they dragged the mean ray residual to 39 degrees. The
        // structure could not be right, and on screen it was an unreadable
        // haze.
        //
        // Pushed just in front of the nearest camera that sees it rather than
        // discarded: the next iteration re-solves from there, and a point that
        // genuinely has support in front will find it. Discarding would throw
        // away a track that is merely badly initialised.
        for (const Ray& r : rays) {
            const Vec3 rel = pt[size_t(r.track)] - camPos[size_t(r.camera)];
            const double depth = rel.Dot(r.dir);
            if (depth > 1e-6) continue;
            // A nominal distance along the ray. The scale is a gauge freedom
            // (see the header), so any positive value is as good as another;
            // what matters is only that the point moves to the correct side.
            pt[size_t(r.track)] = camPos[size_t(r.camera)] + r.dir * 1.0;
        }

        // --- cameras, given points ---
        //
        // DELIBERATELY UNWEIGHTED, and it is worth saying why rather than
        // leaving the asymmetry to look like an oversight.
        //
        // The weight is a property of the CAMERA, and this solve is separable:
        // each camera gets its own 3x3 system from only its own rays. A factor
        // constant across every term of one system scales A and b alike and
        // cancels exactly in the solution. Applying it here would be a no-op
        // that merely looked symmetric.
        //
        // The weight bites in the POINT solve above, where rays from different
        // cameras compete to place the same point -- which is precisely where
        // a starved camera should lose the argument.
        std::vector<double> CA(size_t(nCam) * 9, 0.0);
        std::vector<Vec3>   cb;
        cb.resize(size_t(nCam));
        for (const Ray& r : rays) {
            const Vec3& p = pt[size_t(r.track)];
            const Vec3& d = r.dir;
            double* a = &CA[size_t(r.camera) * 9];
            a[0] += 1.0 - d.x * d.x; a[1] += -d.x * d.y;      a[2] += -d.x * d.z;
            a[3] += -d.y * d.x;      a[4] += 1.0 - d.y * d.y; a[5] += -d.y * d.z;
            a[6] += -d.z * d.x;      a[7] += -d.z * d.y;      a[8] += 1.0 - d.z * d.z;
            const Vec3 pp{p.x - d.x * d.Dot(p), p.y - d.y * d.Dot(p),
                          p.z - d.z * d.Dot(p)};
            cb[size_t(r.camera)] = cb[size_t(r.camera)] + pp;
        }
        for (int c = 0; c < nCam; ++c) {
            if (!cloud->cameras[size_t(c)].solved) continue;
            Vec3 x;
            if (solve3(&CA[size_t(c) * 9], cb[size_t(c)], &x)) camPos[size_t(c)] = x;
        }
    }

    // --- refinement: the angle, not the distance -------------------------------
    // The alternating passes above are only the starting point; see
    // RefineAngular for why they cannot be the answer.
    if (int(m_refine) > 0) {
        std::vector<bool> isSolved(size_t(nCam), false);
        for (int c = 0; c < nCam; ++c) isSolved[size_t(c)] = cloud->cameras[size_t(c)].solved;

        // A SECOND START, from the pairs' own translation DIRECTIONS chained
        // along the sequence at equal steps -- and the refinement run from
        // both, keeping whichever explains the rays better.
        //
        // Why: the angle cannot see scale, so a stretch of cameras tied to
        // the rest by few tracks can sit at almost any scale for almost no
        // cost -- and the alternating start above, which minimises DISTANCE,
        // hands it a shrunken one. Measured on a 100-frame video: frames 18
        // to 71 crushed into a clump, with a jump of 2.26 radii either side,
        // at a 0.47 degree residual that looked healthy. A chain of equal
        // steps has no such bias: for video, where frames are evenly spaced
        // in time, it is close to the truth, and for photographs it is at
        // least a start with one scale throughout.
        //
        // For a long while it was built from directions relative_pose stored
        // REVERSED, so it was a mirror image -- every camera behind its
        // points -- and lost to the alternating start every time. Measured on
        // a hand-held video of a chrome tape measure, where 107 tracks crossed
        // one quick turn: the alternating start gave each side its own scale,
        // and the reconstruction came back as two copies of the subject. From
        // the corrected chain it is one orbit, with 39 tracks triangulated
        // across the turn instead of none.
        std::vector<Vec3> chainCam = camPos, chainPt = pt;
        bool chainOk = false;
        {
            struct E { int a, b, gap, inl; Vec3 dirW; };
            std::vector<E> es;
            for (const PointCloud::ViewEdge& e : cloud->edges) {
                if (e.i < 0 || e.i >= nCam || e.j < 0 || e.j >= nCam) continue;
                if (!isSolved[size_t(e.i)] || !isSolved[size_t(e.j)]) continue;
                // direction is i-to-j in camera i's frame; into the world:
                es.push_back({e.i, e.j, std::abs(e.j - e.i), e.inliers,
                              cloud->cameras[size_t(e.i)].R.Transpose() * e.direction});
            }
            // Nearest neighbours first, then the best supported: a chain
            // along the sequence, not a leap across it.
            std::sort(es.begin(), es.end(), [](const E& x, const E& y) {
                return x.gap != y.gap ? x.gap < y.gap : x.inl > y.inl;
            });
            std::vector<char> known(size_t(nCam), 0);
            int root = -1;
            for (int c = 0; c < nCam && root < 0; ++c) if (isSolved[size_t(c)]) root = c;
            if (root >= 0) {
                known[size_t(root)] = 1;
                chainCam[size_t(root)] = Vec3{0, 0, 0};
                int reached = 1;
                bool grew = true;
                while (grew) {
                    grew = false;
                    for (const E& e : es) {
                        if (known[size_t(e.a)] && !known[size_t(e.b)]) {
                            chainCam[size_t(e.b)] = chainCam[size_t(e.a)] + e.dirW;
                            known[size_t(e.b)] = 1; ++reached; grew = true;
                        } else if (known[size_t(e.b)] && !known[size_t(e.a)]) {
                            chainCam[size_t(e.a)] = chainCam[size_t(e.b)] - e.dirW;
                            known[size_t(e.a)] = 1; ++reached; grew = true;
                        }
                    }
                }
                int want = 0;
                for (int c = 0; c < nCam; ++c) want += isSolved[size_t(c)] ? 1 : 0;
                chainOk = reached == want;
            }
            if (chainOk) {
                // Points where their rays pass closest, then in front of the
                // cameras that saw them -- the same step the alternating
                // solve takes, once.
                std::vector<double> A(size_t(nTrk) * 9, 0.0);
                std::vector<Vec3> b;
                b.resize(size_t(nTrk));
                for (const Ray& r : rays) {
                    const Vec3& o = chainCam[size_t(r.camera)];
                    const Vec3& d = r.dir;
                    double* a = &A[size_t(r.track) * 9];
                    a[0] += 1.0 - d.x * d.x; a[1] += -d.x * d.y; a[2] += -d.x * d.z;
                    a[3] += -d.y * d.x; a[4] += 1.0 - d.y * d.y; a[5] += -d.y * d.z;
                    a[6] += -d.z * d.x; a[7] += -d.z * d.y; a[8] += 1.0 - d.z * d.z;
                    b[size_t(r.track)] = b[size_t(r.track)] +
                        Vec3{o.x - d.x * d.Dot(o), o.y - d.y * d.Dot(o), o.z - d.z * d.Dot(o)};
                }
                for (int t = 0; t < nTrk; ++t) {
                    Vec3 x;
                    if (solve3(&A[size_t(t) * 9], b[size_t(t)], &x)) chainPt[size_t(t)] = x;
                }
                for (const Ray& r : rays) {
                    const Vec3 rel = chainPt[size_t(r.track)] - chainCam[size_t(r.camera)];
                    if (rel.Dot(r.dir) <= 1e-6)
                        chainPt[size_t(r.track)] = chainCam[size_t(r.camera)] + r.dir * 1.0;
                }
            }
        }

        // A SHORT TRIAL, when there are two starts, and only the leader is
        // refined in full. Both used to get the full budget and the better
        // was kept, but that is twice the work for little: on a 179-frame
        // video the two stood at 551 and 5.9 after ten steps and ended at 97
        // and 0.79, and on a 1577-frame room scan they ended within 1% of
        // each other (64.0 and 63.4) -- the same answer reached twice.
        // Refining only the trial's leader cut positioning on the room from
        // 132 s to 76 s, and bundle adjustment then took 50 s, not 84. The
        // price was the room's final reprojection error at 0.993 px against
        // 0.973, because there the trial's leader was the slightly worse of
        // the two in the end: a fiftieth of a pixel, judged not worth twice
        // the time. The winner is refined from its OWN start, not continued
        // from the trial, so where the trial and the full run agree the
        // answer is exactly what it was.
        // This stage's progress across a refinement: [lo, hi] of the bar.
        auto span = [this](double lo, double hi, const char* what) {
            return std::function<void(double)>(
                [this, lo, hi, what](double f) { GroupProgress(lo + (hi - lo) * f, what); });
        };
        int skip = 0;   // 1: the alternating start lost the trial, 2: the chained
        double ta = 0.0, tc = 0.0;
        const int trial = std::min(10, int(m_refine));
        if (chainOk && trial < int(m_refine)) {
            std::vector<Vec3> aCam = camPos, aPt = pt, cCam = chainCam, cPt = chainPt;
            ta = RefineAngular(rays, isSolved, trial, 0.035, &aCam, &aPt, span(0.10, 0.15, "trial"));
            tc = RefineAngular(rays, isSolved, trial, 0.035, &cCam, &cPt, span(0.15, 0.20, "trial"));
            skip = tc < ta ? 1 : 2;
        }
        if (skip == 1) {
            RefineAngular(rays, isSolved, int(m_refine), 0.035, &chainCam, &chainPt,
                          span(0.20, 1.0, "refine"));
            camPos.swap(chainCam);
            pt.swap(chainPt);
            m_start = "chained directions";
        }
        const double costAlt =
            skip == 1 ? 0.0
                      : RefineAngular(rays, isSolved, int(m_refine), 0.035, &camPos, &pt,
                                      span(0.20, skip == 0 && chainOk ? 0.6 : 1.0, "refine"));
        if (skip != 1) m_start = "alternating";
        double costChain = 0.0;
        if (chainOk && skip == 0) {
            costChain = RefineAngular(rays, isSolved, int(m_refine), 0.035, &chainCam, &chainPt,
                                      span(0.6, 1.0, "refine"));
            if (costChain < costAlt) {
                camPos.swap(chainCam);
                pt.swap(chainPt);
                m_start = "chained directions";
            }
        }
        // What decided it, for the report.
        if (chainOk) {
            char buf[160];
            if (skip != 0)
                std::snprintf(buf, sizeof(buf), " (after %d trial steps, alternating %.4g against chained %.4g)",
                              trial, ta, tc);
            else
                std::snprintf(buf, sizeof(buf), " (trial %.4g against %.4g; refined in full, %.4g against %.4g)",
                              ta, tc, costAlt, costChain);
            m_start += buf;
        }
    }

    // --- gauge: centre at the origin, unit mean camera distance -------------
    //
    // Scale and position are free (see the header), so a convention is needed
    // or successive runs produce differently sized copies of the same answer.
    Vec3 centre;
    int solved = 0;
    for (int c = 0; c < nCam; ++c)
        if (cloud->cameras[size_t(c)].solved) { centre = centre + camPos[size_t(c)]; ++solved; }
    if (solved > 0) centre = centre * (1.0 / double(solved));

    double spread = 0.0;
    for (int c = 0; c < nCam; ++c)
        if (cloud->cameras[size_t(c)].solved)
            spread += (camPos[size_t(c)] - centre).Norm();
    spread = (solved > 0 && spread > 1e-12) ? (double(solved) / spread) : 1.0;

    for (int c = 0; c < nCam; ++c) {
        Camera& cam = cloud->cameras[size_t(c)];
        if (!cam.solved) continue;
        const Vec3 pos = (camPos[size_t(c)] - centre) * spread;
        // Stored as world-to-camera: t = -R * centre. See geometry.h.
        cam.t = cam.R * pos * -1.0;
    }
    for (int t = 0; t < nTrk; ++t) {
        Track& tr = cloud->tracks[size_t(t)];
        tr.point = (pt[size_t(t)] - centre) * spread;
        tr.hasPoint = true;
    }

    // Residual: mean angle between each ray and the direction from its camera
    // to the point it landed on. The number this stage is actually minimising.
    double resid = 0.0;
    int counted = 0;
    for (const Ray& r : rays) {
        const Vec3 to = cloud->tracks[size_t(r.track)].point -
                        cloud->cameras[size_t(r.camera)].Center();
        const double len = to.Norm();
        if (len < 1e-9) continue;
        double c = r.dir.Dot(to * (1.0 / len));
        c = c < -1.0 ? -1.0 : (c > 1.0 ? 1.0 : c);
        resid += std::acos(c);
        ++counted;
    }
    resid = counted ? resid / double(counted) : 0.0;

    char buf[256];
    std::snprintf(buf, sizeof buf,
                  "position: joint, %d cameras and %d points from %d rays; "
                  "mean ray residual %.3f deg",
                  solved, nTrk, int(rays.size()),
                  resid * 180.0 / 3.14159265358979);
    m_note = buf;
    if (!m_start.empty()) m_note += "; refined from the " + m_start + " start";
    return true;
}

// --- classic translation averaging ------------------------------------------
//
// Positions from edge directions alone: move each camera so the direction to
// its neighbour matches what two-view geometry measured.
//
// Here as the COMPARISON ARM rather than as a recommendation. It is what the
// pre-2024 global pipelines used, it is genuinely faster (no points in the
// problem), and its collinear degeneracy is the specific failure the joint
// method was designed to remove -- which is only visible with both in front of
// you.
//
// Implemented as iterative projection: repeatedly move each camera toward where
// its edges say it should be, keeping the edge directions fixed. That is the
// simplest scheme that exhibits the right behaviour in both the good case and
// the degenerate one, which is what this arm is for.
bool GlobalPosition::SolveAveraging(PointCloud* cloud, std::string* err) {
    const int nCam = int(cloud->cameras.size());
    if (cloud->edges.empty()) {
        *err = "global_position: no view-graph edges -- run relative_pose first";
        return false;
    }

    // Each edge gives a direction from camera i to camera j, in WORLD space:
    // the two-view direction is in frame i's coordinates, so rotating by
    // R_i^T puts it in the world.
    struct Dir { int i, j; Vec3 d; double w; };
    std::vector<Dir> dirs;
    for (const PointCloud::ViewEdge& e : cloud->edges) {
        if (e.i < 0 || e.i >= nCam || e.j < 0 || e.j >= nCam) continue;
        const Camera& ci = cloud->cameras[size_t(e.i)];
        if (!ci.solved) continue;
        const Vec3 world = (ci.R.Transpose() * e.direction).Normalized();
        if (world.Norm() < 0.5) continue;
        dirs.push_back(Dir{e.i, e.j, world, std::sqrt(double(std::max(1, e.inliers)))});
    }
    if (dirs.empty()) {
        *err = "global_position: no usable edge directions";
        return false;
    }

    std::mt19937 rng(uint32_t(int(m_seed)) * 2654435761u + 7u);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    std::vector<Vec3> pos;
    pos.resize(size_t(nCam));
    for (Vec3& p : pos) p = Vec3{uni(rng), uni(rng), uni(rng)};

    for (int it = 0; it < int(m_iterations); ++it) {
        if (GroupCancelled()) { *err = "cancelled"; return false; }   // see SetGroupCancel
        std::vector<Vec3>   acc;
        acc.resize(size_t(nCam));
        std::vector<double> wsum(size_t(nCam), 0.0);

        for (const Dir& d : dirs) {
            // The current separation, projected onto the measured direction:
            // how far apart the edge says they should be, given where they are.
            const Vec3 sep = pos[size_t(d.j)] - pos[size_t(d.i)];
            double len = sep.Dot(d.d);
            if (len < 1e-3) len = 1e-3;   // never let a pair collapse
            acc[size_t(d.j)] = acc[size_t(d.j)] + (pos[size_t(d.i)] + d.d * len) * d.w;
            acc[size_t(d.i)] = acc[size_t(d.i)] + (pos[size_t(d.j)] - d.d * len) * d.w;
            wsum[size_t(d.i)] += d.w;
            wsum[size_t(d.j)] += d.w;
        }
        for (int c = 0; c < nCam; ++c)
            if (wsum[size_t(c)] > 1e-12)
                pos[size_t(c)] = acc[size_t(c)] * (1.0 / wsum[size_t(c)]);
    }

    Vec3 centre;
    int solved = 0;
    for (int c = 0; c < nCam; ++c)
        if (cloud->cameras[size_t(c)].solved) { centre = centre + pos[size_t(c)]; ++solved; }
    if (solved > 0) centre = centre * (1.0 / double(solved));

    double spread = 0.0;
    for (int c = 0; c < nCam; ++c)
        if (cloud->cameras[size_t(c)].solved) spread += (pos[size_t(c)] - centre).Norm();
    spread = (solved > 0 && spread > 1e-12) ? (double(solved) / spread) : 1.0;

    for (int c = 0; c < nCam; ++c) {
        Camera& cam = cloud->cameras[size_t(c)];
        if (!cam.solved) continue;
        cam.t = cam.R * ((pos[size_t(c)] - centre) * spread) * -1.0;
    }

    double resid = 0.0;
    for (const Dir& d : dirs) {
        const Vec3 sep = (cloud->cameras[size_t(d.j)].Center() -
                          cloud->cameras[size_t(d.i)].Center());
        const double len = sep.Norm();
        if (len < 1e-9) { resid += 3.14159265358979 * 0.5; continue; }
        double c = d.d.Dot(sep * (1.0 / len));
        c = c < -1.0 ? -1.0 : (c > 1.0 ? 1.0 : c);
        resid += std::acos(c);
    }
    resid /= double(dirs.size());

    char buf[256];
    std::snprintf(buf, sizeof buf,
                  "position: translation averaging, %d cameras from %d edges; "
                  "mean direction residual %.3f deg",
                  solved, int(dirs.size()), resid * 180.0 / 3.14159265358979);
    m_note = buf;
    return true;
}

bool GlobalPosition::RunReconstruct(const std::vector<Image>*, PointCloud* cloud,
                                    std::string* err) {
    if (cloud->cameras.size() < 2) {
        *err = "global_position needs a reconstruction with at least two cameras";
        return false;
    }
    return int(m_method) == 0 ? SolveJoint(cloud, err) : SolveAveraging(cloud, err);
}

REGISTER_ALGORITHM(GlobalPosition);

}  // namespace
}  // namespace tglab
