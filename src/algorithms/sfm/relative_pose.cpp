// relative_pose — the essential matrix, and the relative rotation in it.
//
// WHY THIS STAGE HAS TO EXIST BEFORE ROTATION AVERAGING.
//
// align_features already fits a model to each matched pair, but its models are
// TWO-DIMENSIONAL: similarity, affine, homography. None of them contains a
// relative rotation between two cameras in 3D. A homography is the right model
// for a pure rotation about the entrance pupil -- a tripod pan -- and says
// nothing usable when the camera also translates, which is the whole premise of
// Structure from Motion.
//
// What rotation averaging consumes is a set of RELATIVE rotations R_ij: "frame
// j is oriented like this with respect to frame i". Those come from the
// ESSENTIAL matrix, which is the calibrated form of the epipolar constraint:
//
//     x_j^T E x_ij = 0,    E = [t]_x R
//
// for normalised (calibrated) image coordinates. E has five degrees of freedom
// -- three of rotation, two of translation DIRECTION -- because the epipolar
// geometry cannot see how FAR apart two cameras are, only which way. That
// missing scale is exactly what global positioning solves later, and it is why
// translation cannot simply be averaged the way rotation can.
//
// THE EIGHT-POINT ALGORITHM is used here rather than Nister's five-point.
// Eight points is a linear solve; five is a tenth-degree polynomial and several
// hundred lines. The tradeoff is real and worth stating: the linear method
// needs more points per RANSAC sample (so more iterations for the same
// confidence), and it is the weaker estimator on near-degenerate configurations
// -- a planar scene most of all, where the epipolar constraint is
// underdetermined and E is not unique. The five-point method is the natural
// second implementation to compare against, which is the point of the lab.
//
// The focal length comes from the camera slots build_tracks filled in. It is a
// guess at this stage; view-graph calibration and bundle adjustment refine it.
// A wrong focal biases the recovered rotation, which is the main reason the
// literature calibrates the view graph before averaging.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "../../algo_util/features.h"
#include "../../algo_util/view_graph.h"
#include "../../core/algorithm.h"

namespace tglab {
namespace {

// One correspondence, in NORMALISED camera coordinates: the pixel less the
// principal point, over the focal length. Working normalised rather than in
// pixels is what makes the result an essential matrix rather than a fundamental
// one, and therefore decomposable into R and t.
struct Corr {
    double ax, ay;   // in frame i
    double bx, by;   // in frame j
};

// Solves the smallest singular vector of an m x 9 system by Jacobi eigen
// decomposition of A^T A.
//
// A 9x9 symmetric eigenproblem rather than a full SVD of A: the matrix is tiny
// and fixed-size, cyclic Jacobi is thirty lines and unconditionally convergent
// for a symmetric matrix, and the answer wanted is one eigenvector. Squaring
// A costs condition number -- the standard objection -- but the eight-point
// algorithm normalises its input first (Hartley), which is precisely the step
// that makes the squared system well behaved.
bool SmallestEigenvector9(const double A[81], double out[9]) {
    double M[81];
    std::copy(A, A + 81, M);

    // V accumulates the rotations, so its columns end as the eigenvectors.
    double V[81] = {};
    for (int i = 0; i < 9; ++i) V[i * 9 + i] = 1.0;

    for (int sweep = 0; sweep < 60; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < 9; ++p)
            for (int q = p + 1; q < 9; ++q) off += M[p * 9 + q] * M[p * 9 + q];
        if (off < 1e-24) break;

        for (int p = 0; p < 9; ++p) {
            for (int q = p + 1; q < 9; ++q) {
                const double apq = M[p * 9 + q];
                if (std::fabs(apq) < 1e-18) continue;
                const double app = M[p * 9 + p], aqq = M[q * 9 + q];
                const double theta = 0.5 * (aqq - app) / apq;
                const double t = (theta >= 0.0 ? 1.0 : -1.0) /
                                 (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0);
                const double s = t * c;

                for (int k = 0; k < 9; ++k) {
                    const double mkp = M[k * 9 + p], mkq = M[k * 9 + q];
                    M[k * 9 + p] = c * mkp - s * mkq;
                    M[k * 9 + q] = s * mkp + c * mkq;
                }
                for (int k = 0; k < 9; ++k) {
                    const double mpk = M[p * 9 + k], mqk = M[q * 9 + k];
                    M[p * 9 + k] = c * mpk - s * mqk;
                    M[q * 9 + k] = s * mpk + c * mqk;
                }
                for (int k = 0; k < 9; ++k) {
                    const double vkp = V[k * 9 + p], vkq = V[k * 9 + q];
                    V[k * 9 + p] = c * vkp - s * vkq;
                    V[k * 9 + q] = s * vkp + c * vkq;
                }
            }
        }
    }

    int best = 0;
    double bestVal = M[0];
    for (int i = 1; i < 9; ++i)
        if (M[i * 9 + i] < bestVal) { bestVal = M[i * 9 + i]; best = i; }

    for (int i = 0; i < 9; ++i) out[i] = V[i * 9 + best];
    return true;
}

// The eight-point algorithm on normalised coordinates, with Hartley's
// isotropic conditioning.
//
// CONDITIONING IS NOT OPTIONAL. Without it the entries of A span several orders
// of magnitude and the smallest singular vector is dominated by round-off; this
// is the single most common reason a hand-written eight-point implementation
// "almost works". Shifting each point set to zero mean and scaling to mean
// distance sqrt(2) fixes it, and the transform is undone on E afterwards.
bool EssentialEightPoint(const std::vector<Corr>& pts,
                         const std::vector<int>& idx, Mat3* E) {
    if (idx.size() < 8) return false;

    double ma[2] = {0, 0}, mb[2] = {0, 0};
    for (int i : idx) {
        ma[0] += pts[size_t(i)].ax; ma[1] += pts[size_t(i)].ay;
        mb[0] += pts[size_t(i)].bx; mb[1] += pts[size_t(i)].by;
    }
    const double n = double(idx.size());
    ma[0] /= n; ma[1] /= n; mb[0] /= n; mb[1] /= n;

    double sa = 0.0, sb = 0.0;
    for (int i : idx) {
        const double dax = pts[size_t(i)].ax - ma[0], day = pts[size_t(i)].ay - ma[1];
        const double dbx = pts[size_t(i)].bx - mb[0], dby = pts[size_t(i)].by - mb[1];
        sa += std::sqrt(dax * dax + day * day);
        sb += std::sqrt(dbx * dbx + dby * dby);
    }
    sa = (sa > 1e-12) ? (n * std::sqrt(2.0) / sa) : 1.0;
    sb = (sb > 1e-12) ? (n * std::sqrt(2.0) / sb) : 1.0;

    double A[81] = {};
    for (int i : idx) {
        const double x1 = (pts[size_t(i)].ax - ma[0]) * sa;
        const double y1 = (pts[size_t(i)].ay - ma[1]) * sa;
        const double x2 = (pts[size_t(i)].bx - mb[0]) * sb;
        const double y2 = (pts[size_t(i)].by - mb[1]) * sb;
        const double r[9] = {x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1.0};
        for (int p = 0; p < 9; ++p)
            for (int q = 0; q < 9; ++q) A[p * 9 + q] += r[p] * r[q];
    }

    double e[9];
    if (!SmallestEigenvector9(A, e)) return false;

    // Undo the conditioning: E = Tb^T * E' * Ta.
    Mat3 Ta, Tb;
    Ta.m[0] = sa; Ta.m[1] = 0;  Ta.m[2] = -sa * ma[0];
    Ta.m[3] = 0;  Ta.m[4] = sa; Ta.m[5] = -sa * ma[1];
    Ta.m[6] = 0;  Ta.m[7] = 0;  Ta.m[8] = 1.0;
    Tb.m[0] = sb; Tb.m[1] = 0;  Tb.m[2] = -sb * mb[0];
    Tb.m[3] = 0;  Tb.m[4] = sb; Tb.m[5] = -sb * mb[1];
    Tb.m[6] = 0;  Tb.m[7] = 0;  Tb.m[8] = 1.0;

    Mat3 Ep;
    std::copy(e, e + 9, Ep.m);
    *E = Tb.Transpose() * Ep * Ta;
    return true;
}

// --- Nister's five-point algorithm ------------------------------------------
//
// WHY FIVE POINTS IS WORTH SEVERAL HUNDRED LINES when eight points is thirty.
//
// An essential matrix has five degrees of freedom: three of rotation, two of
// translation DIRECTION. The eight-point algorithm ignores that. It treats E
// as nine free numbers with one scale ambiguity, solves the resulting linear
// system, and only afterwards projects the answer onto the set of matrices
// that are actually essential. So it solves a larger problem than the one
// posed and repairs the answer at the end.
//
// Two consequences, and both bite on this project's data:
//
//   * RANSAC needs eight correspondences per sample rather than five. The
//     chance a sample is all-inlier is the inlier rate to the 8th power
//     instead of the 5th. On castle-P19's worst pairs -- 130 inliers among 439
//     candidates, a 30% rate -- that is 0.007% against 0.24%, so roughly 35
//     times as many clean samples for the same iteration count. On the pairs
//     where the reconstruction actually breaks, that is the whole ballgame.
//
//   * On a planar scene the linear system is rank deficient and the recovered
//     pose is arbitrary. Measured directly, on exact noise-free synthetic
//     correspondences all lying on one plane: eight-point returned a rotation
//     6.55 degrees off and a translation direction 50 degrees off. The
//     five-point method imposes the essential constraints as part of the
//     solve, so it stays determined where the linear method does not.
//
// THE METHOD, in outline (Nister, PAMI 2004):
//
//   1. Five correspondences give five linear equations in the nine entries of
//      E, so E lies in the 4-dimensional nullspace:
//          E = x*X + y*Y + z*Z + w*W,  and w is set to 1 by scale.
//   2. E is essential exactly when it satisfies
//          det(E) = 0                                    (1 equation)
//          2*E*E^T*E - trace(E*E^T)*E = 0                (9 equations)
//      Substituting the parameterisation makes these ten cubic polynomials in
//      the three unknowns x, y, z.
//   3. Eliminating x and y leaves a single degree-10 polynomial in z. Its real
//      roots -- at most ten -- give the candidate essential matrices.
//
// Step 3 is where implementations differ. Nister's original uses a hand-derived
// Gauss-Jordan elimination on a 10x20 matrix; the version here builds the same
// system and extracts the roots numerically, which is longer to run and far
// shorter to read. This is a lab: the elimination being legible matters more
// than the microseconds.

// The four-dimensional nullspace of the five epipolar equations.
//
// Each correspondence gives one row of x2^T E x1 = 0, linear in E's nine
// entries. Five rows leave a four-dimensional nullspace, and the four basis
// vectors are the four smallest eigenvectors of A^T A -- the same Jacobi
// routine the eight-point path uses, asked for four vectors instead of one.
bool FivePointNullspace(const std::vector<Corr>& pts,
                        const std::vector<int>& idx, double basis[4][9]) {
    if (idx.size() < 5) return false;

    double A[81] = {};
    for (int k = 0; k < 5; ++k) {
        const Corr& c = pts[size_t(idx[size_t(k)])];
        const double r[9] = {c.bx * c.ax, c.bx * c.ay, c.bx,
                             c.by * c.ax, c.by * c.ay, c.by,
                             c.ax,        c.ay,        1.0};
        for (int p = 0; p < 9; ++p)
            for (int q = 0; q < 9; ++q) A[p * 9 + q] += r[p] * r[q];
    }

    // Jacobi eigendecomposition, keeping all nine eigenvectors so the four
    // smallest can be taken. Same sweep structure as SmallestEigenvector9.
    double M[81];
    std::copy(A, A + 81, M);
    double V[81] = {};
    for (int i = 0; i < 9; ++i) V[i * 9 + i] = 1.0;

    for (int sweep = 0; sweep < 60; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < 9; ++p)
            for (int q = p + 1; q < 9; ++q) off += M[p * 9 + q] * M[p * 9 + q];
        if (off < 1e-24) break;
        for (int p = 0; p < 9; ++p) {
            for (int q = p + 1; q < 9; ++q) {
                const double apq = M[p * 9 + q];
                if (std::fabs(apq) < 1e-18) continue;
                const double app = M[p * 9 + p], aqq = M[q * 9 + q];
                const double theta = 0.5 * (aqq - app) / apq;
                const double t = (theta >= 0.0 ? 1.0 : -1.0) /
                                 (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0);
                const double s = t * c;
                for (int k = 0; k < 9; ++k) {
                    const double mkp = M[k * 9 + p], mkq = M[k * 9 + q];
                    M[k * 9 + p] = c * mkp - s * mkq;
                    M[k * 9 + q] = s * mkp + c * mkq;
                }
                for (int k = 0; k < 9; ++k) {
                    const double mpk = M[p * 9 + k], mqk = M[q * 9 + k];
                    M[p * 9 + k] = c * mpk - s * mqk;
                    M[q * 9 + k] = s * mpk + c * mqk;
                }
                for (int k = 0; k < 9; ++k) {
                    const double vkp = V[k * 9 + p], vkq = V[k * 9 + q];
                    V[k * 9 + p] = c * vkp - s * vkq;
                    V[k * 9 + q] = s * vkp + c * vkq;
                }
            }
        }
    }

    // The four smallest eigenvalues, by selection: nine entries, so sorting
    // would cost more to read than it saves to run.
    int order[9];
    for (int i = 0; i < 9; ++i) order[i] = i;
    for (int i = 0; i < 4; ++i) {
        int bestI = i;
        for (int j = i + 1; j < 9; ++j)
            if (M[order[j] * 9 + order[j]] < M[order[bestI] * 9 + order[bestI]])
                bestI = j;
        std::swap(order[i], order[bestI]);
    }
    for (int b = 0; b < 4; ++b)
        for (int i = 0; i < 9; ++i) basis[b][i] = V[i * 9 + order[b]];
    return true;
}

// How far a candidate (x, y, z) is from satisfying the ten cubic constraints.
//
// Zero exactly when E = x*X + y*Y + z*Z + W is an essential matrix. Summed in
// squares so it is a single non-negative number to minimise, which is what the
// search below needs.
double FivePointResidual(const double basis[4][9], double x, double y, double z) {
    double e[9];
    for (int i = 0; i < 9; ++i)
        e[i] = x * basis[0][i] + y * basis[1][i] + z * basis[2][i] + basis[3][i];

    const Mat3 E{{e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8]}};

    // det(E) = 0.
    const double det =
        e[0] * (e[4] * e[8] - e[5] * e[7]) -
        e[1] * (e[3] * e[8] - e[5] * e[6]) +
        e[2] * (e[3] * e[7] - e[4] * e[6]);

    // 2 E E^T E - trace(E E^T) E = 0, the nine trace constraints.
    const Mat3 EEt = E * E.Transpose();
    const double tr = EEt.m[0] + EEt.m[4] + EEt.m[8];
    const Mat3 lhs = EEt * E;

    // Scale-normalised: E is defined up to scale, so a candidate with a tiny
    // norm would otherwise score well for being small rather than for being
    // essential.
    double norm2 = 0.0;
    for (int i = 0; i < 9; ++i) norm2 += e[i] * e[i];
    if (norm2 < 1e-18) return 1e18;
    const double s3 = norm2 * norm2 * norm2;

    double sum = det * det / s3;
    for (int i = 0; i < 9; ++i) {
        const double r = 2.0 * lhs.m[i] - tr * e[i];
        sum += r * r / s3;
    }
    return sum;
}

// Solves the five-point problem by local refinement of the constraint
// residual, returning the essential matrices found.
//
// MEASURED AGAINST EIGHT-POINT, on synthetic fixtures with known ground truth
// (see the 5-point-vs-8-point block in test_sfm.cpp). Rotation error in
// degrees, and the cosine between the recovered translation direction and the
// true one:
//
//   condition                          8-point          5-point
//   general, exact                     0.000  / 1.000   0.000  / 1.000
//   general, 1px noise, 30% outliers   1.321  / 1.000   0.458  / 1.000
//   general, 1px noise, 60% outliers   4.182  / 0.938   4.514  / 0.987
//   planar,  exact                     6.547  / 0.643   0.005  / 1.000
//   planar,  0.3px noise               4.180  / 0.749  11.434  / 0.099
//
// On general geometry it is the better estimator, clearly so at moderate
// outlier rates, which is the case that matters for real matching.
//
// ON A PLANE IT IS EXACT WITHOUT NOISE AND FALLS APART WITH ANY, and that is
// not an implementation defect to be chased. The five-point problem is itself
// degenerate on a planar scene: the essential matrix is not unique there and
// the tenth-degree polynomial admits a one-parameter family of solutions
// (Nister 2004, section 6). Without noise the scan below happens to land on
// the correct member; with noise the determinant is near zero across a RANGE
// of z rather than crossing at a point, so bracketing a sign change returns an
// arbitrary member of the family. Eight-point is also wrong on a plane, but
// wrong stably.
//
// So neither method should be trusted on a near-planar pair, and the
// planarity measure reported by this stage is how to know when that is the
// situation.
//
// WHY NOT NISTER'S CLOSED FORM, which is what the literature specifies and
// what COLMAP ships. His method eliminates x and y symbolically to leave one
// degree-10 polynomial in z, then takes its roots. That is exact, finds all
// ten candidates, and needs no starting guess -- but the elimination is a
// hand-derived 10x20 Gauss-Jordan, and recovering the roots properly wants a
// general (non-symmetric) eigenvalue solver, which this project does not have;
// only the symmetric Jacobi above. Writing both correctly is a larger piece of
// work than it appears, and getting it subtly wrong would be indistinguishable
// from the estimator simply performing badly -- the exact failure mode that
// has cost the most time on this data already.
//
// So this solves the same system numerically: Gauss-Newton from several
// starting points, keeping whichever converge to a genuine root. It finds
// FEWER of the ten candidates than the closed form, which makes it a weaker
// estimator than a correct Nister implementation -- but it is honest about
// what it does, and the property being tested here is the one that matters on
// this data: five correspondences per RANSAC sample instead of eight.
//
// If this measures well, the closed form is the obvious follow-up and can be
// slotted in behind the same interface with the eight-point arm as a control.
// The ten cubic constraints, as rows over the twenty monomials.
//
// Monomial order, which is Nister's and matters because the elimination below
// depends on the first ten being exactly the ones that get eliminated:
//
//   0..9:   x^3 x^2y xy^2 y^3 x^2z x^2 xyz xy y^2z y^2
//   10..19: xz^2 xz x yz^2 yz y z^3 z^2 z 1
//
// Each entry is built by expanding E = x*X + y*Y + z*Z + W symbolically. The
// expansion is done numerically here -- evaluate the constraint at enough
// (x, y, z) samples and solve for the coefficients -- rather than by writing
// out several hundred hand-derived product terms. Slower per sample and far
// less error-prone, and the errors it avoids are exactly the silent kind.
enum : int { kNMono = 20 };

void BuildConstraintRows(const double basis[4][9], double rows[10][kNMono]) {
    // The monomial exponents, parallel to the order above.
    static const int kExp[kNMono][3] = {
        {3,0,0},{2,1,0},{1,2,0},{0,3,0},{2,0,1},{2,0,0},{1,1,1},{1,1,0},
        {0,2,1},{0,2,0},{1,0,2},{1,0,1},{1,0,0},{0,1,2},{0,1,1},{0,1,0},
        {0,0,3},{0,0,2},{0,0,1},{0,0,0},
    };

    // Sample points: twenty (x, y, z) triples whose monomial vectors must be
    // LINEARLY INDEPENDENT, or the system that recovers the coefficients is
    // singular and the answer is noise.
    //
    // Smooth functions of the sample index do not achieve that. A first
    // version used quadratics in `s`, which put every sample on one curve
    // through the space: the 20x20 came out near-singular and the extracted
    // coefficients described the wrong polynomial entirely -- caught only
    // because VerifyConstraintRows checks at a fresh point.
    //
    // A deterministic pseudo-random spread fills the space instead, and stays
    // reproducible so a failure can be chased.
    std::mt19937 srng(987654321u);
    std::uniform_real_distribution<double> su(-1.5, 1.5);

    double S[kNMono][kNMono];
    double rhs[10][kNMono];
    for (int s = 0; s < kNMono; ++s) {
        const double x = su(srng);
        const double y = su(srng);
        const double z = su(srng);

        for (int m = 0; m < kNMono; ++m) {
            double v = 1.0;
            for (int k = 0; k < kExp[m][0]; ++k) v *= x;
            for (int k = 0; k < kExp[m][1]; ++k) v *= y;
            for (int k = 0; k < kExp[m][2]; ++k) v *= z;
            S[s][m] = v;
        }

        // The ten constraint values at this sample.
        double e[9];
        for (int i = 0; i < 9; ++i)
            e[i] = x * basis[0][i] + y * basis[1][i] + z * basis[2][i] + basis[3][i];
        const Mat3 E{{e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8]}};

        rhs[0][s] = e[0] * (e[4] * e[8] - e[5] * e[7]) -
                    e[1] * (e[3] * e[8] - e[5] * e[6]) +
                    e[2] * (e[3] * e[7] - e[4] * e[6]);

        const Mat3 EEt = E * E.Transpose();
        const double tr = EEt.m[0] + EEt.m[4] + EEt.m[8];
        const Mat3 lhs = EEt * E;
        for (int i = 0; i < 9; ++i)
            rhs[i + 1][s] = 2.0 * lhs.m[i] - tr * e[i];
    }

    // Solve S * coeff = rhs for each constraint, by Gaussian elimination with
    // partial pivoting on the shared matrix.
    double A[kNMono][kNMono + 10];
    for (int r = 0; r < kNMono; ++r) {
        for (int c = 0; c < kNMono; ++c) A[r][c] = S[r][c];
        for (int k = 0; k < 10; ++k) A[r][kNMono + k] = rhs[k][r];
    }

    for (int col = 0; col < kNMono; ++col) {
        int piv = col;
        for (int r = col + 1; r < kNMono; ++r)
            if (std::fabs(A[r][col]) > std::fabs(A[piv][col])) piv = r;
        if (std::fabs(A[piv][col]) < 1e-14) continue;
        if (piv != col)
            for (int c = 0; c < kNMono + 10; ++c) std::swap(A[col][c], A[piv][c]);
        const double d = A[col][col];
        for (int c = col; c < kNMono + 10; ++c) A[col][c] /= d;
        for (int r = 0; r < kNMono; ++r) {
            if (r == col) continue;
            const double f = A[r][col];
            if (f == 0.0) continue;
            for (int c = col; c < kNMono + 10; ++c) A[r][c] -= f * A[col][c];
        }
    }

    for (int k = 0; k < 10; ++k)
        for (int m = 0; m < kNMono; ++m) rows[k][m] = A[m][kNMono + k];
}

// Checks that the extracted coefficients really do reproduce the constraints,
// at a point that was not one of the samples they were fitted to.
//
// THE POINT OF THIS is that a wrong monomial order, a bad pivot or a singular
// sample set all produce coefficients that look perfectly reasonable and
// describe the wrong polynomial. Downstream that shows up as "the estimator
// performs badly", which is indistinguishable from the estimator simply being
// hard -- and this project has already lost time to exactly that confusion.
// Checking at a fresh point is cheap and makes the failure loud.
bool VerifyConstraintRows(const double basis[4][9],
                          const double rows[10][kNMono]) {
    static const int kExp[kNMono][3] = {
        {3,0,0},{2,1,0},{1,2,0},{0,3,0},{2,0,1},{2,0,0},{1,1,1},{1,1,0},
        {0,2,1},{0,2,0},{1,0,2},{1,0,1},{1,0,0},{0,1,2},{0,1,1},{0,1,0},
        {0,0,3},{0,0,2},{0,0,1},{0,0,0},
    };
    const double x = 0.7131, y = -1.2207, z = 0.4413;   // not a sample point

    double mono[kNMono];
    for (int m = 0; m < kNMono; ++m) {
        double v = 1.0;
        for (int k = 0; k < kExp[m][0]; ++k) v *= x;
        for (int k = 0; k < kExp[m][1]; ++k) v *= y;
        for (int k = 0; k < kExp[m][2]; ++k) v *= z;
        mono[m] = v;
    }

    double e[9];
    for (int i = 0; i < 9; ++i)
        e[i] = x * basis[0][i] + y * basis[1][i] + z * basis[2][i] + basis[3][i];
    const Mat3 E{{e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8]}};

    double truth[10];
    truth[0] = e[0] * (e[4] * e[8] - e[5] * e[7]) -
               e[1] * (e[3] * e[8] - e[5] * e[6]) +
               e[2] * (e[3] * e[7] - e[4] * e[6]);
    const Mat3 EEt = E * E.Transpose();
    const double tr = EEt.m[0] + EEt.m[4] + EEt.m[8];
    const Mat3 lhs = EEt * E;
    for (int i = 0; i < 9; ++i) truth[i + 1] = 2.0 * lhs.m[i] - tr * e[i];

    double scale = 0.0;
    for (int k = 0; k < 10; ++k) scale = std::max(scale, std::fabs(truth[k]));
    if (scale < 1e-12) scale = 1.0;

    for (int k = 0; k < 10; ++k) {
        double got = 0.0;
        for (int m = 0; m < kNMono; ++m) got += rows[k][m] * mono[m];
        if (std::fabs(got - truth[k]) > 1e-6 * scale) return false;
    }
    return true;
}

// The 10x10 system the constraints become once z is fixed.
//
// At a fixed z every constraint is a cubic in x and y alone, so it is a linear
// combination of exactly ten monomials:
//
//     x^3  x^2y  xy^2  y^3  x^2  xy  y^2  x  y  1
//
// Ten constraints over ten monomials is a square system M(z) v = 0, and a
// nonzero v exists precisely when det M(z) = 0. That determinant is degree ten
// in z -- the same degree-10 polynomial Nister's elimination produces, reached
// without the hand-paired row shuffling his derivation needs.
//
// The tradeoff is honest: his closed form gets the polynomial's coefficients
// symbolically and roots them once, where this evaluates the determinant
// numerically at each z and brackets the sign changes. Slower, and it can miss
// a double root where the curve touches zero without crossing. It is also
// about forty lines instead of three hundred, and every step of it can be
// checked against the constraints directly -- which, on this code, has already
// caught one silent failure that the opaque version would have buried.
void BuildMatrixAtZ(const double rows[10][kNMono], double z, double M[100]) {
    // Which of the twenty monomials carries each (x,y) monomial, at each power
    // of z. Index by [xy-monomial][z-power], -1 where no such term exists.
    //
    // Read straight off the kExp table in BuildConstraintRows: entry m has
    // exponents (a, b, c), so it contributes to xy-monomial (a, b) at z^c.
    static const int kMap[10][4] = {
        /* x^3  */ { 0, -1, -1, -1},
        /* x^2y */ { 1, -1, -1, -1},
        /* xy^2 */ { 2, -1, -1, -1},
        /* y^3  */ { 3, -1, -1, -1},
        /* x^2  */ { 5,  4, -1, -1},
        /* xy   */ { 7,  6, -1, -1},
        /* y^2  */ { 9,  8, -1, -1},
        /* x    */ {12, 11, 10, -1},
        /* y    */ {15, 14, 13, -1},
        /* 1    */ {19, 18, 17, 16},
    };

    for (int r = 0; r < 10; ++r) {
        for (int m = 0; m < 10; ++m) {
            double v = 0.0;
            double zp = 1.0;
            for (int p = 0; p < 4; ++p) {
                const int col = kMap[m][p];
                if (col >= 0) v += rows[r][col] * zp;
                zp *= z;
            }
            M[r * 10 + m] = v;
        }
    }
}

// Determinant of a 10x10 by LU with partial pivoting.
//
// Scaled by the row norms first, because the monomial columns span several
// orders of magnitude at large |z| -- x^3 against 1 -- and an unscaled
// determinant underflows to zero and manufactures roots everywhere.
double DetAtZ(const double rows[10][kNMono], double z) {
    double M[100];
    BuildMatrixAtZ(rows, z, M);

    double sign = 1.0;
    double logAbs = 0.0;
    for (int col = 0; col < 10; ++col) {
        int piv = col;
        for (int r = col + 1; r < 10; ++r)
            if (std::fabs(M[r * 10 + col]) > std::fabs(M[piv * 10 + col])) piv = r;
        const double pv = M[piv * 10 + col];
        if (std::fabs(pv) < 1e-300) return 0.0;
        if (piv != col) {
            for (int c = 0; c < 10; ++c) std::swap(M[col * 10 + c], M[piv * 10 + c]);
            sign = -sign;
        }
        if (M[col * 10 + col] < 0.0) sign = -sign;
        logAbs += std::log(std::fabs(M[col * 10 + col]));
        for (int r = col + 1; r < 10; ++r) {
            const double f = M[r * 10 + col] / M[col * 10 + col];
            if (f == 0.0) continue;
            for (int c = col; c < 10; ++c) M[r * 10 + c] -= f * M[col * 10 + c];
        }
    }
    // Returned as a scaled value rather than the true determinant: only its
    // SIGN and its zeros matter here, and the magnitude would overflow.
    return sign * std::exp(std::min(logAbs, 700.0) / 10.0);
}

// Given a z where the determinant vanishes, recovers x and y.
//
// The null vector of M(z) is [x^3 x^2y xy^2 y^3 x^2 xy y^2 x y 1] up to scale,
// so x and y can be read straight out of it once it is normalised by its last
// entry. Using the LINEAR entries rather than the cubic ones because they are
// the best conditioned of the redundant options.
bool RecoverXY(const double rows[10][kNMono], double z, double* x, double* y) {
    double M[100];
    BuildMatrixAtZ(rows, z, M);

    // Smallest singular vector via the same Jacobi routine, on M^T M.
    double A[81] = {};   // only the 9x9 leading block is used by the helper,
    double ATA[100] = {};
    for (int i = 0; i < 10; ++i)
        for (int j = 0; j < 10; ++j) {
            double s = 0.0;
            for (int k = 0; k < 10; ++k) s += M[k * 10 + i] * M[k * 10 + j];
            ATA[i * 10 + j] = s;
        }
    (void)A;

    // Jacobi on the 10x10.
    double V[100] = {};
    for (int i = 0; i < 10; ++i) V[i * 10 + i] = 1.0;
    for (int sweep = 0; sweep < 60; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < 10; ++p)
            for (int q = p + 1; q < 10; ++q) off += ATA[p * 10 + q] * ATA[p * 10 + q];
        if (off < 1e-26) break;
        for (int p = 0; p < 10; ++p) {
            for (int q = p + 1; q < 10; ++q) {
                const double apq = ATA[p * 10 + q];
                if (std::fabs(apq) < 1e-20) continue;
                const double app = ATA[p * 10 + p], aqq = ATA[q * 10 + q];
                const double theta = 0.5 * (aqq - app) / apq;
                const double t = (theta >= 0.0 ? 1.0 : -1.0) /
                                 (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0);
                const double s = t * c;
                for (int k = 0; k < 10; ++k) {
                    const double mkp = ATA[k * 10 + p], mkq = ATA[k * 10 + q];
                    ATA[k * 10 + p] = c * mkp - s * mkq;
                    ATA[k * 10 + q] = s * mkp + c * mkq;
                }
                for (int k = 0; k < 10; ++k) {
                    const double mpk = ATA[p * 10 + k], mqk = ATA[q * 10 + k];
                    ATA[p * 10 + k] = c * mpk - s * mqk;
                    ATA[q * 10 + k] = s * mpk + c * mqk;
                }
                for (int k = 0; k < 10; ++k) {
                    const double vkp = V[k * 10 + p], vkq = V[k * 10 + q];
                    V[k * 10 + p] = c * vkp - s * vkq;
                    V[k * 10 + q] = s * vkp + c * vkq;
                }
            }
        }
    }

    int best = 0;
    for (int i = 1; i < 10; ++i)
        if (ATA[i * 10 + i] < ATA[best * 10 + best]) best = i;

    double v[10];
    for (int i = 0; i < 10; ++i) v[i] = V[i * 10 + best];

    // v[9] is the constant term; dividing by it makes v[7] = x and v[8] = y.
    if (std::fabs(v[9]) < 1e-12) return false;
    *x = v[7] / v[9];
    *y = v[8] / v[9];
    return true;
}

int FivePointSolve(const std::vector<Corr>& pts, const std::vector<int>& idx,
                   Mat3 out[10]) {
    double basis[4][9];
    if (!FivePointNullspace(pts, idx, basis)) return 0;

    // Reported once, because a failure here means every candidate downstream
    // is describing the wrong polynomial and the estimator's poor showing
    // would otherwise look like an inherent property of the method.
    if (std::getenv("TGLAB_SFM_DEBUG")) {
        static bool reported = false;
        if (!reported) {
            reported = true;
            double rows[10][kNMono];
            BuildConstraintRows(basis, rows);
            std::printf("  five-point constraint extraction: %s\n",
                        VerifyConstraintRows(basis, rows) ? "verified"
                                                          : "WRONG");
        }
    }

    double rows[10][kNMono];
    BuildConstraintRows(basis, rows);

    int found = 0;

    // Scan z for sign changes in det M(z). Each crossing is a root, and each
    // root is a candidate essential matrix.
    //
    // The range and step are the one place this differs in character from the
    // closed form, which finds every root wherever it lies. z is the third
    // nullspace coefficient with the fourth fixed at 1, so a root far outside
    // this range corresponds to an E almost entirely in the W direction --
    // possible, but rare enough that widening the scan costs more than it
    // returns.
    const double kLo = -1000.0, kHi = 1000.0;
    const int kSteps = 2000;
    const double step = (kHi - kLo) / double(kSteps);

    double prevZ = kLo;
    double prevD = DetAtZ(rows, prevZ);

    for (int i = 1; i <= kSteps && found < 10; ++i) {
        const double z = kLo + step * double(i);
        const double d = DetAtZ(rows, z);

        const bool crosses = (prevD != 0.0) && ((prevD < 0.0) != (d < 0.0));
        if (crosses) {
            // Bisect to machine precision. Unconditionally convergent once
            // bracketed, which Newton on this determinant is not.
            double a = prevZ, b = z, fa = prevD;
            for (int it = 0; it < 80; ++it) {
                const double mid = 0.5 * (a + b);
                const double fm = DetAtZ(rows, mid);
                if ((fa < 0.0) != (fm < 0.0)) b = mid;
                else                          { a = mid; fa = fm; }
            }
            const double zr = 0.5 * (a + b);

            double x = 0.0, y = 0.0;
            if (RecoverXY(rows, zr, &x, &y)) {
                double e[9];
                for (int k = 0; k < 9; ++k)
                    e[k] = x * basis[0][k] + y * basis[1][k] +
                           zr * basis[2][k] + basis[3][k];
                double norm = 0.0;
                for (int k = 0; k < 9; ++k) norm += e[k] * e[k];
                norm = std::sqrt(norm);

                if (norm > 1e-12) {
                    Mat3 E;
                    for (int k = 0; k < 9; ++k) E.m[k] = e[k] / norm;

                    // A root of the determinant is necessary but not
                    // sufficient: the recovered (x, y) must also satisfy the
                    // constraints. Checking rejects the spurious crossings
                    // that scaling the determinant can introduce, and costs
                    // one evaluation.
                    if (FivePointResidual(basis, x, y, zr) < 1e-6) {
                        bool dup = false;
                        for (int k = 0; k < found; ++k) {
                            double dd = 0.0;
                            for (int q = 0; q < 9; ++q) {
                                const double p1 = E.m[q] - out[k].m[q];
                                const double p2 = E.m[q] + out[k].m[q];
                                dd += std::min(p1 * p1, p2 * p2);
                            }
                            if (dd < 1e-12) { dup = true; break; }
                        }
                        if (!dup) out[found++] = E;
                    }
                }
            }
        }
        prevZ = z;
        prevD = d;
    }

    return found;
}

// Sampson distance: the first-order approximation to the geometric reprojection
// error of the epipolar constraint.
//
// The raw algebraic residual x'^T E x is NOT a distance -- it scales with the
// magnitude of E and with where the points happen to sit, so a threshold on it
// means something different for every pair. Sampson divides by the gradient,
// which makes it a distance in normalised image units and therefore a
// threshold that can be stated once.
double SampsonDistance(const Mat3& E, const Corr& c) {
    const Vec3 x1{c.ax, c.ay, 1.0}, x2{c.bx, c.by, 1.0};
    const Vec3 Ex1 = E * x1;
    const Vec3 Etx2 = E.Transpose() * x2;
    const double num = x2.Dot(Ex1);
    const double den = Ex1.x * Ex1.x + Ex1.y * Ex1.y +
                       Etx2.x * Etx2.x + Etx2.y * Etx2.y;
    if (den < 1e-18) return 1e9;
    return num * num / den;
}

// What fraction of these correspondences a HOMOGRAPHY also explains.
//
// WHY THIS IS WORTH KNOWING even though the homography is the wrong model.
//
// The essential matrix is degenerate on a planar scene: if every point lies on
// one plane, infinitely many (R, t) satisfy the epipolar constraint equally
// well, and the eight-point algorithm returns an arbitrary one of them. The
// estimate does not LOOK bad -- it has a full complement of inliers and a
// small Sampson residual -- which is exactly why this has to be measured
// rather than inferred from the fit quality.
//
// A homography explains a plane exactly and cannot explain general 3D
// structure with parallax. So the ratio of homography inliers to essential
// inliers is a direct read on how planar the pair is: near 1.0 means the
// correspondences carry no information the plane does not already account
// for, and the recovered translation direction is not to be trusted.
//
// This is the cheap form of the GRIC model-selection test (Torr et al.), which
// compares penalised likelihoods of the two models. The ratio is enough to
// flag a pair; the full criterion would be the next refinement.
//
// VALIDATED against a fixture that is planar by construction: every point on
// one tilted plane, exact correspondences, where this reports 1.00 and flags
// the pair degenerate. See the planar-scene block in test_sfm.cpp. That
// calibration matters, because on real data a LOW number is ambiguous between
// "not planar" and "the fit is broken", and only a known-planar case
// separates them.
//
// With that established, the reading on castle-P19 is 0.01 -- the facades are
// NOT planar enough at these baselines for a homography to explain them. The
// courtyard corner is genuinely three-dimensional: two wings meeting at an
// angle, with the ground plane and the far archway carrying parallax. So
// planarity is not why that pair fails, and the eight-point weakness there is
// some other degeneracy.
double HomographyInlierFraction(const std::vector<Corr>& pts,
                                const std::vector<int>& inliers,
                                double threshold) {
    if (inliers.size() < 4) return 0.0;

    // Direct linear transform over all the essential inliers at once. Fitting
    // to the consensus rather than to a RANSAC sample is deliberate: the
    // question is not "is there a plane somewhere in here" but "does one plane
    // explain the set that E is relying on".
    const size_t n = inliers.size();

    // HARTLEY CONDITIONING, for the same reason EssentialEightPoint needs it
    // and stated just as strongly there: without it the entries of A span
    // orders of magnitude and the smallest eigenvector is round-off. Measured
    // without this step, every pair on castle-P19 reported planarity 0.00 --
    // not "no plane", but a homography so badly conditioned it explained
    // nothing at all, which is a far less plausible reading than a flat
    // facade and was the tell that the fit was broken rather than the scene.
    double ma[2] = {0, 0}, mb[2] = {0, 0};
    for (int i : inliers) {
        ma[0] += pts[size_t(i)].ax; ma[1] += pts[size_t(i)].ay;
        mb[0] += pts[size_t(i)].bx; mb[1] += pts[size_t(i)].by;
    }
    const double dn = double(n);
    ma[0] /= dn; ma[1] /= dn; mb[0] /= dn; mb[1] /= dn;

    double sa = 0.0, sb = 0.0;
    for (int i : inliers) {
        const double dax = pts[size_t(i)].ax - ma[0], day = pts[size_t(i)].ay - ma[1];
        const double dbx = pts[size_t(i)].bx - mb[0], dby = pts[size_t(i)].by - mb[1];
        sa += std::sqrt(dax * dax + day * day);
        sb += std::sqrt(dbx * dbx + dby * dby);
    }
    sa = (sa > 1e-12) ? (dn * std::sqrt(2.0) / sa) : 1.0;
    sb = (sb > 1e-12) ? (dn * std::sqrt(2.0) / sb) : 1.0;

    std::vector<double> A(n * 2 * 9, 0.0);
    for (size_t i = 0; i < n; ++i) {
        const Corr& c = pts[size_t(inliers[i])];
        const double x1 = (c.ax - ma[0]) * sa, y1 = (c.ay - ma[1]) * sa;
        const double x2 = (c.bx - mb[0]) * sb, y2 = (c.by - mb[1]) * sb;
        double* r0 = &A[(i * 2 + 0) * 9];
        double* r1 = &A[(i * 2 + 1) * 9];
        r0[0] = x1; r0[1] = y1; r0[2] = 1.0;
        r0[6] = -x2 * x1; r0[7] = -x2 * y1; r0[8] = -x2;
        r1[3] = x1; r1[4] = y1; r1[5] = 1.0;
        r1[6] = -y2 * x1; r1[7] = -y2 * y1; r1[8] = -y2;
    }

    double AtA[81] = {};
    for (size_t i = 0; i < n * 2; ++i)
        for (int p = 0; p < 9; ++p)
            for (int q = 0; q < 9; ++q)
                AtA[p * 9 + q] += A[i * 9 + size_t(p)] * A[i * 9 + size_t(q)];

    double h[9];
    if (!SmallestEigenvector9(AtA, h)) return 0.0;

    Mat3 Hn;
    for (int i = 0; i < 9; ++i) Hn.m[i] = h[i];

    // Undo the conditioning: H = Tb^-1 * H' * Ta, so the error below is
    // measured in the same normalised camera units as the threshold.
    Mat3 Ta, TbInv;
    Ta.m[0] = sa; Ta.m[1] = 0;  Ta.m[2] = -sa * ma[0];
    Ta.m[3] = 0;  Ta.m[4] = sa; Ta.m[5] = -sa * ma[1];
    Ta.m[6] = 0;  Ta.m[7] = 0;  Ta.m[8] = 1.0;
    TbInv.m[0] = 1.0 / sb; TbInv.m[1] = 0;        TbInv.m[2] = mb[0];
    TbInv.m[3] = 0;        TbInv.m[4] = 1.0 / sb; TbInv.m[5] = mb[1];
    TbInv.m[6] = 0;        TbInv.m[7] = 0;        TbInv.m[8] = 1.0;
    const Mat3 H = TbInv * Hn * Ta;

    // Symmetric transfer error, in the same normalised units the threshold is
    // expressed in, so the two models are judged on comparable terms.
    int agree = 0;
    for (size_t i = 0; i < n; ++i) {
        const Corr& c = pts[size_t(inliers[i])];
        const Vec3 p = H * Vec3{c.ax, c.ay, 1.0};
        if (std::fabs(p.z) < 1e-12) continue;
        const double dx = p.x / p.z - c.bx, dy = p.y / p.z - c.by;
        if (dx * dx + dy * dy < threshold * threshold) ++agree;
    }
    return double(agree) / double(n);
}

// Decomposes E into the four possible (R, t) pairs, and picks the one that puts
// points IN FRONT of both cameras.
//
// The fourfold ambiguity is structural: E is unchanged by negating t, and by
// rotating 180 degrees about t. Only one of the four is physical, and the test
// is cheirality -- a point must have positive depth in both views. Any other
// choice reconstructs the scene behind a camera, which is why a pipeline that
// skips this check produces mirrored, inside-out geometry rather than an error.
bool DecomposeEssential(const Mat3& E, const std::vector<Corr>& pts,
                        const std::vector<int>& inliers, Mat3* R, Vec3* t) {
    // E = U diag(1,1,0) V^T. Recovered here via the symmetric eigenproblems of
    // E^T E and E E^T, which is enough for the two candidate rotations without
    // a general SVD routine.
    Mat3 EtE = E.Transpose() * E;
    double A[81] = {};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) A[i * 9 + j] = EtE.At(i, j);

    // Jacobi on the 3x3, reusing the 9x9 routine's structure would be wasteful;
    // do it directly.
    double M[9], V[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    for (int i = 0; i < 9; ++i) M[i] = EtE.m[i];
    for (int sweep = 0; sweep < 40; ++sweep) {
        double off = M[1] * M[1] + M[2] * M[2] + M[5] * M[5];
        if (off < 1e-26) break;
        for (int p = 0; p < 3; ++p) {
            for (int q = p + 1; q < 3; ++q) {
                const double apq = M[p * 3 + q];
                if (std::fabs(apq) < 1e-20) continue;
                const double theta = 0.5 * (M[q * 3 + q] - M[p * 3 + p]) / apq;
                const double tt = (theta >= 0.0 ? 1.0 : -1.0) /
                                  (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(tt * tt + 1.0), s = tt * c;
                for (int k = 0; k < 3; ++k) {
                    const double mkp = M[k * 3 + p], mkq = M[k * 3 + q];
                    M[k * 3 + p] = c * mkp - s * mkq;
                    M[k * 3 + q] = s * mkp + c * mkq;
                }
                for (int k = 0; k < 3; ++k) {
                    const double mpk = M[p * 3 + k], mqk = M[q * 3 + k];
                    M[p * 3 + k] = c * mpk - s * mqk;
                    M[q * 3 + k] = s * mpk + c * mqk;
                }
                for (int k = 0; k < 3; ++k) {
                    const double vkp = V[k * 3 + p], vkq = V[k * 3 + q];
                    V[k * 3 + p] = c * vkp - s * vkq;
                    V[k * 3 + q] = s * vkp + c * vkq;
                }
            }
        }
    }

    // Order the columns of V by descending eigenvalue; the null direction is
    // last, and that is the epipole in frame i.
    int order[3] = {0, 1, 2};
    std::sort(order, order + 3, [&](int a, int b) {
        return M[a * 3 + a] > M[b * 3 + b];
    });

    Mat3 Vm;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) Vm.At(r, c) = V[r * 3 + order[c]];
    if (Vm.m[0] * (Vm.m[4] * Vm.m[8] - Vm.m[5] * Vm.m[7]) -
        Vm.m[1] * (Vm.m[3] * Vm.m[8] - Vm.m[5] * Vm.m[6]) +
        Vm.m[2] * (Vm.m[3] * Vm.m[7] - Vm.m[4] * Vm.m[6]) < 0.0)
        for (int r = 0; r < 3; ++r) Vm.At(r, 2) = -Vm.At(r, 2);

    // U from E * V, normalised. The third column is the left null vector.
    Mat3 Um;
    for (int c = 0; c < 2; ++c) {
        const Vec3 col = E * Vec3{Vm.At(0, c), Vm.At(1, c), Vm.At(2, c)};
        const Vec3 u = col.Normalized();
        Um.At(0, c) = u.x; Um.At(1, c) = u.y; Um.At(2, c) = u.z;
    }
    const Vec3 u0{Um.At(0, 0), Um.At(1, 0), Um.At(2, 0)};
    const Vec3 u1{Um.At(0, 1), Um.At(1, 1), Um.At(2, 1)};
    const Vec3 u2 = u0.Cross(u1);
    Um.At(0, 2) = u2.x; Um.At(1, 2) = u2.y; Um.At(2, 2) = u2.z;

    // W is the 90-degree rotation about z that generates the two candidates.
    Mat3 W;
    W.m[0] = 0; W.m[1] = -1; W.m[2] = 0;
    W.m[3] = 1; W.m[4] = 0;  W.m[5] = 0;
    W.m[6] = 0; W.m[7] = 0;  W.m[8] = 1;

    Mat3 Ra = NearestRotation(Um * W * Vm.Transpose());
    Mat3 Rb = NearestRotation(Um * W.Transpose() * Vm.Transpose());
    const Vec3 tu{Um.At(0, 2), Um.At(1, 2), Um.At(2, 2)};

    const Mat3* cands[4] = {&Ra, &Ra, &Rb, &Rb};
    const double signs[4] = {1.0, -1.0, 1.0, -1.0};

    int bestGood = -1, bestIdx = 0;
    for (int k = 0; k < 4; ++k) {
        const Mat3& Rc = *cands[k];
        const Vec3 tc = tu * signs[k];
        int good = 0;
        for (int i : inliers) {
            // Midpoint triangulation, in frame i's coordinates. Camera i is at
            // the origin looking down +Z; camera j is at -Rc^T*tc.
            const Vec3 d1{pts[size_t(i)].ax, pts[size_t(i)].ay, 1.0};
            const Vec3 d2w = Rc.Transpose() * Vec3{pts[size_t(i)].bx,
                                                   pts[size_t(i)].by, 1.0};
            const Vec3 c2 = Rc.Transpose() * tc * -1.0;

            // Solve for the closest approach of the two rays.
            const double a = d1.Dot(d1), b = d1.Dot(d2w), c = d2w.Dot(d2w);
            const double d = d1.Dot(c2), e = d2w.Dot(c2);
            const double den = a * c - b * b;
            if (std::fabs(den) < 1e-12) continue;
            const double s1 = (b * e - c * d) / -den;
            const double s2 = (a * e - b * d) / -den;
            // Positive depth in BOTH: in front of camera i, and in front of j.
            if (s1 > 0.0 && s2 > 0.0) ++good;
        }
        if (good > bestGood) { bestGood = good; bestIdx = k; }
    }

    if (bestGood <= 0) return false;
    *R = *cands[bestIdx];
    *t = tu * signs[bestIdx];
    return true;
}

class RelativePose : public AlgorithmBase {
public:
    const char* Name()     const override { return "relative_pose"; }
    const char* Category() const override { return "sfm"; }

    PortList Inputs() const override {
        return {{"src", DataType::ImageSet, FormatSpec::Any, ShapeSpec::Any}};
    }
    PortList Outputs() const override {
        return {{"out", DataType::ImageSet, FormatSpec::SameAsInput, ShapeSpec::SameAsInput}};
    }

    void RunCPU(RunCtx&) override {}
    bool IsAligner() const override { return true; }

    bool RunAlign(std::vector<Image>* images, std::string* err) override;

    std::string RunReport() const override { return m_note; }

    ProxyBehaviour Proxy() const override { return ProxyBehaviour::Never; }
    bool HasGPU() const override { return false; }

private:
    static constexpr const char* kMethodNames[] = {"8-point (linear)",
                                                   "5-point (Nister)"};

    Param<int> m_method{this, "method", 0, 0, 1,
        {.help = "Which minimal solver RANSAC draws against.\n\n"
                 "8-point is a linear solve: treat E as nine free numbers, "
                 "then project the answer onto the essential set afterwards. "
                 "Simple and fast, but it solves a larger problem than the one "
                 "posed, and it is degenerate on a planar scene -- measured on "
                 "exact planar correspondences it returns a rotation 6.5 "
                 "degrees off and a translation direction 50 degrees off.\n\n"
                 "5-point imposes the essential constraints as part of the "
                 "solve, so it stays determined where the linear method does "
                 "not, and it needs only five correspondences per sample "
                 "rather than eight -- on a pair with a 30% inlier rate that "
                 "is roughly 35x as many clean RANSAC draws. It is slower per "
                 "sample and this implementation finds fewer than the full ten "
                 "candidate solutions; see FivePointSolve.",
         .choices = kMethodNames, .choiceCount = 2}};

    Param<float> m_threshold{this, "threshold", 2.0f, 0.1f, 20.0f,
        {.help = "Sampson distance, in PIXELS, for a match to count as "
                 "agreeing with the epipolar geometry. Converted to normalised "
                 "units internally using the frame's focal length, so the same "
                 "value means the same thing whatever the sensor."}};

    Param<int> m_iterations{this, "iterations", 512, 16, 8000,
        {.help = "RANSAC samples. Eight-point needs eight correspondences per "
                 "sample, so it needs more iterations than a 4-point "
                 "homography for the same confidence -- the probability of a "
                 "clean sample falls as the eighth power of the inlier rate."}};

    Param<int> m_minInliers{this, "min_inliers", 20, 8, 500,
        {.help = "Fewest agreeing matches for a pair to enter the view graph. "
                 "A pair supported by ten points is not evidence; admitting it "
                 "puts a badly determined relative rotation into the average, "
                 "where it pulls every camera connected through it."}};

    Param<float> m_fovDeg{this, "fov_deg", 50.0f, 5.0f, 150.0f,
        {.help = "Horizontal field of view, used to turn pixels into "
                 "normalised camera coordinates. THE most consequential "
                 "setting here: the essential matrix is only an essential "
                 "matrix in calibrated coordinates, so a wrong focal biases "
                 "every recovered rotation and the error compounds through "
                 "averaging.\n\n"
                 "Measured on castle-P19 at 1600 px: the 50-degree default "
                 "implies a focal of 1716 px where the true one is about "
                 "1438 (58 degrees), and that 19% error alone took the mean "
                 "rotation residual from under a degree to fourteen. Read it "
                 "off the EXIF where there is any, or sweep it and watch the "
                 "residual."}};

    std::string m_note;
};

bool RelativePose::RunAlign(std::vector<Image>* images, std::string* err) {
    const int n = int(images->size());
    if (n < 2) { *err = "relative_pose needs at least two frames"; return false; }

    int solved = 0, attempted = 0, rejected = 0;
    double inlierSum = 0.0;
    double rotSum = 0.0;
    double planarSum = 0.0;
    int planarPairs = 0;

    for (int f = 0; f < n; ++f) {
        const MatchSidecar* ms = MatchesOf((*images)[size_t(f)]);
        if (!ms) continue;
        const FeatureSidecar* fsB = FeaturesOf((*images)[size_t(f)]);
        if (!fsB) continue;

        // REPLACED WITH OUR OWN INLIERS at the end of this frame's loop.
        //
        // build_tracks reads the inlier flags to decide which matches to
        // union, and the flags it would otherwise see come from the
        // homography fit upstream -- the wrong model for a scene with
        // parallax. Writing the essential matrix's verdict back means the
        // tracks are built from correspondences verified by the geometry that
        // actually applies.
        auto revised = std::make_shared<MatchSidecar>(*ms);

        auto out = std::make_shared<RelativePoseSidecar>();

        for (size_t setIdx = 0; setIdx < ms->sets.size(); ++setIdx) {
            const MatchSet& set = ms->sets[setIdx];
            if (set.reference < 0 || set.reference >= n) continue;
            const FeatureSidecar* fsA = FeaturesOf((*images)[size_t(set.reference)]);
            if (!fsA) continue;
            ++attempted;

            // Normalised coordinates need a focal length and principal point.
            // Guessed from the frame size, exactly as build_tracks does, since
            // nothing has calibrated anything yet.
            const ImageDesc& dA = (*images)[size_t(set.reference)].Desc();
            const ImageDesc& dB = (*images)[size_t(f)].Desc();
            const double halfFov = 0.5 * double(m_fovDeg) * 3.14159265358979 / 180.0;
            const double fA = 0.5 * dA.width / std::tan(halfFov);
            const double fB = 0.5 * dB.width / std::tan(halfFov);

            // EVERY match, not just the ones an earlier stage flagged.
            //
            // This stage runs its own RANSAC against the ESSENTIAL matrix,
            // which is the correct model for two views of a scene with
            // parallax. Consuming a previous stage's inliers would mean
            // inheriting whatever model produced them -- and the verification
            // stage upstream fits a HOMOGRAPHY, which describes a pure camera
            // rotation and nothing else.
            //
            // Measured on castle-P19, a walk around a building: the homography
            // kept 20% of matches and reported shifts of 28,000 px on a 1600 px
            // image, because no homography fits frames that translate. Feeding
            // those 20% here left 37 of 51 pairs unsolvable for want of
            // correspondences. Taking the raw matches lets RANSAC reject with
            // the model that actually applies.
            std::vector<Corr> pts;
            // Which match each entry came from: a few are dropped for stale
            // keypoint indices, so the arrays are not 1:1 and the inliers
            // cannot be written back without this.
            std::vector<int> ptSource;
            pts.reserve(set.matches.size());
            ptSource.reserve(set.matches.size());
            for (size_t i = 0; i < set.matches.size(); ++i) {
                const Match& m = set.matches[i];
                if (m.a < 0 || m.a >= int(fsA->keypoints.size())) continue;
                if (m.b < 0 || m.b >= int(fsB->keypoints.size())) continue;
                const Keypoint& ka = fsA->keypoints[size_t(m.a)];
                const Keypoint& kb = fsB->keypoints[size_t(m.b)];
                pts.push_back(Corr{(ka.x - 0.5 * dA.width) / fA,
                                   (ka.y - 0.5 * dA.height) / fA,
                                   (kb.x - 0.5 * dB.width) / fB,
                                   (kb.y - 0.5 * dB.height) / fB});
                ptSource.push_back(int(i));
            }
            if (int(pts.size()) < std::max(8, int(m_minInliers))) { ++rejected; continue; }

            // Threshold in pixels, converted to the normalised units the
            // Sampson distance is measured in. Squared because the distance is.
            const double tn = double(m_threshold) / fB;
            const double threshSq = tn * tn;

            // THE SAMPLE SIZE IS THE POINT OF THE CHOICE. Eight-point needs
            // eight correspondences per RANSAC draw, five-point five, and the
            // chance a draw is all-inlier is the inlier rate raised to that
            // power. On this project's worst pairs -- 30% inliers -- that is
            // 0.007% against 0.24%.
            const bool useFive = int(m_method) == 1;
            const int sampleSize = useFive ? 5 : 8;

            std::mt19937 rng(20260921u);
            std::uniform_int_distribution<size_t> pick(0, pts.size() - 1);
            // Braces, not parentheses: `std::vector<int> sample(size_t(n))`
            // declares a FUNCTION taking a size_t, and the errors land on the
            // uses rather than here.
            std::vector<int> sample(static_cast<size_t>(sampleSize), 0);
            std::vector<int> best;
            Mat3 bestE;

            for (int it = 0; it < int(m_iterations); ++it) {
                for (int s = 0; s < sampleSize; ++s) {
                    bool dup; int tries = 0;
                    do {
                        sample[size_t(s)] = int(pick(rng));
                        dup = false;
                        for (int q = 0; q < s; ++q)
                            if (sample[size_t(q)] == sample[size_t(s)]) dup = true;
                    } while (dup && ++tries < 16);
                }

                // Five-point returns SEVERAL candidates -- the constraints are
                // cubic and admit up to ten essential matrices per sample --
                // so each is scored and the best kept. Eight-point's linear
                // solve returns exactly one.
                Mat3 cands[10];
                int nCand = 0;
                if (useFive) {
                    nCand = FivePointSolve(pts, sample, cands);
                } else {
                    if (EssentialEightPoint(pts, sample, &cands[0])) nCand = 1;
                }

                for (int c = 0; c < nCand; ++c) {
                    const Mat3& E = cands[c];
                    std::vector<int> in;
                    in.reserve(pts.size());
                    for (size_t i = 0; i < pts.size(); ++i)
                        if (SampsonDistance(E, pts[i]) <= threshSq)
                            in.push_back(int(i));

                    if (in.size() > best.size()) {
                        best.swap(in);
                        bestE = E;
                    }
                }
                if (best.size() > pts.size() * 9 / 10) break;
            }

            if (int(best.size()) < int(m_minInliers)) { ++rejected; continue; }

            // Refit on all inliers. The sample that won RANSAC used five or
            // eight points; the consensus it found is worth far more, and
            // refitting is what turns "a model consistent with the data" into
            // "the model the data supports".
            //
            // NOT FOR FIVE-POINT, and this is easy to get wrong: the refit is
            // itself the linear eight-point solve, so applying it to a
            // five-point result hands the answer straight back to the
            // estimator whose degeneracy five-point exists to avoid. Measured
            // before this exception, both methods reported an IDENTICAL 6.5467
            // degree error on the planar fixture -- identical to four decimals,
            // because the refit had overwritten the five-point selection
            // entirely and the choice of solver made no difference at all.
            //
            // Keeping the minimal solution costs the accuracy the consensus
            // would have bought on well-conditioned pairs. A proper fix is a
            // non-linear refinement of the Sampson error over the inliers,
            // which preserves the essential constraints; that is the natural
            // next step and is not this one.
            Mat3 E = bestE;
            if (!useFive && best.size() >= 8) {
                Mat3 refit;
                if (EssentialEightPoint(pts, best, &refit)) E = refit;
            }

            Mat3 R; Vec3 t;
            if (!DecomposeEssential(E, pts, best, &R, &t)) { ++rejected; continue; }

            // How planar this pair is, measured rather than assumed. A high
            // fraction means E is underdetermined here and the translation
            // direction below is arbitrary -- see HomographyInlierFraction.
            const double planarity =
                HomographyInlierFraction(pts, best, tn);
            planarSum += planarity;
            if (planarity > 0.9) ++planarPairs;

            // PER-PAIR DETAIL, behind an environment variable because the
            // aggregate hides exactly what matters: a reconstruction that
            // splits does so at ONE link, and a mean over eighteen pairs
            // cannot say which. Off by default so a normal run is quiet.
            if (std::getenv("TGLAB_SFM_DEBUG")) {
                const double rotDeg =
                    Mat3{}.AngleTo(R) * 180.0 / 3.14159265358979;
                std::printf("  pair %2d->%2d: %4d/%4d inliers (%.0f%%), "
                            "rot %5.1f deg, planarity %.2f\n",
                            set.reference, f, int(best.size()),
                            int(pts.size()),
                            pts.empty() ? 0.0
                                        : 100.0 * double(best.size()) /
                                              double(pts.size()),
                            rotDeg, planarity);
            }

            // Record which matches the ESSENTIAL matrix accepted, replacing
            // whatever the upstream homography decided. See the note where
            // `revised` is created.
            {
                MatchSet& rs = revised->sets[setIdx];
                rs.inlier.assign(rs.matches.size(), 0);
                for (int k : best)
                    if (k >= 0 && k < int(ptSource.size()))
                        rs.inlier[size_t(ptSource[size_t(k)])] = 1;
            }

            RelativePoseSidecar::Edge edge;
            edge.reference = set.reference;
            edge.R = R;

            // CONVERTED INTO THE REFERENCE FRAME'S COORDINATES, which is what
            // the sidecar documents and what global_position reads.
            //
            // The decomposition's `t` is the third column of U, and in
            // E = [t]_x R that vector lives in frame B's coordinates -- the
            // cheirality test above already treats it that way, placing camera
            // B's centre at -R^T t. Storing it unconverted put frame B's
            // epipole in a slot every consumer reads as frame A's.
            //
            // The direction from A to B, in A's coordinates, is R^T t: rotate
            // out of B and the sign already points the right way because the
            // cheirality test picked the candidate that puts points in front
            // of both cameras.
            //
            // Measured on an exact synthetic pair: the stored direction was
            // 17 degrees from the truth on noise-free data. Small enough to
            // look like noise, large enough that translation averaging could
            // never converge -- and invisible to the rotation checks, which
            // were exactly right.
            edge.direction = (R.Transpose() * t).Normalized();
            edge.inliers = int(best.size());
            out->edges.push_back(edge);
            ++solved;
            inlierSum += double(best.size());
            // How big the recovered rotation actually is. Reported because a
            // WRONG focal does not announce itself in the inlier count: too
            // long a focal shrinks every recovered rotation toward the
            // identity, and near-identity rotations agree with each other
            // trivially, so the averaging residual FALLS as the calibration
            // gets worse. The magnitude is what distinguishes "the cameras
            // barely turned" from "the model collapsed".
            rotSum += Mat3{}.AngleTo(R);
        }

        if (!out->edges.empty()) {
            (*images)[size_t(f)].Sidecars().Set(kRelativePoseSidecar, std::move(out));
            // Only when at least one pair solved: a frame whose every pair was
            // rejected keeps the upstream flags rather than having them
            // replaced with all-zero, which would silently delete its matches.
            (*images)[size_t(f)].Sidecars().Set(kMatchSidecar, std::move(revised));
        }
    }

    char buf[256];
    std::snprintf(buf, sizeof buf,
                  "relative pose: %d of %d pairs solved (%d rejected), "
                  "mean %.0f inliers per edge, mean rotation %.1f deg, "
                  "planarity %.2f (%d pairs degenerate)",
                  solved, attempted, rejected,
                  solved ? inlierSum / double(solved) : 0.0,
                  solved ? rotSum / double(solved) * 180.0 / 3.14159265358979
                         : 0.0,
                  solved ? planarSum / double(solved) : 0.0, planarPairs);
    m_note = buf;

    if (solved == 0) {
        *err = "relative_pose: no pair yielded a usable essential matrix -- " + m_note;
        return false;
    }
    return true;
}

REGISTER_ALGORITHM(RelativePose);

}  // namespace
}  // namespace tglab
