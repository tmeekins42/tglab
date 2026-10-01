// linalg — the dense linear algebra the algorithms share.
//
// Small and dependency-free, like geometry.h's Vec3 and Mat3: the problems
// here are a 3x3 covariance per point, a 9x9 normal matrix per RANSAC sample,
// a reduced camera system of a few hundred unknowns. A general library would
// cost more to build and read than these few routines do.
//
// CONVENTIONS. Matrices are ROW-MAJOR in flat storage, a[i * n + j]. Fixed
// sizes take std::array (or a plain C array) and a template size, so a
// per-point call allocates nothing; the solvers for systems whose size is
// only known at run time take a pointer and n. Routines that factor in place
// say so, and destroy their input.
//
// Each of these used to exist as one or more private copies inside the
// algorithms that needed it -- five Jacobi eigen-solvers, three Cholesky
// solves, two Gaussian eliminations -- which drifted apart in their
// tolerances and their care. One copy each, so a fix reaches every caller.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <utility>

namespace tglab {
namespace linalg {

// --- symmetric eigen-decomposition ----------------------------------------------
//
// CYCLIC JACOBI: sweeps of plane rotations, each zeroing one off-diagonal
// entry, until the off-diagonal is negligible. Unconditionally convergent for
// a symmetric matrix, accurate in the small eigenvalues -- which is what the
// callers want, the least eigenvector being a nullspace -- and short. For the
// sizes here (3 to 9) it beats anything cleverer.
//
// The rotation is Rutishauser's: t = tan(theta) from the smaller root of
// t^2 + 2 theta t - 1 = 0, which stays accurate when the diagonal entries are
// nearly equal, where computing the angle by atan2 and then its sine and
// cosine loses digits.
//
// BOTH TOLERANCES ARE RELATIVE to the matrix's size, so a covariance of
// millimetre offsets converges like one of metres:
//
//   * done when the off-diagonal's squared norm is 1e-30 of the whole's;
//   * an entry under 1e-18 of the norm is left alone. It carries no
//     information -- double precision resolves about 1e-16 -- and rotating
//     it away is NOT harmless: between two nearly equal eigenvalues the
//     rotation is about 45 degrees, mixing their eigenvectors arbitrarily.
//     The five-point solver's nullspace is four eigenvalues near zero, and
//     the basis it starts from sets its conditioning: rotating such entries
//     moved fountain-P11's estimated field of view from 56.5 to 70 degrees.
//     (Earlier copies skipped them by an ABSOLUTE 1e-18, right only for
//     matrices of order one.)
//
// `values` come back ASCENDING; `vectors` holds the matching eigenvectors as
// its COLUMNS (vectors[r * N + c] is entry r of eigenvector c), unit length.
template <int N>
void SymmetricEigen(const double* A, double* values, double* vectors) {
    std::array<double, N * N> M;
    std::array<double, N * N> V{};
    for (int i = 0; i < N * N; ++i) M[size_t(i)] = A[i];
    for (int i = 0; i < N; ++i) V[size_t(i * N + i)] = 1.0;

    double scale = 0.0;
    for (int i = 0; i < N * N; ++i) scale += M[size_t(i)] * M[size_t(i)];
    const double tiny = 1e-18 * std::sqrt(scale);
    for (int sweep = 0; sweep < 64; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < N; ++p)
            for (int q = p + 1; q < N; ++q) off += M[size_t(p * N + q)] * M[size_t(p * N + q)];
        if (off <= 1e-30 * scale || off == 0.0) break;

        for (int p = 0; p < N; ++p)
            for (int q = p + 1; q < N; ++q) {
                const double apq = M[size_t(p * N + q)];
                if (std::fabs(apq) <= tiny) continue;
                const double theta = 0.5 * (M[size_t(q * N + q)] - M[size_t(p * N + p)]) / apq;
                const double t = (theta >= 0.0 ? 1.0 : -1.0) /
                                 (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
                for (int k = 0; k < N; ++k) {
                    const double kp = M[size_t(k * N + p)], kq = M[size_t(k * N + q)];
                    M[size_t(k * N + p)] = c * kp - s * kq;
                    M[size_t(k * N + q)] = s * kp + c * kq;
                }
                for (int k = 0; k < N; ++k) {
                    const double pk = M[size_t(p * N + k)], qk = M[size_t(q * N + k)];
                    M[size_t(p * N + k)] = c * pk - s * qk;
                    M[size_t(q * N + k)] = s * pk + c * qk;
                }
                for (int k = 0; k < N; ++k) {
                    const double kp = V[size_t(k * N + p)], kq = V[size_t(k * N + q)];
                    V[size_t(k * N + p)] = c * kp - s * kq;
                    V[size_t(k * N + q)] = s * kp + c * kq;
                }
            }
    }

    std::array<int, N> order;
    for (int i = 0; i < N; ++i) order[size_t(i)] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return M[size_t(a * N + a)] < M[size_t(b * N + b)];
    });
    for (int c = 0; c < N; ++c) {
        const int src = order[size_t(c)];
        values[c] = M[size_t(src * N + src)];
        for (int r = 0; r < N; ++r) vectors[r * N + c] = V[size_t(r * N + src)];
    }
}

// The eigenvector of the smallest eigenvalue: the least-squares solution of a
// homogeneous system A x = 0 when given A^T A, unit length.
template <int N>
void SmallestEigenvector(const double* A, double* out) {
    double values[N], vectors[N * N];
    SymmetricEigen<N>(A, values, vectors);
    for (int r = 0; r < N; ++r) out[r] = vectors[r * N];
}

// --- linear solves ---------------------------------------------------------------

// A x = b for symmetric positive-definite A, by Cholesky, IN PLACE: `a`
// (n x n) is overwritten by its factor L, `b` by x. Only the lower triangle
// of `a` is read.
//
// For normal equations, which are SPD by construction: half the work of a
// general elimination, and backward stable without pivoting. False when A is
// not positive definite -- with Levenberg damping applied that means the
// problem is genuinely degenerate, and a caller raises the damping and tries
// again.
inline bool CholeskySolve(double* a, double* b, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j <= i; ++j) {
            double s = a[size_t(i) * size_t(n) + size_t(j)];
            for (int k = 0; k < j; ++k)
                s -= a[size_t(i) * size_t(n) + size_t(k)] * a[size_t(j) * size_t(n) + size_t(k)];
            if (i == j) {
                if (!(s > 0.0)) return false;
                a[size_t(i) * size_t(n) + size_t(i)] = std::sqrt(s);
            } else {
                a[size_t(i) * size_t(n) + size_t(j)] = s / a[size_t(j) * size_t(n) + size_t(j)];
            }
        }
    for (int i = 0; i < n; ++i) {   // L y = b
        double s = b[i];
        for (int k = 0; k < i; ++k) s -= a[size_t(i) * size_t(n) + size_t(k)] * b[k];
        b[i] = s / a[size_t(i) * size_t(n) + size_t(i)];
    }
    for (int i = n - 1; i >= 0; --i) {   // L^T x = y
        double s = b[i];
        for (int k = i + 1; k < n; ++k) s -= a[size_t(k) * size_t(n) + size_t(i)] * b[k];
        b[i] = s / a[size_t(i) * size_t(n) + size_t(i)];
    }
    return true;
}

// A x = b for a general square A, by Gaussian elimination with partial
// pivoting, IN PLACE: `a` is destroyed and `b` becomes x.
//
// Pivoting is what turns a near-singular system -- four nearly collinear
// points in a RANSAC sample -- into a clean "singular" (false) rather than
// garbage from dividing by a tiny leading element. `tiny` is the smallest
// pivot accepted.
inline bool LuSolve(double* a, double* b, int n, double tiny = 1e-12) {
    for (int col = 0; col < n; ++col) {
        int piv = col;
        for (int r = col + 1; r < n; ++r)
            if (std::fabs(a[size_t(r) * size_t(n) + size_t(col)]) >
                std::fabs(a[size_t(piv) * size_t(n) + size_t(col)]))
                piv = r;
        if (std::fabs(a[size_t(piv) * size_t(n) + size_t(col)]) < tiny) return false;
        if (piv != col) {
            for (int c = 0; c < n; ++c)
                std::swap(a[size_t(col) * size_t(n) + size_t(c)], a[size_t(piv) * size_t(n) + size_t(c)]);
            std::swap(b[col], b[piv]);
        }
        const double d = a[size_t(col) * size_t(n) + size_t(col)];
        for (int r = col + 1; r < n; ++r) {
            const double f = a[size_t(r) * size_t(n) + size_t(col)] / d;
            if (f == 0.0) continue;
            for (int c = col; c < n; ++c)
                a[size_t(r) * size_t(n) + size_t(c)] -= f * a[size_t(col) * size_t(n) + size_t(c)];
            b[r] -= f * b[col];
        }
    }
    for (int r = n - 1; r >= 0; --r) {
        double s = b[r];
        for (int c = r + 1; c < n; ++c) s -= a[size_t(r) * size_t(n) + size_t(c)] * b[c];
        b[r] = s / a[size_t(r) * size_t(n) + size_t(r)];
    }
    return true;
}

}  // namespace linalg
}  // namespace tglab
