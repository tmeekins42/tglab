// bench_sparse -- the block-sparse Cholesky on a synthetic reduced camera
// system shaped like a long video's: each camera coupled to the next
// `window`, revisits of earlier passes as a walk around a room makes them,
// and one size-1 node coupled to all of them (a shared focal). Times
// analysis, factorisation and solve, and checks the solve by its residual.
//
//   bench_sparse [cameras] [window] [revisits] [runs]
//
// The default, 1600 cameras of 6 unknowns, a window of 15 and 2 revisits a
// camera, is about the 1577-frame room scan (IMG_1537) that sparse_cholesky
// was written for -- and runs in seconds instead of the quarter hour that
// clip takes to reach bundle adjustment.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "../src/algo_util/linalg.h"
#include "../src/algo_util/sparse_cholesky.h"

using namespace tglab;

static double Ms(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

int main(int argc, char** argv) {
    const int nCam = argc > 1 ? std::atoi(argv[1]) : 1600;
    const int window = argc > 2 ? std::atoi(argv[2]) : 15;
    const int revisits = argc > 3 ? std::atoi(argv[3]) : 2;
    const int runs = argc > 4 ? std::atoi(argv[4]) : 5;
    const int bs = 6;

    std::mt19937 rng(11);
    std::uniform_real_distribution<double> U(-1.0, 1.0);

    std::vector<int> sizes(static_cast<size_t>(nCam), bs);
    sizes.push_back(1);
    const int focal = nCam;
    std::vector<std::pair<int, int>> pairs;
    for (int c = 0; c < nCam; ++c) {
        for (int d = 1; d <= window && c + d < nCam; ++d) pairs.push_back({c + d, c});
        // Revisits as a walk makes them: passing an earlier stretch again,
        // each frame tied to a few neighbouring frames of an earlier pass.
        // (Uniformly random pairs instead fill the factor thirty times over,
        // where the room scan's fills it twice.)
        for (int r = 1; r <= revisits; ++r) {
            const int o = c - r * 337;
            for (int d = -1; d <= 1; ++d)
                if (o + d >= 0) pairs.push_back({c, o + d});
        }
        pairs.push_back({focal, c});
    }

    linalg::BlockSparseCholesky sp;
    auto t0 = std::chrono::steady_clock::now();
    if (!sp.Analyse(sizes, pairs)) { std::printf("analyse failed\n"); return 1; }
    auto t1 = std::chrono::steady_clock::now();
    const size_t dim = size_t(sp.Dim());
    std::printf("%d cameras, window %d, %d revisits each: %zu unknowns, %zu values, factor %zu "
                "(%.2fx), %d supernodes; analysis %.0f ms\n",
                nCam, window, revisits, dim, sp.NumValues(), sp.FactorValues(),
                double(sp.FactorValues()) / double(sp.NumValues()), sp.Supernodes(), Ms(t0, t1));

    // Values: random blocks, then a dominant diagonal so the matrix is
    // positive definite whatever the blocks are.
    std::vector<double> vals(sp.NumValues(), 0.0);
    std::vector<double> rowAbs(dim, 0.0);
    std::vector<double> blk;
    auto add = [&](int i, int j) {
        blk.assign(size_t(sp.Size(i)) * size_t(sp.Size(j)), 0.0);
        for (auto& v : blk) v = U(rng);
        sp.Add(vals.data(), i, j, blk.data());
        for (int a = 0; a < sp.Size(i); ++a)
            for (int b = 0; b < sp.Size(j); ++b) {
                const double v = std::fabs(blk[size_t(a * sp.Size(j) + b)]);
                rowAbs[size_t(sp.Scalar(i) + a)] += v;
                rowAbs[size_t(sp.Scalar(j) + b)] += v;
            }
    };
    for (const auto& p : pairs) add(p.first, p.second);
    for (int n = 0; n <= nCam; ++n) {
        const long long off = sp.Offset(n, n);
        for (int a = 0; a < sp.Size(n); ++a)
            vals[size_t(off) + size_t(a * sp.Size(n) + a)] = rowAbs[size_t(sp.Scalar(n) + a)] + 1.0;
    }

    std::vector<double> rhs(dim);
    for (auto& v : rhs) v = U(rng);
    double best = 1e30;
    std::vector<double> x;
    for (int r = 0; r < runs; ++r) {
        x = rhs;
        auto f0 = std::chrono::steady_clock::now();
        if (!sp.Factor(vals.data())) { std::printf("factor failed\n"); return 1; }
        auto f1 = std::chrono::steady_clock::now();
        sp.Solve(x.data());
        auto f2 = std::chrono::steady_clock::now();
        best = std::min(best, Ms(f0, f1));
        std::printf("  run %d: factor %.1f ms, solve %.1f ms\n", r + 1, Ms(f0, f1), Ms(f1, f2));
    }
    std::printf("best factor %.1f ms\n", best);

    // The residual, |A x - b| / |b|, from the sparse values directly.
    std::vector<double> Ax(dim, 0.0);
    for (int j = 0; j <= nCam; ++j)
        for (int i = j; i <= nCam; ++i) {
            const long long off = sp.Offset(i, j);
            if (off < 0) continue;
            const int si = sp.Size(i), sj = sp.Size(j);
            for (int a = 0; a < si; ++a)
                for (int b = 0; b < sj; ++b) {
                    const double v = vals[size_t(off) + size_t(a * sj + b)];
                    Ax[size_t(sp.Scalar(i) + a)] += v * x[size_t(sp.Scalar(j) + b)];
                    if (i != j) Ax[size_t(sp.Scalar(j) + b)] += v * x[size_t(sp.Scalar(i) + a)];
                }
        }
    double num = 0.0, den = 0.0;
    for (size_t k = 0; k < dim; ++k) {
        num += (Ax[k] - rhs[k]) * (Ax[k] - rhs[k]);
        den += rhs[k] * rhs[k];
    }
    std::printf("relative residual %.2e\n", std::sqrt(num / den));
    return std::sqrt(num / den) < 1e-9 ? 0 : 1;
}
