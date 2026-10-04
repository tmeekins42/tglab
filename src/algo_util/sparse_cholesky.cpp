// sparse_cholesky — see the header.

#include "sparse_cholesky.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <set>

#include "../core/parallel.h"
#include "linalg.h"

namespace tglab {
namespace linalg {

bool BlockSparseCholesky::Analyse(const std::vector<int>& sizes,
                                  const std::vector<std::pair<int, int>>& pairs) {
    const int n = int(sizes.size());
    m_size = sizes;
    m_scalar.assign(size_t(n), 0);
    m_dim = 0;
    for (int i = 0; i < n; ++i) {
        if (sizes[size_t(i)] < 1) return false;
        m_scalar[size_t(i)] = m_dim;
        m_dim += sizes[size_t(i)];
    }

    // --- the caller's pattern, by column ---------------------------------
    std::vector<std::vector<int>> lower(static_cast<size_t>(n));   // rows i > j, per column j
    std::vector<std::vector<int>> adj(static_cast<size_t>(n));     // the graph, both directions
    for (const auto& pr : pairs) {
        int i = pr.first, j = pr.second;
        if (i < 0 || j < 0 || i >= n || j >= n) return false;
        if (i == j) continue;
        if (i < j) std::swap(i, j);
        lower[size_t(j)].push_back(i);
        adj[size_t(i)].push_back(j);
        adj[size_t(j)].push_back(i);
    }
    m_aColPtr.assign(size_t(n) + 1, 0);
    m_aRow.clear();
    m_aOff.clear();
    m_numValues = 0;
    for (int j = 0; j < n; ++j) {
        auto& b = lower[size_t(j)];
        std::sort(b.begin(), b.end());
        b.erase(std::unique(b.begin(), b.end()), b.end());
        m_aColPtr[size_t(j)] = m_aRow.size();
        m_aRow.push_back(j);   // the diagonal, first
        m_aOff.push_back(m_numValues);
        m_numValues += size_t(sizes[size_t(j)]) * size_t(sizes[size_t(j)]);
        for (int i : b) {
            m_aRow.push_back(i);
            m_aOff.push_back(m_numValues);
            m_numValues += size_t(sizes[size_t(i)]) * size_t(sizes[size_t(j)]);
        }
    }
    m_aColPtr[size_t(n)] = m_aRow.size();
    for (auto& a : adj) {
        std::sort(a.begin(), a.end());
        a.erase(std::unique(a.begin(), a.end()), a.end());
    }

    // --- minimum degree, on the explicit elimination graph ---------------
    //
    // Eliminating v joins all of its remaining neighbours into a clique; the
    // neighbours at that moment ARE column v's pattern in the factor, fill
    // included, so ordering and symbolic factorisation are one pass.
    //
    // Explicit rather than a quotient graph: the graphs here are a camera per
    // node -- thousands, not millions -- and the plain version is short
    // enough to check by reading. Ties go to the lower node number, which
    // keeps a video close to its own sequence.
    m_perm.assign(size_t(n), 0);
    m_pos.assign(size_t(n), -1);
    std::vector<std::vector<int>> colNodes(static_cast<size_t>(n));   // per position, neighbours at elimination
    std::set<std::pair<int, int>> queue;   // (degree, node)
    for (int v = 0; v < n; ++v) queue.insert({int(adj[size_t(v)].size()), v});
    std::vector<int> merged;
    for (int k = 0; k < n; ++k) {
        const int v = queue.begin()->second;
        queue.erase(queue.begin());
        m_perm[size_t(k)] = v;
        m_pos[size_t(v)] = k;
        std::vector<int> nb = std::move(adj[size_t(v)]);
        adj[size_t(v)].clear();
        for (int u : nb) {
            // adj[u] := (adj[u] U nb) \ {u, v}
            std::vector<int>& au = adj[size_t(u)];
            queue.erase({int(au.size()), u});
            merged.clear();
            merged.reserve(au.size() + nb.size());
            std::set_union(au.begin(), au.end(), nb.begin(), nb.end(), std::back_inserter(merged));
            au.clear();
            for (int w : merged)
                if (w != u && w != v) au.push_back(w);
            queue.insert({int(au.size()), u});
        }
        colNodes[size_t(k)] = std::move(nb);
    }

    // --- the factor's layout ---------------------------------------------
    m_posScalar.assign(size_t(n) + 1, 0);
    for (int k = 0; k < n; ++k)
        m_posScalar[size_t(k) + 1] = m_posScalar[size_t(k)] + sizes[size_t(m_perm[size_t(k)])];
    std::vector<std::vector<int>> below(static_cast<size_t>(n));   // per position, later rows
    for (int k = 0; k < n; ++k) {
        auto& rows = below[size_t(k)];
        for (int u : colNodes[size_t(k)]) rows.push_back(m_pos[size_t(u)]);
        std::sort(rows.begin(), rows.end());
    }

    // SUPERNODES. Column k+1 joins k's run when k's later rows are exactly
    // k+1 and then k+1's own: the run then has one pattern below it, and its
    // columns can be stored and worked on as a single dense panel. A video's
    // columns nest like this for long stretches, and the work becomes long
    // dot products instead of 6x6 blocks one at a time.
    m_super.clear();
    m_superOf.assign(size_t(n), 0);
    m_colOff.assign(size_t(n), 0);
    m_sRow.clear();
    m_sRowOff.clear();
    m_numL = 0;
    for (int k = 0; k < n;) {
        int last = k + 1;
        while (last < n) {
            const auto& prev = below[size_t(last - 1)];
            const auto& next = below[size_t(last)];
            if (prev.empty() || prev[0] != last || prev.size() != next.size() + 1 ||
                !std::equal(next.begin(), next.end(), prev.begin() + 1))
                break;
            ++last;
        }
        Super sn;
        sn.first = k;
        sn.last = last;
        sn.width = m_posScalar[size_t(last)] - m_posScalar[size_t(k)];
        sn.rowPtr = m_sRow.size();
        int rowsDown = sn.width;
        for (int r : below[size_t(last - 1)]) {
            m_sRow.push_back(r);
            m_sRowOff.push_back(rowsDown);
            rowsDown += sizes[size_t(m_perm[size_t(r)])];
        }
        sn.nBelow = int(below[size_t(last - 1)].size());
        sn.rows = rowsDown;
        sn.panel = m_numL;
        m_numL += size_t(sn.rows) * size_t(sn.width);
        for (int q = k; q < last; ++q) {
            m_superOf[size_t(q)] = int(m_super.size());
            m_colOff[size_t(q)] = m_posScalar[size_t(q)] - m_posScalar[size_t(k)];
        }
        m_super.push_back(sn);
        k = last;
    }
    m_L.assign(m_numL, 0.0);

    // Where each of the caller's blocks lands: the caller's (i, j), i >= j,
    // is the factor's (pos i, pos j) when the ordering kept i after j, and
    // its transpose otherwise -- in the supernode of the earlier one.
    m_scatter.clear();
    m_scatter.reserve(m_aRow.size());
    for (int j = 0; j < n; ++j)
        for (size_t q = m_aColPtr[size_t(j)]; q < m_aColPtr[size_t(j) + 1]; ++q) {
            const int i = m_aRow[q];
            const int pi = m_pos[size_t(i)], pj = m_pos[size_t(j)];
            const int lo = std::min(pi, pj), hi = std::max(pi, pj);
            const int s = m_superOf[size_t(lo)];
            const Super& sn = m_super[size_t(s)];
            Scatter sc;
            sc.from = m_aOff[q];
            sc.rows = sizes[size_t(i)];
            sc.cols = sizes[size_t(j)];
            sc.ld = sn.width;
            sc.transposed = pi < pj;
            sc.to = sn.panel + size_t(RowIn(s, hi)) * size_t(sn.width) + size_t(m_colOff[size_t(lo)]);
            m_scatter.push_back(sc);
        }
    return true;
}

int BlockSparseCholesky::RowIn(int s, int pos) const {
    const Super& sn = m_super[size_t(s)];
    if (pos < sn.last) return m_colOff[size_t(pos)];   // one of its own
    const auto b = m_sRow.begin() + std::ptrdiff_t(sn.rowPtr);
    const auto e = b + sn.nBelow;
    const auto it = std::lower_bound(b, e, pos);
    return m_sRowOff[size_t(it - m_sRow.begin())];
}

long long BlockSparseCholesky::Offset(int i, int j) const {
    if (i < j) std::swap(i, j);
    const auto b = m_aRow.begin() + std::ptrdiff_t(m_aColPtr[size_t(j)]);
    const auto e = m_aRow.begin() + std::ptrdiff_t(m_aColPtr[size_t(j) + 1]);
    const auto it = std::lower_bound(b, e, i);
    if (it == e || *it != i) return -1;
    return (long long)m_aOff[size_t(it - m_aRow.begin())];
}

void BlockSparseCholesky::Offsets(int j, const int* rows, int count, long long* out) const {
    size_t q = m_aColPtr[size_t(j)];
    const size_t e = m_aColPtr[size_t(j) + 1];
    for (int k = 0; k < count; ++k) {
        const int r = rows[k];
        while (q < e && m_aRow[q] < r) ++q;
        out[k] = (q < e && m_aRow[q] == r) ? (long long)m_aOff[q] : -1;
    }
}

void BlockSparseCholesky::Add(double* values, int i, int j, const double* m) const {
    const long long off = Offset(i, j);
    if (off < 0) return;
    double* dst = values + off;
    const int si = m_size[size_t(i)], sj = m_size[size_t(j)];
    if (i >= j) {
        for (int a = 0; a < si * sj; ++a) dst[a] += m[a];
    } else {
        // Stored as block (j, i): sj x si.
        for (int a = 0; a < si; ++a)
            for (int b = 0; b < sj; ++b) dst[b * si + a] += m[a * sj + b];
    }
}

bool BlockSparseCholesky::Factor(const double* values) {
    std::fill(m_L.begin(), m_L.end(), 0.0);
    for (const Scatter& sc : m_scatter) {
        const double* src = values + sc.from;
        double* dst = m_L.data() + sc.to;
        const size_t ld = size_t(sc.ld);
        if (!sc.transposed) {
            for (int a = 0; a < sc.rows; ++a)
                for (int b = 0; b < sc.cols; ++b) dst[size_t(a) * ld + size_t(b)] = src[a * sc.cols + b];
        } else {
            for (int a = 0; a < sc.rows; ++a)
                for (int b = 0; b < sc.cols; ++b) dst[size_t(b) * ld + size_t(a)] = src[a * sc.cols + b];
        }
    }

    auto sizeAt = [&](int pos) { return m_size[size_t(m_perm[size_t(pos)])]; };

    for (size_t si = 0; si < m_super.size(); ++si) {
        const Super& sn = m_super[si];
        const int W = sn.width;
        double* P = m_L.data() + sn.panel;
        auto row = [&](int i) { return P + size_t(i) * size_t(W); };

        // THE PANEL, factored as the dense Cholesky factors a matrix -- with
        // its kernel, because a panel IS one: a tall matrix whose top W x W
        // is the diagonal block, the rows under it filling only those W
        // columns. Blocked by 64 columns; once a block's own rows are done,
        // every row under it fills its share of them independently.
        constexpr int kBlock = 64;
        for (int j0 = 0; j0 < W; j0 += kBlock) {
            const int j1 = std::min(W, j0 + kBlock);
            for (int i = j0; i < j1; ++i)
                for (int j = j0; j <= i; ++j) {
                    const double v = detail::CholeskyEntry(P, W, i, j);
                    if (i == j) {
                        if (!(v > 0.0)) return false;
                        row(i)[i] = std::sqrt(v);
                    } else {
                        row(i)[j] = v / row(j)[j];
                    }
                }
            const int under = sn.rows - j1;
            auto fill = [&](size_t r) {
                const int i = j1 + int(r);
                for (int j = j0; j < j1; ++j) row(i)[j] = detail::CholeskyEntry(P, W, i, j) / row(j)[j];
            };
            if (size_t(under) * size_t(j1 - j0) * size_t(j1) >= 200000) ParallelFor(size_t(under), fill);
            else
                for (int r = 0; r < under; ++r) fill(size_t(r));
        }

        // THE UPDATE of every later column the supernode touches: for each
        // pair of its below rows a >= b, block (a, b) -= X_a X_b^T, a dot
        // product W long per entry. Block (a, b) lives in b's supernode at
        // a's row there; the rows below b in this panel are a subset of that
        // supernode's rows, by the elimination property. Each b is its own
        // set of columns, so they can go in parallel.
        const int nb = sn.nBelow;
        if (nb == 0) continue;
        auto update = [&](size_t qb) {
            const int b = m_sRow[sn.rowPtr + qb];
            const int sb = sizeAt(b);
            const int tsI = m_superOf[size_t(b)];
            const Super& ts = m_super[size_t(tsI)];
            const size_t tW = size_t(ts.width);
            double* T = m_L.data() + ts.panel;
            const int cb = m_colOff[size_t(b)];
            const double* Xb = row(m_sRowOff[sn.rowPtr + qb]);
            // a's row in the target, by a walk: the a ascend, and so do the
            // target's rows -- its own nodes, then those below it.
            size_t tq = ts.rowPtr;
            const size_t tqEnd = ts.rowPtr + size_t(ts.nBelow);
            for (int qa = int(qb); qa < nb; ++qa) {
                const int a = m_sRow[sn.rowPtr + size_t(qa)];
                const int sa = sizeAt(a);
                int ra;
                if (a < ts.last) {
                    ra = m_colOff[size_t(a)];
                } else {
                    while (tq < tqEnd && m_sRow[tq] < a) ++tq;
                    ra = m_sRowOff[tq];
                }
                const double* Xa = row(m_sRowOff[sn.rowPtr + size_t(qa)]);
                for (int r = 0; r < sa; ++r) {
                    const double* xa = Xa + size_t(r) * size_t(W);
                    double* t = T + size_t(ra + r) * tW + size_t(cb);
                    for (int c = 0; c < sb; ++c) {
                        const double* xb = Xb + size_t(c) * size_t(W);
                        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
                        int k = 0;
                        for (; k + 4 <= W; k += 4) {
                            s0 += xa[k] * xb[k];
                            s1 += xa[k + 1] * xb[k + 1];
                            s2 += xa[k + 2] * xb[k + 2];
                            s3 += xa[k + 3] * xb[k + 3];
                        }
                        for (; k < W; ++k) s0 += xa[k] * xb[k];
                        t[c] -= (s0 + s1) + (s2 + s3);
                    }
                }
            }
        };
        const size_t work = size_t(nb) * size_t(nb) * size_t(W) * 18;
        if (nb >= 4 && work >= 200000) ParallelFor(size_t(nb), update);
        else
            for (int q = 0; q < nb; ++q) update(size_t(q));
    }
    return true;
}

void BlockSparseCholesky::Solve(double* b) const {
    const int n = Nodes();
    std::vector<double> y(static_cast<size_t>(m_dim));
    for (int k = 0; k < n; ++k) {
        const int v = m_perm[size_t(k)];
        for (int a = 0; a < m_size[size_t(v)]; ++a)
            y[size_t(m_posScalar[size_t(k)] + a)] = b[m_scalar[size_t(v)] + a];
    }
    auto sizeAt = [&](int pos) { return m_size[size_t(m_perm[size_t(pos)])]; };

    // L z = y, a supernode at a time: its own block forward, then its rows
    // below pushed into the later unknowns.
    for (const Super& sn : m_super) {
        const size_t W = size_t(sn.width);
        const double* P = m_L.data() + sn.panel;
        double* yk = y.data() + m_posScalar[size_t(sn.first)];
        for (size_t c = 0; c < W; ++c) {
            double v = yk[c];
            for (size_t t = 0; t < c; ++t) v -= P[c * W + t] * yk[t];
            yk[c] = v / P[c * W + c];
        }
        for (int q = 0; q < sn.nBelow; ++q) {
            const int r = m_sRow[sn.rowPtr + size_t(q)];
            const int sr = sizeAt(r);
            const double* X = P + size_t(m_sRowOff[sn.rowPtr + size_t(q)]) * W;
            double* yr = y.data() + m_posScalar[size_t(r)];
            for (int a = 0; a < sr; ++a) {
                double v = 0.0;
                for (size_t c = 0; c < W; ++c) v += X[size_t(a) * W + c] * yk[c];
                yr[a] -= v;
            }
        }
    }
    // L^T x = z, in reverse.
    for (size_t si = m_super.size(); si-- > 0;) {
        const Super& sn = m_super[si];
        const size_t W = size_t(sn.width);
        const double* P = m_L.data() + sn.panel;
        double* yk = y.data() + m_posScalar[size_t(sn.first)];
        for (int q = 0; q < sn.nBelow; ++q) {
            const int r = m_sRow[sn.rowPtr + size_t(q)];
            const int sr = sizeAt(r);
            const double* X = P + size_t(m_sRowOff[sn.rowPtr + size_t(q)]) * W;
            const double* yr = y.data() + m_posScalar[size_t(r)];
            for (int a = 0; a < sr; ++a)
                for (size_t c = 0; c < W; ++c) yk[c] -= X[size_t(a) * W + c] * yr[a];
        }
        for (size_t c = W; c-- > 0;) {
            double v = yk[c];
            for (size_t t = c + 1; t < W; ++t) v -= P[t * W + c] * yk[t];
            yk[c] = v / P[c * W + c];
        }
    }
    for (int k = 0; k < n; ++k) {
        const int v = m_perm[size_t(k)];
        for (int a = 0; a < m_size[size_t(v)]; ++a)
            b[m_scalar[size_t(v)] + a] = y[size_t(m_posScalar[size_t(k)] + a)];
    }
}

} // namespace linalg
} // namespace tglab
