// init_splats — a Gaussian per point, before any training.
//
// WHAT THIS IS FOR. Gaussian splatting (Kerbl et al. 2023) optimises millions
// of Gaussians against the photographs, and every Gaussian has to start
// somewhere. The paper starts from the sparse SfM points; this pipeline has a
// dense cloud as well, which is a much better start. This stage makes one
// Gaussian per point and nothing more -- training is a later stage, and this
// is what it will begin from.
//
// It is also worth running on its own. An untrained splat set drawn as sorted,
// blended ellipses already reads as a continuous surface where the same cloud
// drawn as dots reads as a scatter, because each Gaussian covers the gap to
// its neighbours instead of leaving it.
//
// HOW EACH GAUSSIAN IS SHAPED. The paper initialises every Gaussian as a
// sphere sized to the mean distance of its three nearest neighbours. That is
// right for an optimiser, which will reshape them anyway, and wrong for
// viewing untrained: a sphere on a wall bulges out of the wall by its own
// radius. So this goes one step further and fits the surface:
//
//   * the k nearest neighbours give a local covariance, and its smallest
//     eigenvector is the surface normal -- the same PCA bench_sfm uses to
//     measure a wall's thickness, applied per point
//   * the Gaussian is a DISC in that plane: `size` times the neighbour spacing
//     across, `flatness` times that through
//
// Where a point has too few neighbours to fit a plane it stays a sphere,
// which is the paper's choice and the honest one when the shape is unknown.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "../../algo_util/linalg.h"
#include "../../core/algorithm.h"
#include "../../core/parallel.h"

namespace tglab {
namespace {

// A uniform grid over the points, for nearest-neighbour search.
//
// SORTED KEYS rather than a hash map: one sort of (cell, index) pairs and a
// binary search per lookup. For a few million points that is both faster and
// far lighter than a map of vectors.
struct Grid {
    double cell = 1.0;
    Vec3   lo;
    std::vector<std::pair<uint64_t, int>> keyed;

    static uint64_t Key(int x, int y, int z) {
        const uint64_t mask = (1ull << 21) - 1;
        return (uint64_t(x) & mask) | ((uint64_t(y) & mask) << 21) |
               ((uint64_t(z) & mask) << 42);
    }
    void CellOf(const Vec3& p, int* x, int* y, int* z) const {
        *x = int(std::floor((p.x - lo.x) / cell));
        *y = int(std::floor((p.y - lo.y) / cell));
        *z = int(std::floor((p.z - lo.z) / cell));
    }

    void Build(const std::vector<Vec3>& pts, double cellSize) {
        cell = cellSize;
        lo = pts.empty() ? Vec3{0, 0, 0} : pts[0];
        for (const Vec3& p : pts) {
            lo.x = std::min(lo.x, p.x);
            lo.y = std::min(lo.y, p.y);
            lo.z = std::min(lo.z, p.z);
        }
        keyed.resize(pts.size());
        for (size_t i = 0; i < pts.size(); ++i) {
            int x, y, z;
            CellOf(pts[i], &x, &y, &z);
            keyed[i] = {Key(x, y, z), int(i)};
        }
        std::sort(keyed.begin(), keyed.end());
    }

    // Every point in the cube of cells within `ring` of p's cell.
    template <class F>
    void Visit(const Vec3& p, int ring, F&& f) const {
        int cx, cy, cz;
        CellOf(p, &cx, &cy, &cz);
        for (int dz = -ring; dz <= ring; ++dz)
            for (int dy = -ring; dy <= ring; ++dy)
                for (int dx = -ring; dx <= ring; ++dx) {
                    const int x = cx + dx, y = cy + dy, z = cz + dz;
                    if (x < 0 || y < 0 || z < 0) continue;
                    const uint64_t k = Key(x, y, z);
                    auto it = std::lower_bound(
                        keyed.begin(), keyed.end(), std::make_pair(k, -1));
                    for (; it != keyed.end() && it->first == k; ++it)
                        f(it->second);
                }
    }
};

class InitSplats : public AlgorithmBase {
public:
    const char* Name()     const override { return "init_splats"; }
    const char* Category() const override { return "sfm"; }

    PortList Inputs() const override {
        return {{"src", DataType::PointCloud, FormatSpec::Any, ShapeSpec::Any}};
    }
    PortList Outputs() const override {
        return {{"out", DataType::PointCloud, FormatSpec::Any, ShapeSpec::Any}};
    }

    void RunCPU(RunCtx&) override {}
    bool IsReconstruct() const override { return true; }
    ProxyBehaviour Proxy() const override { return ProxyBehaviour::Never; }
    bool HasGPU() const override { return false; }

    bool RunReconstruct(const std::vector<Image>*, PointCloud* cloud,
                        std::string* err) override {
        std::vector<Vec3> pts;
        std::vector<Vec3> cols;
        pts.reserve(cloud->tracks.size());
        cols.reserve(cloud->tracks.size());
        for (const Track& t : cloud->tracks) {
            if (!t.hasPoint) continue;
            pts.push_back(t.point);
            const bool has = t.color.x != 0.0 || t.color.y != 0.0 || t.color.z != 0.0;
            cols.push_back(has ? t.color : Vec3{0.72, 0.72, 0.74});
        }
        // NO MORE THAN TRAINING CAN KEEP. A fused room scan is sixteen million
        // points; training holds at most max_gaussians (a million by
        // default) and spent its first densify steps pruning the rest, at a
        // quarter of a second an iteration while they lasted -- and fitting
        // a disc to each one here took 49 s. A random draw, fixed seed so a
        // rerun gives the same splats, keeps the cloud's coverage; the
        // spacing below is then measured on what was kept, so each Gaussian
        // grows to cover the gap the dropped ones leave.
        const size_t total = pts.size();
        const size_t cap = size_t(std::max(4, int(m_maxSplats)));
        if (pts.size() > cap) {
            std::mt19937 rng(12345u);
            for (size_t i = 0; i < cap; ++i) {
                std::uniform_int_distribution<size_t> pick(i, pts.size() - 1);
                const size_t j = pick(rng);
                std::swap(pts[i], pts[j]);
                std::swap(cols[i], cols[j]);
            }
            pts.resize(cap);
            cols.resize(cap);
        }
        const int n = int(pts.size());
        if (n < 4) {
            *err = "init_splats: fewer than four points -- run triangulate or "
                   "fuse_depth first";
            return false;
        }

        // CELL SIZE FROM THE POINT SPACING, assuming the points lie on
        // surfaces -- which is what a reconstruction is. N points over a
        // surface of extent E sit about E / sqrt(N) apart, and a cell of twice
        // that holds a handful of them, so the 27-cell neighbourhood searched
        // below holds on the order of a hundred candidates. Sizing for a
        // VOLUME (E / cbrt(N)) puts hundreds of points in every cell on a
        // surface and makes the search thousands of times slower.
        //
        // The extent is the 2-98 percentile box, for the reason the viewer
        // frames on percentiles: a few strays decide a bounding box.
        const double extent = RobustExtent(pts);
        const double cell = std::max(1e-9, 2.0 * extent / std::sqrt(double(n)));
        Grid grid;
        grid.Build(pts, cell);

        const int k = std::clamp(int(m_neighbours), 3, 32);
        const double size = double(m_size), flat = double(m_flatness);
        const double opacity = std::clamp(double(m_opacity), 0.01, 1.0);

        std::vector<Splat> out(static_cast<size_t>(n));
        std::vector<char>  planar(static_cast<size_t>(n), 0);
        std::vector<double> spacings(static_cast<size_t>(n), 0.0);
        std::vector<int>    found(static_cast<size_t>(n), 0);

        ParallelFor(n, [&](int i) {
            const Vec3& p = pts[size_t(i)];

            // The k nearest, widening the search until enough are found. A
            // point on an isolated fragment may not find k within two rings,
            // and gets whatever it has.
            std::vector<std::pair<double, int>> near;
            for (int ring = 1; ring <= 2; ++ring) {
                near.clear();
                grid.Visit(p, ring, [&](int j) {
                    if (j == i) return;
                    const Vec3 d = pts[size_t(j)] - p;
                    near.emplace_back(d.Dot(d), j);
                });
                if (int(near.size()) >= k) break;
            }
            const int m = std::min(k, int(near.size()));
            std::partial_sort(near.begin(), near.begin() + m, near.end());

            // Spacing: the mean distance to the neighbours found.
            double spacing = 0.0;
            for (int j = 0; j < m; ++j) spacing += std::sqrt(near[size_t(j)].first);
            spacing = (m > 0) ? spacing / double(m) : cell * 0.5;
            spacings[size_t(i)] = spacing;
            found[size_t(i)] = m;

            Splat s;
            s.mean = p;
            s.color = cols[size_t(i)];
            s.opacity = opacity;

            const double r = std::max(1e-9, spacing * size);
            if (m >= 3) {
                // Local covariance about the centroid of the neighbourhood.
                Vec3 c = p;
                for (int j = 0; j < m; ++j) c = c + pts[size_t(near[size_t(j)].second)];
                c = c * (1.0 / double(m + 1));
                double a[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
                auto add = [&](const Vec3& q) {
                    const double d[3] = {q.x - c.x, q.y - c.y, q.z - c.z};
                    for (int r2 = 0; r2 < 3; ++r2)
                        for (int c2 = 0; c2 < 3; ++c2) a[r2][c2] += d[r2] * d[c2];
                };
                add(p);
                for (int j = 0; j < m; ++j) add(pts[size_t(near[size_t(j)].second)]);

                double ev[3];
                Mat3 V;
                linalg::SymmetricEigen<3>(&a[0][0], ev, V.m);

                // Columns: the two in-plane axes, then the normal (the
                // SMALLEST eigenvector). Rebuilt as e0 x e1 so the frame is
                // right-handed whatever sign Jacobi returned -- a reflection
                // is not a rotation and has no quaternion.
                const Vec3 e0 = V.Column(2), e1 = V.Column(1);
                const Vec3 nrm = e0.Cross(e1).Normalized();
                Mat3 R;
                R.m[0] = e0.x; R.m[1] = e1.x; R.m[2] = nrm.x;
                R.m[3] = e0.y; R.m[4] = e1.y; R.m[5] = nrm.y;
                R.m[6] = e0.z; R.m[7] = e1.z; R.m[8] = nrm.z;
                MatToQuat(R, s.rot);
                s.scale = Vec3{r, r, std::max(1e-9, r * flat)};
                planar[size_t(i)] = 1;
            } else {
                s.scale = Vec3{r, r, r};
            }
            out[size_t(i)] = s;
        });

        // ISOLATED POINTS ARE DROPPED. A point whose neighbours are several
        // times further off than is typical for this cloud is not on a
        // surface with the rest: it is a stray that survived fusion -- on a
        // selfie video, a sprinkle of wall far behind the head. Kept, each
        // becomes a Gaussian that training grows into a blob to paint the
        // background it sits in front of.
        int dropped = 0;
        const double factor = double(m_isolated);
        if (factor > 0.0) {
            std::vector<double> sorted = spacings;
            std::nth_element(sorted.begin(), sorted.begin() + long(sorted.size() / 2),
                             sorted.end());
            const double limit = factor * sorted[sorted.size() / 2];
            std::vector<Splat> kept;
            std::vector<char> keptPlanar;
            kept.reserve(out.size());
            for (size_t i = 0; i < out.size(); ++i) {
                if (found[i] < 3 || spacings[i] > limit) { ++dropped; continue; }
                kept.push_back(out[i]);
                keptPlanar.push_back(planar[i]);
            }
            out.swap(kept);
            planar.swap(keptPlanar);
        }

        int nPlanar = 0;
        for (char c : planar) nPlanar += c;
        const int nOut = int(out.size());
        cloud->splats = std::move(out);
        // New Gaussians have plain colour; any coefficients were someone else's.
        cloud->splatSh.clear();
        cloud->shDegree = 0;
        cloud->splatRefl.clear();
        cloud->envMap.clear();
        cloud->envRes = 0;

        char buf[320];
        std::snprintf(buf, sizeof(buf),
                      "init_splats: %d Gaussians from %d points, %d oriented to "
                      "the surface, %d left spherical (too few neighbours); "
                      "%d isolated points dropped",
                      nOut, n, nPlanar, nOut - nPlanar, dropped);
        m_note = buf;
        if (total > size_t(n))
            m_note += "; drawn at random from " + std::to_string(total) + " (max_splats)";
        return true;
    }

    std::string RunReport() const override { return m_note; }

private:
    static double RobustExtent(std::vector<Vec3> pts) {
        auto pct = [&](auto get) {
            std::vector<double> v;
            v.reserve(pts.size());
            for (const Vec3& p : pts) v.push_back(get(p));
            const size_t lo = size_t(0.02 * double(v.size() - 1));
            const size_t hi = size_t(0.98 * double(v.size() - 1));
            std::nth_element(v.begin(), v.begin() + long(lo), v.end());
            const double a = v[lo];
            std::nth_element(v.begin(), v.begin() + long(hi), v.end());
            return v[hi] - a;
        };
        const double ex = pct([](const Vec3& p) { return p.x; });
        const double ey = pct([](const Vec3& p) { return p.y; });
        const double ez = pct([](const Vec3& p) { return p.z; });
        return std::max(1e-9, std::sqrt(ex * ex + ey * ey + ez * ez));
    }

    // See the draw in RunReconstruct. Matches train_splats' max_gaussians.
    Param<int> m_maxSplats{this, "max_splats", 1000000, 4, 100000000,
        {.help = "At most this many Gaussians, drawn at random from the points "
                 "when there are more. Training keeps no more than its "
                 "max_gaussians anyway, so making more only costs time.",
         .softMax = 10000000.0}};

    Param<float> m_isolated{this, "isolated", 3.0f, 0.0f, 20.0f,
        {.help = "Drop a point whose nearest neighbours are more than this "
                 "many times further off than the cloud's median spacing -- a "
                 "stray, not part of a surface. 0 keeps every point.",
         .step = 0.5}};

    Param<int> m_neighbours{this, "neighbours", 8, 3, 32,
        {.help = "How many nearest points define each Gaussian's size and the "
                 "surface it lies in. More gives a smoother orientation and "
                 "blurs fine detail; fewer follows the surface more closely "
                 "and is noisier."}};

    // How wide each disc is relative to the spacing of its neighbours. A
    // Gaussian's visible extent is about two standard deviations, so 0.6
    // makes neighbouring discs just overlap: less leaves gaps, more smears.
    Param<float> m_size{this, "size", 0.6f, 0.05f, 4.0f,
        {.help = "Each Gaussian's width as a fraction of the distance to its "
                 "neighbours. Too small leaves the surface full of holes; too "
                 "large smears detail across neighbours.",
         .step = 0.05, .softMax = 2.0}};

    Param<float> m_flatness{this, "flatness", 0.1f, 0.01f, 1.0f,
        {.help = "Thickness through the surface as a fraction of the width. "
                 "1 is a sphere, as the Gaussian-splatting paper starts; small "
                 "values make discs lying in the fitted surface, which look "
                 "far better before any training.",
         .step = 0.01}};

    Param<float> m_opacity{this, "opacity", 0.9f, 0.01f, 1.0f,
        {.help = "Starting opacity for every Gaussian. Training will set each "
                 "one; untrained, high opacity reads as solid surface.",
         .step = 0.05}};

    std::string m_note;
};

REGISTER_ALGORITHM(InitSplats);

}  // namespace
}  // namespace tglab
