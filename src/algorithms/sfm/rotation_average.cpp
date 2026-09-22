// rotation_average — global camera orientations from pairwise relative ones.
//
// THE PROBLEM. The view graph gives relative rotations R_ij between pairs of
// frames. Wanted is one GLOBAL rotation R_i per frame, consistent with all of
// them at once:
//
//     R_ij  =  R_j * R_i^T        for every edge (i, j)
//
// Overdetermined and inconsistent, which is the whole point. Measurement noise
// means composing rotations around a loop does not return to the identity, and
// a chain that simply multiplies relative rotations accumulates that error
// without bound -- drift, the characteristic failure of incremental methods.
// Averaging uses every edge simultaneously, so a loop closure constrains the
// frames inside it and the drift has nowhere to accumulate.
//
// WHY ROTATIONS CAN BE AVERAGED AND TRANSLATIONS CANNOT. A relative rotation is
// fully determined by two views. A relative translation is known only in
// DIRECTION -- two-view geometry cannot see baseline length -- so translations
// have a missing scale per edge that has to be solved jointly with the
// structure. That asymmetry is why these are separate stages in that order, and
// why rotation averaging is the well-conditioned half of global SfM.
//
// GAUGE FREEDOM, which every test here has to respect. Rotating every camera by
// the same R leaves all relative rotations unchanged, so the solution is only
// determined up to a global rotation. There is no "correct" absolute
// orientation to recover. The convention below pins frame 0 to the identity,
// which is a choice, not a result.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <queue>
#include <string>
#include <vector>

#include "../../algo_util/view_graph.h"
#include "../../core/algorithm.h"

namespace tglab {
namespace {

// One edge of the view graph, flattened out of the per-frame sidecars.
struct Edge {
    int i = -1, j = -1;   // R_ij takes frame i's coordinates into frame j's
    Mat3 R;
    double weight = 1.0;
};

// --- spanning-tree initialisation -------------------------------------------
//
// Both averaging methods below are ITERATIVE and both need somewhere to start.
// Starting from the identity everywhere is a poor choice: the residuals are
// then large and the linearisation that L2 relies on is least trustworthy
// exactly where it matters.
//
// Propagating along a maximum spanning tree instead gives an estimate already
// close to the answer. The tree is chosen by edge weight (inlier count), so the
// path to every frame goes through the best-supported edges available, and the
// error that does accumulate along it accumulates through the most reliable
// measurements. This is what every implementation in the literature does, and
// skipping it is a common reason a hand-written averager converges to
// something plausible but wrong.
bool SpanningTreeInit(int n, const std::vector<Edge>& edges,
                      std::vector<Mat3>* R, std::vector<bool>* seen) {
    std::vector<std::vector<int>> adj;
    adj.resize(size_t(n));
    for (size_t e = 0; e < edges.size(); ++e) {
        adj[size_t(edges[e].i)].push_back(int(e));
        adj[size_t(edges[e].j)].push_back(int(e));
    }

    R->assign(size_t(n), Mat3::Identity());
    seen->assign(size_t(n), false);

    // Prim's, by descending weight. A priority queue over edges rather than
    // over frames, because the weight lives on the edge.
    struct Cand { double w; int edge; int to; };
    auto worse = [](const Cand& a, const Cand& b) { return a.w < b.w; };
    std::priority_queue<Cand, std::vector<Cand>, decltype(worse)> pq(worse);

    (*seen)[0] = true;
    for (int e : adj[0]) {
        const Edge& ed = edges[size_t(e)];
        pq.push(Cand{ed.weight, e, ed.i == 0 ? ed.j : ed.i});
    }

    int reached = 1;
    while (!pq.empty()) {
        const Cand c = pq.top();
        pq.pop();
        if ((*seen)[size_t(c.to)]) continue;

        const Edge& ed = edges[size_t(c.edge)];
        // R_ij = R_j R_i^T, so R_j = R_ij R_i and R_i = R_ij^T R_j.
        if (c.to == ed.j) (*R)[size_t(ed.j)] = ed.R * (*R)[size_t(ed.i)];
        else              (*R)[size_t(ed.i)] = ed.R.Transpose() * (*R)[size_t(ed.j)];

        (*seen)[size_t(c.to)] = true;
        ++reached;
        for (int e : adj[size_t(c.to)]) {
            const Edge& nx = edges[size_t(e)];
            const int other = nx.i == c.to ? nx.j : nx.i;
            if (!(*seen)[size_t(other)]) pq.push(Cand{nx.weight, e, other});
        }
    }
    return reached > 1;
}

// The angular residual of an edge against the current estimate, in radians.
double EdgeResidual(const Edge& e, const std::vector<Mat3>& R) {
    const Mat3 predicted = R[size_t(e.j)] * R[size_t(e.i)].Transpose();
    return predicted.AngleTo(e.R);
}

class RotationAverage : public AlgorithmBase {
public:
    const char* Name()     const override { return "rotation_average"; }
    const char* Category() const override { return "sfm"; }

    PortList Inputs() const override {
        return {{"src", DataType::ImageSet, FormatSpec::Any, ShapeSpec::Any}};
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
    // --- L2: one linear pass, then project back onto SO(3) ------------------
    //
    // Martinec & Pajdla (CVPR 2007). Relaxes the rotation constraint entirely:
    // treat each R_i as nine free numbers, minimise ||R_j - R_ij R_i||^2 over
    // all edges, then project the result back to the nearest rotation.
    //
    // FAST AND NOT ROBUST, and both halves matter. It is a linear least-squares
    // problem so there are no local minima to get stuck in, which is exactly
    // why it makes a good initialiser. But least squares gives every edge equal
    // say regardless of how wrong it is, so one grossly mis-estimated relative
    // rotation -- a mismatched pair that survived RANSAC -- pulls every camera
    // connected through it. That is what IRLS below is for.
    //
    // Implemented as iterated local averaging rather than one global solve: each
    // frame is set to the weighted average of what its neighbours predict, then
    // reprojected. This converges to the same fixed point, needs no sparse
    // linear algebra, and lets the same loop serve both methods.
    void SolveL2(int n, const std::vector<Edge>& edges, std::vector<Mat3>* R,
                 int iters) const {
        std::vector<Mat3> next;
        next.resize(size_t(n));
        for (int it = 0; it < iters; ++it) {
            std::vector<double> acc(size_t(n) * 9, 0.0);
            std::vector<double> wsum(size_t(n), 0.0);

            for (const Edge& e : edges) {
                // What edge e says frame j should be, given frame i.
                const Mat3 predJ = e.R * (*R)[size_t(e.i)];
                const Mat3 predI = e.R.Transpose() * (*R)[size_t(e.j)];
                for (int k = 0; k < 9; ++k) {
                    acc[size_t(e.j) * 9 + size_t(k)] += e.weight * predJ.m[k];
                    acc[size_t(e.i) * 9 + size_t(k)] += e.weight * predI.m[k];
                }
                wsum[size_t(e.i)] += e.weight;
                wsum[size_t(e.j)] += e.weight;
            }

            for (int f = 0; f < n; ++f) {
                if (wsum[size_t(f)] < 1e-12) { next[size_t(f)] = (*R)[size_t(f)]; continue; }
                Mat3 avg;
                for (int k = 0; k < 9; ++k)
                    avg.m[k] = acc[size_t(f) * 9 + size_t(k)] / wsum[size_t(f)];
                // The average of rotations is not a rotation; project.
                next[size_t(f)] = NearestRotation(avg);
            }
            *R = next;
        }
    }

    // --- L1-IRLS: the same loop, with edges reweighted by their residual -----
    //
    // Chatterjee & Govindu (ICCV 2013), which is what GLOMAP itself uses.
    //
    // The idea is that a squared cost is the wrong one for data with outliers:
    // an edge wrong by a radian contributes a hundred times what one wrong by a
    // tenth does, so the fit bends toward the worst measurement. Iteratively
    // reweighted least squares replaces it with a robust cost by solving a
    // WEIGHTED least-squares problem repeatedly, where each edge's weight falls
    // as its residual grows.
    //
    // The weight here is Huber-like: full weight inside the threshold, falling
    // as 1/residual outside it. An edge that disagrees with everything else
    // ends up contributing almost nothing, which is the desired behaviour --
    // and note that it is the GRAPH that decides an edge is wrong, not any test
    // on the edge itself. That is why averaging catches outliers a pairwise
    // verification cannot.
    void SolveIrls(int n, std::vector<Edge> edges, std::vector<Mat3>* R,
                   int iters, double sigmaRad) const {
        // Start from L2 so the first residuals mean something. Reweighting from
        // an identity start would judge every edge against a wrong estimate and
        // could down-weight the good ones.
        SolveL2(n, edges, R, 8);

        const std::vector<double> base = [&] {
            std::vector<double> w;
            w.reserve(edges.size());
            for (const Edge& e : edges) w.push_back(e.weight);
            return w;
        }();

        // The tail is SQUARED past the threshold rather than 1/r. Measured on
        // castle-P19 at window 2, a 1/r tail left an edge wrong by 162 degrees
        // -- a relative pose that is essentially backwards -- still carrying 3%
        // of a good edge's weight; squaring takes it to 0.1%, while an edge
        // just outside the threshold is barely touched (20 deg: 25% -> 6%).
        //
        // Graduated rather than a hard cut on purpose: the residual is measured
        // against an estimate that is itself still wrong in the early
        // iterations, so a hard cut would discard good edges before the
        // estimate settles.
        //
        // NOTE, because it is easy to over-credit this: on castle-P19 this
        // changes almost nothing (mean residual 9.10 vs 9.08 deg), and neither
        // does switching to plain L2 (9.21 deg). When averaging cannot improve
        // on its own initialisation, the edges are wrong in BULK rather than
        // individually, and no reweighting can help -- see relative_pose.
        for (int it = 0; it < iters; ++it) {
            for (size_t k = 0; k < edges.size(); ++k) {
                const double r = EdgeResidual(edges[k], *R);
                double scale = 1.0;
                if (r > sigmaRad) {
                    const double t = sigmaRad / r;
                    scale = t * t;
                }
                edges[k].weight = base[k] * scale;
            }
            SolveL2(n, edges, R, 4);
        }
    }

    static constexpr const char* kMethodNames[] = {"L2 (Martinec-Pajdla)",
                                                   "L1-IRLS (Chatterjee-Govindu)"};

    Param<int> m_method{this, "method", 1, 0, 1,
        {.help = "L2 is one linear averaging pass, fast and with no local "
                 "minima, but least squares gives a badly wrong edge as much "
                 "say as a good one. L1-IRLS reweights each edge by how far it "
                 "disagrees with the rest of the graph, so a mismatched pair "
                 "that survived RANSAC stops pulling every camera connected "
                 "through it. IRLS is what GLOMAP uses.",
         .choices = kMethodNames, .choiceCount = 2}};

    Param<int> m_iterations{this, "iterations", 32, 1, 200,
        {.help = "Averaging passes. Convergence is fast on a well-connected "
                 "graph and slow on a long thin one, where information has to "
                 "travel the length of the chain one hop per pass."}};

    Param<float> m_outlierDeg{this, "outlier_deg", 5.0f, 0.5f, 60.0f,
        {.help = "Angular disagreement, in degrees, past which IRLS starts "
                 "discounting an edge. Below it an edge keeps full weight; "
                 "above, its influence falls as 1/residual. Ignored by L2."}};

    std::string m_note;
};

bool RotationAverage::RunReconstruct(const std::vector<Image>* images,
                                     PointCloud* cloud, std::string* err) {
    // FRAMES OR A RECONSTRUCTION, whichever the pipeline handed over.
    //
    // This stage can be first in a chain -- reading relative poses out of the
    // frames' sidecars -- or it can follow build_tracks, in which case the
    // input is a PointCloud and there are no frames at all. Reading the view
    // graph out of the cloud when it is already there is what lets the two
    // orders both work, and it is why PointCloud carries `edges`.
    const bool haveFrames = images && !images->empty();
    const int n = haveFrames ? int(images->size())
                             : int(cloud->cameras.size());
    if (n < 2) {
        *err = "rotation_average needs at least two frames";
        return false;
    }

    // Cameras may already exist from build_tracks; if this stage runs first,
    // create the slots.
    if (int(cloud->cameras.size()) != n) {
        cloud->cameras.assign(size_t(n), Camera{});
        if (haveFrames) {
            for (int f = 0; f < n; ++f) {
                const ImageDesc& d = (*images)[size_t(f)].Desc();
                Camera& c = cloud->cameras[size_t(f)];
                c.width = d.width; c.height = d.height;
                c.cx = 0.5 * d.width; c.cy = 0.5 * d.height;
                c.focal = 0.5 * d.width /
                          std::tan(0.5 * 50.0 * 3.14159265358979 / 180.0);
            }
        }
    }

    // Read the view graph out of the frames when they are here, and record it
    // on the cloud so later stages do not have to.
    if (haveFrames && cloud->edges.empty()) {
        for (int f = 0; f < n; ++f) {
            const RelativePoseSidecar* rp = RelativePosesOf((*images)[size_t(f)]);
            if (!rp) continue;
            for (const auto& e : rp->edges) {
                if (e.reference < 0 || e.reference >= n || e.reference == f) continue;
                PointCloud::ViewEdge ve;
                ve.i = e.reference;
                ve.j = f;
                ve.R = e.R;
                ve.direction = e.direction;
                ve.inliers = e.inliers;
                cloud->edges.push_back(ve);
            }
        }
    }

    std::vector<Edge> edges;
    for (const PointCloud::ViewEdge& ve : cloud->edges) {
        if (ve.i < 0 || ve.i >= n || ve.j < 0 || ve.j >= n) continue;
        Edge ed;
        ed.i = ve.i;
        ed.j = ve.j;
        ed.R = ve.R;
        // Weighted by support. sqrt rather than the raw count: an edge with
        // four hundred inliers is better than one with a hundred, but not
        // four times better -- the marginal value of another correspondence
        // falls off, and a linear weight lets one dense pair dominate a
        // whole neighbourhood.
        ed.weight = std::sqrt(double(std::max(1, ve.inliers)));
        edges.push_back(ed);
    }

    if (edges.empty()) {
        *err = "rotation_average: no relative poses -- run relative_pose first";
        return false;
    }

    std::vector<Mat3> R;
    std::vector<bool> seen;
    if (!SpanningTreeInit(n, edges, &R, &seen)) {
        *err = "rotation_average: the view graph is disconnected at frame 0";
        return false;
    }

    // Residual before, so the report can say whether averaging helped. Measured
    // against the spanning-tree estimate, which is what a chain would produce
    // -- so this is precisely the drift comparison.
    double before = 0.0;
    for (const Edge& e : edges) before += EdgeResidual(e, R);
    before /= double(edges.size());

    const double sigma = double(m_outlierDeg) * 3.14159265358979 / 180.0;
    if (int(m_method) == 0) SolveL2(n, edges, &R, int(m_iterations));
    else                    SolveIrls(n, edges, &R, int(m_iterations), sigma);

    // Gauge: pin frame 0 to the identity. Any global rotation is equally
    // correct (see the header), so a convention is needed and this is the
    // conventional one.
    const Mat3 gauge = R[0].Transpose();
    for (Mat3& r : R) r = r * gauge;

    double after = 0.0, worst = 0.0;
    for (const Edge& e : edges) {
        const double r = EdgeResidual(e, R);
        after += r;
        worst = std::max(worst, r);
    }
    after /= double(edges.size());

    int solvedCount = 0;
    for (int f = 0; f < n; ++f) {
        if (!seen[size_t(f)]) continue;   // unreachable: leave it unsolved
        cloud->cameras[size_t(f)].R = R[size_t(f)];
        cloud->cameras[size_t(f)].solved = true;
        ++solvedCount;
    }

    const double toDeg = 180.0 / 3.14159265358979;
    char buf[320];
    std::snprintf(buf, sizeof buf,
                  "rotation: %s, %d of %d frames from %d edges; mean residual "
                  "%.2f -> %.2f deg, worst %.2f deg",
                  int(m_method) == 0 ? "L2" : "L1-IRLS", solvedCount, n,
                  int(edges.size()), before * toDeg, after * toDeg, worst * toDeg);
    m_note = buf;

    if (solvedCount < 2) {
        *err = "rotation_average: fewer than two frames reachable -- " + m_note;
        return false;
    }
    return true;
}

REGISTER_ALGORITHM(RotationAverage);

}  // namespace
}  // namespace tglab
