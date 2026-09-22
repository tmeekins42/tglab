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
// The error is a NORMALISED DIRECTION DIFFERENCE, bounded to [0, 1]: the angle
// between where a camera says a point is and where the current estimate puts
// it, expressed so that a wildly wrong observation contributes no more than a
// merely bad one. That bound is what lets the solve start from uniform random
// positions -- which it does, in the paper and here -- with no initialisation
// from anything.
//
// SCALE IS NOT RECOVERABLE. The whole reconstruction can be scaled freely and
// every measurement stays satisfied, exactly as the whole thing can be rotated
// freely (see rotation_average). Both are gauge freedoms, both are fixed by
// convention here, and both have to be accounted for in any test.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "../../core/algorithm.h"

namespace tglab {
namespace {

// One camera-to-point observation, resolved into a world-space ray.
struct Ray {
    int  camera = -1;
    int  track = -1;
    Vec3 dir;       // unit, world space: which way the camera saw the point
    double weight = 1.0;
};

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

    Param<int> m_seed{this, "seed", 1, 0, 100000,
        {.help = "Random seed for the initial positions. The joint method "
                 "starts from uniform random -- the GLOMAP formulation is "
                 "bounded so it does not need a good starting point -- and "
                 "exposing the seed makes that claim testable rather than "
                 "merely asserted."}};

    std::string m_note;
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
