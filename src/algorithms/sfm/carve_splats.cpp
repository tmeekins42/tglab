// carve_splats — remove the Gaussians that cameras can see through.
//
// WHAT IT IS FOR. Training fits Gaussians to the photographs, and a
// photograph only constrains what it sees. Where a surface appears in just a
// few views and has no geometry of its own -- on fountain-P11, a textureless
// pale wall at the right, seen from the left-hand cameras -- the optimiser is
// free to paint it with whatever Gaussians happen to project there. From
// those few views the result is right; from anywhere else it is a cloud of
// pale blobs hanging in the air beside the fountain. Opacity and size
// pruning leave them alone, because to the views that made them they are
// doing useful work.
//
// THE TEST is free-space carving, the oldest idea in multi-view
// reconstruction: if a camera measured a surface BEHIND a point, along its
// own line of sight, then it looked straight through the place that point
// claims is solid, so nothing solid is there. One camera saying so can be a
// depth error; several saying so is evidence. The depth maps are the
// plane sweep's, so a camera's opinion is independent of the training that
// made the floater in the first place.
//
// Only a Gaussian's centre is tested, not its footprint. A large Gaussian
// straddling a depth edge could be judged by the wrong side of it; small
// ones, which is nearly all of them, cannot. (Testing the tips of the longest
// axis as well was tried: these floaters are not needles -- longest axis
// 1.2x the next on average -- and it mostly removed more real surface.)
//
// MEASURED on fountain-P11, judged by rendering from 30 and 60 degrees off
// the middle camera (bench_sfm --dump renders those views):
//
//   max_seen_through   removed    floaters            real surface
//                  1    14787     most gone           untouched
//                  0    61512     slightly fewer      holes in the basin
//
// At 0 one camera's depth noise is enough to condemn a Gaussian, and it
// shows. What 1 leaves are a few large blobs where NO camera measured
// anything behind them -- dark background, so nothing to see through. Free
// space that nobody measured cannot be carved; that needs another signal.
#include <algorithm>
#include <array>
#include <cstdio>
#include <string>
#include <vector>

#include "../../algo_util/depth_views.h"
#include "../../core/algorithm.h"
#include "../../core/parallel.h"

namespace tglab {
namespace {

// Histogram buckets: 0, 1, 2, 3, 4 and "5 or more" cameras seeing through.
constexpr int kHist = 6;

class CarveSplats : public AlgorithmBase {
public:
    const char* Name()     const override { return "carve_splats"; }
    const char* Category() const override { return "sfm"; }

    PortList Inputs() const override {
        return {{"src",   DataType::PointCloud, FormatSpec::Any, ShapeSpec::Any},
                {"depth", DataType::ImageSet,   FormatSpec::Any, ShapeSpec::Any}};
    }
    PortList Outputs() const override {
        return {{"out", DataType::PointCloud, FormatSpec::Any, ShapeSpec::Any}};
    }

    void RunCPU(RunCtx&) override {}
    bool IsReconstruct() const override { return true; }
    // Depths and intrinsics are in image pixels; nothing rescales them.
    ProxyBehaviour Proxy() const override { return ProxyBehaviour::Never; }
    bool HasGPU() const override { return false; }

    bool RunReconstruct(const std::vector<Image>* images, PointCloud* cloud,
                        std::string* err) override {
        if (!cloud) { *err = "carve_splats: no reconstruction"; return false; }
        if (cloud->splats.empty()) {
            *err = "carve_splats: no Gaussians -- run init_splats (and "
                   "train_splats) first";
            return false;
        }
        if (!images || images->empty()) {
            *err = "carve_splats: needs the depth maps as its second input -- "
                   "carve_splats(splats, depth)";
            return false;
        }
        const int nCam = int(cloud->cameras.size());
        std::vector<DepthView> views;
        std::vector<ImageView> holds;      // keep the mappings alive
        if (!ReadDepthViews(*images, nCam, "carve_splats", &views, &holds, err))
            return false;

        const int    maxSeen = int(m_maxSeenThrough);
        const double seenFrac = double(m_maxSeenFraction);
        const double tol     = double(m_tolerance);
        const double minConf = double(m_minConfidence);

        // The scene's scale, measured as train_splats measures it -- the
        // cameras' radius about their centre, times 1.1 -- so `large_size`
        // means the same fraction here as max_world_size does there.
        double extent = 0.0;
        {
            Vec3 mean{0, 0, 0};
            int nc = 0;
            for (const Camera& c : cloud->cameras)
                if (c.solved) { mean = mean + c.Center(); ++nc; }
            mean = mean * (1.0 / double(std::max(1, nc)));
            for (const Camera& c : cloud->cameras)
                if (c.solved) extent = std::max(extent, (c.Center() - mean).Norm());
            extent = std::max(1e-6, extent * 1.1);
        }
        const double largeLimit = double(m_largeSize) * extent;
        const int    minViews   = int(m_largeMinViews);

        const size_t n = cloud->splats.size();
        std::vector<int> seen(n, 0), inView(n, 0), measured(n, 0);
        const size_t chunk = 4096;
        ParallelFor((n + chunk - 1) / chunk, [&](size_t c) {
            const size_t end = std::min(n, (c + 1) * chunk);
            for (size_t i = c * chunk; i < end; ++i) {
                const Vec3& m = cloud->splats[i].mean;
                seen[i] = CountSeenThrough(*cloud, views, m, -1, minConf, tol, &measured[i]);
                int v = 0;
                for (const Camera& cam : cloud->cameras) {
                    if (!cam.solved) continue;
                    if ((cam.R * m + cam.t).z <= 1e-9) continue;
                    double px = 0.0, py = 0.0;
                    if (!cam.Project(m, &px, &py)) continue;
                    if (px >= 0.0 && py >= 0.0 && px < cam.width && py < cam.height) ++v;
                }
                inView[i] = v;
            }
        });

        std::array<long long, kHist> hist{};
        size_t kept = 0, thinLarge = 0, seenDropped = 0;
        for (size_t i = 0; i < n; ++i) {
            ++hist[size_t(std::min(seen[i], kHist - 1))];
            if (seen[i] > maxSeen && double(seen[i]) >= seenFrac * double(measured[i])) {
                ++seenDropped;
                continue;
            }
            const Splat& s = cloud->splats[i];
            if (minViews > 0 && inView[i] < minViews &&
                std::max({s.scale.x, s.scale.y, s.scale.z}) > largeLimit) {
                ++thinLarge;
                continue;
            }
            // The view-dependent colour moves with its Gaussian.
            if (cloud->shDegree > 0 && cloud->splatSh.size() >= (i + 1) * 45)
                std::copy_n(cloud->splatSh.begin() + long(i * 45), 45,
                            cloud->splatSh.begin() + long(kept * 45));
            // ...and so does its reflectivity and normal.
            if (cloud->splatRefl.size() >= (i + 1) * 4)
                std::copy_n(cloud->splatRefl.begin() + long(i * 4), 4,
                            cloud->splatRefl.begin() + long(kept * 4));
            cloud->splats[kept++] = cloud->splats[i];
        }
        cloud->splats.resize(kept);
        if (cloud->shDegree > 0) cloud->splatSh.resize(kept * 45);
        if (!cloud->splatRefl.empty()) cloud->splatRefl.resize(kept * 4);

        char buf[420];
        std::snprintf(buf, sizeof(buf),
                      "carve_splats: removed %zu of %zu Gaussians -- %zu seen "
                      "through by more than %d camera%s and %.0f%% of those that "
                      "measured there, %zu large and in view "
                      "of fewer than %d; cameras seeing through each: 0: %lld, "
                      "1: %lld, 2: %lld, 3: %lld, 4: %lld, 5+: %lld",
                      n - kept, n, seenDropped, maxSeen,
                      maxSeen == 1 ? "" : "s", seenFrac * 100.0, thinLarge, minViews, hist[0],
                      hist[1], hist[2], hist[3], hist[4], hist[5]);
        m_note = buf;
        return true;
    }

    std::string RunReport() const override { return m_note; }

private:
    // AND A FRACTION, because a count alone means different things at
    // different camera counts. With eleven cameras, two seeing through a
    // point is a real signal; with a hundred, a point measured by fifty of
    // them collects two such votes from depth noise alone -- on a 100-frame
    // video it removed two thirds of the Gaussians, holes and all. Requiring
    // the dissenters to be a fifth of the cameras that measured there too
    // leaves an eleven-camera capture judged as before and makes a hundred-
    // camera one need about ten.
    Param<float> m_maxSeenFraction{this, "max_seen_fraction", 0.2f, 0.0f, 1.0f,
        {.help = "A Gaussian is removed only when the cameras seeing through "
                 "it are ALSO at least this fraction of the cameras that "
                 "measured a depth there. Keeps a capture with many cameras "
                 "from losing real surface to depth noise. 0 uses the count "
                 "alone.",
         .step = 0.05}};

    Param<int> m_maxSeenThrough{this, "max_seen_through", 1, 0, 16,
        {.help = "Remove a Gaussian when more than this many cameras measured "
                 "a surface clearly BEHIND it along their own line of sight: "
                 "they looked through where it claims to be. One camera can be "
                 "a depth error, so 1 tolerates a single dissent; 0 is the "
                 "strictest. 16 removes nothing."}};

    Param<float> m_tolerance{this, "tolerance", 0.05f, 0.005f, 0.5f,
        {.help = "How far behind the Gaussian a camera's measured surface must "
                 "be, as a fraction of the distance, before that camera counts "
                 "as seeing through it. Too tight and a Gaussian sitting on a "
                 "surface is judged by that surface's own depth noise.",
         .step = 0.01}};

    Param<float> m_minConfidence{this, "min_confidence", 0.0f, 0.0f, 1.0f,
        {.help = "Lowest sweep confidence a camera's depth may have and still "
                 "vote. Raise it if real geometry is being removed because of "
                 "bad depths behind it.",
         .step = 0.05}};

    // LARGE AND BARELY SEEN. The second kind of junk, and not a free-space
    // one: at the edges of the capture a region is in view of only one or two
    // cameras, and there nothing constrains a Gaussian's size or depth -- a
    // smear across a blank wall costs nothing in the one photograph that sees
    // it. Measured on fountain-P11 after 3000 iterations: 1.5% of all
    // Gaussians are in view of three cameras or fewer, but 49% of those with
    // an axis over 2% of the scene extent are.
    //
    // Small Gaussians there are left alone: they are fine detail the few
    // views do support. Only size is condemned by a lack of views.
    Param<float> m_largeSize{this, "large_size", 0.02f, 0.001f, 0.2f,
        {.help = "What counts as LARGE for large_min_views: a longest axis "
                 "above this fraction of the scene extent (the cameras' "
                 "spread, as train_splats measures it).",
         .step = 0.005, .softMax = 0.1}};

    Param<int> m_largeMinViews{this, "large_min_views", 4, 0, 16,
        {.help = "Remove a LARGE Gaussian in view of fewer than this many "
                 "cameras. At the edges of a capture, where one or two views "
                 "see a region, nothing stops a Gaussian smearing across it. "
                 "0 disables."}};

    std::string m_note;
};

REGISTER_ALGORITHM(CarveSplats);

}  // namespace
}  // namespace tglab
