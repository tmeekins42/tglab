// fuse_depth — depth maps in, a dense point cloud out.
//
// WHY FUSION IS A SEPARATE STAGE FROM THE SWEEP THAT PRODUCED THE MAPS.
//
// A depth map is a statement about ONE camera: for each of its pixels, how far
// away that pixel is. Turning N of those into a single cloud is a different
// problem, and it is the one where a dense reconstruction is usually won or
// lost, because the maps DISAGREE. The same piece of wall is measured by every
// camera that saw it, each with its own errors, and naively unprojecting all of
// them gives N overlapping copies of the scene -- a cloud N times too large,
// fuzzy to the thickness of the disagreement, with every error preserved.
//
// So this stage does two things the sweep cannot:
//
//   * CONSISTENCY CHECKING. A depth is believed only when other cameras agree
//     with it. That is the single most effective filter available here,
//     because a wrong depth is wrong in one view's private way and rarely
//     lines up with another view's mistake.
//   * MERGING. Agreeing measurements of one surface become ONE point, at
//     their average, rather than a small cloud of near-duplicates.
//
// Keeping it separate from plane_sweep also means PMVS can reuse it unchanged
// and the two dense methods can be compared through identical fusion -- which
// is the comparison worth making, since otherwise a difference in the fuser
// masquerades as a difference in the method.
//
// THE CHECK, concretely. Take pixel p in camera i with depth d. Unproject it
// to a world point X. For every other camera j, project X into j and read j's
// own depth map there. If j says the surface is at roughly the distance X
// actually is from j, then j AGREES -- two independent measurements of the
// same surface. Require `min_views` agreements and X survives.
//
// Three things this deliberately does not do, stated so the gap is visible:
// it does not estimate normals (PMVS will; a point here is a bare position),
// it does not do sub-voxel surface extraction the way TSDF fusion does, and
// it does not detect that a nearer surface OCCLUDES a measurement rather than
// contradicting it -- an occluded pixel simply fails to find support and is
// dropped, which is conservative but loses real geometry at depth edges.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "../../algo_util/view_graph.h"
#include "../../core/algorithm.h"
#include "../../core/parallel.h"

namespace tglab {
namespace {

// A depth map and its confidence, read back out of the ImageSet plane_sweep
// produced. Held as raw planes for the same reason the sweep did it: the inner
// loop runs per pixel per camera and a format switch there would dominate.
struct DepthView {
    const float* depth = nullptr;
    const float* conf  = nullptr;
    // The reference frame's own pixels, RGBA32F, carried through by the sweep
    // so a dense point takes the colour of the pixel it was measured from.
    const float* rgb   = nullptr;
    int w = 0, h = 0;
    bool ok = false;

    Vec3 Colour(int x, int y) const {
        if (!rgb || x < 0 || y < 0 || x >= w || y >= h)
            return Vec3{0.72, 0.72, 0.74};
        const float* p = &rgb[(size_t(y) * size_t(w) + size_t(x)) * 4];
        return Vec3{double(p[0]), double(p[1]), double(p[2])};
    }

    // Depth at an integer pixel, or 0 where nothing was measured. The sweep
    // writes exactly 0.0 for an unmeasured pixel, which is distinguishable
    // from a real depth because a real one is strictly positive.
    float At(int x, int y) const {
        if (x < 0 || y < 0 || x >= w || y >= h) return 0.0f;
        return depth[size_t(y) * size_t(w) + size_t(x)];
    }
    float Conf(int x, int y) const {
        if (!conf || x < 0 || y < 0 || x >= w || y >= h) return 0.0f;
        return conf[size_t(y) * size_t(w) + size_t(x)];
    }
};

class FuseDepth : public AlgorithmBase {
public:
    const char* Name()     const override { return "fuse_depth"; }
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

    // Depth values and camera intrinsics are in IMAGE PIXELS and nothing
    // rescales them, so a proxy run would unproject with the wrong focal.
    ProxyBehaviour Proxy() const override { return ProxyBehaviour::Never; }

    bool RunReconstruct(const std::vector<Image>* images, PointCloud* cloud,
                        std::string* err) override {
        if (!cloud) { *err = "fuse_depth: no reconstruction"; return false; }
        if (!images || images->empty()) {
            *err = "fuse_depth: needs the depth maps as its second input -- "
                   "fuse_depth(cloud, depth)";
            return false;
        }

        const int nCam = int(cloud->cameras.size());
        if (nCam < 2) {
            *err = "fuse_depth: fewer than two cameras";
            return false;
        }

        // plane_sweep emits depth and confidence INTERLEAVED, two images per
        // frame. Checked rather than assumed: handing this the colour frames
        // by mistake is an easy error and produces nonsense silently.
        if (int(images->size()) != nCam * 3) {
            *err = "fuse_depth: expected " + std::to_string(nCam * 3) +
                   " images (a depth, confidence and colour map per camera), "
                   "got " + std::to_string(images->size()) + " -- the second "
                   "input should be plane_sweep's output";
            return false;
        }

        std::vector<DepthView> views;
        views.resize(size_t(nCam));
        std::vector<ImageView> holds;      // keep the mappings alive
        holds.reserve(size_t(nCam) * 2);

        for (int c = 0; c < nCam; ++c) {
            ImageView dv =
                const_cast<Image&>((*images)[size_t(c) * 3 + 0]).MapCpuRead();
            ImageView cv =
                const_cast<Image&>((*images)[size_t(c) * 3 + 1]).MapCpuRead();
            ImageView rv =
                const_cast<Image&>((*images)[size_t(c) * 3 + 2]).MapCpuRead();
            holds.push_back(dv);
            holds.push_back(cv);
            holds.push_back(rv);

            if (!dv.Valid() || dv.desc.format != Format::R32F) continue;
            if (!dv.desc.isDepth) {
                *err = "fuse_depth: input 1 image " +
                       std::to_string(c * 3) + " is not a depth map -- the "
                       "second input should be plane_sweep's output";
                return false;
            }

            DepthView& v = views[size_t(c)];
            v.depth = dv.At<float>(0, 0);
            v.w = dv.desc.width;
            v.h = dv.desc.height;
            v.conf = (cv.Valid() && cv.desc.format == Format::R32F)
                         ? cv.At<float>(0, 0) : nullptr;
            v.rgb = (rv.Valid() && rv.desc.format == Format::RGBA32F)
                        ? rv.At<float>(0, 0) : nullptr;
            v.ok = true;
        }

        // COLOUR COMES FROM THE SWEEP'S THIRD MAP, one RGBA plane per frame
        // carrying that frame's own pixels. A dense point therefore takes the
        // colour of the exact pixel it was measured from.
        //
        // The first version instead gave each dense point the colour of the
        // nearest SPARSE point, reasoning that at this density they would be
        // on the same surface. That was wrong in a way worth recording: 1.2M
        // dense points against 3300 sparse ones means ~360 dense points share
        // every sample, so the cloud rendered as a few thousand FLAT PATCHES
        // -- each one a Voronoi cell of the sparse cloud. It read as a shading
        // artefact and was really a resolution mismatch of three orders of
        // magnitude.
        const int    minViews = std::max(1, int(m_minViews));
        const double relTol   = double(m_tolerance);
        const int    step     = std::max(1, int(m_step));
        const double minConf  = double(m_minConfidence);

        // Fused points, gathered per camera then concatenated. Per-camera
        // vectors rather than one shared: the cameras run in parallel and
        // this needs no lock.
        std::vector<std::vector<Track>> perCam;
        perCam.resize(size_t(nCam));

        std::vector<long long> perTested(size_t(nCam), 0);
        std::vector<long long> perKept(size_t(nCam), 0);

        ParallelFor(nCam, [&](int c) {
            FuseCamera(c, *cloud, views, minViews, relTol, step, minConf,
                       &perCam[size_t(c)], &perTested[size_t(c)],
                       &perKept[size_t(c)]);
        });

        long long tested = 0, kept = 0;
        size_t total = 0;
        for (int c = 0; c < nCam; ++c) {
            tested += perTested[size_t(c)];
            kept   += perKept[size_t(c)];
            total  += perCam[size_t(c)].size();
        }

        // REPLACE the tracks rather than append to them. The sparse points are
        // the same surfaces measured a second, sparser way, and keeping both
        // would double-count every one of them -- the cloud would read as
        // denser than it is and the duplicates would sit a little apart,
        // exactly the fuzz this stage exists to remove.
        //
        // The cameras and the view graph stay: they are the reconstruction,
        // and a viewer still wants to draw them.
        const int sparseBefore = cloud->TriangulatedPoints();
        cloud->tracks.clear();
        cloud->tracks.reserve(total);
        for (int c = 0; c < nCam; ++c)
            for (Track& t : perCam[size_t(c)])
                cloud->tracks.push_back(std::move(t));

        char buf[360];
        std::snprintf(buf, sizeof(buf),
                      "fuse_depth: %lld of %lld depth samples survived "
                      "%d-view agreement (%.1f%%); %d dense points from %d "
                      "sparse, a %.0fx increase",
                      kept, tested, minViews,
                      tested > 0 ? 100.0 * double(kept) / double(tested) : 0.0,
                      int(cloud->tracks.size()), sparseBefore,
                      sparseBefore > 0
                          ? double(cloud->tracks.size()) / double(sparseBefore)
                          : 0.0);
        m_note = buf;
        return true;
    }

    std::string RunReport() const override { return m_note; }
    bool HasGPU() const override { return false; }

private:
    void FuseCamera(int ci, const PointCloud& cloud,
                    const std::vector<DepthView>& views, int minViews,
                    double relTol, int step, double minConf,
                    std::vector<Track>* out, long long* tested,
                    long long* kept) const {
        const Camera& ref = cloud.cameras[size_t(ci)];
        const DepthView& rv = views[size_t(ci)];
        if (!ref.solved || !rv.ok) return;

        const int nCam = int(cloud.cameras.size());
        long long nTested = 0, nKept = 0;

        for (int y = 0; y < rv.h; y += step) {
            for (int x = 0; x < rv.w; x += step) {
                const float d = rv.At(x, y);
                if (d <= 0.0f) continue;              // unmeasured
                if (rv.Conf(x, y) < float(minConf)) continue;
                ++nTested;

                // Unproject: the ray through this pixel, at this depth.
                const Vec3 world = Unproject(ref, double(x), double(y),
                                             double(d));

                // HOW MANY OTHER CAMERAS AGREE. Each one is asked the same
                // question: project the point into you, and does your own
                // depth map say the surface is there?
                int agree = 0;
                double sumD = double(d);
                int    sumN = 1;

                for (int cj = 0; cj < nCam && agree < minViews; ++cj) {
                    if (cj == ci) continue;
                    const Camera& oc = cloud.cameras[size_t(cj)];
                    const DepthView& ov = views[size_t(cj)];
                    if (!oc.solved || !ov.ok) continue;

                    // Where the point sits relative to that camera.
                    const Vec3 local = oc.R * world + oc.t;
                    if (local.z <= 1e-9) continue;     // behind it

                    double px = 0.0, py = 0.0;
                    if (!oc.Project(world, &px, &py)) continue;
                    const int ix = int(px + 0.5), iy = int(py + 0.5);
                    if (ix < 0 || iy < 0 || ix >= ov.w || iy >= ov.h) continue;

                    const float od = ov.At(ix, iy);
                    if (od <= 0.0f) continue;          // it measured nothing

                    // AGREEMENT IS RELATIVE, not absolute. Depth uncertainty
                    // grows with distance -- the same inverse-depth argument
                    // that decides how the sweep spaces its planes -- so a
                    // fixed tolerance would be far too strict far away and
                    // far too loose up close.
                    const double diff = std::fabs(double(od) - local.z);
                    if (diff / local.z > relTol) continue;

                    ++agree;
                    // Averaged over the agreeing views, which is the other
                    // half of what fusion is for: several noisy measurements
                    // of one surface make a better estimate than any of them.
                    sumD += double(od) * (double(d) / local.z);
                    ++sumN;
                }

                if (agree < minViews) continue;

                // Re-unproject at the averaged depth.
                const double fused = sumD / double(sumN);
                const Vec3 p = Unproject(ref, double(x), double(y), fused);

                Track t;
                t.point = p;
                t.hasPoint = true;
                t.color = rv.Colour(x, y);

                // One observation, naming where this point was measured. Not a
                // real track -- nothing was matched to produce it -- but the
                // field is what a viewer and an exporter read, and leaving it
                // empty would make the point look like a failed track.
                Observation o;
                o.frame = ci;
                o.x = float(x);
                o.y = float(y);
                t.obs.push_back(o);

                out->push_back(std::move(t));
                ++nKept;
            }
        }

        *tested = nTested;
        *kept   = nKept;
    }

    static Vec3 Unproject(const Camera& c, double px, double py, double depth) {
        const double f = (std::fabs(c.focal) < 1e-9) ? 1e-9 : c.focal;
        const Vec3 cam{(px - c.cx) / f * depth, (py - c.cy) / f * depth, depth};
        // Camera to world: X = R^T (x_cam - t).
        return c.R.Transpose() * (cam - c.t);
    }

    // HOW MANY OTHER CAMERAS MUST AGREE, and the main quality/quantity dial.
    //
    // 1 is already a real filter -- it means some other view independently
    // measured this surface here -- and 2 is substantially stricter because a
    // coincidence has to happen twice.
    Param<int> m_minViews{this, "min_views", 2, 1, 16,
        {.help = "How many OTHER cameras must independently measure a surface "
                 "at the same place before the point is kept. The single most "
                 "effective filter here: a wrong depth is wrong in one view's "
                 "own way and rarely agrees with another view's mistake."}};

    // HOW CLOSELY THEY MUST AGREE, as a fraction of the distance rather than
    // an absolute, because depth uncertainty scales with depth.
    Param<float> m_tolerance{this, "tolerance", 0.01f, 0.001f, 0.2f,
        {.help = "How close another camera's depth must be to count as "
                 "agreement, as a FRACTION of the distance -- 0.01 is one "
                 "percent. Relative rather than absolute because depth "
                 "uncertainty grows with distance.",
         .step = 0.005, .softMax = 0.05}};

    // PIXEL STRIDE. A depth map has as many samples as pixels, and a 1.0 MP
    // frame times eleven cameras is eleven million candidate points before any
    // filtering -- far more than the structure justifies and more than the
    // viewer wants to draw.
    Param<int> m_step{this, "step", 2, 1, 16,
        {.help = "Take every Nth pixel of each depth map. 1 uses them all, "
                 "which is a great many points for little extra detail: "
                 "neighbouring pixels of a depth map are measurements of the "
                 "same surface, not independent samples of it."}};

    // MEASURED on fountain-P11, and the pattern is the useful part:
    //
    //   min_confidence   samples   agreement rate   dense points
    //             0.00    688989            64.1%         441626
    //             0.15    324358            86.1%         279429
    //             0.30    127260            93.4%         118853
    //
    // The agreement rate CLIMBS as confidence rises, and those are two
    // independent measures -- one is how distinguishable the depth was within
    // a single view's cost curve, the other is whether other cameras put the
    // surface in the same place. That they corroborate each other says the
    // sweep's confidence is a real signal rather than a number that happens
    // to exist.
    //
    // RAISE THIS WHEN A SURFACE COMES OUT IN THE WRONG PLACE. A wall that
    // reconstructs behind where it should be is the sweep picking a competing
    // peak on repetitive texture -- brick matched to the wrong course -- and
    // such a pixel has a small margin by construction. 0.15 to 0.3 removes
    // most of it, at a real cost in coverage.
    //
    // Left at 0 by default because multi-view agreement is the stronger
    // check and should be allowed to do its job first; this is the dial to
    // reach for when it is not enough.
    Param<float> m_minConfidence{this, "min_confidence", 0.0f, 0.0f, 1.0f,
        {.help = "Lowest sweep confidence a sample may have and still be "
                 "considered. Zero accepts everything the sweep measured and "
                 "leans on multi-view agreement instead. Raise it to 0.15-0.3 "
                 "when a surface reconstructs in the wrong place: that is the "
                 "sweep picking a competing peak on repetitive texture, and "
                 "such a pixel has a small margin by construction.",
         .step = 0.05}};

    std::string m_note;
};

REGISTER_ALGORITHM(FuseDepth);

}  // namespace
}  // namespace tglab
