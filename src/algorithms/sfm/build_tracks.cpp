// build_tracks — pairwise matches into feature tracks.
//
// A TRACK is one 3D point and every image that saw it: frame 2's keypoint 41
// and frame 3's keypoint 17 and frame 4's keypoint 88 are all the same corner
// of the same building. Matching produces PAIRS; a reconstruction needs the
// transitive closure of them.
//
// WHY THIS IS A STAGE AT ALL, since it is not obvious from the outside.
//
// Incremental SfM (Bundler, COLMAP's default mapper) has no track-building
// step. A track emerges there as a by-product: a 3D point is triangulated from
// two views, and each newly registered image that matches it adds an
// observation. The track IS the point, accumulated over time.
//
// Global SfM cannot work that way. Rotation averaging and global positioning
// solve for all cameras at once, so they need the entire correspondence graph
// BEFORE anything is solved. Track building is therefore explicit, mandatory,
// and first -- which is why it gets its own stage here rather than living
// inside a solver.
//
// THE ALGORITHM is union-find over the matches (Moulon & Monasse, "Unordered
// Feature Tracking Made Fast and Easy", CVMP 2012). Each (frame, keypoint) is
// a node; each match unions two nodes; each resulting set is a track.
//
// Union-find rather than the graph traversal Bundler used, for two reasons the
// paper measures: it is O(n·α(n)) rather than O(n log n) -- roughly 1.8x faster
// in the median -- and it is ORDER-INDEPENDENT. Bundler's produced different
// tracks depending on the order matches were visited, which makes a
// reconstruction irreproducible for no reason anyone wanted.
//
// WHERE IMPLEMENTATIONS ACTUALLY DIFFER is not the union-find, which is
// settled, but the filtering around it. Those are the parameters below.
#include <algorithm>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../algo_util/features.h"
#include "../../algo_util/view_graph.h"
#include "../../core/algorithm.h"

namespace tglab {
namespace {

// Union-find over (frame, keypoint) pairs, with path compression and union by
// size. Nodes are dense indices; the caller maps them to and from pairs.
class UnionFind {
public:
    explicit UnionFind(size_t n) : m_parent(n), m_size(n, 1) {
        for (size_t i = 0; i < n; ++i) m_parent[i] = i;
    }

    // Iterative rather than recursive: a track spanning hundreds of frames is
    // a chain hundreds deep, and a recursive Find would risk the stack on
    // exactly the large reconstructions this is for.
    size_t Find(size_t x) {
        size_t root = x;
        while (m_parent[root] != root) root = m_parent[root];
        while (m_parent[x] != root) {
            const size_t next = m_parent[x];
            m_parent[x] = root;
            x = next;
        }
        return root;
    }

    void Union(size_t a, size_t b) {
        a = Find(a);
        b = Find(b);
        if (a == b) return;
        if (m_size[a] < m_size[b]) std::swap(a, b);
        m_parent[b] = a;
        m_size[a] += m_size[b];
    }

private:
    std::vector<size_t> m_parent;
    std::vector<size_t> m_size;
};

class BuildTracks : public AlgorithmBase {
public:
    const char* Name()     const override { return "build_tracks"; }
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
                        std::string* err) override {
        if (!images || images->empty()) {
            *err = "build_tracks needs a group of images";
            return false;
        }
        const int n = int(images->size());

        // Node numbering: frame `f`'s keypoint `k` is node base[f] + k. A flat
        // index rather than a map, because union-find is the hot loop and a
        // hash lookup per operation would dominate it.
        std::vector<size_t> base(size_t(n) + 1, 0);
        for (int f = 0; f < n; ++f) {
            const FeatureSidecar* fs = FeaturesOf((*images)[size_t(f)]);
            base[size_t(f) + 1] =
                base[size_t(f)] + (fs ? fs->keypoints.size() : 0);
        }
        const size_t total = base[size_t(n)];
        if (total == 0) {
            *err = "build_tracks: no features on any frame -- run a detector first";
            return false;
        }

        UnionFind uf(total);

        // --- union every verified match ------------------------------------
        int usedMatches = 0, skippedOutliers = 0, pairs = 0;
        for (int f = 0; f < n; ++f) {
            const MatchSidecar* ms = MatchesOf((*images)[size_t(f)]);
            if (!ms) continue;
            for (const MatchSet& set : ms->sets) {
                if (set.reference < 0 || set.reference >= n) continue;
                ++pairs;
                for (size_t i = 0; i < set.matches.size(); ++i) {
                    // INLIERS ONLY. The ratio test and cross-check ask whether
                    // two patches look alike; RANSAC asks whether a match is
                    // consistent with the geometry everything else agrees on,
                    // and it is the only one that catches two genuinely
                    // similar patches in genuinely different places. A
                    // repeating texture -- a row of windows -- produces
                    // exactly that, and one such match merges two unrelated
                    // tracks permanently.
                    if (!set.IsInlier(i)) { ++skippedOutliers; continue; }

                    const Match& m = set.matches[i];
                    const size_t a = base[size_t(set.reference)] + size_t(m.a);
                    const size_t b = base[size_t(f)] + size_t(m.b);
                    if (a >= total || b >= total) continue;   // stale indices
                    uf.Union(a, b);
                    ++usedMatches;
                }
            }
        }

        if (pairs == 0) {
            *err = "build_tracks: no matches on any frame -- run a matcher first";
            return false;
        }

        // --- gather the sets into tracks -----------------------------------
        std::unordered_map<size_t, size_t> rootToTrack;
        std::vector<Track> tracks;
        for (int f = 0; f < n; ++f) {
            const FeatureSidecar* fs = FeaturesOf((*images)[size_t(f)]);
            if (!fs) continue;
            for (size_t k = 0; k < fs->keypoints.size(); ++k) {
                const size_t node = base[size_t(f)] + k;
                const size_t root = uf.Find(node);

                // A singleton is a keypoint nothing matched. Skipping it here
                // rather than filtering later keeps the track list to real
                // correspondences; a length-1 "track" constrains nothing.
                auto it = rootToTrack.find(root);
                if (it == rootToTrack.end()) {
                    rootToTrack.emplace(root, tracks.size());
                    tracks.push_back(Track{});
                    it = rootToTrack.find(root);
                }
                // The pixel position travels WITH the observation: after this
                // stage the pipeline passes a PointCloud and the frames are
                // gone, so a later stage cannot look it up. See Observation.
                Observation ob;
                ob.frame = f;
                ob.keypoint = int(k);
                ob.x = fs->keypoints[k].x;
                ob.y = fs->keypoints[k].y;
                tracks[it->second].obs.push_back(ob);
            }
        }

        // --- filter ---------------------------------------------------------
        const int minLen = std::max(2, int(m_minLength));
        int droppedShort = 0, droppedConflict = 0;

        std::vector<Track> kept;
        kept.reserve(tracks.size());
        for (Track& t : tracks) {
            if (t.Length() < minLen) { ++droppedShort; continue; }

            // A CONFLICTING track sees the same frame twice, which is
            // physically impossible: one 3D point projects to one place in one
            // image. It means union-find merged two distinct points, almost
            // always through a repeated texture that survived RANSAC in two
            // different pairs.
            //
            // Dropped rather than split, because the two halves cannot be told
            // apart -- the merge destroyed the information that would separate
            // them. OpenMVG drops them too. Keeping one would put a wrong
            // observation into every solver downstream, and a wrong
            // correspondence is worse than a missing one: it pulls the fit
            // rather than merely failing to constrain it.
            std::sort(t.obs.begin(), t.obs.end(),
                      [](const Observation& a, const Observation& b) {
                          return a.frame < b.frame;
                      });
            bool conflict = false;
            for (size_t i = 1; i < t.obs.size(); ++i)
                if (t.obs[i].frame == t.obs[i - 1].frame) { conflict = true; break; }
            if (conflict && m_dropConflicts) { ++droppedConflict; continue; }

            kept.push_back(std::move(t));
        }

        // Longest first, so a consumer that caps the count keeps the
        // best-constrained points. A track seen by ten frames is worth more to
        // every downstream stage than five seen by two.
        std::sort(kept.begin(), kept.end(),
                  [](const Track& a, const Track& b) {
                      return a.obs.size() > b.obs.size();
                  });

        const int capped = int(m_maxTracks);
        int droppedCap = 0;
        if (capped > 0 && int(kept.size()) > capped) {
            droppedCap = int(kept.size()) - capped;
            kept.resize(size_t(capped));
        }

        // --- fill in the cameras --------------------------------------------
        //
        // Unsolved, with intrinsics guessed from the frame size. Nothing here
        // solves for pose -- that is rotation averaging and positioning -- but
        // the cloud carries one camera per frame from the start so later stages
        // fill slots rather than resizing.
        cloud->cameras.assign(size_t(n), Camera{});
        for (int f = 0; f < n; ++f) {
            const ImageDesc& d = (*images)[size_t(f)].Desc();
            Camera& c = cloud->cameras[size_t(f)];
            c.width = d.width;
            c.height = d.height;
            c.cx = 0.5 * d.width;
            c.cy = 0.5 * d.height;
            // A 50-degree horizontal field of view, which is an ordinary lens.
            // Only a starting point: the view graph refines it, and bundle
            // adjustment solves it. Stated rather than left at 1.0 so a
            // reprojection before any solve is merely wrong rather than absurd.
            c.focal = 0.5 * d.width / std::tan(0.5 * 50.0 * 3.14159265358979 / 180.0);
        }

        // The view graph, if relative_pose has already run. Carried on the
        // cloud so a later stage does not need the frames back -- see
        // PointCloud::edges. Harmless when absent: rotation_average reads them
        // out of the frames itself when it is the first stage in the chain.
        if (cloud->edges.empty()) {
            for (int f = 0; f < n; ++f) {
                const RelativePoseSidecar* rp = RelativePosesOf((*images)[size_t(f)]);
                if (!rp) continue;
                for (const auto& e : rp->edges) {
                    if (e.reference < 0 || e.reference >= n || e.reference == f)
                        continue;
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

        // --- colour, sampled from the frames that saw each point -----------
        //
        // HERE BECAUSE THIS IS THE LAST STAGE THAT HAS THE FRAMES. After this
        // the pipeline passes a PointCloud and the pixels are gone, so a
        // viewer that wanted colour later would have no way to get it.
        //
        // Worth having beyond looking nice: a grey cloud shows shape and
        // nothing else, while a coloured one shows whether the shape is the
        // RIGHT one. A wall that comes out the colour of the sky, or a facade
        // whose windows land on the lawn, is obvious in colour and invisible
        // in grey.
        //
        // Averaged over every observation rather than taken from the first:
        // one frame may have caught the point in shadow or at a grazing angle,
        // and the mean is what the point actually looks like.
        //
        // Read straight from the ImageView rather than through PixelBuffer,
        // which unpacks a WHOLE frame to float -- the floor that costs 1483 ms
        // at 22 MP. Sampling a few thousand pixels must not pay it.
        {
            std::vector<ImageView> views;
            views.resize(size_t(n));
            for (int f = 0; f < n; ++f)
                views[size_t(f)] = const_cast<Image&>((*images)[size_t(f)]).MapCpuRead();

            for (Track& t : kept) {
                double acc[3] = {0, 0, 0};
                int counted = 0;
                for (const Observation& o : t.obs) {
                    if (o.frame < 0 || o.frame >= n) continue;
                    const ImageView& v = views[size_t(o.frame)];
                    if (!v.Valid()) continue;
                    const int px = int(o.x), py = int(o.y);
                    if (px < 0 || py < 0 ||
                        px >= v.desc.width || py >= v.desc.height) continue;

                    float rgb[3] = {0, 0, 0};
                    switch (v.desc.format) {
                        case Format::RGBA8: {
                            const uint8_t* p = v.At<uint8_t>(px, py);
                            for (int c = 0; c < 3; ++c) rgb[c] = float(p[c]) / 255.0f;
                            break;
                        }
                        case Format::RGBA32F: {
                            const float* p = v.At<float>(px, py);
                            for (int c = 0; c < 3; ++c) rgb[c] = p[c];
                            break;
                        }
                        case Format::RGBA16F: {
                            const uint16_t* p = v.At<uint16_t>(px, py);
                            for (int c = 0; c < 3; ++c) rgb[c] = HalfToFloat(p[c]);
                            break;
                        }
                        default: continue;   // R32F: no colour to sample
                    }
                    for (int c = 0; c < 3; ++c) acc[c] += double(rgb[c]);
                    ++counted;
                }
                if (counted > 0)
                    t.color = Vec3{acc[0] / counted, acc[1] / counted,
                                   acc[2] / counted};
            }
        }

        cloud->tracks = std::move(kept);

        int totalObs = 0, longest = 0;
        // A MEAN HIDES THE SHAPE, and the shape is the diagnosis. Mean 2.2 is
        // equally consistent with "every track is a bare pair plus a handful
        // of long ones" and with "most tracks chain a little" -- and those
        // call for opposite fixes. Measured on castle-P19 it was the former:
        // 87% of tracks were length 2, so only redundancy-building work
        // (re-matching against existing tracks) can move it, not a filter.
        int lenHist[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (const Track& t : cloud->tracks) {
            totalObs += t.Length();
            longest = std::max(longest, t.Length());
            lenHist[std::min(t.Length(), 7)]++;
        }

        const int nTracks = int(cloud->tracks.size());
        char buf[480];
        std::snprintf(buf, sizeof buf,
                      "tracks: %d from %d pairs (%d inlier matches, %d outliers "
                      "skipped); %d obs, mean length %.1f, longest %d; len2 %d "
                      "(%.0f%%) len3 %d len4 %d len5+ %d; dropped "
                      "%d short, %d conflicting%s",
                      nTracks, pairs, usedMatches,
                      skippedOutliers, totalObs,
                      cloud->tracks.empty()
                          ? 0.0
                          : double(totalObs) / double(nTracks),
                      longest,
                      lenHist[2],
                      nTracks ? 100.0 * double(lenHist[2]) / double(nTracks) : 0.0,
                      lenHist[3], lenHist[4],
                      lenHist[5] + lenHist[6] + lenHist[7],
                      droppedShort, droppedConflict,
                      droppedCap ? (", capped" ) : "");
        m_note = buf;

        if (cloud->tracks.empty()) {
            *err = "build_tracks: no tracks survived -- " + m_note;
            return false;
        }
        return true;
    }

    std::string RunReport() const override { return m_note; }

    // Keypoint coordinates are in IMAGE PIXELS and nothing rescales them, so a
    // proxy run would build tracks from positions wrong by the scale factor --
    // silently, since the sidecars are still present and still look valid.
    ProxyBehaviour Proxy() const override { return ProxyBehaviour::Never; }

    bool HasGPU() const override { return false; }

private:
    Param<int> m_minLength{this, "min_length", 2, 2, 20,
        {.help = "Fewest frames a track must appear in. Two is the minimum "
                 "that says anything -- one point seen once constrains "
                 "nothing. Raising it discards weakly supported points, which "
                 "trades coverage for reliability: a track seen by five frames "
                 "is far better determined than one seen by two, but there are "
                 "many fewer of them."}};

    Param<bool> m_dropConflicts{this, "drop_conflicts", true,
        "Discard tracks that see one frame twice. Physically impossible -- a "
        "point projects to one place in one image -- so it means two distinct "
        "points were merged, usually through a repeating texture. Turn this "
        "off only to see how many there are."};

    Param<int> m_maxTracks{this, "max_tracks", 0, 0, 200000,
        {.help = "Cap on how many tracks to keep, longest first. 0 is no cap. "
                 "A scalability control: bundle adjustment cost scales with "
                 "observations, and the longest tracks carry most of the "
                 "constraint."}};

    std::string m_note;
};

REGISTER_ALGORITHM(BuildTracks);

}  // namespace
}  // namespace tglab
