// match_guided — a second matching pass, guided by the geometry the first one
// recovered.
//
// WHY A SECOND PASS EXISTS AT ALL, which is the whole argument for this stage.
//
// The first pass has no idea where a feature's partner should be, so for every
// feature in A it must search all ten thousand descriptors in B and then decide
// whether the winner is trustworthy. That decision is the hard part: the
// descriptors are 486 bits over a small patch, and in a photograph of a brick
// wall thousands of patches genuinely look alike. The ratio test is the usual
// answer -- accept only when the best match is clearly better than the second
// -- and it is a blunt one, because on repetitive texture there is always a
// close second and the correct match is thrown away with the ambiguous ones.
//
// Measured on fountain-P11: the first pass keeps 16% of its candidate pairs,
// and 10000 features per frame yield 3398 tracks. The features are there; they
// are not being paired.
//
// ONCE THE ESSENTIAL MATRIX IS KNOWN, the problem changes completely. A feature
// in A no longer has ten thousand possible partners in B -- it has the ones
// lying on a LINE, its epipolar line, because that is where the geometry says
// its partner must be. Two consequences:
//
//   * The search is one-dimensional rather than two. Most of the ten thousand
//     are eliminated by geometry before any descriptor is compared.
//   * The ambiguity the ratio test guards against largely evaporates. Two
//     patches that look alike but sit off the line are no longer competitors,
//     so a match that the first pass had to discard as ambiguous is often
//     unambiguous here.
//
// This is classical guided matching (Hartley & Zisserman 11.5). What makes it
// worth having as a separate STAGE rather than folding into the matcher is that
// it needs the output of relative_pose, which needs the output of the matcher:
// the dependency genuinely runs in a circle and the only way to express it is
// two passes.
//
// WHAT THIS DOES NOT DO YET, stated so the gap is visible: it does not expand
// along tracks (match A->B, then use the track to predict C), and it does not
// densify into regions with no features at all. Both are natural extensions and
// both are larger than this.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "../../algo_util/features.h"
#include "../../algo_util/view_graph.h"
#include "../../core/algorithm.h"

namespace tglab {
namespace {

// The fundamental matrix for a pair, from its relative pose and a focal guess.
//
// E = [t]_x R is the essential matrix, in CALIBRATED coordinates. The
// correspondences here are in pixels, so the intrinsics have to come back in:
// F = K^-T E K^-1. Folding them in once per pair is cheaper than normalising
// every candidate, and it lets the distance below be stated in pixels, which is
// the only unit a threshold can honestly be given in.
Mat3 FundamentalFrom(const Mat3& R, const Vec3& t, double focal,
                     double cx, double cy) {
    // [t]_x
    Mat3 tx;
    tx.m[0] =  0.0;   tx.m[1] = -t.z;  tx.m[2] =  t.y;
    tx.m[3] =  t.z;   tx.m[4] =  0.0;  tx.m[5] = -t.x;
    tx.m[6] = -t.y;   tx.m[7] =  t.x;  tx.m[8] =  0.0;

    const Mat3 E = tx * R;

    // K^-1, and K^-T. A pinhole with square pixels and no skew, matching what
    // build_tracks fills in.
    Mat3 Ki;
    Ki.m[0] = 1.0 / focal; Ki.m[1] = 0.0;         Ki.m[2] = -cx / focal;
    Ki.m[3] = 0.0;         Ki.m[4] = 1.0 / focal; Ki.m[5] = -cy / focal;
    Ki.m[6] = 0.0;         Ki.m[7] = 0.0;         Ki.m[8] = 1.0;

    return Ki.Transpose() * E * Ki;
}

// Distance from point x2 to the epipolar line F * x1, in pixels.
//
// The line is l = F x1 = (a, b, c), and the perpendicular distance from a point
// to ax + by + c = 0 is |ax + by + c| / sqrt(a^2 + b^2). Normalising by the
// gradient is what turns an algebraic residual into a distance -- the same
// reason Sampson distance divides, and the reason a threshold here means the
// same thing on every pair.
double EpipolarDistance(const Mat3& F, double x1, double y1,
                        double x2, double y2) {
    const Vec3 l = F * Vec3{x1, y1, 1.0};
    const double n = std::sqrt(l.x * l.x + l.y * l.y);
    if (n < 1e-12) return 1e9;
    return std::fabs(l.x * x2 + l.y * y2 + l.z) / n;
}

class MatchGuided : public AlgorithmBase {
public:
    const char* Name()     const override { return "match_guided"; }
    const char* Category() const override { return "features"; }

    PortList Inputs() const override {
        return {{"src", DataType::ImageSet, FormatSpec::Any, ShapeSpec::Any}};
    }
    PortList Outputs() const override {
        return {{"out", DataType::ImageSet, FormatSpec::Any, ShapeSpec::Any}};
    }

    void RunCPU(RunCtx&) override {}
    bool IsAligner() const override { return true; }

    // Sidecar coordinates are in IMAGE PIXELS and nothing rescales them, so a
    // proxy run would search epipolar lines measured in the wrong units.
    ProxyBehaviour Proxy() const override { return ProxyBehaviour::Never; }

    bool RunAlign(std::vector<Image>* images, std::string* err) override {
        if (!images || images->size() < 2) {
            *err = "match_guided needs a group of at least two images";
            return false;
        }
        const int n = int(images->size());

        const double maxDist = double(m_maxDistance);
        const int    maxHam  = int(m_maxHamming);

        int pairsSeen = 0, pairsGuided = 0;
        long long added = 0, before = 0;

        for (int f = 0; f < n; ++f) {
            const RelativePoseSidecar* rp = RelativePosesOf((*images)[size_t(f)]);
            const MatchSidecar* ms = MatchesOf((*images)[size_t(f)]);
            if (!rp || !ms) continue;

            const FeatureSidecar* fsB = FeaturesOf((*images)[size_t(f)]);
            if (!fsB || fsB->keypoints.empty()) continue;

            // Copied so the existing matches survive: this stage ADDS to the
            // first pass rather than replacing it. A match the first pass found
            // is one this one would have to rediscover, and rediscovering it
            // could only lose.
            auto revised = std::make_shared<MatchSidecar>(*ms);

            for (const RelativePoseSidecar::Edge& e : rp->edges) {
                if (e.reference < 0 || e.reference >= n) continue;
                const FeatureSidecar* fsA = FeaturesOf((*images)[size_t(e.reference)]);
                if (!fsA || fsA->keypoints.empty()) continue;
                if (fsA->descriptors.kind != fsB->descriptors.kind ||
                    fsA->descriptors.dim  != fsB->descriptors.dim) continue;

                // Find the MatchSet for this pair, which is where the new
                // matches go and which says what is already paired.
                MatchSet* set = nullptr;
                for (MatchSet& s : revised->sets)
                    if (s.reference == e.reference) { set = &s; break; }
                if (!set) continue;

                ++pairsSeen;
                before += static_cast<long long>(set->matches.size());

                // Which keypoints are already spoken for. A feature matched in
                // the first pass is not a candidate here, in either image: one
                // 3D point projects to one place in one image, so allowing a
                // second match to the same keypoint would create exactly the
                // conflicting track build_tracks then has to drop.
                std::vector<uint8_t> usedA(fsA->keypoints.size(), 0);
                std::vector<uint8_t> usedB(fsB->keypoints.size(), 0);
                for (const Match& m : set->matches) {
                    if (m.a >= 0 && m.a < int(usedA.size())) usedA[size_t(m.a)] = 1;
                    if (m.b >= 0 && m.b < int(usedB.size())) usedB[size_t(m.b)] = 1;
                }

                const ImageDesc& dA = (*images)[size_t(e.reference)].Desc();
                const double focal =
                    0.5 * dA.width / std::tan(0.5 * double(m_fovDeg) *
                                              3.14159265358979 / 180.0);
                const Mat3 F = FundamentalFrom(e.R, e.direction, focal,
                                               0.5 * dA.width, 0.5 * dA.height);

                const int bytes = fsA->descriptors.BytesPerBinary();
                const bool binary = fsA->descriptors.kind == DescriptorKind::Binary;

                int addedHere = 0;
                for (size_t ia = 0; ia < fsA->keypoints.size(); ++ia) {
                    if (usedA[ia]) continue;
                    const Keypoint& ka = fsA->keypoints[ia];

                    // NO RATIO TEST. The epipolar constraint has already done
                    // the job the ratio test was standing in for: a descriptor
                    // that looks similar but lies off the line is not a
                    // competitor, so "is there a close second?" no longer
                    // measures ambiguity. Measured on the first pass, loosening
                    // the ratio from 0.8 to 0.99 slightly IMPROVED the
                    // reconstruction, which says the test was already doing
                    // little there; here it would only discard good matches.
                    //
                    // What replaces it is a hard distance cap: a match must be
                    // on the line AND look alike. Both, not either.
                    int bestDist = maxHam + 1;
                    int bestIdx  = -1;
                    for (size_t ib = 0; ib < fsB->keypoints.size(); ++ib) {
                        if (usedB[ib]) continue;
                        const Keypoint& kb = fsB->keypoints[ib];

                        // GEOMETRY FIRST, because it rejects most candidates
                        // for the cost of a dot product, where a descriptor
                        // comparison is sixty bytes of popcount.
                        if (EpipolarDistance(F, ka.x, ka.y, kb.x, kb.y) > maxDist)
                            continue;

                        // ...and a scale sanity check. Two views of the same
                        // point see it at similar scale unless the baseline is
                        // enormous, and a 4x scale jump is a different feature.
                        const float sr = (ka.scale > 0.0f && kb.scale > 0.0f)
                                             ? (ka.scale / kb.scale) : 1.0f;
                        if (sr < 0.25f || sr > 4.0f) continue;

                        const int d = binary
                            ? DistanceHamming(fsA->descriptors.BinaryAt(ia),
                                              fsB->descriptors.BinaryAt(ib), bytes)
                            : int(DistanceL2Sq(fsA->descriptors.FloatAt(ia),
                                               fsB->descriptors.FloatAt(ib),
                                               fsA->descriptors.dim));
                        if (d < bestDist) { bestDist = d; bestIdx = int(ib); }
                    }

                    if (bestIdx < 0) continue;

                    Match m;
                    m.a = int(ia);
                    m.b = bestIdx;
                    m.distance = float(bestDist);
                    m.ratio = 0.0f;   // not measured here; see the note above
                    set->matches.push_back(m);

                    // The inlier flag travels with the match, and these are
                    // NOT verified: relative_pose runs its own RANSAC and will
                    // decide. Marking them inliers here would be asserting
                    // something this stage has not checked.
                    if (!set->inlier.empty()) set->inlier.push_back(0);

                    usedA[ia] = 1;
                    usedB[size_t(bestIdx)] = 1;
                    ++addedHere;
                }

                if (addedHere > 0) ++pairsGuided;
                added += addedHere;
            }

            (*images)[size_t(f)].Sidecars().Set(kMatchSidecar, std::move(revised));
        }

        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "guided: %lld matches added to %lld across %d pairs "
                      "(%d gained any); %.0f%% more",
                      added, before, pairsSeen, pairsGuided,
                      before ? 100.0 * double(added) / double(before) : 0.0);
        m_note = buf;
        return true;
    }

    std::string RunReport() const override { return m_note; }
    bool HasGPU() const override { return false; }

private:
    Param<float> m_maxDistance{this, "max_distance", 3.0f, 0.5f, 20.0f,
        {.help = "How far, in PIXELS, a candidate may sit from the epipolar "
                 "line its partner must lie on. This is the constraint doing "
                 "the work: it eliminates almost every candidate before a "
                 "descriptor is compared. Raise it when the first pass's poses "
                 "are rough, since the line is only as accurate as they are.",
         .step = 0.5}};

    Param<int> m_maxHamming{this, "max_hamming", 90, 10, 256,
        {.help = "Largest descriptor distance accepted, in bits for a binary "
                 "descriptor. There is no ratio test here -- the epipolar line "
                 "has already removed the ambiguity a ratio test measures -- "
                 "so this is the only appearance check, and it has to be a "
                 "real one. Roughly a fifth of the descriptor length is a "
                 "reasonable starting point."}};

    // MUST MATCH relative_pose's fov_deg, for the same reason build_tracks's
    // does: the pose was solved in that geometry, and the fundamental matrix
    // built here has to be built in it too.
    Param<float> m_fovDeg{this, "fov_deg", 50.0f, 5.0f, 150.0f,
        {.help = "Horizontal field of view, in degrees, used to turn the "
                 "pair's essential matrix into a fundamental matrix in pixel "
                 "coordinates. Set it to the same value as relative_pose's."}};

    std::string m_note;
};

REGISTER_ALGORITHM(MatchGuided);

}  // namespace
}  // namespace tglab
