// match_brute — every descriptor against every other, exactly.
//
// The reference implementation. O(n*m) and slow at scale, which is the point:
// it is exhaustive, so it is what an approximate matcher gets checked against.
// A KNN tree that returns 92% of these is a useful trade; one that returns 40%
// is broken, and without an exact answer to compare with there is no way to
// tell those apart.
//
// LOWE'S RATIO TEST IS THE WHOLE THING, and it is worth being explicit about
// why a distance threshold is not enough. A descriptor's absolute distance to
// its best match says very little: a featureless patch of sky matches every
// other patch of sky closely, and a distinctive corner may match its true
// partner at a larger distance than that. What distinguishes a good match is
// that the best candidate is much better than the SECOND best -- which means
// the match is unambiguous, whatever its absolute distance.
//
//     ratio = d(best) / d(second best)
//
// Lowe measured 0.8 as the value that discards ~90% of false matches while
// losing ~5% of correct ones. That asymmetry is why the test works: false
// matches are overwhelmingly ambiguous ones.
//
// The CROSS CHECK is the other filter, and it catches a different failure. A
// feature in image A may match feature X in image B unambiguously, while X's
// own best match is some other feature entirely -- which happens when A's
// feature has no true partner in B at all (it was occluded, or moved out of
// frame). Requiring the match to be mutual removes those.
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "../../algo_util/features.h"
#include "../../core/algorithm.h"

namespace tglab {
namespace {

// Best and second-best distance from one descriptor to a whole set.
//
// Both, in one pass: the ratio test needs the runner-up, and finding it
// separately would double the work in the hot loop.
struct Best {
    int   index  = -1;
    float d1 = std::numeric_limits<float>::max();   // best
    float d2 = std::numeric_limits<float>::max();   // second best
};

// The runner-up must be a DIFFERENT FEATURE, not the same one seen twice.
//
// A detector that emits several keypoints at one position with different
// orientations -- which SIFT does by design -- gives every such feature a
// near-identical twin. That twin would be its own second-best match, the ratio
// would be near 1, and the ratio test would reject a good correspondence.
//
// HOW MUCH THIS ACTUALLY MATTERS, measured rather than assumed, because it was
// my second wrong theory about why SIFT keeps few matches: on assets/test.png
// it changes nothing. The rejected candidates are genuinely ambiguous, not
// twins -- 165 of 196 sit at ratio 0.9-1.0 and only 35 of those are correct,
// so the test is doing exactly its job. Kept because the reasoning is sound
// and the case is real on other images; not kept because it fixed anything
// here.
//
// `minSep` in pixels, squared here to keep the comparison off the square root.
inline bool FarEnough(const std::vector<Keypoint>& kps, int a, int b, float sepSq) {
    if (sepSq <= 0.0f) return true;
    if (a < 0 || b < 0 || size_t(a) >= kps.size() || size_t(b) >= kps.size()) return true;
    const float dx = kps[size_t(a)].x - kps[size_t(b)].x;
    const float dy = kps[size_t(a)].y - kps[size_t(b)].y;
    return dx * dx + dy * dy >= sepSq;
}

// Best, then the best candidate that is FAR ENOUGH from it to count.
//
// Two passes rather than one, and the reason is worth stating because the
// one-pass version is the obvious thing to write and is wrong: which candidates
// are "too close to the winner" is not known until the winner is known, so a
// single pass has to decide whether to keep a runner-up before it can tell
// whether that runner-up is the winner's twin. Trying it that way left the
// ratio unchanged, because a stale d2 from before the final winner survived.
//
// The second pass costs another sweep of the same array, already in cache.
template <class Dist>
Best NearestWith(size_t n, Dist dist, const std::vector<Keypoint>& kps, float sepSq) {
    Best b;
    for (size_t i = 0; i < n; ++i) {
        const float d = dist(i);
        if (d < b.d1) { b.d1 = d; b.index = int(i); }
    }
    if (b.index < 0) return b;

    for (size_t i = 0; i < n; ++i) {
        if (int(i) == b.index) continue;
        if (!FarEnough(kps, b.index, int(i), sepSq)) continue;
        const float d = dist(i);
        if (d < b.d2) b.d2 = d;
    }
    return b;
}

Best NearestFloat(const float* q, const DescriptorSet& set, int dim,
                  const std::vector<Keypoint>& kps, float sepSq) {
    return NearestWith(set.Count(),
                       [&](size_t i) { return DistanceL2Sq(q, set.FloatAt(i), dim); },
                       kps, sepSq);
}

Best NearestBinary(const uint8_t* q, const DescriptorSet& set, int bytes,
                   const std::vector<Keypoint>& kps, float sepSq) {
    return NearestWith(set.Count(),
                       [&](size_t i) {
                           return float(DistanceHamming(q, set.BinaryAt(i), bytes));
                       },
                       kps, sepSq);
}

// One descriptor against a set, dispatched on the set's KIND.
//
// This is what DescriptorKind exists for. L2 over a binary descriptor's packed
// bytes returns numbers, they order the candidates plausibly, and the matches
// are meaningless -- a failure with no symptom except that everything
// downstream is subtly wrong. The kind travels with the data so the matcher
// cannot be told the wrong thing.
Best Nearest(const DescriptorSet& query, size_t qi, const DescriptorSet& train,
             const std::vector<Keypoint>& trainKps, float sepSq) {
    if (query.kind != train.kind || query.dim != train.dim) return Best{};
    if (query.kind == DescriptorKind::Float)
        return NearestFloat(query.FloatAt(qi), train, query.dim, trainKps, sepSq);
    if (query.kind == DescriptorKind::Binary)
        return NearestBinary(query.BinaryAt(qi), train, query.BytesPerBinary(),
                             trainKps, sepSq);
    return Best{};
}

class MatchBrute : public AlgorithmBase {
public:
    const char* Name()     const override { return "match_brute"; }
    const char* Category() const override { return "match"; }

    PortList Inputs() const override {
        return {{"src", DataType::ImageSet, FormatSpec::Any, ShapeSpec::Any}};
    }
    PortList Outputs() const override {
        return {{"out", DataType::ImageSet, FormatSpec::SameAsInput, ShapeSpec::SameAsInput}};
    }

    void RunCPU(RunCtx&) override {}

    // Handled by the pipeline like an aligner: matching is inherently about
    // PAIRS, so it needs the whole set at once rather than one frame at a time.
    bool IsAligner() const override { return true; }

    bool RunAlign(std::vector<Image>* images, std::string* err) override {
        m_pairs = 0;
        m_total = 0;
        m_kept  = 0;
        m_note.clear();

        if (!images || images->size() < 2) {
            *err = "match_brute needs a group of at least two images";
            return false;
        }

        const int refIdx = std::clamp(int(m_reference), 0, int(images->size()) - 1);
        const FeatureSidecar* ref = FeaturesOf((*images)[size_t(refIdx)]);
        if (!ref) {
            *err = "match_brute: the reference image has no features -- "
                   "run a detector before matching";
            return false;
        }
        if (ref->keypoints.empty()) {
            m_note = "the reference image has no features to match against";
            return true;
        }

        for (size_t i = 0; i < images->size(); ++i) {
            if (int(i) == refIdx) continue;

            Image& img = (*images)[i];
            const FeatureSidecar* fs = FeaturesOf(img);
            if (!fs || fs->keypoints.empty()) continue;

            // A different detector on two frames of one group is a script
            // error rather than something to paper over: the descriptors are
            // not comparable, and matching them would return pairs that mean
            // nothing.
            if (fs->descriptors.kind != ref->descriptors.kind ||
                fs->descriptors.dim  != ref->descriptors.dim) {
                *err = "match_brute: frame " + std::to_string(i) +
                       " has a different descriptor from the reference (" +
                       fs->detector + " vs " + ref->detector +
                       ") -- one detector for the whole group";
                return false;
            }

            auto ms = std::make_shared<MatchSidecar>();
            ms->reference = refIdx;
            ms->matcher   = "brute";
            MatchPair(*ref, *fs, ms.get());

            m_total += ms->considered;
            m_kept  += int(ms->matches.size());
            ++m_pairs;

            img.Sidecars().Set(kMatchSidecar, ms);
        }
        return true;
    }

    std::string RunReport() const override {
        if (!m_note.empty()) return m_note;
        if (m_pairs == 0) return {};
        char buf[160];
        std::snprintf(buf, sizeof buf,
                      "matched %d pair%s: %d of %d candidates kept (%.0f%%)",
                      m_pairs, m_pairs == 1 ? "" : "s", m_kept, m_total,
                      m_total ? 100.0 * double(m_kept) / double(m_total) : 0.0);
        return buf;
    }

    bool HasGPU() const override { return false; }

private:
    // Named MatchPair rather than Match: inside the class, a method called
    // Match HIDES the struct of the same name, and `Match m;` below then reads
    // as a call. The errors point at the variable rather than the collision.
    void MatchPair(const FeatureSidecar& ref, const FeatureSidecar& other,
                   MatchSidecar* out) const {
        const float maxRatio = float(m_ratio);
        const bool  cross    = bool(m_crossCheck);
        const float sep      = std::max(0.0f, float(m_minSeparation));
        const float sepSq    = sep * sep;

        // The reverse direction, computed once, for the cross check. Without
        // caching it the check would re-scan the reference set per candidate
        // and turn an O(n*m) matcher into O(n*m*m).
        std::vector<int> backBest;
        if (cross) {
            backBest.assign(ref.keypoints.size(), -1);
            for (size_t i = 0; i < ref.descriptors.Count(); ++i) {
                const Best b = Nearest(ref.descriptors, i, other.descriptors,
                                       other.keypoints, sepSq);
                backBest[i] = b.index;
            }
        }

        const size_t n = other.descriptors.Count();
        for (size_t i = 0; i < n; ++i) {
            const Best b = Nearest(other.descriptors, i, ref.descriptors,
                                   ref.keypoints, sepSq);
            if (b.index < 0) continue;
            ++out->considered;

            // The ratio, in the units the distance is measured in. L2 is stored
            // SQUARED for speed, so the ratio of the stored values is the
            // SQUARE of the ratio of the distances -- comparing it against
            // Lowe's 0.8 directly would be testing 0.64 without meaning to.
            float ratio;
            if (ref.descriptors.kind == DescriptorKind::Float)
                ratio = (b.d2 > 0.0f) ? std::sqrt(b.d1 / b.d2) : 0.0f;
            else
                ratio = (b.d2 > 0.0f) ? (b.d1 / b.d2) : 0.0f;

            if (ratio > maxRatio) continue;

            // Mutual best: this feature's best match must have it as ITS best
            // match. Catches the case where a feature has no true partner at
            // all and merely resembles something.
            if (cross && backBest[size_t(b.index)] != int(i)) continue;

            Match m;
            m.a = b.index;
            m.b = int(i);
            m.distance = (ref.descriptors.kind == DescriptorKind::Float)
                             ? std::sqrt(b.d1) : b.d1;
            m.ratio = ratio;
            out->matches.push_back(m);
        }
    }

    Param<int> m_reference{this, "reference", 0, 0, 64,
        {.help = "Which frame the others are matched against. Every non-"
                 "reference frame gets a match set pairing it to this one."}};

    Param<float> m_ratio{this, "ratio", 0.8f, 0.1f, 1.0f,
        {.help = "Lowe's ratio test: keep a match only when the best candidate "
                 "is this much closer than the second best. 0.8 discards about "
                 "90% of false matches while losing 5% of correct ones. 1.0 "
                 "disables the test, which is almost never what you want.",
         .step = 0.01, .softMin = 0.6, .softMax = 0.9}};

    Param<bool> m_crossCheck{this, "cross_check", true,
        "Require the match to be mutual -- each feature must be the other's "
        "best. Catches features with no true partner at all, which the ratio "
        "test alone does not: such a feature can still match something "
        "unambiguously."};

    Param<float> m_minSeparation{this, "min_separation", 3.0f, 0.0f, 20.0f,
        {.help = "How far apart two candidates must be for the second to count "
                 "as the runner-up in the ratio test. SIFT emits several "
                 "keypoints at one position with different orientations, and "
                 "each is then its own second-best match -- which makes the "
                 "ratio near 1 and rejects a good correspondence. 0 disables "
                 "this and reproduces the textbook test.",
         .step = 0.5, .softMax = 8.0}};

    int         m_pairs = 0;
    int         m_total = 0;
    int         m_kept  = 0;
    std::string m_note;
};

} // namespace

REGISTER_ALGORITHM(MatchBrute);

} // namespace tglab
