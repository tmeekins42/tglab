// Features: keypoints and descriptors attached to an image.
//
// A detector finds distinctive points and describes their neighbourhoods; a
// matcher pairs them between images; alignment and stitching solve from those
// pairs. Each of those is a separate algorithm, and what connects them is this
// sidecar rather than a chain of ports -- see core/sidecar.h for why.
//
// DESCRIPTORS ARE NOT ONE TYPE, and that is the design problem this file
// exists to solve. SIFT produces 128 floats; SURF 64 or 128; AKAZE's M-LDB is a
// bit string, typically 486 bits; ORB's is 256 bits. They differ in length, in
// element type, and in the distance that compares them -- L2 for the float
// ones, Hamming for the binary ones. A matcher given the wrong distance still
// returns matches, and they are garbage.
//
// So a DescriptorSet carries its own kind and dimension, and the distance
// function is chosen from the kind rather than assumed by the caller. That is
// the whole reason this is a struct with a tag rather than a vector<float>.
//
// THE USER DOES NOT NORMALLY PICK. Tim asked how the descriptor type is
// exposed. The answer here: a detector declares what it produces, and the
// detectors that genuinely offer a choice (SURF's 64 vs 128, "extended")
// expose it as an ordinary parameter. Nothing else has to name a descriptor at
// all -- the matcher reads the kind off the data. Requiring the script to keep
// detector and matcher in agreement would be a class of error the data can
// already prevent.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "../core/image.h"
#include "../core/sidecar.h"

namespace tglab {

// The name features are filed under. A constant rather than a literal at every
// use, for the same reason as kTransformSidecar: a typo would silently mean "no
// features", which looks like a detector that found nothing.
inline constexpr const char* kFeatureSidecar = "features";

// The image's own signal level, as the 99th percentile of its samples.
//
// WHY A DETECTOR NEEDS THIS. Every contrast threshold in the literature --
// Lowe's 0.02, SURF's Hessian threshold -- was calibrated on display-referred
// images, where the median pixel sits near 0.5 and the range really is 0..1. A
// scene-referred float raw is nominally in 0..1 too, and occupies the bottom
// sixth of it: measured across a 15-frame panorama, median 0.058, p99 0.16.
// The same threshold is then roughly ten times too strict, and the detector
// reports a scene with no texture rather than an exposure it was not scaled
// for. Dividing by this makes the published defaults mean what they say.
//
// The 99th percentile rather than the maximum, because a maximum is one hot
// pixel or one specular highlight -- and normalising by an outlier would scale
// the whole image by whatever happened to be brightest in it, which is exactly
// the instability this is meant to remove.
//
// nth_element rather than a sort: linear rather than n log n, and only the one
// order statistic is wanted. Every second pixel in each direction is sampled,
// which on a 45 MP frame is 2.8 M values -- far more than enough for a
// percentile, and a quarter of the work.
inline float Percentile99(const std::vector<float>& v) {
    if (v.empty()) return 0.0f;
    std::vector<float> s;
    s.reserve(v.size() / 4 + 1);
    for (size_t i = 0; i < v.size(); i += 4) s.push_back(std::fabs(v[i]));
    if (s.empty()) return 0.0f;
    const size_t i = size_t(0.99 * double(s.size() - 1));
    std::nth_element(s.begin(), s.begin() + i, s.end());
    return s[i];
}

// One detected point.
//
// Scale and angle are not decoration: a descriptor is computed on a patch
// SIZED by scale and ORIENTED by angle, which is what makes the result
// invariant to zoom and rotation. They are also what a visualiser needs to draw
// a feature honestly -- a fixed-size circle would hide the thing that
// distinguishes a scale-space detector from a corner detector.
struct Keypoint {
    float x = 0.0f, y = 0.0f;   // sub-pixel position, in image coordinates
    float scale = 1.0f;         // radius of the region this describes, in pixels
    float angle = 0.0f;         // dominant orientation, radians, 0 = +x
    float response = 0.0f;      // detector strength; higher is more distinctive
    int   octave = 0;           // which pyramid level it was found at
};

// What kind of descriptor a set holds, and therefore how to compare two.
enum class DescriptorKind {
    None,     // keypoints only, from a detector that does not describe
    Float,    // SIFT, SURF: compared by L2
    Binary,   // AKAZE M-LDB, ORB: compared by Hamming over packed bits
};

// A block of descriptors, one per keypoint, stored contiguously.
//
// Flat rather than a vector of vectors: matching is the hot loop and walks
// these linearly, so one allocation and a stride beats N allocations and a
// pointer chase. `dim` is elements per descriptor for Float, and BITS per
// descriptor for Binary -- the byte count is then (dim + 7) / 8.
struct DescriptorSet {
    DescriptorKind kind = DescriptorKind::None;
    int            dim  = 0;

    std::vector<float>   f;    // Float: count * dim
    std::vector<uint8_t> b;    // Binary: count * ((dim + 7) / 8)

    int BytesPerBinary() const { return (dim + 7) / 8; }

    size_t Count() const {
        if (kind == DescriptorKind::Float)  return dim ? f.size() / size_t(dim) : 0;
        if (kind == DescriptorKind::Binary) {
            const int bpd = BytesPerBinary();
            return bpd ? b.size() / size_t(bpd) : 0;
        }
        return 0;
    }

    const float*   FloatAt(size_t i)  const { return &f[i * size_t(dim)]; }
    const uint8_t* BinaryAt(size_t i) const { return &b[i * size_t(BytesPerBinary())]; }
};

// Squared L2 between two float descriptors.
//
// Squared, and left that way: matching only ever ORDERS distances and compares
// them by ratio, and the square root is monotonic so it changes neither. It is
// the innermost operation in an O(n*m) loop, which makes "cheap" worth more
// than "in the same units as the input".
inline float DistanceL2Sq(const float* a, const float* b, int dim) {
    float s = 0.0f;
    for (int i = 0; i < dim; ++i) {
        const float d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

// Hamming distance between two packed bit strings.
//
// std::popcount would be tidier, but this is called O(n*m) times and the
// table is measurably faster than a byte-at-a-time loop on the paths that do
// not compile to a POPCNT instruction. Worth the sixteen lines.
inline int DistanceHamming(const uint8_t* a, const uint8_t* b, int bytes) {
    static const uint8_t kBits[256] = {
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4, 1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
        1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5, 2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
        1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5, 2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
        2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6, 3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
        1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5, 2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
        2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6, 3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
        2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6, 3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
        3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7, 4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8};
    int s = 0;
    for (int i = 0; i < bytes; ++i) s += kBits[a[i] ^ b[i]];
    return s;
}

// The sidecar itself: what a detector attaches and a matcher reads.
class FeatureSidecar : public SidecarBase {
public:
    std::vector<Keypoint> keypoints;
    DescriptorSet         descriptors;

    // Which detector produced this, for the run report and for a visualiser
    // that wants to say what it is drawing. Not used to decide anything --
    // behaviour follows the descriptor KIND, which is data rather than a name.
    std::string detector;

    // Derived from the pixels, so it is dropped when the image changes. The
    // default, and correct here: features found in an image are wrong the
    // moment that image is re-developed, and silently keeping them would give
    // an alignment solved against an image that no longer exists.
    bool DerivedFromPixels() const override { return true; }
};

// The name matches are filed under.
inline constexpr const char* kMatchSidecar = "matches";

// One correspondence: feature `a` in the reference image is feature `b` here.
struct Match {
    int   a = 0;          // index into the REFERENCE image's keypoints
    int   b = 0;          // index into THIS image's keypoints
    float distance = 0.0f;// descriptor distance; lower is a closer match
    float ratio = 0.0f;   // best/second-best, the confidence measure
};

// Matches attached to an image, pairing its features against a reference.
//
// ATTACHED TO THE NON-REFERENCE IMAGE, and that asymmetry is deliberate. A
// group of N frames matched against frame 0 produces N-1 sets of
// correspondences, and each belongs with the frame it describes -- the same
// arrangement the transform sidecar uses, and for the same reason: whatever
// consumes it is walking the frames, not the pairs.
// One frame's matches against ONE other frame.
//
// Split out of MatchSidecar so a frame can carry several. Bundle adjustment
// needs that: refining a chain against only its immediate neighbours cannot
// remove accumulated drift, because a sequential chain has no constraint saying
// where frame 8 sits relative to frame 5. Matching a WINDOW of neighbours
// supplies exactly those constraints, and they are measurably there --
// on a real 15-frame sweep, pairs three apart still yield about 400 matches at
// 85-91% inliers, all of which a chain throws away.
struct MatchSet {
    std::vector<Match> matches;

    // Which image the `a` indices refer to. -1 when unset.
    int reference = -1;

    // How many candidate pairs were considered. See `considered` below.
    int considered = 0;

    // Which matches survived the geometric check, parallel to `matches`.
    //
    // WHY THIS TRAVELS WITH THE MATCHES. The ratio test and cross check are
    // DESCRIPTOR tests: they ask whether two patches look alike. RANSAC asks a
    // different question -- whether a match is consistent with the transform
    // everything else agrees on -- and it is the only one that can catch two
    // genuinely similar patches in genuinely different places. A repeating
    // texture produces exactly that, and no descriptor test will ever reject
    // it.
    //
    // So the verdict is worth keeping, and worth keeping HERE rather than
    // recomputing. Bundle adjustment reads it: without it, BA takes every match
    // the matcher produced, and on a detector whose matches are 45% outliers
    // that is half its observations pulling the wrong way. Measured, that left
    // it stalled at 25 px RMS where a clean detector reached 4.7 on the same
    // frames.
    //
    // Empty means "not verified", not "none passed" -- a consumer must treat an
    // empty vector as "use them all" so a matcher run without an aligner still
    // works.
    std::vector<uint8_t> inlier;

    bool IsInlier(size_t i) const {
        return inlier.empty() || (i < inlier.size() && inlier[i] != 0);
    }
};

class MatchSidecar : public SidecarBase {
public:
    // Every set this frame matched against, nearest neighbour first.
    //
    // Usually one, which is why the accessors below exist: a matcher that pairs
    // against a single reference writes one set, and every consumer written
    // before windowed matching reads it through `matches` and `reference`
    // without knowing there could be more.
    std::vector<MatchSet> sets;

    // The FIRST set's matches and reference, for the consumers that want one
    // pairing -- the aligner's chain, draw_matches, the reports.
    //
    // References into `sets` rather than copies, so there is one storage
    // location and no way for the two views to disagree. That was the whole
    // reason Param<T> is shaped the way it is, and the same argument applies
    // here: a duplicated `matches` vector would need syncing, and every bug
    // afterwards would be a sync bug.
    const std::vector<Match>& Matches() const {
        static const std::vector<Match> kEmpty;
        return sets.empty() ? kEmpty : sets.front().matches;
    }
    int Reference() const { return sets.empty() ? -1 : sets.front().reference; }

    // What produced these, for the report.
    std::string matcher;

    // How many candidate pairs were considered and how many survived. The
    // difference is the useful number: a matcher that keeps 90% of its
    // candidates has a threshold doing nothing, and one that keeps 2% has a
    // threshold that is probably discarding real matches too.
    int considered = 0;

    bool DerivedFromPixels() const override { return true; }
};

inline const MatchSidecar* MatchesOf(const Image& img) {
    return img.Sidecars().Get<MatchSidecar>(kMatchSidecar);
}

// Reads the features attached to an image, or null when there are none.
//
// Null rather than an empty set, so "no detector ran" is distinguishable from
// "a detector ran and found nothing" -- the first is a script the user has not
// finished writing, the second is a threshold set too high, and they want
// different messages.
inline const FeatureSidecar* FeaturesOf(const Image& img) {
    return img.Sidecars().Get<FeatureSidecar>(kFeatureSidecar);
}

} // namespace tglab
