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
