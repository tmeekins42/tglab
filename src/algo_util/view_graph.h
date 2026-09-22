// The VIEW GRAPH: which frames see each other, and how they are oriented.
//
// Nodes are frames; edges are pairs that share enough verified correspondences
// to recover a relative pose. This is the structure global Structure-from-Motion
// actually consumes -- rotation averaging and global positioning both read it
// and neither reads pixels.
//
// WHY AN EDGE CARRIES A DIRECTION AND NOT A TRANSLATION. Two-view geometry
// determines the relative rotation completely and the relative translation only
// up to scale: the epipolar constraint cannot distinguish a small baseline with
// nearby scene content from a large one with distant content. Every edge is
// therefore a unit vector plus an unknown, and recovering the consistent set of
// scales across the whole graph is exactly what global positioning does later.
//
// That asymmetry is the reason rotation averaging and translation averaging are
// separate stages solved in that order, rather than one joint pose average:
// rotations compose into a solvable problem on their own, translations do not.
#pragma once

#include <string>
#include <vector>

#include "../core/geometry.h"
#include "../core/image.h"
#include "../core/sidecar.h"

namespace tglab {

inline constexpr const char* kRelativePoseSidecar = "relative_pose";

// The relative poses attached to ONE frame, against the frames it was matched
// to. Filed on the non-reference frame, matching how MatchSidecar does it --
// whatever consumes this is walking the frames, not the pairs.
class RelativePoseSidecar : public SidecarBase {
public:
    struct Edge {
        // Which frame this is relative TO.
        int reference = -1;

        // Rotation taking the reference frame's camera coordinates into this
        // frame's: x_this = R * x_ref. Composing along a path through the graph
        // is what rotation averaging exploits, and it only works because this
        // is a rotation between CAMERAS rather than a 2D image transform.
        Mat3 R;

        // Unit translation direction, in the reference frame's coordinates.
        // Its magnitude is unrecoverable from two views -- see the header note.
        Vec3 direction;

        // How many correspondences supported it. THE weight for averaging: an
        // edge backed by four hundred points deserves more say than one backed
        // by twenty, and weighting by it is the cheapest robustness available.
        int inliers = 0;
    };

    std::vector<Edge> edges;

    // Derived from the pixels: the poses were solved from features found in
    // this image, so re-developing it invalidates them.
    bool DerivedFromPixels() const override { return true; }
};

inline const RelativePoseSidecar* RelativePosesOf(const Image& img) {
    return img.Sidecars().Get<RelativePoseSidecar>(kRelativePoseSidecar);
}

}  // namespace tglab
