#pragma once

#include <algorithm>
#include <cmath>

#include "../core/image.h"

namespace tglab {

// WHICH PART OF THE IMAGE A PANEL CAN SEE, in FULL-RESOLUTION image pixels.
//
// Split out of ImageViewPanel::Draw() so it can be tested without a window, a
// device, or an ImGui context. That is not tidiness: this arithmetic was wrong
// in a way the whole region test suite could not see, because those tests hand
// the pipeline a rectangle directly and never ask where a rectangle comes
// from. The bug was a unit mismatch, and a unit mismatch is exactly what a
// pure function of numbers can pin down.
//
// The caller passes the whole image's rectangle in SCREEN pixels -- which is
// what the layout already computes, since it lays out against the full extent
// precisely so the picture does not move when the pipeline switches between a
// crop and the whole frame. Given that, the conversion is just the zoom:
//
//     screen pixels = full-resolution image pixels * zoom
//
// proxyScale does NOT belong in it. It is a MULTIPLIER stating how many of an
// image's own pixels there are per full-resolution pixel (0.25 on a
// quarter-scale proxy), and it has already been accounted for in turning the
// image's extent into the screen rectangle passed here. Applying it a second
// time asks for a rectangle scaled by the proxy factor -- a different part of
// the image, at the wrong size, and only while dragging.
//
// Rounded OUTWARD, because a rectangle half a pixel short leaves a seam at the
// edge of the visible area, which is the one place the user is looking.
struct VisibleRectInput {
    float panelX = 0, panelY = 0, panelW = 0, panelH = 0;  // panel, screen px
    float imageX = 0, imageY = 0, imageW = 0, imageH = 0;  // WHOLE image, screen px
    float zoom = 1.0f;  // screen px per full-resolution image px
};

inline ImageRect ComputeVisibleRect(const VisibleRectInput& in) {
    const float vx0 = std::max(in.panelX, in.imageX);
    const float vy0 = std::max(in.panelY, in.imageY);
    const float vx1 = std::min(in.panelX + in.panelW, in.imageX + in.imageW);
    const float vy1 = std::min(in.panelY + in.panelH, in.imageY + in.imageH);
    if (!(vx1 > vx0) || !(vy1 > vy0) || in.zoom <= 1e-6f) return ImageRect{};

    // The image rect spans the WHOLE image even when the pipeline is currently
    // producing a crop, so this is already in whole-image coordinates. It has
    // to be: the answer feeds back in as the next frame's input, and measuring
    // against the crop would ask for a region inside the region, shrinking on
    // itself every frame until nothing was left.
    ImageRect r;
    r.x = int(std::floor((vx0 - in.imageX) / in.zoom));
    r.y = int(std::floor((vy0 - in.imageY) / in.zoom));
    r.w = int(std::ceil((vx1 - vx0) / in.zoom)) + 1;
    r.h = int(std::ceil((vy1 - vy0) / in.zoom)) + 1;
    r.x = std::max(0, r.x);
    r.y = std::max(0, r.y);
    return r;
}

}  // namespace tglab
