// OrbitCamera — the 3D counterpart to ViewCamera.
//
// Same idea, same sharing: image panels share a ViewCamera so comparing two
// results pans and zooms them together, and 3D panels share one of these so
// comparing two reconstructions turns them together. A reconstruction beside
// the reconstruction it is being measured against is the whole reason to have
// two panels open.
//
// THE CONTROLS MIRROR THE 2D ONES on purpose. ImageViewPanel reads
// IsItemHovered() plus a drag and the wheel; this reads the same gestures and
// means the analogous thing:
//
//     drag                         rotate      (pan, in 2D)
//     right / middle / shift+drag  pan         the scene follows the mouse
//     wheel                        dolly       toward the point under the cursor
//     double-click                 re-centre   orbit about the point clicked
//
// ORBIT RATHER THAN FREE-FLY. A reconstruction is an object to be inspected,
// not a space to be walked through: the useful motion is "turn it over and look
// at it", which is one target point and two angles. Free-fly needs six degrees
// of freedom, a movement speed that depends on scene scale, and a way back when
// the user flies into the middle of the points -- all of which is worse for
// looking at a point cloud.
//
// The maths is kept here rather than in the view so it can be tested without a
// window: a view-projection matrix is a pure function of the camera state, and
// that is exactly the kind of thing that is wrong in a way no screenshot
// reveals.
#pragma once

#include <algorithm>
#include <cmath>

#include "../core/geometry.h"

namespace tglab {

// The axis the orbit turns about. Kept as +Y so the basis is right-handed and
// "right" means right; which way is UP on screen is handled in ViewProj, for
// the reason given there.
inline Vec3 WorldUp() { return Vec3{0, 1, 0}; }

struct OrbitCamera {
    // What the camera looks at, and how far away it sits.
    Vec3   target;
    double distance = 5.0;

    // Spherical angles about the target. Yaw turns around the world's up axis;
    // pitch tilts, CLAMPED short of vertical -- at exactly vertical the up
    // vector and the view direction are parallel, the cross product that builds
    // the basis is zero, and the view flips over. Every orbit camera clamps
    // this and the ones that do not are the ones that spin unusably at the top.
    double yaw   = 0.0;
    double pitch = 0.3;

    double fovY = 50.0 * 3.14159265358979 / 180.0;
    double nearZ = 0.01;
    double farZ  = 1000.0;

    // Where the camera actually is, from the angles and distance.
    Vec3 Eye() const {
        const double cp = std::cos(pitch), sp = std::sin(pitch);
        return target + Vec3{distance * cp * std::sin(yaw), distance * sp,
                             distance * cp * std::cos(yaw)};
    }

    void Rotate(double dYaw, double dPitch) {
        yaw += dYaw;
        // Just short of +-90 degrees: see the note above.
        const double lim = 1.5533;   // 89 degrees
        pitch = std::clamp(pitch + dPitch, -lim, lim);
    }

    // Dolly, multiplicatively. A fixed step would crawl when far out and
    // overshoot the target when close in; a ratio moves the same fraction of
    // the remaining distance every notch, which is what feels linear.
    void Dolly(double notches) {
        distance *= std::pow(1.12, -notches);
        distance = std::clamp(distance, 1e-6, 1e6);
        FitClip();
    }

    // Dolly TOWARD THE CURSOR: the point under (sx, sy) on the target's plane
    // stays where it is on screen while everything else closes in on it, so
    // the wheel zooms into whatever is being pointed at rather than the
    // middle. Pixels from the viewport's top-left.
    void DollyAt(double notches, double sx, double sy, double viewW, double viewH) {
        const Vec3 p = OnTargetPlane(sx, sy, viewW, viewH);
        const double before = distance;
        Dolly(notches);
        const double f = distance / before;
        target = p + (target - p) * f;
    }

    // The point under a pixel, on the plane through the target facing the
    // camera. `up` renders toward the TOP of the screen, so a pixel below
    // the centre is toward -up. (Measured against ViewProj in test_sfm.)
    Vec3 OnTargetPlane(double sx, double sy, double viewW, double viewH) const {
        Vec3 right, up, fwd;
        Basis(&right, &up, &fwd);
        const double halfH = distance * std::tan(fovY * 0.5);
        const double halfW = halfH * viewW / std::max(1.0, viewH);
        const double nx = 2.0 * sx / std::max(1.0, viewW) - 1.0;
        const double ny = 2.0 * sy / std::max(1.0, viewH) - 1.0;
        return target + right * (nx * halfW) - up * (ny * halfH);
    }

    // Where a world point lands on screen, in pixels from the top-left, and
    // its depth along the view; false when it is behind the near plane.
    bool ToScreen(const Vec3& p, double viewW, double viewH, double* sx, double* sy,
                  double* depth) const {
        Vec3 right, up, fwd;
        Basis(&right, &up, &fwd);
        const Vec3 q = p - Eye();
        const double z = q.Dot(fwd);
        if (z <= nearZ) return false;
        const double t = std::tan(fovY * 0.5);
        const double aspect = viewW / std::max(1.0, viewH);
        *sx = (q.Dot(right) / (z * t * aspect) + 1.0) * 0.5 * viewW;
        *sy = (1.0 - q.Dot(up) / (z * t)) * 0.5 * viewH;
        *depth = z;
        return true;
    }

    // Re-centre the orbit on a point without moving the eye's distance to
    // it: the view turns to put it in the middle, and orbiting then turns
    // about it.
    void FocusOn(const Vec3& p) {
        const double d = (Eye() - p).Norm();
        target = p;
        distance = std::clamp(d, 1e-6, 1e6);
        FitClip();
    }

    // The scene's radius, from the last Frame(): the far plane must still
    // reach across it however close the camera has moved to the target.
    double sceneRadius = 1.0;

    // Near and far planes that follow the distance. Fixed at Frame() time,
    // zooming in to a tenth of that distance put the near plane BEHIND the
    // target and sliced away what was being looked at. The near plane stays
    // a small fraction of the distance; the far one reaches past the scene;
    // the ratio between them is capped, since that is what depth precision
    // depends on (see Frame).
    void FitClip() {
        farZ  = distance * 3.0 + sceneRadius * 2.0;
        nearZ = std::max({1e-9, distance * 0.1, farZ / 20000.0});
    }

    // Pan across the view plane, in units scaled by distance so a drag moves
    // the same fraction of the screen however far out the camera is.
    // The camera's basis: right, up and the viewing direction, in world space.
    //
    // ONE DEFINITION for everything that needs it. ViewProj builds its matrix
    // from this, Pan moves along it, and the splat renderer projects each
    // Gaussian's covariance into it. Three copies of the same cross products
    // would be three chances to get the handedness wrong independently, and
    // the handedness has already been wrong once -- see ViewProj.
    void Basis(Vec3* right, Vec3* up, Vec3* fwd) const {
        *fwd = (target - Eye()).Normalized();
        // right = worldUp x fwd, NOT fwd x worldUp: see ViewProj.
        *right = WorldUp().Cross(*fwd).Normalized();
        if (right->Norm() < 0.5) *right = Vec3{1, 0, 0};   // degenerate
        *up = right->Cross(*fwd).Normalized();
    }

    // Pan by a mouse delta in pixels, over a viewport `viewH` pixels tall: the
    // scale is the height of the view at the target's distance over the
    // pixels showing it, so what was under the cursor at the target's depth
    // stays under it -- the drag grabs the scene rather than sliding it.
    void Pan(double dx, double dy, double viewH) {
        Vec3 right, up, fwd;
        Basis(&right, &up, &fwd);
        const double s = 2.0 * distance * std::tan(fovY * 0.5) / std::max(1.0, viewH);
        // A mouse moving DOWN must carry the scene down, so the target moves
        // toward +up: `up` renders toward the top of the screen. This had the
        // opposite sign, and a shift-drag moved the scene against the mouse
        // vertically; the orbit tests in test_sfm now check it against the
        // rendered projection, pixel for pixel.
        target = target + right * (-dx * s) + up * (dy * s);
    }

    // Frames a bounding box: centres on it and backs off far enough to see it.
    //
    // What a 3D view needs on its first frame, and what a "reset" button does.
    // Without it the camera opens at an arbitrary distance from an arbitrary
    // point, and a reconstruction whose scale is a gauge freedom (see
    // global_position) could be anywhere from millimetres to kilometres across.
    void Frame(const Vec3& lo, const Vec3& hi) {
        target = (lo + hi) * 0.5;
        const Vec3 extent = hi - lo;
        const double radius = std::max(1e-3, extent.Norm() * 0.5);
        // Far enough that a sphere of that radius fits the vertical field of
        // view, with a little margin so the cloud is not flush to the edges.
        distance = radius / std::max(1e-6, std::tan(fovY * 0.5)) * 1.4;
        // THE NEAR/FAR RATIO IS WHAT DECIDES DEPTH PRECISION, and it is easy
        // to set catastrophically by reaching for round numbers.
        //
        // A perspective depth buffer spends most of its range near the front:
        // with near at distance/10000 and far at 100, the ratio is 1e5 and
        // essentially every bit of a 32-bit depth buffer is used up on the
        // first fraction of the scene. Points at similar depth then fail to
        // order stably, and rotating the camera makes them flicker in and out
        // -- which is exactly how this presented, as cameras appearing and
        // disappearing while dragging.
        //
        // Both planes are now tied to the scene: near at a tenth of the
        // distance to it, far at three times. A ratio of 30 leaves the depth
        // buffer with precision to spare, and nothing in a framed view falls
        // outside it.
        nearZ = std::max(1e-6, distance * 0.1);
        farZ  = distance * 3.0 + radius * 2.0;
        sceneRadius = radius;
    }

    // The view-projection matrix, for a constant buffer.
    //
    // LAID OUT FOR ROW-VECTOR MULTIPLICATION: a point is transformed as
    // `v * M`, so element (r, c) lives at out[r * 4 + c] and the translation
    // occupies the last ROW. That is HLSL's `mul(float4(p, 1), M)` and the
    // convention D3D samples default to; the opposite arrangement, with
    // translation in the last column, is what OpenGL and `M * v` want.
    //
    // Stated explicitly because the two are indistinguishable by inspection --
    // both are sixteen floats -- and a mismatch produces not an error but a
    // scene that projects to a single point at the origin. That is exactly
    // what the first version of this did.
    //
    // Right-handed view, and a projection mapping depth to [0,1] -- D3D's
    // convention, not OpenGL's [-1,1]. Getting that wrong does not produce an
    // error either; it wastes half the depth buffer's range and shows up as
    // z-fighting.
    void ViewProj(float out[16]) const {
        const Vec3 eye = Eye();
        Vec3 right, up, fwd;
        Basis(&right, &up, &fwd);
        // right = worldUp x fwd, NOT fwd x worldUp.
        //
        // This was backwards, and it mirrored every reconstruction
        // left-to-right. `fwd x worldUp` is the LEFT-handed convention: with
        // +Y up it produces a vector pointing to the viewer's left, so the
        // whole scene rendered as its own mirror image.
        //
        // Nothing in the geometry could reveal it -- a mirrored point cloud
        // is still a plausible point cloud, reprojects correctly, and every
        // numeric check passes. Tim found it with camera tracking markers
        // taped to the wall in fountain-P11: a blue square and a black/white
        // checker whose left-to-right order in the photograph was reversed on
        // screen. The orbit test watched it happen for weeks because it
        // computed its own expectation with the same cross product, so both
        // sides flipped together; see test_sfm.cpp for why that is now
        // derived from the world instead. The cross products live in Basis().

        // View: rotate into the camera basis, then translate.
        const double vx = -right.Dot(eye), vy = -up.Dot(eye), vz = fwd.Dot(eye);

        // THE SCREEN Y IS FLIPPED, and it belongs here rather than in the
        // basis.
        //
        // The reconstruction is in OpenCV's convention -- +X right, +Y DOWN,
        // camera along +Z -- because that is COLMAP's, and geometry.h states
        // it at the top. A viewer built with the graphics habit of +Y up shows
        // every reconstruction VERTICALLY MIRRORED, which is what this did:
        // the fountain rendered upside down, and it took a recognisable object
        // to notice, since a point cloud has no obvious top.
        //
        // NEGATING worldUp DOES NOT FIX IT, which was tried first. `right` is
        // fwd x worldUp and `up` is right x fwd, so negating worldUp negates
        // right and leaves up UNCHANGED -- it mirrors the image horizontally,
        // exactly the wrong axis. Both orientation tests flipped together,
        // which is how that showed.
        //
        // Negating the projection's y term flips the screen and nothing else:
        // the basis stays right-handed, "right" still means right, and the
        // orbit still turns about +Y.
        const double f = 1.0 / std::tan(fovY * 0.5);
        const double aspect = 1.0;   // the caller scales x by width/height
        const double a = f / aspect;
        // NO LONGER NEGATED, because the basis now does it.
        //
        // While `right` was computed left-handed (fwd x worldUp), `up` came
        // out negated too -- up is right x fwd, so flipping one flips the
        // other. This term was added to cancel that and make the scene look
        // upright, which it did, at the cost of leaving the left-to-right
        // mirror in place: two wrongs that together produced a plausible
        // picture, upright and reversed.
        //
        // With the basis right-handed, `up` already points the correct way
        // for a +Y-down reconstruction and negating here would flip the
        // image vertically again. Fixing the handedness is what removes the
        // need for this.
        const double fy = f;
        const double zn = nearZ, zf = farZ;
        const double q = zf / (zf - zn);

        // V * P, written out, with the translation in the last ROW.
        //
        // The view basis rows are (right, up, fwd) scaled by the projection's
        // x, y and z terms. fwd is NOT negated: this is a right-handed view
        // looking along +fwd, so a point in front has positive w -- which is
        // what the perspective divide needs, and what the first version got
        // backwards by copying an OpenGL-style -Z convention.
        const double m[16] = {
            right.x * a,  up.x * fy,  fwd.x * q,        fwd.x,
            right.y * a,  up.y * fy,  fwd.y * q,        fwd.y,
            right.z * a,  up.z * fy,  fwd.z * q,        fwd.z,
            vx * a,       vy * fy,    -vz * q - zn * q, -vz,
        };
        for (int i = 0; i < 16; ++i) out[i] = float(m[i]);
    }
};

}  // namespace tglab
