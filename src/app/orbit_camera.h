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
//     drag            rotate      (pan, in 2D)
//     shift + drag    pan
//     wheel           dolly       (zoom, in 2D)
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
        distance = std::clamp(distance, 1e-3, 1e6);
    }

    // Pan across the view plane, in units scaled by distance so a drag moves
    // the same fraction of the screen however far out the camera is.
    void Pan(double dx, double dy) {
        const Vec3 fwd = (target - Eye()).Normalized();
        const Vec3 worldUp{0, 1, 0};
        const Vec3 right = fwd.Cross(worldUp).Normalized();
        const Vec3 up = right.Cross(fwd).Normalized();
        const double s = distance * 0.002;
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
        const Vec3 fwd = (target - eye).Normalized();
        const Vec3 worldUp{0, 1, 0};
        Vec3 right = fwd.Cross(worldUp).Normalized();
        if (right.Norm() < 0.5) right = Vec3{1, 0, 0};   // degenerate; pick one
        const Vec3 up = right.Cross(fwd).Normalized();

        // View: rotate into the camera basis, then translate.
        const double vx = -right.Dot(eye), vy = -up.Dot(eye), vz = fwd.Dot(eye);

        const double f = 1.0 / std::tan(fovY * 0.5);
        const double aspect = 1.0;   // the caller scales x by width/height
        const double a = f / aspect;
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
            right.x * a,  up.x * f,  fwd.x * q,        fwd.x,
            right.y * a,  up.y * f,  fwd.y * q,        fwd.y,
            right.z * a,  up.z * f,  fwd.z * q,        fwd.z,
            vx * a,       vy * f,    -vz * q - zn * q, -vz,
        };
        for (int i = 0; i < 16; ++i) out[i] = float(m[i]);
    }
};

}  // namespace tglab
