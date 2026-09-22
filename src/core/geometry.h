// 3D geometry: the types every Structure-from-Motion stage passes around.
//
// Small and concrete rather than a linear algebra library. Eigen arrives with
// the bundle adjuster and is the right tool for a solve; it is the wrong tool
// for a struct that crosses a port boundary, where a plain aggregate keeps the
// data inspectable and the headers cheap.
//
// CONVENTIONS, stated once because every sign error in a reconstruction traces
// back to one of them being assumed rather than read:
//
//   * Camera looks down +Z, with +X right and +Y DOWN. This is OpenCV's and
//     COLMAP's convention, not OpenGL's. Chosen because COLMAP's sparse model
//     is the interchange format (see PointCloud below), and converting at the
//     boundary is one place to get it wrong instead of many.
//
//   * A camera's pose is WORLD-TO-CAMERA: x_cam = R * x_world + t. So `t` is
//     NOT the camera's position in the world -- that is -R^T * t, which
//     Camera::Center() returns. Storing world-to-camera because it is what the
//     projection equation uses directly and what COLMAP writes.
//
//   * Rotations are stored as 3x3 row-major matrices rather than quaternions.
//     A matrix is what projection needs, and rotation averaging produces
//     matrices; converting to quaternions for storage would mean converting
//     back at every use. Quaternions appear only at the COLMAP boundary.
#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

namespace tglab {

struct Vec3 {
    double x = 0.0, y = 0.0, z = 0.0;

    Vec3() = default;
    Vec3(double X, double Y, double Z) : x(X), y(Y), z(Z) {}

    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }

    double Dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    Vec3   Cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }
    double Norm() const { return std::sqrt(Dot(*this)); }

    // Unit vector, or the zero vector when this is (near) zero. Returning zero
    // rather than NaN: a degenerate direction is a real case in SfM -- two
    // cameras at the same place have no baseline -- and a NaN would propagate
    // silently through an averaging step, where a zero is at least detectable.
    Vec3 Normalized() const {
        const double n = Norm();
        return n > 1e-12 ? Vec3{x / n, y / n, z / n} : Vec3{};
    }
};

// 3x3, row-major: m[row * 3 + col].
struct Mat3 {
    double m[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};   // identity

    static Mat3 Identity() { return Mat3{}; }

    double  At(int r, int c) const { return m[r * 3 + c]; }
    double& At(int r, int c) { return m[r * 3 + c]; }

    Vec3 operator*(const Vec3& v) const {
        return {m[0] * v.x + m[1] * v.y + m[2] * v.z,
                m[3] * v.x + m[4] * v.y + m[5] * v.z,
                m[6] * v.x + m[7] * v.y + m[8] * v.z};
    }

    Mat3 operator*(const Mat3& o) const {
        Mat3 r;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                double s = 0.0;
                for (int k = 0; k < 3; ++k) s += At(i, k) * o.At(k, j);
                r.At(i, j) = s;
            }
        return r;
    }

    // For a ROTATION the transpose is the inverse, which is the only use this
    // has here -- named Transpose rather than Inverse so a caller who hands it
    // a non-rotation does not get a wrong answer under a reassuring name.
    Mat3 Transpose() const {
        Mat3 r;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) r.At(i, j) = At(j, i);
        return r;
    }

    // Angle of the rotation taking `this` to `other`, in radians.
    //
    // THE measure for rotation averaging: the residual of a relative rotation
    // against the two global rotations it constrains, and what a test compares
    // instead of raw matrices. Comparing element by element would fail on a
    // correct solver, because rotation averaging has a gauge freedom -- the
    // whole solution can be rotated and remain equally correct.
    double AngleTo(const Mat3& other) const {
        const Mat3 d = Transpose() * other;
        const double tr = d.At(0, 0) + d.At(1, 1) + d.At(2, 2);
        // Clamped: floating point puts the trace a hair outside [-1, 3] and
        // acos of 1.0000001 is NaN.
        const double c = (tr - 1.0) * 0.5;
        return std::acos(c < -1.0 ? -1.0 : (c > 1.0 ? 1.0 : c));
    }
};

// --- rotation conversions ---------------------------------------------------
//
// Angle-axis is the minimal parameterisation: three numbers for three degrees
// of freedom, where a matrix has nine with six constraints between them. Every
// solver wants the minimal form (an unconstrained optimiser handed nine numbers
// walks straight off the rotation manifold); everything else wants the matrix.

// Rodrigues' formula: an angle-axis vector to a rotation matrix. The vector's
// length is the angle, its direction the axis.
inline Mat3 AxisAngleToMat(const Vec3& w) {
    const double t = w.Norm();
    Mat3 R;
    if (t < 1e-12) {
        // The small-angle limit is identity plus the skew matrix. Taken
        // explicitly rather than by dividing by t, which is 0/0 at exactly
        // zero -- and zero is where a solve with no prior estimate STARTS.
        R.m[0] = 1.0;   R.m[1] = -w.z;  R.m[2] =  w.y;
        R.m[3] =  w.z;  R.m[4] = 1.0;   R.m[5] = -w.x;
        R.m[6] = -w.y;  R.m[7] =  w.x;  R.m[8] = 1.0;
        return R;
    }
    const double c = std::cos(t), s = std::sin(t), C = 1.0 - c;
    const double x = w.x / t, y = w.y / t, z = w.z / t;
    R.m[0] = c + x * x * C;     R.m[1] = x * y * C - z * s; R.m[2] = x * z * C + y * s;
    R.m[3] = y * x * C + z * s; R.m[4] = c + y * y * C;     R.m[5] = y * z * C - x * s;
    R.m[6] = z * x * C - y * s; R.m[7] = z * y * C + x * s; R.m[8] = c + z * z * C;
    return R;
}

inline Vec3 MatToAxisAngle(const Mat3& R) {
    const double tr = R.m[0] + R.m[4] + R.m[8];
    double c = (tr - 1.0) * 0.5;
    c = c < -1.0 ? -1.0 : (c > 1.0 ? 1.0 : c);
    const double t = std::acos(c);
    if (t < 1e-9) return Vec3{};
    const double k = t / (2.0 * std::sin(t));
    return Vec3{k * (R.m[7] - R.m[5]), k * (R.m[2] - R.m[6]), k * (R.m[3] - R.m[1])};
}

// The nearest true rotation to `M`, by polar decomposition.
//
// WHY ROTATION AVERAGING CANNOT DO WITHOUT THIS. Averaging rotations linearly
// -- summing matrices, or solving a least-squares system over their entries --
// produces something close to a rotation but not one: the columns drift out of
// orthogonality and the determinant off 1. Projecting back onto SO(3) after
// each such step is what makes the linear methods (Martinec-Pajdla) work at
// all.
//
// Implemented by iterating M <- (M + M^-T) / 2, which converges to the
// orthogonal polar factor. Chosen over an SVD because it is twenty lines
// instead of two hundred, converges in a handful of iterations for a matrix
// already near a rotation (which is the only case here), and needs no
// dependency. A full SVD arrives with Eigen in Phase 4 if a harder case turns
// up.
inline Mat3 NearestRotation(const Mat3& M) {
    Mat3 X = M;
    for (int it = 0; it < 24; ++it) {
        // Inverse transpose via the adjugate; for a near-rotation the
        // determinant is near 1, so this is well conditioned.
        const double d =
            X.m[0] * (X.m[4] * X.m[8] - X.m[5] * X.m[7]) -
            X.m[1] * (X.m[3] * X.m[8] - X.m[5] * X.m[6]) +
            X.m[2] * (X.m[3] * X.m[7] - X.m[4] * X.m[6]);
        if (std::fabs(d) < 1e-12) break;   // degenerate; leave it as it is

        Mat3 invT;
        invT.m[0] = (X.m[4] * X.m[8] - X.m[5] * X.m[7]) / d;
        invT.m[1] = (X.m[5] * X.m[6] - X.m[3] * X.m[8]) / d;
        invT.m[2] = (X.m[3] * X.m[7] - X.m[4] * X.m[6]) / d;
        invT.m[3] = (X.m[2] * X.m[7] - X.m[1] * X.m[8]) / d;
        invT.m[4] = (X.m[0] * X.m[8] - X.m[2] * X.m[6]) / d;
        invT.m[5] = (X.m[1] * X.m[6] - X.m[0] * X.m[7]) / d;
        invT.m[6] = (X.m[1] * X.m[5] - X.m[2] * X.m[4]) / d;
        invT.m[7] = (X.m[2] * X.m[3] - X.m[0] * X.m[5]) / d;
        invT.m[8] = (X.m[0] * X.m[4] - X.m[1] * X.m[3]) / d;

        double delta = 0.0;
        Mat3 next;
        for (int i = 0; i < 9; ++i) {
            next.m[i] = 0.5 * (X.m[i] + invT.m[i]);
            delta += std::fabs(next.m[i] - X.m[i]);
        }
        X = next;
        if (delta < 1e-14) break;
    }
    return X;
}

// A camera: where it is, and how it projects.
//
// Intrinsics are a simple pinhole -- one focal length and a principal point.
// No distortion, deliberately: 3D Gaussian Splatting's rasteriser has no
// distortion model either, which is why COLMAP undistorts before feeding it.
// Carrying distortion here would mean carrying it through every stage to be
// discarded at the end.
struct Camera {
    Mat3   R;               // world-to-camera rotation
    Vec3   t;               // world-to-camera translation
    double focal = 1.0;     // pixels
    double cx = 0.0, cy = 0.0;   // principal point, pixels
    int    width = 0, height = 0;

    // Whether this camera has been solved for. A reconstruction routinely
    // contains frames that failed to register, and treating an unsolved
    // identity pose as a real one puts a phantom camera at the origin.
    bool   solved = false;

    // Where the camera SITS in the world. Not `t`: see the header note.
    Vec3 Center() const { return R.Transpose() * t * -1.0; }

    // Projects a world point to pixels. Returns false when the point is behind
    // the camera (the cheirality test), which is a real and common case rather
    // than an error -- a triangulation that lands behind the camera is exactly
    // what the test exists to reject.
    bool Project(const Vec3& world, double* px, double* py) const {
        const Vec3 c = R * world + t;
        if (c.z <= 1e-9) return false;
        *px = focal * c.x / c.z + cx;
        *py = focal * c.y / c.z + cy;
        return true;
    }

    // The viewing ray for a pixel, in WORLD space, as a unit vector from the
    // camera centre. What global positioning optimises against.
    Vec3 RayThrough(double px, double py) const {
        const Vec3 cam{(px - cx) / focal, (py - cy) / focal, 1.0};
        return (R.Transpose() * cam).Normalized();
    }
};

// One image's observation of a 3D point: which frame, and which keypoint in it.
struct Observation {
    int frame = -1;      // index into the group
    int keypoint = -1;   // index into that frame's FeatureSet

    // WHERE IT WAS SEEN, in that frame's pixels.
    //
    // Copied in rather than looked up through `keypoint`, because after the
    // first reconstruction stage the frames are no longer an input -- the
    // pipeline is passing a PointCloud -- and every later stage needs these.
    // Positioning turns them into rays, triangulation intersects those rays,
    // and bundle adjustment measures reprojection against them.
    //
    // `keypoint` is kept alongside for provenance: it is what lets a viewer
    // draw a track back onto its source image, and what a debugging session
    // needs to find the feature a bad observation came from.
    float x = 0.0f, y = 0.0f;
};

// A TRACK: one 3D point, and every image that saw it.
//
// The central structure of global SfM. Incremental pipelines have no such
// thing -- a track emerges as a 3D point accumulates observations while images
// register -- but a global method needs the whole correspondence graph before
// it can solve anything, so tracks are built explicitly and first.
//
// The `point` is only meaningful once triangulation has run; `HasPoint` says
// whether it has. A track with observations and no point is the normal state
// between track building and triangulation, not an error.
struct Track {
    std::vector<Observation> obs;
    Vec3 point;
    // 0..1, averaged over every frame that saw this point. Set by
    // build_tracks, which is the last stage holding the source pixels.
    //
    // EXACTLY ZERO means unset rather than black: a point sampled from real
    // pixels is never all three channels at exactly 0.0, and a viewer needs to
    // tell "no colour was sampled" from "this point is dark" so it can fall
    // back to a neutral grey instead of drawing an invisible black dot.
    Vec3 color;
    bool hasPoint = false;

    int Length() const { return int(obs.size()); }
};

}  // namespace tglab
