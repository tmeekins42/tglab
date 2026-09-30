// splat_sh — view-dependent colour for Gaussian splats, by spherical harmonics.
//
// A Gaussian with one colour looks the same from everywhere, so anything whose
// colour changes with the viewpoint -- a sheen, a highlight, a glossy book
// cover -- can only be baked in at one angle or faked with floaters that show
// it from some cameras and hide it from others. The paper (Kerbl et al. 2023)
// gives each Gaussian a colour that is a function of the viewing direction:
// a low-order spherical-harmonic expansion per channel, up to degree 3.
//
// HOW IT IS STORED. The degree-0 term is the Gaussian's ordinary colour, left
// exactly where it was (Splat::color, SplatParam::color), so nothing that only
// knows plain colour changes. The higher terms -- 3, 8 and 15 coefficients per
// channel at degrees 1, 2 and 3 -- sit in a separate array beside the splats,
// kShRest floats per Gaussian, coefficient-major: rest[k * 3 + channel]. A
// cloud without them has an empty array and costs nothing.
//
// THE CONVENTION IS THE REFERENCE CODE'S, so an exported .ply means the same
// thing in every splat viewer: the colour seen along direction d (from the
// camera centre to the Gaussian) is
//
//     base + sum_k  Y_k(d) * rest_k
//
// with the real SH basis Y_k and constants below. The reference stores the
// base as a DC coefficient with a 0.5 offset (base = C0 * f_dc + 0.5); the
// .ply reader and writer convert, and the rest terms need no conversion.
//
// EVALUATED ONCE PER GAUSSIAN PER VIEW, before rasterising, which is why the
// rasterisers did not change: they composite whatever colour each Gaussian
// has for this camera. The backward pass likewise receives dLoss/dColour per
// Gaussian, and the coefficients' gradient is that times Y_k(d). (The paper
// also lets the direction's dependence on the Gaussian's position feed the
// position gradient; that term is small and is left out, as several
// reimplementations do.)
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "../core/geometry.h"

namespace tglab {

constexpr int kShMaxDegree = 3;
constexpr int kShCoeffs    = 15;                 // beyond the DC term, at degree 3
constexpr int kShRest      = kShCoeffs * 3;      // floats per Gaussian
static_assert(kShRest == 45, "PointCloud::ShOf (core/data.h) assumes 45 per Gaussian");

// Coefficients per channel beyond the DC term, at a degree.
inline int ShCoeffsAt(int degree) {
    const int d = std::clamp(degree, 0, kShMaxDegree);
    return (d + 1) * (d + 1) - 1;
}

// The real SH basis, excluding Y_0, for a unit direction. Fills
// ShCoeffsAt(degree) values; the constants are the reference code's.
inline void ShBasis(double x, double y, double z, int degree, double out[kShCoeffs]) {
    if (degree < 1) return;
    const double C1 = 0.4886025119029199;
    out[0] = -C1 * y;
    out[1] =  C1 * z;
    out[2] = -C1 * x;
    if (degree < 2) return;
    const double xx = x * x, yy = y * y, zz = z * z;
    const double xy = x * y, yz = y * z, xz = x * z;
    out[3] =  1.0925484305920792 * xy;
    out[4] = -1.0925484305920792 * yz;
    out[5] =  0.31539156525252005 * (2.0 * zz - xx - yy);
    out[6] = -1.0925484305920792 * xz;
    out[7] =  0.5462742152960396 * (xx - yy);
    if (degree < 3) return;
    out[8]  = -0.5900435899266435 * y * (3.0 * xx - yy);
    out[9]  =  2.890611442640554 * xy * z;
    out[10] = -0.4570457994644658 * y * (4.0 * zz - xx - yy);
    out[11] =  0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy);
    out[12] = -0.4570457994644658 * x * (4.0 * zz - xx - yy);
    out[13] =  1.445305721320277 * z * (xx - yy);
    out[14] = -0.5900435899266435 * x * (xx - 3.0 * yy);
}

// The colour a Gaussian at `mean` shows a camera centred at `eye`, clamped to
// 0..1 as base colours are. `rest` is this Gaussian's kShRest floats.
template <typename F>
inline Vec3 ShColour(const Vec3& base, const F* rest, int degree, const Vec3& mean,
                     const Vec3& eye) {
    const int nk = ShCoeffsAt(degree);
    if (nk == 0 || !rest) return base;
    Vec3 d = mean - eye;
    const double len = d.Norm();
    if (len < 1e-12) return base;
    d = d * (1.0 / len);
    double Y[kShCoeffs];
    ShBasis(d.x, d.y, d.z, degree, Y);
    double c[3] = {base.x, base.y, base.z};
    for (int k = 0; k < nk; ++k)
        for (int ch = 0; ch < 3; ++ch) c[ch] += Y[k] * double(rest[k * 3 + ch]);
    return Vec3{std::clamp(c[0], 0.0, 1.0), std::clamp(c[1], 0.0, 1.0),
                std::clamp(c[2], 0.0, 1.0)};
}

}  // namespace tglab
