// splat_reflect — reflections for Gaussian splats, by deferred shading.
//
// Spherical harmonics (splat_sh.h) let a Gaussian's colour change smoothly
// with the viewpoint, which is a sheen, not a reflection: a mirror-like
// surface shows a sharp image of its surroundings that moves across it as
// the camera does, and no low-order expansion per Gaussian can draw that.
// Plain 3DGS falls back on floaters -- Gaussians hanging behind the surface
// to paint the reflected image -- which is the noise over the shiny book in
// the cat video.
//
// THE MODEL is 3DGS-DR's (Ye et al., "3D Gaussian Splatting with Deferred
// Reflection", SIGGRAPH 2024):
//
//   * each Gaussian carries a REFLECTIVITY r in 0..1 and a surface NORMAL,
//     which starts as the thin axis of the disc init_splats fitted;
//   * the scene carries an ENVIRONMENT: a cube map of what surrounds it,
//     learned from the photographs like everything else;
//   * per pixel, after compositing, colour C_d, reflectivity R and normal N
//     are each the alpha blend of the Gaussians' own, and the pixel is
//
//         C = (1 - R) C_d  +  R  Env( reflect(view ray, N / |N|) )
//
// DEFERRED because shading the blended normal rather than each Gaussian's is
// what lets a reflection be sharp: the environment is looked up once per
// pixel along one well-defined mirror direction, and the normal gets a
// gradient from what the reflection should have shown.
//
// THREE PASSES OF THE SAME RASTERISER. Compositing is linear in whatever a
// Gaussian carries, with weights set by the geometry alone, so C_d, R and N
// are the rasteriser run three times with three payloads -- colour, (r,r,r)
// and the camera-facing normal -- and nothing in it had to change. Backward
// runs it three times too, each with its map's gradient, and the geometry
// gradients add: the pixel depends on position, shape and opacity through
// all three maps, and the chain rule is exactly that sum.
#pragma once

#include <vector>

#include <algorithm>
#include <cmath>

#include "../core/data.h"
#include "splat_raster.h"

namespace tglab {

// The environment: six square faces of res x res RGB texels, +X -X +Y -Y +Z
// -Z, sampled bilinearly within a face. Seams between faces are not
// filtered across -- at the resolutions trained here they do not show.
struct EnvMap {
    int res = 0;
    std::vector<double> texels;   // 6 * res * res * 3

    void Init(int r, double grey) {
        res = r;
        texels.assign(size_t(6) * size_t(r) * size_t(r) * 3, grey);
    }
    bool Valid() const { return res > 0 && texels.size() == size_t(6) * size_t(res) * size_t(res) * 3; }

    // The colour seen along direction d (need not be unit).
    Vec3 Sample(const Vec3& d) const;
    // Adds `w * dColour` into `grad` (sized like texels) at the texels
    // Sample(d) reads, with its bilinear weights.
    void Splat(const Vec3& d, const Vec3& dColour, std::vector<double>* grad) const;
    // dLoss/d(direction) for a read along d whose colour has gradient
    // dColour: the face coordinates' change with direction times the
    // bilinear slope. Exact within a face; a seam or texel edge is a kink.
    Vec3 DirectionGrad(const Vec3& d, const Vec3& dColour) const;
};

// Per Gaussian: reflectivity (a logit, so any value is a valid 0..1) and a
// normal (any length; normalised when used).
struct ReflParam {
    double reflLogit = -4.6;   // sigmoid ~ 0.01: starts out a mirror of nothing
    double normal[3] = {0, 0, 1};
};

// The Gaussians with colour replaced by (r, r, r), for the R pass.
void ReflPayload(const std::vector<SplatParam>& params, const std::vector<ReflParam>& refl,
                 std::vector<SplatParam>* out);
// ...and by each unit normal turned to face the camera at `eye`, for the N
// pass. `flip` receives +1 or -1 per Gaussian, which the backward pass needs.
void NormalPayload(const std::vector<SplatParam>& params, const std::vector<ReflParam>& refl,
                   const Vec3& eye, std::vector<SplatParam>* out, std::vector<double>* flip);

// Deferred shading, per pixel. Cd, Rm, Nm are the three composited images
// (w*h*3 each; Rm's three channels are equal). Writes the shaded image.
void ShadeDeferred(const std::vector<double>& Cd, const std::vector<double>& Rm,
                   const std::vector<double>& Nm, const SplatCam& cam, const EnvMap& env,
                   std::vector<double>* out);

// Its backward pass: given dLoss/dOut, fills dLoss/dCd, dLoss/dRm (in
// channel 0, the others zero -- the R payload's colour gradient then lands
// in channel 0 alone) and dLoss/dNm, and ADDS dLoss/dtexel into envGrad.
void ShadeDeferredBackward(const std::vector<double>& Cd, const std::vector<double>& Rm,
                           const std::vector<double>& Nm, const SplatCam& cam,
                           const EnvMap& env, const std::vector<double>& dOut,
                           std::vector<double>* dCd, std::vector<double>* dRm,
                           std::vector<double>* dNm, std::vector<double>* envGrad);

// A cloud's stored reflections as the renderer wants them: empty `refl` and
// an invalid `env` when it has none. The reflectivity goes back to a logit.
inline void ReflFromCloud(const PointCloud& cloud, std::vector<ReflParam>* refl, EnvMap* env) {
    refl->clear();
    *env = EnvMap{};
    const size_t n = cloud.splats.size();
    if (cloud.envRes <= 0 || cloud.splatRefl.size() != n * 4 ||
        cloud.envMap.size() != size_t(6) * size_t(cloud.envRes) * size_t(cloud.envRes) * 3)
        return;
    refl->resize(n);
    for (size_t i = 0; i < n; ++i) {
        const double r = std::clamp(double(cloud.splatRefl[i * 4]), 1e-6, 1.0 - 1e-6);
        (*refl)[i].reflLogit = std::log(r / (1.0 - r));
        for (int k = 0; k < 3; ++k) (*refl)[i].normal[k] = cloud.splatRefl[i * 4 + 1 + size_t(k)];
    }
    env->res = cloud.envRes;
    env->texels.assign(cloud.envMap.begin(), cloud.envMap.end());
}

// A full render: colour (with spherical harmonics) and, when `refl` is
// non-empty and `env` valid, the reflection passes and deferred shading.
// What every render-only caller uses, so they all show the same thing.
template <typename F>
void RenderShaded(SplatRaster& raster, const std::vector<SplatParam>& params,
                  const std::vector<F>& sh, int shDegree, const std::vector<ReflParam>& refl,
                  const EnvMap& env, const SplatCam& cam, const RasterOptions& opt,
                  std::vector<double>* rgb, std::vector<double>* depth = nullptr) {
    std::vector<SplatParam> shaded;
    ShadeForView(params, sh, shDegree, CentreOf(cam), &shaded);
    if (refl.size() != params.size() || !env.Valid()) {
        raster.Forward(shaded, cam, opt, rgb, depth);
        return;
    }
    std::vector<double> Cd, Rm, Nm;
    raster.Forward(shaded, cam, opt, &Cd, depth);
    RasterOptions black = opt;
    black.background = Vec3{0, 0, 0};
    std::vector<SplatParam> pr, pn;
    std::vector<double> flip;
    ReflPayload(params, refl, &pr);
    NormalPayload(params, refl, CentreOf(cam), &pn, &flip);
    SplatRaster r2, r3;
    r2.Forward(pr, cam, black, &Rm);
    r3.Forward(pn, cam, black, &Nm);
    ShadeDeferred(Cd, Rm, Nm, cam, env, rgb);
}

}  // namespace tglab
