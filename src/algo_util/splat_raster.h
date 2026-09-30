// splat_raster — a differentiable Gaussian-splat rasteriser, on the CPU.
//
// The core of Gaussian-splatting training (Kerbl et al. 2023): render the
// Gaussians from a camera, compare with the photograph, and push the error
// back into every Gaussian's position, shape, rotation, opacity and colour.
// The forward pass is the renderer; the backward pass is the chain rule
// through it, written out by hand.
//
// WHY THE CPU FIRST. The reference implementation is custom CUDA. A GPU
// version here would fight the same constraints the plane sweep did -- four
// texture bindings, and a device that hangs on unusual dispatch patterns --
// and a backward pass is exactly the kind of code that is plausibly wrong in
// ways nothing reveals. So this is the REFERENCE: double precision,
// deterministic, and checked against finite differences in test_sfm.cpp. A
// GPU trainer later will be measured against it, as the GPU sweep was
// against the CPU one.
//
// THE FORWARD PASS, as the paper does it:
//
//   1. Project every Gaussian: its centre to a pixel, its 3D covariance to a
//      2D one through the Jacobian of the projection (EWA splatting), plus a
//      third of a pixel so nothing is smaller than a pixel.
//   2. Bin the projected ellipses into 16x16 tiles and sort each tile's list
//      by depth.
//   3. Per pixel, composite front to back: alpha_i = opacity_i * G_i(pixel),
//      C = sum c_i alpha_i T_i with T the transmittance so far, stopping when
//      T is negligible.
//
// THE BACKWARD PASS walks each pixel's list in reverse, which is what lets
// the gradient of every alpha be computed from the colour that was BEHIND it
// without storing per-pixel lists. Then per Gaussian it chains from the 2D
// quantities (centre, conic, colour, opacity) back to the parameters.
//
// PARAMETERS ARE THE OPTIMISER'S, not the Splat struct's: log-scale so a
// scale can never go negative, a raw quaternion normalised in the forward pass
// so any four numbers are a rotation, and opacity as a logit so it stays in
// (0,1). ToParams/FromParams convert.
#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "../core/geometry.h"
#include "splat_sh.h"

namespace tglab {

class ComputeContext;

// A pinhole camera in the rasteriser's pixel frame: OpenCV conventions, as
// the rest of the pipeline -- +X right, +Y DOWN, looking along +Z.
struct SplatCam {
    Mat3   R;              // world to camera
    Vec3   t;
    double fx = 1.0, fy = 1.0, cx = 0.0, cy = 0.0;
    int    w = 0, h = 0;
};

// One Gaussian as the optimiser sees it. Fourteen numbers.
struct SplatParam {
    double mean[3]     = {0, 0, 0};
    double logScale[3] = {0, 0, 0};
    double quat[4]     = {1, 0, 0, 0};   // w x y z, not necessarily unit
    double opacity     = 0.0;            // logit
    double color[3]    = {0, 0, 0};

    static constexpr int kCount = 14;
    double*       Data()       { return mean; }
    const double* Data() const { return mean; }

    // ALL fourteen zero, for a GRADIENT buffer.
    //
    // A default-constructed SplatParam is not zero: its quaternion is the
    // identity rotation, w = 1. That is right for a Gaussian and wrong for a
    // gradient, and using one as the other added exactly 1.0 to every w
    // gradient -- found by the finite-difference test, where every other
    // parameter matched to 1e-7 and w alone was off by precisely one.
    static SplatParam Zero() {
        SplatParam p;
        p.quat[0] = 0.0;
        return p;
    }
};
static_assert(sizeof(SplatParam) == SplatParam::kCount * sizeof(double),
              "SplatParam must be exactly its fourteen doubles, in order");

SplatParam ToParam(const Splat& s);
Splat      FromParam(const SplatParam& p);

// The parameters with each colour replaced by the one it shows the camera
// centred at `eye`: base colour plus spherical harmonics (splat_sh.h). `sh`
// holds kShRest values per Gaussian in the same order, or is empty for plain
// colour, in which case `out` is a copy. The rasterisers composite whatever
// colour they are given, which is why view dependence needs nothing else.
template <typename F>
void ShadeForView(const std::vector<SplatParam>& params, const std::vector<F>& sh,
                  int degree, const Vec3& eye, std::vector<SplatParam>* out) {
    *out = params;
    if (degree < 1 || sh.size() < params.size() * size_t(kShRest)) return;
    for (size_t i = 0; i < params.size(); ++i) {
        SplatParam& p = (*out)[i];
        const Vec3 c = ShColour(Vec3{p.color[0], p.color[1], p.color[2]},
                                &sh[i * size_t(kShRest)], degree,
                                Vec3{p.mean[0], p.mean[1], p.mean[2]}, eye);
        p.color[0] = c.x; p.color[1] = c.y; p.color[2] = c.z;
    }
}

// The centre of a rasteriser camera, in world space.
inline Vec3 CentreOf(const SplatCam& c) { return c.R.Transpose() * c.t * -1.0; }

// A solved camera as the rasteriser wants it, rendering at w x h.
//
// The reconstruction's cameras were solved at whatever resolution the frames
// had then; training and rendering usually want another. Scaling is about
// pixel CENTRES, not edges: pixel x at the solved size covers [x-0.5, x+0.5],
// so a point at x maps to (x + 0.5) * s - 0.5 at scale s. Scaling cx by s
// alone would shift every render by up to half a pixel -- small, and exactly
// the kind of offset that makes training blur edges to hedge.
inline SplatCam SplatCamFrom(const Camera& c, int w, int h) {
    SplatCam o;
    o.R = c.R;
    o.t = c.t;
    const double sx = (c.width  > 0) ? double(w) / double(c.width)  : 1.0;
    const double sy = (c.height > 0) ? double(h) / double(c.height) : 1.0;
    o.fx = c.focal * sx;
    o.fy = c.focal * sy;
    o.cx = (c.cx + 0.5) * sx - 0.5;
    o.cy = (c.cy + 0.5) * sy - 0.5;
    o.w = w;
    o.h = h;
    return o;
}

// THE DEPTH LOSS, shared by every training path so they cannot drift apart.
//
// Per pixel with a target t (t > 0; 0 means "no measurement here"):
//
//     weight * | D - A t | / t,        A = 1 - T, the pixel's coverage
//
// normalised per pixel as the colour L1 is. RELATIVE because a
// reconstruction's units are arbitrary and depth uncertainty grows with
// distance, the same reasoning fuse_depth's tolerance uses.
//
// AGAINST A t, NOT t. The rendered D is sum(w_i z_i) over a background of
// depth 0, so a partly transparent pixel renders NEARER than its surface. The
// first version compared D with t directly, and to close that gap the loss
// pushed Gaussians BEHIND the surface and up in opacity -- smearing brick
// across the walls from every novel view (fountain-P11, 3000 iterations,
// weight 1). Against A t the loss is zero whenever every Gaussian sits at the
// measured depth, however much of the pixel they cover: coverage is colour's
// business, depth only says where.
//
// In the backward pass that is one extra channel whose value for Gaussian i
// is (z_i - t) -- A is the same blend with value 1 -- so the kernels take the
// target as a per-pixel SHIFT beside dLoss/dD. See SplatRaster::Backward.
//
// Writes dLoss/dD into `dD` (sized like `rendered`); adds the summed relative
// residual |D - A t| / t and the pixels it covered to `errSum`/`count`.
template <typename R, typename T2, typename G>
void DepthLossGrad(const std::vector<R>& rendered, const std::vector<T2>& finalT,
                   const std::vector<float>& target, double weight,
                   std::vector<G>* dD, double* errSum, long long* count) {
    const size_t np = rendered.size();
    dD->assign(np, G(0));
    const double k = weight / double(std::max<size_t>(1, np));
    for (size_t i = 0; i < np && i < target.size(); ++i) {
        const double t = double(target[i]);
        if (t <= 0.0) continue;
        const double cover = 1.0 - double(finalT[i]);
        const double diff = double(rendered[i]) - cover * t;
        *errSum += std::fabs(diff) / t;
        ++*count;
        (*dD)[i] = G(diff > 0.0 ? k / t : (diff < 0.0 ? -k / t : 0.0));
    }
}

struct RasterOptions {
    // An alpha below this is skipped. It also sets how far out a Gaussian is
    // evaluated: to where opacity * G falls below it, which for a fully
    // opaque Gaussian is about 3.3 standard deviations.
    double minAlpha = 1.0 / 255.0;
    // Capped just below one so nothing hides everything behind it
    // completely -- and so the backward pass never divides by 1 - alpha = 0.
    double maxAlpha = 0.99;
    // Compositing stops once transmittance falls below this.
    double tStop    = 1e-4;
    // Added to the 2D covariance so every Gaussian covers about a pixel.
    double lowPass  = 0.3;
    Vec3   background;          // what a pixel shows where nothing covers it
};

class SplatRaster {
public:
    SplatRaster();
    ~SplatRaster();
    SplatRaster(const SplatRaster&)            = delete;
    SplatRaster& operator=(const SplatRaster&) = delete;

    // THE PER-PIXEL PASSES ON THE GPU, when given a device.
    //
    // Measured on fountain-P11 (490k Gaussians in view, 320x213), an
    // iteration was 206 ms compositing and 223 ms in the per-pixel backward
    // pass, against 264 ms for everything per-Gaussian combined: at training
    // resolution half a million Gaussians overlap heavily, and each pixel
    // walks a hundred or more of them. That walk is what a GPU is for.
    //
    // Only the per-pixel work moves. Projection, sorting, the per-Gaussian
    // chain rule and the optimiser stay here, so the GPU path produces
    // exactly the intermediate the CPU one does -- a gradient per tile-list
    // entry -- and everything after it is shared.
    //
    // FALLS BACK TO THE CPU on any device failure, per call, and says so in
    // GpuNote(). Null turns the GPU off.
    void SetGpu(ComputeContext* gpu);
    bool UsedGpu() const { return m_usedGpu; }
    const std::string& GpuNote() const { return m_gpuNote; }

    // Renders `splats` from `cam` into `rgb` (w*h*3, row-major). Keeps what
    // Backward needs, so Backward must follow with the same splats and camera.
    //
    // `depth`, when given, receives each pixel's EXPECTED depth: the
    // camera-space depth of every Gaussian's centre, blended with the same
    // weights as its colour -- alpha times the transmittance reaching it --
    // over a background of depth 0. Depth is composited exactly as a fourth
    // colour channel would be, which is what makes its gradient the same
    // machinery as colour's plus one term: moving a Gaussian toward or away
    // from the camera changes the depth it contributes.
    void Forward(const std::vector<SplatParam>& splats, const SplatCam& cam,
                 const RasterOptions& opt, std::vector<double>* rgb,
                 std::vector<double>* depth = nullptr);

    // Given dLoss/dRGB for the image Forward produced, ADDS dLoss/dparam into
    // `grad` (same size as the splats, filled with SplatParam::Zero() by the
    // caller -- not default-constructed; see Zero()). `dDepth`, when given,
    // is dLoss/d(expected depth) per pixel (w*h), and `depthShift` the
    // per-pixel value subtracted from every Gaussian's depth in that
    // channel -- the depth TARGET, for DepthLossGrad's loss; see there.
    // Null shift is zero.
    void Backward(const std::vector<SplatParam>& splats, const SplatCam& cam,
                  const RasterOptions& opt, const std::vector<double>& dRgb,
                  std::vector<SplatParam>* grad,
                  const std::vector<double>* dDepth = nullptr,
                  const std::vector<float>* depthShift = nullptr);

    // Per pixel, from the last Forward: the transmittance left after every
    // Gaussian -- 1 minus the pixel's coverage.
    const std::vector<double>& FinalT() const { return m_finalT; }

    // How many Gaussians the last Forward found in view.
    int Visible() const { return m_visible; }

    // Milliseconds spent in each phase, accumulated over every call since
    // the raster was made. Kept permanently rather than added when needed:
    // the question "where does an iteration go" decides what is worth
    // moving to the GPU, and guessing at it has been wrong before.
    struct Timings {
        double project = 0, bin = 0, composite = 0;          // forward
        double backPixel = 0, backSum = 0, backGaussian = 0; // backward
    };
    const Timings& Time() const { return m_time; }

    // Per Gaussian, from the last Backward: the magnitude of the loss
    // gradient with respect to its SCREEN position, in pixels -- or -1 where
    // it was not in view.
    //
    // This is the signal densification runs on (Kerbl et al. 2023, 5.2). A
    // Gaussian that the optimiser keeps trying to push around the screen is
    // one that cannot, on its own, represent what is under it: either detail
    // it is too big to follow, or a gap it is being stretched to cover.
    const std::vector<double>& ScreenGrad() const { return m_screenGrad; }

    // Per Gaussian, from the last Forward: its radius on screen in pixels --
    // three standard deviations along its longest axis -- or -1 where it was
    // not in view. What the paper's screen-size pruning measures: a Gaussian
    // that covers a large part of the image is almost always a floater
    // stretched over a region nothing else covers.
    const std::vector<double>& ScreenRadius() const { return m_screenRadius; }

private:
    // What projection works out once per Gaussian per image.
    struct Proj {
        bool   ok = false;
        double u = 0, v = 0;           // centre, pixels
        double depth = 0;              // camera-space z
        double conic[3] = {0, 0, 0};   // inverse 2D covariance: a, b, c
        double reach = 0;              // pixels out to where alpha < minAlpha
        double cov2[3] = {0, 0, 0};    // 2D covariance: A, B, C
        double tc[3] = {0, 0, 0};      // camera-space centre
        double J[6] = {0, 0, 0, 0, 0, 0};   // projection Jacobian, 2x3
        double M[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};  // W Sigma W^T
        double sigma[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};  // world covariance
        double Rq[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};     // rotation
        double scale[3] = {0, 0, 0};
        double qn[4] = {1, 0, 0, 0};   // normalised quaternion
        double qlen = 1.0;
        double opac = 0.0;             // sigmoid(opacity logit)
        double radius = 0.0;           // 3 sigma along the long axis, pixels
        int    x0 = 0, y0 = 0, x1 = 0, y1 = 0;   // tile rectangle, inclusive
    };

    // The 2D quantities the per-pixel backward pass produces, per Gaussian.
    struct Grad2 {
        double u = 0, v = 0;
        double conic[3] = {0, 0, 0};
        double color[3] = {0, 0, 0};
        double opacity = 0;   // w.r.t. the logit
        double depth = 0;     // w.r.t. the centre's camera-space depth
        void Add(const Grad2& o) {
            u += o.u; v += o.v;
            for (int i = 0; i < 3; ++i) { conic[i] += o.conic[i]; color[i] += o.color[i]; }
            opacity += o.opacity;
            depth += o.depth;
        }
    };

    void Project(const std::vector<SplatParam>& splats, const SplatCam& cam,
                 const RasterOptions& opt);
    void BinTiles(const SplatCam& cam);

    // The GPU halves of Forward and Backward. Each returns false, having
    // changed nothing the CPU path depends on, if the device fails.
    struct Gpu;
    bool CompositeGpu(const std::vector<SplatParam>& splats, const SplatCam& cam,
                      const RasterOptions& opt, std::vector<double>* rgb,
                      std::vector<double>* depth);
    bool BackPixelGpu(const SplatCam& cam, const RasterOptions& opt,
                      const std::vector<double>& dRgb,
                      const std::vector<double>* dDepth,
                      const std::vector<float>* depthShift,
                      const std::vector<size_t>& offset,
                      std::vector<Grad2>* entry);
    std::unique_ptr<Gpu> m_gpu;
    bool        m_usedGpu = false;
    std::string m_gpuNote;

    static constexpr int kTile = 16;

    std::vector<Proj>              m_proj;
    std::vector<std::vector<int>>  m_tiles;      // per tile, Gaussians near to far
    int                            m_tilesX = 0, m_tilesY = 0;
    std::vector<double>            m_finalT;     // per pixel
    std::vector<int>               m_lastIdx;    // per pixel: list entries used
    std::vector<double>            m_screenGrad; // per Gaussian, see ScreenGrad
    std::vector<double>            m_screenRadius;   // see ScreenRadius
    Timings                        m_time;
    int                            m_visible = 0;
};

}  // namespace tglab
