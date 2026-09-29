// train_splats — fit the Gaussians to the photographs.
//
// This is what makes Gaussian splatting what it is. init_splats gives every
// dense point a disc; this moves, reshapes, turns, recolours and fades those
// discs until rendering them from each camera reproduces that camera's
// photograph. The loop, from Kerbl et al. 2023:
//
//   pick a camera, render the splats from it, measure the difference from its
//   photograph, backpropagate that difference into every Gaussian's
//   parameters, take an Adam step -- and repeat, cycling through the cameras.
//
// The rendering and the gradients are SplatRaster, whose backward pass is
// checked against finite differences in test_sfm.cpp. This file is only the
// loop around it: the loss, the optimiser and the bookkeeping.
//
// DENSIFICATION (the paper's adaptive density control) runs every
// `densify_every` iterations: Gaussians the optimiser keeps pushing around the
// screen are cloned if small or split if large, and near-transparent ones are
// removed. Measured on fountain-P11 (508k Gaussians, 320x213, 11 views):
//
//   iterations   Gaussians after   PSNR before -> after   time
//          100           511,012        12.83 -> 13.73       85 s
//          300           515,437        12.83 -> 17.74      248 s
//
// And on a synthetic scene started from twelve coarse blobs where the truth
// has sixty small Gaussians, 600 iterations reach 35.9 dB with it and 20.9
// without -- the case it exists for.
//
// FLOATER REMOVAL, as the paper does it: after the first opacity reset,
// Gaussians larger than 10% of the scene extent or 20 pixels on screen are
// pruned, and a periodic opacity reset lets the unneeded ones fade and be
// pruned as faint. Measured on fountain-P11, 300 iterations:
//
//   pruning                 pruned as too large   PSNR after
//   none                                      0     17.74 dB
//   paper's limits (20 px)                    3     17.63 dB
//   8 px                                    646     16.90 dB
//
// The paper's limits barely fire here, and a tighter one costs fit, because
// the soft blobs this was meant to remove are mostly NOT floaters. They sit
// where the photographs show the white wall on the right, which the dense
// cloud never covered: the optimiser is painting a missing surface with the
// few Gaussians it has nearby. Pruning them only brings back the black gap.
// The fix for that is coverage -- Gaussians where the surface is -- not
// removal. The 8-pixel limit did clear one genuine floater, a grey smudge in
// front of the basin's edge.
//
// WHAT THIS DOES NOT DO YET, stated so the gap is visible:
//
//   * L1 loss only. The paper mixes in 20% D-SSIM, which sharpens structure;
//     L1 alone is simpler to verify and is most of the signal.
//   * colour does not vary with viewing direction. The paper's higher
//     spherical-harmonic bands do that; they matter for shiny surfaces and
//     little for stone.
//   * the CPU only, and it is slow -- see the `downscale` parameter.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "../../algo_util/depth_views.h"
#include "../../algo_util/splat_raster.h"
#include "../../algo_util/splat_train_gpu.h"
#include "../../core/algorithm.h"
#include "../../core/parallel.h"

namespace tglab {
namespace {

// One training view: a camera and its photograph at training resolution.
struct View {
    SplatCam            cam;
    std::vector<double> rgb;   // w*h*3, 0..1
    // The plane sweep's depth at training resolution, w*h, 0 where there is
    // no confident measurement. Empty without a depth input.
    std::vector<float>  depth;
};

// A camera's sweep depth resampled to the training view's w x h, nearest
// pixel by pixel centre, keeping only depths at or above `minConf`. The
// sweep ran on the same frames at their own resolution, so the two cover
// the same field of view and relative coordinates line up.
//
// Nearest rather than averaged: averaging depths across an edge invents a
// surface halfway between the two that neither camera saw.
std::vector<float> DepthTarget(const DepthView& dv, int w, int h, double minConf) {
    std::vector<float> out(size_t(w) * size_t(h), 0.0f);
    if (!dv.ok) return out;
    for (int y = 0; y < h; ++y) {
        const int sy = std::min(dv.h - 1, int((double(y) + 0.5) * dv.h / h));
        for (int x = 0; x < w; ++x) {
            const int sx = std::min(dv.w - 1, int((double(x) + 0.5) * dv.w / w));
            const float d = dv.At(sx, sy);
            if (d > 0.0f && dv.Conf(sx, sy) >= float(minConf))
                out[size_t(y) * size_t(w) + size_t(x)] = d;
        }
    }
    return out;
}

// Reads a frame as RGB and box-downsamples it by an integer factor.
//
// A box filter rather than point sampling, because training compares against
// these pixels: a point-sampled target is aliased, and the splats would learn
// the aliasing.
bool LoadView(const Image& img, int k, std::vector<double>* rgb, int* ow,
              int* oh) {
    ImageView v = const_cast<Image&>(img).MapCpuRead();
    if (!v.Valid()) return false;
    const int w = v.desc.width / k, h = v.desc.height / k;
    if (w < 8 || h < 8) return false;
    rgb->assign(size_t(w) * size_t(h) * 3, 0.0);

    auto px = [&](int x, int y, double* c) {
        switch (v.desc.format) {
            case Format::RGBA8: {
                const uint8_t* p = v.At<uint8_t>(x, y);
                for (int i = 0; i < 3; ++i) c[i] = double(p[i]) / 255.0;
                break;
            }
            case Format::RGBA32F: {
                const float* p = v.At<float>(x, y);
                for (int i = 0; i < 3; ++i) c[i] = double(p[i]);
                break;
            }
            case Format::RGBA16F: {
                const uint16_t* p = v.At<uint16_t>(x, y);
                for (int i = 0; i < 3; ++i) c[i] = double(HalfToFloat(p[i]));
                break;
            }
            case Format::R32F: {
                const double g = double(*v.At<float>(x, y));
                c[0] = c[1] = c[2] = g;
                break;
            }
            default: c[0] = c[1] = c[2] = 0.0; break;
        }
    };

    const double inv = 1.0 / double(k * k);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            double s[3] = {0, 0, 0}, c[3];
            for (int dy = 0; dy < k; ++dy)
                for (int dx = 0; dx < k; ++dx) {
                    px(x * k + dx, y * k + dy, c);
                    s[0] += c[0]; s[1] += c[1]; s[2] += c[2];
                }
            double* o = &(*rgb)[(size_t(y) * size_t(w) + size_t(x)) * 3];
            o[0] = s[0] * inv; o[1] = s[1] * inv; o[2] = s[2] * inv;
        }
    *ow = w;
    *oh = h;
    return true;
}

// Mean absolute error and PSNR of a render against its photograph.
// THE PIXELS TRAINING IS ALLOWED TO FIT: those the starting Gaussians --
// the fused dense cloud -- cover when projected into this view, grown by
// `grow` pixels. Everything else has its target colour set to -1, which the
// loss treats as background (see the random background in the loop) and
// every score skips.
//
// WHY. A photograph shows more than the reconstruction holds -- the wall
// behind a face, sky over a building -- and with nothing in the scene to
// explain those pixels, training grows and stretches Gaussians near the
// subject until they paint the background in. From any other viewpoint
// those are the smears and blobs around the edge.
//
// FROM THE FUSED CLOUD, not the depth maps. A single view's depth map still
// holds speckles of background it measured on its own; grown, they covered
// most of a selfie's wall and left only 28% masked. The fused points are
// the ones several cameras agreed on, which is exactly the subject.
//
// GROWN because the cloud has thin gaps ON the subject -- mortar lines on a
// brick wall, a horizontal edge the camera moved along -- and those should
// still be fitted. A few pixels closes them; a background region is far
// wider and stays out.
void MaskUncovered(const std::vector<Splat>& points, const SplatCam& cam, int grow,
                   std::vector<double>* rgb) {
    const int w = cam.w, h = cam.h;
    std::vector<uint8_t> in(size_t(w) * size_t(h), 0);
    for (const Splat& s : points) {
        const Vec3 q = cam.R * s.mean + cam.t;
        if (q.z <= 1e-9) continue;
        const int x = int(cam.fx * q.x / q.z + cam.cx);
        const int y = int(cam.fy * q.y / q.z + cam.cy);
        if (x >= 0 && y >= 0 && x < w && y < h) in[size_t(y) * size_t(w) + size_t(x)] = 1;
    }
    // Dilation by a square of side 2*grow+1, separably: rows, then columns.
    std::vector<uint8_t> tmp(in.size(), 0);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            uint8_t v = 0;
            for (int d = -grow; d <= grow && !v; ++d) {
                const int xx = x + d;
                if (xx >= 0 && xx < w) v = in[size_t(y) * size_t(w) + size_t(xx)];
            }
            tmp[size_t(y) * size_t(w) + size_t(x)] = v;
        }
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            uint8_t v = 0;
            for (int d = -grow; d <= grow && !v; ++d) {
                const int yy = y + d;
                if (yy >= 0 && yy < h) v = tmp[size_t(yy) * size_t(w) + size_t(x)];
            }
            if (!v)
                for (int ch = 0; ch < 3; ++ch)
                    (*rgb)[(size_t(y) * size_t(w) + size_t(x)) * 3 + size_t(ch)] = -1.0;
        }
}

// L1 and PSNR over the pixels with a target (see MaskUnmeasured).
void Score(const std::vector<double>& a, const std::vector<double>& b,
           double* l1, double* psnr) {
    double s1 = 0.0, s2 = 0.0;
    size_t used = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (b[i] < 0.0) continue;
        const double d = a[i] - b[i];
        s1 += std::fabs(d);
        s2 += d * d;
        ++used;
    }
    const double n = double(std::max<size_t>(1, used));
    *l1 = s1 / n;
    const double mse = s2 / n;
    *psnr = (mse > 1e-20) ? 10.0 * std::log10(1.0 / mse) : 99.0;
}

class TrainSplats : public AlgorithmBase {
public:
    const char* Name()     const override { return "train_splats"; }
    const char* Category() const override { return "sfm"; }

    PortList Inputs() const override {
        // `depth`, optional: plane_sweep's maps, to supervise geometry as
        // well as colour. See depth_weight.
        return {{"src",    DataType::PointCloud, FormatSpec::Any, ShapeSpec::Any},
                {"frames", DataType::ImageSet,   FormatSpec::Any, ShapeSpec::Any},
                {"depth",  DataType::ImageSet,   FormatSpec::Any, ShapeSpec::Any,
                 true}};
    }
    PortList Outputs() const override {
        return {{"out", DataType::PointCloud, FormatSpec::Any, ShapeSpec::Any}};
    }

    void RunCPU(RunCtx&) override {}
    bool IsReconstruct() const override { return true; }
    ProxyBehaviour Proxy() const override { return ProxyBehaviour::Never; }
    bool HasGPU() const override { return false; }

    bool RunReconstruct(const std::vector<Image>* images, PointCloud* cloud,
                        std::string* err) override {
        const auto t0 = std::chrono::steady_clock::now();
        if (cloud->splats.empty()) {
            *err = "train_splats: no splats -- run init_splats first";
            return false;
        }
        if (!images || images->size() != cloud->cameras.size()) {
            *err = "train_splats: needs the frames the cameras were solved "
                   "from as its second input -- train_splats(splats, frames)";
            return false;
        }

        // --- training views, and any held out ----------------------------------
        // HELD-OUT VIEWS are the honest score. PSNR on the photographs being
        // trained on rewards a fit that is perfect from those viewpoints and
        // wrong between them -- which is exactly what a few-view capture
        // produces. A camera left out of training says how the splats look
        // from somewhere they were not fitted to.
        const int k = std::clamp(int(m_downscale), 1, 16);
        const int holdEvery = std::max(0, int(m_holdout));

        // DEPTH, when the third input carries plane_sweep's maps.
        std::vector<DepthView> depthViews;
        std::vector<ImageView> depthHolds;   // keep the mappings alive
        const double depthWeight = double(m_depthWeight);
        const bool useDepth = ReconstructExtra() && depthWeight > 0.0;
        // The mask marks where the dense cloud is, so it needs the dense
        // path; the depth input is what says this is one.
        const bool useMask = ReconstructExtra() && bool(m_mask);
        if (useDepth &&
            !ReadDepthViews(*ReconstructExtra(), int(cloud->cameras.size()),
                            "train_splats", &depthViews, &depthHolds, err))
            return false;
        std::vector<View> views, heldOut;
        size_t maskedPx = 0, totalPx = 0;   // for the report
        for (size_t i = 0; i < cloud->cameras.size(); ++i) {
            const Camera& c = cloud->cameras[i];
            if (!c.solved) continue;
            View vw;
            int w = 0, h = 0;
            if (!LoadView((*images)[i], k, &vw.rgb, &w, &h)) continue;
            vw.cam = SplatCamFrom(c, w, h);
            if (useDepth)
                vw.depth = DepthTarget(depthViews[i], w, h, double(m_depthConfidence));
            if (useMask) {
                MaskUncovered(cloud->splats, vw.cam, std::max(0, int(m_maskGrow)), &vw.rgb);
                for (double v : vw.rgb) if (v < 0.0) ++maskedPx;
                totalPx += vw.rgb.size();
            }
            // Every Nth camera from the middle of each run of N, so the held-
            // out ones sit BETWEEN training cameras: interpolation, which is
            // the question. Holding out an end camera would ask about
            // extrapolation instead, which no amount of fitting fixes.
            const bool hold = holdEvery >= 2 && int(i) % holdEvery == holdEvery / 2;
            (hold ? heldOut : views).push_back(std::move(vw));
        }
        if (views.empty()) {
            *err = "train_splats: no solved camera has a usable frame";
            return false;
        }

        std::vector<SplatParam> params;
        params.reserve(cloud->splats.size());
        for (const Splat& s : cloud->splats) params.push_back(ToParam(s));

        // THE CAP APPLIES TO THE START TOO. max_gaussians used to limit only
        // what densification added, so a dense cloud bigger than it went
        // straight through -- and a hundred-frame video fused to 2.6 million
        // points, whose GPU state (13 texels a Gaussian) needs a texture past
        // Direct3D's 16384-row limit. Training fell back to the CPU, which at
        // that size is hours. Thinned evenly, with each kept Gaussian widened
        // to cover the area of those it stands for: thinning by r leaves each
        // flat disc r times the surface, so its two larger axes grow by
        // sqrt(r).
        const size_t cap = size_t(std::max(1, int(m_maxGaussians)));
        size_t thinnedFrom = 0;
        if (params.size() > cap) {
            thinnedFrom = params.size();
            const double r = double(params.size()) / double(cap);
            const double grow = 0.5 * std::log(r);
            std::vector<SplatParam> kept;
            kept.reserve(cap);
            for (size_t k = 0; k < cap; ++k) {
                SplatParam p = params[size_t(double(k) * r)];
                int flat = 0;   // the thin axis stays as it is
                for (int a = 1; a < 3; ++a)
                    if (p.logScale[a] < p.logScale[flat]) flat = a;
                for (int a = 0; a < 3; ++a)
                    if (a != flat) p.logScale[a] += grow;
                kept.push_back(p);
            }
            params.swap(kept);
        }
        size_t n = params.size();   // changes as densification adds and prunes

        RasterOptions opt;
        const double bg = double(m_background);
        opt.background = Vec3{bg, bg, bg};

        // --- how well it fits before -----------------------------------------
        double l1Before = 0.0, psnrBefore = 0.0;
        double depthBefore = -1.0, depthAfter = -1.0;
        Evaluate(params, views, opt, GroupGpu(), &l1Before, &psnrBefore,
                 useDepth ? &depthBefore : nullptr);
        double heldBefore = 0.0, heldAfter = 0.0, heldL1 = 0.0;
        if (!heldOut.empty())
            Evaluate(params, heldOut, opt, GroupGpu(), &heldL1, &heldBefore);

        // --- the scene's scale, for the position learning rate ------------------
        // A step size for positions has to be in scene units, and a
        // reconstruction's units are arbitrary. The paper scales by the
        // radius of the cameras about their centre, times 1.1.
        double extent = 0.0;
        {
            Vec3 mean{0, 0, 0};
            int nc = 0;
            for (const Camera& c : cloud->cameras)
                if (c.solved) { mean = mean + c.Center(); ++nc; }
            mean = mean * (1.0 / double(std::max(1, nc)));
            for (const Camera& c : cloud->cameras)
                if (c.solved) extent = std::max(extent, (c.Center() - mean).Norm());
            extent = std::max(1e-6, extent * 1.1);
        }

        // Zero is allowed and means "measure only": the report then says how
        // well the untrained splats fit, which is the baseline every run of
        // training should be compared with.
        const int iters = std::max(0, int(m_iterations));

        // --- Adam ---------------------------------------------------------------
        // Learning rates from the paper, per parameter group. Positions decay
        // exponentially by a factor of 100 over the run, as the paper's do
        // over 30000 steps: large moves early to get things roughly placed,
        // then small ones so the fine fit is not shaken apart.
        const double lrMean0 = 1.6e-4 * extent * double(m_lrScale);
        const double lrMean1 = 1.6e-6 * extent * double(m_lrScale);
        const double lrGroup[5] = {0.0, 5e-3, 1e-3, 5e-2, 2.5e-3};  // mean set per step
        const int groupOf[14] = {0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 4, 4, 4};
        const double b1 = 0.9, b2 = 0.999, adamEps = 1e-15;

        std::vector<double> m1(n * 14, 0.0), m2(n * 14, 0.0);
        std::vector<SplatParam> grad(n, SplatParam::Zero());
        std::vector<double> img, dImg;
        std::vector<double> maskedTarget;   // see the random background below
        std::vector<double> rDepth, dDepth;          // the CPU path's depth term
        double    cpuDepthErr = 0.0;
        long long cpuDepthCount = 0;
        SplatRaster raster;
        // The per-pixel passes on the device when the run has one; the
        // rasteriser falls back to the CPU on its own if the device fails.
        raster.SetGpu(GroupGpu());

        // --- densification bookkeeping ------------------------------------------
        // The screen-position gradient, summed per Gaussian over the
        // iterations it was in view, and how many those were. Averaged at
        // each densification step and reset after it.
        std::vector<double> gradAccum(n, 0.0);
        std::vector<int>    gradCount(n, 0);
        // The largest each Gaussian has appeared on screen since the last
        // densification step, in training pixels.
        std::vector<double> maxScreen(n, 0.0);
        const double worldLimit  = double(m_maxWorldSize) * extent;
        const double screenLimit = double(m_maxScreenSize);
        const int    resetEvery  = std::max(0, int(m_opacityResetEvery));
        long long prunedFaint = 0, prunedBig = 0;
        int resets = 0;
        const bool densify = bool(m_densify) && iters > 0;
        const int  every = std::max(1, int(m_densifyEvery));
        const int  until = int(double(iters) * double(m_densifyUntil));
        const double threshold = double(m_gradThreshold);
        const size_t maxCount = size_t(std::max(1, int(m_maxGaussians)));
        const double pruneOpacity = double(m_pruneOpacity);
        const bool   clampColour  = bool(m_clampColour);
        long long cloned = 0, split = 0, pruned = 0;
        int densifySteps = 0;
        int pruneSteps   = 0;   // after densify_until: prune, never grow

        // Cameras in a shuffled order each pass, as the paper does, so no
        // camera is always followed by the same one. A fixed seed keeps runs
        // reproducible.
        std::vector<int> order;
        uint32_t seed = 0x9E3779B9u;
        auto rnd = [&](uint32_t mod) {
            seed = seed * 1664525u + 1013904223u;
            return (seed >> 8) % mod;
        };

        // A standard normal, for sampling split positions. Box-Muller on the
        // same generator as the camera order, so a run is reproducible.
        auto normal = [&]() {
            const double u1 = (double(rnd(1u << 23)) + 0.5) / double(1u << 23);
            const double u2 = (double(rnd(1u << 23)) + 0.5) / double(1u << 23);
            return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
        };

        // Where each iteration goes, beside the rasteriser's own phases.
        using Clock = std::chrono::steady_clock;
        auto msSince = [](Clock::time_point t) {
            return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
        };
        double msAdam = 0.0, msDensify = 0.0;
        long long visibleSum = 0;

        // --- the whole state on the device, when there is one -----------------
        //
        // See SplatTrainerGpu. The CPU loop below stays as the reference and
        // the fallback: if the device fails part way, the state is read back
        // from the last good step and training carries on here.
        std::unique_ptr<SplatTrainerGpu> gpuTrainer;
        std::string gpuNote;
        if (ComputeContext* dev = GroupGpu()) {
            auto t = std::make_unique<SplatTrainerGpu>(dev);
            t->SetClampColour(clampColour);
            if (t->Upload(params, m1, m2, gradAccum, gradCount, maxScreen, &gpuNote))
                gpuTrainer = std::move(t);
        }
        int gpuIters = 0;

        for (int it = 0; it < iters; ++it) {
            if (order.empty()) {
                order.resize(views.size());
                for (size_t i = 0; i < views.size(); ++i) order[i] = int(i);
                for (size_t i = order.size(); i > 1; --i)
                    std::swap(order[i - 1], order[rnd(uint32_t(i))]);
            }
            const View& vw = views[size_t(order.back())];
            order.pop_back();

            // MASKED PIXELS TARGET A RANDOM BACKGROUND. Leaving them out of
            // the loss alone stops training painting the background in, but
            // it also makes a Gaussian already sitting there free: nothing
            // pulls on it, so the blobs grown from the subject's edge stayed.
            // Rendering over a colour drawn fresh each step, and asking the
            // masked pixels to BE that colour, can only be met by being
            // transparent there -- so anything in front of the background is
            // driven to zero opacity and pruned. (The trick object-centric
            // NeRF and splatting work uses with masks.) Measured pixels see
            // the same background wherever the Gaussians are not yet opaque,
            // which pushes the subject solid as well.
            const std::vector<double>* target = &vw.rgb;
            if (useMask) {
                const Vec3 bgc{rnd(256) / 255.0, rnd(256) / 255.0, rnd(256) / 255.0};
                opt.background = bgc;
                maskedTarget = vw.rgb;
                for (size_t i = 0; i < maskedTarget.size(); i += 3)
                    if (maskedTarget[i] < 0.0) {
                        maskedTarget[i] = bgc.x;
                        maskedTarget[i + 1] = bgc.y;
                        maskedTarget[i + 2] = bgc.z;
                    }
                target = &maskedTarget;
            }

            const double frac = (iters > 1) ? double(it) / double(iters - 1) : 1.0;
            const double lrMean = lrMean0 * std::pow(lrMean1 / lrMean0, frac);
            const double c1 = 1.0 - std::pow(b1, double(it + 1));
            const double c2 = 1.0 - std::pow(b2, double(it + 1));

            // One step on the device; if it fails, bring the state home and
            // take this step, and every later one, on the CPU.
            bool stepped = false;
            if (gpuTrainer) {
                if (gpuTrainer->Step(vw.cam, opt, *target, lrMean, c1, c2, &gpuNote,
                                     vw.depth.empty() ? nullptr : &vw.depth,
                                     depthWeight)) {
                    visibleSum += gpuTrainer->Visible();
                    ++gpuIters;
                    stepped = true;
                } else {
                    std::string derr;
                    if (!gpuTrainer->Download(&params, &m1, &m2, &gradAccum,
                                              &gradCount, &maxScreen, &derr)) {
                        *err = "train_splats: the GPU failed (" + gpuNote +
                               ") and its state could not be recovered (" + derr + ")";
                        return false;
                    }
                    n = params.size();
                    grad.assign(n, SplatParam::Zero());
                    gpuTrainer.reset();
                }
            }

            if (!stepped) {
            raster.Forward(params, vw.cam, opt, &img,
                           vw.depth.empty() ? nullptr : &rDepth);
            visibleSum += raster.Visible();

            // L1: the gradient is the sign of each difference, over the mean
            // -- of the pixels with a target; masked ones pull nothing.
            dImg.resize(img.size());
            size_t used = 0;
            for (double v : *target) if (v >= 0.0) ++used;
            const double invN = 1.0 / double(std::max<size_t>(1, used));
            for (size_t i = 0; i < img.size(); ++i) {
                if ((*target)[i] < 0.0) { dImg[i] = 0.0; continue; }
                const double d = img[i] - (*target)[i];
                dImg[i] = (d > 0.0 ? invN : (d < 0.0 ? -invN : 0.0));
            }

            std::fill(grad.begin(), grad.end(), SplatParam::Zero());
            if (!vw.depth.empty())
                DepthLossGrad(rDepth, raster.FinalT(), vw.depth, depthWeight, &dDepth,
                              &cpuDepthErr, &cpuDepthCount);
            raster.Backward(params, vw.cam, opt, dImg, &grad,
                            vw.depth.empty() ? nullptr : &dDepth,
                            vw.depth.empty() ? nullptr : &vw.depth);

            // Accumulate the screen-position gradient for densification, in
            // the paper's units. The paper measures it against NDC, which
            // spans half the image width per unit; the rasteriser reports it
            // per pixel. Multiplying by half the width makes the paper's
            // threshold of 0.0002 mean the same thing here.
            if (densify) {
                const std::vector<double>& sg = raster.ScreenGrad();
                const std::vector<double>& sr = raster.ScreenRadius();
                const double toNdc = 0.5 * double(vw.cam.w);
                for (size_t i = 0; i < n; ++i) {
                    if (sg[i] >= 0.0) { gradAccum[i] += sg[i] * toNdc; ++gradCount[i]; }
                    if (sr[i] >= 0.0) maxScreen[i] = std::max(maxScreen[i], sr[i]);
                }
            }

            auto tAdam = Clock::now();
            ParallelFor(n, [&](size_t i) {
                double* p = params[i].Data();
                const double* g = grad[i].Data();
                double* a = &m1[i * 14];
                double* b = &m2[i * 14];
                for (int q = 0; q < 14; ++q) {
                    a[q] = b1 * a[q] + (1.0 - b1) * g[q];
                    b[q] = b2 * b[q] + (1.0 - b2) * g[q] * g[q];
                    const double lr = (groupOf[q] == 0) ? lrMean : lrGroup[groupOf[q]];
                    p[q] -= lr * (a[q] / c1) / (std::sqrt(b[q] / c2) + adamEps);
                }
                if (clampColour)
                    for (int q = 11; q < 14; ++q) p[q] = std::clamp(p[q], 0.0, 1.0);
            });

            msAdam += msSince(tAdam);
            }   // the CPU step

            auto tDens = Clock::now();

            // Densification and the opacity reset work on the CPU copy, so on
            // the GPU path the state comes down before either and goes back
            // up after. Every `densify_every` iterations, not every one.
            // Growth stops at densify_until; PRUNING DOES NOT, see below.
            const bool growing   = it + 1 <= until;
            const bool doDensify = densify && (it + 1) % every == 0;
            const bool doReset = densify && resetEvery > 0 &&
                                 (it + 1) % resetEvery == 0 && it + 1 < until;
            if (gpuTrainer && (doDensify || doReset)) {
                std::string derr;
                if (!gpuTrainer->Download(&params, &m1, &m2, &gradAccum,
                                          &gradCount, &maxScreen, &derr)) {
                    *err = "train_splats: could not read the GPU state back to "
                           "densify: " + derr;
                    return false;
                }
                n = params.size();
            }

            // --- densify and prune ----------------------------------------------
            // PRUNING OUTLIVES GROWTH. After densify_until nothing is cloned
            // or split, but the faint and the oversized are still removed
            // every `densify_every` iterations, to the end.
            //
            // The paper stops both together, and with a hundred-plus views it
            // can: every Gaussian is seen from enough angles that none can
            // grow unnoticed. With eleven it cannot. A flat Gaussian seen
            // edge-on from every training camera is a thin line in every
            // photograph, so nothing in the loss stops its other two axes
            // growing -- log-scale moves up to ~0.005 an iteration, a factor
            // of 400 over 1200 unchecked iterations. At 3000 iterations
            // on fountain-P11 that made sheets spanning the whole scene,
            // invisible from the cameras and a wall of smear from anywhere
            // else. At 300 there were only 120 iterations after the cutoff,
            // too few for it to show.
            //
            // Only the WORLD-size limit applies once growth has stopped; the
            // screen-size one does not. The sheets are small on screen by
            // construction, so the world limit is the one that catches them,
            // while the screen limit then removes big-on-screen Gaussians
            // that nothing can replace any more. Measured on fountain-P11:
            //
            //   late pruning          300 iterations   3000 iterations
            //   none (the paper's)        17.59 dB      sheets across the scene
            //                                           (seen at full resolution)
            //   world and screen          16.63 dB        28.37 dB
            //   world only                17.59 dB        30.92 dB
            if (doDensify) {
                std::vector<SplatParam> np;
                std::vector<double> nm1, nm2;
                np.reserve(n + n / 4);
                nm1.reserve((n + n / 4) * 14);
                nm2.reserve((n + n / 4) * 14);

                // Kept: carries its own optimiser state. New: starts with
                // none, as the paper's new tensors do -- which is also what
                // lets a clone drift apart from the original it copies,
                // since the two then take different steps.
                auto keep = [&](size_t i) {
                    np.push_back(params[i]);
                    nm1.insert(nm1.end(), m1.begin() + long(i * 14), m1.begin() + long(i * 14 + 14));
                    nm2.insert(nm2.end(), m2.begin() + long(i * 14), m2.begin() + long(i * 14 + 14));
                };
                auto fresh = [&](const SplatParam& p) {
                    np.push_back(p);
                    nm1.insert(nm1.end(), 14, 0.0);
                    nm2.insert(nm2.end(), 14, 0.0);
                };

                // WHICH GAUSSIANS MAY GROW: those whose average screen
                // gradient passed the threshold, strongest first, so that
                // when the cap binds it is the most needed ones that divide.
                std::vector<std::pair<double, size_t>> grow;
                for (size_t i = 0; i < n; ++i) {
                    if (gradCount[i] == 0) continue;
                    const double avg = gradAccum[i] / double(gradCount[i]);
                    if (avg >= threshold) grow.emplace_back(avg, i);
                }
                std::sort(grow.begin(), grow.end(), std::greater<>());
                const size_t room = (growing && maxCount > n) ? maxCount - n : 0;
                std::vector<char> grows(n, 0);
                for (size_t k = 0; k < grow.size() && k < room; ++k)
                    grows[grow[k].second] = 1;

                // SMALL MEANS CLONE, LARGE MEANS SPLIT. The paper's dividing
                // line is 1% of the scene extent: a small Gaussian in a region
                // of high error is under-covering it, so a copy is added
                // alongside; a large one is over-covering detail it cannot
                // follow, so it is replaced by two smaller ones drawn from
                // inside it, each 1/1.6 of its size.
                const double smallLimit = 0.01 * extent;
                for (size_t i = 0; i < n; ++i) {
                    const SplatParam& p = params[i];
                    // PRUNE the nearly transparent: they cost a slot in every
                    // tile they touch and contribute nothing.
                    if (1.0 / (1.0 + std::exp(-p.opacity)) < pruneOpacity) {
                        ++pruned;
                        ++prunedFaint;
                        continue;
                    }
                    // PRUNE THE HUGE, which is what floaters are. A Gaussian
                    // the optimiser stretched over a region the initial cloud
                    // never covered -- sky, a blank wall -- lowers the loss a
                    // little by smearing an average colour across it, and
                    // then sits in the air in front of everything, visible
                    // from any other angle. The paper removes them by size in
                    // the world and on screen; both limits are zero to
                    // disable.
                    //
                    // ONLY AFTER THE FIRST OPACITY RESET, as the paper's code
                    // does. Early on a large Gaussian is usually a legitimate
                    // coarse one that has not been split yet, not a floater;
                    // pruning it by size first deletes it before it can
                    // divide. Measured on the coarse-start test: size limits
                    // from the first step pruned all twelve starting blobs,
                    // leaving nothing to train (12.5 dB, against 20.9 with no
                    // densification at all). Without resets, from the second
                    // step on.
                    const bool sizePrune = (resetEvery > 0)
                                               ? (it + 1 > resetEvery)
                                               : (densifySteps > 0);
                    if (sizePrune) {
                        const double big = std::exp(std::max({p.logScale[0],
                                                              p.logScale[1],
                                                              p.logScale[2]}));
                        if ((worldLimit > 0.0 && big > worldLimit) ||
                            (growing && screenLimit > 0.0 &&
                             maxScreen[i] > screenLimit)) {
                            ++pruned;
                            ++prunedBig;
                            continue;
                        }
                    }
                    if (!grows[i]) { keep(i); continue; }

                    const double s[3] = {std::exp(p.logScale[0]),
                                         std::exp(p.logScale[1]),
                                         std::exp(p.logScale[2])};
                    if (std::max({s[0], s[1], s[2]}) <= smallLimit) {
                        keep(i);
                        fresh(p);
                        ++cloned;
                        continue;
                    }

                    const Splat asSplat = FromParam(p);
                    const Mat3 R = asSplat.Rotation();
                    for (int c = 0; c < 2; ++c) {
                        SplatParam child = p;
                        const double z[3] = {normal() * s[0], normal() * s[1],
                                             normal() * s[2]};
                        for (int a = 0; a < 3; ++a)
                            child.mean[a] += R.m[a * 3 + 0] * z[0] +
                                             R.m[a * 3 + 1] * z[1] +
                                             R.m[a * 3 + 2] * z[2];
                        for (int a = 0; a < 3; ++a)
                            child.logScale[a] -= std::log(1.6);
                        fresh(child);
                    }
                    ++split;
                }

                params.swap(np);
                m1.swap(nm1);
                m2.swap(nm2);
                n = params.size();
                grad.assign(n, SplatParam::Zero());
                gradAccum.assign(n, 0.0);
                gradCount.assign(n, 0);
                maxScreen.assign(n, 0.0);
                if (growing) ++densifySteps; else ++pruneSteps;
            }

            msDensify += msSince(tDens);
            // --- opacity reset ---------------------------------------------------
            // Every Gaussian's opacity is capped at 0.01, and its optimiser
            // state for opacity cleared. The ones the photographs need climb
            // straight back; the ones that were only smearing an average over
            // an empty region do not, fall under prune_opacity, and are
            // removed at the next densification step. The paper's other
            // weapon against floaters, and only run while densification is,
            // since pruning is what finishes the job.
            if (densify && resetEvery > 0 && (it + 1) % resetEvery == 0 &&
                it + 1 < until) {
                const double cap = std::log(0.01 / 0.99);
                for (size_t i = 0; i < n; ++i) {
                    params[i].opacity = std::min(params[i].opacity, cap);
                    m1[i * 14 + 10] = 0.0;
                    m2[i * 14 + 10] = 0.0;
                }
                ++resets;
            }

            if (gpuTrainer && (doDensify || doReset)) {
                if (!gpuTrainer->Upload(params, m1, m2, gradAccum, gradCount,
                                        maxScreen, &gpuNote)) {
                    // Nothing lost: the CPU copy is current, so the run simply
                    // continues there.
                    gpuTrainer.reset();
                    grad.assign(n, SplatParam::Zero());
                }
            }
        }

        // The final state, home from the device.
        if (gpuTrainer) {
            std::string derr;
            if (!gpuTrainer->Download(&params, &m1, &m2, &gradAccum, &gradCount,
                                      &maxScreen, &derr)) {
                *err = "train_splats: could not read the trained splats back from "
                       "the GPU: " + derr;
                return false;
            }
            n = params.size();
        }

        // --- how well it fits after ------------------------------------------
        opt.background = Vec3{bg, bg, bg};   // the random one was for training
        double l1After = l1Before, psnrAfter = psnrBefore;
        depthAfter = depthBefore;
        if (iters > 0)
            Evaluate(params, views, opt, GroupGpu(), &l1After, &psnrAfter,
                     useDepth ? &depthAfter : nullptr);
        heldAfter = heldBefore;
        if (iters > 0 && !heldOut.empty())
            Evaluate(params, heldOut, opt, GroupGpu(), &heldL1, &heldAfter);

        const size_t startCount = cloud->splats.size();
        cloud->splats.resize(n);
        for (size_t i = 0; i < n; ++i) cloud->splats[i] = FromParam(params[i]);

        const double secs = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - t0).count();
        char dens[260] = "";
        if (densify)
            std::snprintf(dens, sizeof(dens),
                          "; %d densify steps and %d prune-only: %lld cloned, "
                          "%lld split, %lld "
                          "pruned (%lld faint, %lld too large), %d opacity "
                          "reset%s, %d -> %d Gaussians",
                          densifySteps, pruneSteps, cloned, split, pruned, prunedFaint,
                          prunedBig, resets, resets == 1 ? "" : "s",
                          int(startCount), int(n));
        // Per iteration, so it reads the same at any iteration count.
        char timing[400] = "";
        if (gpuIters > 0 && gpuTrainer) {
            // Stated every time, with the count: a fallback part way through
            // reads as "the GPU is slow" unless the report says it happened.
            const SplatTrainerGpu::Timings& tm = gpuTrainer->Time();
            const double k = 1.0 / double(gpuIters);
            std::snprintf(timing, sizeof(timing),
                          "; %d of %d iterations on the GPU; per iteration: "
                          "project %.0f, bin %.0f, composite %.0f, loss %.0f, "
                          "backward %.0f, update %.0f, densify %.0f ms",
                          gpuIters, iters, tm.project * k, tm.bin * k,
                          tm.composite * k, tm.loss * k, tm.backward * k,
                          tm.update * k, msDensify / double(iters));
        } else if (iters > 0) {
            const SplatRaster::Timings& tm = raster.Time();
            const double k = 1.0 / double(iters);
            // Which path ran, stated every time: a GPU fallback that happens
            // silently reads as "the GPU is slow".
            std::snprintf(timing, sizeof(timing),
                          "; %s; per iteration: project %.0f, "
                          "bin %.0f, composite "
                          "%.0f, back-pixel %.0f, back-sum %.0f, back-Gaussian "
                          "%.0f, Adam %.0f, densify %.0f ms",
                          raster.UsedGpu() ? "pixels on the GPU"
                                           : (gpuNote.empty() ? "on the CPU"
                                                              : ("on the CPU -- GPU failed: " + gpuNote).c_str()),
                          tm.project * k, tm.bin * k, tm.composite * k,
                          tm.backPixel * k, tm.backSum * k, tm.backGaussian * k,
                          msAdam * k, msDensify * k);
        }
        char buf[900];
        std::snprintf(buf, sizeof(buf),
                      "train_splats: %d iterations over %d views at %dx%d, %d "
                      "Gaussians (%.0f in view on average); L1 %.4f -> %.4f, "
                      "PSNR %.2f -> %.2f dB; %.1f s%s%s",
                      iters, int(views.size()), views[0].cam.w, views[0].cam.h,
                      int(n), double(visibleSum) / double(std::max(1, iters)),
                      l1Before, l1After, psnrBefore, psnrAfter, secs, dens,
                      timing);
        m_note = buf;
        if (thinnedFrom > 0)
            m_note += "; started from " + std::to_string(thinnedFrom) +
                      " Gaussians, thinned evenly to max_gaussians";
        if (useDepth) {
            char db[200];
            std::snprintf(db, sizeof(db),
                          "; DEPTH (weight %.3g): mean error against the sweep "
                          "%.2f%% -> %.2f%%",
                          depthWeight, depthBefore * 100.0, depthAfter * 100.0);
            m_note += db;
        }
        if (useMask && totalPx > 0) {
            char mb[160];
            std::snprintf(mb, sizeof(mb),
                          "; %.0f%% of pixels masked out (no depth measured there), "
                          "scores over the rest",
                          100.0 * double(maskedPx) / double(totalPx));
            m_note += mb;
        }
        if (!heldOut.empty()) {
            char hb[200];
            std::snprintf(hb, sizeof(hb),
                          "; HELD OUT (%d view%s, not trained on): PSNR %.2f -> "
                          "%.2f dB",
                          int(heldOut.size()), heldOut.size() == 1 ? "" : "s",
                          heldBefore, heldAfter);
            m_note += hb;
        }
        return true;
    }

    std::string RunReport() const override { return m_note; }

private:
    // Mean L1 and PSNR over every view: the number that says whether
    // training helped, measured on all the cameras rather than the last few
    // it happened to step on. `depthErr`, when given, receives the mean
    // RELATIVE depth error against the views' sweep depth, over the pixels
    // that have one -- or -1 if none do.
    static void Evaluate(const std::vector<SplatParam>& params,
                         const std::vector<View>& views,
                         const RasterOptions& opt, ComputeContext* gpu,
                         double* l1, double* psnr, double* depthErr = nullptr) {
        SplatRaster r;
        r.SetGpu(gpu);
        std::vector<double> img, dep, unused;
        double sl = 0.0, sp = 0.0, de = 0.0;
        long long dn = 0;
        for (const View& v : views) {
            const bool withDepth = depthErr && !v.depth.empty();
            r.Forward(params, v.cam, opt, &img, withDepth ? &dep : nullptr);
            double a = 0.0, b = 0.0;
            Score(img, v.rgb, &a, &b);
            sl += a;
            sp += b;
            if (withDepth)
                DepthLossGrad(dep, r.FinalT(), v.depth, 1.0, &unused, &de, &dn);
        }
        *l1 = sl / double(views.size());
        *psnr = sp / double(views.size());
        if (depthErr) *depthErr = dn > 0 ? de / double(dn) : -1.0;
    }

    Param<int> m_iterations{this, "iterations", 300, 0, 100000,
        {.help = "Optimisation steps, each rendering one camera and updating "
                 "every Gaussian. The paper runs 30000 on a GPU; on the CPU a "
                 "few hundred already shows the fit improving. 0 trains "
                 "nothing and only reports how well the splats fit.",
         .softMax = 3000.0}};

    // DEPTH SUPERVISION, when the optional third input carries plane_sweep's
    // maps: train_splats(splats, frames, dense).
    //
    // With few photographs, colour alone under-determines geometry: a
    // Gaussian can sit in front of or behind the surface and still paint the
    // right colour into the views that see it, and from anywhere else it is
    // in the wrong place. Rendering each pixel's expected depth and pulling
    // it toward the sweep's pins Gaussians to the surface the cameras
    // measured. Where the sweep measured nothing -- a blank wall -- there is
    // no depth term, and colour alone decides as before.
    //
    // MEASURED on fountain-P11 with cameras 2 and 7 held out (holdout = 5),
    // PSNR on the training views / the held-out ones:
    //
    //   iterations   depth_weight   training   held out
    //         1000              0    24.97      22.82
    //         1000            0.1    25.40      23.40
    //         1000            0.3    24.40      23.98
    //         1000            0.5    24.02      23.86
    //         3000              0    31.17      23.32
    //         3000            0.3    30.66      24.59
    //         3000            1.0    27.39      22.55
    //
    // 0.3 is best at both lengths, and at 1000 iterations it narrows the gap
    // between the photographs trained on and the ones held out from 2.2 dB
    // to 0.4 -- the gap that IS overfitting. Stronger, and the sweep's own
    // errors start to win over the photographs.
    // See MaskUnmeasured. Needs the depth input; without it every pixel trains.
    Param<bool> m_mask{this, "mask", true,
        "Fit only the pixels the dense cloud covers (grown by "
        "mask_grow). Background with no geometry behind it -- a wall behind "
        "a face -- is then left out rather than painted in by Gaussians "
        "stretched around the subject. Needs the depth input."};
    Param<int> m_maskGrow{this, "mask_grow", 4, 0, 32,
        {.help = "Pixels, at training resolution, the measured region is "
                 "grown by before masking, so thin unmeasured gaps on the "
                 "subject -- mortar lines, edges along the camera's motion -- "
                 "are still fitted. Wide unmeasured regions stay out."}};

    Param<float> m_depthWeight{this, "depth_weight", 0.3f, 0.0f, 10.0f,
        {.help = "How strongly rendered depth is pulled toward the plane "
                 "sweep's, relative to the colour loss. The error is "
                 "relative (a fraction of the distance), so the weight means "
                 "the same at any scene scale. Used only when the depth maps "
                 "are passed as the third input; 0 disables.",
         .step = 0.05, .softMax = 2.0}};

    Param<float> m_depthConfidence{this, "depth_confidence", 0.0f, 0.0f, 1.0f,
        {.help = "Lowest sweep confidence a depth may have to be used as a "
                 "target. Raise it if a wrong sweep depth -- repetitive "
                 "texture matched to the wrong place -- is pulling the "
                 "splats with it.",
         .step = 0.05}};

    Param<int> m_holdout{this, "holdout", 0, 0, 16,
        {.help = "Leave every Nth camera out of training and report PSNR on "
                 "those separately: how the splats look from viewpoints they "
                 "were not fitted to. The honest score for a few-view capture, "
                 "where training-view PSNR rewards overfitting. 0 trains on "
                 "every camera."}};

    Param<int> m_downscale{this, "downscale", 2, 1, 16,
        {.help = "Train on the frames shrunk by this factor. The cost is "
                 "proportional to the pixel count, so 2 is four times faster "
                 "than 1; the Gaussians still live in full-resolution space."}};

    Param<float> m_lrScale{this, "position_lr", 1.0f, 0.0f, 10.0f,
        {.help = "Multiplier on the position learning rate. 0 freezes the "
                 "Gaussians in place and trains only their shape, opacity and "
                 "colour -- useful for seeing what each contributes.",
         .step = 0.1, .softMax = 4.0}};

    Param<float> m_background{this, "background", 0.0f, 0.0f, 1.0f,
        {.help = "Grey level wherever no Gaussian covers a pixel, in the "
                 "render compared with the photograph.",
         .step = 0.05}};

    // --- adaptive density control (Kerbl et al. 2023, section 5.2) -------------
    //
    // Without it training can only reshape the Gaussians it started with. A
    // region the dense cloud covered thinly stays thin, and fine detail under
    // one big Gaussian stays blurred, however long it runs.
    Param<bool> m_densify{this, "densify", true,
        "Periodically split Gaussians too large for the detail under them, "
        "clone small ones in under-covered regions, and remove ones that have "
        "faded to near-transparent. The count changes as training runs."};

    // THE PAPER DENSIFIES EVERY 100 OF 30000 ITERATIONS; a CPU run is a few
    // hundred in total, so the interval is shorter to let it happen at all.
    // Too short and a Gaussian is judged on the gradient of a handful of
    // views, which is noise.
    Param<int> m_densifyEvery{this, "densify_every", 50, 5, 5000,
        {.help = "Iterations between densification steps. Each step judges a "
                 "Gaussian on the gradients accumulated since the last, so "
                 "shorter intervals decide on less evidence.",
         .softMax = 500.0}};

    // Stop before the end, as the paper does at half way: the Gaussians a
    // late step creates need iterations of their own to settle, and without
    // them the final render is worse than if they had not been made.
    Param<float> m_densifyUntil{this, "densify_until", 0.6f, 0.0f, 1.0f,
        {.help = "Fraction of the run after which the count is frozen, so the "
                 "last Gaussians created have time to be fitted.",
         .step = 0.05}};

    Param<float> m_gradThreshold{this, "grad_threshold", 0.0002f, 0.00001f, 0.01f,
        {.help = "Average screen-position gradient, in the paper's NDC units, "
                 "above which a Gaussian is split or cloned. Lower grows the "
                 "set more aggressively.",
         .step = 0.00005, .softMax = 0.002}};

    // A CAP THE PAPER DOES NOT HAVE, because the paper is not on a CPU. Every
    // Gaussian costs time on every iteration, so an unbounded set grows the
    // cost of the run as it goes. When the cap binds, the Gaussians with the
    // largest gradients divide first.
    Param<int> m_maxGaussians{this, "max_gaussians", 1000000, 1, 100000000,
        {.help = "Upper limit on the number of Gaussians. Training on the CPU "
                 "costs time in proportion to it.",
         .softMax = 5000000.0}};

    // THE PAPER'S LIMITS: 10% of the scene extent in the world, and 20 pixels
    // on screen at training resolution.
    //
    // The world limit has a known weakness, inherited with it: "extent" is
    // how far the CAMERAS spread, which only measures the scene when the
    // cameras surround it. For a capture whose cameras sit close together in
    // front of a large subject, the limit comes out smaller than the real
    // Gaussians and deletes them. The synthetic test in test_sfm.cpp is such a
    // capture -- cameras 0.8 apart, true Gaussians up to 0.18 across, limit
    // 0.044 -- and disables it for that reason.
    Param<float> m_maxWorldSize{this, "max_world_size", 0.1f, 0.0f, 1.0f,
        {.help = "Gaussians larger than this fraction of the scene's extent "
                 "(the spread of the cameras) are removed. Wrong for captures "
                 "whose cameras are close together relative to the subject -- "
                 "set 0 there. 0 disables.",
         .step = 0.01}};

    // IN PIXELS, as the paper states it, not as a fraction of the width. A
    // fraction was tried first and on a 64-pixel test image came out at three
    // pixels -- smaller than the true Gaussians, which it then deleted.
    Param<float> m_maxScreenSize{this, "max_screen_size", 20.0f, 0.0f, 1000.0f,
        {.help = "Gaussians whose on-screen radius, in training pixels, has "
                 "exceeded this since the last densification step are "
                 "removed. Big faint blobs over blank regions are what this "
                 "catches. 0 disables.",
         .step = 1.0, .softMax = 100.0}};

    // The paper resets every 3000 of its 30000 iterations. A CPU run is a
    // few hundred, so the default here is one reset partway through -- and
    // it must come a full densify interval before `densify_until`, or the
    // Gaussians it fades are never pruned.
    Param<int> m_opacityResetEvery{this, "opacity_reset_every", 100, 0, 100000,
        {.help = "Iterations between opacity resets, while densification "
                 "runs. Every opacity is dropped to 0.01; Gaussians the "
                 "photographs need climb back, and the rest fade and are "
                 "pruned. 0 disables.",
         .softMax = 3000.0}};

    Param<float> m_pruneOpacity{this, "prune_opacity", 0.005f, 0.0f, 0.5f,
        {.help = "Gaussians whose opacity has fallen below this are removed at "
                 "each densification step.",
         .step = 0.005, .softMax = 0.1}};

    // COLOUR IS KEPT BETWEEN 0 AND 1, clamped after every step.
    //
    // Unclamped, the optimiser learns to CANCEL: a Gaussian four times too
    // red behind a darker one in front blends to the right colour from the
    // training views. From any other view the pair no longer lines up and
    // the excess shows -- saturated red on a corner of the fountain, as seen
    // after 3000 iterations. Measured there before clamping: 43333 of 540229
    // Gaussians had a channel above 1.05 (up to 4.24), 3309 below -0.05
    // (down to -0.80). The paper allows this -- it clamps only at 0 -- but it
    // has a hundred-plus views and view-dependent colour to absorb it; with
    // eleven views and one colour per Gaussian, the cancellation is the fit.
    Param<bool> m_clampColour{this, "clamp_colour", true,
        "Keep every Gaussian's colour within 0..1. Off lets the optimiser use "
        "over-bright colours that cancel against others -- a slightly better "
        "fit to the photographs, and discoloured patches from any other "
        "viewpoint."};

    std::string m_note;
};

REGISTER_ALGORITHM(TrainSplats);

}  // namespace
}  // namespace tglab
