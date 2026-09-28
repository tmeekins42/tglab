// splat_train_gpu — Gaussian-splat training with its state on the GPU.
//
// The step after moving only the per-pixel passes. That halved an iteration,
// and what was left was the per-Gaussian work -- projecting, the chain rule,
// Adam -- plus the traffic between the two halves: every iteration uploaded
// 24 MB of projected Gaussians and read back ~72 MB of per-entry gradients.
//
// Here the parameters, both Adam moments and the densification counters live
// on the device for the whole run, and an iteration is:
//
//   project        GPU, one thread per Gaussian
//   bin and sort   CPU, threaded, from an 8 MB readback of screen positions
//   composite      GPU  (the kernel SplatRaster uses)
//   loss           CPU, from the rendered image -- a megabyte each way
//   backward       GPU, per pixel (SplatRaster's kernel), then per Gaussian:
//                  sum each Gaussian's per-tile gradients, apply the chain
//                  rule, take the Adam step
//
// BINNING STAYS ON THE CPU for now, on purpose. Sorting on the GPU takes many
// dispatches, and many dispatches is exactly the pattern that has hung this
// machine. What crosses the bus for it is small; everything large stays put.
//
// THE STATE IS PING-PONGED rather than updated in place: each update reads the
// old texture and writes a new one. Reading and writing one texture would need
// typed UAV loads of four-float formats, which D3D12 does not guarantee on all
// hardware; two textures need nothing unusual.
//
// Every dispatch is submitted on its own -- one per submission is the pattern
// this machine has never hung on.
//
// BINNING is threaded by contiguous chunks of Gaussians and sorts each tile
// with a stable radix sort, which reproduces the CPU rasteriser's (depth,
// index) order exactly. It was 49 ms an iteration single-threaded with
// std::sort and four freshly allocated staging buffers; it is now 17, about
// half of that the readback and uploads themselves.
//
// MEASURED on fountain-P11 at `downscale` 2 (about half a million Gaussians),
// per iteration: bin 16 ms, backward 12 (31 before it was batched; see
// splat_kernels.h), composite 6, loss, project and update 1 each. 300
// iterations take 15 s against 308 s on the CPU, to the same PSNR (17.59 vs
// 17.63 dB). Binning is the largest cost again, and half of it is the
// transfers.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "splat_raster.h"

namespace tglab {

class ComputeContext;

class SplatTrainerGpu {
public:
    explicit SplatTrainerGpu(ComputeContext* gpu);
    ~SplatTrainerGpu();
    SplatTrainerGpu(const SplatTrainerGpu&)            = delete;
    SplatTrainerGpu& operator=(const SplatTrainerGpu&) = delete;

    // The whole training state to and from the device: parameters, Adam's
    // first and second moments (14 per Gaussian each), and the densification
    // counters. Uploaded once at the start and again after every
    // densification step, which changes the count; downloaded before one.
    bool Upload(const std::vector<SplatParam>& params,
                const std::vector<double>& m1, const std::vector<double>& m2,
                const std::vector<double>& gradAccum,
                const std::vector<int>& gradCount,
                const std::vector<double>& maxScreen, std::string* err);
    bool Download(std::vector<SplatParam>* params, std::vector<double>* m1,
                  std::vector<double>* m2, std::vector<double>* gradAccum,
                  std::vector<int>* gradCount, std::vector<double>* maxScreen,
                  std::string* err);

    // One iteration against one photograph. `target` is w*h*3, as
    // train_splats holds it. `lrMean` is this iteration's position rate and
    // c1, c2 Adam's bias corrections; the other groups' rates are the
    // paper's, fixed in the kernel.
    //
    // `depthTarget`, when given, is a depth per pixel (w*h, 0 = none) that
    // the rendered expected depth is pulled toward with `depthWeight`; see
    // DepthLossGrad.
    bool Step(const SplatCam& cam, const RasterOptions& opt,
              const std::vector<double>& target, double lrMean, double c1,
              double c2, std::string* err,
              const std::vector<float>* depthTarget = nullptr,
              double depthWeight = 0.0);

    // Summed relative depth error, and pixels it covered, over every Step
    // that had a depth target.
    double DepthErrSum() const { return m_depthErr; }
    long long DepthErrCount() const { return m_depthCount; }

    // Clamp each colour to 0..1 after every step, as train_splats'
    // clamp_colour does on the CPU. On by default.
    void SetClampColour(bool on) { m_clampColour = on; }

    int Visible() const { return m_visible; }

    // Milliseconds per phase, accumulated over every Step.
    struct Timings {
        double project = 0, bin = 0, composite = 0, loss = 0, backward = 0,
               update = 0;
    };
    const Timings& Time() const { return m_time; }

private:
    struct Impl;
    std::unique_ptr<Impl> m;
    int     m_visible = 0;
    bool    m_clampColour = true;
    double    m_depthErr = 0.0;
    long long m_depthCount = 0;
    Timings m_time;
};

}  // namespace tglab
