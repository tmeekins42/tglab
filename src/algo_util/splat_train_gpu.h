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
//   bin and sort   GPU: a prefix sum and a radix sort (see kBinCommon); only
//                  the entry count comes back
//   composite      GPU  (the kernel SplatRaster uses)
//   loss           CPU, from the rendered image -- a megabyte each way
//   backward       GPU, per pixel (SplatRaster's kernel), then per Gaussian:
//                  sum each Gaussian's per-tile gradients, apply the chain
//                  rule, take the Adam step
//
// BINNING STAYED ON THE CPU at first, on purpose: sorting on the GPU takes
// many dispatches, and many dispatches had hung this machine. It moved once
// it was the largest cost left -- about twenty dispatches a step, still each
// submitted on its own.
//
// THE STATE IS PING-PONGED rather than updated in place: each update reads the
// old texture and writes a new one. Reading and writing one texture would need
// typed UAV loads of four-float formats, which D3D12 does not guarantee on all
// hardware; two textures need nothing unusual.
//
// DISPATCHES ARE BATCHED: each is recorded, and a batch is submitted by the
// next transfer, at the end of a phase (so the timings below stay honest), or
// by the framework's own limits -- its 16 MP pixel budget, which is what
// keeps several large dispatches out of one submission (the hang this
// machine has seen, see compute.cpp), and its descriptor heap. Every dispatch
// used to be submitted on its own, the one pattern never seen to hang, and
// TGLAB_SPLAT_NOBATCH brings that back for bisecting. A texture is never
// released while a recorded dispatch may still use it (Impl::Ensure).
//
// BINNING reproduces the CPU rasteriser's (tile, depth, index) order exactly:
// the entries are made in index order and every radix pass is stable. On the
// CPU it was 49 ms an iteration single-threaded, then 17 threaded -- half of
// that the transfers. On the device, on a 100-frame video (720k Gaussians in
// view, 180x320): 29 ms on the CPU, 4 on the GPU, and the run's every count
// and score unchanged to the last digit.
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
#include "splat_reflect.h"

namespace tglab {

class ComputeContext;

// What a Step needs to train reflections: the environment it shades with,
// where to add its gradient, and the rates for reflectivity and normals.
struct ReflStepArgs {
    const EnvMap*        env = nullptr;
    std::vector<double>* envGrad = nullptr;
    double               lrRefl = 0.0, lrNormal = 0.0;
    // train_splats' reflect_sparsity: lambda times the mean reflectivity over
    // the measured pixels, added to the loss.
    double               sparsity = 0.0;
};

// All densification decides from, per Gaussian: see train_splats. `gradAvg`
// is the mean screen gradient over the views that saw it, -1 for none.
struct DensifyIn {
    double opacityLogit = 0.0, maxLogScale = 0.0, gradAvg = -1.0, maxScreen = 0.0;
};

// One Gaussian of the state after densification: which one it comes from,
// and how. A kept one carries its optimiser state; a copy starts without;
// a split child also moves by R * (z * scale) and shrinks by 1.6.
struct DensifyPlan {
    enum Kind : uint32_t { kKeep = 0, kCopy = 1, kSplit = 2 };
    uint32_t src = 0;
    uint32_t kind = kKeep;
    float    z[3] = {0, 0, 0};
};

class SplatTrainerGpu {
public:
    explicit SplatTrainerGpu(ComputeContext* gpu);
    ~SplatTrainerGpu();
    SplatTrainerGpu(const SplatTrainerGpu&)            = delete;
    SplatTrainerGpu& operator=(const SplatTrainerGpu&) = delete;

    // The whole training state to and from the device: parameters, Adam's
    // first and second moments (14 per Gaussian each), and the densification
    // counters. Uploaded once at the start and downloaded at the end;
    // densification in between stays on the device (ApplyPlan).
    bool Upload(const std::vector<SplatParam>& params,
                const std::vector<double>& m1, const std::vector<double>& m2,
                const std::vector<double>& gradAccum,
                const std::vector<int>& gradCount,
                const std::vector<double>& maxScreen, std::string* err);
    bool Download(std::vector<SplatParam>* params, std::vector<double>* m1,
                  std::vector<double>* m2, std::vector<double>* gradAccum,
                  std::vector<int>* gradCount, std::vector<double>* maxScreen,
                  std::string* err);

    // Just the Gaussians -- and their view-dependent colour when `sh` is
    // given and there is any -- for a look at training as it goes, without
    // unpacking Adam's moments or the densification counters.
    bool DownloadParams(std::vector<SplatParam>* params, std::vector<double>* sh,
                        std::string* err);

    // One iteration against one photograph. `target` is w*h*3, as
    // train_splats holds it. `lrMean` is this iteration's position rate and
    // c1, c2 Adam's bias corrections; the other groups' rates are the
    // paper's, fixed in the kernel.
    //
    // `depthTarget`, when given, is a depth per pixel (w*h, 0 = none) that
    // the rendered expected depth is pulled toward with `depthWeight`; see
    // DepthLossGrad.
    //
    // `shDegree` > 0 colours each Gaussian for this camera with its
    // spherical harmonics up to that degree and steps them at `shLr`; it
    // needs UploadSh first, and is ignored without it.
    //
    // `ra`, when given (and UploadRefl has run), trains REFLECTIONS as well
    // (splat_reflect.h): the colour, reflectivity and normal are composited
    // in turn, the loss is taken on the deferred shading of the three, and
    // each pass's backward runs. The shading and its backward run on the
    // device too (kShade, kShadeBwd); the environment is uploaded each step
    // and its gradient read back and ADDED into *ra->envGrad, for the caller
    // to step on the CPU -- a few hundred kilobytes each way.
    bool Step(const SplatCam& cam, const RasterOptions& opt,
              const std::vector<double>& target, double lrMean, double c1,
              double c2, std::string* err,
              const std::vector<float>* depthTarget = nullptr,
              double depthWeight = 0.0, int shDegree = 0, double shLr = 0.0,
              const ReflStepArgs* ra = nullptr);

    // Renders the uploaded splats from `cam` into `rgb` (w*h*3), through
    // exactly the forward pass training uses: view-dependent colour up to
    // `shDegree` (UploadSh first) and, when `env` is given and UploadRefl
    // has run, reflections. Changes nothing.
    bool Render(const SplatCam& cam, const RasterOptions& opt, int shDegree,
                const EnvMap* env, std::vector<double>* rgb, std::string* err,
                std::vector<double>* depth = nullptr, std::vector<double>* finalT = nullptr);

    // Reflectivity logits and normals (splat_reflect.h), with Adam's moments
    // (4 per Gaussian each), beside Upload and Download as UploadSh is.
    bool UploadRefl(const std::vector<ReflParam>& refl, const std::vector<double>& m1,
                    const std::vector<double>& m2, std::string* err);
    bool DownloadRefl(std::vector<ReflParam>* refl, std::vector<double>* m1,
                      std::vector<double>* m2, std::string* err);

    // The view-dependent colour (splat_sh.h): kShRest coefficients per
    // Gaussian and Adam's moments for each, in parameter order. Sent after
    // every Upload -- which may change the count and so drops them -- and
    // read back beside every Download.
    bool UploadSh(const std::vector<double>& sh, const std::vector<double>& m1,
                  const std::vector<double>& m2, std::string* err);
    bool DownloadSh(std::vector<double>* sh, std::vector<double>* m1,
                    std::vector<double>* m2, std::string* err);

    // DENSIFICATION ON THE DEVICE. Only the summary comes down (32 bytes a
    // Gaussian, against ~800 for the whole state with view-dependent colour)
    // and the plan made from it goes back up; the device rebuilds parameters,
    // moments, colour and reflections from the plan, and clears every
    // Gaussian's densification counters, as the CPU path does.
    bool DensifyStats(std::vector<DensifyIn>* out, std::string* err);
    bool ApplyPlan(const std::vector<DensifyPlan>& plan, std::string* err);
    // Caps every opacity logit at `cap` and clears its Adam moments.
    bool ResetOpacity(double cap, std::string* err);

    // FOR TESTS: the deferred shading kernels alone, on maps given as
    // splat_reflect.h's ShadeDeferred takes them (w*h*3; R in channel 0),
    // with dOut the loss's gradient -- so both passes can be checked against
    // ShadeDeferred and ShadeDeferredBackward. `sparsity` is added to every
    // pixel's dR, as a Step's reflect_sparsity is. Needs an Upload first,
    // which builds the kernels.
    bool ShadeCheck(const SplatCam& cam, const EnvMap& env, const std::vector<double>& Cd,
                    const std::vector<double>& Rm, const std::vector<double>& Nm,
                    const std::vector<double>& dOut, double sparsity,
                    std::vector<double>* shaded, std::vector<double>* dCd,
                    std::vector<double>* dRm, std::vector<double>* dNm,
                    std::vector<double>* envGrad, std::string* err);

    // Summed relative depth error, and pixels it covered, over every Step
    // that had a depth target.
    double DepthErrSum() const { return m_depthErr; }
    long long DepthErrCount() const { return m_depthCount; }

    // Clamp each colour to 0..1 after every step, as train_splats'
    // clamp_colour does on the CPU. On by default.
    void SetClampColour(bool on) { m_clampColour = on; }

    // Hold each Gaussian's largest axis within `ratio` times its middle one
    // after every step, as train_splats' max_elongation does on the CPU.
    // 0 (or anything up to 1) leaves them free.
    void SetMaxElongation(double ratio) { m_maxElong = ratio; }

    int Visible() const { return m_visible; }

    // Milliseconds per phase, accumulated over every Step.
    struct Timings {
        double project = 0, bin = 0, composite = 0, loss = 0, backward = 0,
               update = 0;
    };
    const Timings& Time() const { return m_time; }

private:
    bool StepImpl(const SplatCam& cam, const RasterOptions& opt,
                  const std::vector<double>& target, double lrMean, double c1, double c2,
                  std::string* err, const std::vector<float>* depthTarget, double depthWeight,
                  int shDegree, double shLr, const ReflStepArgs* ra,
                  std::vector<double>* renderOut);

    struct Impl;
    std::unique_ptr<Impl> m;
    int     m_visible = 0;
    bool    m_clampColour = true;
    double  m_maxElong = 0.0;
    // Render's optional depth and coverage outputs, for the call in flight.
    std::vector<double>* m_renderDepth = nullptr;
    std::vector<double>* m_renderT = nullptr;
    double    m_depthErr = 0.0;
    long long m_depthCount = 0;
    Timings m_time;
};

}  // namespace tglab
