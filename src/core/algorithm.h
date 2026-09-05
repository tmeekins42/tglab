// AlgorithmBase: the interface every algorithm derives from.
//
// Ports and parameters are both *declared* rather than implied by a signature,
// so the script, the UI, and the stage cache all discover them the same way.
#pragma once

#include <cmath>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "cancel.h"
#include "data.h"
#include "reduction.h"
#include "image.h"
#include "param.h"

namespace tglab {

// Defined in gpu/compute.h. Declared here so RunCtx can hand a CPU-shaped
// algorithm the compute device without core/ taking a dependency on gpu/.
//
// This exists for algorithms whose OUTPUT is not an image -- a feature
// detector emits a sidecar -- which the image-in/image-out RunGPU path
// cannot express. They stay CPU algorithms that offload their inner loops.
class ComputeContext;

struct Port {
    const char* name;
    DataType    type   = DataType::Image;
    FormatSpec  format = FormatSpec::Any;   // only meaningful for Image ports

    // How many images this port carries. Scalar -- one image -- is the default,
    // so every algorithm written before shape existed declares the right thing
    // by saying nothing. See shape.h for why the default matters.
    ShapeSpec   shape  = ShapeSpec::Scalar;
};

using PortList = std::vector<Port>;

// Handed to an algorithm at run time. Indices match the declared port order.
class RunCtx {
public:
    RunCtx(std::span<const Data* const> in, std::span<Data> out,
           const CancelToken* cancel = nullptr,
           ComputeContext* gpu = nullptr)
        : m_in(in), m_out(out), m_cancel(cancel), m_gpu(gpu) {}

    // True when this run has been superseded and should stop.
    //
    // A long algorithm should check this every row or two and return early.
    // Returning with a partly-written output is safe: the pipeline marks a
    // cancelled stage invalid, so the partial result is never displayed or
    // cached, and the newer run recomputes it from scratch.
    bool Cancelled() const { return m_cancel && m_cancel->Cancelled(); }

    // Image convenience accessors (M1 has only image ports).
    ImageView In(size_t i) const;
    ImageView Out(size_t i) const;

    const ImageDesc& InDesc(size_t i) const;

    // Raw port data, for composite algorithms that forward an input straight
    // into a sub-algorithm (see canny.cpp).
    const Data* InData(size_t i) const { return i < m_in.size() ? m_in[i] : nullptr; }

    // The output as an Image, for attaching a SIDECAR.
    //
    // Out() hands back an ImageView, which is pixels and a descriptor -- enough
    // for anything that writes pixels, and not enough for an algorithm whose
    // product is a sidecar. A feature detector's output IS its sidecar: it
    // passes the image through unchanged and attaches what it found.
    //
    // Null when the port does not hold an Image, so a caller that gets one is
    // holding something it can write to.
    Image* OutImage(size_t i) {
        if (i >= m_out.size()) return nullptr;
        return std::get_if<Image>(&m_out[i]);
    }

    // The input as an Image, for READING a sidecar an earlier stage attached.
    const Image* InImage(size_t i) const {
        if (i >= m_in.size() || !m_in[i]) return nullptr;
        return std::get_if<Image>(m_in[i]);
    }

    size_t NumIn()  const { return m_in.size(); }
    size_t NumOut() const { return m_out.size(); }

    // The compute device, or null when there is none.
    //
    // For CPU algorithms that want to offload an inner loop and read the
    // result back -- the case RunGPU cannot serve, because RunGPU is
    // image-in/image-out and a detector's product is a sidecar.
    //
    // Null is ordinary, not an error: no device, a lost device, or a build
    // with the GPU disabled. Every caller needs a CPU path anyway, so treat
    // this as an optimisation that may decline rather than a capability.
    ComputeContext* Gpu() const { return m_gpu; }

    // Converts a parameter expressed in PIXELS to this run's pixels.
    //
    // While a slider is dragged the pipeline may be running on a reduced-
    // resolution proxy, and a radius stated in pixels means a different
    // fraction of the picture there -- a blur of sigma 20 on a quarter-scale
    // image must run at sigma 5 to look the same. Every algorithm whose
    // parameters are lengths passes them through this.
    //
    // A HELPER RATHER THAN EACH SITE MULTIPLYING, because the failure mode is
    // silence: an algorithm that forgets simply previews wrong, and "the blur
    // looks too strong while dragging" is easy to dismiss as the preview being
    // approximate. One named call is greppable, and reads as a decision rather
    // than as arithmetic.
    //
    // Returns the value unchanged at full resolution, so it is always safe to
    // apply.
    float PixelScale(size_t port = 0) const {
        const float s = InDesc(port).proxyScale;
        return (s > 0.0f && s < 1.0f) ? s : 1.0f;
    }

    // The same, applied. `ctx.ScaledPx(m_sigma)` is the whole change for most
    // algorithms.
    float ScaledPx(float pixels, size_t port = 0) const {
        return pixels * PixelScale(port);
    }

    // For an integer radius. Rounds rather than truncating, and never below 1:
    // a radius of 0 is not a smaller blur, it is no blur at all, and a proxy
    // that silently disables an effect is worse than one that approximates it.
    int ScaledRadius(int pixels, size_t port = 0) const {
        if (pixels <= 0) return pixels;
        return std::max(1, int(std::lround(double(pixels) * double(PixelScale(port)))));
    }

private:
    std::span<const Data* const> m_in;
    std::span<Data>              m_out;
    const CancelToken*           m_cancel = nullptr;
    ComputeContext*              m_gpu    = nullptr;
};


// What is known about the image an algorithm is about to run on, for deriving
// parameter defaults.
//
// Two kinds of fact, and the distinction matters. The white balance is what the
// CAMERA recorded -- metadata, free to read. The exposure figures are MEASURED
// from the pixels, which costs a pass over the image and so is computed once
// when the file loads rather than per run.
struct SourceFacts {
    bool isMosaic = false;

    // The camera's own white balance, decomposed. Zero when the file carries no
    // daylight reference to measure against.
    float asShotKelvin = 0.0f;
    float asShotTint   = 0.0f;

    // Measured suggestions, in the units of basic_adjust's own controls.
    // `hasExposure` is false for a non-raw source or one too small to judge.
    bool  hasExposure = false;
    float autoExposure = 0.0f;   // stops
    float autoHighlights = 0.0f;   // negative recovers
    float autoShadows  = 0.0f;
    float autoBlacks   = 0.0f;
};

class AlgorithmBase {
    // FIRST, before any Param member -- including m_enabled below and every
    // one a derived class declares.
    //
    // A Param registers itself with its owner from its constructor, and
    // members are constructed in DECLARATION order. With this at the bottom of
    // the class, as it used to be, m_enabled pushed into the vector and the
    // vector was then default-constructed on top of it -- so the base's own
    // parameter silently vanished while every derived one survived. The panel
    // showed sigma and no `enabled`, with nothing reported.
    //
    // Derived params are unaffected either way: a derived class's members are
    // constructed after the whole base, vector included. It is only the base's
    // own that needs this.
    std::vector<ParamBase*> m_params;
    float m_proxyScale = 1.0f;   // see SetProxyScale
    friend class ParamBase;

public:
    AlgorithmBase()          = default;
    virtual ~AlgorithmBase() = default;

    // MUST stay deleted: this object holds raw pointers to its own Param
    // members, so a default copy/move would leave them dangling into the
    // old object. Algorithms are always heap-allocated via the registry.
    AlgorithmBase(const AlgorithmBase&)            = delete;
    AlgorithmBase& operator=(const AlgorithmBase&) = delete;
    AlgorithmBase(AlgorithmBase&&)                 = delete;
    AlgorithmBase& operator=(AlgorithmBase&&)      = delete;

    virtual const char* Name() const = 0;

    // Groups interchangeable implementations (e.g. "quantize"). Used by the
    // UI and, from M2, by choose() to offer every algorithm in a category.
    virtual const char* Category() const { return ""; }

    virtual PortList Inputs()  const = 0;
    virtual PortList Outputs() const = 0;

    virtual void RunCPU(RunCtx& ctx) = 0;

    // --- reduction (multi-image) --------------------------------------------
    // An algorithm that consumes several images along one named axis and
    // produces one. Opting in means overriding these AND declaring an input
    // port with ShapeSpec::Any and an output with ShapeSpec::Reduced.
    //
    // The script names the axis: merge_hdr(frames, over="exposure"). `over=` is
    // handled by the framework rather than being a parameter, because the axis
    // decides the SHAPE of the result and so has to be known while the pipeline
    // is being built, not when the stage runs.
    virtual bool IsReduction() const { return false; }

    // Reinterprets its input's shape without touching pixels. The pipeline
    // handles it directly, since it needs the whole Data rather than the
    // per-image view RunCPU gets. See algorithms/merge/reshape.cpp.
    virtual bool IsReshape() const { return false; }

    // Annotates a whole group at once -- it solves every frame against a common
    // reference, so it cannot be run one frame at a time. Like a reshape, the
    // pipeline handles it directly. See algorithms/merge/align.cpp.
    virtual bool IsAligner() const { return false; }

    // True when this stage's settings would leave the image unchanged, so the
    // pipeline can skip it ENTIRELY -- no allocation, no dispatch, no copy.
    //
    // The point is stacking. A script emulating a full develop pipeline wants
    // twenty effects available and three of them used, and an unused effect
    // must cost nothing rather than "one cheap pass". Twenty cheap passes over
    // a 45 MP image is not cheap, and every one of them also allocates a
    // full-size intermediate.
    //
    // Skipping is safe because a bypassed stage ALIASES its input: Resolve()
    // follows the PortRef through, so downstream reads the upstream image
    // directly. That means no copy either -- the saving is the whole stage,
    // not just its inner loop.
    //
    // Two things make it correct rather than merely fast:
    //
    //   - Only a stage whose output port would have the same FORMAT and SHAPE
    //     as its input may be bypassed. A demosaic turning R32F into RGBA16F
    //     cannot be, even at settings that "do nothing", because the type
    //     changes. The pipeline checks this; an algorithm cannot opt out of it.
    //   - IsNoOp is folded into the parameter hash, so toggling it re-runs the
    //     stages downstream exactly as any other parameter change does.
    //
    // Default false: an algorithm that says nothing is always run, which is the
    // safe answer. Override it where a setting genuinely means "off" -- amount
    // at 0, a disabled toggle -- and be strict, since a wrong `true` silently
    // drops the effect.
    virtual bool IsNoOp() const { return false; }

    // How this algorithm behaves when run on a reduced-resolution proxy.
    //
    //   Exact   Per-pixel. The proxy IS the answer, just smaller -- brightness,
    //           a LUT, a tone curve. Nothing to adjust.
    //   Scaled  Correct once pixel-unit parameters are multiplied by
    //           ImageDesc::proxyScale. A blur, a bloom, a window radius.
    //   Never   The output genuinely differs and no parameter fixes it, so the
    //           pipeline must not proxy this stage at all. A demosaic reads a
    //           CFA mosaic that downsampling destroys; a feature detector on a
    //           proxy finds DIFFERENT keypoints rather than approximate ones.
    //
    // SCALED IS THE DEFAULT ON PURPOSE. It is right for most algorithms, and an
    // algorithm that ignores proxyScale while claiming Scaled produces a
    // slightly wrong PREVIEW -- visible, and fixable. Defaulting to Never would
    // instead silently keep the pipeline slow, which is the failure nobody
    // notices because nothing looks wrong.
    // --- region processing --------------------------------------------------
    //
    // How far, in pixels, this algorithm reads outside the pixel it is writing.
    //
    // Zooming in makes the SCALE axis useless -- at 1:1 the correct scale is
    // 1.0 and nothing is saved -- while 95% of the computed pixels are off
    // screen. Computing only the visible rectangle is the other half of "what
    // the viewer can actually see", and this is what makes it possible: to
    // produce a correct tile, a stage needs its input grown by its own reach,
    // and the pipeline grows the region backward through the chain by summing
    // these.
    //
    // 0 means per-pixel, which needs no margin at all -- and that is the right
    // DEFAULT because it is true of most algorithms. An algorithm that reads
    // neighbours and forgets to say so produces visible SEAMS at tile edges,
    // which is loud and immediately traceable to the stage that caused it. The
    // opposite default would make every pipeline quietly refuse to tile.
    //
    // Stated in the pixels of THIS run, so it must scale like any other
    // pixel-unit value -- see RunCtx::ScaledPx.
    virtual int ReachPixels() const { return 0; }

    // Whether this algorithm's output depends on pixels beyond any margin.
    //
    // A DIFFERENT QUESTION FROM ProxyBehaviour, and the distinction is the
    // reason this is a separate declaration. `global_threshold` builds a
    // histogram of the whole frame: at reduced SCALE that histogram is a good
    // approximation, so it proxies happily. On a REGION it is a histogram of
    // the wrong population -- the threshold chosen for a bright corner is not
    // the threshold for the picture -- and no margin fixes it, because the
    // dependency is on everything.
    //
    // The same is true of auto-exposure, dehaze's airlight estimate, and any
    // percentile. They are safe to shrink and unsafe to crop.
    virtual bool RegionSafe() const { return true; }

    enum class ProxyBehaviour { Exact, Scaled, Never };
    virtual ProxyBehaviour Proxy() const {
        // DEMOSAIC IS DERIVED FROM THE CATEGORY, because there the category is
        // exact: every demosaicer reads a CFA mosaic, which downsampling
        // destroys -- the pattern IS the data. Deciding it here rather than
        // per-file means the next demosaicer is safe without anyone
        // remembering, and the failure it prevents (a mosaic averaged into
        // mush) would look like a demosaic bug rather than a proxy one.
        //
        // THE SIDECAR ALGORITHMS OVERRIDE INDIVIDUALLY, and the reason is worth
        // recording. Anything producing or consuming a sidecar is equally
        // unsafe -- sidecar coordinates are in image pixels and nothing
        // rescales them -- but they do NOT share a category. Measured:
        // detect_* and draw_* say "features", match_* says "match",
        // align_features and bundle_adjust say "merge". A category test for
        // "features" read correctly and silently missed two thirds of them,
        // which is exactly the quiet failure this default exists to avoid.
        return std::strcmp(Category(), "demosaic") == 0
                   ? ProxyBehaviour::Never
                   : ProxyBehaviour::Scaled;
    }

    // Restates the output descriptor for port `p`, given what the pipeline
    // would otherwise have allocated.
    //
    // The pipeline sizes every output from input 0 and then runs the algorithm
    // INTO that buffer, which is right for the overwhelming majority: a blur, a
    // tone map and a demosaic all produce one output pixel per input pixel. An
    // algorithm that changes the raster -- a crop, and a resize when one is
    // written -- has no way to express that, and this is it.
    //
    // A HOOK RATHER THAN A PORT SPEC, because the size is not a property of the
    // algorithm the way its format is: the same crop produces a different
    // descriptor for every setting of its sliders. FormatSpec can stay a
    // compile-time declaration precisely because a format does not vary with a
    // parameter; a size does.
    //
    // Returning `in` unchanged (the default) means "same size as the input",
    // which is what every existing algorithm wants and gets without saying so.
    //
    // Whatever comes back is validated by the caller, so a crop that computes
    // an empty rectangle produces a named error rather than a zero-sized
    // allocation that fails somewhere later and less clearly.
    virtual ImageDesc OutputDesc(int /*p*/, const ImageDesc& in) const { return in; }

    // Solves and attaches transforms across a whole group. Called only when
    // IsAligner() is true.
    virtual bool RunAlign(std::vector<Image>* /*images*/, std::string* /*err*/) { return true; }

    // The streaming accumulator. See core/reduction.h for why it is Begin /
    // Accept / Finish rather than "here are all N images".
    virtual Reducer* AsReducer() { return nullptr; }

    // --- GPU path (M3) ------------------------------------------------------
    // An algorithm opts in by returning true from HasGPU() and providing the
    // HLSL for a compute kernel. The framework handles residency, descriptor
    // binding and dispatch, so an algorithm only writes the kernel itself.
    virtual bool HasGPU() const { return false; }

    // Compute shader source with a `main` entry point. Bindings by convention:
    //   t0..t3  inputs        u0..u3  outputs
    //   b0      uint Width, uint Height, then GpuConstants() below
    virtual const char* GpuSource() const { return nullptr; }

    // Called before the GPU path runs, with the input descriptors.
    //
    // Most algorithms need nothing here: their constants come from parameters.
    // A demosaic does -- the CFA pattern and the sensor's black and white
    // levels ride on the *image*, not on any parameter, and GpuConstants() is
    // otherwise the only place to read them, by which point the descriptors are
    // out of reach. RunCPU() gets them from RunCtx, but the GPU path never
    // calls RunCPU, so without this the shader would use stale values.
    virtual void PrepareGpu(const std::vector<ImageDesc>& inputs) { (void)inputs; }

    // The proxy scale of the input this stage is about to run on.
    //
    // Set by the pipeline before every run, so a GPU algorithm can convert its
    // pixel-unit parameters in GpuConstants() -- where the input descriptors
    // are already out of reach and there is no RunCtx to ask.
    //
    // Stored on the base rather than requiring a PrepareGpu override in every
    // algorithm with a radius, because an algorithm that forgets to override
    // does not fail: it silently previews at the wrong strength while dragging,
    // and looks merely approximate. Making the value available by default means
    // the only thing an algorithm has to remember is to USE it.
    void SetProxyScale(float s) { m_proxyScale = (s > 0.0f) ? s : 1.0f; }

    // Multiplies a pixel-unit parameter into this run's pixels. 1.0 at full
    // resolution, so it is always safe to apply. Mirrors RunCtx::ScaledPx for
    // the GPU side.
    float GpuScaledPx(float pixels) const { return pixels * m_proxyScale; }
    int   GpuScaledRadius(int pixels) const {
        if (pixels <= 0) return pixels;
        return std::max(1, int(std::lround(double(pixels) * double(m_proxyScale))));
    }

    // True if this algorithm must MEASURE its input before the GPU path runs.
    //
    // Separate from PrepareGpu, and opt-in, because it costs something real: it
    // forces the input's pixels to the CPU, which on a GPU-resident
    // intermediate means a readback -- ~86 ms at 21 MP. Every algorithm whose
    // constants come from parameters or from the descriptor should leave this
    // false and pay nothing.
    //
    // What needs it: an operator whose shader constants depend on the CONTENT
    // rather than on the format. A tone mapper is the case -- where it places
    // its curve depends on the scene's own percentiles, and a merged bracket's
    // scale is arbitrary, so no fixed anchor can be right. Without this the
    // choice is a CPU-only operator or a shader working from numbers that do
    // not describe the image in front of it.
    virtual bool GpuNeedsInputPixels() const { return false; }

    // Called with the input pixels when GpuNeedsInputPixels() is true, before
    // any pass is dispatched. Whatever is measured here is expected to reach
    // the shader through GpuConstants/GpuPassConstants.
    virtual void MeasureForGpu(const std::vector<const Image*>& inputs) { (void)inputs; }

    // --- caching a measurement across runs ----------------------------------
    //
    // MeasureForGpu runs on every slider tick, and for some algorithms the
    // measurement depends only on the INPUT -- not on any parameter the user is
    // dragging. Dehaze is the case: its airlight is a property of the
    // photograph, so moving the strength slider re-measures an answer that
    // cannot have changed.
    //
    // An algorithm that says so here gets its measurement cached by the
    // pipeline, keyed on the identity of the input that produced it. The
    // pipeline owns the key because only it knows when an input is genuinely
    // the same image -- the algorithm object itself does not survive between
    // runs, since the script is re-interpreted and every stage rebuilt.
    //
    // The contract is deliberately narrow: return true ONLY if the measurement
    // is a pure function of the input pixels. An algorithm whose measurement
    // also depends on a parameter must leave this false, or dragging that
    // parameter will silently use a stale answer -- and a stale measurement
    // does not look like a bug, it looks like a slightly wrong picture.
    virtual bool MeasurementDependsOnlyOnInput() const { return false; }

    // Serialise the last measurement, and restore one. Opaque to the pipeline,
    // which only stores and returns the bytes.
    //
    // Empty means "nothing to cache", which is also what an algorithm that has
    // not measured anything yet should return -- the pipeline then simply does
    // not cache, rather than storing a blob that decodes to garbage.
    virtual std::vector<uint32_t> SaveMeasurement() const { return {}; }
    virtual bool RestoreMeasurement(const std::vector<uint32_t>&) { return false; }

    // Extra textures the ALGORITHM supplies, bound as SRVs after the port
    // inputs. An empty vector -- the default -- costs nothing.
    //
    // What this is for: data too large for the constant buffer and not shaped
    // like the image. A 33^3 colour LUT is 431 KB against the cbuffer's 64 KB
    // limit, so it cannot be a constant; it is not an input port, because it
    // comes from a file rather than from another stage; and it cannot be a
    // scratch plane, because those are allocated at IMAGE size. Without a hook
    // like this the only remaining option is to keep such an algorithm on the
    // CPU.
    //
    // The images are owned by the algorithm and must outlive the dispatch --
    // hold them as members. Returning the same images on every call is fine and
    // expected: they are uploaded once and stay GPU-resident, so rebuilding
    // them per run would upload per run. Rebuild only when the thing they
    // describe actually changes.
    //
    // Indexing follows GpuPass::reads: with one input port the extra textures
    // begin at -2, so a pass reading {-1, -2} gets the image and then the first
    // extra. The single-kernel path binds them in the same order.
    virtual std::vector<const Image*> GpuExtraInputs() const { return {}; }

    // Lets an algorithm set parameter DEFAULTS from the source it will run on,
    // before the script declares its controls.
    //
    // Distinct from PrepareGpu, which happens far too late for this: by then the
    // UI has already built its sliders. Only basic_adjust uses it, to open the
    // kelvin control at the temperature the camera actually chose rather than at
    // a sentinel meaning "leave it alone".
    //
    // A default, not a value: whatever the user or the script set wins. That is
    // what makes "auto" a starting point rather than a mode -- an untouched
    // slider moves to the suggestion, a touched one is left alone, and a re-run
    // changes nothing because the same image measures the same way.
    //
    // A struct rather than a parameter list: this started as three floats and
    // grew, and every addition would otherwise touch every override.
    virtual void PrepareDefaults(const SourceFacts& /*facts*/) {}

    // Extra root constants, bit-cast to uint. Floats go through asfloat() in
    // the shader. Order must match the cbuffer declaration.
    //
    // `iteration` is the 0-based pass index, for iterative algorithms; it is 0
    // for everything else. A kernel that behaves the same every pass can ignore
    // it entirely.
    virtual std::vector<uint32_t> GpuConstants(int iteration = 0) const {
        (void)iteration;
        return {};
    }

    // --- multi-pass GPU (several DIFFERENT kernels) -------------------------
    //
    // GpuIterations() runs ONE kernel repeatedly. That covers separable filters
    // and iterative schemes, and it is not enough for an algorithm whose passes
    // do different things: AHD demosaicing interpolates green, then reconstructs
    // two colour-difference planes from it, then medians those, then combines --
    // four distinct kernels over three intermediate planes that must persist
    // between them.
    //
    // Such an algorithm returns a non-empty GpuPasses(). Each pass names its own
    // HLSL and declares which buffers it reads and writes, by index into a pool
    // the framework allocates: negative indices are the stage's real inputs and
    // outputs, non-negative ones are scratch planes.
    //
    // Deliberately explicit about buffers rather than inferring them. Getting a
    // ping-pong wrong is silent -- a pass reads the buffer it is writing and the
    // result is a race that shows as noise on some hardware and not others --
    // and a declaration the framework can check turns that into an error at
    // build time.
    struct GpuPass {
        const char* source = nullptr;   // HLSL with a `main` entry point
        const char* name   = "";        // for compile errors and PIX markers

        // Buffer indices. Negative values address the stage's own ports:
        //   -1 = input 0,  -2 = input 1, ...
        //   -1 in `writes` = output 0, -2 = output 1, ...
        // Non-negative values index the scratch pool, sized by GpuScratchCount().
        std::vector<int> reads;
        std::vector<int> writes;
    };

    virtual std::vector<GpuPass> GpuPasses() const { return {}; }

    // How many scratch planes the passes above share, and what format they are.
    //
    // Allocated once and cached on the stage, like the iterative scratch, since
    // reallocating three full-size planes per slider nudge would cost more than
    // the passes themselves.
    virtual int        GpuScratchCount()  const { return 0; }
    virtual FormatSpec GpuScratchPlanes() const { return FormatSpec::R32F; }

    // Root constants for a specific pass, when they differ between passes.
    // Defaults to GpuConstants(pass), so an algorithm whose passes share their
    // constants needs no override.
    virtual std::vector<uint32_t> GpuPassConstants(int pass) const {
        return GpuConstants(pass);
    }

    // How many times to dispatch the kernel, feeding each pass's output back in
    // as the next pass's input. 1 (the default) is the ordinary single-dispatch
    // case.
    //
    // Iterative schemes -- Perona-Malik being the motivating one -- cannot be
    // expressed as a single dispatch: each step must see the *completed*
    // previous step, and threads within one dispatch have no such ordering. The
    // framework ping-pongs between two GPU images and hands back whichever
    // holds the final result, so the algorithm only writes one pass.
    virtual int GpuIterations() const { return 1; }

    // One short line about what the last run actually did, for the info panel.
    //
    // Some algorithms know something the pixels do not show. hot_pixel_repair
    // counts the sensels it replaced -- 12 out of 44 million on one file --
    // which is invisible in the result and exactly what tells you whether the
    // threshold is set sensibly: a count that suddenly jumps into the thousands
    // means it has started eating detail rather than defects.
    //
    // Empty by default, and an empty report is simply not shown. Deliberately a
    // string rather than a number: what is worth reporting differs per
    // algorithm, and a scheme general enough to hold all of them would be more
    // machinery than the feature is worth.
    //
    // Called on the UI thread against the algorithm the worker just ran, so it
    // must only read values RunCPU/RunGPU already computed -- never recompute.
    virtual std::string RunReport() const { return {}; }

    // Format for the ping-pong scratch image of an iterative GPU stage.
    //
    // FormatSpec::SameAsInput (the default) means "match the output", which is
    // what a filter wants: its intermediate is the same kind of thing as its
    // result.
    //
    // A threshold is where that breaks down. Its output is a single-channel
    // R32F mask, but the intermediate it needs is richer -- Bernsen carries a
    // window minimum AND maximum, the Niblack family carries a running sum and
    // sum-of-squares. Sized from the output those extra channels are silently
    // dropped: the first version of the Bernsen kernel lost its maximum and got
    // 50% of pixels wrong, with nothing reported. Declaring the scratch format
    // separately is what makes those algorithms expressible.
    virtual FormatSpec GpuScratchFormat() const { return FormatSpec::SameAsInput; }

    // Every algorithm gets an on/off switch, declared here rather than in each
    // one so there is no algorithm that forgot to have it.
    //
    // Distinct from IsNoOp(), and both are worth having. IsNoOp says "these
    // settings happen to change nothing", which is free but requires
    // NEUTRALISING the controls -- and that loses the settings you spent time
    // getting right. This is the switch you flip to compare with and without,
    // and flip back to exactly what you had.
    //
    // First in the panel because it is declared before any derived member: a
    // Param registers with its owner at construction, and base members are
    // constructed first. That ordering is load-bearing rather than incidental,
    // so a stage's switch always sits above the controls it governs.
    //
    // Named `enabled` rather than `bypass` so the checked state is the active
    // one -- a control that has to be OFF for the effect to happen reads
    // backwards on every panel it appears in.
    Param<bool> m_enabled{this, "enabled", true,
        "Turn this stage off without losing its settings. A disabled stage is "
        "skipped entirely -- no allocation, no dispatch, no copy -- and passes "
        "its input straight through."};

    // True when the stage should not run at all: switched off, or at settings
    // that would change nothing. The pipeline asks this rather than IsNoOp
    // directly, so an algorithm overriding IsNoOp never has to remember the
    // switch.
    bool ShouldBypass() const { return !bool(m_enabled) || IsNoOp(); }

    std::span<ParamBase* const> Params() const { return m_params; }
    ParamBase* FindParam(std::string_view name) const;

    // Folds every parameter value into one hash for dirty detection.
    uint64_t ParamHash() const;

};

// ---------------------------------------------------------------------------
// Registry — self-registration, so adding an algorithm touches no central file.

class Registry {
public:
    using Factory = std::unique_ptr<AlgorithmBase> (*)();

    static Registry& Get();

    void Add(const char* name, Factory f);
    std::unique_ptr<AlgorithmBase> Create(std::string_view name) const;
    bool Contains(std::string_view name) const;

    std::vector<std::string> Names() const;
    std::vector<std::string> NamesInCategory(std::string_view category) const;

private:
    struct Entry {
        std::string name;
        Factory     factory;
    };
    std::vector<Entry> m_entries;
};

#define REGISTER_ALGORITHM(T)                                                  \
    namespace {                                                                \
    struct T##_Registrar {                                                     \
        T##_Registrar() {                                                      \
            ::tglab::Registry::Get().Add(                                      \
                T{}.Name(),                                                    \
                []() -> std::unique_ptr<::tglab::AlgorithmBase> {              \
                    return std::make_unique<T>();                              \
                });                                                            \
        }                                                                      \
    };                                                                         \
    T##_Registrar g_##T##_registrar;                                           \
    }

} // namespace tglab
