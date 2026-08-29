// Pipeline: the recorded stage list plus the dirty-by-hash execution.
//
// Phase 1 (interpreter) RECORDS stages; it never runs an algorithm.
// Phase 2 (Execute) runs from the first stage whose inputs or parameters
// changed, reusing cached outputs before that point.
#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "algorithm.h"
#include "data.h"
#include "image_io.h"
#include "progress.h"

namespace tglab {

// Diagnostic hook: called once per frame inside a fused reduction. Null in
// normal builds. See pipeline.cpp.
extern std::function<void(int)> g_frameTrace;

// Reference to one output port of a stage. stage == -1 means a source image
// from the palette, with `port` as its palette index.
struct PortRef {
    int stage = -1;
    int port  = 0;

    bool operator==(const PortRef&) const = default;
};

struct ComputeKernel;   // gpu/compute.h
struct GpuImage;        // gpu/gpu_image.h

struct Stage {
    std::unique_ptr<AlgorithmBase> algo;
    std::string          algoName;
    std::vector<PortRef> inputs;
    std::vector<Data>    outputs;
    uint64_t             paramHash = 0;

    // The axis this stage reduces over, from the script's over="..." argument.
    // Empty when the stage is not a reduction. Part of SameStage, so changing
    // it re-runs rather than reusing a result reduced over a different axis.
    std::string          reduceAxis;

    // Target shape for a reshape stage, computed by the interpreter so that
    // later lines can be checked against it at build time. Empty otherwise.
    Shape                reshapeTo;

    // When this stage was skipped as a no-op, the port its output aliases.
    // stage == -2 means "not bypassed"; -1 would be ambiguous with a palette
    // source, which is a legitimate thing to alias.
    PortRef              bypassOf{-2, 0};

    // When this stage was consumed by a fused reduction, the stage that owns
    // its result. -1 otherwise. The cache scan uses it to look past a stage
    // that legitimately holds nothing.
    int                  fusedInto = -1;
    // Versions of the palette images this stage read, so replacing one is
    // detected as a change rather than looking identical.
    uint64_t             sourceHash = 0;
    bool                 valid     = false;   // outputs hold a usable result
    int                  line      = 0;       // for error messages

    // Compiled kernel, cached so dragging a slider does not recompile HLSL
    // every frame. shared_ptr because Stage is moved between pipelines and the
    // kernel outlives any single run.
    std::shared_ptr<ComputeKernel> kernel;

    // Second GPU image for iterative stages to ping-pong against, cached for
    // the same reason as the kernel: reallocating it per slider drag would cost
    // more than the iterations themselves.
    std::shared_ptr<GpuImage> gpuScratch;
    ImageDesc                 scratchDesc{};

    // Multi-pass stages: one compiled kernel per declared pass, and the pool of
    // scratch planes they share. Cached for the same reason as the above --
    // recompiling four kernels and reallocating three full-size planes on every
    // slider nudge would dwarf the work itself.
    std::vector<std::shared_ptr<ComputeKernel>> passKernels;
    std::vector<std::shared_ptr<GpuImage>>      gpuPlanes;
    ImageDesc                                   planeDesc{};
};

// A viewer declared by the script via display().
struct ViewerDecl {
    std::string name;
    PortRef     source;
};

// A save() the script declared. Recorded during the build like a viewer, and
// acted on after the run rather than during it -- writing a file from inside
// Execute would put disk IO on the worker's critical path and would happen
// again on every slider tick.
struct SaveDecl {
    std::string path;        // as written; a group gets a number appended
    PortRef     source;
    SaveFormat  format = SaveFormat::Png;
    int         quality = 92;

    // What to do when the file already exists.
    //
    //   Increment  out.png -> out_1.png. The default, because a script that
    //              silently overwrote the result of the last run would be a
    //              bad surprise in a tool built around re-running.
    //   Overwrite  write over it, for the "export the current state" case
    //              where accumulating numbered copies is the surprise.
    //   Skip       leave the existing file and do nothing.
    enum class Existing { Increment, Overwrite, Skip };
    Existing existing = Existing::Increment;
};

// Which implementation to run. Auto prefers the GPU when the algorithm has
// one; the explicit modes exist for correctness comparison and benchmarking.
enum class ExecMode : uint8_t { Auto, ForceCPU, ForceGPU };

class ComputeContext;   // fwd; pipeline only needs a pointer

class Pipeline {
public:
    // --- recording (phase 1) ---
    void Clear();
    int  AddStage(std::unique_ptr<AlgorithmBase> algo, std::string name,
                  std::vector<PortRef> inputs, size_t numOutputs, int line,
                  std::string reduceAxis = {}, Shape reshapeTo = {});
    void AddViewer(std::string name, PortRef src);
    void AddSave(SaveDecl s);

    // --- execution (phase 2) ---
    // `sources` are the palette images, indexed by PortRef::port when stage==-1.
    // Reuses cached outputs from `prev` where the stage is unchanged.
    // `gpu` may be null, in which case everything runs on the CPU.
    // `sourceVersions` runs parallel to `sources`: a value that changes when
    // that palette image is replaced. Without it a stage reading PortRef{-1, i}
    // looks identical no matter which file backs slot i, so swapping an image
    // would reuse the cached output.
    // `cancel`, if given, is polled between stages and handed to each algorithm
    // so a slow one can abandon its own inner loop. Execute returns false with
    // `err` set to kCancelled; the caller is expected to discard the result
    // rather than report it, since a newer run is already on its way.
    bool Execute(std::vector<Data>* sources, Pipeline* prev, std::string* err,
                 ComputeContext* gpu = nullptr, ExecMode mode = ExecMode::Auto,
                 const std::vector<uint64_t>* sourceVersions = nullptr,
                 const CancelToken* cancel = nullptr,
                 Progress* progress = nullptr);

    // Sentinel error text for an abandoned run, so the caller can tell "the
    // user moved on" apart from a real failure worth showing.
    static constexpr const char* kCancelled = "cancelled";

    // How the last run was split. These add up to the pipeline stage count:
    // stages skipped by the dirty-hash cache are counted separately rather
    // than vanishing, so "0 CPU, 0 GPU" never means "nothing happened".
    int GpuStageCount()    const { return m_gpuStages; }

    // Stages that wanted the GPU and did not get it, with the reason.
    //
    // A GPU failure falls back to the CPU rather than failing the run, which is
    // right -- a broken kernel should degrade to a slow correct result. But a
    // SILENT fallback is indistinguishable from the GPU merely being slow, and
    // that is exactly what hid a device hang on 45 MP raws: every develop was
    // falling back and recovering, for months, with nothing said.
    const std::vector<std::string>& GpuFallbacks() const { return m_gpuFallbacks; }
    int CpuStageCount()    const { return m_cpuStages; }
    int CachedStageCount() const { return m_cachedStages; }

    // Stages skipped because their settings would change nothing. Reported so a
    // stacked script can show "12 stages, 9 bypassed" rather than leaving the
    // user to wonder whether an effect they turned off is still costing them.
    int BypassedStageCount() const { return m_bypassedStages; }

    // Index of the first stage that actually re-ran. Everything below it kept
    // its cached output, so a viewer reading from there shows the same pixels
    // it showed last run -- which is what lets the UI skip re-uploading it.
    size_t FirstDirtyStage() const { return m_firstDirty; }

    const Data* Resolve(PortRef r, const std::vector<Data>* sources) const;

    std::vector<Stage>&       Stages()       { return m_stages; }
    const std::vector<Stage>& Stages() const { return m_stages; }
    const std::vector<ViewerDecl>& Viewers() const { return m_viewers; }
    const std::vector<SaveDecl>&   Saves()   const { return m_saves; }

    // Writes everything the script's save() calls asked for, AFTER a run.
    //
    // Separate from Execute rather than folded into it: a save is disk IO on
    // the worker's critical path, and Execute runs on every slider tick while
    // a save should happen when the caller decides it should.
    //
    // A group writes one file per frame with a number appended, since a single
    // path cannot name N images. Returns false if any write failed, with `err`
    // naming the first -- and keeps going, because a full disk on frame 3 is
    // not a reason to skip frames 4 through 8.
    bool RunSaves(std::vector<Data>* sources, std::string* err,
                  std::vector<std::string>* written = nullptr);

private:
    // True if `a` and `b` are the same algorithm with the same wiring/params.
    static bool SameStage(const Stage& a, const Stage& b);

    // Runs one stage on the GPU. Returns false (with `err` set) if anything
    // about the stage is unsupported, so the caller can fall back to the CPU.
    std::vector<int> FusableChain(int reduceStage, PortRef* srcPort) const;
    bool RunFusedReduction(int reduceStage, const std::vector<int>& chain,
                           PortRef srcPort, const std::vector<Data>* sources,
                           ComputeContext* gpu, ExecMode mode,
                           const CancelToken* cancel, Progress* progress,
                           std::string* err);
    bool RunStageOnce(Stage& s, const std::vector<const Data*>& in,
                      ComputeContext* gpu, ExecMode mode,
                      const CancelToken* cancel, std::string* err);
    bool BroadcastStage(Stage& s, const std::vector<const Data*>& in,
                        ComputeContext* gpu, ExecMode mode,
                        const CancelToken* cancel, Progress* progress,
                        std::string* err);
    bool RunReduction(Stage& s, const std::vector<const Data*>& in,
                      const CancelToken* cancel, std::string* err);
    bool RunStageGpu(Stage& s, const std::vector<const Data*>& in,
                     ComputeContext* gpu, std::string* err);

    std::vector<Stage>      m_stages;
    std::vector<ViewerDecl> m_viewers;
    std::vector<SaveDecl>   m_saves;
    int                     m_gpuStages = 0;
    std::vector<std::string> m_gpuFallbacks;
    int                     m_cpuStages = 0;
    int                     m_cachedStages = 0;
    int                     m_bypassedStages = 0;
    size_t                  m_firstDirty = 0;

    // When the current Execute() began, so the live tallies published to
    // Progress can report elapsed time. A member rather than a local because
    // the per-frame loops -- which is where a slow run actually spends its
    // time -- publish from inside the helpers, not from Execute's own scope.
    std::chrono::steady_clock::time_point m_runStart{};

    // Publishes the running CPU/GPU/elapsed tallies, if anyone is listening.
    // One place, so the several call sites cannot drift apart.
    //
    // The GPU figure comes from the compute context rather than from timing
    // stages: a "GPU stage" only RECORDS commands, and the device does not
    // start until a flush submits them. Wall clock around such a stage measures
    // recording -- near zero -- so a split derived that way would report almost
    // all of a GPU-heavy run as CPU time.
    void PublishStats(Progress* progress, const ComputeContext* gpu) const;
};

} // namespace tglab
