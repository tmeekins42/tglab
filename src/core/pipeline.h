// Pipeline: the recorded stage list plus the dirty-by-hash execution.
//
// Phase 1 (interpreter) RECORDS stages; it never runs an algorithm.
// Phase 2 (Execute) runs from the first stage whose inputs or parameters
// changed, reusing cached outputs before that point.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "algorithm.h"
#include "data.h"

namespace tglab {

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
                  std::string reduceAxis = {});
    void AddViewer(std::string name, PortRef src);

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
                 const CancelToken* cancel = nullptr);

    // Sentinel error text for an abandoned run, so the caller can tell "the
    // user moved on" apart from a real failure worth showing.
    static constexpr const char* kCancelled = "cancelled";

    // How the last run was split. These add up to the pipeline stage count:
    // stages skipped by the dirty-hash cache are counted separately rather
    // than vanishing, so "0 CPU, 0 GPU" never means "nothing happened".
    int GpuStageCount()    const { return m_gpuStages; }
    int CpuStageCount()    const { return m_cpuStages; }
    int CachedStageCount() const { return m_cachedStages; }

    // Index of the first stage that actually re-ran. Everything below it kept
    // its cached output, so a viewer reading from there shows the same pixels
    // it showed last run -- which is what lets the UI skip re-uploading it.
    size_t FirstDirtyStage() const { return m_firstDirty; }

    const Data* Resolve(PortRef r, const std::vector<Data>* sources) const;

    std::vector<Stage>&       Stages()       { return m_stages; }
    const std::vector<Stage>& Stages() const { return m_stages; }
    const std::vector<ViewerDecl>& Viewers() const { return m_viewers; }

private:
    // True if `a` and `b` are the same algorithm with the same wiring/params.
    static bool SameStage(const Stage& a, const Stage& b);

    // Runs one stage on the GPU. Returns false (with `err` set) if anything
    // about the stage is unsupported, so the caller can fall back to the CPU.
    bool RunStageOnce(Stage& s, const std::vector<const Data*>& in,
                      ComputeContext* gpu, ExecMode mode,
                      const CancelToken* cancel, std::string* err);
    bool BroadcastStage(Stage& s, const std::vector<const Data*>& in,
                        ComputeContext* gpu, ExecMode mode,
                        const CancelToken* cancel, std::string* err);
    bool RunReduction(Stage& s, const std::vector<const Data*>& in,
                      const CancelToken* cancel, std::string* err);
    bool RunStageGpu(Stage& s, const std::vector<const Data*>& in,
                     ComputeContext* gpu, std::string* err);

    std::vector<Stage>      m_stages;
    std::vector<ViewerDecl> m_viewers;
    int                     m_gpuStages = 0;
    int                     m_cpuStages = 0;
    int                     m_cachedStages = 0;
    size_t                  m_firstDirty = 0;
};

} // namespace tglab
