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

struct Stage {
    std::unique_ptr<AlgorithmBase> algo;
    std::string          algoName;
    std::vector<PortRef> inputs;
    std::vector<Data>    outputs;
    uint64_t             paramHash = 0;
    // Versions of the palette images this stage read, so replacing one is
    // detected as a change rather than looking identical.
    uint64_t             sourceHash = 0;
    bool                 valid     = false;   // outputs hold a usable result
    int                  line      = 0;       // for error messages

    // Compiled kernel, cached so dragging a slider does not recompile HLSL
    // every frame. shared_ptr because Stage is moved between pipelines and the
    // kernel outlives any single run.
    std::shared_ptr<ComputeKernel> kernel;
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
                  std::vector<PortRef> inputs, size_t numOutputs, int line);
    void AddViewer(std::string name, PortRef src);

    // --- execution (phase 2) ---
    // `sources` are the palette images, indexed by PortRef::port when stage==-1.
    // Reuses cached outputs from `prev` where the stage is unchanged.
    // `gpu` may be null, in which case everything runs on the CPU.
    // `sourceVersions` runs parallel to `sources`: a value that changes when
    // that palette image is replaced. Without it a stage reading PortRef{-1, i}
    // looks identical no matter which file backs slot i, so swapping an image
    // would reuse the cached output.
    bool Execute(std::vector<Data>* sources, Pipeline* prev, std::string* err,
                 ComputeContext* gpu = nullptr, ExecMode mode = ExecMode::Auto,
                 const std::vector<uint64_t>* sourceVersions = nullptr);

    // How the last run was split. These add up to the pipeline stage count:
    // stages skipped by the dirty-hash cache are counted separately rather
    // than vanishing, so "0 CPU, 0 GPU" never means "nothing happened".
    int GpuStageCount()    const { return m_gpuStages; }
    int CpuStageCount()    const { return m_cpuStages; }
    int CachedStageCount() const { return m_cachedStages; }

    const Data* Resolve(PortRef r, const std::vector<Data>* sources) const;

    std::vector<Stage>&       Stages()       { return m_stages; }
    const std::vector<Stage>& Stages() const { return m_stages; }
    const std::vector<ViewerDecl>& Viewers() const { return m_viewers; }

private:
    // True if `a` and `b` are the same algorithm with the same wiring/params.
    static bool SameStage(const Stage& a, const Stage& b);

    // Runs one stage on the GPU. Returns false (with `err` set) if anything
    // about the stage is unsupported, so the caller can fall back to the CPU.
    bool RunStageGpu(Stage& s, const std::vector<const Data*>& in,
                     ComputeContext* gpu, std::string* err);

    std::vector<Stage>      m_stages;
    std::vector<ViewerDecl> m_viewers;
    int                     m_gpuStages = 0;
    int                     m_cpuStages = 0;
    int                     m_cachedStages = 0;
};

} // namespace tglab
