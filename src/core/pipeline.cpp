#include "pipeline.h"

#include <cassert>

#include "../gpu/compute.h"
#include "../gpu/gpu_image.h"

#include <windows.h>
#include <cstdio>

namespace tglab {

void Pipeline::Clear() {
    m_stages.clear();
    m_viewers.clear();
}

int Pipeline::AddStage(std::unique_ptr<AlgorithmBase> algo, std::string name,
                       std::vector<PortRef> inputs, size_t numOutputs, int line) {
    Stage s;
    s.paramHash = algo->ParamHash();
    s.algo      = std::move(algo);
    s.algoName  = std::move(name);
    s.inputs    = std::move(inputs);
    s.outputs.resize(numOutputs);
    s.line      = line;
    m_stages.push_back(std::move(s));
    return int(m_stages.size()) - 1;
}

void Pipeline::AddViewer(std::string name, PortRef src) {
    // Re-declaring the same viewer name updates it rather than duplicating.
    for (ViewerDecl& v : m_viewers) {
        if (v.name == name) {
            v.source = src;
            return;
        }
    }
    m_viewers.push_back({std::move(name), src});
}

const Data* Pipeline::Resolve(PortRef r, const std::vector<Data>* sources) const {
    if (r.stage < 0) {
        if (!sources || r.port < 0 || size_t(r.port) >= sources->size()) return nullptr;
        return &(*sources)[size_t(r.port)];
    }
    if (size_t(r.stage) >= m_stages.size()) return nullptr;
    const Stage& s = m_stages[size_t(r.stage)];
    if (r.port < 0 || size_t(r.port) >= s.outputs.size()) return nullptr;
    return &s.outputs[size_t(r.port)];
}

bool Pipeline::SameStage(const Stage& a, const Stage& b) {
    return a.algoName == b.algoName && a.inputs == b.inputs &&
           a.paramHash == b.paramHash && a.sourceHash == b.sourceHash &&
           a.outputs.size() == b.outputs.size();
}

bool Pipeline::Execute(std::vector<Data>* sources, Pipeline* prev, std::string* err,
                       ComputeContext* gpu, ExecMode mode,
                       const std::vector<uint64_t>* sourceVersions,
                       const CancelToken* cancel) {
    // Stamp each stage with the versions of the palette images it reads, before
    // any cache comparison. PortRef{-1, i} is identical whichever file backs
    // slot i, so without this a swapped image reuses the cached output -- which
    // looks like the app ignoring the new file until some other parameter
    // changes and incidentally busts the cache.
    if (sourceVersions) {
        for (Stage& s : m_stages) {
            uint64_t h = 1469598103934665603ull;   // FNV-1a
            for (const PortRef& r : s.inputs) {
                if (r.stage >= 0) continue;        // not a palette image
                const uint64_t v = size_t(r.port) < sourceVersions->size()
                                       ? (*sourceVersions)[size_t(r.port)] : 0;
                for (int b = 0; b < 8; ++b) {
                    h ^= (v >> (b * 8)) & 0xff;
                    h *= 1099511628211ull;
                }
            }
            s.sourceHash = h;
        }
    }

    m_gpuStages = 0;
    m_cpuStages = 0;
    m_cachedStages = 0;
    // Find the first stage that differs from the previous run. Everything
    // before it can reuse its cached output. Comparing by hash means there is
    // no dirty flag to forget to set.
    size_t firstDirty = 0;
    if (prev) {
        const size_t n = std::min(m_stages.size(), prev->m_stages.size());
        while (firstDirty < n &&
               prev->m_stages[firstDirty].valid &&
               SameStage(m_stages[firstDirty], prev->m_stages[firstDirty])) {
            // Steal the cached outputs — Data is move-only, and the previous
            // pipeline is discarded right after this call. The compiled kernel
            // comes along too, so an unchanged stage never recompiles.
            m_stages[firstDirty].outputs = std::move(prev->m_stages[firstDirty].outputs);
            m_stages[firstDirty].kernel  = prev->m_stages[firstDirty].kernel;
            m_stages[firstDirty].valid   = true;
            ++m_cachedStages;
            ++firstDirty;
        }
    }

    // Dirty stages still reuse their compiled kernel when they are the same
    // algorithm as last run — a parameter change does not alter the HLSL, and
    // recompiling on every slider tick would defeat the point of the GPU path.
    if (prev) {
        for (size_t i = firstDirty; i < m_stages.size(); ++i) {
            if (i >= prev->m_stages.size()) break;
            if (m_stages[i].algoName == prev->m_stages[i].algoName)
                m_stages[i].kernel = prev->m_stages[i].kernel;
        }
    }

    // Any exit from the stage loop -- an error, a cancellation, or falling off
    // the end -- must submit what the GPU stages recorded. Dispatch() no longer
    // flushes, so leaving early would strand a half-built command list, and the
    // *next* run's BeginRecording() would append to it rather than starting
    // clean. There are several early returns below, so this is a guard rather
    // than a line repeated at each of them.
    struct GpuBatchGuard {
        ComputeContext* gpu;
        ~GpuBatchGuard() {
            if (!gpu || !gpu->Ready()) return;
            std::string ignored;
            gpu->Flush(&ignored);   // the caller already has the real error
        }
    } batchGuard{gpu};

    for (size_t i = firstDirty; i < m_stages.size(); ++i) {
        // Between stages as well as within them: a pipeline of several
        // moderately slow stages should abandon at the next boundary even if no
        // single algorithm polls the token itself.
        if (cancel && cancel->Cancelled()) { *err = kCancelled; return false; }

        Stage& s = m_stages[i];
        s.valid = false;

        // Gather inputs.
        std::vector<const Data*> in;
        in.reserve(s.inputs.size());
        for (const PortRef& r : s.inputs) {
            const Data* d = Resolve(r, sources);
            if (!d || TypeOf(*d) == DataType::None) {
                *err = "line " + std::to_string(s.line) + ": '" + s.algoName +
                       "' has an input that produced no data";
                return false;
            }
            in.push_back(d);
        }

        // Allocate outputs. M1: every port is an image sized from input 0,
        // with the format resolved from the port's FormatSpec.
        const PortList outPorts = s.algo->Outputs();
        ImageDesc base{};
        if (!in.empty() && std::holds_alternative<Image>(*in[0]))
            base = std::get<Image>(*in[0]).Desc();

        for (size_t p = 0; p < s.outputs.size(); ++p) {
            ImageDesc d = base;
            switch (outPorts[p].format) {
                case FormatSpec::RGBA8:       d.format = Format::RGBA8;   break;
                case FormatSpec::R32F:        d.format = Format::R32F;    break;
                case FormatSpec::RGBA32F:     d.format = Format::RGBA32F; break;
                case FormatSpec::SameAsInput: /* keep base.format */      break;
                case FormatSpec::Any:         /* keep base.format */      break;
            }
            if (!d.Valid()) {
                *err = "line " + std::to_string(s.line) + ": '" + s.algoName +
                       "' could not determine output size";
                return false;
            }
            Image img;
            img.Alloc(d);
            s.outputs[p] = Data{std::move(img)};
        }

        // GPU when asked for and available; otherwise CPU. A GPU failure falls
        // back rather than failing the run — a broken kernel should degrade to
        // a slow correct result, not an empty viewer.
        const bool wantGpu = gpu && mode != ExecMode::ForceCPU &&
                             s.algo->HasGPU() && s.algo->GpuSource();
        bool ranOnGpu = false;
        if (wantGpu) {
            std::string gpuErr;
            if (RunStageGpu(s, in, gpu, &gpuErr)) {
                ranOnGpu = true;
                ++m_gpuStages;
            } else if (mode == ExecMode::ForceGPU) {
                *err = "line " + std::to_string(s.line) + ": '" + s.algoName +
                       "' GPU path failed: " + gpuErr;
                return false;
            }
        }

        if (!ranOnGpu) {
            RunCtx ctx(in, s.outputs, cancel);
            s.algo->RunCPU(ctx);
            ++m_cpuStages;

            // An algorithm that honoured the token has written only part of its
            // output. Leaving the stage valid would cache that partial result
            // and, worse, let a later run reuse it as though it were finished.
            if (cancel && cancel->Cancelled()) {
                s.valid = false;
                *err = kCancelled;
                return false;
            }
        }
        s.valid = true;
    }

    // Submit explicitly on the success path so a failure here is *reported*.
    // The guard above would flush anyway, but it deliberately swallows the
    // error -- on an early return the caller already has a better one, whereas
    // here this is the only thing that went wrong. Flushing twice is harmless:
    // the second call finds nothing pending.
    if (gpu && gpu->Ready()) {
        std::string flushErr;
        if (!gpu->Flush(&flushErr)) {
            *err = flushErr;
            return false;
        }
    }

    return true;
}

bool Pipeline::RunStageGpu(Stage& s, const std::vector<const Data*>& in,
                           ComputeContext* gpu, std::string* err) {
    // M3 supports the common shape: image inputs and image outputs, one
    // dispatch. Anything else stays on the CPU.
    for (const Data* d : in)
        if (!d || !std::holds_alternative<Image>(*d)) { *err = "non-image input"; return false; }
    for (const Data& d : s.outputs)
        if (!std::holds_alternative<Image>(d)) { *err = "non-image output"; return false; }

    // Kernels are compiled once per stage and cached on the stage, so dragging
    // a slider does not recompile HLSL every frame.
    if (!s.kernel) {
        s.kernel = std::make_shared<ComputeKernel>();
        std::string compileErr;
        if (!gpu->CreateKernel(s.algo->GpuSource(), "main", s.algoName,
                               s.kernel.get(), &compileErr)) {
            s.kernel.reset();
            *err = compileErr;
            return false;
        }
    }

    // Acquire through Image, which owns its GPU resource and only transfers
    // when a side is stale. Chained GPU stages therefore upload once at the
    // head and never read back in the middle: the intermediate images stay
    // GPU-resident, and nothing asks for their CPU pixels.
    std::vector<const GpuImage*> gin;
    std::vector<GpuImage*>       gout;
    gin.reserve(in.size());
    gout.reserve(s.outputs.size());

    for (const Data* d : in) {
        Image& img = const_cast<Image&>(std::get<Image>(*d));
        GpuResidency* g = img.AcquireGpuRead(*gpu);
        if (!g) { *err = "could not make an input GPU-resident"; return false; }
        gin.push_back(&g->image);
    }
    for (Data& d : s.outputs) {
        Image& img = std::get<Image>(d);
        GpuResidency* g = img.AcquireGpuWrite(*gpu);
        if (!g) { *err = "could not allocate a GPU output"; return false; }
        gout.push_back(&g->image);
    }

    // TGLAB_GPUDBG=1 logs every dispatch. Worth keeping: when the device
    // hung after a script switch, the absence of these lines is what proved
    // the fault was not in compute at all, but in freeing view textures.
    if (GetEnvironmentVariableA("TGLAB_GPUDBG", nullptr, 0) > 0) {
        const ImageDesc& od = std::get<Image>(s.outputs[0]).Desc();
        std::fprintf(stderr, "[gpu] dispatch %s %dx%d\n",
                     s.algoName.c_str(), od.width, od.height);
        std::fflush(stderr);
    }
    const int iterations = std::max(1, s.algo->GpuIterations());

    if (iterations == 1) {
        if (!gpu->Dispatch(*s.kernel, gin, gout, s.algo->GpuConstants(0), err)) return false;
        // No readback here. Outputs stay GPU-resident; whoever needs CPU pixels
        // (a later CPU stage, or a viewer) triggers the transfer via
        // MapCpuRead().
        return true;
    }

    // Iterative: ping-pong between the real output and one scratch image.
    //
    // Each pass must read the *completed* previous pass, which a single
    // dispatch cannot provide -- threads within one dispatch have no ordering
    // relative to each other. Writing into the image being read would be a
    // race, so two buffers alternate.
    //
    // Only single-input, single-output stages iterate; anything else has no
    // obvious pairing to alternate between.
    if (gin.size() != 1 || gout.size() != 1) {
        *err = "iterative GPU stages must have exactly one input and one output";
        return false;
    }

    const ImageDesc desc = std::get<Image>(s.outputs[0]).Desc();
    if (!s.gpuScratch) {
        s.gpuScratch = std::make_shared<GpuImage>();
        if (!gpu->CreateImage(desc, s.gpuScratch.get())) {
            s.gpuScratch.reset();
            *err = "could not allocate the scratch image for an iterative stage";
            return false;
        }
        s.scratchDesc = desc;
    } else if (s.scratchDesc.width != desc.width || s.scratchDesc.height != desc.height ||
               s.scratchDesc.format != desc.format) {
        // The image changed size, so the cached scratch no longer matches.
        s.gpuScratch = std::make_shared<GpuImage>();
        if (!gpu->CreateImage(desc, s.gpuScratch.get())) {
            s.gpuScratch.reset();
            *err = "could not resize the scratch image for an iterative stage";
            return false;
        }
        s.scratchDesc = desc;
    }

    // Choose the starting parity so the LAST pass lands in the real output;
    // otherwise the result would sit in scratch and be silently discarded.
    // With an odd count, start by writing into the output; with an even count,
    // start in scratch so the final flip ends up there.
    const GpuImage* readFrom = gin[0];
    GpuImage*       writeTo  = (iterations % 2 == 1) ? gout[0] : s.gpuScratch.get();

    for (int i = 0; i < iterations; ++i) {
        const std::vector<const GpuImage*> passIn{readFrom};
        const std::vector<GpuImage*>       passOut{writeTo};
        if (!gpu->Dispatch(*s.kernel, passIn, passOut, s.algo->GpuConstants(i), err))
            return false;

        // This pass's output becomes the next pass's input, and the buffer just
        // read becomes the next target -- except on the first pass, where the
        // input is the upstream image and must not be overwritten.
        // Submit before the next pass, which READS what this one just wrote.
        //
        // Dispatch()'s UAV barrier orders write-against-write on the same
        // binding; it does not order a write followed by a read through an SRV.
        // A state-transition barrier does not help either -- transitioning out
        // and back gives the driver nothing to synchronise against -- so the
        // ordering has to come from a submit. That was free while every
        // dispatch flushed, and became a garbage image the moment they started
        // batching.
        //
        // The cost is bounded: only iterative stages pay it, and only between
        // their own passes. Ordinary chains still batch, which is where the
        // measured win came from.
        if (i + 1 < iterations) {
            std::string fe;
            if (!gpu->Flush(&fe)) { *err = fe; return false; }
        }

        readFrom = writeTo;
        writeTo  = (writeTo == gout[0]) ? s.gpuScratch.get() : gout[0];
    }

    return true;
}

} // namespace tglab
