// GPU compute: a dedicated compute queue plus the plumbing to dispatch an
// HLSL kernel over an image.
//
// This runs on the worker thread, so it gets its OWN command queue and
// allocators. Sharing the direct queue with ImGui's submission is the classic
// source of intermittent, hard-to-reproduce corruption.
#pragma once

#include <d3d12.h>

#include <string>
#include <vector>

#include "../core/image.h"
#include "shader.h"

namespace tglab {

class Device;

// A GPU-resident image. Created as a UAV so a compute shader can write it,
// with SIMULTANEOUS_ACCESS so the UI's direct queue may read it for display
// without a cross-queue state transition.
struct GpuImage {
    ID3D12Resource* res = nullptr;
    ImageDesc       desc{};

    // What state the resource is currently in, so Dispatch transitions it only
    // when it actually needs to change.
    //
    // Tracked rather than assumed, because the same texture is bound as a UAV
    // by one stage and an SRV by the next. Without it every dispatch wrote a
    // resource still sitting in COMMON, which GPU-based validation reported as
    // "Incompatible texture barrier layout" on all 14 GPU algorithms.
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;

    bool Valid() const { return res != nullptr; }
    void Release() { if (res) { res->Release(); res = nullptr; } }
};

// One compiled kernel plus its root signature and PSO.
// Owns its D3D12 objects: stages hold these in a shared_ptr, so the destructor
// is what actually frees them when the last pipeline referencing it goes away.
struct ComputeKernel {
    ID3D12RootSignature* root = nullptr;
    ID3D12PipelineState* pso  = nullptr;

    ComputeKernel() = default;
    ~ComputeKernel() { Release(); }

    ComputeKernel(const ComputeKernel&)            = delete;
    ComputeKernel& operator=(const ComputeKernel&) = delete;

    bool Valid() const { return root && pso; }
    void Release() {
        if (pso)  { pso->Release();  pso  = nullptr; }
        if (root) { root->Release(); root = nullptr; }
    }
};

class ComputeContext {
public:
    bool Init(ID3D12Device* device);
    void Shutdown();

    // False once a dispatch has hung or the device was removed; the pipeline
    // then stops offering the GPU path for the rest of the session.
    bool Ready() const { return m_device != nullptr && !m_deviceLost; }
    bool DeviceLost() const { return m_deviceLost; }

    // Builds a kernel from HLSL source. Root signature is fixed by convention:
    //   t0..t3  input SRVs
    //   u0..u3  output UAVs
    //   b0      32 root constants (width, height, then algorithm parameters)
    bool CreateKernel(const std::string& hlsl, const std::string& entry,
                      const std::string& debugName,
                      ComputeKernel* out, std::string* errors);

    bool CreateImage(const ImageDesc& d, GpuImage* out);
    bool Upload(const ImageView& src, GpuImage* dst);
    bool Readback(const GpuImage& src, ImageView* dst);

    // Records and submits one dispatch, then waits. `constants` are the b0
    // root constants beyond the automatic width/height pair.
    bool Dispatch(const ComputeKernel& k,
                  const std::vector<const GpuImage*>& inputs,
                  const std::vector<GpuImage*>& outputs,
                  const std::vector<uint32_t>& constants,
                  std::string* err);

    ShaderCompiler& Compiler() { return m_compiler; }

    // --- histogram ----------------------------------------------------------
    //
    // The info panel's histogram, computed where the pixels already are.
    //
    // The point is not that binning is slow on the CPU; it is that getting the
    // pixels there is. The stats worker subsamples to 512x512, but Subsample()
    // begins by mapping the *whole* image, so a 21 MP result paid a 66 ms
    // readback of 84 MB in order to keep 262k pixels -- 0.3% of what it
    // fetched. Here only the bins come back: 4 KB.
    //
    // Bins are 256 per channel over [rangeMin, rangeMax], matching
    // algo_util/Histogram so the panel's statistics are the same either way.
    // For an RGBA8 image the range is fixed at 0..255; for a float one it is
    // the observed luma range, because a float image has no natural bounds.
    struct HistogramResult {
        std::vector<uint32_t> r, g, b, luma;   // 256 each; rgb empty for R32F
        double   rangeMin = 0.0;
        double   rangeMax = 255.0;
        uint64_t count    = 0;
    };
    bool BuildHistogram(const GpuImage& src, HistogramResult* out, std::string* err);

    // Submits any recorded work and waits for it.
    //
    // Dispatch() only *records*; consecutive dispatches accumulate into one
    // command list and are submitted together. Nothing needs the pixels between
    // two GPU stages -- the intermediate stays on the device and a UAV barrier
    // orders them -- so flushing after each one was a submit-and-block per
    // stage for no benefit. Measured at ~20% of a 12-stage chain.
    //
    // Upload() and Readback() flush on their own, because those genuinely move
    // pixels across the bus; callers only need this when they want completion
    // for its own sake, such as before tearing down resources.
    bool Flush(std::string* err);

private:
    // Opens the command list, reusing an in-progress batch rather than
    // discarding it. See the definition for why this matters.
    bool BeginRecording();

    // Histogram scratch, created on first use and reused. The bins live in a
    // 256x4 R32_UINT texture (rows R, G, B, luma) and the luma range in a 2x1
    // one; integer textures because HLSL atomics need them, and textures rather
    // than buffers because the fixed root signature binds texture UAVs.
    bool CreateHistogramResources(std::string* err);
    ComputeKernel   m_histRangeKernel{};
    ComputeKernel   m_histBinKernel{};
    ID3D12Resource* m_histBins  = nullptr;
    ID3D12Resource* m_histRange = nullptr;
    ID3D12Resource* m_histRead  = nullptr;   // readback staging for both

    // ClearUnorderedAccessViewUint wants two handles for the same UAV: a
    // shader-visible one and a CPU-readable one. The main heap is shader
    // visible, which makes it CPU write-only, so the clear needs its own
    // non-shader-visible heap. Two slots: bins and range.
    ID3D12DescriptorHeap* m_histClearHeap = nullptr;


    // True when work is recorded but not yet submitted, so BeginRecording()
    // knows not to reset the list out from under it.
    bool                       m_pendingWork = false;

    ID3D12Device*              m_device    = nullptr;
    ID3D12CommandQueue*        m_queue     = nullptr;
    ID3D12CommandAllocator*    m_alloc     = nullptr;
    ID3D12GraphicsCommandList* m_list      = nullptr;
    ID3D12Fence*               m_fence     = nullptr;
    HANDLE                     m_event     = nullptr;
    UINT64                     m_fenceVal  = 0;

    // Non-shader-visible staging heap for the descriptors a dispatch needs,
    // copied into a shader-visible heap at record time.
    ID3D12DescriptorHeap* m_srvHeap = nullptr;
    UINT                  m_srvStride = 0;

    // Descriptors are consumed by the GPU when the command list *executes*,
    // not when Dispatch() records it. Batched dispatches must therefore each
    // own a distinct slice of the heap -- reusing slot 0 every time silently
    // gives every dispatch in a batch the last one's bindings.
    UINT                  m_heapCursor = 0;

    std::vector<ID3D12Resource*> m_staging;   // upload/readback buffers, freed on flush
    ShaderCompiler               m_compiler;
    bool                         m_deviceLost = false;
};

} // namespace tglab
