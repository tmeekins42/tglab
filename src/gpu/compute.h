// GPU compute: a dedicated compute queue plus the plumbing to dispatch an
// HLSL kernel over an image.
//
// This runs on the worker thread, so it gets its OWN command queue and
// allocators. Sharing the direct queue with ImGui's submission is the classic
// source of intermittent, hard-to-reproduce corruption.
#pragma once

#include <d3d12.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
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

    // Frees the resource when the last owner goes away.
    //
    // Without this, holding a GpuImage in a shared_ptr leaked the texture: the
    // wrapper was freed and the ID3D12Resource was not. The iterative scratch
    // is held exactly that way and is reallocated whenever a parameter changes,
    // so dragging a slider leaked one full-size texture per tick -- 16 MB a
    // move for an RGBA32F scratch at 1024x1024, until the card ran out.
    //
    // Copying is deleted rather than refcounted: a GpuImage is an owner, and a
    // copy would double-free. Move transfers ownership and leaves the source
    // empty.
    ~GpuImage() { Release(); }

    GpuImage() = default;
    GpuImage(const GpuImage&)            = delete;
    GpuImage& operator=(const GpuImage&) = delete;

    GpuImage(GpuImage&& o) noexcept
        : res(o.res), desc(o.desc), state(o.state) { o.res = nullptr; }

    GpuImage& operator=(GpuImage&& o) noexcept {
        if (this != &o) {
            Release();
            res = o.res; desc = o.desc; state = o.state;
            o.res = nullptr;
        }
        return *this;
    }

    bool Valid() const { return res != nullptr; }
    void Release() { if (res) { res->Release(); res = nullptr; } }
};

// A GPU-resident flat array of bytes, bound as a raw buffer
// (ByteAddressBuffer / RWByteAddressBuffer).
//
// WHY BUFFERS AS WELL AS IMAGES. Everything else here is an image, and an
// image is the right shape for pixels. A neural network's tensors are not: a
// 64-channel feature map at 1024x1024 is 64 planes, and a texture side is
// capped at 16384, so stacking them overflows; its weights are a 512x512x3x3
// block with no 2D meaning at all. A flat array indexes both without packing
// tricks, and the same descriptor tables bind it -- an SRV range takes a
// buffer view as readily as a texture view, so kernels and the root signature
// are unchanged.
//
// Owned and move-only for GpuImage's reasons.
struct GpuBuffer {
    ID3D12Resource*       res   = nullptr;
    uint64_t              bytes = 0;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;

    ~GpuBuffer() { Release(); }

    GpuBuffer() = default;
    GpuBuffer(const GpuBuffer&)            = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;

    GpuBuffer(GpuBuffer&& o) noexcept : res(o.res), bytes(o.bytes), state(o.state) {
        o.res = nullptr;
    }
    GpuBuffer& operator=(GpuBuffer&& o) noexcept {
        if (this != &o) {
            Release();
            res = o.res; bytes = o.bytes; state = o.state;
            o.res = nullptr;
        }
        return *this;
    }

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
    // True only when Init() ran to completion.
    //
    // Deliberately NOT "m_device != nullptr": Init() assigns the device
    // first and can fail at any of six later steps, so that test reported
    // ready for a context with no queue, no command list, or -- as on a
    // GPU-less CI runner -- no shader compiler. Every guard written as
    // `if (!Ready()) return;` was then a no-op, and the first dispatch
    // dereferenced a null kernel.
    bool Ready() const { return m_ready && !m_deviceLost; }
    bool DeviceLost() const { return m_deviceLost; }

    // Builds a kernel from HLSL source. Root signature is fixed by convention:
    //   t0..t3  input SRVs
    //   u0..u3  output UAVs
    //   b0      32 root constants (width, height, then algorithm parameters)
    bool CreateKernel(const std::string& hlsl, const std::string& entry,
                      const std::string& debugName,
                      ComputeKernel* out, std::string* errors);

    // A kernel compiled once per context and kept until Shutdown, under
    // `key` -- which must name the source exactly, variants included.
    //
    // WHY. A helper that owns its kernels (the network engine, the k-NN
    // search) is naturally created per call, and each creation recompiled
    // every kernel through DXC: measured, ten kernels at tens of milliseconds
    // each turned an 85 ms network into a 550 ms one inside the pipeline,
    // where a fresh engine serves every frame. Thread-safe; null on failure.
    const ComputeKernel* SharedKernel(const std::string& key, const std::string& hlsl,
                                      std::string* err);

    bool CreateImage(const ImageDesc& d, GpuImage* out);
    bool Upload(const ImageView& src, GpuImage* dst);
    bool Readback(const GpuImage& src, ImageView* dst);

    // Records and submits one dispatch, then waits. `constants` are the b0
    // root constants beyond the automatic width/height pair.
    //
    // `groupsX` / `groupsY`, when non-zero, set the thread-group grid
    // directly instead of deriving it from the first output's size. For a
    // kernel whose groups are not 8x8 pixels of that output -- the splat
    // rasteriser runs one 16x16 group per screen tile, so that a tile's
    // threads can reduce their gradients together in group-shared memory.
    bool Dispatch(const ComputeKernel& k,
                  const std::vector<const GpuImage*>& inputs,
                  const std::vector<GpuImage*>& outputs,
                  const std::vector<uint32_t>& constants,
                  std::string* err, uint32_t groupsX = 0,
                  uint32_t groupsY = 0);

    // --- raw buffers --------------------------------------------------------
    //
    // `bytes` is rounded up to a multiple of 4, the unit of a raw view.
    bool CreateBuffer(uint64_t bytes, GpuBuffer* out);
    // Copies `bytes` from `src` into `dst` at byte `offset`, and waits.
    bool UploadBuffer(const void* src, uint64_t bytes, GpuBuffer* dst, uint64_t offset = 0);
    // Copies the first `bytes` of `src` back to `dst`, and waits.
    bool ReadbackBuffer(const GpuBuffer& src, void* dst, uint64_t bytes);

    // A scratch buffer of at least `bytes`, reused when one of a suitable size
    // has been handed back -- see RecycleBuffer. Contents are unspecified.
    //
    // WHY A POOL. A network allocates every intermediate fresh: about a
    // gigabyte per frame for the DeDoDe descriptor at 784 pixels, dozens of
    // committed resources, each paid for in the OS zeroing its pages. The
    // shapes are identical frame to frame, so after the first frame every
    // request can be met from what the previous one gave back.
    //
    // Reuse while a recorded batch still references the buffer is safe: the
    // work is on one queue, recorded in order, and every bind records the
    // state transition (or UAV barrier) that orders the new use after the
    // old one. Thread-safe.
    bool AcquireBuffer(uint64_t bytes, GpuBuffer* out);
    // Hands a buffer back for reuse; past the pool's budget the largest
    // pooled ones are freed instead.
    void RecycleBuffer(GpuBuffer&& b);

    // Dispatch with buffers bound instead of images: inputs at t0.., outputs
    // at u0.., as raw views. Unlike Dispatch there is no image to size the
    // grid from, so the caller gives it, and ALL 32 root constants are the
    // caller's -- there is no width/height pair at the front.
    bool DispatchBuffers(const ComputeKernel& k,
                         const std::vector<const GpuBuffer*>& inputs,
                         const std::vector<GpuBuffer*>& outputs,
                         const std::vector<uint32_t>& constants,
                         uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ,
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

    // Milliseconds spent submitting batches and waiting for them, accumulated
    // since the last ResetGpuMs().
    //
    // This is the only number that can honestly be called "GPU time". Dispatch
    // merely records into a batch; the device does no work until Flush submits
    // and waits. Timing a stage's wall clock therefore measures RECORDING and
    // reads as near-zero, with the real cost landing on whichever stage happens
    // to trigger the flush.
    double GpuMs() const { return m_gpuMs; }
    void   ResetGpuMs() { m_gpuMs = 0.0; }

    // --- concurrency --------------------------------------------------------
    //
    // NOTHING ELSE IN THIS CLASS IS THREAD-SAFE. One command queue, one
    // allocator, one in-progress batch: two threads recording at once corrupt
    // the batch, and two flushing at once deadlock on the fence. That was
    // acceptable while only the pipeline worker ever touched the device.
    //
    // It stopped being acceptable when broadcast stages began running their
    // frames in parallel, because the scale-space detectors are CPU algorithms
    // that offload an inner loop through RunCtx::Gpu(). Nineteen threads
    // entered one command queue and the application froze.
    //
    // A CALLER-HELD LOCK rather than locking each method, deliberately. The
    // detectors do upload / dispatch / readback as one logical operation, and
    // per-method locks would let another thread interleave between them --
    // still correct for the queue, still wrong for the result. Holding it
    // across the whole offload is what makes it one operation.
    //
    // So the rule is: any code reaching the device from somewhere other than
    // the pipeline worker takes this first. See GpuLock below.
    std::mutex& SubmitMutex() { return m_submitMtx; }

private:
    // Opens the command list, reusing an in-progress batch rather than
    // discarding it. See the definition for why this matters.
    bool BeginRecording();

    // Reserves this dispatch's slice of the descriptor heap, flushing first
    // when the batch has used it up. See Dispatch.
    bool TakeHeapSlice(UINT* base, std::string* err);

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
    double                     m_gpuMs = 0.0;   // submit+wait, see GpuMs()
    uint64_t                   m_batchPixels = 0;
    static uint64_t            BatchPixelBudget();

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

    std::vector<ID3D12Resource*> m_staging;   // upload buffers in flight, pooled on flush

    // A reference to every buffer a recorded DispatchBuffers binds, dropped
    // once its batch has run. A dispatch is only RECORDED until the flush, so
    // a caller freeing a scratch buffer right after recording -- a network's
    // temporaries going out of scope -- would otherwise free memory the GPU
    // has yet to touch.
    std::vector<ID3D12Resource*> m_inflight;

    // Buffers handed back by RecycleBuffer, for AcquireBuffer. Own lock: a
    // tensor can be freed by a thread not holding the submit mutex.
    std::mutex                               m_poolMtx;
    struct PooledBuffer {
        uint64_t              bytes;
        ID3D12Resource*       res;
        D3D12_RESOURCE_STATES state;   // where it was left: its next bind transitions from here
    };
    std::vector<PooledBuffer>                m_bufferPool;

    // SharedKernel's cache. unique_ptr so a returned pointer stays valid as
    // the map grows.
    std::mutex                                                     m_kernelMtx;
    std::unordered_map<std::string, std::unique_ptr<ComputeKernel>> m_kernels;
    uint64_t                                 m_poolBytes = 0;

    // Idle staging buffers for reuse; see TakeStaging().
    struct PooledStaging {
        ID3D12Resource*  res;
        UINT64           size;
        D3D12_HEAP_TYPE  type;
    };
    std::vector<PooledStaging> m_stagingPool;
    ID3D12Resource* TakeStaging(D3D12_HEAP_TYPE type, UINT64 size);
    void            RecycleStaging(ID3D12Resource* r);
    ShaderCompiler               m_compiler;
    bool                         m_deviceLost = false;
    std::mutex           m_submitMtx;   // see SubmitMutex()
    bool                 m_ready      = false;
};

} // namespace tglab
