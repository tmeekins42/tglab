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

private:
    bool Flush(std::string* err);   // execute + wait

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

    std::vector<ID3D12Resource*> m_staging;   // upload/readback buffers, freed on flush
    ShaderCompiler               m_compiler;
    bool                         m_deviceLost = false;
};

} // namespace tglab
