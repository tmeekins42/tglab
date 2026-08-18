// SrvHeap — free-list allocator over one shader-visible CBV/SRV/UAV heap.
//
// Required by ImGui_ImplDX12_InitInfo (SrvDescriptorAllocFn/FreeFn). The same
// heap mints SRVs for displayed images, so it must be sized for the font atlas
// plus every live view texture.
#pragma once

#include <d3d12.h>

#include <vector>

namespace tglab {

class SrvHeap {
public:
    bool Init(ID3D12Device* device, uint32_t capacity);
    void Shutdown();

    void Alloc(D3D12_CPU_DESCRIPTOR_HANDLE* outCpu, D3D12_GPU_DESCRIPTOR_HANDLE* outGpu);
    void Free(D3D12_CPU_DESCRIPTOR_HANDLE cpu, D3D12_GPU_DESCRIPTOR_HANDLE gpu);

    ID3D12DescriptorHeap* Heap() const { return m_heap; }

private:
    ID3D12DescriptorHeap*       m_heap = nullptr;
    D3D12_CPU_DESCRIPTOR_HANDLE m_cpuStart{};
    D3D12_GPU_DESCRIPTOR_HANDLE m_gpuStart{};
    UINT                        m_stride   = 0;
    uint32_t                    m_capacity = 0;
    std::vector<uint32_t>       m_free;      // free indices, LIFO
};

} // namespace tglab
