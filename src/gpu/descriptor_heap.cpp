#include "descriptor_heap.h"

#include <cassert>

namespace tglab {

bool SrvHeap::Init(ID3D12Device* device, uint32_t capacity) {
    D3D12_DESCRIPTOR_HEAP_DESC desc = {};
    desc.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    desc.NumDescriptors = capacity;
    desc.Flags          = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&m_heap)))) return false;

    m_capacity = capacity;
    m_stride   = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    m_cpuStart = m_heap->GetCPUDescriptorHandleForHeapStart();
    m_gpuStart = m_heap->GetGPUDescriptorHandleForHeapStart();

    m_free.reserve(capacity);
    for (uint32_t i = 0; i < capacity; ++i) m_free.push_back(capacity - 1 - i);
    return true;
}

void SrvHeap::Shutdown() {
    if (m_heap) { m_heap->Release(); m_heap = nullptr; }
    m_free.clear();
    m_capacity = 0;
}

void SrvHeap::Alloc(D3D12_CPU_DESCRIPTOR_HANDLE* outCpu, D3D12_GPU_DESCRIPTOR_HANDLE* outGpu) {
    assert(!m_free.empty() && "SRV heap exhausted — raise the capacity");
    const uint32_t idx = m_free.back();
    m_free.pop_back();
    outCpu->ptr = m_cpuStart.ptr + SIZE_T(idx) * m_stride;
    outGpu->ptr = m_gpuStart.ptr + UINT64(idx) * m_stride;
}

void SrvHeap::Free(D3D12_CPU_DESCRIPTOR_HANDLE cpu, D3D12_GPU_DESCRIPTOR_HANDLE gpu) {
    (void)gpu;
    const uint32_t idx = uint32_t((cpu.ptr - m_cpuStart.ptr) / m_stride);
    assert(idx < m_capacity);
    m_free.push_back(idx);
}

} // namespace tglab
