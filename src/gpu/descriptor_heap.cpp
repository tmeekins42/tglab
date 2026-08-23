#include "descriptor_heap.h"

#include <algorithm>
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
    // Cleared so a late Free() is detected rather than computing an index from
    // a stale base pointer.
    m_cpuStart = {};
    m_gpuStart = {};
    m_stride   = 0;
}


bool SrvHeap::Reserve(uint32_t count, D3D12_CPU_DESCRIPTOR_HANDLE* outCpu,
                      D3D12_GPU_DESCRIPTOR_HANDLE* outGpu) {
    if (count == 0 || count > m_capacity) return false;
    if (m_free.size() < count) return false;

    // Find a run of `count` consecutive free indices.
    //
    // Deliberately NOT "the block must start at 0": ImGui allocates its font
    // atlas descriptor during init, so index 0 is already gone by the time the
    // display pipeline asks. An earlier version required [0, count) and simply
    // returned false, which made EnsureDisplayPipeline fail and the viewer fall
    // back to the CPU path -- silently, since the fallback is a legitimate
    // route for CPU-only stages. Fail loudly or not at all.
    std::vector<uint32_t> sorted(m_free.begin(), m_free.end());
    std::sort(sorted.begin(), sorted.end());

    uint32_t runStart = 0, runLen = 0;
    for (size_t i = 0; i < sorted.size(); ++i) {
        if (runLen && sorted[i] == runStart + runLen) {
            ++runLen;
        } else {
            runStart = sorted[i];
            runLen   = 1;
        }
        if (runLen == count) break;
    }
    if (runLen < count) return false;

    // Remove the reserved indices from the free list.
    const uint32_t lo = runStart, hi = runStart + count;
    m_free.erase(std::remove_if(m_free.begin(), m_free.end(),
                                [lo, hi](uint32_t v) { return v >= lo && v < hi; }),
                 m_free.end());

    outCpu->ptr = m_cpuStart.ptr + SIZE_T(runStart) * m_stride;
    outGpu->ptr = m_gpuStart.ptr + UINT64(runStart) * m_stride;
    return true;
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

    // Freeing after Shutdown() is a lifetime bug in the caller (a GpuTexture
    // outliving the device), but it must not corrupt the free list or crash on
    // the way out. Ignore it here and let the assert flag it in debug builds.
    if (!m_heap || m_stride == 0) return;
    if (cpu.ptr < m_cpuStart.ptr) {
        assert(false && "SRV freed against a different/destroyed heap — "
                        "release GpuTextures before Device::Shutdown()");
        return;
    }

    const uint32_t idx = uint32_t((cpu.ptr - m_cpuStart.ptr) / m_stride);
    if (idx >= m_capacity) {
        assert(false && "SRV index out of range — descriptor freed twice, or "
                        "after the heap was destroyed");
        return;
    }
    m_free.push_back(idx);
}

} // namespace tglab
