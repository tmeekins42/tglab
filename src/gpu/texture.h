// GpuTexture — an app-owned D3D12 texture displayed via ImGui::Image().
//
// For the DX12 backend ImTextureID *is* a D3D12_GPU_DESCRIPTOR_HANDLE, so
// display costs one SRV and no copy back to the CPU. In M3 an algorithm's
// GPU output feeds this directly with zero readback.
#pragma once

#include <d3d12.h>

#include "../core/image.h"
#include "descriptor_heap.h"

namespace tglab {

class Device;

// Upper bound for the per-frame staging buffers below; must be >= the
// device's kNumFramesInFlight.
inline constexpr int kMaxFramesInFlight = 4;

class GpuTexture {
public:
    ~GpuTexture() { Release(); }

    // Uploads CPU pixels, recreating the resource if the size/format changed.
    // Converts R32F/RGBA32F to RGBA8 for display. Re-uploads only when
    // `contentVersion` differs from the last call.
    bool Update(Device& dev, Image& img, uint64_t contentVersion);
    void Release();

    bool Valid() const { return m_res != nullptr; }
    D3D12_GPU_DESCRIPTOR_HANDLE Handle() const { return m_gpu; }

    int Width()  const { return m_desc.width; }
    int Height() const { return m_desc.height; }

private:
    bool Create(Device& dev, const ImageDesc& d);

    Device*                     m_dev = nullptr;
    ID3D12Resource*             m_res = nullptr;
    D3D12_CPU_DESCRIPTOR_HANDLE m_cpu{};
    D3D12_GPU_DESCRIPTOR_HANDLE m_gpu{};
    ImageDesc                   m_desc{};
    bool                        m_freshlyCreated = false;   // starts in COPY_DEST

    // One staging buffer per frame-in-flight. Reusing the slot for frame N
    // is safe because BeginFrame() has already waited on that frame's fence;
    // a single shared buffer would be freed while the GPU still reads it.
    ID3D12Resource* m_upload[kMaxFramesInFlight] = {};
    UINT64          m_uploadSize[kMaxFramesInFlight] = {};
    uint64_t        m_version = UINT64_MAX;   // last uploaded content version
};

} // namespace tglab
