#include "texture.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include "device.h"

namespace tglab {

void GpuTexture::Release() {
    // Frames already submitted may still be sampling this texture or copying
    // out of its staging buffers -- resizing one mid-flight is the common case,
    // since a dropped image rarely matches the slot's current dimensions.
    // Releasing now faults the GPU, and a removed device never signals its
    // fence, so the next BeginFrame() waits forever. Hand them to the device
    // instead; it frees them once the fence says the GPU is done.
    const bool deferrable = m_dev && m_dev->Ready();

    if (m_res) {
        // Srv().Free() tolerates being called after the heap is gone, but the
        // descriptor is only meaningful while the device lives.
        if (deferrable) {
            m_dev->Srv().Free(m_cpu, m_gpu);
            m_dev->DeferRelease(m_res);
        } else {
            m_res->Release();
        }
        m_res = nullptr;
    }
    for (int i = 0; i < kMaxFramesInFlight; ++i) {
        if (m_upload[i]) {
            if (deferrable) m_dev->DeferRelease(m_upload[i]);
            else            m_upload[i]->Release();
            m_upload[i] = nullptr;
        }
        m_uploadSize[i] = 0;
    }
    m_desc    = {};
    m_version = UINT64_MAX;
}

bool GpuTexture::Create(Device& dev, const ImageDesc& d) {
    Release();
    m_dev = &dev;

    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width            = UINT64(d.width);
    rd.Height           = UINT(d.height);
    rd.DepthOrArraySize = 1;
    rd.MipLevels        = 1;
    rd.Format           = DXGI_FORMAT_R8G8B8A8_UNORM;   // display format
    rd.SampleDesc.Count = 1;
    rd.Layout           = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    rd.Flags            = D3D12_RESOURCE_FLAG_NONE;

    if (FAILED(dev.Get()->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                                                  D3D12_RESOURCE_STATE_COPY_DEST,
                                                  nullptr, IID_PPV_ARGS(&m_res))))
        return false;

    D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Format                  = rd.Format;
    srv.ViewDimension           = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Texture2D.MipLevels     = 1;

    dev.Srv().Alloc(&m_cpu, &m_gpu);
    dev.Get()->CreateShaderResourceView(m_res, &srv, m_cpu);

    m_desc = d;
    m_freshlyCreated = true;
    return true;
}

bool GpuTexture::Update(Device& dev, Image& img, uint64_t contentVersion) {
    if (!img.Valid()) return false;
    const ImageDesc& d = img.Desc();

    const bool recreated = (!m_res || m_desc.width != d.width || m_desc.height != d.height);
    if (recreated) {
        if (!Create(dev, d)) return false;
    }

    // Skip the upload entirely when nothing changed — the common case once
    // the pipeline is idle, and what keeps a static frame free.
    if (!recreated && contentVersion == m_version) return true;
    m_version = contentVersion;

    // Convert to RGBA8 for display. Float formats are normalized so that
    // signed data (e.g. gradients) is visible rather than clipped to black.
    ImageView v = img.MapCpuRead();
    const int w = d.width, h = d.height;

    std::vector<uint8_t> rgba;
    const uint8_t* srcBytes = nullptr;

    if (d.format == Format::RGBA8) {
        srcBytes = v.data;
    } else if (d.format == Format::R32F) {
        float lo = 0.0f, hi = 1.0f;
        const float* p = reinterpret_cast<const float*>(v.data);
        lo = hi = p[0];
        for (int i = 1; i < w * h; ++i) { lo = std::min(lo, p[i]); hi = std::max(hi, p[i]); }
        const float range = (hi - lo) > 1e-8f ? (hi - lo) : 1.0f;

        rgba.resize(size_t(w) * size_t(h) * 4);
        for (int i = 0; i < w * h; ++i) {
            const uint8_t g = uint8_t(std::clamp((p[i] - lo) / range, 0.0f, 1.0f) * 255.0f);
            rgba[size_t(i) * 4 + 0] = g;
            rgba[size_t(i) * 4 + 1] = g;
            rgba[size_t(i) * 4 + 2] = g;
            rgba[size_t(i) * 4 + 3] = 255;
        }
        srcBytes = rgba.data();
    } else if (d.format == Format::RGBA32F) {
        const float* p = reinterpret_cast<const float*>(v.data);
        rgba.resize(size_t(w) * size_t(h) * 4);
        for (int i = 0; i < w * h * 4; ++i)
            rgba[size_t(i)] = uint8_t(std::clamp(p[i], 0.0f, 1.0f) * 255.0f);
        srcBytes = rgba.data();
    } else {
        return false;
    }

    // Upload through a staging buffer, respecting the 256-byte row alignment.
    const UINT rowPitch = UINT(w) * 4;
    const UINT aligned  = (rowPitch + D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1) &
                          ~(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1);
    const UINT64 uploadSize = UINT64(aligned) * UINT64(h);

    // Use this frame's slot. BeginFrame() already waited on its fence, so the
    // GPU is finished with whatever was here last time round.
    const UINT slot = dev.FrameSlot() % kMaxFramesInFlight;
    ID3D12Resource*& upload = m_upload[slot];
    if (upload && m_uploadSize[slot] < uploadSize) {
        upload->Release();
        upload = nullptr;
    }

    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_UPLOAD;

    D3D12_RESOURCE_DESC bd = {};
    bd.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    bd.Width            = uploadSize;
    bd.Height           = 1;
    bd.DepthOrArraySize = 1;
    bd.MipLevels        = 1;
    bd.Format           = DXGI_FORMAT_UNKNOWN;
    bd.SampleDesc.Count = 1;
    bd.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    if (!upload) {
        if (FAILED(dev.Get()->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd,
                                                      D3D12_RESOURCE_STATE_GENERIC_READ,
                                                      nullptr, IID_PPV_ARGS(&upload))))
            return false;
        m_uploadSize[slot] = uploadSize;
    }

    void* mapped = nullptr;
    D3D12_RANGE noRead = {0, 0};
    if (FAILED(upload->Map(0, &noRead, &mapped))) return false;
    for (int y = 0; y < h; ++y) {
        std::memcpy(static_cast<uint8_t*>(mapped) + size_t(y) * aligned,
                    srcBytes + size_t(y) * rowPitch, rowPitch);
    }
    upload->Unmap(0, nullptr);

    // Record the copy on the frame's command list. It runs before ImGui's
    // draw commands sample the texture, so no extra sync is needed.
    ID3D12GraphicsCommandList* cl = dev.CurrentCommandList();
    if (!cl) return false;

    D3D12_TEXTURE_COPY_LOCATION dst = {};
    dst.pResource        = m_res;
    dst.Type             = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.SubresourceIndex = 0;

    D3D12_TEXTURE_COPY_LOCATION src = {};
    src.pResource                          = upload;
    src.Type                               = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint.Footprint.Format   = DXGI_FORMAT_R8G8B8A8_UNORM;
    src.PlacedFootprint.Footprint.Width    = UINT(w);
    src.PlacedFootprint.Footprint.Height   = UINT(h);
    src.PlacedFootprint.Footprint.Depth    = 1;
    src.PlacedFootprint.Footprint.RowPitch = aligned;

    D3D12_RESOURCE_BARRIER toCopy = {};
    toCopy.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    toCopy.Transition.pResource   = m_res;
    toCopy.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    toCopy.Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    toCopy.Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_DEST;

    // A freshly created resource is already in COPY_DEST.
    if (!m_freshlyCreated) cl->ResourceBarrier(1, &toCopy);
    m_freshlyCreated = false;

    cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

    D3D12_RESOURCE_BARRIER toSrv = toCopy;
    toSrv.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    toSrv.Transition.StateAfter  = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    cl->ResourceBarrier(1, &toSrv);

    return true;
}

} // namespace tglab
