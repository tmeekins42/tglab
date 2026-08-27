// GpuTexture — an app-owned D3D12 texture displayed via ImGui::Image().
//
// For the DX12 backend ImTextureID *is* a D3D12_GPU_DESCRIPTOR_HANDLE, so
// display costs one SRV and no copy back to the CPU. In M3 an algorithm's
// GPU output feeds this directly with zero readback.
#pragma once

#include <d3d12.h>

#include "../core/image.h"
#include "descriptor_heap.h"
#include "device.h"

namespace tglab {

// One staging buffer per frame-in-flight slot, indexed by Device::FrameSlot().
// Tied to the device's count rather than merely ">=" it: the slot index is what
// makes "BeginFrame() already waited on this slot's fence" true, and that
// reasoning only holds if the two cycle together.
inline constexpr int kMaxFramesInFlight = kNumFramesInFlight;

class GpuTexture {
public:
    GpuTexture() = default;
    ~GpuTexture() { Release(); }

    // Owns an ID3D12Resource and an SRV descriptor slot, so copying it is
    // always a bug: both objects would release the same resource and free the
    // same descriptor.
    //
    // It was implicitly copyable, and PaletteEntry holds one BY VALUE inside a
    // std::vector. Adding an entry reallocated that vector, which copied every
    // existing texture to new storage and then destroyed the originals --
    // releasing resources the copies still pointed at. The freed descriptor
    // slot was immediately handed to the new entry's thumbnail, so two palette
    // rows drew the same SRV (Tim saw "test" showing a newly dropped photo's
    // thumbnail), and the next Update() on the stale one dereferenced a
    // released resource and took the app down.
    //
    // Deleted rather than defined: there is no sane copy of a GPU resource
    // handle, and a vector only needs the move.
    GpuTexture(const GpuTexture&)            = delete;
    GpuTexture& operator=(const GpuTexture&) = delete;

    GpuTexture(GpuTexture&& o) noexcept { MoveFrom(o); }
    GpuTexture& operator=(GpuTexture&& o) noexcept {
        if (this != &o) { Release(); MoveFrom(o); }
        return *this;
    }

    // Uploads CPU pixels, recreating the resource if the size/format changed.
    // Converts R32F/RGBA32F to RGBA8 for display. Re-uploads only when
    // `contentVersion` differs from the last call.
    bool Update(Device& dev, Image& img, uint64_t contentVersion);

    // Reads a small rectangle of a GPU-resident result back to the CPU.
    //
    // For the 1:1 loupe, which point-samples individual pixels and prints their
    // values -- it genuinely needs the numbers, not a magnified picture. The
    // rectangle is tens of pixels across, so this is a few kilobytes rather
    // than the whole image, and it only happens while the loupe is switched on.
    //
    // Blocking: it waits for the copy. Acceptable for a deliberate, hovering
    // inspection gesture; not something to put on the per-frame path.
    //
    // `outPixels` receives width*height*BytesPerPixel bytes in the source's own
    // format. Returns false if the rectangle is empty or the copy fails.
    static bool ReadRegion(Device& dev, const SharedGpuTexture& src,
                           int x, int y, int w, int h,
                           std::vector<uint8_t>* outPixels);

    // Converts straight from a result that is still on the GPU, with no
    // readback and no CPU conversion.
    //
    // The alternative -- and what this replaces -- was to map the pixels, walk
    // them on the CPU into RGBA8, and upload the result back. At 21 MP that was
    // ~86 ms of readback and ~47 ms of conversion per viewer per frame, sending
    // the image across the bus twice to draw it once.
    //
    // The compute context and ImGui share one device and compute images carry
    // ALLOW_SIMULTANEOUS_ACCESS, so the UI's direct queue can sample the
    // worker's texture while the compute queue owns it.
    bool UpdateFromGpu(Device& dev, const SharedGpuTexture& src,
                       uint64_t contentVersion);
    void Release();

    // Which out-of-range conditions the viewers highlight, as a bit mask.
    //
    // A global rather than a per-viewer setting or a parameter threaded through
    // UpdateFromGpu: it is one display preference, set from a menu, and every
    // viewer should agree. Threading it through the call chain would touch four
    // layers to express something that has exactly one value.
    //
    // Judged on the LINEAR values, before the tone curve. After the curve
    // everything is inside 0..1 by construction -- the shoulder maps 4.0 linear
    // to 0.91 -- so a blown highlight would look merely bright and the
    // condition would be untestable.
    enum ClipOverlay : uint32_t {
        kOverlayNone       = 0,
        kOverlayWhite      = 1,   // at or above the white level: detail gone
        kOverlayBlack      = 2,   // at or below zero: crushed
        kOverlayOutOfGamut = 4,   // a negative channel: real colour sRGB cannot
                                  // represent, recoverable by desaturating
    };
    static void     SetClipOverlay(uint32_t mask);
    static uint32_t GetClipOverlay();

    bool Valid() const { return m_res != nullptr; }
    D3D12_GPU_DESCRIPTOR_HANDLE Handle() const { return m_gpu; }

    int Width()  const { return m_desc.width; }
    int Height() const { return m_desc.height; }

private:
    bool Create(Device& dev, const ImageDesc& d);

    // Transfers every owning handle and leaves `o` empty, so its destructor
    // releases nothing. Every raw pointer here is an owner -- the resource, the
    // descriptor slot, and each per-frame staging buffer -- so missing one
    // would leak rather than crash, which is harder to notice.
    void MoveFrom(GpuTexture& o) noexcept {
        m_dev            = o.m_dev;
        m_res            = o.m_res;
        m_cpu            = o.m_cpu;
        m_gpu            = o.m_gpu;
        m_desc           = o.m_desc;
        m_freshlyCreated = o.m_freshlyCreated;
        m_version        = o.m_version;
        m_overlay        = o.m_overlay;
        for (int i = 0; i < kMaxFramesInFlight; ++i) {
            m_upload[i]     = o.m_upload[i];
            m_uploadSize[i] = o.m_uploadSize[i];
            o.m_upload[i]     = nullptr;
            o.m_uploadSize[i] = 0;
        }
        o.m_dev = nullptr;
        o.m_res = nullptr;
        o.m_cpu = {};
        o.m_gpu = {};
        o.m_desc = {};
        o.m_version = UINT64_MAX;
        o.m_overlay = UINT32_MAX;
    }

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

    // The overlay mask the cached conversion was produced with.
    //
    // Part of the cache key, not just state. The conversion is skipped when the
    // content version is unchanged -- which is what keeps a static frame free
    // -- but the overlay changes what the shader DRAWS from identical content.
    // Without this, toggling an overlay did nothing until something else
    // happened to bump the version, so it appeared to work only when a slider
    // was touched.
    uint32_t        m_overlay = UINT32_MAX;
};

// Frees the shared display-conversion pipeline. One shader and one descriptor
// heap serve every view, so they outlive any single GpuTexture; call this once
// at shutdown, before the device goes away.
void ReleaseDisplayPipeline();

} // namespace tglab
