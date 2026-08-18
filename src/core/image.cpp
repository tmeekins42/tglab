#include "image.h"

#include <cassert>
#include <cstring>

namespace tglab {

// Set by the GPU layer at startup (gpu/gpu_image.cpp). A function pointer
// rather than a direct call so core/ does not depend on gpu/ — the dependency
// only ever points gpu -> core.
bool (*g_readbackFromGpu)(Image&) = nullptr;

static bool ReadbackFromGpu(Image& img) {
    return g_readbackFromGpu ? g_readbackFromGpu(img) : false;
}

int BytesPerPixel(Format f) {
    switch (f) {
        case Format::RGBA8:   return 4;
        case Format::R32F:    return 4;
        case Format::RGBA32F: return 16;
        case Format::Unknown: return 0;
    }
    return 0;
}

const char* FormatName(Format f) {
    switch (f) {
        case Format::RGBA8:   return "RGBA8";
        case Format::R32F:    return "R32F";
        case Format::RGBA32F: return "RGBA32F";
        case Format::Unknown: return "Unknown";
    }
    return "Unknown";
}

size_t ImageDesc::SizeInBytes() const {
    return size_t(width) * size_t(height) * size_t(BytesPerPixel(format));
}

// Out-of-line so GpuResidency stays incomplete in the header.
Image::Image() = default;
Image::Image(const ImageDesc& d) { Alloc(d); }
Image::~Image() = default;
Image::Image(Image&&) noexcept = default;
Image& Image::operator=(Image&&) noexcept = default;

void Image::EnsureCpuStorage() {
    if (m_desc.Valid() && m_cpu.size() != m_desc.SizeInBytes())
        m_cpu.assign(m_desc.SizeInBytes(), 0);
}

void Image::Alloc(const ImageDesc& d) {
    m_desc = d;
    m_gpu.reset();
    if (!d.Valid()) {
        m_cpu.clear();
        m_res = Residency::None;
        return;
    }
    m_cpu.assign(d.SizeInBytes(), 0);
    m_res = Residency::Cpu;
}

void Image::Reset() {
    m_desc = {};
    m_cpu.clear();
    m_gpu.reset();
    m_res = Residency::None;
}

ImageView Image::MapCpuRead() {
    // The one blocking sync point. When the pixels live only on the GPU, this
    // is where they come back — which is why displaying a GPU result should go
    // through the SRV instead of calling this.
    if (!HasCpu() && HasGpu()) {
        EnsureCpuStorage();
        if (ReadbackFromGpu(*this)) {
            m_res = Residency::Both;
        } else {
            // Returning the uninitialised buffer here would look like a
            // correct-but-wrong image, which is far harder to diagnose than a
            // crash. This only fires if the GPU layer failed to install its
            // hook, which is a programming error rather than a runtime one.
            assert(false && "GPU-resident image could not be read back — "
                            "was the GPU residency hook installed?");
        }
    }

    ImageView v;
    v.data = m_cpu.empty() ? nullptr : m_cpu.data();
    v.desc = m_desc;
    return v;
}

ImageView Image::MapCpuWrite() {
    // Writing on the CPU invalidates any GPU copy.
    EnsureCpuStorage();
    m_res = Residency::Cpu;

    ImageView v;
    v.data = m_cpu.empty() ? nullptr : m_cpu.data();
    v.desc = m_desc;
    return v;
}

void Image::MarkGpuResident() {
    // The GPU now holds the authoritative pixels; the CPU copy is stale.
    m_res = Residency::Gpu;
}

ImageView Image::CpuBufferForFill() {
    EnsureCpuStorage();
    ImageView v;
    v.data = m_cpu.empty() ? nullptr : m_cpu.data();
    v.desc = m_desc;
    return v;
}

Image Image::Clone() const {
    // A GPU-only image has no CPU pixels to copy, so pull them back first —
    // otherwise Clone() would silently hand back a blank image. Const-cast is
    // safe here: the readback only populates the cache, it does not change
    // what the image represents.
    if (!HasCpu() && HasGpu()) const_cast<Image*>(this)->MapCpuRead();

    Image out;
    out.m_desc = m_desc;
    out.m_cpu  = m_cpu;
    // Only the CPU side is cloned. A clone that shared the GPU resource would
    // alias it; one that copied it would cost VRAM for a copy usually made
    // just to hand pixels to the UI.
    out.m_res = HasCpu() ? Residency::Cpu : Residency::None;
    return out;
}

} // namespace tglab
