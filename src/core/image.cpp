#include "image.h"

#include <cassert>
#include <cstring>

namespace tglab {

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

void Image::Alloc(const ImageDesc& d) {
    m_desc = d;
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
    m_res = Residency::None;
}

ImageView Image::MapCpuRead() {
    // M3: if only GPU-resident, this is where the readback fence is awaited.
    ImageView v;
    v.data = m_cpu.empty() ? nullptr : m_cpu.data();
    v.desc = m_desc;
    return v;
}

ImageView Image::MapCpuWrite() {
    // Writing on the CPU invalidates any GPU copy (M3).
    m_res = Residency::Cpu;
    ImageView v;
    v.data = m_cpu.empty() ? nullptr : m_cpu.data();
    v.desc = m_desc;
    return v;
}

Image Image::Clone() const {
    Image out;
    out.m_desc = m_desc;
    out.m_cpu  = m_cpu;
    out.m_res  = m_res;
    return out;
}

} // namespace tglab
