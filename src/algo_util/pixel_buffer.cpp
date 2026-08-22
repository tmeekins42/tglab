#include "pixel_buffer.h"

#include <algorithm>

namespace tglab {

void PixelBuffer::Unpack(const ImageView& v) {
    if (!v.Valid()) {
        m_w = m_h = m_ch = 0;
        m_data.clear();
        return;
    }

    m_w      = v.desc.width;
    m_h      = v.desc.height;
    m_format = v.desc.format;
    m_ch     = (m_format == Format::R32F) ? 1 : 4;
    m_scale  = (m_format == Format::RGBA8) ? 255.0f : 1.0f;

    m_data.assign(size_t(m_w) * size_t(m_h) * size_t(m_ch), 0.0f);

    // One switch here rather than one per pixel access later.
    switch (m_format) {
        case Format::R32F:
            for (int y = 0; y < m_h; ++y)
                for (int x = 0; x < m_w; ++x)
                    m_data[Index(x, y)] = *v.At<float>(x, y);
            break;

        case Format::RGBA32F:
            for (int y = 0; y < m_h; ++y)
                for (int x = 0; x < m_w; ++x) {
                    const float* p = v.At<float>(x, y);
                    for (int c = 0; c < 4; ++c) m_data[Index(x, y) + size_t(c)] = p[c];
                }
            break;

        case Format::RGBA16F:
            for (int y = 0; y < m_h; ++y)
                for (int x = 0; x < m_w; ++x) {
                    const uint16_t* p = v.At<uint16_t>(x, y);
                    for (int c = 0; c < 4; ++c)
                        m_data[Index(x, y) + size_t(c)] = HalfToFloat(p[c]);
                }
            break;

        default:   // RGBA8
            for (int y = 0; y < m_h; ++y)
                for (int x = 0; x < m_w; ++x) {
                    const uint8_t* p = v.At<uint8_t>(x, y);
                    for (int c = 0; c < 4; ++c)
                        m_data[Index(x, y) + size_t(c)] = float(p[c]);
                }
            break;
    }
}

void PixelBuffer::PackInto(ImageView& dst) const {
    if (!dst.Valid() || !Valid()) return;

    const int w = std::min(m_w, dst.desc.width);
    const int h = std::min(m_h, dst.desc.height);

    switch (dst.desc.format) {
        case Format::R32F:
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x)
                    *dst.At<float>(x, y) = m_data[Index(x, y)];
            break;

        case Format::RGBA32F:
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x) {
                    float* p = dst.At<float>(x, y);
                    for (int c = 0; c < 4; ++c)
                        p[c] = m_ch == 1 ? m_data[Index(x, y)]
                                         : m_data[Index(x, y) + size_t(c)];
                }
            break;

        case Format::RGBA16F:
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x) {
                    uint16_t* p = dst.At<uint16_t>(x, y);
                    for (int c = 0; c < 4; ++c) {
                        const float s = m_ch == 1 ? m_data[Index(x, y)]
                                                  : m_data[Index(x, y) + size_t(c)];
                        p[c] = FloatToHalf(s);
                    }
                    // A 1-channel buffer carries no alpha; 1.0 keeps the image
                    // opaque rather than fully transparent.
                    if (m_ch == 1) p[3] = FloatToHalf(1.0f);
                }
            break;

        default: {  // RGBA8
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x) {
                    uint8_t* p = dst.At<uint8_t>(x, y);
                    for (int c = 0; c < 4; ++c) {
                        const float s = m_ch == 1 ? m_data[Index(x, y)]
                                                  : m_data[Index(x, y) + size_t(c)];
                        p[c] = uint8_t(std::clamp(s, 0.0f, 255.0f));
                    }
                    // A filter working on a 1-channel buffer has no alpha to
                    // carry, so keep the image opaque rather than transparent.
                    if (m_ch == 1) p[3] = 255;
                }
            break;
        }
    }
}

void PixelBuffer::AllocLike(const PixelBuffer& other) {
    m_w      = other.m_w;
    m_h      = other.m_h;
    m_ch     = other.m_ch;
    m_scale  = other.m_scale;
    m_format = other.m_format;
    m_data.assign(other.m_data.size(), 0.0f);
}

const float* PixelBuffer::AtClamped(int x, int y) const {
    x = std::clamp(x, 0, m_w - 1);
    y = std::clamp(y, 0, m_h - 1);
    return &m_data[Index(x, y)];
}

} // namespace tglab
