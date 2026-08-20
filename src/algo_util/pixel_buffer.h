// PixelBuffer — an image unpacked to flat floats, and packed back afterwards.
//
// Every spatial filter needs the same two things: random access to neighbouring
// pixels without a per-access format branch, and a way to write the result back
// in whatever format the output port asked for. Doing that inline in each
// algorithm means repeating the format switch (and forgetting RGBA32F, as the
// first version of gaussian_blur did).
//
// Values are kept in the source's natural scale -- 0..255 for RGBA8, unchanged
// for the float formats -- so a filter that only averages neighbours needs no
// rescaling. Filters that reason about intensity differences (bilateral,
// Perona-Malik) care about that scale, so ValueScale() reports it.
#pragma once

#include <vector>

#include "../core/image.h"

namespace tglab {

class PixelBuffer {
public:
    // Unpacks `v`. Channels is 1 for R32F and 4 otherwise.
    void Unpack(const ImageView& v);

    // Writes back, clamping to the destination format's range.
    void PackInto(ImageView& dst) const;

    // Sized like Unpack(src) would produce, but zero-filled: for scratch
    // buffers that a filter fills itself.
    void AllocLike(const PixelBuffer& other);

    int Width()    const { return m_w; }
    int Height()   const { return m_h; }
    int Channels() const { return m_ch; }

    // 255 for RGBA8, 1 for the float formats. The yardstick for parameters
    // expressed as an intensity difference, so they mean the same thing
    // whichever format an image arrives in.
    float ValueScale() const { return m_scale; }

    float*       At(int x, int y)       { return &m_data[Index(x, y)]; }
    const float* At(int x, int y) const { return &m_data[Index(x, y)]; }

    // Edge-clamped access, which is what every windowed filter wants at the
    // border. Clamping (rather than zero-padding) keeps borders from darkening.
    const float* AtClamped(int x, int y) const;

    float        Get(int x, int y, int c) const { return m_data[Index(x, y) + size_t(c)]; }
    void         Set(int x, int y, int c, float v) { m_data[Index(x, y) + size_t(c)] = v; }

    std::vector<float>&       Data()       { return m_data; }
    const std::vector<float>& Data() const { return m_data; }

    bool Valid() const { return m_w > 0 && m_h > 0; }

private:
    size_t Index(int x, int y) const {
        return (size_t(y) * size_t(m_w) + size_t(x)) * size_t(m_ch);
    }

    int   m_w = 0, m_h = 0, m_ch = 0;
    float m_scale = 1.0f;
    Format m_format = Format::Unknown;
    std::vector<float> m_data;
};

} // namespace tglab
