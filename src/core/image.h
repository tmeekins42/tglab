// Image: the primary data type flowing between algorithm ports.
//
// M1 is CPU-only. The Residency machinery is declared but only the Cpu bit is
// ever set; M3 adds the GPU side. The acquire/invalidate API is fixed now so
// that adding GPU residency later does not touch any algorithm.
#pragma once

#include <cstdint>
#include <vector>

namespace tglab {

enum class Format : uint8_t {
    Unknown = 0,
    RGBA8,      // 4 x uint8  — loading, display
    R32F,       // 1 x float  — intermediates, signed gradients
    RGBA32F,    // 4 x float  — high-precision intermediates
};

int BytesPerPixel(Format f);
const char* FormatName(Format f);

// Port format constraints. Algorithms declare what they accept/produce.
enum class FormatSpec : uint8_t {
    Any,            // accepts whatever it is given
    SameAsInput,    // output matches input 0's format
    RGBA8,
    R32F,
    RGBA32F,
};

struct ImageDesc {
    int    width  = 0;
    int    height = 0;
    Format format = Format::Unknown;

    bool operator==(const ImageDesc&) const = default;
    size_t SizeInBytes() const;
    bool   Valid() const { return width > 0 && height > 0 && format != Format::Unknown; }
};

// Non-owning view of CPU pixels. Algorithms operate through this.
struct ImageView {
    uint8_t*  data   = nullptr;
    ImageDesc desc{};

    bool Valid() const { return data != nullptr && desc.Valid(); }
    int  Pitch() const { return desc.width * BytesPerPixel(desc.format); }

    template <class T> T*       At(int x, int y)       { return reinterpret_cast<T*>(data + size_t(y) * Pitch() + size_t(x) * BytesPerPixel(desc.format)); }
    template <class T> const T* At(int x, int y) const { return reinterpret_cast<const T*>(data + size_t(y) * Pitch() + size_t(x) * BytesPerPixel(desc.format)); }
};

enum class Residency : uint8_t { None = 0, Cpu = 1, Gpu = 2, Both = 3 };

class Image {
public:
    Image() = default;
    explicit Image(const ImageDesc& d) { Alloc(d); }

    // Movable, not copyable — images are large and stage-owned.
    Image(Image&&) noexcept            = default;
    Image& operator=(Image&&) noexcept = default;
    Image(const Image&)                = delete;
    Image& operator=(const Image&)     = delete;

    void Alloc(const ImageDesc& d);
    void Reset();

    const ImageDesc& Desc() const { return m_desc; }
    bool Valid() const { return m_desc.Valid() && !m_cpu.empty(); }

    // CPU access. In M3 MapCpuRead() becomes the one blocking sync point
    // (it must wait on the GPU readback fence when only GPU-resident).
    ImageView MapCpuRead();
    ImageView MapCpuWrite();   // invalidates the GPU copy

    // Deep copy — explicit, since Image is move-only.
    Image Clone() const;

private:
    ImageDesc            m_desc{};
    std::vector<uint8_t> m_cpu;
    Residency            m_res = Residency::None;
};

} // namespace tglab
