// Image: the primary data type flowing between algorithm ports.
//
// M1 is CPU-only. The Residency machinery is declared but only the Cpu bit is
// ever set; M3 adds the GPU side. The acquire/invalidate API is fixed now so
// that adding GPU residency later does not touch any algorithm.
#pragma once

#include <cstdint>
#include <memory>
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

// The GPU side of an Image. Defined in gpu/, so core/ only ever holds a
// pointer to it — that keeps the dependency pointing one way (gpu -> core).
struct GpuResidency;

// Frees a GpuResidency without core/ needing its definition.
struct GpuResidencyDeleter {
    void operator()(GpuResidency*) const noexcept;
};

class Image {
public:
    Image();
    explicit Image(const ImageDesc& d);
    ~Image();

    // Movable, not copyable — images are large and stage-owned.
    Image(Image&&) noexcept;
    Image& operator=(Image&&) noexcept;
    Image(const Image&)            = delete;
    Image& operator=(const Image&) = delete;

    void Alloc(const ImageDesc& d);
    void Reset();

    const ImageDesc& Desc() const { return m_desc; }
    bool Valid() const { return m_desc.Valid() && (HasCpu() || HasGpu()); }

    bool HasCpu() const { return (uint8_t(m_res) & uint8_t(Residency::Cpu)) != 0; }
    bool HasGpu() const { return (uint8_t(m_res) & uint8_t(Residency::Gpu)) != 0; }
    Residency GetResidency() const { return m_res; }

    // --- CPU access ---------------------------------------------------------
    // MapCpuRead() is the ONLY blocking sync point: when the image is
    // GPU-resident only, it must read back before returning valid pixels.
    // Callers that already know the image is CPU-side pay nothing.
    ImageView MapCpuRead();
    ImageView MapCpuWrite();   // invalidates the GPU copy

    // --- GPU access ---------------------------------------------------------
    // Implemented in gpu/gpu_image.cpp. Acquiring for write invalidates the
    // CPU copy; acquiring for read uploads only if the GPU copy is stale.
    // The rule "write clears the other side, read transfers only when unset"
    // is what keeps a chain of GPU stages free of intermediate transfers.
    GpuResidency* AcquireGpuRead(class ComputeContext& ctx);
    GpuResidency* AcquireGpuWrite(class ComputeContext& ctx);

    // Called by the GPU layer once it has populated the GPU copy.
    void MarkGpuResident();
    GpuResidency* RawGpu() const { return m_gpu.get(); }

    // Writable view of the CPU buffer WITHOUT touching residency. Used by the
    // readback path, which is filling the CPU cache rather than authoring new
    // pixels — MapCpuWrite() there would clear the GPU bit mid-readback and
    // make the result look stale.
    ImageView CpuBufferForFill();

    // Deep copy of whatever is CPU-side (does not clone GPU memory).
    Image Clone() const;

private:
    void EnsureCpuStorage();

    ImageDesc            m_desc{};
    std::vector<uint8_t> m_cpu;
    Residency            m_res = Residency::None;
    std::unique_ptr<GpuResidency, GpuResidencyDeleter> m_gpu;
};

} // namespace tglab
