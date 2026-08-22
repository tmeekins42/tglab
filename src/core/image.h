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
    R32F,       // 1 x float  — intermediates, signed gradients, Bayer mosaics
    RGBA32F,    // 4 x float  — high-precision intermediates
    RGBA16F,    // 4 x half   — the working format for raw
};

// Colour filter array layout, naming the 2x2 tile's top-left quad.
//
// A Bayer mosaic is deliberately NOT a new Data type: it is a single-channel
// 2D array of samples, which R32F already describes. What it needs beyond that
// is metadata saying which colour each pixel carries -- a property of the
// image, not a new kind of data. Keeping it an Image means PixelBuffer, the
// stage cache, FormatSpec and the GPU path all handle it unchanged, and a
// "passthrough" demosaic is simply the identity function.
enum class CfaPattern : uint8_t {
    None = 0,   // an ordinary image; every existing algorithm sees this
    RGGB,
    BGGR,
    GRBG,
    GBRG,
    XTrans,     // Fuji's 6x6; the slot is reserved, the algorithms are not written
};

const char* CfaPatternName(CfaPattern p);

// IEEE 754 half conversion.
//
// RGBA16F is the working format for raw: it halves the bandwidth of RGBA32F,
// which matters because chained GPU stages are bandwidth-bound, and its 11 bits
// of mantissa still comfortably exceed a 14-bit sensor's usable range. There is
// no native half type in C++, so pixels are stored as uint16_t and converted at
// the edges.
uint16_t FloatToHalf(float f);
float    HalfToFloat(uint16_t h);

// The colour of one mosaic sample. Index 0=R, 1=G, 2=B, matching RGB order.
int CfaColorAt(CfaPattern p, int x, int y);

int BytesPerPixel(Format f);
const char* FormatName(Format f);

// Port format constraints. Algorithms declare what they accept/produce.
enum class FormatSpec : uint8_t {
    Any,            // accepts whatever it is given
    SameAsInput,    // output matches input 0's format
    RGBA8,
    R32F,
    RGBA32F,
    RGBA16F,
};

struct ImageDesc {
    int    width  = 0;
    int    height = 0;
    Format format = Format::Unknown;

    // Sensor metadata, meaningful only for an undemosaiced mosaic. Declared
    // after `format` so the twenty-odd existing {w, h, Format::X} brace
    // initialisations keep working and simply take the defaults.
    //
    // These participate in operator==, which drives the stage cache. That is
    // correct: a mosaic and an RGB image of the same size are genuinely
    // different, and a cached result for one must not be reused for the other.
    CfaPattern cfa        = CfaPattern::None;
    float      blackLevel = 0.0f;   // sensor value that means black
    float      whiteLevel = 1.0f;   // ... and saturation

    // As-shot white balance, as per-channel gains normalised so green is 1.
    //
    // A sensor's green photosites are roughly twice as sensitive as its red and
    // blue ones, so raw data is heavily green without these -- which is exactly
    // what an undemosaiced image looks like if you forget them.
    float camMul[3] = {1.0f, 1.0f, 1.0f};

    // Camera colour space -> sRGB, row-major. Sensor primaries are not sRGB
    // primaries, so without this even correctly balanced data has the wrong
    // hues. Identity for a non-raw image.
    float rgbCam[9] = {1.0f, 0.0f, 0.0f,
                       0.0f, 1.0f, 0.0f,
                       0.0f, 0.0f, 1.0f};

    bool operator==(const ImageDesc&) const = default;
    size_t SizeInBytes() const;
    bool   Valid() const { return width > 0 && height > 0 && format != Format::Unknown; }

    // True when this image still carries an undemosaiced sensor pattern.
    bool IsMosaic() const { return cfa != CfaPattern::None; }
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
