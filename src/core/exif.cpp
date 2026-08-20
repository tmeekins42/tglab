#include "exif.h"
#include <algorithm>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace tglab {
namespace {

// Every accessor is bounds-checked. This parses whatever file the user dropped
// in, so a truncated or hostile header must yield an empty result rather than
// read past the buffer.
class Reader {
public:
    Reader(const uint8_t* data, size_t size, bool bigEndian)
        : m_data(data), m_size(size), m_big(bigEndian) {}

    bool U16(size_t off, uint16_t* out) const {
        if (off + 2 > m_size) return false;
        *out = m_big ? uint16_t((m_data[off] << 8) | m_data[off + 1])
                     : uint16_t((m_data[off + 1] << 8) | m_data[off]);
        return true;
    }

    bool U32(size_t off, uint32_t* out) const {
        if (off + 4 > m_size) return false;
        *out = m_big ? (uint32_t(m_data[off]) << 24 | uint32_t(m_data[off + 1]) << 16 |
                        uint32_t(m_data[off + 2]) << 8 | uint32_t(m_data[off + 3]))
                     : (uint32_t(m_data[off + 3]) << 24 | uint32_t(m_data[off + 2]) << 16 |
                        uint32_t(m_data[off + 1]) << 8 | uint32_t(m_data[off]));
        return true;
    }

    // ASCII tag value, trimmed of trailing NULs and spaces.
    std::string Str(size_t off, uint32_t count) const {
        if (off >= m_size) return {};
        const size_t n = std::min(size_t(count), m_size - off);
        std::string s(reinterpret_cast<const char*>(m_data + off), n);
        while (!s.empty() && (s.back() == '\0' || s.back() == ' ')) s.pop_back();
        return s;
    }

    size_t Size() const { return m_size; }

private:
    const uint8_t* m_data;
    size_t         m_size;
    bool           m_big;
};

// EXIF tag numbers used below. Named rather than inline so the IFD walk reads
// as a list of what is wanted, not a list of magic numbers.
enum : uint16_t {
    kMake         = 0x010F,
    kModel        = 0x0110,
    kDateTime     = 0x0132,
    kExposureTime = 0x829A,
    kFNumber      = 0x829D,
    kExifIfd      = 0x8769,
    kIsoSpeed     = 0x8827,
    kDateTimeOrig = 0x9003,
    kFocalLength  = 0x920A,
    kLensModel    = 0xA434,
};

std::string FormatExposure(double seconds) {
    char buf[64];
    if (seconds <= 0.0) return {};
    // Fast shutter speeds read naturally as a fraction, slow ones as decimals --
    // the convention every camera and photo tool uses.
    if (seconds < 1.0) std::snprintf(buf, sizeof buf, "1/%.0f s", 1.0 / seconds);
    else               std::snprintf(buf, sizeof buf, "%.1f s", seconds);
    return buf;
}

} // namespace

ExifData ReadExif(const std::string& path) {
    ExifData out;

    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return out;

    // Only the head of the file is needed: APP1 must appear near the start, and
    // reading a whole 40 MB scan to find a 64 KB block would be wasteful.
    std::vector<uint8_t> buf(128 * 1024);
    const size_t got = std::fread(buf.data(), 1, buf.size(), f);
    std::fclose(f);
    buf.resize(got);
    if (got < 4) return out;

    if (buf[0] != 0xFF || buf[1] != 0xD8) return out;   // not a JPEG

    // Walk JPEG markers looking for APP1 with an "Exif\0\0" signature.
    size_t p = 2;
    size_t tiff = 0;
    size_t tiffLen = 0;
    while (p + 4 <= buf.size()) {
        if (buf[p] != 0xFF) break;
        const uint8_t marker = buf[p + 1];
        if (marker == 0xD8 || marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7)) {
            p += 2;
            continue;
        }
        if (marker == 0xDA || marker == 0xD9) break;   // image data begins

        const size_t segLen = size_t(buf[p + 2]) << 8 | buf[p + 3];
        if (segLen < 2 || p + 2 + segLen > buf.size()) break;

        if (marker == 0xE1 && segLen >= 8 &&
            std::memcmp(&buf[p + 4], "Exif\0\0", 6) == 0) {
            tiff    = p + 10;                 // TIFF header starts after the sig
            tiffLen = segLen - 8;
            break;
        }
        p += 2 + segLen;
    }
    if (!tiff || tiff + 8 > buf.size()) return out;

    // TIFF header: byte order, magic 42, offset to IFD0. Offsets inside are
    // relative to the TIFF header, not the file.
    const bool big = (buf[tiff] == 'M' && buf[tiff + 1] == 'M');
    if (!big && !(buf[tiff] == 'I' && buf[tiff + 1] == 'I')) return out;

    const size_t avail = std::min(tiffLen, buf.size() - tiff);
    Reader r(buf.data() + tiff, avail, big);

    uint16_t magic = 0;
    if (!r.U16(2, &magic) || magic != 42) return out;

    uint32_t ifd0 = 0;
    if (!r.U32(4, &ifd0)) return out;

    // Reads one entry's value. Values of 4 bytes or fewer are stored inline in
    // the offset field; anything longer is at the offset it names.
    auto valueOffset = [&](size_t entry, uint16_t type, uint32_t count,
                           size_t* off) -> bool {
        static const int kSize[] = {0, 1, 1, 2, 4, 8, 1, 1, 2, 4, 8, 4, 8};
        if (type == 0 || type > 12) return false;
        const uint64_t bytes = uint64_t(kSize[type]) * count;
        if (bytes <= 4) { *off = entry + 8; return true; }
        uint32_t o = 0;
        if (!r.U32(entry + 8, &o)) return false;
        *off = o;
        return *off < r.Size();
    };

    auto rational = [&](size_t off, double* out) -> bool {
        uint32_t num = 0, den = 0;
        if (!r.U32(off, &num) || !r.U32(off + 4, &den) || den == 0) return false;
        *out = double(num) / double(den);
        return true;
    };

    // Walks one IFD. `exifIfd` receives the offset of the Exif sub-IFD when the
    // pointer tag is seen, since the interesting tags live there rather than in
    // IFD0.
    auto walk = [&](size_t ifd, uint32_t* exifIfd) {
        uint16_t count = 0;
        if (!r.U16(ifd, &count)) return;
        // A corrupt count could otherwise send this walking off the end.
        if (count > 512) return;

        for (uint16_t i = 0; i < count; ++i) {
            const size_t e = ifd + 2 + size_t(i) * 12;
            uint16_t tag = 0, type = 0;
            uint32_t n = 0;
            if (!r.U16(e, &tag) || !r.U16(e + 2, &type) || !r.U32(e + 4, &n)) return;

            size_t off = 0;
            if (!valueOffset(e, type, n, &off)) continue;

            switch (tag) {
                case kMake:         out.cameraMake  = r.Str(off, n); break;
                case kModel:        out.cameraModel = r.Str(off, n); break;
                case kLensModel:    out.lens        = r.Str(off, n); break;
                case kDateTimeOrig: out.dateTaken   = r.Str(off, n); break;
                case kDateTime:
                    // Only as a fallback: DateTimeOriginal is the capture time,
                    // while DateTime is when the file was last written.
                    if (out.dateTaken.empty()) out.dateTaken = r.Str(off, n);
                    break;

                case kExposureTime: {
                    double v = 0;
                    if (rational(off, &v)) out.exposureTime = FormatExposure(v);
                    break;
                }
                case kFNumber: {
                    double v = 0;
                    if (rational(off, &v) && v > 0) {
                        char b[32];
                        std::snprintf(b, sizeof b, "f/%.1f", v);
                        out.aperture = b;
                    }
                    break;
                }
                case kFocalLength: {
                    double v = 0;
                    if (rational(off, &v) && v > 0) {
                        char b[32];
                        std::snprintf(b, sizeof b, "%.0f mm", v);
                        out.focalLength = b;
                    }
                    break;
                }
                case kIsoSpeed: {
                    uint16_t v = 0;
                    if (r.U16(off, &v) && v > 0) out.iso = "ISO " + std::to_string(v);
                    break;
                }
                case kExifIfd: {
                    uint32_t v = 0;
                    if (r.U32(e + 8, &v)) *exifIfd = v;
                    break;
                }
                default: break;
            }
        }
    };

    uint32_t exifIfd = 0;
    walk(ifd0, &exifIfd);
    if (exifIfd && exifIfd < r.Size()) {
        uint32_t ignored = 0;
        walk(exifIfd, &ignored);
    }

    out.present = out.Any();
    return out;
}

} // namespace tglab
