#include "raw_io.h"

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>

#include "libraw/libraw.h"

namespace tglab {
namespace {

// Extensions LibRaw claims to handle. Not exhaustive -- LibRaw supports many
// hundreds of camera models -- but these cover the formats a file actually
// arrives as. An unlisted extension falls through to stb_image, which will
// report its own error, so a wrong guess here degrades rather than breaks.
const char* const kRawExtensions[] = {
    ".3fr", ".arw", ".cr2", ".cr3", ".crw", ".dcr", ".dng", ".erf",
    ".iiq", ".kdc", ".mef", ".mos", ".mrw", ".nef", ".nrw", ".orf",
    ".pef", ".raf", ".raw", ".rw2", ".rwl", ".sr2", ".srf", ".srw",
    ".x3f",
};

// LibRaw pads its fixed-size char arrays, and some cameras write trailing
// spaces into the make/model strings themselves.
std::string Trim(const char* s) {
    if (!s) return {};
    std::string v(s);
    while (!v.empty() && (v.back() == ' ' || v.back() == '\0')) v.pop_back();
    return v;
}

template <class... Args>
std::string Fmt(const char* fmt, Args... args) {
    char buf[64];
    std::snprintf(buf, sizeof buf, fmt, args...);
    return buf;
}

// Fast shutter speeds read as a fraction, slow ones as decimals -- the same
// convention the JPEG path uses, so the panel is consistent.
std::string FormatExposure(double seconds) {
    if (seconds <= 0.0) return {};
    if (seconds < 1.0) return Fmt("1/%.0f s", 1.0 / seconds);
    return Fmt("%.1f s", seconds);
}

std::string LowerExtension(const std::string& path) {
    const size_t dot = path.find_last_of('.');
    if (dot == std::string::npos) return {};
    // A dot in a directory name is not an extension.
    if (path.find_first_of("/\\", dot) != std::string::npos) return {};

    std::string ext = path.substr(dot);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return char(std::tolower(c)); });
    return ext;
}

} // namespace

bool IsRawExtension(const std::string& path) {
    const std::string ext = LowerExtension(path);
    if (ext.empty()) return false;
    for (const char* known : kRawExtensions)
        if (ext == known) return true;
    return false;
}

bool ReadRawMetadata(const std::string& path, ExifData* out) {
    if (!IsRawExtension(path)) return false;

    LibRaw raw;
    // open_file() alone parses the metadata; unpack() and dcraw_process() are
    // the expensive parts and are not needed to read the capture settings.
    if (raw.open_file(path.c_str()) != LIBRAW_SUCCESS) return false;

    const auto& idata = raw.imgdata.idata;
    const auto& other = raw.imgdata.other;

    out->cameraMake  = Trim(idata.make);
    out->cameraModel = Trim(idata.model);
    out->lens        = Trim(raw.imgdata.lens.Lens);

    if (other.shutter > 0.0f)   out->exposureTime = FormatExposure(other.shutter);
    if (other.aperture > 0.0f)  out->aperture     = Fmt("f/%.1f", other.aperture);
    if (other.iso_speed > 0.0f) out->iso          = Fmt("ISO %.0f", other.iso_speed);
    if (other.focal_len > 0.0f) out->focalLength  = Fmt("%.0f mm", other.focal_len);

    if (other.timestamp > 0) {
        // Matches the EXIF spelling the JPEG path produces, so the panel shows
        // one format regardless of where the data came from.
        std::tm tm{};
        if (localtime_s(&tm, &other.timestamp) == 0) {
            char buf[32];
            std::strftime(buf, sizeof buf, "%Y:%m:%d %H:%M:%S", &tm);
            out->dateTaken = buf;
        }
    }

    out->present = out->Any();
    return out->present;
}

bool LoadRawFile(const std::string& path, Image* out, std::string* err) {
    LibRaw raw;

    // 8-bit sRGB output, which is what makes a raw file behave like any other
    // image here. Set explicitly rather than relying on LibRaw's defaults, so a
    // change in those cannot silently alter what the lab sees.
    raw.imgdata.params.output_bps    = 8;
    raw.imgdata.params.output_color  = 1;      // sRGB
    raw.imgdata.params.use_camera_wb = 1;      // as the camera metered it
    raw.imgdata.params.user_flip     = -1;     // honour the orientation tag

    // Auto-brightening ON, which is not the obvious choice for a lab and is
    // worth explaining.
    //
    // It sounds like "silently rescaling the data", so the first version turned
    // it off -- and a correctly-exposed 45 MP CR3 came out at a mean level of
    // 20/255, essentially black. The reason is that disabling it does not give
    // unscaled data: LibRaw still applies a gamma curve, just against a fixed
    // white point of 0x2000 rather than one derived from the image. A sensor
    // whose highlights fall below that ceiling is then mapped far too dark.
    //
    // With it on, LibRaw picks the white point by percentile (auto_bright_thr,
    // 0.1% clipped by default). That is a defensible choice rather than an
    // arbitrary constant, and it preserves relative exposure between frames --
    // three shots from one session came out at 108, 106 and 53, not all
    // normalised to the same level.
    //
    // TGLAB_RAW_NOBRIGHT=1 turns it off for anyone who wants the fixed white
    // point instead. The genuinely unscaled data is step two's business: it
    // needs the sensor mosaic and a viewer with exposure control, neither of
    // which this path has.
    raw.imgdata.params.no_auto_bright =
        (std::getenv("TGLAB_RAW_NOBRIGHT") != nullptr) ? 1 : 0;

    if (const int rc = raw.open_file(path.c_str()); rc != LIBRAW_SUCCESS) {
        *err = "could not open raw '" + path + "': " + libraw_strerror(rc);
        return false;
    }
    if (const int rc = raw.unpack(); rc != LIBRAW_SUCCESS) {
        *err = "could not unpack raw '" + path + "': " + libraw_strerror(rc);
        return false;
    }
    if (const int rc = raw.dcraw_process(); rc != LIBRAW_SUCCESS) {
        *err = "could not process raw '" + path + "': " + libraw_strerror(rc);
        return false;
    }

    int rc = LIBRAW_SUCCESS;
    libraw_processed_image_t* img = raw.dcraw_make_mem_image(&rc);
    if (!img) {
        *err = "raw produced no image: " + std::string(libraw_strerror(rc));
        return false;
    }

    // LibRaw can return a bitmap or an embedded JPEG thumbnail; only the former
    // is what was asked for here.
    if (img->type != LIBRAW_IMAGE_BITMAP || img->colors != 3 || img->bits != 8) {
        LibRaw::dcraw_clear_mem(img);
        *err = "unexpected raw output format from '" + path + "'";
        return false;
    }

    ImageDesc d;
    d.width  = img->width;
    d.height = img->height;
    d.format = Format::RGBA8;
    if (!d.Valid()) {
        LibRaw::dcraw_clear_mem(img);
        *err = "raw '" + path + "' has no usable dimensions";
        return false;
    }

    out->Alloc(d);
    ImageView v = out->MapCpuWrite();

    // RGB -> RGBA. Opaque alpha, since a raw capture has no transparency and
    // leaving it zero would make the image vanish in the viewer.
    const uint8_t* src = img->data;
    for (int y = 0; y < d.height; ++y) {
        uint8_t* dst = v.At<uint8_t>(0, y);
        for (int x = 0; x < d.width; ++x) {
            dst[x * 4 + 0] = src[0];
            dst[x * 4 + 1] = src[1];
            dst[x * 4 + 2] = src[2];
            dst[x * 4 + 3] = 255;
            src += 3;
        }
    }

    LibRaw::dcraw_clear_mem(img);
    return true;
}

} // namespace tglab
