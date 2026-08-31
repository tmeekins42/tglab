#include "raw_io.h"

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <memory>
#include <vector>

#include "libraw/libraw.h"
#include "stb_image.h"

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

namespace {

} // namespace

// Declared in the header; see there for why this is worth exposing.
//
// Derived from the same coordinate map the copy loop uses rather than from a
// hand-written table, so the two cannot disagree. That also handles the case a
// table would get wrong: with an ODD width or height the rotation shifts the
// CFA phase, because the tile no longer lands on an even boundary.
CfaPattern RotateCfa(CfaPattern src, int flip, int w, int h) {
    if (src == CfaPattern::None || src == CfaPattern::XTrans) return src;
    if (flip != 3 && flip != 5 && flip != 6) return src;

    // Where each of the source tile's four sensels ends up, and what colour it
    // carries. Reading the destination tile back gives the rotated pattern.
    int dest[2][2] = {{-1, -1}, {-1, -1}};
    for (int sy = 0; sy < 2; ++sy) {
        for (int sx = 0; sx < 2; ++sx) {
            int dx = sx, dy = sy;
            switch (flip) {
                case 3: dx = w - 1 - sx; dy = h - 1 - sy; break;
                case 5: dx = sy;         dy = w - 1 - sx; break;
                case 6: dx = h - 1 - sy; dy = sx;         break;
            }
            dest[dy & 1][dx & 1] = CfaColorAt(src, sx, sy);
        }
    }

    const int tl = dest[0][0], tr = dest[0][1];
    const int bl = dest[1][0], br = dest[1][1];
    if (tl == 0 && tr == 1 && bl == 1 && br == 2) return CfaPattern::RGGB;
    if (tl == 2 && tr == 1 && bl == 1 && br == 0) return CfaPattern::BGGR;
    if (tl == 1 && tr == 0 && bl == 2 && br == 1) return CfaPattern::GRBG;
    if (tl == 1 && tr == 2 && bl == 0 && br == 1) return CfaPattern::GBRG;
    return src;   // should not happen; leaving it alone beats guessing
}

namespace {

// LibRaw reports the CFA layout through filters(), a packed 32-bit code. Rather
// than decode that bit layout, ask it which colour sits at each of the four
// positions in the 2x2 tile -- COLOR() handles every vendor quirk internally.
CfaPattern CfaFromLibRaw(LibRaw& raw) {
    if (raw.imgdata.idata.filters == 9) return CfaPattern::XTrans;
    if (raw.imgdata.idata.filters == 0) return CfaPattern::None;   // already RGB

    // LibRaw's colour indices are 0=R, 1=G, 2=B, 3=G2. Fold the second green
    // onto the first, since the two are interchangeable for our purposes.
    auto at = [&](int r, int c) {
        const int v = raw.COLOR(r, c);
        return v == 3 ? 1 : v;
    };

    const int tl = at(0, 0), tr = at(0, 1), bl = at(1, 0), br = at(1, 1);
    if (tl == 0 && tr == 1 && bl == 1 && br == 2) return CfaPattern::RGGB;
    if (tl == 2 && tr == 1 && bl == 1 && br == 0) return CfaPattern::BGGR;
    if (tl == 1 && tr == 0 && bl == 2 && br == 1) return CfaPattern::GRBG;
    if (tl == 1 && tr == 2 && bl == 0 && br == 1) return CfaPattern::GBRG;
    return CfaPattern::None;
}

} // namespace

bool LoadRawMosaic(const std::string& path, Image* out, std::string* err) {
    LibRaw raw;

    if (const int rc = raw.open_file(path.c_str()); rc != LIBRAW_SUCCESS) {
        *err = "could not open raw '" + path + "': " + libraw_strerror(rc);
        return false;
    }
    if (const int rc = raw.unpack(); rc != LIBRAW_SUCCESS) {
        *err = "could not unpack raw '" + path + "': " + libraw_strerror(rc);
        return false;
    }

    const ushort* raw_image = raw.imgdata.rawdata.raw_image;
    if (!raw_image) {
        // Foveon and some medium-format backs have no Bayer mosaic at all.
        *err = "'" + path + "' has no Bayer mosaic (a three-colour sensor?)";
        return false;
    }

    const CfaPattern cfa = CfaFromLibRaw(raw);
    if (cfa == CfaPattern::None) {
        *err = "'" + path + "' has an unrecognised colour filter array";
        return false;
    }

    const auto& s = raw.imgdata.sizes;

    // The visible frame, not the full sensor readout. The margins are masked
    // pixels used for black-level calibration; including them would put a black
    // band down two edges of every image.
    const int w = s.width;
    const int h = s.height;
    if (w <= 0 || h <= 0) {
        *err = "'" + path + "' reports no visible image area";
        return false;
    }

    // Cropping shifts the CFA phase. A left or top margin with an odd offset
    // means the visible frame starts on a different colour than the sensor
    // does, and getting this wrong swaps red and blue across the whole image.
    const int xoff = s.left_margin;
    const int yoff = s.top_margin;
    CfaPattern visibleCfa = cfa;
    if ((xoff & 1) || (yoff & 1)) {
        // Re-derive from the CFA colour at the visible frame's own origin.
        auto at = [&](int r, int c) {
            const int v = raw.COLOR(r + yoff, c + xoff);
            return v == 3 ? 1 : v;
        };
        const int tl = at(0, 0), tr = at(0, 1), bl = at(1, 0), br = at(1, 1);
        if      (tl == 0 && tr == 1 && bl == 1 && br == 2) visibleCfa = CfaPattern::RGGB;
        else if (tl == 2 && tr == 1 && bl == 1 && br == 0) visibleCfa = CfaPattern::BGGR;
        else if (tl == 1 && tr == 0 && bl == 2 && br == 1) visibleCfa = CfaPattern::GRBG;
        else if (tl == 1 && tr == 2 && bl == 0 && br == 1) visibleCfa = CfaPattern::GBRG;
    }

    // The camera's orientation tag, applied here rather than left to the
    // viewer.
    //
    // LibRaw reports this in dcraw's encoding, which is NOT the EXIF
    // orientation number: 0 none, 3 = 180, 5 = 90 CCW, 6 = 90 CW. Tim's
    // _dsc0139.arw declares 6 and was displayed on its side; his other files
    // declare 0, which is why this went unnoticed.
    //
    // Rotating the MOSAIC rather than the demosaiced result, because the
    // palette holds the mosaic: leaving it to a later stage would rotate the
    // picture but leave the thumbnail, the loupe and every measurement working
    // in sensor space, which is a worse kind of wrong than not rotating at all.
    const int flip = s.flip;
    const bool swapAxes = (flip == 5 || flip == 6);
    const int outW = swapAxes ? h : w;
    const int outH = swapAxes ? w : h;

    ImageDesc d;
    d.width      = outW;
    d.height     = outH;
    d.format     = Format::R32F;
    d.cfa        = RotateCfa(visibleCfa, flip, w, h);
    // cblack[] holds per-channel offsets on top of the global black level; the
    // first four cover the 2x2 tile. Averaging them is an approximation, but
    // the per-channel spread is small on every sensor this has been tried on,
    // and a single scalar keeps the descriptor simple.
    {
        double black = raw.imgdata.color.black;
        double sum = 0;
        for (int i = 0; i < 4; ++i) sum += raw.imgdata.color.cblack[i];
        black += sum / 4.0;
        d.blackLevel = float(black);
    }
    d.whiteLevel = float(raw.imgdata.color.maximum);
    if (d.whiteLevel <= d.blackLevel) d.whiteLevel = d.blackLevel + 1.0f;

    // ...and then MEASURED from the data, because color.maximum is not
    // reliably the sensor's saturation point.
    //
    // THIS MATTERS FAR MORE THAN IT SOUNDS. Everything downstream that has to
    // know "is this pixel blown" tests a normalised value against a threshold
    // near 1.0, and normalisation divides by (whiteLevel - blackLevel). Get
    // whiteLevel wrong and the test means the wrong thing.
    //
    // What that costs, measured on a night shot at 3313 K: the sensor really
    // saturates at 15284 against a declared 15488, so a blown pixel normalises
    // to 0.9848 and the clipped-channel repair -- which triggers at 0.99 --
    // never fires. White balance then multiplies the channels apart (blue x3.08
    // at that temperature), the camera matrix's negative green coefficients
    // crush what is left, and a neutral saturated pixel develops as R 1.77,
    // G 0.06, B 4.48. Every street light in the frame came out MAGENTA, and
    // 66.5% of the bright pixels were affected.
    //
    // The value is genuinely per-file, so a constant cannot fix it: measured
    // across three bodies, real saturation lands at 0.9848, 1.0000 and 1.0013
    // of the declared white level -- on BOTH sides of it.
    //
    // A saturating sensor produces a SPIKE: everything brighter than the
    // ceiling reads the same value, so pixels pile up there. Searching down
    // from the top for a bin holding far more than its neighbours finds it.
    // Restricted to the top half of the range, because a frame with nothing
    // blown has its largest pile-up at the noise floor instead.
    {
        const int lo = int(d.blackLevel);
        const int top = int(d.whiteLevel * 1.15f);   // room to find it ABOVE
        if (top > lo + 16) {
            std::vector<int> hist(size_t(top - lo + 1), 0);
            // Every fourth pixel in each direction: the spike is thousands of
            // samples, so a sixteenth of them still resolves it, and this runs
            // on every raw load.
            // `w` and `h`, NOT d.width and d.height.
            //
            // d.width/d.height are the ROTATED dimensions -- outW/outH above
            // swap them when the EXIF orientation is a quarter turn. The raw
            // buffer is never rotated, so indexing it with the rotated extents
            // reads off the end of every row on a portrait frame and walks past
            // the end of the allocation.
            //
            // A Canon R5 file with flip = 6 segfaulted before printing a single
            // line, which is what made it look like a loader failure rather
            // than an indexing one: the crash happens inside the raw load,
            // before anything downstream has run.
            for (int y = 0; y < h; y += 4)
                for (int x = 0; x < w; x += 4) {
                    const int v = int(raw_image[size_t(y + s.top_margin) *
                                                size_t(s.raw_width) +
                                                size_t(x + s.left_margin)]);
                    if (v >= lo && v <= top) ++hist[size_t(v - lo)];
                }

            const int floorBin = (int(d.whiteLevel) - lo) / 2;
            for (int v = top - lo; v > floorBin; --v) {
                const long long c = hist[size_t(v)];
                if (c < 32) continue;
                long long below = 0;
                int n = 0;
                for (int t = std::max(0, v - 200); t < v; ++t) { below += hist[size_t(t)]; ++n; }
                const double avg = n ? double(below) / double(n) : 0.0;
                // Five times the local average AND an absolute floor: a dark
                // frame has neither, and a bright one has both by a wide
                // margin.
                if (double(c) > std::max(avg * 5.0, 32.0)) {
                    const float found = float(v + lo);
                    // Only ever LOWERED towards the measurement, and never by
                    // more than a stop: a spike far below the declared white is
                    // more likely a histogram artefact than a saturation point,
                    // and trusting it would clip real highlights.
                    if (found > d.blackLevel + 1.0f && found < d.whiteLevel)
                        d.whiteLevel = std::max(found, d.blackLevel +
                                                (d.whiteLevel - d.blackLevel) * 0.5f);
                    break;
                }
            }
        }
    }

    // As-shot white balance, normalised to green.
    //
    // Without this the image is overwhelmingly green: a sensor's green
    // photosites are far more sensitive than its red and blue ones, and the
    // camera's own multipliers are what correct for that. Measured on real
    // files: a Canon 5D3 needs R x1.45 / B x2.37, a Sony RX100M5 R x2.26 /
    // B x1.79 -- large enough that omitting them is immediately visible.
    {
        const float* m = raw.imgdata.color.cam_mul;
        const float g = (m[1] > 0.0f) ? m[1] : 1.0f;
        d.camMul[0] = (m[0] > 0.0f) ? m[0] / g : 1.0f;
        d.camMul[1] = 1.0f;
        d.camMul[2] = (m[2] > 0.0f) ? m[2] / g : 1.0f;
    }

    // Exposure settings as numbers, for an HDR merge to divide out. Kept
    // alongside the formatted EXIF rather than instead of it: the info panel
    // wants "1/250 s" and merge_hdr wants 0.004.
    {
        const auto& o = raw.imgdata.other;
        d.shutter  = (o.shutter   > 0.0f) ? o.shutter   : 0.0f;
        d.aperture = (o.aperture  > 0.0f) ? o.aperture  : 0.0f;
        d.iso      = (o.iso_speed > 0.0f) ? o.iso_speed : 0.0f;
    }

    // The camera's DAYLIGHT reference: the gains that make a D65 white neutral
    // on this sensor. Distinct from cam_mul, which is what the camera chose for
    // this particular shot.
    //
    // Without it a temperature control can only be relative -- "a bit warmer
    // than however this was shot" -- because nothing says where neutral is.
    // With it, the ratio between as-shot and daylight gives the shot's actual
    // colour temperature, and the slider can name a Kelvin value and mean it.
    //
    // Measured: a 5D3 frame shot under tungsten reads as-shot 1.454/1/2.370
    // against a daylight 2.251/1/1.418 -- a factor of 0.65 in red and 1.67 in
    // blue, far outside the +-40% a relative control could reach.
    {
        const float* p = raw.imgdata.color.pre_mul;
        const float g = (p[1] > 0.0f) ? p[1] : 1.0f;
        if (p[0] > 0.0f && p[2] > 0.0f) {
            d.preMul[0] = p[0] / g;
            d.preMul[1] = 1.0f;
            d.preMul[2] = p[2] / g;
            d.hasDaylightWb = true;
        }
    }

    // Camera primaries -> sRGB. Sensor filters are not sRGB primaries, so even
    // perfectly balanced data has the wrong hues without this -- greens too
    // yellow, blues too purple. LibRaw derives it from the embedded profile.
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            d.rgbCam[r * 3 + c] = raw.imgdata.color.rgb_cam[r][c];

    // A file with no embedded matrix leaves rgb_cam zeroed, which would map
    // every pixel to black. Identity is wrong but visible; black is not.
    {
        float sum = 0.0f;
        for (int i = 0; i < 9; ++i) sum += std::abs(d.rgbCam[i]);
        if (sum < 1e-6f) {
            for (int i = 0; i < 9; ++i) d.rgbCam[i] = (i % 4 == 0) ? 1.0f : 0.0f;
        }
    }

    out->Alloc(d);
    ImageView v = out->MapCpuWrite();

    // Stored as raw sensor counts rather than normalised here: the demosaic
    // applies the black/white levels, so every algorithm sees the same
    // convention and the levels stay inspectable on the descriptor.
    // Rotation happens here rather than as a second pass: the destination
    // index is a function of the source one, so this costs an addressing
    // change rather than another full-size buffer. At 21 MP that matters.
    const int stride = s.raw_width;
    for (int y = 0; y < h; ++y) {
        const ushort* row = raw_image + size_t(y + yoff) * size_t(stride) + size_t(xoff);
        for (int x = 0; x < w; ++x) {
            int dx = x, dy = y;
            switch (flip) {
                case 3: dx = w - 1 - x; dy = h - 1 - y; break;   // 180
                case 5: dx = y;         dy = w - 1 - x; break;   // 90 CCW
                case 6: dx = h - 1 - y; dy = x;         break;   // 90 CW
                default: break;                                  // 0, or unknown
            }
            *v.At<float>(dx, dy) = float(row[x]);
        }
    }

    return true;
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

bool LoadRawPreview(const std::string& path, int maxSide, Image* out, std::string* err) {
    if (!out) return false;

    // Heap rather than the stack. LibRaw is around a megabyte, and while the
    // three instances above live happily on the app's 1 MB main-thread stack,
    // this one also gets called from the loader thread and from test binaries,
    // whose stacks are smaller. A bare console exe with one on the stack
    // overflows before reaching main -- observed while investigating this very
    // feature, as a silent exit with code 0xC00000FD.
    auto raw = std::make_unique<LibRaw>();

    if (raw->open_file(path.c_str()) != LIBRAW_SUCCESS) return false;
    if (raw->unpack_thumb() != LIBRAW_SUCCESS) return false;

    const libraw_thumbnail_t& t = raw->imgdata.thumbnail;
    if (!t.thumb || t.tlength == 0) return false;

    // Only JPEG previews. LibRaw also reports bitmap thumbnails, but those are
    // the small low-quality ones -- the full-frame render that makes this
    // worthwhile is always JPEG.
    if (t.tformat != LIBRAW_THUMBNAIL_JPEG) return false;

    int w = 0, h = 0, comp = 0;
    // Decode at reduced size where stb can do it for us... it cannot, so decode
    // fully and downsample below. Acceptable because this runs once per file on
    // the loader thread, never per frame.
    unsigned char* pixels = stbi_load_from_memory(
        reinterpret_cast<const unsigned char*>(t.thumb),
        int(t.tlength), &w, &h, &comp, 4);
    if (!pixels) {
        if (err) *err = "embedded preview is not decodable JPEG";
        return false;
    }

    // The preview is stored in the sensor's orientation, so it needs the same
    // flip the mosaic gets. Without this a portrait frame's thumbnail comes out
    // sideways while its developed image is upright.
    const int flip = raw->imgdata.sizes.flip;
    const bool swapAxes = (flip == 5 || flip == 6);
    const int rw = swapAxes ? h : w;
    const int rh = swapAxes ? w : h;

    // Box-filter down to the requested size in one pass, straight out of the
    // decoded buffer. Nearest-neighbour would alias badly on a 45 MP source
    // reduced to 96 pixels -- fine detail like the brickwork turns into noise.
    const int scale = std::max(1, std::min(rw / std::max(maxSide, 1),
                                           rh / std::max(maxSide, 1)));
    const int ow = std::max(1, rw / scale);
    const int oh = std::max(1, rh / scale);

    ImageDesc d;
    d.width  = ow;
    d.height = oh;
    d.format = Format::RGBA8;
    d.linear = false;          // a JPEG is already display-encoded
    out->Alloc(d);
    if (!out->Valid()) {
        stbi_image_free(pixels);
        if (err) *err = "could not allocate preview image";
        return false;
    }
    ImageView v = out->MapCpuWrite();

    for (int y = 0; y < oh; ++y) {
        uint8_t* dst = v.At<uint8_t>(0, y);
        for (int x = 0; x < ow; ++x) {
            uint32_t acc[4] = {0, 0, 0, 0};
            int n = 0;
            for (int sy = 0; sy < scale; ++sy) {
                for (int sx = 0; sx < scale; ++sx) {
                    // Map the output pixel back through the rotation to find
                    // which source pixel feeds it. Inverting the flip here
                    // rather than rotating a second buffer keeps this to one
                    // pass over the data.
                    //
                    // The inverse is expressed in terms of the SOURCE
                    // dimensions (w, h), not the rotated ones. Writing it with
                    // the rotated extents is the obvious mistake and a quiet
                    // one: for a 90-degree flip it produces coordinates up to
                    // the long edge, the bounds check below skips them, and
                    // the skipped samples simply do not contribute -- so the
                    // image still appears, just darker. Measured on the
                    // Petaluma CR3, that read as mean 0.140 against the
                    // decoded preview's true 0.308.
                    const int rx = x * scale + sx;
                    const int ry = y * scale + sy;
                    if (rx >= rw || ry >= rh) continue;
                    int ux = rx, uy = ry;
                    switch (flip) {
                        case 3: ux = w - 1 - rx; uy = h - 1 - ry; break;  // 180
                        case 5: ux = w - 1 - ry; uy = rx;         break;  // 90 CCW
                        case 6: ux = ry;         uy = h - 1 - rx; break;  // 90 CW
                        default: break;
                    }
                    if (ux < 0 || ux >= w || uy < 0 || uy >= h) continue;
                    const unsigned char* s = pixels + (size_t(uy) * size_t(w) + size_t(ux)) * 4;
                    acc[0] += s[0]; acc[1] += s[1]; acc[2] += s[2]; acc[3] += s[3];
                    ++n;
                }
            }
            if (n == 0) n = 1;
            dst[x * 4 + 0] = uint8_t(acc[0] / uint32_t(n));
            dst[x * 4 + 1] = uint8_t(acc[1] / uint32_t(n));
            dst[x * 4 + 2] = uint8_t(acc[2] / uint32_t(n));
            dst[x * 4 + 3] = 255;
        }
    }

    stbi_image_free(pixels);
    return true;
}

} // namespace tglab
