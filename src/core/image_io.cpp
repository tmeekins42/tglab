#include "image_io.h"

#include "raw_io.h"

#include "../algo_util/pixel_buffer.h"
#include "../algo_util/tone_curve.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <vector>

#include <cstdlib>
#include <cstring>
#include <windows.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_WINDOWS_UTF8
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBIW_WINDOWS_UTF8
#include "stb_image_write.h"

namespace tglab {

bool LoadImageFile(const std::string& path, Image* out, std::string* err) {
    // TGLAB_SLOWLOAD=<ms> simulates a slow or network drive, so the "loading..."
    // indicator and the background loader can be exercised from a local disk.
    if (const char* slow = std::getenv("TGLAB_SLOWLOAD")) Sleep(DWORD(std::atoi(slow)));

    // Camera raw goes to LibRaw; everything else to stb_image. Dispatching on
    // the extension keeps both call sites (the drop handler and the loader
    // thread) unaware that raw exists at all.
    if (IsRawExtension(path)) {
        // The mosaic is the point of the raw path: it carries the sensor's full
        // range (a Canon CR2 measures ~13,400 distinct levels) where LibRaw's
        // finished output has already crushed that to 256. Demosaicing then
        // happens in the pipeline, where it can be compared and swapped.
        //
        // TGLAB_RAW_RGB=1 falls back to LibRaw's own conversion, which is the
        // escape hatch if a camera's mosaic cannot be read.
        if (!std::getenv("TGLAB_RAW_RGB")) {
            std::string mosaicErr;
            if (LoadRawMosaic(path, out, &mosaicErr)) return true;
            // Fall through rather than fail: a sensor with no Bayer mosaic
            // (Foveon, some medium format) still has a perfectly good RGB
            // conversion, and refusing to open it would be worse.
        }
        return LoadRawFile(path, out, err);
    }

    int w = 0, h = 0, comp = 0;
    unsigned char* pixels = stbi_load(path.c_str(), &w, &h, &comp, 4);
    if (!pixels) {
        const char* reason = stbi_failure_reason();
        *err = "could not load '" + path + "'" + (reason ? std::string(": ") + reason : "");
        return false;
    }

    ImageDesc d;
    d.width  = w;
    d.height = h;
    d.format = Format::RGBA8;

    out->Alloc(d);
    ImageView v = out->MapCpuWrite();
    std::memcpy(v.data, pixels, d.SizeInBytes());
    stbi_image_free(pixels);
    return true;
}

bool SavePng(const std::string& path, Image& img, std::string* err) {
    if (!img.Valid() || img.Desc().format != Format::RGBA8) {
        *err = "SavePng expects a valid RGBA8 image";
        return false;
    }
    ImageView v = img.MapCpuRead();
    if (!stbi_write_png(path.c_str(), v.desc.width, v.desc.height, 4, v.data, v.Pitch())) {
        *err = "could not write '" + path + "'";
        return false;
    }
    return true;
}

SaveFormat SaveFormatFromPath(const std::string& path) {
    const auto dot = path.find_last_of('.');
    std::string ext = (dot == std::string::npos) ? "" : path.substr(dot + 1);
    for (char& c : ext) c = char(std::tolower(static_cast<unsigned char>(c)));

    if (ext == "jpg" || ext == "jpeg") return SaveFormat::Jpg;
    if (ext == "bmp")                  return SaveFormat::Bmp;
    if (ext == "tga")                  return SaveFormat::Tga;
    if (ext == "hdr")                  return SaveFormat::Hdr;
    return SaveFormat::Png;
}

std::string NextFreePath(const std::string& path) {
    if (GetFileAttributesA(path.c_str()) == INVALID_FILE_ATTRIBUTES) return path;

    const auto dot = path.find_last_of('.');
    // Only an extension AFTER the last separator counts: "a.b/out" has a dot
    // but no extension, and splitting there would produce "a_1.b/out".
    const auto slash = path.find_last_of("/\\");
    const bool hasExt = dot != std::string::npos &&
                        (slash == std::string::npos || dot > slash);
    const std::string stem = hasExt ? path.substr(0, dot) : path;
    const std::string ext  = hasExt ? path.substr(dot) : "";

    for (int i = 1; i < 10000; ++i) {
        const std::string tryPath = stem + "_" + std::to_string(i) + ext;
        if (GetFileAttributesA(tryPath.c_str()) == INVALID_FILE_ATTRIBUTES) return tryPath;
    }
    return path;   // give up and let the write overwrite rather than hang
}

bool SaveImage(const std::string& path, Image& img, SaveFormat fmt,
               int quality, std::string* err) {
    if (!img.Valid()) { *err = "nothing to save"; return false; }

    // Create the directory rather than failing on it. A script saying
    // save(img, "export/out.png") plainly means to put a file in export/, and
    // stb's failure for a missing directory is an unhelpful "could not write".
    {
        const auto slash = path.find_last_of("/\\");
        if (slash != std::string::npos && slash > 0) {
            std::error_code ec;
            std::filesystem::create_directories(path.substr(0, slash), ec);
            // Not checked: if it failed the write below reports it, and a
            // directory that already exists is a success reported as an error
            // by some implementations.
        }
    }

    ImageView v = img.MapCpuRead();
    if (!v.data) { *err = "could not read the image"; return false; }

    const int w = v.desc.width, h = v.desc.height;

    // Radiance .hdr: the LINEAR values, unconverted. See the header for why
    // this format exists here at all.
    if (fmt == SaveFormat::Hdr) {
        PixelBuffer pb;
        pb.Unpack(v);
        if (!pb.Valid()) { *err = "unsupported pixel format"; return false; }

        const int ch = pb.Channels();
        const float scale = pb.ValueScale();
        std::vector<float> rgb(size_t(w) * size_t(h) * 3);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const float* p = pb.At(x, y);
                float* o = &rgb[(size_t(y) * size_t(w) + size_t(x)) * 3];
                for (int c = 0; c < 3; ++c) o[c] = p[(ch >= 3) ? c : 0] / scale;
            }
        if (!stbi_write_hdr(path.c_str(), w, h, 3, rgb.data())) {
            *err = "could not write '" + path + "'";
            return false;
        }
        return true;
    }

    // Everything else is 8-bit, so the image has to be brought into 0..255 the
    // same way the viewer draws it.
    std::vector<uint8_t> out(size_t(w) * size_t(h) * 4);

    if (v.desc.format == Format::RGBA8) {
        std::memcpy(out.data(), v.data, out.size());
    } else {
        PixelBuffer pb;
        pb.Unpack(v);
        if (!pb.Valid()) { *err = "unsupported pixel format"; return false; }

        const int ch = pb.Channels();
        const float scale = pb.ValueScale();
        const bool linear = v.desc.linear;

        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const float* p = pb.At(x, y);
                uint8_t* o = &out[(size_t(y) * size_t(w) + size_t(x)) * 4];
                for (int c = 0; c < 3; ++c) {
                    const float lin = p[(ch >= 3) ? c : 0] / scale;
                    // Identical rule to image_view.cpp's loupe: tone curve for
                    // linear data, clamp for anything already display referred.
                    const float d = linear ? ToneCurve(lin) : std::clamp(lin, 0.0f, 1.0f);
                    o[c] = uint8_t(std::clamp(d * 255.0f + 0.5f, 0.0f, 255.0f));
                }
                o[3] = (ch == 4) ? uint8_t(std::clamp(p[3] / scale * 255.0f + 0.5f, 0.0f, 255.0f))
                                 : 255;
            }
    }

    const int stride = w * 4;
    int ok = 0;
    switch (fmt) {
        case SaveFormat::Jpg:
            // JPEG has no alpha, and stb writes 3-channel from a 4-channel
            // buffer only if told the component count -- so pack down first.
            {
                std::vector<uint8_t> rgb(size_t(w) * size_t(h) * 3);
                for (size_t i = 0, n = size_t(w) * size_t(h); i < n; ++i) {
                    rgb[i * 3 + 0] = out[i * 4 + 0];
                    rgb[i * 3 + 1] = out[i * 4 + 1];
                    rgb[i * 3 + 2] = out[i * 4 + 2];
                }
                ok = stbi_write_jpg(path.c_str(), w, h, 3, rgb.data(),
                                    std::clamp(quality, 1, 100));
            }
            break;
        case SaveFormat::Bmp: ok = stbi_write_bmp(path.c_str(), w, h, 4, out.data()); break;
        case SaveFormat::Tga: ok = stbi_write_tga(path.c_str(), w, h, 4, out.data()); break;
        case SaveFormat::Hdr: break;   // handled above
        case SaveFormat::Png:
        default:
            ok = stbi_write_png(path.c_str(), w, h, 4, out.data(), stride);
            break;
    }

    if (!ok) { *err = "could not write '" + path + "'"; return false; }
    return true;
}

bool SaveImage(const std::string& path, Image& img, std::string* err) {
    return SaveImage(path, img, SaveFormatFromPath(path), 92, err);
}

} // namespace tglab
