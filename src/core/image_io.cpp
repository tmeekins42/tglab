#include "image_io.h"

#include "raw_io.h"

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

} // namespace tglab
