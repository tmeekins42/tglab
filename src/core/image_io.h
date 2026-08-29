#pragma once

#include <string>

#include "image.h"

namespace tglab {

// Loads any format stb_image handles (PNG/JPG/BMP/TGA/HDR) as RGBA8.
bool LoadImageFile(const std::string& path, Image* out, std::string* err);

// Writes RGBA8 as PNG.
bool SavePng(const std::string& path, Image& img, std::string* err);

// What to write. Chosen from the extension unless a script names one.
enum class SaveFormat { Png, Jpg, Bmp, Tga, Hdr };

// Picks a format from a path's extension. Png when there is nothing to go on,
// since it is the lossless one that every viewer opens.
SaveFormat SaveFormatFromPath(const std::string& path);

// Writes an image of ANY format, converting as the target requires.
//
// The conversion is the part worth stating, because getting it wrong produces a
// file that does not match what the app showed:
//
//   - a LINEAR image written to an 8-bit format goes through the display tone
//     curve, exactly as the viewer draws it. Writing linear values straight to
//     8 bits is what made the loupe darker than the image beside it -- middle
//     grey at 0.18 linear becomes 46/255 rather than the 117/255 the curve puts
//     it at.
//   - a gamma-encoded image is already display referred and is only clamped;
//     running the curve again would double-encode.
//   - .hdr keeps the LINEAR values, undoing nothing. That is the whole reason
//     to offer it: a merged bracket reaching 900 linear cannot survive an 8-bit
//     format, and clipping it silently would throw away the headroom the merge
//     existed to capture.
//
// `quality` is 1..100 and only meaningful for JPEG.
bool SaveImage(const std::string& path, Image& img, SaveFormat fmt,
               int quality, std::string* err);

// Convenience: format from the extension, default quality.
bool SaveImage(const std::string& path, Image& img, std::string* err);

// Adds a numeric suffix until the path does not exist: "out.png" becomes
// "out_1.png", then "out_2.png". Returns `path` unchanged when it is free.
//
// The number goes before the extension rather than after, so the result is
// still a .png to everything that reads it.
std::string NextFreePath(const std::string& path);

} // namespace tglab
