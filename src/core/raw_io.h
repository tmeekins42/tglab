// Camera raw loading, via LibRaw.
//
// This is step one of two. Here LibRaw does the whole conversion -- black
// level, white balance, demosaic, colour matrix, gamma -- and hands back RGB,
// so a .CR2 or .NEF behaves exactly like a JPEG and every existing filter
// works on it unchanged.
//
// Step two is to expose the sensor's Bayer mosaic instead, so demosaicing
// becomes an algorithm *in* the lab rather than something that happened before
// the image arrived. That is the interesting version for research; this one is
// the version that makes raw files usable today. The two are not in conflict:
// this path stays as the "just show me the picture" default.
#pragma once

#include <string>

#include "exif.h"
#include "image.h"

namespace tglab {

// True when the extension is one LibRaw might handle. Cheap check on the name
// alone -- deciding by content would mean opening every dropped file twice.
bool IsRawExtension(const std::string& path);

// Decodes a raw file to RGBA8 using LibRaw's own processing.
//
// 16-bit output is deliberately not used yet: the viewer clamps float data to
// 0..1 with no exposure control, so linear raw would display as nearly black.
// Steps two brings both together.
bool LoadRawFile(const std::string& path, Image* out, std::string* err);

// Capture settings from a raw file.
//
// Raw formats do carry EXIF, but not where a JPEG does: CR3 is ISO-BMFF, and
// CR2/ARW/NEF are TIFF variants whose IFD layout differs per vendor. Rather
// than teach the JPEG reader several container formats, this asks LibRaw --
// which already parses all of them to decode the image at all.
//
// Returns false when the file is not raw or has no usable metadata, leaving
// `out` untouched.
bool ReadRawMetadata(const std::string& path, ExifData* out);

} // namespace tglab
