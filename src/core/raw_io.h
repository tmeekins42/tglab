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

// Loads the sensor's undemosaiced Bayer mosaic: one sample per pixel, linear,
// as R32F with the CFA pattern and black/white levels on the descriptor.
//
// This is the path that actually buys dynamic range. LoadRawFile() asks LibRaw
// for finished 8-bit sRGB, which crushes a 14-bit sensor's 16384 levels to 256
// before any adjustment sees them -- pulling exposure down then recovers
// nothing, because the highlight information is already gone. The mosaic keeps
// all of it, and demosaicing becomes an algorithm the lab can compare rather
// than something that already happened.
//
// The visible frame only: LibRaw's raw buffer includes masked border pixels
// used for black-level calibration, which are not part of the picture.
bool LoadRawMosaic(const std::string& path, Image* out, std::string* err);

// The CFA pattern a mosaic has after being rotated by `flip`, in LibRaw's
// (dcraw's) encoding: 0 none, 3 = 180, 5 = 90 CCW, 6 = 90 CW.
//
// Exposed because it is the subtle half of honouring an orientation tag and
// deserves testing on its own. Rotating a mosaic is not a transpose: the colour
// filter rotates with the pixels, so an RGGB sensor turned 90 degrees clockwise
// reads GRBG, and getting that wrong swaps red and blue across the whole image
// -- which looks like a demosaic bug rather than an orientation one.
//
// `w` and `h` are the pre-rotation dimensions; they matter because an odd size
// shifts the CFA phase.
CfaPattern RotateCfa(CfaPattern src, int flip, int w, int h);

// The camera's own JPEG rendering of the frame, embedded in the raw file.
//
// Nearly every raw carries one, and it is the picture the body's own processing
// produced -- picture style, tone curve and white balance already applied. On a
// Canon CR3 from an R5 it is the full 8192x5464 frame at 5.2 MB. It is what
// Windows Explorer shows for a raw, and what makes those icons look so much
// better than a naive decode of the sensor data.
//
// Worth having because it is both faster and better-looking than developing the
// mosaic ourselves for a 48-pixel icon: no demosaic, no measurement pass, and
// the result carries the manufacturer's rendering rather than our
// approximation of one.
//
// Scaled down to fit `maxSide` on load, since the only caller wants a
// thumbnail and decoding 45 megapixels to draw an icon would defeat the point.
// Honours the raw's orientation flag, like LoadRawMosaic does.
//
// Returns false when the file carries no usable preview, which is a normal
// outcome and not an error -- the caller should fall back to developing the
// mosaic. `err` is only set for a genuine failure.
bool LoadRawPreview(const std::string& path, int maxSide, Image* out, std::string* err);

} // namespace tglab
