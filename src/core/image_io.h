#pragma once

#include <string>

#include "image.h"

namespace tglab {

// Loads any format stb_image handles (PNG/JPG/BMP/TGA/HDR) as RGBA8.
bool LoadImageFile(const std::string& path, Image* out, std::string* err);

// Writes RGBA8 as PNG.
bool SavePng(const std::string& path, Image& img, std::string* err);

} // namespace tglab
