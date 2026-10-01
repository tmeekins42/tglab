// color — the colour conventions every algorithm shares.
//
// ONE DEFINITION OF LUMA. Algorithms used to pick their own weights: sixteen
// files used Rec. 709 (0.2126, 0.7152, 0.0722) and nine used Rec. 601 (0.299,
// 0.587, 0.114), so a feature detector, a threshold and the histogram could
// each mean a different thing by "brightness" of the same pixel. Rec. 709 is
// the one: it is what sRGB's primaries are defined by, so it is the luminance
// of the linear values these images hold. 601 belongs to standard-definition
// television's primaries, which nothing here uses.
//
// The same weights for HLSL are in kColorHlsl below: a kernel prepends it to
// its source and calls Luma(), SrgbToLinear() and LinearToSrgb() exactly as
// C++ code does.
#pragma once

#include <cmath>

namespace tglab {

// Rec. 709 luma weights, for linear RGB.
inline constexpr double kLumaR = 0.2126, kLumaG = 0.7152, kLumaB = 0.0722;

template <typename T>
inline T Luma(T r, T g, T b) {
    return T(kLumaR) * r + T(kLumaG) * g + T(kLumaB) * b;
}
// Of an RGB(A) pixel, as three consecutive values.
template <typename T>
inline T Luma(const T* rgb) {
    return Luma(rgb[0], rgb[1], rgb[2]);
}

// The sRGB transfer curve, per channel: the IEC 61966-2-1 piecewise form -- a
// linear toe, then a 2.4 power -- not the 2.2-power approximation, since the
// toe is where shadow adjustments live and where the two disagree most.
// Encoding clamps to 0..1 first: it is for display, and a negative has no
// power.
inline float SrgbToLinear(float c) {
    return (c <= 0.04045f) ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}
inline float LinearToSrgb(float c) {
    c = c < 0.0f ? 0.0f : (c > 1.0f ? 1.0f : c);
    return (c <= 0.0031308f) ? c * 12.92f : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
}

// CIE XYZ (D65) to linear sRGB.
inline void XyzToLinearSrgb(float X, float Y, float Z, float* r, float* g, float* b) {
    *r =  3.2404542f * X - 1.5371385f * Y - 0.4985314f * Z;
    *g = -0.9692660f * X + 1.8760108f * Y + 0.0415560f * Z;
    *b =  0.0556434f * X - 0.2040259f * Y + 1.0572252f * Z;
}

// The same for kernels: prepend to an HLSL source.
inline constexpr const char* kColorHlsl = R"(
static const float3 kLumaW = float3(0.2126, 0.7152, 0.0722);   // Rec. 709
float Luma(float3 c) { return dot(c, kLumaW); }
// select(), not ?: -- SM 6.x requires it for a per-component condition.
float3 SrgbToLinear(float3 c) {
    return select(c <= 0.04045, c / 12.92, pow((c + 0.055) / 1.055, 2.4));
}
float3 LinearToSrgb(float3 c) {
    c = saturate(c);
    return select(c <= 0.0031308, c * 12.92, 1.055 * pow(c, 1.0 / 2.4) - 0.055);
}
)";

}  // namespace tglab
