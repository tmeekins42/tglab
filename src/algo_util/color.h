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

// OkLab (Ottosson 2020), from and to LINEAR sRGB. A perceptual space whose
// hue angle stays put as lightness and chroma change -- which is what lets a
// stage pick out "purple" or "green" by angle and change chroma alone. The
// cube root is sign-preserving: a linear pipeline carries out-of-gamut
// negatives, and std::cbrt of one is defined, pow is not.
inline void LinearSrgbToOkLab(const float* rgb, float* lab) {
    const float l = 0.4122214708f * rgb[0] + 0.5363325363f * rgb[1] + 0.0514459929f * rgb[2];
    const float m = 0.2119034982f * rgb[0] + 0.6806995451f * rgb[1] + 0.1073969566f * rgb[2];
    const float s = 0.0883024619f * rgb[0] + 0.2817188376f * rgb[1] + 0.6299787005f * rgb[2];
    const float l_ = std::cbrt(l), m_ = std::cbrt(m), s_ = std::cbrt(s);
    lab[0] = 0.2104542553f * l_ + 0.7936177850f * m_ - 0.0040720468f * s_;
    lab[1] = 1.9779984951f * l_ - 2.4285922050f * m_ + 0.4505937099f * s_;
    lab[2] = 0.0259040371f * l_ + 0.7827717662f * m_ - 0.8086757660f * s_;
}
inline void OkLabToLinearSrgb(const float* lab, float* rgb) {
    const float l_ = lab[0] + 0.3963377774f * lab[1] + 0.2158037573f * lab[2];
    const float m_ = lab[0] - 0.1055613458f * lab[1] - 0.0638541728f * lab[2];
    const float s_ = lab[0] - 0.0894841775f * lab[1] - 1.2914855480f * lab[2];
    const float l = l_ * l_ * l_, m = m_ * m_ * m_, s = s_ * s_ * s_;
    rgb[0] =  4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s;
    rgb[1] = -1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s;
    rgb[2] = -0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s;
}

// The same in HLSL, kept beside the C++ so the two cannot drift.
inline constexpr const char* kOkLabHlsl = R"(
float CbrtS(float x) { return sign(x) * pow(abs(x), 1.0 / 3.0); }
float3 LinearSrgbToOkLab(float3 c) {
    float l = 0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b;
    float m = 0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b;
    float s = 0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b;
    float l_ = CbrtS(l), m_ = CbrtS(m), s_ = CbrtS(s);
    return float3(0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
                  1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
                  0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_);
}
float3 OkLabToLinearSrgb(float3 lab) {
    float l_ = lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z;
    float m_ = lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z;
    float s_ = lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z;
    float l = l_ * l_ * l_, m = m_ * m_ * m_, s = s_ * s_ * s_;
    return float3( 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
                  -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
                  -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s);
}
)";

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
