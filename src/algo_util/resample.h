// resample — sampling an image at a fractional position, shared by the stages
// that warp: lens_correct and upright.
//
// Catmull-Rom, with the result held to the range of the four nearest samples:
// the cubic's overshoot is a dark or bright ring along every hard edge, worst
// on a blown light against a night sky, and a resample that only moves the
// image must not add one. Coordinates are in the raster's own pixels with
// pixel centres at integers; samples past the edge clamp to it.
#pragma once

#include <algorithm>
#include <cmath>

namespace tglab {

inline float SampleCubicClamped(const float* d, int w, int h, int ch, int c, double fx, double fy) {
    const int x0 = int(std::floor(fx)), y0 = int(std::floor(fy));
    const float tx = float(fx - x0), ty = float(fy - y0);
    auto wts = [](float t, float* o) {
        const float t2 = t * t, t3 = t2 * t;
        o[0] = 0.5f * (-t3 + 2 * t2 - t);
        o[1] = 0.5f * (3 * t3 - 5 * t2 + 2);
        o[2] = 0.5f * (-3 * t3 + 4 * t2 + t);
        o[3] = 0.5f * (t3 - t2);
    };
    float wx[4], wy[4];
    wts(tx, wx); wts(ty, wy);
    float sum = 0, lo = 3.4e38f, hi = -3.4e38f;
    for (int j = 0; j < 4; ++j) {
        const int yy = std::clamp(y0 - 1 + j, 0, h - 1);
        float row = 0;
        for (int i = 0; i < 4; ++i) {
            const int xx = std::clamp(x0 - 1 + i, 0, w - 1);
            const float v = d[(size_t(yy) * size_t(w) + size_t(xx)) * size_t(ch) + size_t(c)];
            row += wx[i] * v;
            if ((i == 1 || i == 2) && (j == 1 || j == 2)) { lo = std::min(lo, v); hi = std::max(hi, v); }
        }
        sum += wy[j] * row;
    }
    return std::clamp(sum, lo, hi);
}

// The same in HLSL, sampling texture `Src` (which the including shader
// declares) of size Width x Height. Channel `c` of a float4.
inline constexpr const char* kSampleCubicHlsl = R"(
float4 CubicW(float t) {
    float t2 = t * t, t3 = t2 * t;
    return 0.5 * float4(-t3 + 2 * t2 - t, 3 * t3 - 5 * t2 + 2, -3 * t3 + 4 * t2 + t, t3 - t2);
}
float SampleCubic(float2 f, uint c) {
    int2 p0 = int2(floor(f));
    float2 t = f - float2(p0);
    float4 wx = CubicW(t.x), wy = CubicW(t.y);
    float sum = 0, lo = 3.4e38, hi = -3.4e38;
    [unroll] for (int j = 0; j < 4; ++j) {
        int yy = clamp(p0.y - 1 + j, 0, int(Height) - 1);
        float row = 0;
        [unroll] for (int i = 0; i < 4; ++i) {
            int xx = clamp(p0.x - 1 + i, 0, int(Width) - 1);
            float v = Src.Load(int3(xx, yy, 0))[c];
            row += wx[i] * v;
            if ((i == 1 || i == 2) && (j == 1 || j == 2)) { lo = min(lo, v); hi = max(hi, v); }
        }
        sum += wy[j] * row;
    }
    return clamp(sum, lo, hi);
}
)";

}  // namespace tglab
