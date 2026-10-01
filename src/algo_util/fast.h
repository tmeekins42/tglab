// fast — the FAST-9 corner test (Rosten & Drummond), shared by the detectors
// that find corners with it (ORB, BRISK). The GPU pyramid has the same test
// in HLSL; keep the two in step.
#pragma once

namespace tglab {

// The Bresenham circle of radius 3 a FAST test samples, clockwise from the
// top.
inline constexpr int kFastCircleX[16] = { 0,  1,  2,  3,  3,  3,  2,  1,
                                          0, -1, -2, -3, -3, -3, -2, -1};
inline constexpr int kFastCircleY[16] = {-3, -3, -2, -1,  0,  1,  2,  3,
                                          3,  3,  2,  1,  0, -1, -2, -3};

// Is (x, y) a corner at threshold t: nine contiguous circle pixels all
// brighter than the centre by t, or all darker? `img` is anything with
// At(x, y) returning the pixel's value.
//
// The early rejection is the whole reason this is fast. Pixels 0, 4, 8 and 12
// are the compass points; for 9 contiguous of 16 to pass, at least three of
// those four must pass too. Testing them first rejects the great majority of
// pixels after four reads instead of sixteen.
template <typename Img>
bool FastCorner(const Img& img, int x, int y, float t) {
    const float c = img.At(x, y);
    const float hi = c + t, lo = c - t;

    int brightAxis = 0, darkAxis = 0;
    for (int i = 0; i < 16; i += 4) {
        const float p = img.At(x + kFastCircleX[i], y + kFastCircleY[i]);
        if (p > hi) ++brightAxis;
        else if (p < lo) ++darkAxis;
    }
    if (brightAxis < 3 && darkAxis < 3) return false;

    // The full ring, walked twice so a run can wrap around the end.
    int runBright = 0, runDark = 0;
    for (int i = 0; i < 32; ++i) {
        const int k = i & 15;
        const float p = img.At(x + kFastCircleX[k], y + kFastCircleY[k]);
        if (p > hi) { runDark = 0; if (++runBright >= 9) return true; }
        else if (p < lo) { runBright = 0; if (++runDark >= 9) return true; }
        else { runBright = runDark = 0; }
    }
    return false;
}

}  // namespace tglab
