#include "white_balance.h"

#include <algorithm>
#include <cmath>

namespace tglab {
namespace {

// sRGB/Rec.709 primaries with a D65 white: the matrix taking XYZ to linear sRGB.
inline void XyzToLinearSrgb(float X, float Y, float Z, float* r, float* g, float* b) {
    *r =  3.2404542f * X - 1.5371385f * Y - 0.4985314f * Z;
    *g = -0.9692660f * X + 1.8760108f * Y + 0.0415560f * Z;
    *b =  0.0556434f * X - 0.2040259f * Y + 1.0572252f * Z;
}

} // namespace

void PlanckianXy(float kelvin, float* outX, float* outY) {
    const float T = std::clamp(kelvin, 1667.0f, 25000.0f);
    const float t1 = 1e3f / T, t2 = 1e6f / (T * T), t3 = 1e9f / (T * T * T);

    float x;
    if (T <= 4000.0f) x = -0.2661239f * t3 - 0.2343589f * t2 + 0.8776956f * t1 + 0.179910f;
    else              x = -3.0258469f * t3 + 2.1070379f * t2 + 0.2226347f * t1 + 0.240390f;

    const float x2 = x * x, x3 = x2 * x;
    float y;
    if (T <= 2222.0f)      y = -1.1063814f * x3 - 1.34811020f * x2 + 2.18555832f * x - 0.20219683f;
    else if (T <= 4000.0f) y = -0.9549476f * x3 - 1.37418593f * x2 + 2.09137015f * x - 0.16748867f;
    else                   y =  3.0817580f * x3 - 5.87338670f * x2 + 3.75112997f * x - 0.37001483f;

    *outX = x;
    *outY = y;
}

void KelvinGains(float kelvin, float tint, float* gr, float* gg, float* gb) {
    float x = 0.0f, y = 0.0f;
    PlanckianXy(kelvin, &x, &y);

    // Tint moves perpendicular to the locus: +y greener, -y magenta.
    y = std::clamp(y - tint * 0.05f, 0.05f, 0.95f);

    // xy -> XYZ at unit luminance.
    const float X = x / y;
    const float Y = 1.0f;
    const float Z = (1.0f - x - y) / y;

    float r = 0.0f, g = 0.0f, b = 0.0f;
    XyzToLinearSrgb(X, Y, Z, &r, &g, &b);

    // Floored well above zero rather than at an epsilon: see kKelvinMin for
    // what a near-zero divisor did here.
    constexpr float kMinComponent = 0.02f;
    r = std::max(r, kMinComponent);
    g = std::max(g, kMinComponent);
    b = std::max(b, kMinComponent);

    *gr = g / r;
    *gg = 1.0f;
    *gb = g / b;
}

void AsShotWhiteBalance(const ImageDesc& d, float* kelvin, float* tint) {
    *kelvin = 0.0f;
    *tint   = 0.0f;
    if (!d.hasDaylightWb) return;

    // What the camera applied, relative to its own daylight reference.
    const float shotR = d.camMul[0] / std::max(d.preMul[0], 1e-6f);
    const float shotB = d.camMul[2] / std::max(d.preMul[2], 1e-6f);

    float dr = 1.0f, dg = 1.0f, db = 1.0f;
    KelvinGains(kD65Kelvin, 0.0f, &dr, &dg, &db);

    // The two axes are NOT independent: moving tint shifts y, which changes the
    // R/B ratio as well as the green level -- measured, tint +0.6 moves R/B by
    // 12% at 4000 K. Solving for temperature first and then tint therefore does
    // not converge on the right pair; my first version did exactly that and left
    // the round trip 8% off.
    //
    // So alternate. Each pass re-solves temperature for the current tint and
    // then re-solves tint for the new temperature; the coupling is weak enough
    // that a handful of passes settles it to well under a percent.
    float k = 5000.0f;
    float t = 0.0f;

    for (int pass = 0; pass < 8; ++pass) {
        // Temperature, by bisection on R/B at the current tint. Monotonic in k,
        // so this needs no convergence criterion.
        const float target = shotR / std::max(shotB, 1e-6f);
        float lo = kKelvinMin, hi = kKelvinMax;
        for (int i = 0; i < 32; ++i) {
            const float mid = 0.5f * (lo + hi);
            float wr = 1.0f, wg = 1.0f, wb = 1.0f;
            KelvinGains(mid, t, &wr, &wg, &wb);
            const float ratio = (wr / std::max(dr, 1e-6f)) /
                                std::max(wb / std::max(db, 1e-6f), 1e-6f);
            if (ratio < target) lo = mid; else hi = mid;
        }
        k = 0.5f * (lo + hi);

        // Tint, by bisection on the green level at the new temperature. The
        // camera's green gain is 1 by normalisation, so the target is how far
        // the current pair leaves red and blue from where it put them.
        float loT = -1.0f, hiT = 1.0f;
        for (int i = 0; i < 32; ++i) {
            const float midT = 0.5f * (loT + hiT);
            float wr = 1.0f, wg = 1.0f, wb = 1.0f;
            KelvinGains(k, midT, &wr, &wg, &wb);
            const float offR = (wr / std::max(dr, 1e-6f)) / std::max(shotR, 1e-6f);
            const float offB = (wb / std::max(db, 1e-6f)) / std::max(shotB, 1e-6f);
            const float green = 0.5f * (offR + offB);
            // Green above 1 means the pair is too green, which more tint fixes.
            if (green > 1.0f) loT = midT; else hiT = midT;
        }
        t = 0.5f * (loT + hiT);
    }

    *kelvin = k;
    *tint   = std::clamp(t, -1.0f, 1.0f);
}

} // namespace tglab
