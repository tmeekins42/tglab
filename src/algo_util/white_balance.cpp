#include "white_balance.h"

#include <algorithm>
#include <cmath>

#include "color.h"

namespace tglab {

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

namespace {

// The white of (kelvin, tint) in linear sRGB, at unit luminance, each
// component floored well above zero: see kKelvinMin for what a near-zero
// divisor did.
void KelvinWhite(float kelvin, float tint, float* r, float* g, float* b) {
    float x = 0.0f, y = 0.0f;
    PlanckianXy(kelvin, &x, &y);

    // Tint moves perpendicular to the locus: +y greener, -y magenta.
    y = std::clamp(y - tint * 0.05f, 0.05f, 0.95f);

    // xy -> XYZ at unit luminance.
    const float X = x / y;
    const float Y = 1.0f;
    const float Z = (1.0f - x - y) / y;

    XyzToLinearSrgb(X, Y, Z, r, g, b);
    constexpr float kMinComponent = 0.02f;
    *r = std::max(*r, kMinComponent);
    *g = std::max(*g, kMinComponent);
    *b = std::max(*b, kMinComponent);
}

bool Invert3(const float* m, float* inv) {
    const double a = m[0], b = m[1], c = m[2], d = m[3], e = m[4], f = m[5],
                 g = m[6], h = m[7], k = m[8];
    const double det = a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g);
    if (std::abs(det) < 1e-9) return false;
    const double r = 1.0 / det;
    inv[0] = float((e * k - f * h) * r); inv[1] = float((c * h - b * k) * r); inv[2] = float((b * f - c * e) * r);
    inv[3] = float((f * g - d * k) * r); inv[4] = float((a * k - c * g) * r); inv[5] = float((c * d - a * f) * r);
    inv[6] = float((d * h - e * g) * r); inv[7] = float((b * g - a * h) * r); inv[8] = float((a * e - b * d) * r);
    return true;
}

}  // namespace

void KelvinGains(float kelvin, float tint, float* gr, float* gg, float* gb) {
    float r = 0.0f, g = 0.0f, b = 0.0f;
    KelvinWhite(kelvin, tint, &r, &g, &b);
    *gr = g / r;
    *gg = 1.0f;
    *gb = g / b;
}

bool CameraWhiteMultipliers(const ImageDesc& d, float kelvin, float tint, float* mul) {
    if (!d.hasDaylightWb) return false;
    float inv[9];
    if (!Invert3(d.rgbCam, inv)) return false;
    float s[3];
    KelvinWhite(kelvin, tint, &s[0], &s[1], &s[2]);
    // The illuminant's white in the camera's daylight-balanced space: rgbCam
    // maps daylight-balanced camera RGB to sRGB, so its inverse takes the white
    // back. Divided by the daylight gains, that is what the sensor itself
    // records under this light -- and the multipliers are its reciprocal.
    float c[3], top = 0.0f;
    for (int i = 0; i < 3; ++i) {
        c[i] = inv[i * 3] * s[0] + inv[i * 3 + 1] * s[1] + inv[i * 3 + 2] * s[2];
        top = std::max(top, c[i]);
    }
    if (top <= 0.0f) return false;
    for (int i = 0; i < 3; ++i) mul[i] = d.preMul[i] / std::max(c[i], 0.02f * top);
    const float g = std::max(mul[1], 1e-9f);
    for (int i = 0; i < 3; ++i) mul[i] /= g;
    return true;
}

bool CameraWhiteBalanceMatrix(const ImageDesc& d, float kelvin, float tint, float* m) {
    float mul[3], inv[9];
    if (!CameraWhiteMultipliers(d, kelvin, tint, mul) || !Invert3(d.rgbCam, inv)) return false;
    float g[3];
    for (int i = 0; i < 3; ++i) g[i] = mul[i] / std::max(d.camMul[i], 1e-6f);
    // rgbCam * diag(g) * inv(rgbCam).
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) {
            float v = 0.0f;
            for (int k = 0; k < 3; ++k) v += d.rgbCam[r * 3 + k] * g[k] * inv[k * 3 + c];
            m[r * 3 + c] = v;
        }
    // Normalised on the luminance it gives white, so a change of white
    // balance is not also a change of exposure.
    float w[3];
    for (int r = 0; r < 3; ++r) w[r] = m[r * 3] + m[r * 3 + 1] + m[r * 3 + 2];
    const float lum = Luma(w[0], w[1], w[2]);
    if (lum <= 1e-6f) return false;
    for (int i = 0; i < 9; ++i) m[i] /= lum;
    return true;
}

void AsShotWhiteBalance(const ImageDesc& d, float* kelvin, float* tint) {
    *kelvin = 0.0f;
    *tint   = 0.0f;
    if (!d.hasDaylightWb) return;

    // The camera multipliers each (kelvin, tint) implies, in the same model
    // basic_adjust applies -- so feeding the answer back is the identity.
    auto model = [&](float k, float t, float* mr, float* mb) {
        float mul[3] = {1.0f, 1.0f, 1.0f};
        CameraWhiteMultipliers(d, k, t, mul);
        *mr = mul[0]; *mb = mul[2];
    };
    const float camR = std::max(d.camMul[0], 1e-6f), camB = std::max(d.camMul[2], 1e-6f);

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
        const float target = camR / camB;
        float lo = kKelvinMin, hi = kKelvinMax;
        for (int i = 0; i < 32; ++i) {
            const float mid = 0.5f * (lo + hi);
            float mr, mb;
            model(mid, t, &mr, &mb);
            const float ratio = mr / std::max(mb, 1e-6f);
            if (ratio < target) lo = mid; else hi = mid;
        }
        k = 0.5f * (lo + hi);

        // Tint, by bisection on the green level at the new temperature. The
        // camera's green gain is 1 by normalisation, so the target is how far
        // the current pair leaves red and blue from where it put them.
        float loT = -1.0f, hiT = 1.0f;
        for (int i = 0; i < 32; ++i) {
            const float midT = 0.5f * (loT + hiT);
            float mr, mb;
            model(k, midT, &mr, &mb);
            const float offR = mr / camR;
            const float offB = mb / camB;
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
