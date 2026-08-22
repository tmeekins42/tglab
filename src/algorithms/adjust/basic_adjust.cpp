// basic_adjust — the standard photographic controls, in one pass.
//
// Deliberately one algorithm rather than ten, which is against this lab's usual
// grain. The reason is measured: chained GPU stages are bandwidth-bound, not
// compute-bound. At 36 MP a *trivial* one-fetch-per-pixel kernel still costs
// ~50 ms per stage, because each stage reads and writes a full 144 MB
// intermediate; a kernel doing nine times the arithmetic costs only ~20% more.
// Ten chained adjustments would move ~2.9 GB and cost ~500 ms per slider nudge.
//
// Fused, the image is read once and written once -- ~290 MB -- and all ten
// adjustments happen in registers, where they are effectively free. That is a
// ~10x difference, and it comes from the memory traffic rather than the
// instruction count.
//
// These controls are also settled: they are the same operations in every raw
// developer, and are not what this lab exists to experiment with. Anything
// genuinely experimental should stay a separate algorithm, where choose() and
// the compare panel can get at it.
//
// Order matters and follows the conventional raw-developer sequence:
//
//   white balance -> exposure -> highlights/shadows -> whites/blacks
//   -> contrast -> vibrance/saturation
//
// White balance comes first because it is a property of the capture, not a
// look. Tonal recovery precedes contrast so that contrast operates on already-
// recovered detail. Saturation comes last so it sees final luminance.
//
// Everything happens in LINEAR light. The image arrives gamma-encoded (sRGB),
// so the shader decodes on read and re-encodes on write. This is not
// pedantry: exposure is a multiply, and multiplying gamma-encoded values gives
// the wrong answer -- highlights roll off incorrectly and colours shift as they
// brighten. Doing it right costs two transfer functions per pixel, which is
// nothing when the cost is bandwidth.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"

namespace tglab {
namespace {

// sRGB transfer functions. The piecewise form, not the 2.2 power
// approximation: the linear toe matters for shadow adjustments, which is
// exactly where the two disagree most.
inline float SrgbToLinear(float c) {
    return (c <= 0.04045f) ? c / 12.92f
                           : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

inline float LinearToSrgb(float c) {
    c = std::clamp(c, 0.0f, 1.0f);
    return (c <= 0.0031308f) ? c * 12.92f
                             : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
}

// Scene-linear input (a demosaiced raw) is ALREADY linear, so the transfer
// functions must not be applied to it -- and its values must not be clamped on
// write. Both of those were being done unconditionally, which decoded linear
// data as though it were gamma-encoded and then threw away every value above
// 1.0. On a bright frame that is most of a stop of highlight detail: measured
// 1.77 coming out of the demosaic, clamped to 0.9995 on the way out.
//
// The linear path stays linear end to end. Whatever displays or writes the
// image applies a transfer function at that point, which is where it belongs.
inline float DecodeIn(float c, bool linear) {
    return linear ? c : SrgbToLinear(c);
}

inline float EncodeOut(float c, bool linear) {
    // No clamp on the linear path -- the headroom is the point.
    return linear ? std::max(c, 0.0f) : LinearToSrgb(c);
}

// Rec. 709 luminance, matching the primaries the image is already in.
inline float Luma(float r, float g, float b) {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

// Where the highlight band tops out for scene-linear input.
//
// NOT the image's peak, and not 1.0. Measured across six raws from two bodies,
// peak *luminance* after demosaic ran 0.25 to 0.75 -- every one of them below
// the 1.0 the band previously assumed, so their highlights sat in the band's
// dead zone and the slider did nothing. That is the "ranges are too small"
// symptom: the control was fine, the band it drove was pitched for
// gamma-encoded data.
//
// Deliberately a constant rather than per-image: a peak-relative band would
// mean something different on every frame, so the same slider position would
// give different results shot to shot. Pitched here, a highlight at 0.35 gets
// partial correction and anything at or above 0.7 gets the full amount, which
// covers the observed range with room above it.
//
// Note that a single saturated *channel* is not a bright pixel: one frame here
// peaks at 1.82 in blue while its luminance never exceeds 0.25. That is a
// colour clip, not a blown highlight, and a luminance-driven band correctly
// leaves it alone.
constexpr float kLinearShoulder = 0.70f;

// The white point the sRGB primaries assume, and therefore the reference the
// camera daylight gains are expressed against.
constexpr float kD65Kelvin = 6504.0f;

// A smooth 0..1 window used to target a tonal band without a hard edge, which
// would show as banding on a gradient.
inline float SmoothBand(float x, float lo, float hi) {
    const float t = std::clamp((x - lo) / (hi - lo), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}


// --- colour temperature ------------------------------------------------------

// The chromaticity of a black body at `kelvin`, as CIE xy.
//
// Kim et al.'s cubic approximation of the Planckian locus, the same one every
// raw developer uses. Accurate to within a few units of xy over 1667..25000 K,
// which is far finer than anyone can see, and vastly cheaper than integrating
// Planck's law per slider tick.
inline void PlanckianXy(float kelvin, float* outX, float* outY) {
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

// sRGB/Rec.709 primaries with a D65 white, as the matrix taking linear sRGB to
// XYZ. Its inverse takes XYZ back.
inline void XyzToLinearSrgb(float X, float Y, float Z, float* r, float* g, float* b) {
    *r =  3.2404542f * X - 1.5371385f * Y - 0.4985314f * Z;
    *g = -0.9692660f * X + 1.8760108f * Y + 0.0415560f * Z;
    *b =  0.0556434f * X - 0.2040259f * Y + 1.0572252f * Z;
}

// The per-channel gains that render an illuminant at `kelvin` (offset along the
// green/magenta axis by `tint`) as neutral.
//
// The illuminant's chromaticity becomes an sRGB triple; dividing by it is what
// makes that colour come out grey. Normalised on green, so the control does not
// double as an exposure slider.
inline void KelvinGains(float kelvin, float tint, float* gr, float* gg, float* gb) {
    float x = 0.0f, y = 0.0f;
    PlanckianXy(kelvin, &x, &y);

    // Tint moves perpendicular to the locus: +y is greener, -y magenta. The
    // scale is chosen so the slider's -1..1 covers the useful range without
    // leaving the region where the cubic above is meaningful.
    y = std::clamp(y - tint * 0.05f, 0.05f, 0.95f);
    if (y < 1e-4f) y = 1e-4f;

    // xy -> XYZ at unit luminance.
    const float X = x / y;
    const float Y = 1.0f;
    const float Z = (1.0f - x - y) / y;

    float r, g, b;
    XyzToLinearSrgb(X, Y, Z, &r, &g, &b);

    // Below about 1900 K the Planckian locus leaves the sRGB gamut and the blue
    // component goes NEGATIVE -- measured, b is -0.036 at 1700 K and only turns
    // positive around 1900 K. Clamping it to a small epsilon and dividing
    // produced a blue gain of 32x at 2000 K and broke monotonicity: the red gain
    // stopped falling as the temperature rose, so dragging the slider one way
    // moved the colour back the other.
    //
    // Floored at a value that keeps the ratio sane rather than at zero. The
    // parameter range starts at 2000 K for the same reason, so this is the last
    // line of defence rather than the normal path.
    constexpr float kMinComponent = 0.02f;
    r = std::max(r, kMinComponent);
    g = std::max(g, kMinComponent);
    b = std::max(b, kMinComponent);

    *gr = g / r;
    *gg = 1.0f;
    *gb = g / b;
}
} // namespace

class BasicAdjust : public AlgorithmBase {
public:
    const char* Name()     const override { return "basic_adjust"; }
    const char* Category() const override { return "adjust"; }

    PortList Inputs()  const override { return {{"src", DataType::Image, FormatSpec::Any}}; }
    PortList Outputs() const override { return {{"out", DataType::Image, FormatSpec::SameAsInput}}; }

    void RunCPU(RunCtx& ctx) override {
        const ImageView src = ctx.In(0);
        ImageView       dst = ctx.Out(0);
        if (!src.Valid() || !dst.Valid()) return;

        m_in.Unpack(src);
        if (!m_in.Valid()) return;
        m_out.AllocLike(m_in);

        const int w = m_in.Width(), h = m_in.Height(), ch = m_in.Channels();
        const float scale = m_in.ValueScale();
        const bool  linear = src.desc.linear;
        CaptureWhiteBalance(src.desc);

        // Where the highlight band tops out.
        //
        // For gamma-encoded input this is 1.0, the definitional maximum. For
        // scene-linear input it is NOT the image's peak: three frames from the
        // same camera measured peaks of 0.39, 0.86 and 1.06, so a
        // peak-relative band would mean something different on every shot and
        // the slider would behave differently frame to frame.
        //
        // kLinearShoulder instead marks where highlights *begin* in linear
        // light -- above middle grey, around a stop under diffuse white. The
        // band saturates there and stays saturated for everything brighter, so
        // detail at 1.8 gets the full correction rather than being lumped in
        // with detail at 1.0. That is what makes highlight recovery reach the
        // headroom a raw actually carries.
        const float white = linear ? kLinearShoulder : 1.0f;

        // White balance gains, derived once rather than per pixel.
        float wbR = 1.0f, wbG = 1.0f, wbB = 1.0f;
        WhiteBalanceGains(&wbR, &wbG, &wbB);
        const float expGain = std::pow(2.0f, float(m_exposure));

        for (int y = 0; y < h; ++y) {
            if (ctx.Cancelled()) return;
            for (int x = 0; x < w; ++x) {
                const float* p = m_in.At(x, y);

                // A single-channel image has no colour to balance; treat it as
                // grey so the tonal controls still work.
                float r, g, b;
                if (ch == 1) {
                    r = g = b = DecodeIn(p[0] / scale, linear);
                } else {
                    r = DecodeIn(p[0] / scale, linear);
                    g = DecodeIn(p[1] / scale, linear);
                    b = DecodeIn(p[2] / scale, linear);
                }

                Apply(&r, &g, &b, wbR, wbG, wbB, expGain, white);

                if (ch == 1) {
                    m_out.Set(x, y, 0, EncodeOut(Luma(r, g, b), linear) * scale);
                } else {
                    m_out.Set(x, y, 0, EncodeOut(r, linear) * scale);
                    m_out.Set(x, y, 1, EncodeOut(g, linear) * scale);
                    m_out.Set(x, y, 2, EncodeOut(b, linear) * scale);
                    if (ch == 4) m_out.Set(x, y, 3, p[3]);
                }
            }
        }

        m_out.PackInto(dst);
    }

    // --- GPU implementation -------------------------------------------------
    // The whole point of the algorithm: one read, one write, ten adjustments in
    // between. See the file header for the measurements.
    bool HasGPU() const override { return true; }

    const char* GpuSource() const override {
        return R"(
Texture2D<float4>   Src : register(t0);
RWTexture2D<float4> Dst : register(u0);

cbuffer Params : register(b0) {
    uint  Width;
    uint  Height;
    uint  WbR, WbG, WbB;        // precomputed white-balance gains
    uint  ExpGain;              // 2^exposure
    uint  Contrast;
    uint  Highlights, Shadows;
    uint  Whites, Blacks;
    uint  Vibrance, Saturation;
    uint  Linear;               // 1 = scene-linear input: no transfer, no clamp
    uint  HighlightTop;         // top of the highlight band (1.0 unless linear)
};

static const float3 kLumaW = float3(0.2126, 0.7152, 0.0722);

float3 SrgbToLinear(float3 c) {
    // select(), not ?: -- SM 6.x requires it for a per-component condition.
    return select(c <= 0.04045, c / 12.92, pow((c + 0.055) / 1.055, 2.4));
}
float3 LinearToSrgb(float3 c) {
    c = saturate(c);
    return select(c <= 0.0031308, c * 12.92, 1.055 * pow(c, 1.0 / 2.4) - 0.055);
}
float SmoothBand(float x, float lo, float hi) {
    float t = saturate((x - lo) / (hi - lo));
    return t * t * (3.0 - 2.0 * t);
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;

    float4 texel = Src[int2(tid.xy)];

    // Scene-linear input is already linear -- decoding it again would be wrong.
    float3 c = (Linear != 0) ? texel.rgb : SrgbToLinear(texel.rgb);

    // --- white balance ---
    c *= float3(asfloat(WbR), asfloat(WbG), asfloat(WbB));

    // --- exposure: a straight multiply, which is only correct in linear ---
    c *= asfloat(ExpGain);

    // --- highlights / shadows ---
    // Each targets a tonal band via a smooth window, so the correction fades
    // out rather than leaving a visible edge where it stops applying.
    float lum = dot(c, kLumaW);

    float hi = asfloat(Highlights);
    if (abs(hi) > 1e-4) {
        float wp = asfloat(HighlightTop);
        float w = SmoothBand(lum, 0.35, wp);

        // Pull each channel toward the band top rather than scaling all three.
        // See the CPU path: a uniform multiply preserves channel ratios, so a
        // saturated highlight only darkens and never desaturates -- which reads
        // as colour crushing rather than recovery.
        float amount = -hi * w;
        if (amount > 0.0) {
            c -= max(c - wp, 0.0) * amount;
            c *= (1.0 - 0.35 * amount);
        } else if (amount < 0.0) {
            c *= (1.0 - amount);
        }
    }
    float sh = asfloat(Shadows);
    if (abs(sh) > 1e-4) {
        float w = 1.0 - SmoothBand(lum, 0.0, 0.5);
        c *= (1.0 + sh * w);
    }

    // --- whites / blacks: the endpoints, not bands ---
    float wh = asfloat(Whites);
    if (abs(wh) > 1e-4) c *= (1.0 + wh * saturate(dot(c, kLumaW)));
    float bl = asfloat(Blacks);
    // Shifts the black point: negative crushes, positive lifts.
    if (abs(bl) > 1e-4) c = max(c + bl * 0.1 * (1.0 - saturate(dot(c, kLumaW))), 0.0);

    // --- contrast, pivoted on middle grey (0.18 in linear) ---
    float ct = asfloat(Contrast);
    if (abs(ct) > 1e-4) c = max((c - 0.18) * (1.0 + ct) + 0.18, 0.0);

    // --- vibrance / saturation ---
    lum = dot(c, kLumaW);
    float sat = asfloat(Saturation);
    float vib = asfloat(Vibrance);
    if (abs(vib) > 1e-4) {
        // Vibrance is saturation weighted by how unsaturated a pixel already
        // is, so muted colours move and already-vivid ones are left alone --
        // which is what keeps skin tones from going lurid.
        float mx = max(c.r, max(c.g, c.b));
        float mn = min(c.r, min(c.g, c.b));
        float current = (mx > 1e-6) ? (mx - mn) / mx : 0.0;
        c = lerp(lum.xxx, c, 1.0 + vib * (1.0 - current));
    }
    if (abs(sat) > 1e-4) c = lerp(lum.xxx, c, 1.0 + sat);

    // On the linear path the values stay linear AND unclamped: the headroom
    // above 1.0 is exactly what the highlight controls need to recover.
    c = max(c, 0.0);
    Dst[tid.xy] = float4((Linear != 0) ? c : LinearToSrgb(c), texel.a);
}
)";
    }

    std::vector<uint32_t> GpuConstants(int) const override {
        float wbR = 1.0f, wbG = 1.0f, wbB = 1.0f;
        WhiteBalanceGains(&wbR, &wbG, &wbB);

        auto bits = [](float f) {
            uint32_t u;
            std::memcpy(&u, &f, sizeof(u));
            return u;
        };
        return {bits(wbR), bits(wbG), bits(wbB),
                bits(std::pow(2.0f, float(m_exposure))),
                bits(float(m_contrast)),
                bits(float(m_highlights)), bits(float(m_shadows)),
                bits(float(m_whites)),     bits(float(m_blacks)),
                bits(float(m_vibrance)),   bits(float(m_saturation)),
                uint32_t(m_linear ? 1 : 0), bits(m_white)};
    }

    // HasGPU() is consulted before PrepareGpu(), so this is where the input
    // descriptor first becomes available -- the GPU path never calls RunCPU().
    // The white-balance references the Kelvin control needs. Shared by both
    // paths so they cannot disagree.
    void CaptureWhiteBalance(const ImageDesc& d) {
        for (int i = 0; i < 3; ++i) {
            m_camMul[i] = d.camMul[i];
            m_preMul[i] = d.preMul[i];
        }
        m_hasDaylightWb = d.hasDaylightWb;
    }

    void PrepareGpu(const std::vector<ImageDesc>& inputs) override {
        m_linear = !inputs.empty() && inputs[0].linear;
        m_white  = m_linear ? kLinearShoulder : 1.0f;
        if (!inputs.empty()) CaptureWhiteBalance(inputs[0]);
    }

private:
    // The white-balance gains, from the absolute Kelvin control and the
    // relative nudge together.
    //
    // The Kelvin part answers "what light was this shot under?", and needs two
    // things from the file: the as-shot gains the camera chose (camMul) and the
    // camera's daylight reference (preMul). The demosaic has already applied
    // camMul, so the image arrives balanced as the camera saw it -- meaning a
    // Kelvin setting must be applied RELATIVE to that, not from scratch. Undo
    // the camera's choice, apply the requested illuminant, and the two ratios
    // cancel to exactly 1.0 when the request matches the shot.
    //
    // Without preMul there is no way to know where neutral is, so kelvin does
    // nothing and only the relative control applies. That is the honest
    // behaviour for a JPEG, which carries no such record.
    //
    // Gains are normalised on luminance, so changing the white balance does not
    // double as an exposure change.
    void WhiteBalanceGains(float* r, float* g, float* b) const {
        *r = 1.0f;
        *g = 1.0f;
        *b = 1.0f;

        const float k = float(m_kelvin);
        if (k > 1.0f && m_hasDaylightWb) {
            // The illuminant the user asked for, and the one the camera assumed,
            // both expressed as gains on this sensor. preMul is the camera's
            // D65 reference, so scaling it by the ratio of the requested
            // illuminant to D65 gives the requested illuminant in camera space.
            float wr = 1.0f, wg = 1.0f, wb = 1.0f;
            KelvinGains(k, float(m_tint), &wr, &wg, &wb);

            float dr = 1.0f, dg = 1.0f, db = 1.0f;
            KelvinGains(kD65Kelvin, 0.0f, &dr, &dg, &db);

            // What the camera already applied, relative to its daylight
            // reference: this is the correction to undo.
            const float shotR = m_camMul[0] / std::max(m_preMul[0], 1e-6f);
            const float shotB = m_camMul[2] / std::max(m_preMul[2], 1e-6f);

            // ...and what the requested illuminant asks for instead.
            const float wantR = wr / std::max(dr, 1e-6f);
            const float wantB = wb / std::max(db, 1e-6f);

            *r = wantR / std::max(shotR, 1e-6f);
            *b = wantB / std::max(shotB, 1e-6f);
        }

        // The relative nudge, on top of whatever the above decided. Red and
        // blue in opposition for temperature; green against magenta for tint.
        const float t = float(m_temperature);
        const float n = float(m_tint);
        *r *= 1.0f + t * 0.4f;
        *b *= 1.0f - t * 0.4f;

        // Tint is folded into the Kelvin path when that is active (it shifts the
        // illuminant off the locus), so apply it here only when it is not.
        if (!(k > 1.0f && m_hasDaylightWb)) {
            *g *= 1.0f - n * 0.3f;
            *r *= 1.0f + n * 0.15f;
            *b *= 1.0f + n * 0.15f;
        }

        *r = std::max(*r, 0.0f);
        *g = std::max(*g, 0.0f);
        *b = std::max(*b, 0.0f);

        const float lum = Luma(*r, *g, *b);
        if (lum > 1e-6f) {
            *r /= lum;
            *g /= lum;
            *b /= lum;
        }
    }
    // The CPU path, matching the shader step for step. Kept in sync because
    // compare mode checks one against the other.
    //
    // `white` is where the highlight band tops out; see kLinearShoulder and
    // the caller for why it is not 1.0 on scene-linear input.

    void Apply(float* rr, float* gg, float* bb,
               float wbR, float wbG, float wbB, float expGain, float white) const {
        float r = *rr * wbR * expGain;
        float g = *gg * wbG * expGain;
        float b = *bb * wbB * expGain;

        float lum = Luma(r, g, b);

        const float hi = float(m_highlights);
        if (std::abs(hi) > 1e-4f) {
            const float w = SmoothBand(lum, 0.35f, white);

            // Recovery pulls each channel toward the band's top, rather than
            // scaling all three by one factor.
            //
            // A uniform multiply preserves the ratios between channels, so a
            // saturated highlight only gets darker -- measured, a 1.20/0.35/0.20
            // red held saturation 0.833 at every slider position, coming out
            // dark and MORE colour-dense than its surroundings. That is the
            // colour crushing, and it is what "recovery" must not do: a blown
            // highlight is blown because one channel clipped, so bringing it
            // back means moving that channel down toward the others.
            //
            // Pulling toward `white` does exactly that. A channel already below
            // the target is left alone (max with 0), so a dark blue in a bright
            // red does not get dragged upward.
            const float amount = -hi * w;   // hi is negative when recovering
            if (amount > 0.0f) {
                r -= std::max(r - white, 0.0f) * amount;
                g -= std::max(g - white, 0.0f) * amount;
                b -= std::max(b - white, 0.0f) * amount;

                // Below the top there is nothing clipped to pull back, so fall
                // back to a gentle overall darkening -- which is what the
                // control means for a merely-bright region.
                const float k = 1.0f - 0.35f * amount;
                r *= k; g *= k; b *= k;
            } else if (amount < 0.0f) {
                // Positive highlights: push the band up, ratios preserved.
                const float k = 1.0f - amount;
                r *= k; g *= k; b *= k;
            }
        }
        const float sh = float(m_shadows);
        if (std::abs(sh) > 1e-4f) {
            const float w = 1.0f - SmoothBand(lum, 0.0f, 0.5f);
            const float k = 1.0f + sh * w;
            r *= k; g *= k; b *= k;
        }

        const float wh = float(m_whites);
        if (std::abs(wh) > 1e-4f) {
            const float k = 1.0f + wh * std::clamp(Luma(r, g, b), 0.0f, 1.0f);
            r *= k; g *= k; b *= k;
        }
        const float bl = float(m_blacks);
        if (std::abs(bl) > 1e-4f) {
            const float k = bl * 0.1f * (1.0f - std::clamp(Luma(r, g, b), 0.0f, 1.0f));
            r = std::max(r + k, 0.0f);
            g = std::max(g + k, 0.0f);
            b = std::max(b + k, 0.0f);
        }

        const float ct = float(m_contrast);
        if (std::abs(ct) > 1e-4f) {
            r = std::max((r - 0.18f) * (1.0f + ct) + 0.18f, 0.0f);
            g = std::max((g - 0.18f) * (1.0f + ct) + 0.18f, 0.0f);
            b = std::max((b - 0.18f) * (1.0f + ct) + 0.18f, 0.0f);
        }

        lum = Luma(r, g, b);
        const float vib = float(m_vibrance);
        if (std::abs(vib) > 1e-4f) {
            const float mx = std::max(r, std::max(g, b));
            const float mn = std::min(r, std::min(g, b));
            const float current = (mx > 1e-6f) ? (mx - mn) / mx : 0.0f;
            const float k = 1.0f + vib * (1.0f - current);
            r = lum + (r - lum) * k;
            g = lum + (g - lum) * k;
            b = lum + (b - lum) * k;
        }
        const float sat = float(m_saturation);
        if (std::abs(sat) > 1e-4f) {
            const float k = 1.0f + sat;
            r = lum + (r - lum) * k;
            g = lum + (g - lum) * k;
            b = lum + (b - lum) * k;
        }

        *rr = std::max(r, 0.0f);
        *gg = std::max(g, 0.0f);
        *bb = std::max(b, 0.0f);
    }

    // 0 means "leave the camera's own white balance alone", which is why this
    // is not simply defaulted to 5500: a raw arrives already balanced as shot,
    // and a control that jumped to daylight the moment the image loaded would
    // change every picture before the user touched anything.
    Param<float> m_kelvin{
        this, "kelvin", 0.0f, 0.0f, 25000.0f,
        {.help = "Absolute colour temperature of the light the scene was under, "
                 "in Kelvin -- 2800 tungsten, 5500 daylight, 7000 shade. Setting "
                 "it to the ACTUAL light neutralises the cast: a photograph shot "
                 "under tungsten needs ~2800 here, not a cooler number. 0 keeps "
                 "the camera's own white balance. Needs a raw file; a JPEG has "
                 "no record of where neutral was.",
         .step = 50.0, .softMin = 2000.0, .softMax = 10000.0}};

    Param<float> m_temperature{
        this, "temperature", 0.0f, -1.0f, 1.0f,
        {.help = "Warm/cool nudge, relative to whatever white balance is already "
                 "in effect -- the camera's, or the one `kelvin` set. Positive "
                 "warms (more red, less blue). Use this for a small correction "
                 "by eye; use kelvin to name the actual illuminant.",
         .step = 0.01, .softMin = -0.5, .softMax = 0.5}};
    Param<float> m_tint{
        this, "tint", 0.0f, -1.0f, 1.0f,
        {.help = "Green/magenta balance, the axis perpendicular to temperature. "
                 "Negative is greener, positive more magenta.",
         .step = 0.01, .softMin = -0.5, .softMax = 0.5}};
    Param<float> m_exposure{
        this, "exposure", 0.0f, -5.0f, 5.0f,
        {.help = "Overall brightness in stops: +1 doubles the light, -1 halves "
                 "it. Applied in linear light, so it behaves like the camera's "
                 "own exposure rather than a brightness slider.",
         .step = 0.05, .softMin = -3.0, .softMax = 3.0}};
    Param<float> m_contrast{
        this, "contrast", 0.0f, -1.0f, 1.0f,
        {.help = "Spread of tones around middle grey. Positive deepens shadows "
                 "and brightens highlights; negative flattens the image.",
         .step = 0.01, .softMin = -0.5, .softMax = 0.5}};
    Param<float> m_highlights{
        this, "highlights", 0.0f, -1.0f, 1.0f,
        {.help = "Brightness of the upper tones only. NEGATIVE recovers blown "
                 "highlights, which is the usual direction; positive pushes "
                 "them further.",
         .step = 0.01, .softMin = -1.0, .softMax = 0.5}};
    Param<float> m_shadows{
        this, "shadows", 0.0f, -1.0f, 1.0f,
        {.help = "Brightness of the lower tones only. POSITIVE opens up shadow "
                 "detail; negative deepens it.",
         .step = 0.01, .softMin = -0.5, .softMax = 1.0}};
    Param<float> m_whites{
        this, "whites", 0.0f, -1.0f, 1.0f,
        {.help = "The white point -- where the brightest tones land. Sets how "
                 "far the top of the range extends, as opposed to highlights, "
                 "which reshapes tones already up there.",
         .step = 0.01, .softMin = -0.5, .softMax = 0.5}};
    Param<float> m_blacks{
        this, "blacks", 0.0f, -1.0f, 1.0f,
        {.help = "The black point. Negative crushes shadows towards pure black "
                 "for a deeper image; positive lifts them for a faded look.",
         .step = 0.01, .softMin = -0.5, .softMax = 0.5}};
    Param<float> m_vibrance{
        this, "vibrance", 0.0f, -1.0f, 1.0f,
        {.help = "Saturation weighted towards muted colours: already-vivid "
                 "areas are left alone. Safer than saturation on skin tones.",
         .step = 0.01, .softMin = -0.5, .softMax = 1.0}};
    Param<float> m_saturation{
        this, "saturation", 0.0f, -1.0f, 1.0f,
        {.help = "Colour intensity, applied evenly. -1 is fully greyscale.",
         .step = 0.01, .softMin = -1.0, .softMax = 1.0}};

    bool  m_linear = false;   // set by PrepareGpu from the input descriptor
    float m_white  = 1.0f;

    // White-balance references from the input descriptor, needed by the Kelvin
    // control. Set in both paths -- RunCPU reads the descriptor directly, and
    // PrepareGpu is where the GPU path first sees it.
    float m_camMul[3]     = {1.0f, 1.0f, 1.0f};
    float m_preMul[3]     = {1.0f, 1.0f, 1.0f};
    bool  m_hasDaylightWb = false;

    PixelBuffer m_in, m_out;
};

REGISTER_ALGORITHM(BasicAdjust);

} // namespace tglab
