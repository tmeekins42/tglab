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

// A smooth 0..1 window used to target a tonal band without a hard edge, which
// would show as banding on a gradient.
inline float SmoothBand(float x, float lo, float hi) {
    const float t = std::clamp((x - lo) / (hi - lo), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
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
        // Floored: see the CPU path -- at full weight this reaches exactly 0
        // for hi = -1 and erases the highlights instead of recovering them.
        c *= max(1.0 + hi * w, 0.15);
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
    void PrepareGpu(const std::vector<ImageDesc>& inputs) override {
        m_linear = !inputs.empty() && inputs[0].linear;
        m_white  = m_linear ? kLinearShoulder : 1.0f;
    }

private:
    // Temperature and tint as channel gains.
    //
    // A physically-correct version would convert the image to XYZ, apply a
    // chromatic adaptation transform for the target illuminant, and convert
    // back. That needs the capture's own white point, which is not available
    // once LibRaw has already applied camera white balance -- so this is the
    // approximation every editor uses for a *relative* warm/cool control:
    // scale red and blue in opposition, and green against magenta.
    //
    // Gains are normalised to preserve luminance, so moving temperature does
    // not also change exposure.
    void WhiteBalanceGains(float* r, float* g, float* b) const {
        const float t = float(m_temperature);
        const float n = float(m_tint);

        *r = 1.0f + t * 0.4f;
        *b = 1.0f - t * 0.4f;
        *g = 1.0f - n * 0.3f;
        // Tint pushes green against magenta, so red and blue take the other half.
        *r += n * 0.15f;
        *b += n * 0.15f;

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
    // symptom: not the slider, the band it drives.
    void Apply(float* rr, float* gg, float* bb,
               float wbR, float wbG, float wbB, float expGain, float white) const {
        float r = *rr * wbR * expGain;
        float g = *gg * wbG * expGain;
        float b = *bb * wbB * expGain;

        float lum = Luma(r, g, b);

        const float hi = float(m_highlights);
        if (std::abs(hi) > 1e-4f) {
            const float w = SmoothBand(lum, 0.35f, white);
            // Floored well above zero: at full weight, k = 1 + hi*w reaches
            // exactly 0 for hi = -1, wiping highlights to black rather than
            // recovering them. Harmless on 8-bit input, where the band seldom
            // reached full weight, but reachable on scene-linear data, where
            // it now does. -1 means "pull hard", not "erase".
            const float k = std::max(1.0f + hi * w, 0.15f);
            r *= k; g *= k; b *= k;
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

    Param<float> m_temperature{
        this, "temperature", 0.0f, -1.0f, 1.0f,
        {.help = "Warm/cool balance. Positive warms (more red, less blue), "
                 "negative cools. Relative to the camera's own white balance, "
                 "not an absolute Kelvin value.",
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

    PixelBuffer m_in, m_out;
};

REGISTER_ALGORITHM(BasicAdjust);

} // namespace tglab
