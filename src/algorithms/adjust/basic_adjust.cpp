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
#include "../../algo_util/white_balance.h"
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
    // The linear path is passed through UNCLAMPED, at both ends.
    //
    // The comment here used to say "no clamp on the linear path -- the headroom
    // is the point" while the code clamped negatives to zero. The headroom
    // argument was only half applied: values above 1.0 were preserved, which is
    // what highlight recovery needs, and values below 0.0 were destroyed.
    //
    // Both ends matter for the same reason. A colour the sensor recorded that
    // sRGB cannot represent lands outside the gamut triangle, which in linear
    // sRGB coordinates means a channel below zero -- and such a value can come
    // back INTO gamut after a desaturation or a white-balance change. Clamping
    // it here makes that impossible.
    //
    // Nothing downstream requires non-negative input: ToneCurve() maps anything
    // at or below zero to black, which is the correct behaviour at display and
    // the wrong behaviour in the middle of an edit.
    return linear ? c : LinearToSrgb(c);
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
            // Compress the excess above the top rather than pulling every
            // channel onto it -- see the CPU path for why. Subtracting the full
            // excess put every bright pixel on the same value, which on a
            // tonemapped sky is one flat grey plate.
            // max(c - wp, 0) rather than a per-channel branch: a channel below
            // the top contributes nothing, so the same expression covers both
            // cases without HLSL needing a vector condition.
            float shed = 0.5 * amount;
            c -= max(c - wp, 0.0) * shed;
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


    // Opens the kelvin slider at the temperature the camera chose, and tint at
    // the green/magenta offset it applied alongside -- so the controls describe
    // this photograph before anyone touches them.
    //
    // Both are defaults rather than values: a script that sets kelvin, or a
    // slider the user has already moved, wins. Left at 0 for a source with no
    // daylight reference, where 0 correctly means "nothing to correct against".
    void PrepareDefaults(const SourceFacts& f) override {
        if (!f.isMosaic) return;

        // White balance from what the camera recorded.
        if (f.asShotKelvin > 0.0f) {
            m_kelvin.SetDefault(f.asShotKelvin);
            m_tint.SetDefault(f.asShotTint);
        }

        // Exposure from what the sensor actually captured.
        //
        // `hasExposure` is the script's decision -- params(basic_adjust,
        // auto_exposure = 1) -- rather than a parameter here. It is not an
        // adjustment: it decides WHERE the adjustments start, which is a
        // property of the call. As a checkbox it read as something you could
        // toggle to see the effect, and could not honestly behave that way,
        // because by the time a control has a value the defaults have already
        // been chosen.
        //
        // These are defaults like any other: drag one and it stops following
        // the measurement, double-click and it returns to it.
        if (f.hasExposure) {
            m_exposure.SetDefault(f.autoExposure);
            m_highlights.SetDefault(f.autoHighlights);
            m_shadows.SetDefault(f.autoShadows);
            m_blacks.SetDefault(f.autoBlacks);
        }
    }
    void PrepareGpu(const std::vector<ImageDesc>& inputs) override {
        m_linear = !inputs.empty() && inputs[0].linear;
        m_white  = m_linear ? kLinearShoulder : 1.0f;
        if (!inputs.empty()) CaptureWhiteBalance(inputs[0]);
    }

private:
    // The white-balance gains, from kelvin and tint.
    //
    // Two axes, which is what white balance actually has: a colour temperature
    // along the Planckian locus, and a green/magenta offset perpendicular to it
    // that no temperature can express. Every raw developer uses this pair, and
    // it is what the camera's own choice decomposes into -- measured, matching a
    // file's temperature alone still left red and blue both off by 1.078, equal
    // in both, which is a pure green shift by definition.
    //
    // A third relative "temperature" nudge used to sit on top. It was dropped
    // once kelvin could start at the camera's own value: a small warm/cool
    // adjustment is just a small move in Kelvin, and having two controls for one
    // axis meant neither said clearly what it did.
    //
    // The demosaic has already applied camMul, so the image arrives balanced as
    // the camera saw it -- meaning a request is applied RELATIVE to that. Undo
    // the camera's choice, apply the asked-for illuminant, and the two cancel
    // exactly when the two agree. That cancellation is what makes the number
    // mean something rather than being an arbitrary curve.
    //
    // Without preMul there is nothing to measure against, so this leaves the
    // image alone. That is the honest behaviour for a JPEG, which carries no
    // record of where neutral was.
    //
    // Gains are normalised on luminance, so changing the white balance does not
    // double as an exposure change.
    void WhiteBalanceGains(float* r, float* g, float* b) const {
        *r = 1.0f;
        *g = 1.0f;
        *b = 1.0f;

        const float k = float(m_kelvin);
        if (k > 1.0f && m_hasDaylightWb) {
            // The requested illuminant, and D65, both as gains on this sensor.
            // preMul is the camera's D65 reference, so the ratio between the two
            // expresses the request in camera space.
            float wr = 1.0f, wg = 1.0f, wb = 1.0f;
            KelvinGains(k, float(m_tint), &wr, &wg, &wb);

            float dr = 1.0f, dg = 1.0f, db = 1.0f;
            KelvinGains(kD65Kelvin, 0.0f, &dr, &dg, &db);

            // What the camera already applied, relative to its daylight
            // reference: the correction to undo.
            const float shotR = m_camMul[0] / std::max(m_preMul[0], 1e-6f);
            const float shotB = m_camMul[2] / std::max(m_preMul[2], 1e-6f);

            *r = (wr / std::max(dr, 1e-6f)) / std::max(shotR, 1e-6f);
            *g = wg / std::max(dg, 1e-6f);
            *b = (wb / std::max(db, 1e-6f)) / std::max(shotB, 1e-6f);
        } else {
            // No reference to anchor to. Tint is still meaningful on its own --
            // it is a straightforward green/magenta push, not something that
            // needs to know where neutral was -- so it stays available.
            const float n = float(m_tint);
            *g = 1.0f - n * 0.3f;
            *r = 1.0f + n * 0.15f;
            *b = 1.0f + n * 0.15f;
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
                // Compress the excess above the band top, rather than pulling
                // every channel to a fixed value.
                //
                // The previous version subtracted `(c - white) * amount`, which
                // at full strength lands EVERY channel of EVERY bright pixel on
                // `white` exactly -- one flat grey plate. On an SDR image that
                // was nearly invisible, because little sits far above 0.70 and
                // SmoothBand's window still varied. On tonemapped HDR the whole
                // sky sits above the top, SmoothBand saturates at 1.0 across all
                // of it, and the result was the solid grey Tim reported.
                //
                // Scaling the excess instead keeps the ORDER and the spacing of
                // what is up there: a cloud edge brighter than its centre stays
                // brighter, just by less. At full strength the excess is halved
                // rather than erased, which is a recovery a photographer would
                // recognise.
                //
                // Per channel still, not a uniform multiply: a blown highlight
                // is blown because ONE channel clipped, so recovery has to move
                // that channel down relative to the others. A uniform scale only
                // darkens and leaves the cast behind -- measured, a 1.20/0.35/
                // 0.20 red held saturation 0.833 at every slider position.
                // Written as "subtract a FRACTION of the excess" rather than
                // "scale toward the top" so the GPU path can use the same
                // expression -- HLSL cannot take a ternary on a vector
                // condition, and max(c - white, 0) covers both cases.
                const float shed = 0.5f * amount;
                r -= std::max(r - white, 0.0f) * shed;
                g -= std::max(g - white, 0.0f) * shed;
                b -= std::max(b - white, 0.0f) * shed;

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

    // Defaults to the temperature the camera actually used, recovered from the
    // file when the image loads -- so the slider opens showing what this
    // photograph IS, and moving it away is what changes the picture.
    //
    // It previously defaulted to 0 meaning "leave the camera's balance alone",
    // which is a poor answer to "what temperature is this?" when the file
    // knows. 0 remains valid and still means that, for a JPEG or a raw whose
    // profile carries no daylight reference.
    Param<float> m_kelvin{
        this, "kelvin", 0.0f, 0.0f, kKelvinMax,
        {.help = "Colour temperature of the light the scene was under, in Kelvin "
                 "-- 2800 tungsten, 5500 daylight, 7000 shade. Setting it to the "
                 "ACTUAL light neutralises the cast: a photograph shot under "
                 "tungsten needs ~2800 here, not a cooler number. Starts at the "
                 "value the camera chose. Needs a raw file; a JPEG carries no "
                 "record of where neutral was, so this does nothing there.",
         .step = 50.0, .softMin = 2000.0, .softMax = 10000.0}};

    Param<float> m_tint{
        this, "tint", 0.0f, -1.0f, 1.0f,
        {.help = "Green/magenta balance, the axis colour temperature cannot "
                 "express. Negative is greener, positive more magenta. Also "
                 "starts where the camera had it: matching a raw's white balance "
                 "needs both axes, not just the temperature.",
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
