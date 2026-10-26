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
#include "../../algo_util/color.h"
#include "../../core/algorithm.h"
#include "../../core/parallel.h"

namespace tglab {
namespace {

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

// Where the shadow band tops out, for scene-linear input.
//
// The band was hardcoded 0.0 to 0.5, which is a GAMMA-ENCODED assumption: in
// sRGB, 0.5 is roughly a midtone, so a band ending there covers the lower half
// of the picture and nothing above it. In linear light 0.5 is well ABOVE the
// midtones -- middle grey is 0.18 -- so the same band gave middle grey 70% of
// the full shadow lift and only stopped short of the brightest highlights.
//
// That is precisely the symptom Tim reported on a tonemapped merge: "the
// shadows slider seems to brighten the entire scene except clouds". It was not
// a shadow control at all, it was a global brightener with the highlights
// masked out.
//
// 0.09 -- a stop BELOW middle grey, not at it.
//
// Topping the band at middle grey was the first correction, and it was still
// too generous. Middle grey is the middle of the picture, so a band reaching it
// covers everything darker than the midtones -- which on a dark scene is most
// of the frame. Tim's second report, after that change: "the shadow slider
// seems to have reverted back boosting the entire scene". It had not reverted;
// the scene had moved. His valley merge has a median of 0.098, so the median
// itself sat inside the band and took 43% of the lift.
//
// A stop below grey is what "shadows" means to a photographer: the tones under
// the midtones, not the lower half of everything. Middle grey now gets zero
// lift rather than 43%, and the deep shadows still get essentially all of it.
//
// Symmetry with the highlight band is worth noting, since the two were pitched
// independently and by different reasoning: kLinearShoulder sits at 0.70, about
// two stops ABOVE grey, and this sits one stop below. They are not mirror
// images because the tonal ranges are not -- there is far more room above
// middle grey in linear light than below it.
//
// Constant rather than per-image, for the same reason kLinearShoulder is: a
// band pitched off the image's own statistics would make the same slider
// position mean something different on every frame.
constexpr float kLinearShadowTop = 0.09f;

// A smooth 0..1 window used to target a tonal band without a hard edge, which
// would show as banding on a gradient.
inline float SmoothBand(float x, float lo, float hi) {
    const float t = std::clamp((x - lo) / (hi - lo), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// The midtones band: centred on middle grey (0.18 linear), falling away over
// about a stop and a quarter either side in log -- so it moves the middle of
// the picture and leaves both ends where they are. Applied as a gain in
// stops: +1 brightens middle grey by a stop.
inline float MidtoneWeight(float lum) {
    const float d = std::log2(std::max(lum, 1e-6f) / 0.18f) / 1.25f;
    return std::exp(-d * d);
}

// THE LOCAL BASE, which is what makes highlights, shadows and midtones edge-
// aware: a guided filter (He, Sun & Tang) of log luminance at kBaseRadius
// full-resolution pixels, threshold kBaseEps stops. The tonal bands are chosen
// by the base -- the brightness of the REGION a pixel is in -- and applied as
// a gain, so the texture inside a region keeps its contrast: pulling down a
// sky's highlights darkens the sky without flattening its clouds, where a
// per-pixel band compresses the bright half of every cloud and leaves them grey.
constexpr int   kBaseRadius = 48;
constexpr float kBaseEps    = 0.4f;

void BoxMeanBA(const std::vector<float>& in, std::vector<float>& out, int w, int h, int r) {
    std::vector<float> tmp(in.size());
    const float norm = 1.0f / float(2 * r + 1);
    ParallelFor(size_t(h), [&](size_t y) {
        const size_t row = y * size_t(w);
        float sum = 0.0f;
        for (int i = -r; i <= r; ++i) sum += in[row + size_t(std::clamp(i, 0, w - 1))];
        for (int x = 0; x < w; ++x) {
            tmp[row + size_t(x)] = sum * norm;
            sum += in[row + size_t(std::clamp(x + r + 1, 0, w - 1))] -
                   in[row + size_t(std::clamp(x - r, 0, w - 1))];
        }
    });
    ParallelFor(size_t(w), [&](size_t x) {
        float sum = 0.0f;
        for (int i = -r; i <= r; ++i) sum += tmp[size_t(std::clamp(i, 0, h - 1)) * size_t(w) + x];
        for (int y = 0; y < h; ++y) {
            out[size_t(y) * size_t(w) + x] = sum * norm;
            sum += tmp[size_t(std::clamp(y + r + 1, 0, h - 1)) * size_t(w) + x] -
                   tmp[size_t(std::clamp(y - r, 0, h - 1)) * size_t(w) + x];
        }
    });
}

// The guided filter's a and b, smoothed: base = a * l + b.
void GuidedAB(const std::vector<float>& l, int w, int h, int r, float eps,
              std::vector<float>* a, std::vector<float>* b) {
    const size_t n = l.size();
    std::vector<float> mean(n), meanSq(n), sq(n);
    for (size_t i = 0; i < n; ++i) sq[i] = l[i] * l[i];
    BoxMeanBA(l, mean, w, h, r);
    BoxMeanBA(sq, meanSq, w, h, r);
    std::vector<float> ra(n), rb(n);
    for (size_t i = 0; i < n; ++i) {
        const float var = std::max(0.0f, meanSq[i] - mean[i] * mean[i]);
        ra[i] = var / (var + eps * eps);
        rb[i] = (1.0f - ra[i]) * mean[i];
    }
    a->resize(n); b->resize(n);
    BoxMeanBA(ra, *a, w, h, r);
    BoxMeanBA(rb, *b, w, h, r);
}

// The base passes on the GPU: log luminance (and its square), running-sum boxes
// a run of kBaseRun pixels per thread, the a/b coefficients, boxes again.
const char* const kBaseCommon = R"(
Texture2D<float4>   T0 : register(t0);
RWTexture2D<float4> U0 : register(u0);
cbuffer Params : register(b0) {
    uint Width;
    uint Height;
    uint Radius;
    uint EpsBits;
    uint Linear;
};
int2 ClampXY(int x, int y) { return int2(clamp(x, 0, int(Width) - 1), clamp(y, 0, int(Height) - 1)); }
)";
const char* const kBaseLog = R"(
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    float3 c = T0[tid.xy].rgb;
    c = (Linear != 0) ? c : SrgbToLinear(c);
    float l = log2(max(Luma(c), 1e-6));
    U0[tid.xy] = float4(l, l * l, 0, 1);
}
)";
const char* const kBaseBoxH = R"(
static const int kRun = 64;
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    int x0 = int(tid.x) * kRun, y = int(tid.y);
    if (x0 >= int(Width) || y >= int(Height)) return;
    int r = int(Radius);
    float inv = 1.0 / float(2 * r + 1);
    float2 sum = 0.0;
    for (int i = -r; i <= r; ++i) sum += T0[ClampXY(x0 + i, y)].xy;
    int x1 = min(x0 + kRun, int(Width));
    for (int x = x0; x < x1; ++x) {
        if (x > x0) sum += T0[ClampXY(x + r, y)].xy - T0[ClampXY(x - r - 1, y)].xy;
        U0[int2(x, y)] = float4(sum * inv, 0, 1);
    }
}
)";
const char* const kBaseBoxV = R"(
static const int kRun = 64;
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    int x = int(tid.x), y0 = int(tid.y) * kRun;
    if (x >= int(Width) || y0 >= int(Height)) return;
    int r = int(Radius);
    float inv = 1.0 / float(2 * r + 1);
    float2 sum = 0.0;
    for (int i = -r; i <= r; ++i) sum += T0[ClampXY(x, y0 + i)].xy;
    int y1 = min(y0 + kRun, int(Height));
    for (int y = y0; y < y1; ++y) {
        if (y > y0) sum += T0[ClampXY(x, y + r)].xy - T0[ClampXY(x, y - r - 1)].xy;
        U0[int2(x, y)] = float4(sum * inv, 0, 1);
    }
}
)";
const char* const kBaseAb = R"(
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    float2 ms = T0[int2(tid.xy)].xy;
    float var = max(0.0, ms.y - ms.x * ms.x);
    float eps = asfloat(EpsBits);
    float a = var / (var + eps * eps);
    U0[tid.xy] = float4(a, (1.0 - a) * ms.x, 0, 1);
}
)";
constexpr uint32_t kBaseRun = 64;   // must match the HLSL above


} // namespace

class BasicAdjust : public AlgorithmBase {
public:
    const char* Name()     const override { return "basic_adjust"; }
    const char* Category() const override { return "adjust"; }

    PortList Inputs()  const override { return {{"src", DataType::Image, FormatSpec::Any}}; }
    PortList Outputs() const override { return {{"out", DataType::Image, FormatSpec::SameAsInput}}; }

    void RunCPU(RunCtx& ctx) override {
        // LOCAL, NOT A MEMBER. One algorithm instance is mapped across every
        // frame of a group, so scratch kept on the instance is shared between
        // the threads running those frames -- and the symptom is not a crash
        // but frames holding each other's pixels. See TestNoSharedScratch.
        PixelBuffer m_in, m_out;
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
        const float shadowTop = linear ? kLinearShadowTop : 0.5f;

        // The white balance, derived once rather than per pixel.
        float wb[9];
        WhiteBalanceMatrix(wb);
        const float expGain = std::pow(2.0f, float(m_exposure));

        // The local base, when a tonal band needs it.
        const bool useBase = NeedBase();
        std::vector<float> logL, baseA, baseB;
        if (useBase) {
            logL.resize(size_t(w) * size_t(h));
            ParallelFor(size_t(h), [&](size_t y) {
                for (int x = 0; x < w; ++x) {
                    const float* p = m_in.At(x, int(y));
                    float r, g, b;
                    if (ch == 1) r = g = b = DecodeIn(p[0] / scale, linear);
                    else { r = DecodeIn(p[0] / scale, linear); g = DecodeIn(p[1] / scale, linear);
                           b = DecodeIn(p[2] / scale, linear); }
                    logL[y * size_t(w) + size_t(x)] = std::log2(std::max(Luma(r, g, b), 1e-6f));
                }
            });
            GuidedAB(logL, w, h, std::max(1, int(std::lround(ctx.ScaledPx(float(kBaseRadius))))), kBaseEps, &baseA, &baseB);
        }

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

                const float r0 = r, g0 = g, b0 = b;
                r = wb[0] * r0 + wb[1] * g0 + wb[2] * b0;
                g = wb[3] * r0 + wb[4] * g0 + wb[5] * b0;
                b = wb[6] * r0 + wb[7] * g0 + wb[8] * b0;
                // The base, as a linear luminance the input had in this
                // region, carried through the same white balance and exposure
                // the pixel was: scaled by the pixel's own change.
                float base = -1.0f;
                if (useBase) {
                    const size_t i = size_t(y) * size_t(w) + size_t(x);
                    const float lumIn = std::max(Luma(r0, g0, b0), 1e-6f);
                    const float lumAdj = Luma(r, g, b) * expGain;
                    base = std::exp2(baseA[i] * logL[i] + baseB[i]) * lumAdj / lumIn;
                }
                Apply(&r, &g, &b, 1.0f, 1.0f, 1.0f, expGain, white, shadowTop, base);

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

    // Per pixel, unless the local base is in use: then the two box stages of
    // its guided filter reach twice its radius.
    int ReachPixels() const override { return NeedBase() ? 2 * kBaseRadius : 0; }

    const char* GpuSource() const override { return kMainHlsl; }

    // With a local base: six passes build it -- they depend only on the input,
    // so dragging any of this stage's sliders skips them -- then the main pass.
    // Without one (no tonal band set, or `local` off) the single kernel above.
    bool NeedBase() const {
        return bool(m_local) && (std::abs(float(m_highlights)) > 1e-4f ||
                                 std::abs(float(m_shadows)) > 1e-4f ||
                                 std::abs(float(m_midtones)) > 1e-4f);
    }
    int        GpuScratchCount()  const override { return NeedBase() ? 4 : 0; }
    FormatSpec GpuScratchPlanes() const override { return FormatSpec::RG32F; }
    std::vector<GpuPass> GpuPasses() const override {
        if (!NeedBase()) return {};
        static const std::string logp = std::string(kBaseCommon) + kBaseLog;
        static const std::string boxh = std::string(kBaseCommon) + kBaseBoxH;
        static const std::string boxv = std::string(kBaseCommon) + kBaseBoxV;
        static const std::string ab   = std::string(kBaseCommon) + kBaseAb;
        static const std::string mainp = std::string("#define USE_BASE 1\n") + kMainHlsl;
        return {{logp.c_str(), "baseLog",  {-1}, {0}},
                {boxh.c_str(), "baseBoxH", {0}, {1}, kBaseRun, 1},
                {boxv.c_str(), "baseBoxV", {1}, {2}, 1, kBaseRun},
                {ab.c_str(),   "baseAb",   {2}, {1}},
                {boxh.c_str(), "baseAbH",  {1}, {2}, kBaseRun, 1},
                {boxv.c_str(), "baseAbV",  {2}, {3}, 1, kBaseRun},
                {mainp.c_str(), "adjust",  {-1, 3}, {-1}}};
    }
    std::vector<uint32_t> GpuPassConstants(int pass) const override {
        if (NeedBase() && pass < 6)
            return {uint32_t(std::max(1, int(std::lround(GpuScaledPx(float(kBaseRadius)))))), FloatBits(kBaseEps),
                    m_linear ? 1u : 0u};
        return GpuConstants(0);
    }

    // The adjustment kernel. With USE_BASE it reads the local base's a/b plane
    // at t1 and drives the tonal bands from it; see Apply.
    static constexpr const char* kMainHlsl = R"(
Texture2D<float4>   Src : register(t0);
#ifdef USE_BASE
Texture2D<float4>   BaseAB : register(t1);
#endif
RWTexture2D<float4> Dst : register(u0);

cbuffer Params : register(b0) {
    uint  Width;
    uint  Height;
    uint  Wb0, Wb1, Wb2, Wb3, Wb4, Wb5, Wb6, Wb7, Wb8;   // white balance, row-major 3x3
    uint  ExpGain;              // 2^exposure
    uint  Contrast;
    uint  Highlights, Shadows;
    uint  Whites, Blacks;
    uint  Vibrance, Saturation;
    uint  Linear;               // 1 = scene-linear input: no transfer, no clamp
    uint  HighlightTop;         // top of the highlight band (1.0 unless linear)
    uint  ShadowTop;            // top of the shadow band (0.5 unless linear)
    uint  Midtones;
};
float MidtoneWeight(float lum) {
    float d = log2(max(lum, 1e-6) / 0.18) / 1.25;
    return exp(-d * d);
}

// kLumaW, Luma, SrgbToLinear and LinearToSrgb: color.h's kColorHlsl,
// prepended at compile time.
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
    float lumIn = max(Luma(c), 1e-6);

    // --- white balance: a 3x3, since on a raw it is done in camera space ---
    c = float3(dot(float3(asfloat(Wb0), asfloat(Wb1), asfloat(Wb2)), c),
               dot(float3(asfloat(Wb3), asfloat(Wb4), asfloat(Wb5)), c),
               dot(float3(asfloat(Wb6), asfloat(Wb7), asfloat(Wb8)), c));

    // --- exposure: a straight multiply, which is only correct in linear ---
    c *= asfloat(ExpGain);

    // --- highlights / shadows ---
    // Each targets a tonal band via a smooth window, so the correction fades
    // out rather than leaving a visible edge where it stops applying.
    float lum = Luma(c);
    float drive = lum;
    float hi = asfloat(Highlights);
#ifdef USE_BASE
    // The region's brightness, carried through this pixel's white balance and
    // exposure; and highlights as a gain on it -- see Apply on the CPU.
    float2 ab = BaseAB[int2(tid.xy)].xy;
    drive = exp2(ab.x * log2(lumIn) + ab.y) * lum / lumIn;
    if (abs(hi) > 1e-4) {
        float wp = asfloat(HighlightTop);
        float w = SmoothBand(drive, 0.35, wp);
        float amount = -hi * w;
        if (amount > 0.0) {
            float shed = 0.5 * amount;
            float b1 = drive - max(drive - wp, 0.0) * shed;
            b1 *= (1.0 - 0.35 * amount);
            c *= b1 / max(drive, 1e-6);
            float l0 = Luma(c);
            float3 c2 = c - max(c - wp, 0.0) * shed;
            float l2 = Luma(c2);
            c = l2 > 1e-6 ? c2 * (l0 / l2) : c;
        } else if (amount < 0.0) {
            c *= (1.0 - amount);
        }
    }
#else
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
#endif
    float sh = asfloat(Shadows);
    if (abs(sh) > 1e-4) {
        float w = 1.0 - SmoothBand(drive, 0.0, asfloat(ShadowTop));
        c *= (1.0 + sh * w);
    }
    float md = asfloat(Midtones);
    if (abs(md) > 1e-4) c *= exp2(md * MidtoneWeight(drive));

    // --- whites / blacks: the endpoints, not bands ---
    float wh = asfloat(Whites);
    if (abs(wh) > 1e-4) c *= (1.0 + wh * saturate(Luma(c)));
    float bl = asfloat(Blacks);
    // Shifts the black point: negative crushes, positive lifts.
    if (abs(bl) > 1e-4) c = max(c + bl * 0.1 * (1.0 - saturate(Luma(c))), 0.0);

    // --- contrast, pivoted on middle grey (0.18 in linear) ---
    float ct = asfloat(Contrast);
    if (abs(ct) > 1e-4) c = max((c - 0.18) * (1.0 + ct) + 0.18, 0.0);

    // --- vibrance / saturation ---
    lum = Luma(c);
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

    std::vector<uint32_t> GpuConstants(int) const override {
        float wb[9];
        WhiteBalanceMatrix(wb);

        return {FloatBits(wb[0]), FloatBits(wb[1]), FloatBits(wb[2]),
                FloatBits(wb[3]), FloatBits(wb[4]), FloatBits(wb[5]),
                FloatBits(wb[6]), FloatBits(wb[7]), FloatBits(wb[8]),
                FloatBits(std::pow(2.0f, float(m_exposure))),
                FloatBits(float(m_contrast)),
                FloatBits(float(m_highlights)), FloatBits(float(m_shadows)),
                FloatBits(float(m_whites)),     FloatBits(float(m_blacks)),
                FloatBits(float(m_vibrance)),   FloatBits(float(m_saturation)),
                uint32_t(m_linear ? 1 : 0), FloatBits(m_white), FloatBits(m_shadowTop),
                FloatBits(float(m_midtones))};
    }

    // HasGPU() is consulted before PrepareGpu(), so this is where the input
    // descriptor first becomes available -- the GPU path never calls RunCPU().
    // The white-balance references the Kelvin control needs. Shared by both
    // paths so they cannot disagree.
    void CaptureWhiteBalance(const ImageDesc& d) {
        for (int i = 0; i < 3; ++i) {
            m_wbDesc.camMul[i] = d.camMul[i];
            m_wbDesc.preMul[i] = d.preMul[i];
        }
        for (int i = 0; i < 9; ++i) m_wbDesc.rgbCam[i] = d.rgbCam[i];
        m_wbDesc.hasDaylightWb = d.hasDaylightWb;
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
        m_shadowTop = m_linear ? kLinearShadowTop : 0.5f;
        if (!inputs.empty()) CaptureWhiteBalance(inputs[0]);
    }

private:
    // The white balance, from kelvin and tint, as a 3x3 on linear sRGB.
    //
    // Two axes, which is what white balance actually has: a colour temperature
    // along the Planckian locus, and a green/magenta offset perpendicular to it
    // that no temperature can express. Every raw developer uses this pair, and
    // it is what the camera's own choice decomposes into -- measured, matching a
    // file's temperature alone still left red and blue both off by 1.078, equal
    // in both, which is a pure green shift by definition.
    //
    // The demosaic has already applied camMul, so the image arrives balanced as
    // the camera saw it -- meaning a request is applied RELATIVE to that, and the
    // two cancel exactly when they agree.
    //
    // IN CAMERA SPACE. White balance is a gain per photosite colour, before the
    // camera matrix; the demosaic applied the camera's as rgbCam * diag(camMul).
    // So a new balance is the matrix undone, the camera's gains swapped for the
    // requested illuminant's, and the matrix put back -- one 3x3 per image,
    // CameraWhiteBalanceMatrix. Gains applied to sRGB after the matrix (what this
    // used to do) neutralise a grey just as well but shift every other colour
    // differently, since sRGB's primaries are not the sensor's; and they mixed
    // camera-space ratios (camMul / preMul) with sRGB-space gains in one product.
    //
    // Without preMul there is nothing to measure against, so this leaves the
    // image alone. That is the honest behaviour for a JPEG, which carries no
    // record of where neutral was. Tint is still meaningful on its own -- a
    // plain green/magenta push -- so it stays available there.
    //
    // Normalised on luminance, so changing the white balance does not double as
    // an exposure change.
    void WhiteBalanceMatrix(float* m) const {
        for (int i = 0; i < 9; ++i) m[i] = (i % 4 == 0) ? 1.0f : 0.0f;

        const float k = float(m_kelvin);
        if (k > 1.0f && m_wbDesc.hasDaylightWb) {
            CameraWhiteBalanceMatrix(m_wbDesc, k, float(m_tint), m);
            return;
        }
        const float n = float(m_tint);
        float r = std::max(1.0f + n * 0.15f, 0.0f);
        float g = std::max(1.0f - n * 0.3f, 0.0f);
        float b = std::max(1.0f + n * 0.15f, 0.0f);
        const float lum = Luma(r, g, b);
        if (lum > 1e-6f) { r /= lum; g /= lum; b /= lum; }
        m[0] = r; m[4] = g; m[8] = b;
    }
    // The CPU path, matching the shader step for step. Kept in sync because
    // compare mode checks one against the other.
    //
    // `white` is where the highlight band tops out; see kLinearShoulder and
    // the caller for why it is not 1.0 on scene-linear input.

    // `base`: the local base's luminance (see kBaseRadius), or negative for the
    // per-pixel bands.
    void Apply(float* rr, float* gg, float* bb,
               float wbR, float wbG, float wbB, float expGain, float white,
               float shadowTop, float base = -1.0f) const {
        float r = *rr * wbR * expGain;
        float g = *gg * wbG * expGain;
        float b = *bb * wbB * expGain;

        float lum = Luma(r, g, b);
        // What chooses the tonal bands: the region's brightness when local.
        const bool local = base >= 0.0f;
        const float drive = local ? base : lum;

        const float hi = float(m_highlights);
        if (local && std::abs(hi) > 1e-4f) {
            // LOCAL: the base decides how much, and the change is a GAIN, so
            // the region's texture keeps its contrast. The same compression of
            // the excess above the band top as below, done to the base.
            const float w = SmoothBand(drive, 0.35f, white);
            const float amount = -hi * w;
            if (amount > 0.0f) {
                const float shed = 0.5f * amount;
                float b1 = drive - std::max(drive - white, 0.0f) * shed;
                b1 *= 1.0f - 0.35f * amount;
                const float gain = b1 / std::max(drive, 1e-6f);
                r *= gain; g *= gain; b *= gain;
                // And still per channel for COLOUR: a blown highlight is blown
                // because one channel clipped, so recovery pulls that channel
                // toward the others -- with the lightness the gain set kept.
                const float l0 = Luma(r, g, b);
                float r2 = r - std::max(r - white, 0.0f) * shed;
                float g2 = g - std::max(g - white, 0.0f) * shed;
                float b2 = b - std::max(b - white, 0.0f) * shed;
                const float l2 = Luma(r2, g2, b2);
                const float k = l2 > 1e-6f ? l0 / l2 : 1.0f;
                r = r2 * k; g = g2 * k; b = b2 * k;
            } else if (amount < 0.0f) {
                const float k = 1.0f - amount;
                r *= k; g *= k; b *= k;
            }
        } else if (std::abs(hi) > 1e-4f) {
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
            const float w = 1.0f - SmoothBand(drive, 0.0f, shadowTop);
            const float k = 1.0f + sh * w;
            r *= k; g *= k; b *= k;
        }
        const float mid = float(m_midtones);
        if (std::abs(mid) > 1e-4f) {
            const float k = std::exp2(mid * MidtoneWeight(drive));
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
    Param<float> m_midtones{
        this, "midtones", 0.0f, -1.0f, 1.0f,
        {.help = "Brightness of the middle tones only, in stops at middle grey, "
                 "falling away toward both ends -- so the picture opens up or "
                 "deepens without the highlights or shadows moving.",
         .step = 0.01, .softMin = -0.5, .softMax = 0.5}};

    Param<float> m_shadows{
        this, "shadows", 0.0f, -1.0f, 1.0f,
        {.help = "Brightness of the lower tones only. POSITIVE opens up shadow "
                 "detail; negative deepens it.",
         .step = 0.01, .softMin = -0.5, .softMax = 1.0}};

    Param<bool> m_local{this, "local_tones", true,
        "Highlights, midtones and shadows by REGION rather than by pixel: the "
        "band each pixel falls in is decided by the brightness around it "
        "(edge-aware, so a sky and the land below it are separate regions), and "
        "texture inside a region keeps its contrast -- clouds stay crisp while "
        "the sky is pulled down. Off: each pixel by its own brightness."};
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
    float m_shadowTop = 0.5f;   // see kLinearShadowTop

    // White-balance references from the input descriptor, needed by the Kelvin
    // control. Set in both paths -- RunCPU reads the descriptor directly, and
    // PrepareGpu is where the GPU path first sees it. Only the white-balance
    // fields (camMul, preMul, rgbCam, hasDaylightWb) are filled.
    ImageDesc m_wbDesc{};

};

REGISTER_ALGORITHM(BasicAdjust);

} // namespace tglab
