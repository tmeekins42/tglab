// orton — the darkroom sandwich: a sharp frame over a blurred, bright copy.
//
// Michael Orton's original was two slides of the same scene taped together and
// projected as one: a sharp exposure and a defocused one, both overexposed so
// that stacking them came back to a normal density. The result is a glow that
// spills out of the highlights while the edges stay readable -- soft without
// being out of focus.
//
// THE COMBINE IS SCREEN, NOT AN AVERAGE, and that is the whole effect. Two
// slides in a sandwich multiply their TRANSMISSIONS, which for a negative --
// where density is inverted -- is screen on the positive:
//
//     screen(a, b) = 1 - (1 - a)(1 - b)
//
// Screen can only brighten, it saturates smoothly toward white rather than
// clipping, and it leaves black alone: where either layer is black, the result
// is the other layer. Averaging the two instead gives a flat, hazy image with
// no glow at all -- the highlights come DOWN to meet the shadows rather than
// blooming past them, which is the look people mean when they call a bad Orton
// "just a blurry overlay".
//
// Screen needs values in 0..1 to mean anything: (1-a) is nonsense at a = 4.
// Scene-linear data from a raw file routinely runs above 1, so the operator is
// applied to a tone-mapped copy and the result is scaled back -- see
// ScreenHDR, which is the one piece of arithmetic here that is not obvious.
//
// WHY BRIGHTEN BEFORE BLURRING. Orton overexposed both slides in the camera,
// which is not the same as brightening the sandwich afterwards: the blurred
// layer is made bright FIRST, so its highlights bleed into the surrounding
// dark areas as they spread. Brightening after the blur lifts everything
// uniformly and produces a washed-out image rather than a glow. `brightness`
// therefore applies to the blurred layer only, before it is combined.
//
// WHAT THE CONTROLS DO, in the order they matter:
//
//   blur       how far the soft layer spreads. The single most visible control.
//   strength   how much of the sandwich shows against the original.
//   brightness how far the blurred layer is lifted before combining, which is
//              what turns a soft overlay into a glow.
//   contrast   an S-curve on the result. Orton prints are usually contrastier
//              than the sum of their parts, and screening alone flattens.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"

namespace tglab {
namespace {

// Screen, on data that may exceed 1.0.
//
// Screen is defined on 0..1 and its whole character comes from that: it
// approaches white and stops. Scene-linear values from a raw file routinely run
// above 1, where the formula stops meaning anything -- 1 - (1 - 8)(1 - b)
// grows without bound as b rises.
//
// The operands are therefore clamped into range for the screen itself, and the
// pixel's own headroom is added back afterwards. Below 1.0 this is exactly
// ordinary screen. Above it, a genuinely blown pixel passes through at its own
// value rather than being pulled down to white -- two blown slides sandwiched
// are still blown.
//
// A ROUND TRIP THROUGH x/(1+x) WAS TRIED FIRST AND IS WRONG. Compressing both
// operands, screening, then expanding with y/(1-y) looks principled but is
// not: screen of two compressed values is not itself a compressed value, so
// the inverse does not invert it, it amplifies. Measured, a 0.9 square with
// brightness 1.4 came out at 2.79 and a 6.0 highlight at 51.75 -- runaway
// rather than glow.
float ScreenHDR(float a, float b) {
    const float ca = std::clamp(a, 0.0f, 1.0f);
    const float cb = std::clamp(b, 0.0f, 1.0f);
    const float s  = 1.0f - (1.0f - ca) * (1.0f - cb);
    // Headroom the screen could not represent, carried through untouched.
    const float over = std::max(std::max(a, b) - 1.0f, 0.0f);
    return s + over;
}

// An S-curve about mid-grey. Symmetric, so it neither brightens nor darkens
// overall -- it only steepens.
float SCurve(float v, float amount, float white) {
    if (amount == 0.0f) return v;
    const float t = v / std::max(white, 1e-6f);
    // smoothstep is the S; blending toward it by `amount` keeps 0 as identity.
    const float s = t * t * (3.0f - 2.0f * t);
    const float mixed = t + (s - t) * amount;
    return mixed * white;
}

}  // namespace

class Orton : public AlgorithmBase {
public:
    const char* Name()     const override { return "orton"; }
    const char* Category() const override { return "filter"; }

    PortList Inputs() const override {
        return {{"src", DataType::Image, FormatSpec::Any}};
    }
    PortList Outputs() const override {
        return {{"out", DataType::Image, FormatSpec::SameAsInput}};
    }

    void RunCPU(RunCtx& ctx) override {
        const ImageView src = ctx.In(0);
        ImageView       dst = ctx.Out(0);
        if (!src.Valid() || !dst.Valid()) return;

        m_in.Unpack(src);
        if (!m_in.Valid()) return;
        m_out.AllocLike(m_in);

        const int w = m_in.Width(), h = m_in.Height(), ch = m_in.Channels();
        if (w <= 0 || h <= 0) return;

        const float white    = m_in.ValueScale();
        const float sigma    = std::max(0.01f, float(m_blur));
        const float strength = float(m_strength);
        const float bright   = float(m_brightness);
        const float contrast = float(m_contrast);

        const int colours = (ch == 4) ? 3 : ch;
        const int radius  = std::clamp(int(std::ceil(sigma * 3.0f)), 1, 64);
        const float twoSS = 2.0f * sigma * sigma;

        // Separable blur of the BRIGHTENED source. Brightening first is what
        // makes the highlights bleed outward -- see the note at the top.
        m_tmp.assign(size_t(w) * size_t(h) * size_t(colours), 0.0f);
        m_soft.assign(m_tmp.size(), 0.0f);

        for (int y = 0; y < h; ++y) {
            if (ctx.Cancelled()) return;
            for (int x = 0; x < w; ++x) {
                float sum[4] = {0, 0, 0, 0}, weight = 0.0f;
                for (int i = -radius; i <= radius; ++i) {
                    const int sx = std::clamp(x + i, 0, w - 1);
                    const float wt = std::exp(-float(i * i) / twoSS);
                    for (int c = 0; c < colours; ++c)
                        sum[c] += m_in.Get(sx, y, c) * bright * wt;
                    weight += wt;
                }
                const size_t o = (size_t(y) * size_t(w) + size_t(x)) * size_t(colours);
                for (int c = 0; c < colours; ++c)
                    m_tmp[o + size_t(c)] = sum[c] / std::max(weight, 1e-6f);
            }
        }
        for (int y = 0; y < h; ++y) {
            if (ctx.Cancelled()) return;
            for (int x = 0; x < w; ++x) {
                float sum[4] = {0, 0, 0, 0}, weight = 0.0f;
                for (int i = -radius; i <= radius; ++i) {
                    const int sy = std::clamp(y + i, 0, h - 1);
                    const float wt = std::exp(-float(i * i) / twoSS);
                    const size_t s = (size_t(sy) * size_t(w) + size_t(x)) * size_t(colours);
                    for (int c = 0; c < colours; ++c)
                        sum[c] += m_tmp[s + size_t(c)] * wt;
                    weight += wt;
                }
                const size_t o = (size_t(y) * size_t(w) + size_t(x)) * size_t(colours);
                for (int c = 0; c < colours; ++c)
                    m_soft[o + size_t(c)] = sum[c] / std::max(weight, 1e-6f);
            }
        }

        // Sandwich the two, then blend against the original by `strength`.
        for (int y = 0; y < h; ++y) {
            if (ctx.Cancelled()) return;
            for (int x = 0; x < w; ++x) {
                const size_t o = (size_t(y) * size_t(w) + size_t(x)) * size_t(colours);
                for (int c = 0; c < colours; ++c) {
                    const float v = m_in.Get(x, y, c);
                    // Screen works in 0..1, so normalise by the image's own
                    // white before combining and scale back afterwards.
                    const float a = v / white;
                    const float b = m_soft[o + size_t(c)] / white;
                    const float sandwich = ScreenHDR(a, b) * white;
                    const float mixed = v + (sandwich - v) * strength;
                    m_out.Set(x, y, c, SCurve(mixed, contrast, white));
                }
                if (ch == 4) m_out.Set(x, y, 3, m_in.Get(x, y, 3));
            }
        }

        m_out.PackInto(dst);
    }

    // Strength 0 leaves the image alone whatever the other controls say.
    bool IsNoOp() const override { return float(m_strength) == 0.0f; }

    // --- GPU implementation -------------------------------------------------
    //
    // Three passes over two scratch planes: brighten-and-blur horizontally,
    // blur vertically, then combine with the ORIGINAL. That last read is why
    // this cannot use the ping-pong GpuIterations() path that gaussian_blur
    // uses -- ping-pong overwrites the source, and the sandwich needs it.
    bool HasGPU() const override { return true; }

    int        GpuScratchCount()  const override { return 2; }
    FormatSpec GpuScratchPlanes() const override { return FormatSpec::RGBA32F; }

    std::vector<GpuPass> GpuPasses() const override {
        // Assembled once: GpuPasses returns raw pointers, so rebuilding the
        // strings per call would dangle them the moment the vector went away.
        static const std::string blr = std::string(kCommon) + kBlurHlsl;
        static const std::string cmb = std::string(kCommon) + kCombineHlsl;

        std::vector<GpuPass> p;
        p.push_back({blr.c_str(), "blurH",   {-1},    {0}});
        p.push_back({blr.c_str(), "blurV",   {0},     {1}});
        p.push_back({cmb.c_str(), "combine", {-1, 1}, {-1}});
        return p;
    }

    std::vector<uint32_t> GpuPassConstants(int pass) const override {
        auto bits = [](float f) { uint32_t u; std::memcpy(&u, &f, sizeof u); return u; };
        // Pass 0 blurs horizontally and pass 1 vertically. The brightening is
        // folded into the horizontal pass so it happens before the spread; the
        // vertical pass must not apply it a second time.
        const float dir    = (pass == 1) ? 1.0f : 0.0f;
        const float bright = (pass == 0) ? float(m_brightness) : 1.0f;
        return {bits(std::max(0.01f, float(m_blur))),
                bits(float(m_strength)),
                bits(bright),
                bits(float(m_contrast)),
                bits(dir)};
    }

private:
    static constexpr const char* kCommon = R"(
cbuffer Constants : register(b0) {
    uint Width;
    uint Height;
    uint C0; uint C1; uint C2; uint C3; uint C4;
};
float Blur()      { return asfloat(C0); }
float Strength()  { return asfloat(C1); }
float Bright()    { return asfloat(C2); }
float Contrast()  { return asfloat(C3); }
float Dir()       { return asfloat(C4); }
)";

    static constexpr const char* kBlurHlsl = R"(
Texture2D<float4>   Src : register(t0);
RWTexture2D<float4> Dst : register(u0);

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;

    float sigma  = max(Blur(), 0.01);
    // Matches the CPU path's radius exactly, so the two agree rather than
    // merely resembling each other.
    int   radius = clamp(int(ceil(sigma * 3.0)), 1, 64);
    float twoSS  = 2.0 * sigma * sigma;

    float3 sum = 0.0;
    float  weight = 0.0;
    for (int i = -radius; i <= radius; ++i) {
        int2 p = (Dir() < 0.5) ? int2(int(tid.x) + i, int(tid.y))
                               : int2(int(tid.x), int(tid.y) + i);
        p = clamp(p, int2(0, 0), int2(int(Width) - 1, int(Height) - 1));
        float w = exp(-float(i * i) / twoSS);
        sum    += Src[p].rgb * w;
        weight += w;
    }
    // Bright() is 1.0 on the vertical pass, so the lift happens once, before
    // the spread -- which is what makes highlights bleed outward.
    Dst[tid.xy] = float4(sum / max(weight, 1e-6) * Bright(), 1.0);
}
)";

    static constexpr const char* kCombineHlsl = R"(
Texture2D<float4>   Src  : register(t0);
Texture2D<float4>   Soft : register(t1);
RWTexture2D<float4> Dst  : register(u0);

// Screen on data that may exceed 1.0 -- see ScreenHDR in the CPU path for why
// the operands are clamped and the headroom added back rather than round
// tripped through a tone map.
float3 ScreenHDR(float3 a, float3 b) {
    float3 ca = saturate(a);
    float3 cb = saturate(b);
    float3 s  = 1.0 - (1.0 - ca) * (1.0 - cb);
    float3 over = max(max(a, b) - 1.0, 0.0);
    return s + over;
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;

    float4 c = Src[int2(tid.xy)];
    float3 soft = Soft[int2(tid.xy)].rgb;

    // A UNORM SRV hands the shader 0..1 whatever the storage format, so white
    // is 1.0 here and no normalisation is needed -- unlike the CPU path, which
    // works in the image's own units.
    float3 sandwich = ScreenHDR(c.rgb, soft);
    float3 mixed = lerp(c.rgb, sandwich, Strength());

    // The same symmetric S-curve as the CPU path.
    float amount = Contrast();
    float3 s = mixed * mixed * (3.0 - 2.0 * mixed);
    float3 outv = lerp(mixed, s, amount);

    Dst[tid.xy] = float4(outv, c.a);
}
)";

    Param<float> m_blur{
        this, "blur", 12.0f, 0.0f, 128.0f,
        {.help = "How far the soft layer spreads, in pixels. The most visible "
                 "control: small values give a gentle diffusion, large ones "
                 "the dreamy glow the effect is known for.",
         .step = 0.5, .softMin = 2.0, .softMax = 40.0}};

    Param<float> m_strength{
        this, "strength", 0.5f, 0.0f, 1.0f,
        {.help = "How much of the sandwich shows against the original. 0 is "
                 "off, 1 is the full effect.",
         .step = 0.01}};

    Param<float> m_brightness{
        this, "brightness", 1.4f, 1.0f, 3.0f,
        {.help = "How far the blurred layer is lifted BEFORE it is blurred, "
                 "as Orton overexposed his slides in camera. This is what "
                 "makes highlights bleed outward into the shadows rather than "
                 "simply washing the frame out.",
         .step = 0.01}};

    Param<float> m_contrast{
        this, "contrast", 0.2f, 0.0f, 1.0f,
        {.help = "An S-curve on the result. Screening flattens, and Orton "
                 "prints are usually contrastier than the sum of their parts, "
                 "so a little of this puts the bite back.",
         .step = 0.01}};

    PixelBuffer        m_in, m_out;
    std::vector<float> m_tmp, m_soft;
};

REGISTER_ALGORITHM(Orton);

}  // namespace tglab
