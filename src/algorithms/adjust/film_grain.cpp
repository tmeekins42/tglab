// film_grain — add grain, with a controllable grain size.
//
// WHAT GRAIN SIZE MEANS, because a per-pixel random number has no size at all.
// Sampling noise once per pixel gives the finest possible texture and nothing
// else: it is the same on a 4 MP crop as on a 45 MP frame, and turning it up
// only makes it louder, never coarser. Real grain is silver crystals of a
// physical size, so on a bigger enlargement they are BIGGER, not more numerous.
//
// So the noise is sampled on a LATTICE whose spacing is the grain size in
// pixels, and interpolated between lattice points. Size 1 is per-pixel; size 4
// makes each grain about four pixels across. The interpolation is smoothstep
// rather than linear, which matters: linear interpolation between random
// lattice values leaves visible creases along the lattice lines, and on a
// smooth sky those read as a grid rather than as grain.
//
// THE VARIANCE PROBLEM, which is the part that is easy to get wrong. Averaging
// or interpolating random values reduces their variance -- interpolating a
// 4-pixel lattice gives noticeably flatter noise than a 1-pixel one, so
// dragging the size slider changes the apparent STRENGTH as well, and the two
// controls fight. Compensating restores it: bilinear interpolation of
// independent samples has variance sum(w^2), which for the smoothstep weights
// used here comes out below 1, so the result is divided by its own RMS. The
// grain then keeps the same loudness at every size, and `strength` means one
// thing. See kVarianceComp.
//
// GRAIN IS MULTIPLICATIVE, NOT ADDITIVE, and this is what separates film grain
// from sensor noise. A film grain is a developed crystal: it modulates how much
// light gets through, so its effect scales with the exposure there. Black stays
// black -- there is nothing to modulate -- and the midtones carry the most
// visible texture. Adding a constant-amplitude noise instead lifts the shadows
// into a grey haze, which is what digital sensor noise looks like and not what
// anyone means by "film grain".
//
// The shadow behaviour is the giveaway. On a properly exposed frame the deep
// shadows of a film scan are comparatively clean and the midtones are where the
// grain lives; on a noisy digital file it is the other way round.
//
// MONOCHROME BY DEFAULT. One noise field shared by all three channels moves the
// pixel along the luminance axis, which is what a black-and-white negative does
// and what colour film approximates, since its three layers are physically
// stacked and largely coincident. `colour` blends toward an independent field
// per channel, which is the chroma-noise look of a pushed colour stock or a
// high-ISO digital file. At 0 the grain has no hue at all.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"

namespace tglab {
namespace {

// RMS of smoothstep-weighted bilinear interpolation of four independent
// unit-variance samples: sqrt of sum(w^2) averaged over the unit cell.
// Dividing the interpolated noise by this restores unit variance, so grain
// size does not change apparent strength.
//
// INTEGRATED NUMERICALLY, not assumed. The smoothstep weights are not the
// linear ones, so the familiar bilinear figure does not apply -- mean variance
// over the cell is 0.5518, and it is that number's square root that belongs
// here. Writing the variance itself would have made the grain 29% too loud.
constexpr float kVarianceComp = 0.742857f;

// ...but the compensation only applies where interpolation actually happens.
//
// At size exactly 1 the lattice spacing equals the pixel spacing, so every
// pixel lands on an integer lattice point, both fractional offsets are 0, and
// the "interpolated" value is a raw sample of variance 1. Dividing that by
// kVarianceComp anyway inflated it by 1/0.7429 = 1.346, and size 1 measured
// 35% louder than every other size -- a visible step exactly where the slider
// starts. Blending the compensation in across the first octave removes it and
// leaves size 2 and above untouched.
inline float VarianceComp(float size) {
    const float t = std::clamp(size - 1.0f, 0.0f, 1.0f);
    return 1.0f + (kVarianceComp - 1.0f) * t;
}

}  // namespace

class FilmGrain : public AlgorithmBase {
public:
    const char* Name()     const override { return "film_grain"; }
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
        if (w <= 0 || h <= 0) return;

        // The modulation is a fraction of the pixel's own value, so it needs no
        // unit scaling -- but the midtone weighting compares against white,
        // which is 255 on an RGBA8 image and 1.0 on a float one.
        const float white = m_in.ValueScale();

        const float strength = float(m_strength);

        // SCALED FOR THE PROXY, but the goal here is different from a blur's
        // and worth stating, because "scale the size" is right for an unobvious
        // reason.
        //
        // A blur reproduces the full-resolution LOOK on a smaller canvas. Grain
        // cannot: it is a random FIELD, not a function of the image, so
        // downsampling one realisation does not approximate another. There is
        // no per-pixel answer to converge to.
        //
        // What the preview owes the user is grain the same size ON SCREEN as
        // the final image will have. A grain of S full-resolution pixels viewed
        // at scale k occupies S*k screen pixels; a proxy is viewed 1:1, so it
        // needs P = S*k. The same multiply, for a different reason.
        //
        // It DEGRADES rather than being exact, and the floor is where: below a
        // size of 1 the lattice cannot go finer than a pixel, so a heavy zoom-
        // out previews grain coarser than the result. That is the honest limit
        // of the approximation, not a bug -- and it errs toward showing MORE
        // texture than the final image, which is the safer direction for a
        // control whose whole job is judging how much grain to add.
        //
        // VarianceComp below is computed from this same scaled size, so the
        // strength compensation follows without a second adjustment.
        const float size     = std::max(1.0f, ctx.ScaledPx(float(m_size)));
        const float colour   = float(m_colour);
        const float shadows  = float(m_shadows);
        const uint32_t seed  = uint32_t(int(m_seed));

        const int colours = (ch == 4) ? 3 : ch;
        const float comp = VarianceComp(size);

        for (int y = 0; y < h; ++y) {
            if (ctx.Cancelled()) return;
            for (int x = 0; x < w; ++x) {
                const float mono =
                    Noise(float(x) / size, float(y) / size, seed, 0, comp);

                for (int c = 0; c < colours; ++c) {
                    const float v = m_in.Get(x, y, c);

                    // Per-channel noise only where `colour` asks for it: the
                    // extra lattice lookups are the expensive part, and at
                    // colour 0 they are pure waste.
                    float n = mono;
                    if (colour > 0.0f && c > 0) {
                        const float ci = Noise(float(x) / size, float(y) / size,
                                               seed, uint32_t(c), comp);
                        n = mono + (ci - mono) * colour;
                    }

                    m_out.Set(x, y, c,
                              Modulate(v, n, strength, shadows, white));
                }
                if (ch == 4) m_out.Set(x, y, 3, m_in.Get(x, y, 3));
            }
        }

        m_out.PackInto(dst);
    }

    bool IsNoOp() const override { return float(m_strength) == 0.0f; }

    // --- GPU implementation -------------------------------------------------
    bool HasGPU() const override { return true; }

    const char* GpuSource() const override {
        return R"(
Texture2D<float4>   Src : register(t0);
RWTexture2D<float4> Dst : register(u0);

cbuffer Params : register(b0) {
    uint Width;
    uint Height;
    uint StrengthBits;
    uint SizeBits;
    uint ColourBits;
    uint ShadowsBits;
    uint Seed;
};

// Integer hash. Wang/Jenkins style: cheap, and good enough that neighbouring
// lattice points do not correlate -- which is the only property that matters
// here. A poor hash shows as a visible diagonal or plaid pattern in the grain.
uint Hash(uint x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// A unit-variance value at an integer lattice point.
float LatticeAt(int ix, int iy, uint seed, uint channel) {
    uint hv = Hash(uint(ix) * 73856093u ^ uint(iy) * 19349663u ^
                   seed * 83492791u ^ channel * 2654435761u);
    // 0..1 -> roughly zero-mean unit-variance. Uniform noise has variance
    // 1/12, so scale by sqrt(12) after centring.
    return (float(hv) * (1.0 / 4294967296.0) - 0.5) * 3.4641016;
}

float Noise(float fx, float fy, uint seed, uint channel, float comp) {
    int ix = int(floor(fx)), iy = int(floor(fy));
    float tx = fx - float(ix), ty = fy - float(iy);
    // Smoothstep, not linear: linear interpolation between random lattice
    // values leaves creases along the lattice lines, which read as a grid.
    tx = tx * tx * (3.0 - 2.0 * tx);
    ty = ty * ty * (3.0 - 2.0 * ty);

    float a = LatticeAt(ix,     iy,     seed, channel);
    float b = LatticeAt(ix + 1, iy,     seed, channel);
    float c = LatticeAt(ix,     iy + 1, seed, channel);
    float d = LatticeAt(ix + 1, iy + 1, seed, channel);

    float n = lerp(lerp(a, b, tx), lerp(c, d, tx), ty);
    // Interpolation reduces variance, so grain size would otherwise change
    // apparent strength. See kVarianceComp / VarianceComp in the CPU path --
    // `comp` is 1.0 at size 1, where nothing is actually interpolated.
    return n / comp;
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;

    float strength = asfloat(StrengthBits);
    float size     = max(asfloat(SizeBits), 1.0);
    float colour   = asfloat(ColourBits);
    float shadows  = asfloat(ShadowsBits);

    float fx = float(tid.x) / size, fy = float(tid.y) / size;
    float comp = lerp(1.0, 0.742857, saturate(size - 1.0));
    float mono = Noise(fx, fy, Seed, 0, comp);

    float4 c = Src[int2(tid.xy)];
    float3 n = float3(mono, mono, mono);
    if (colour > 0.0) {
        n.g = lerp(mono, Noise(fx, fy, Seed, 1, comp), colour);
        n.b = lerp(mono, Noise(fx, fy, Seed, 2, comp), colour);
    }

    // Multiplicative and midtone-weighted -- see the note at the top of the
    // CPU file for why grain is not simply added.
    float3 v = c.rgb;
    float3 t = saturate(v);                      // white is 1.0 to the shader
    float3 weight = lerp(4.0 * t * (1.0 - t), 1.0, shadows);
    Dst[tid.xy] = float4(v * (1.0 + n * strength * weight), c.a);
}
)";
    }

    std::vector<uint32_t> GpuConstants(int) const override {
        auto bits = [](float f) {
            uint32_t u;
            std::memcpy(&u, &f, sizeof(u));
            return u;
        };
        return {bits(float(m_strength)), bits(std::max(1.0f, GpuScaledPx(float(m_size)))),
                bits(float(m_colour)),   bits(float(m_shadows)),
                uint32_t(int(m_seed))};
    }

private:
    // Kept in step with the HLSL above; the CPU/GPU agreement test is what
    // holds the two together.
    static uint32_t Hash(uint32_t x) {
        x ^= x >> 16; x *= 0x7feb352du;
        x ^= x >> 15; x *= 0x846ca68bu;
        x ^= x >> 16;
        return x;
    }

    static float LatticeAt(int ix, int iy, uint32_t seed, uint32_t channel) {
        const uint32_t hv = Hash(uint32_t(ix) * 73856093u ^
                                 uint32_t(iy) * 19349663u ^
                                 seed * 83492791u ^
                                 channel * 2654435761u);
        return (float(hv) * (1.0f / 4294967296.0f) - 0.5f) * 3.4641016f;
    }

    static float Noise(float fx, float fy, uint32_t seed, uint32_t channel,
                       float comp) {
        const int ix = int(std::floor(fx)), iy = int(std::floor(fy));
        float tx = fx - float(ix), ty = fy - float(iy);
        tx = tx * tx * (3.0f - 2.0f * tx);
        ty = ty * ty * (3.0f - 2.0f * ty);

        const float a = LatticeAt(ix,     iy,     seed, channel);
        const float b = LatticeAt(ix + 1, iy,     seed, channel);
        const float c = LatticeAt(ix,     iy + 1, seed, channel);
        const float d = LatticeAt(ix + 1, iy + 1, seed, channel);

        const float top = a + (b - a) * tx;
        const float bot = c + (d - c) * tx;
        return (top + (bot - top) * ty) / comp;
    }

    // Multiplicative, weighted toward the midtones. `shadows` blends that
    // weighting away, toward uniform amplitude everywhere -- which is the
    // digital-sensor look, and is occasionally what is wanted.
    static float Modulate(float v, float n, float strength, float shadows,
                          float white) {
        const float t = std::clamp(v / std::max(white, 1e-6f), 0.0f, 1.0f);
        // 4t(1-t) peaks at 1.0 in the midtones and falls to zero at both ends,
        // so black stays black and clipped white stays clipped.
        const float weight = 4.0f * t * (1.0f - t) +
                             (1.0f - 4.0f * t * (1.0f - t)) * shadows;
        return v * (1.0f + n * strength * weight);
    }

    Param<float> m_strength{
        this, "strength", 0.0f, 0.0f, 1.0f,
        {.help = "How strong the grain is, as a fraction of the pixel's own "
                 "value. Multiplicative, so black stays black and the "
                 "midtones carry the most texture.",
         .step = 0.005, .softMin = 0.0, .softMax = 0.3}};

    Param<float> m_size{
        this, "size", 1.0f, 1.0f, 32.0f,
        {.help = "Grain size in pixels. 1 is per-pixel; larger clumps the "
                 "grain, as a bigger enlargement of the same film would. "
                 "Strength is compensated, so this changes only the texture.",
         .step = 0.1, .softMin = 1.0, .softMax = 8.0}};

    Param<float> m_colour{
        this, "colour", 0.0f, 0.0f, 1.0f,
        {.help = "0 gives monochrome grain -- one field shared by all three "
                 "channels, so it moves along luminance only, as a "
                 "black-and-white negative does. 1 gives an independent field "
                 "per channel, the chroma-noise look of a pushed colour stock.",
         .step = 0.01}};

    Param<float> m_shadows{
        this, "shadows", 0.0f, 0.0f, 1.0f,
        {.help = "0 keeps grain out of the deep shadows and blown highlights, "
                 "which is how film behaves. 1 spreads it evenly over the "
                 "whole tonal range, which is the digital sensor look.",
         .step = 0.01}};

    Param<int> m_seed{
        this, "seed", 1, 1, 9999,
        {.help = "Changes the grain pattern without changing its character. "
                 "Vary it between frames of a sequence so the grain does not "
                 "sit still.",
         .step = 1}};

    PixelBuffer m_in, m_out;
};

REGISTER_ALGORITHM(FilmGrain);

}  // namespace tglab
