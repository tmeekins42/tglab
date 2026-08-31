// bloom — threshold the highlights, spread them, add them back.
//
// The simple version is three steps and Tim's description of it is exactly
// right: threshold, blur, merge. What makes this one worth writing as a single
// algorithm is that HALATION, STARBURSTS and ANAMORPHIC STREAKS are all the
// same three steps with a different spread. Only the kernel changes, so one
// algorithm covers them and the effects compose instead of stacking as separate
// stages that each re-threshold the image.
//
// WHAT THE OPTICS ACTUALLY ARE, because the parameters follow from it and
// "bloom with a red tint" is the wrong model for halation.
//
//   BLOOM is scattering BEFORE the sensor -- veiling glare in the lens, dust,
//   the coating. It is broadly achromatic and roughly Gaussian, and it spreads
//   from bright areas in every direction.
//
//   HALATION happens INSIDE film. Light passes through the emulsion, reflects
//   off the base or the pressure plate behind it, and re-exposes the layers
//   from below. Red penetrates deepest, so it is what comes back -- which is
//   why halation is red-orange rather than white, and why it hugs the EDGES of
//   bright areas rather than filling them: the returning light is offset by
//   twice the base thickness.
//
//   So halation is not bloom that has been tinted. It is a spread that differs
//   PER CHANNEL, which is why `dispersion` scales each channel's radius rather
//   than colouring the result. At dispersion 1 the red channel spreads furthest
//   and blue least, and the glow around a highlight goes warm at its edge
//   because that is where only the red has reached.
//
//   STARBURST is diffraction at the aperture blades. An n-bladed iris produces
//   n spikes for even n and 2n for odd -- a six-bladed iris gives six, a
//   five-bladed one gives ten -- because an odd polygon's opposite edges are
//   not parallel. That is why `blades` is a real parameter and not decoration.
//
//   ANAMORPHIC streaks are the horizontal flares from a cylindrical element,
//   and are just the starburst with two spikes and no rotation. Same code path,
//   `blades = 1` and `streak = 1`.
//
// WHY ONE ALGORITHM RATHER THAN FOUR. Every one of these thresholds the same
// highlights and adds back into the same image. Split into separate stages,
// each would re-threshold and re-add, so two of them together would count the
// highlights twice and the second would bloom the first one's glow. Here the
// threshold happens once and the spreads are summed before the merge, which is
// both correct and cheaper.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"

namespace tglab {
namespace {

// Shared by every pass: the constants and the soft-knee threshold.
const char* kCommon = R"(
cbuffer Params : register(b0) {
    uint Width;
    uint Height;
    uint ThresholdBits;
    uint KneeBits;
    uint RadiusBits;      // base blur radius in pixels
    uint DispersionBits;  // per-channel radius spread, 0 = achromatic
    uint IntensityBits;
    uint StreakBits;      // 0 = none, 1 = full directional
    uint BladesBits;      // spike count driver
    uint AngleBits;       // starburst rotation, radians
    uint TintRBits;
    uint TintGBits;
    uint TintBBits;
    uint PassDirBits;     // 0 = horizontal, 1 = vertical, else a spike index
};

float T()   { return asfloat(ThresholdBits); }
float Knee(){ return asfloat(KneeBits); }
float Rad() { return asfloat(RadiusBits); }
float Disp(){ return asfloat(DispersionBits); }

// The SOFT KNEE, which is what stops the effect switching on at a hard edge.
//
// A plain `max(v - t, 0)` makes every pixel either bloom or not, so a gradient
// crossing the threshold gets a visible contour where the glow begins. The knee
// blends quadratically over a band below the threshold, so the contribution
// grows from zero smoothly -- the standard formulation, and the reason bloom
// on a sky looks like light rather than like a mask.
float3 Highlight(float3 c) {
    float t = T();
    float k = max(Knee(), 1e-4);
    float br = max(c.r, max(c.g, c.b));
    float soft = clamp(br - t + k, 0.0, 2.0 * k);
    soft = soft * soft / (4.0 * k);
    float contrib = max(soft, br - t) / max(br, 1e-5);
    return c * max(contrib, 0.0);
}
)";

// Pass 1: extract the highlights.
const char* kThresholdHlsl = R"(
Texture2D<float4>   Src : register(t0);
RWTexture2D<float4> Dst : register(u0);

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    float4 c = Src[int2(tid.xy)];
    Dst[tid.xy] = float4(Highlight(c.rgb), 1.0);
}
)";

// Passes 2 and 3: a separable Gaussian with a PER-CHANNEL radius.
//
// The per-channel part is the whole of halation and dispersion. One kernel
// weight per channel per tap, so the three colours spread by different amounts
// in a single pass rather than needing three blurs.
//
// The loop bound is the LARGEST of the three radii, and the narrower channels
// simply contribute nothing past their own -- which costs a few wasted taps and
// avoids three separate passes.
const char* kBlurHlsl = R"(
Texture2D<float4>   Src : register(t0);
RWTexture2D<float4> Dst : register(u0);

// 1 where |x| <= edge, 0 beyond, per channel. Written out because HLSL's
// step() takes the edge first and reads backwards here.
float3 Step3(float x, float3 edge) {
    return float3(x <= edge.r ? 1.0 : 0.0,
                  x <= edge.g ? 1.0 : 0.0,
                  x <= edge.b ? 1.0 : 0.0);
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;

    // Red spreads furthest, blue least: light of a longer wavelength
    // penetrates the emulsion deeper before it reflects.
    float d = Disp();
    float3 sigma = Rad() * float3(1.0 + 0.6 * d, 1.0, max(1.0 - 0.5 * d, 0.15));
    int    R     = (int)ceil(max(sigma.r, max(sigma.g, sigma.b)) * 2.5);
    R = min(R, 128);

    int2 step = (PassDirBits == 0) ? int2(1, 0) : int2(0, 1);

    float3 sum = float3(0, 0, 0);
    float3 wsum = float3(0, 0, 0);
    for (int i = -R; i <= R; ++i) {
        int2 p = int2(tid.xy) + step * i;
        p = clamp(p, int2(0, 0), int2(Width - 1, Height - 1));
        float3 c = Src[p].rgb;

        float fi = (float)i;
        float3 w = exp(-(fi * fi) / (2.0 * sigma * sigma));
        // A channel whose sigma is tiny would otherwise pick up a long tail of
        // near-zero weights that sum to something; zeroing past 2.5 sigma keeps
        // each channel's support honest.
        w *= Step3(abs(fi), sigma * 2.5);

        sum  += c * w;
        wsum += w;
    }
    Dst[tid.xy] = float4(sum / max(wsum, 1e-6), 1.0);
}
)";

// Pass 4: the directional streaks, accumulated over every spike.
//
// One pass walks all the spikes rather than one pass each: a spike is a 1D
// blur along a direction, and the count is small enough (2 to 16) that doing
// them in one dispatch beats sixteen dispatches with their own barriers.
const char* kStreakHlsl = R"(
Texture2D<float4>   Src : register(t0);   // the thresholded highlights
RWTexture2D<float4> Dst : register(u0);

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;

    float streak = asfloat(StreakBits);
    if (streak <= 0.0) { Dst[tid.xy] = float4(0, 0, 0, 1); return; }

    int   blades = max((int)asfloat(BladesBits), 1);
    float angle  = asfloat(AngleBits);

    // An n-bladed iris gives n spikes when n is even and 2n when it is odd,
    // because an odd polygon has no pair of parallel edges -- each edge
    // diffracts into its own spike instead of sharing one with its opposite.
    int spikes = (blades % 2 == 0) ? blades : blades * 2;
    spikes = min(spikes, 32);

    // LINES, not spikes. The inner loop runs i from -N to +N, so one line
    // already draws two opposite rays -- iterating `spikes` times over pi
    // therefore drew twice as many rays as asked for, and put them at half the
    // intended spacing.
    //
    // The six-blade case hid it: 6 lines over pi gave 12 rays at 30 degrees,
    // which still looks like a plausible starburst. blades = 1 is what exposed
    // it -- 2 lines over pi is a horizontal AND a vertical, so the "anamorphic"
    // streak came out as a cross.
    int lines = max(spikes / 2, 1);

    // Long and thin: a streak reaches much further than the round glow, which
    // is what makes it read as a diffraction spike rather than a smear.
    float len = Rad() * 4.0;
    int   N   = min((int)len, 192);

    float d = Disp();
    float3 chroma = float3(1.0 + 0.6 * d, 1.0, max(1.0 - 0.5 * d, 0.15));

    float3 sum = float3(0, 0, 0);

    for (int s = 0; s < lines; ++s) {
        float th = angle + 3.14159265 * (float)s / (float)lines;
        float2 dir = float2(cos(th), sin(th));

        // Both directions along the axis, so a spike is a full line rather
        // than a ray -- diffraction is symmetric about the source.
        for (int i = -N; i <= N; ++i) {
            if (i == 0) continue;
            float2 p = float2(tid.xy) + dir * (float)i;
            if (p.x < 0.0 || p.y < 0.0 || p.x >= Width || p.y >= Height) continue;

            // 1/d falloff rather than Gaussian: a diffraction spike decays far
            // more slowly than scattering, which is exactly why it reads as a
            // line reaching across the frame instead of a blob.
            float w = 1.0 / (1.0 + abs((float)i) * 4.0 / max(len, 1.0));
            sum += Src[int2(p)].rgb * w * chroma;
        }
    }
    // NOT DIVIDED BY THE TOTAL WEIGHT, and this is the one place where doing
    // the physically tidy thing gives the visually wrong answer.
    //
    // The round blur normalises because it REDISTRIBUTES light: the same energy
    // is spread wider, so a small bright source spreading over 20x the area
    // gets 20x dimmer. That is correct for scattering, and it is what a
    // Gaussian conserving energy means.
    //
    // A diffraction spike is not that. The light in the spike did not come out
    // of the core -- it is the part of the wavefront the aperture edge bent,
    // arriving in ADDITION to what went straight through. Normalising it over
    // the length of the line divided a point source by about a hundred and made
    // the spikes invisible while the data insisted they were there.
    //
    // So the falloff is applied per tap and the sum is taken as-is, scaled only
    // by a length normaliser that keeps the brightness roughly independent of
    // `spread` -- otherwise widening the glow would silently dim the spikes.
    Dst[tid.xy] = float4(sum * streak / max(len * 0.25, 1.0), 1.0);
}
)";

// Pass 5: add the glow back.
const char* kCombineHlsl = R"(
Texture2D<float4>   Src    : register(t0);   // the original image
Texture2D<float4>   Glow   : register(t1);   // blurred highlights
Texture2D<float4>   Streak : register(t2);   // directional spikes
RWTexture2D<float4> Dst    : register(u0);

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;

    float4 base = Src[int2(tid.xy)];
    float3 g    = Glow[int2(tid.xy)].rgb + Streak[int2(tid.xy)].rgb;

    float3 tint = float3(asfloat(TintRBits), asfloat(TintGBits), asfloat(TintBBits));
    g *= tint * asfloat(IntensityBits);

    // ADDED, not blended. Bloom is light that arrived in addition to what was
    // already there -- scattered from elsewhere in the frame -- so it adds
    // energy. A lerp would darken the highlight it is supposed to be spilling
    // from, which is the giveaway of a bloom done as a compositing operation
    // rather than as light.
    //
    // Alpha carried through untouched: glow does not change transparency.
    Dst[tid.xy] = float4(base.rgb + g, base.a);
}
)";

class Bloom : public AlgorithmBase {
public:
    const char* Name()     const override { return "bloom"; }
    const char* Category() const override { return "filter"; }

    PortList Inputs()  const override { return {{"src", DataType::Image, FormatSpec::Any}}; }
    PortList Outputs() const override {
        return {{"out", DataType::Image, FormatSpec::SameAsInput}};
    }

    bool IsNoOp() const override { return float(m_intensity) <= 0.0f; }

    void RunCPU(RunCtx& ctx) override;

    bool HasGPU() const override { return true; }

    int        GpuScratchCount()  const override { return 3; }
    FormatSpec GpuScratchPlanes() const override { return FormatSpec::RGBA32F; }

    std::vector<GpuPass> GpuPasses() const override {
        // Assembled once: GpuPasses returns raw pointers, so rebuilding the
        // strings per call would dangle them the moment the vector went away.
        static const std::string thr = std::string(kCommon) + kThresholdHlsl;
        static const std::string blr = std::string(kCommon) + kBlurHlsl;
        static const std::string stk = std::string(kCommon) + kStreakHlsl;
        static const std::string cmb = std::string(kCommon) + kCombineHlsl;

        std::vector<GpuPass> p;
        // 0 holds the highlights, 1 and 2 ping-pong the blur, 2 ends up
        // holding the round glow and 0 is reused for the streaks -- which is
        // safe because the streaks read the highlights and the highlights are
        // no longer needed once the blur has consumed them... except they ARE,
        // so the streak pass reads 0 and writes 1, and 0 survives.
        p.push_back({thr.c_str(), "threshold", {-1},          {0}});
        p.push_back({blr.c_str(), "blurH",     {0},           {2}});
        p.push_back({blr.c_str(), "blurV",     {2},           {1}});
        p.push_back({stk.c_str(), "streak",    {0},           {2}});
        p.push_back({cmb.c_str(), "combine",   {-1, 1, 2},    {-1}});
        return p;
    }

    std::vector<uint32_t> GpuPassConstants(int pass) const override {
        auto bits = [](float f) { uint32_t u; std::memcpy(&u, &f, sizeof u); return u; };
        // Pass 1 is the horizontal blur and pass 2 the vertical; everything
        // else ignores the direction.
        const float dir = (pass == 2) ? 1.0f : 0.0f;
        return {bits(float(m_threshold)),
                bits(std::max(0.01f, float(m_knee))),
                bits(std::max(0.1f, float(m_radius))),
                bits(float(m_dispersion)),
                bits(float(m_intensity)),
                bits(float(m_streak)),
                bits(float(m_blades)),
                bits(float(m_angle) * 3.14159265f / 180.0f),
                bits(float(m_tintR)),
                bits(float(m_tintG)),
                bits(float(m_tintB)),
                bits(dir)};
    }

private:
    Param<float> m_threshold{this, "threshold", 0.8f, 0.0f, 4.0f,
        {.help = "How bright a pixel must be to bloom. On a scene-linear image "
                 "with headroom above 1.0 this can usefully go past 1 -- only "
                 "the genuinely blown highlights then glow.",
         .step = 0.02, .softMax = 2.0}};

    Param<float> m_knee{this, "knee", 0.15f, 0.01f, 1.0f,
        {.help = "Width of the soft transition below the threshold. A hard "
                 "cutoff puts a visible contour wherever a gradient crosses "
                 "it; the knee is what makes the glow read as light.",
         .step = 0.01}};

    Param<float> m_radius{this, "spread", 24.0f, 0.5f, 256.0f,
        {.help = "How far the glow reaches, in pixels. This is the blur sigma "
                 "-- the visible glow extends about two and a half times it.",
         .step = 1.0, .softMax = 80.0}};

    Param<float> m_intensity{this, "intensity", 0.35f, 0.0f, 4.0f,
        {.help = "How much glow is added back. 0 turns the whole stage off, "
                 "and the pipeline then skips it entirely.",
         .step = 0.02, .softMax = 1.5}};

    // HALATION, expressed as a spread that differs per channel rather than as
    // a tint. See the file header: halation is light returning through the
    // emulsion from the backing, and red gets deepest, so what makes it look
    // right is red reaching further -- not the whole glow being coloured.
    Param<float> m_dispersion{this, "dispersion", 0.0f, 0.0f, 1.0f,
        {.help = "Spread the channels by different amounts -- red furthest, "
                 "blue least. This is halation: the warm fringe appears where "
                 "only the longer wavelengths have reached, so it hugs the "
                 "edge of a highlight rather than filling it. 0 is achromatic "
                 "bloom.",
         .step = 0.05}};

    Param<float> m_streak{this, "streak", 0.0f, 0.0f, 2.0f,
        {.help = "Diffraction spikes from the aperture blades. 0 is none; "
                 "raise it for a starburst on point highlights.",
         .step = 0.05, .softMax = 1.0}};

    Param<float> m_blades{this, "blades", 6.0f, 1.0f, 16.0f,
        {.help = "Aperture blades. An EVEN count gives that many spikes, an "
                 "ODD count gives twice as many -- an odd polygon has no "
                 "parallel edges, so each diffracts into its own spike. Set 1 "
                 "for a two-spike anamorphic streak.",
         .step = 1.0}};

    Param<float> m_angle{this, "streak_angle", 0.0f, 0.0f, 180.0f,
        {.help = "Rotation of the spike pattern, in degrees. 0 puts the first "
                 "spike horizontal, which with blades = 1 is the anamorphic "
                 "look.",
         .step = 5.0}};

    Param<float> m_tintR{this, "tint_r", 1.0f, 0.0f, 2.0f,
        {.help = "Colour of the glow, multiplied in after the spread. Leave at "
                 "1,1,1 for neutral; warm it for a filmic halation on top of "
                 "the per-channel spread.",
         .step = 0.05}};
    Param<float> m_tintG{this, "tint_g", 1.0f, 0.0f, 2.0f,
        {.help = "Green component of the glow tint.", .step = 0.05}};
    Param<float> m_tintB{this, "tint_b", 1.0f, 0.0f, 2.0f,
        {.help = "Blue component of the glow tint.", .step = 0.05}};
};

// The CPU path, which exists so the algorithm works without a GPU and so the
// two can be compared. Same arithmetic, laid out as three buffers rather than
// five passes.
void Bloom::RunCPU(RunCtx& ctx) {
    const ImageView src = ctx.In(0);
    ImageView       dst = ctx.Out(0);
    if (!src.Valid() || !dst.Valid()) return;

    PixelBuffer in;
    in.Unpack(src);
    if (!in.Valid()) return;

    PixelBuffer out;
    out.Unpack(dst);
    if (!out.Valid()) return;

    const int w = in.Width(), h = in.Height(), ch = in.Channels();
    const float scale = in.ValueScale();

    const float thr  = float(m_threshold) * scale;
    const float knee = std::max(0.01f, float(m_knee)) * scale;
    const float rad  = std::max(0.1f, float(m_radius));
    const float disp = float(m_dispersion);
    const float amt  = float(m_intensity);
    const float strk = float(m_streak);

    // Highlights.
    std::vector<float> hi(size_t(w) * size_t(h) * 3, 0.0f);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const float* p = in.At(x, y);
            const float r = p[0], g = (ch >= 3) ? p[1] : p[0], b = (ch >= 3) ? p[2] : p[0];
            const float br = std::max(r, std::max(g, b));
            float soft = std::clamp(br - thr + knee, 0.0f, 2.0f * knee);
            soft = soft * soft / (4.0f * knee);
            const float c = std::max(std::max(soft, br - thr), 0.0f) / std::max(br, 1e-5f);
            const size_t i = (size_t(y) * size_t(w) + size_t(x)) * 3;
            hi[i + 0] = r * c;
            hi[i + 1] = g * c;
            hi[i + 2] = b * c;
        }

    // Per-channel sigma, matching the shader.
    const float sig[3] = {rad * (1.0f + 0.6f * disp), rad,
                          std::max(rad * (1.0f - 0.5f * disp), rad * 0.15f)};
    const int R = std::min(128, int(std::ceil(std::max(sig[0], std::max(sig[1], sig[2])) * 2.5f)));

    std::vector<float> tmp(hi.size(), 0.0f), glow(hi.size(), 0.0f);
    for (int axis = 0; axis < 2; ++axis) {
        const std::vector<float>& s = (axis == 0) ? hi : tmp;
        std::vector<float>&       d = (axis == 0) ? tmp : glow;
        for (int y = 0; y < h; ++y) {
            if (ctx.Cancelled()) return;
            for (int x = 0; x < w; ++x) {
                float sum[3] = {0, 0, 0}, wsum[3] = {0, 0, 0};
                for (int i = -R; i <= R; ++i) {
                    const int sx = std::clamp(axis == 0 ? x + i : x, 0, w - 1);
                    const int sy = std::clamp(axis == 0 ? y : y + i, 0, h - 1);
                    const size_t si = (size_t(sy) * size_t(w) + size_t(sx)) * 3;
                    for (int c = 0; c < 3; ++c) {
                        if (std::abs(float(i)) > sig[c] * 2.5f) continue;
                        const float wgt = std::exp(-float(i * i) / (2.0f * sig[c] * sig[c]));
                        sum[c]  += s[si + size_t(c)] * wgt;
                        wsum[c] += wgt;
                    }
                }
                const size_t di = (size_t(y) * size_t(w) + size_t(x)) * 3;
                for (int c = 0; c < 3; ++c)
                    d[di + size_t(c)] = sum[c] / std::max(wsum[c], 1e-6f);
            }
        }
    }

    // Streaks, when asked for.
    std::vector<float> streak;
    if (strk > 0.0f) {
        streak.assign(hi.size(), 0.0f);
        const int blades = std::max(1, int(float(m_blades)));
        int spikes = (blades % 2 == 0) ? blades : blades * 2;
        spikes = std::min(spikes, 32);
        // Lines, not spikes -- see the shader: one line draws two rays.
        const int lines = std::max(spikes / 2, 1);
        const float ang = float(m_angle) * 3.14159265f / 180.0f;
        const float len = rad * 4.0f;
        const int   N   = std::min(192, int(len));
        const float chroma[3] = {1.0f + 0.6f * disp, 1.0f,
                                 std::max(1.0f - 0.5f * disp, 0.15f)};

        for (int y = 0; y < h; ++y) {
            if (ctx.Cancelled()) return;
            for (int x = 0; x < w; ++x) {
                float sum[3] = {0, 0, 0};
                for (int s = 0; s < lines; ++s) {
                    const float th = ang + 3.14159265f * float(s) / float(lines);
                    const float dx = std::cos(th), dy = std::sin(th);
                    for (int i = -N; i <= N; ++i) {
                        if (i == 0) continue;
                        const int sx = int(std::lround(float(x) + dx * float(i)));
                        const int sy = int(std::lround(float(y) + dy * float(i)));
                        if (sx < 0 || sy < 0 || sx >= w || sy >= h) continue;
                        const float wgt = 1.0f / (1.0f + std::abs(float(i)) * 4.0f /
                                                        std::max(len, 1.0f));
                        const size_t si = (size_t(sy) * size_t(w) + size_t(sx)) * 3;
                        for (int c = 0; c < 3; ++c)
                            sum[c] += hi[si + size_t(c)] * wgt * chroma[c];
                    }
                }
                const size_t di = (size_t(y) * size_t(w) + size_t(x)) * 3;
                // Not divided by wsum -- see the shader for why a spike adds
                // light rather than redistributing it.
                for (int c = 0; c < 3; ++c)
                    streak[di + size_t(c)] =
                        sum[c] * strk / std::max(len * 0.25f, 1.0f);
            }
        }
    }

    const float tint[3] = {float(m_tintR), float(m_tintG), float(m_tintB)};
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const float* p = in.At(x, y);
            float*       q = out.At(x, y);
            const size_t i = (size_t(y) * size_t(w) + size_t(x)) * 3;
            for (int c = 0; c < ch; ++c) {
                if (c >= 3) { q[c] = p[c]; continue; }   // alpha untouched
                float g = glow[i + size_t(c)];
                if (!streak.empty()) g += streak[i + size_t(c)];
                q[c] = p[c] + g * tint[c] * amt;
            }
        }
    out.PackInto(dst);
}

REGISTER_ALGORITHM(Bloom);

} // namespace
} // namespace tglab
