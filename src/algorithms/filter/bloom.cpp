// bloom — threshold the highlights, spread them, add them back.
//
// Three steps, and that is the whole algorithm: threshold, Gaussian blur, add.
// The one addition is that the blur radius is SEPARATE PER CHANNEL, which is
// what turns the same three steps into halation and dispersion without any
// extra machinery.
//
//   spread_r == spread_g == spread_b   ordinary achromatic bloom
//   spread_r >  the others             halation -- warm fringe at the edge
//   all three different                dispersion, a spectral edge
//
// WHY PER-CHANNEL RADIUS RATHER THAN A TINT. Halation happens inside film:
// light passes through the emulsion, reflects off the base behind it, and
// re-exposes the layers from below. Red penetrates deepest, so red is what
// comes back. That makes halation a spread that DIFFERS PER CHANNEL, not a
// bloom that has been coloured. The difference is visible: with a per-channel
// spread the glow is neutral in its core, where all three channels reach, and
// goes warm only at the edge where red alone has got to. A tint would colour
// the whole glow uniformly and look nothing like film.
//
// THE THRESHOLD IS ON LUMINANCE, NOT PER CHANNEL, and that is deliberate.
// Thresholding each channel separately would mean a saturated red lamp bloomed
// only in red -- its red channel clears the bar, its green and blue do not --
// so a red light could never throw a white halo, and the three spread sliders
// would collapse into one effect. Deciding WHAT is bright once, on luminance,
// and then letting the sliders control only HOW FAR each channel travels,
// keeps the two decisions independent. The extracted highlight keeps its own
// colour; only its reach is per channel.
//
// AN EARLIER VERSION ALSO DID STARBURSTS AND ANAMORPHIC STREAKS. They were
// removed: they never looked good at any setting, and they cost four of the
// eleven parameters. Diffraction spikes are a genuinely different phenomenon
// from scattering -- a spike ADDS light along a line rather than redistributing
// it -- and folding them into the same threshold-and-blur made both worse. If
// they come back it should be as their own stage.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"

namespace tglab {
namespace {

// Shared by every pass: the constants and the highlight extraction.
const char* kCommon = R"(
cbuffer Constants : register(b0) {
    uint Width;
    uint Height;
    uint C0; uint C1; uint C2; uint C3; uint C4; uint C5; uint C6;
};
float T()         { return asfloat(C0); }
float Knee()      { return asfloat(C1); }
float SpreadR()   { return asfloat(C2); }
float SpreadG()   { return asfloat(C3); }
float SpreadB()   { return asfloat(C4); }
float Intensity() { return asfloat(C5); }
float Dir()       { return asfloat(C6); }

// Intensity, compensated for how much the spread dimmed the glow.
//
// The blur conserves energy, so a source spread over sigma falls as 1/sigma^2.
// Dividing that back out means intensity 1 is roughly "the glow reads as
// bright as the light that made it" whatever the spread, instead of meaning
// something different at every radius. Referenced to sigma 8 so the familiar
// mid-range settings keep about the strength they had.
float Gain() {
    float s = max((SpreadR() + SpreadG() + SpreadB()) / 3.0, 1.0);
    return Intensity() * (s * s) / 64.0;
}

// How much of a pixel survives the threshold, as a fraction of itself.
//
// Decided on LUMINANCE so the choice of what is bright is made once, for the
// pixel as a whole -- see the note at the top of the file. The soft knee
// blends quadratically over a band below the threshold, so the contribution
// grows from zero smoothly rather than switching on at a hard edge, which is
// what stops bloom on a sky looking like a mask.
//
// Rec.709 luma: green dominates because the eye does. A saturated blue light
// therefore has to be far brighter than a green one before it blooms, which is
// correct -- it IS dimmer to look at.
float3 Highlight(float3 c) {
    float t = T();
    float k = max(Knee(), 1e-4);
    float lum = dot(c, float3(0.2126, 0.7152, 0.0722));
    float soft = clamp(lum - t + k, 0.0, 2.0 * k);
    soft = soft * soft / (4.0 * k);
    float contrib = max(max(soft, lum - t), 0.0) / max(lum, 1e-5);
    return c * contrib;
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

// Passes 2 and 3: a separable Gaussian with a PER-CHANNEL sigma.
//
// One kernel weight per channel per tap, so all three colours spread by their
// own amount in a single pass rather than needing three separate blurs. The
// loop bound is the largest of the three radii; narrower channels contribute
// nothing past their own, which costs a few wasted taps and saves two passes.
//
// Each channel is normalised by ITS OWN weight sum. That is what makes this
// redistribute light rather than add or lose it: a channel cut off by the loop
// bound still integrates to 1.
const char* kBlurHlsl = R"(
Texture2D<float4>   Src : register(t0);
RWTexture2D<float4> Dst : register(u0);

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    int2 p = int2(tid.xy);

    float3 sig = max(float3(SpreadR(), SpreadG(), SpreadB()), 0.05);
    float rmax = max(sig.r, max(sig.g, sig.b));
    int R = min(128, int(ceil(rmax * 2.5)));

    float3 sum = 0.0, wsum = 0.0;
    for (int i = -R; i <= R; ++i) {
        int2 q = (Dir() < 0.5) ? int2(p.x + i, p.y) : int2(p.x, p.y + i);
        q = clamp(q, int2(0, 0), int2(int(Width) - 1, int(Height) - 1));
        float3 c = Src[q].rgb;
        float fi = float(i);
        // Per channel: contributes nothing beyond that channel's own reach.
        float3 inside = step(abs(fi), sig * 2.5);
        float3 w = exp(-(fi * fi) / (2.0 * sig * sig)) * inside;
        sum  += c * w;
        wsum += w;
    }
    Dst[tid.xy] = float4(sum / max(wsum, 1e-6), 1.0);
}
)";

// Pass 4: add the glow back.
//
// The glow is scaled by GAIN, which is intensity times a compensation for how
// far the light was spread. A normalised Gaussian CONSERVES energy, so a small
// source spread wide goes faint as 1/sigma^2 -- a 4x4 lamp at sigma 24 peaks
// at 0.4% of itself. That is physically right and made the control unusable:
// widening the spread dimmed the glow, so intensity had to be re-tuned every
// time, and at wide radii the old 0..4 range could not reach a strong look at
// all. Compensating by sigma^2 decouples the two, so intensity means "how
// bright" and spread means "how far", independently. See kCommon's Gain().
const char* kCombineHlsl = R"(
Texture2D<float4>   Src  : register(t0);
Texture2D<float4>   Glow : register(t1);
RWTexture2D<float4> Dst  : register(u0);

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    float4 c = Src[int2(tid.xy)];
    float3 g = Glow[int2(tid.xy)].rgb;
    // ADDITIVE, not a blend. Light adds; the original is never dimmed to make
    // room for the glow, which is what keeps a blown highlight blown.
    Dst[tid.xy] = float4(c.rgb + g * Gain(), c.a);
}
)";

class Bloom : public AlgorithmBase {
public:
    const char* Name()     const override { return "bloom"; }
    const char* Category() const override { return "filter"; }

    PortList Inputs() const override {
        return {{"src", DataType::Image, FormatSpec::Any}};
    }
    PortList Outputs() const override {
        return {{"out", DataType::Image, FormatSpec::SameAsInput}};
    }

    // At intensity 0 nothing is added back, so the pipeline can alias this
    // stage away rather than running four passes to produce its input.
    bool IsNoOp() const override { return float(m_intensity) <= 0.0f; }

    void RunCPU(RunCtx& ctx) override;
    bool HasGPU() const override { return true; }

    // Two scratch planes: one holds the thresholded highlights and takes the
    // vertical pass, the other takes the horizontal. The starburst version
    // needed three.
    int        GpuScratchCount()  const override { return 2; }
    FormatSpec GpuScratchPlanes() const override { return FormatSpec::RGBA32F; }

    std::vector<GpuPass> GpuPasses() const override {
        // Assembled once: GpuPasses returns raw pointers, so rebuilding the
        // strings per call would dangle them the moment the vector went away.
        static const std::string thr = std::string(kCommon) + kThresholdHlsl;
        static const std::string blr = std::string(kCommon) + kBlurHlsl;
        static const std::string cmb = std::string(kCommon) + kCombineHlsl;

        std::vector<GpuPass> p;
        // Plane 0 holds the highlights, 1 takes the horizontal pass, and 0 is
        // reused for the vertical result -- safe because the highlights have
        // been consumed by then.
        p.push_back({thr.c_str(), "threshold", {-1},    {0}});
        p.push_back({blr.c_str(), "blurH",     {0},     {1}});
        p.push_back({blr.c_str(), "blurV",     {1},     {0}});
        p.push_back({cmb.c_str(), "combine",   {-1, 0}, {-1}});
        return p;
    }

    std::vector<uint32_t> GpuPassConstants(int pass) const override {
        auto bits = [](float f) { uint32_t u; std::memcpy(&u, &f, sizeof u); return u; };
        // Pass 1 blurs horizontally and pass 2 vertically; the rest ignore it.
        const float dir = (pass == 2) ? 1.0f : 0.0f;
        // THE SPREAD SCALES; THE GAIN MUST NOT.
        //
        // Gain() in the shader is Intensity * s^2 / 64, compensating for a
        // normalised Gaussian conserving energy -- a wide spread makes a faint
        // glow, so the compensation keeps "intensity 1" meaning one thing at
        // every radius.
        //
        // But s is in PIXELS, and a proxy changes what a pixel is. Handing the
        // shader the scaled spread made it compute the compensation for a
        // narrower blur than the user asked for: at a third scale, spread 24
        // becomes 8 and the gain drops by 9x. The glow vanished during a drag
        // and reappeared at full resolution -- looking like the spread sliders
        // did nothing until the mouse came up.
        //
        // The user changed nothing; only the resolution did. So the intensity
        // is pre-multiplied by the ratio between the gain the UNSCALED spread
        // would have produced and the one the scaled spread will, leaving the
        // shader's arithmetic untouched.
        const float s0 = std::max((float(m_spreadR) + float(m_spreadG) +
                                   float(m_spreadB)) / 3.0f, 1.0f);
        const float s1 = std::max((GpuScaledPx(float(m_spreadR)) +
                                   GpuScaledPx(float(m_spreadG)) +
                                   GpuScaledPx(float(m_spreadB))) / 3.0f, 1.0f);
        const float gainFix = (s1 > 1e-6f) ? (s0 * s0) / (s1 * s1) : 1.0f;

        return {bits(float(m_threshold)),
                bits(std::max(0.01f, float(m_knee))),
                bits(std::max(0.05f, GpuScaledPx(float(m_spreadR)))),
                bits(std::max(0.05f, GpuScaledPx(float(m_spreadG)))),
                bits(std::max(0.05f, GpuScaledPx(float(m_spreadB)))),
                bits(float(m_intensity) * gainFix),
                bits(dir)};
    }

    // The widest channel, at the 2.5-sigma cutoff the blur pass uses. Bloom
    // spreads FAR -- a 24 px spread reaches 60 -- so this is the stage most
    // likely to make a region unprofitable, which is honest rather than a
    // reason to understate it.
    int ReachPixels() const override {
        const float s = std::max(float(m_spreadR),
                                 std::max(float(m_spreadG), float(m_spreadB)));
        return std::min(128, std::max(1, int(std::ceil(s * 2.5f))));
    }

private:
    // 0.6 rather than the 0.8 this had when it thresholded on max(r,g,b).
    // Luminance is always the lower number for anything that is not neutral --
    // a fully saturated red is 0.21 of its own channel value -- so keeping 0.8
    // would have quietly stopped most coloured highlights blooming at all.
    Param<float> m_threshold{this, "threshold", 0.6f, 0.0f, 4.0f,
        {.help = "Luminance above which a pixel blooms. Scene-linear, so 1.0 "
                 "is white and above it is the headroom a raw file holds."}};

    Param<float> m_knee{this, "knee", 0.15f, 0.01f, 1.0f,
        {.help = "Width of the soft band below the threshold. Without it the "
                 "bloom switches on at a hard edge and reads as a mask."}};

    // The three that do the work. Equal = ordinary bloom; red largest =
    // halation; all different = dispersion.
    Param<float> m_spreadR{this, "spread_r", 24.0f, 0.0f, 256.0f,
        {.help = "How far red spreads, in pixels. Raise it above the others "
                 "for halation -- the warm fringe film gives a bright edge."}};

    Param<float> m_spreadG{this, "spread_g", 24.0f, 0.0f, 256.0f,
        {.help = "How far green spreads, in pixels."}};

    Param<float> m_spreadB{this, "spread_b", 24.0f, 0.0f, 256.0f,
        {.help = "How far blue spreads, in pixels. All three equal gives an "
                 "ordinary achromatic bloom; vary them for a spectral edge."}};

    // Compensated for spread (see Gain()), so this means the same thing at
    // every radius. 1.0 is roughly "as bright as the light that made it";
    // above that is a fog or a heavy diffusion filter.
    Param<float> m_intensity{this, "intensity", 0.5f, 0.0f, 8.0f,
        {.help = "How bright the glow is. Additive, so the highlight itself "
                 "is never dimmed. Compensated for spread, so widening the "
                 "spread no longer dims the glow. ~1 reads as strong as the "
                 "source; 3+ is a foggy night."}};
};

REGISTER_ALGORITHM(Bloom);

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
    const float sig[3] = {std::max(0.05f, ctx.ScaledPx(float(m_spreadR))),
                          std::max(0.05f, ctx.ScaledPx(float(m_spreadG))),
                          std::max(0.05f, ctx.ScaledPx(float(m_spreadB)))};

    // Spread compensation, matching Gain() in the shader -- see the note above
    // kCombineHlsl for why the raw intensity is not used directly.
    //
    // From the UNSCALED spreads. The compensation is a statement about the
    // user's setting, not about the raster: it exists so "intensity 1" means
    // one thing at every radius, and s is in pixels, so computing it from the
    // proxy-scaled values would make it mean something different at every
    // resolution. At a third scale that cost a factor of nine and the glow
    // disappeared for the length of a drag.
    const float sAvg = std::max(1.0f, (float(m_spreadR) + float(m_spreadG) +
                                       float(m_spreadB)) / 3.0f);
    const float amt  = float(m_intensity) * (sAvg * sAvg) / 64.0f;

    // Highlights, thresholded on luminance -- see the note at the top.
    std::vector<float> hi(size_t(w) * size_t(h) * 3, 0.0f);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const float* p = in.At(x, y);
            const float r = p[0];
            const float g = (ch >= 3) ? p[1] : p[0];
            const float b = (ch >= 3) ? p[2] : p[0];
            const float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            float soft = std::clamp(lum - thr + knee, 0.0f, 2.0f * knee);
            soft = soft * soft / (4.0f * knee);
            const float c = std::max(std::max(soft, lum - thr), 0.0f) /
                            std::max(lum, 1e-5f);
            const size_t i = (size_t(y) * size_t(w) + size_t(x)) * 3;
            hi[i + 0] = r * c;
            hi[i + 1] = g * c;
            hi[i + 2] = b * c;
        }

    const int R = std::min(128, int(std::ceil(
        std::max(sig[0], std::max(sig[1], sig[2])) * 2.5f)));

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
                        if (std::fabs(float(i)) > sig[c] * 2.5f) continue;
                        const float wgt =
                            std::exp(-float(i * i) / (2.0f * sig[c] * sig[c]));
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

    // Add back. Additive, so a blown highlight stays blown.
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const float* p = in.At(x, y);
            float* o = out.At(x, y);
            const size_t i = (size_t(y) * size_t(w) + size_t(x)) * 3;
            for (int c = 0; c < ch; ++c) {
                if (c < 3) o[c] = p[c] + glow[i + size_t(c)] * amt;
                else       o[c] = p[c];
            }
        }

    // Unpack gave a working COPY, not a live view of the destination -- the
    // writes above go nowhere without this. Omitting it produces an entirely
    // black output with no error, which is exactly what test_filters' "no
    // algorithm leaves its output untouched" guard exists to catch.
    out.PackInto(dst);
}

}  // namespace
}  // namespace tglab
