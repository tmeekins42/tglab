// bloom — threshold the highlights, spread them, add them back.
//
// Three steps, and that is the whole algorithm: threshold, Gaussian blur, add.
//
// THE BLUR IS THREE BOX FILTERS PER DIRECTION, each a running sum, which is a
// Gaussian to within a few percent and costs the same at every radius. A direct
// Gaussian had to cap its window (128 px), and a spread of 94 at full
// resolution needs 235: cut off at 1.4 sigma and renormalised, the kernel was
// nearly flat, and a flat kernel run across and then down is a SQUARE. The
// preview, at a quarter of the spread, sat inside the cap and looked right --
// the glow turned into a box the moment the slider was let go.
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
#include "../../algo_util/color.h"
#include "../../core/algorithm.h"

namespace tglab {
namespace {

// Shared by every pass: the constants and the highlight extraction.
const char* kCommon = R"(
cbuffer Constants : register(b0) {
    uint Width;
    uint Height;
    uint C0; uint C1; uint C2; uint C3; uint C4; uint C5; uint C6;
    uint RadR; uint RadG; uint RadB;   // box radii, per channel
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
    float lum = Luma(c);
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

// Passes 2-7: three box passes across, then three down, each a running sum
// with a PER-CHANNEL radius -- so all three colours spread by their own amount
// in the same passes. A thread walks kBoxRun pixels, gathering each channel's
// window once at the start of its run and sliding it (add the sample entering,
// drop the one leaving); re-gathering every run keeps float error in the sums
// from building. Samples past the edge clamp to it, as before.
const char* kBlurHlsl = R"(
Texture2D<float4>   Src : register(t0);
RWTexture2D<float4> Dst : register(u0);
static const int kBoxRun = 64;

float4 At(int i, int fixedC, bool across) {
    int2 q = across ? int2(clamp(i, 0, int(Width) - 1), fixedC)
                    : int2(fixedC, clamp(i, 0, int(Height) - 1));
    return Src[q];
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    bool across = Dir() < 0.5;
    int len   = across ? int(Width) : int(Height);
    int other = across ? int(Height) : int(Width);
    int i0 = int(across ? tid.x : tid.y) * kBoxRun;
    int fc = int(across ? tid.y : tid.x);
    if (i0 >= len || fc >= other) return;
    int3 r = int3(RadR, RadG, RadB);
    float3 sum = 0.0;
    [unroll] for (int c = 0; c < 3; ++c)
        for (int k = -r[c]; k <= r[c]; ++k) sum[c] += At(i0 + k, fc, across)[c];
    float3 inv = 1.0 / float3(2 * r + 1);
    int i1 = min(i0 + kBoxRun, len);
    for (int i = i0; i < i1; ++i) {
        if (i > i0) {
            [unroll] for (int c = 0; c < 3; ++c)
                sum[c] += At(i + r[c], fc, across)[c] - At(i - r[c] - 1, fc, across)[c];
        }
        int2 o = across ? int2(i, fc) : int2(fc, i);
        Dst[o] = float4(sum * inv, 1.0);
    }
}
)";
constexpr uint32_t kBoxRun = 64;   // must match the HLSL above

// The box radius whose three passes give a Gaussian of `sigma`: three boxes of
// width n have variance 3 (n^2 - 1) / 12.
inline int BoxRadius(float sigma) {
    const float n = std::sqrt(4.0f * sigma * sigma + 1.0f);
    return std::max(0, int(std::lround((n - 1.0f) * 0.5f)));
}

// One box pass over three-channel float data, exactly as the shader: per
// channel radius, edge clamped, a running sum restarted every kBoxRun pixels.
void BoxPass(const std::vector<float>& src, std::vector<float>& dst, int w, int h,
             const int* r, bool across) {
    const int len = across ? w : h, other = across ? h : w;
    auto at = [&](int i, int f, int c) {
        const int x = across ? std::clamp(i, 0, w - 1) : f;
        const int y = across ? f : std::clamp(i, 0, h - 1);
        return src[(size_t(y) * size_t(w) + size_t(x)) * 3 + size_t(c)];
    };
    for (int f = 0; f < other; ++f)
        for (int i0 = 0; i0 < len; i0 += int(kBoxRun)) {
            float sum[3] = {0, 0, 0};
            for (int c = 0; c < 3; ++c)
                for (int k = -r[c]; k <= r[c]; ++k) sum[c] += at(i0 + k, f, c);
            const int i1 = std::min(i0 + int(kBoxRun), len);
            for (int i = i0; i < i1; ++i) {
                if (i > i0)
                    for (int c = 0; c < 3; ++c)
                        sum[c] += at(i + r[c], f, c) - at(i - r[c] - 1, f, c);
                const int x = across ? i : f, y = across ? f : i;
                for (int c = 0; c < 3; ++c)
                    dst[(size_t(y) * size_t(w) + size_t(x)) * 3 + size_t(c)] =
                        sum[c] * (1.0f / float(2 * r[c] + 1));
            }
        }
}

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
    // An EFFECT, not a correction: a glow is a look, chosen per image, not a default state.
    bool DefaultOff() const override { return true; }

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
        // Plane 0 holds the highlights; the six box passes ping-pong between
        // the two planes and leave the glow in 0.
        p.push_back({thr.c_str(), "threshold", {-1},    {0}});
        p.push_back({blr.c_str(), "boxH1",     {0},     {1}, kBoxRun, 1});
        p.push_back({blr.c_str(), "boxH2",     {1},     {0}, kBoxRun, 1});
        p.push_back({blr.c_str(), "boxH3",     {0},     {1}, kBoxRun, 1});
        p.push_back({blr.c_str(), "boxV1",     {1},     {0}, 1, kBoxRun});
        p.push_back({blr.c_str(), "boxV2",     {0},     {1}, 1, kBoxRun});
        p.push_back({blr.c_str(), "boxV3",     {1},     {0}, 1, kBoxRun});
        p.push_back({cmb.c_str(), "combine",   {-1, 0}, {-1}});
        return p;
    }

    std::vector<uint32_t> GpuPassConstants(int pass) const override {
        // Passes 1-3 blur across, 4-6 down; the rest ignore it.
        const float dir = (pass >= 4 && pass <= 6) ? 1.0f : 0.0f;
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

        return {FloatBits(float(m_threshold)),
                FloatBits(std::max(0.01f, float(m_knee))),
                FloatBits(std::max(0.05f, GpuScaledPx(float(m_spreadR)))),
                FloatBits(std::max(0.05f, GpuScaledPx(float(m_spreadG)))),
                FloatBits(std::max(0.05f, GpuScaledPx(float(m_spreadB)))),
                FloatBits(float(m_intensity) * gainFix),
                FloatBits(dir),
                uint32_t(BoxRadius(std::max(0.05f, GpuScaledPx(float(m_spreadR))))),
                uint32_t(BoxRadius(std::max(0.05f, GpuScaledPx(float(m_spreadG))))),
                uint32_t(BoxRadius(std::max(0.05f, GpuScaledPx(float(m_spreadB)))))};
    }

    // The widest channel: three box passes reach three box radii. Bloom
    // spreads FAR -- a 24 px spread reaches ~72 -- so this is the stage most
    // likely to make a region unprofitable, which is honest rather than a
    // reason to understate it. No cap: the blur has none now either.
    int ReachPixels() const override {
        const float s = std::max(float(m_spreadR),
                                 std::max(float(m_spreadG), float(m_spreadB)));
        return std::max(1, 3 * BoxRadius(s));
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
            const float lum = Luma(r, g, b);
            float soft = std::clamp(lum - thr + knee, 0.0f, 2.0f * knee);
            soft = soft * soft / (4.0f * knee);
            const float c = std::max(std::max(soft, lum - thr), 0.0f) /
                            std::max(lum, 1e-5f);
            const size_t i = (size_t(y) * size_t(w) + size_t(x)) * 3;
            hi[i + 0] = r * c;
            hi[i + 1] = g * c;
            hi[i + 2] = b * c;
        }

    // Three box passes across, three down: see kBlurHlsl.
    const int r[3] = {BoxRadius(sig[0]), BoxRadius(sig[1]), BoxRadius(sig[2])};
    std::vector<float> glow(hi.size(), 0.0f);
    for (int pass = 0; pass < 6; ++pass) {
        if (ctx.Cancelled()) return;
        BoxPass(hi, glow, w, h, r, pass < 3);
        hi.swap(glow);
    }
    glow.swap(hi);

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
