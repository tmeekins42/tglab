// color_grade — three-way colour grading: shadows, midtones, highlights.
//
// Each range has a colour wheel -- a hue and a strength -- and a lightness, and
// a fourth, global wheel applies to every pixel. In OkLab:
//
//   tint       the wheel's colour is ADDED to the pixel's a/b, scaled by the
//              range's weight: strength 1 adds 0.10 of OkLab chroma, a strong
//              but not garish grade. Adding rather than rotating is what a
//              grading wheel does -- a grey shadow takes the colour, and a
//              coloured one is pushed toward it.
//   lightness  the range's weight times up to 0.15 of OkLab L, either way.
//
// THE RANGES come from OkLab lightness, which is perceptual, so "midtones"
// means what it looks like: shadows fall from 1 at black to 0 at the pivot,
// highlights rise from 0 at the pivot to 1 at white, and midtones are what is
// left, peaking at the pivot. `balance` moves the pivot (middle grey, L 0.57,
// at 0): positive gives the highlights more of the picture. `blending` widens
// or narrows the ranges' reach -- low is a crisp split, high a gradual one.
//
// Pure black is not tinted -- a coloured black is out of gamut -- the tint
// rising over the first sixth of lightness. Encoded images are taken to
// linear and back; a linear image's highlights above 1 count as highlights.
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "../../algo_util/color.h"
#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"
#include "../../core/parallel.h"

namespace tglab {
namespace {

constexpr float kTint = 0.10f;    // OkLab chroma at strength 1
constexpr float kLight = 0.15f;   // OkLab L at lightness +-1
constexpr float kGrey = 0.57f;    // OkLab L of 18% grey: the pivot at balance 0

inline float Smoothstep(float a, float b, float x) {
    const float t = std::clamp((x - a) / (b - a), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// One wheel, as the a/b offset and the lightness offset it applies at weight 1.
struct Wheel { float a, b, l; };

const char* const kGradeHead = R"(
Texture2D<float4>   Src : register(t0);
RWTexture2D<float4> Dst : register(u0);
cbuffer Params : register(b0) {
    uint Width;
    uint Height;
    uint Linear;
    uint PivotBits, PowerBits;
    uint SaBits, SbBits, SlBits;    // shadows: a, b, lightness offsets
    uint MaBits, MbBits, MlBits;    // midtones
    uint HaBits, HbBits, HlBits;    // highlights
    uint GaBits, GbBits, GlBits;    // global
};
)";

const char* const kGradeMain = R"(
float3 ToLinear(float3 c) {
    return Linear != 0 ? c
        : select(c <= 0.04045, c / 12.92, pow((max(c, 0.0) + 0.055) / 1.055, 2.4));
}
float3 FromLinear(float3 c) {
    if (Linear != 0) return max(c, 0.0);
    c = saturate(c);
    return select(c <= 0.0031308, c * 12.92, 1.055 * pow(c, 1.0 / 2.4) - 0.055);
}
float Smooth(float a, float b, float x) { float t = saturate((x - a) / (b - a)); return t * t * (3.0 - 2.0 * t); }

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    float4 c = Src[tid.xy];
    float3 lab = LinearSrgbToOkLab(ToLinear(c.rgb));
    float p = asfloat(PivotBits), e = asfloat(PowerBits);
    float t = saturate(lab.x);
    float ws = pow(1.0 - Smooth(0.0, p, t), e);
    float wh = pow(Smooth(p, 1.0, t), e);
    float wm = max(0.0, 1.0 - ws - wh);
    float3 d = ws * float3(asfloat(SaBits), asfloat(SbBits), asfloat(SlBits)) +
               wm * float3(asfloat(MaBits), asfloat(MbBits), asfloat(MlBits)) +
               wh * float3(asfloat(HaBits), asfloat(HbBits), asfloat(HlBits)) +
                    float3(asfloat(GaBits), asfloat(GbBits), asfloat(GlBits));
    lab.yz += d.xy * saturate(lab.x * 6.0);
    lab.x = max(0.0, lab.x + d.z);
    Dst[tid.xy] = float4(FromLinear(OkLabToLinearSrgb(lab)), c.a);
}
)";

class ColorGrade : public AlgorithmBase {
public:
    const char* Name()     const override { return "color_grade"; }
    const char* Category() const override { return "color"; }

    PortList Inputs()  const override { return {{"src", DataType::Image, FormatSpec::Any}}; }
    PortList Outputs() const override { return {{"out", DataType::Image, FormatSpec::SameAsInput}}; }

    bool IsNoOp() const override {
        return float(m_shSat) == 0 && float(m_shLum) == 0 && float(m_midSat) == 0 &&
               float(m_midLum) == 0 && float(m_hiSat) == 0 && float(m_hiLum) == 0 &&
               float(m_glSat) == 0 && float(m_glLum) == 0;
    }

    void RunCPU(RunCtx& ctx) override {
        const ImageView src = ctx.In(0);
        ImageView dst = ctx.Out(0);
        if (!src.Valid() || !dst.Valid()) return;
        PixelBuffer in, out;
        in.Unpack(src);
        if (!in.Valid()) return;
        out.AllocLike(in);
        const int w = in.Width(), h = in.Height(), ch = in.Channels();
        const float vs = in.ValueScale();
        const bool linear = ctx.InDesc(0).linear;
        const float* px = in.Data().data();
        float* o = out.Data().data();
        const size_t n = size_t(w) * size_t(h);
        if (ch < 3) {
            std::copy(px, px + n * size_t(ch), o);
            out.PackInto(dst);
            return;
        }
        const Wheel sh = Of(m_shHue, m_shSat, m_shLum), mid = Of(m_midHue, m_midSat, m_midLum),
                    hi = Of(m_hiHue, m_hiSat, m_hiLum), gl = Of(m_glHue, m_glSat, m_glLum);
        const float p = Pivot(), e = Power();
        auto lin = [&](float v) { return linear ? v / vs : SrgbToLinear(v / vs); };
        auto enc = [&](float v) {
            return linear ? std::max(v, 0.0f) * vs : LinearToSrgb(std::clamp(v, 0.0f, 1.0f)) * vs;
        };
        ParallelFor(size_t(h), [&](size_t y) {
            for (int x = 0; x < w; ++x) {
                const size_t i = y * size_t(w) + size_t(x);
                const float* q0 = &px[i * size_t(ch)];
                float* q = &o[i * size_t(ch)];
                for (int c = 0; c < ch; ++c) q[c] = q0[c];
                const float rgb[3] = {lin(q0[0]), lin(q0[1]), lin(q0[2])};
                float lab[3];
                LinearSrgbToOkLab(rgb, lab);
                const float t = std::clamp(lab[0], 0.0f, 1.0f);
                const float ws = std::pow(1.0f - Smoothstep(0.0f, p, t), e);
                const float wh = std::pow(Smoothstep(p, 1.0f, t), e);
                const float wm = std::max(0.0f, 1.0f - ws - wh);
                const float da = ws * sh.a + wm * mid.a + wh * hi.a + gl.a;
                const float db = ws * sh.b + wm * mid.b + wh * hi.b + gl.b;
                const float dl = ws * sh.l + wm * mid.l + wh * hi.l + gl.l;
                const float k = std::clamp(lab[0] * 6.0f, 0.0f, 1.0f);
                lab[1] += da * k;
                lab[2] += db * k;
                lab[0] = std::max(0.0f, lab[0] + dl);
                float back[3];
                OkLabToLinearSrgb(lab, back);
                for (int c = 0; c < 3; ++c) q[c] = enc(back[c]);
            }
        });
        out.PackInto(dst);
    }

    // --- GPU ----------------------------------------------------------------
    bool HasGPU() const override { return true; }

    void PrepareGpu(const std::vector<ImageDesc>& inputs) override {
        if (!inputs.empty()) m_linear = inputs[0].linear;
    }

    const char* GpuSource() const override {
        static const std::string src = std::string(kGradeHead) + kOkLabHlsl + kGradeMain;
        return src.c_str();
    }

    std::vector<uint32_t> GpuConstants(int) const override {
        std::vector<uint32_t> k = {m_linear ? 1u : 0u, FloatBits(Pivot()), FloatBits(Power())};
        for (const Wheel& wh : {Of(m_shHue, m_shSat, m_shLum), Of(m_midHue, m_midSat, m_midLum),
                                Of(m_hiHue, m_hiSat, m_hiLum), Of(m_glHue, m_glSat, m_glLum)}) {
            k.push_back(FloatBits(wh.a));
            k.push_back(FloatBits(wh.b));
            k.push_back(FloatBits(wh.l));
        }
        return k;
    }

private:
    static Wheel Of(const Param<float>& hue, const Param<float>& sat, const Param<float>& lum) {
        const float r = float(hue) * (3.14159265f / 180.0f);
        return {kTint * float(sat) * std::cos(r), kTint * float(sat) * std::sin(r), kLight * float(lum)};
    }
    float Pivot() const { return std::clamp(kGrey - 0.25f * float(m_balance), 0.2f, 0.9f); }
    float Power() const { return 1.5f - float(m_blending); }

    static constexpr const char* kHueHelp =
        "The wheel's colour, as an OkLab hue angle: red ~29, orange ~58, yellow "
        "~110, green ~142, cyan ~195, blue ~264, magenta ~328.";

    Param<float> m_shHue{this, "shadows_hue", 220.0f, 0.0f, 360.0f, {.help = kHueHelp, .step = 1.0}};
    Param<float> m_shSat{this, "shadows_strength", 0.0f, 0.0f, 1.0f,
        {.help = "How much of the shadow wheel's colour to add to the shadows.", .step = 0.01}};
    Param<float> m_shLum{this, "shadows_lightness", 0.0f, -1.0f, 1.0f,
        {.help = "Lift (+) or sink (-) the shadows.", .step = 0.01}};
    Param<float> m_midHue{this, "midtones_hue", 60.0f, 0.0f, 360.0f, {.help = kHueHelp, .step = 1.0}};
    Param<float> m_midSat{this, "midtones_strength", 0.0f, 0.0f, 1.0f,
        {.help = "How much of the midtone wheel's colour to add to the midtones.", .step = 0.01}};
    Param<float> m_midLum{this, "midtones_lightness", 0.0f, -1.0f, 1.0f,
        {.help = "Brighten (+) or darken (-) the midtones.", .step = 0.01}};
    Param<float> m_hiHue{this, "highlights_hue", 60.0f, 0.0f, 360.0f, {.help = kHueHelp, .step = 1.0}};
    Param<float> m_hiSat{this, "highlights_strength", 0.0f, 0.0f, 1.0f,
        {.help = "How much of the highlight wheel's colour to add to the highlights.", .step = 0.01}};
    Param<float> m_hiLum{this, "highlights_lightness", 0.0f, -1.0f, 1.0f,
        {.help = "Brighten (+) or pull down (-) the highlights.", .step = 0.01}};
    Param<float> m_glHue{this, "global_hue", 60.0f, 0.0f, 360.0f, {.help = kHueHelp, .step = 1.0}};
    Param<float> m_glSat{this, "global_strength", 0.0f, 0.0f, 1.0f,
        {.help = "A tint over the whole picture.", .step = 0.01}};
    Param<float> m_glLum{this, "global_lightness", 0.0f, -1.0f, 1.0f,
        {.help = "Lightness over the whole picture.", .step = 0.01}};
    Param<float> m_balance{this, "balance", 0.0f, -1.0f, 1.0f,
        {.help = "Where shadows end and highlights begin: positive gives more "
                 "of the picture to the highlights, negative to the shadows.", .step = 0.01}};
    Param<float> m_blending{this, "blending", 0.5f, 0.0f, 1.0f,
        {.help = "How far each range reaches into its neighbours: low is a "
                 "crisp split, high a gradual one.", .step = 0.01}};

    bool m_linear = false;   // set by PrepareGpu
};

REGISTER_ALGORITHM(ColorGrade);

}  // namespace
}  // namespace tglab
