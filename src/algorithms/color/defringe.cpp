// defringe — take the colour out of purple and green fringes along edges.
//
// WHAT A FRINGE IS. Axial (longitudinal) chromatic aberration: the lens brings
// the colours to focus at different DEPTHS, so at a high-contrast edge one
// colour is sharp and the others are a halo of blur -- purple or magenta on the
// bright side of a backlit edge, green on the far side of focus. Lateral CA
// moves the colour planes sideways and lens_correct undoes it by scaling them;
// axial CA cannot be undone geometrically, because nothing is displaced -- the
// colour is a blur that exists only near edges. Demosaicing a hard edge,
// especially beside a blown highlight, adds false colour of the same kind.
//
// HOW. In OkLab, where hue is an angle that stays put as lightness and chroma
// change:
//
//   1. Near an edge: the spread of OkLab lightness (brightest minus darkest)
//      over `radius` pixels. A fringe is a few pixels wide and hugs a strong
//      edge; a purple OBJECT has a flat interior, so it is left alone there.
//   2. In a fringe's hue: inside the purple window (or the green one), with
//      soft shoulders so the edge of the window does not band.
//   3. Chroma scaled toward zero by amount x hue x edge. Lightness is never
//      touched, so the edge keeps its contrast -- only its colour goes.
//
// (lightcraft defringes the same way, in OkLab hue windows next to high local
// contrast: crates/pipeline/src/optics.rs.)
//
// Encoded images are taken to linear for OkLab and back.
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

// Lightness spread over the window, in OkLab L (0..1 for SDR), from no effect
// to the full amount: a gentle gradient is not an edge.
constexpr float kEdgeLo = 0.06f, kEdgeHi = 0.20f;
// Hue window shoulders, in degrees.
constexpr float kHueSoft = 15.0f;

inline float Smoothstep(float a, float b, float x) {
    const float t = std::clamp((x - a) / (b - a), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// Membership of hue h (degrees) in [lo, hi], which may wrap past 360.
inline float HueWeight(float h, float lo, float hi) {
    auto wrap = [](float d) { d = std::fmod(d, 360.0f); return d < 0.0f ? d + 360.0f : d; };
    h = wrap(h); lo = wrap(lo); hi = wrap(hi);
    const bool inside = lo <= hi ? (h >= lo && h <= hi) : (h >= lo || h <= hi);
    if (inside) return 1.0f;
    auto circ = [](float a, float b) { const float d = std::fabs(a - b); return std::min(d, 360.0f - d); };
    const float d = std::min(circ(h, lo), circ(h, hi));
    return 1.0f - Smoothstep(0.0f, kHueSoft, d);
}

int RadiusPx(float r) { return std::clamp(int(std::lround(r)), 1, 32); }

const char* const kCommon = R"(
Texture2D<float4>   T0 : register(t0);
Texture2D<float4>   T1 : register(t1);
RWTexture2D<float4> U0 : register(u0);
cbuffer Params : register(b0) {
    uint Width;
    uint Height;
    uint Radius;
    uint Vertical;
    uint Linear;
    uint PurpleBits, PurpleLoBits, PurpleHiBits;
    uint GreenBits, GreenLoBits, GreenHiBits;
};
static const float kEdgeLo = 0.06, kEdgeHi = 0.20, kHueSoft = 15.0;
float Smooth(float a, float b, float x) { float t = saturate((x - a) / (b - a)); return t * t * (3.0 - 2.0 * t); }
float3 ToLinear(float3 c) {
    return Linear != 0 ? c
        : select(c <= 0.04045, c / 12.92, pow((max(c, 0.0) + 0.055) / 1.055, 2.4));
}
float3 FromLinear(float3 c) {
    if (Linear != 0) return c;
    c = saturate(c);
    return select(c <= 0.0031308, c * 12.92, 1.055 * pow(c, 1.0 / 2.4) - 0.055);
}
float Wrap360(float d) { d = fmod(d, 360.0); return d < 0.0 ? d + 360.0 : d; }
float HueWeight(float h, float lo, float hi) {
    h = Wrap360(h); lo = Wrap360(lo); hi = Wrap360(hi);
    bool inside = lo <= hi ? (h >= lo && h <= hi) : (h >= lo || h <= hi);
    if (inside) return 1.0;
    float d1 = abs(h - lo); d1 = min(d1, 360.0 - d1);
    float d2 = abs(h - hi); d2 = min(d2, 360.0 - d2);
    return 1.0 - Smooth(0.0, kHueSoft, min(d1, d2));
}
)";

const char* const kLightness = R"(
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    float L = LinearSrgbToOkLab(ToLinear(T0[tid.xy].rgb)).x;
    U0[tid.xy] = float4(L, L, 0, 1);
}
)";

// Min and max over the radius, across then down.
const char* const kMinMax = R"(
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    int r = int(Radius);
    float lo = 1e30, hi = -1e30;
    for (int k = -r; k <= r; ++k) {
        int2 q = Vertical != 0 ? int2(tid.x, clamp(int(tid.y) + k, 0, int(Height) - 1))
                               : int2(clamp(int(tid.x) + k, 0, int(Width) - 1), tid.y);
        float2 v = T0[q].xy;
        lo = min(lo, v.x); hi = max(hi, v.y);
    }
    U0[tid.xy] = float4(lo, hi, 0, 1);
}
)";

const char* const kApply = R"(
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    float4 c = T0[tid.xy];
    float2 mm = T1[tid.xy].xy;
    float edge = Smooth(kEdgeLo, kEdgeHi, mm.y - mm.x);
    if (edge <= 0.0) { U0[tid.xy] = c; return; }
    float3 lab = LinearSrgbToOkLab(ToLinear(c.rgb));
    float h = degrees(atan2(lab.z, lab.y));
    float k = asfloat(PurpleBits) * HueWeight(h, asfloat(PurpleLoBits), asfloat(PurpleHiBits)) +
              asfloat(GreenBits)  * HueWeight(h, asfloat(GreenLoBits),  asfloat(GreenHiBits));
    float keep = 1.0 - saturate(k * edge);
    lab.yz *= keep;
    U0[tid.xy] = float4(FromLinear(OkLabToLinearSrgb(lab)), c.a);
}
)";

class Defringe : public AlgorithmBase {
public:
    const char* Name()     const override { return "defringe"; }
    const char* Category() const override { return "color"; }

    PortList Inputs()  const override { return {{"src", DataType::Image, FormatSpec::Any}}; }
    PortList Outputs() const override { return {{"out", DataType::Image, FormatSpec::SameAsInput}}; }

    bool IsNoOp() const override { return float(m_purple) == 0.0f && float(m_green) == 0.0f; }
    int ReachPixels() const override { return RadiusPx(float(m_radius)); }

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
        if (ch < 3) {   // no colour, no fringe
            std::copy(px, px + n * size_t(ch), o);
            out.PackInto(dst);
            return;
        }
        auto lin = [&](float v) { return linear ? v / vs : SrgbToLinear(v / vs); };
        auto enc = [&](float v) { return linear ? v * vs : LinearToSrgb(v) * vs; };

        std::vector<float> L(n), lo(n), hi(n), lo2(n), hi2(n);
        ParallelFor(size_t(h), [&](size_t y) {
            for (int x = 0; x < w; ++x) {
                const size_t i = y * size_t(w) + size_t(x);
                const float* p = &px[i * size_t(ch)];
                const float rgb[3] = {lin(p[0]), lin(p[1]), lin(p[2])};
                float lab[3];
                LinearSrgbToOkLab(rgb, lab);
                L[i] = lab[0];
            }
        });
        const int r = RadiusPx(ctx.ScaledPx(float(m_radius)));
        ParallelFor(size_t(h), [&](size_t y) {
            for (int x = 0; x < w; ++x) {
                float a = 1e30f, b = -1e30f;
                for (int k = -r; k <= r; ++k) {
                    const float v = L[y * size_t(w) + size_t(std::clamp(x + k, 0, w - 1))];
                    a = std::min(a, v); b = std::max(b, v);
                }
                lo[y * size_t(w) + size_t(x)] = a; hi[y * size_t(w) + size_t(x)] = b;
            }
        });
        ParallelFor(size_t(h), [&](size_t y) {
            for (int x = 0; x < w; ++x) {
                float a = 1e30f, b = -1e30f;
                for (int k = -r; k <= r; ++k) {
                    const size_t j = size_t(std::clamp(int(y) + k, 0, h - 1)) * size_t(w) + size_t(x);
                    a = std::min(a, lo[j]); b = std::max(b, hi[j]);
                }
                lo2[y * size_t(w) + size_t(x)] = a; hi2[y * size_t(w) + size_t(x)] = b;
            }
        });

        const float pa = float(m_purple), ga = float(m_green);
        const float plo = float(m_purpleLo), phi = float(m_purpleHi);
        const float glo = float(m_greenLo), ghi = float(m_greenHi);
        ParallelFor(size_t(h), [&](size_t y) {
            for (int x = 0; x < w; ++x) {
                const size_t i = y * size_t(w) + size_t(x);
                const float* p = &px[i * size_t(ch)];
                float* q = &o[i * size_t(ch)];
                for (int c = 0; c < ch; ++c) q[c] = p[c];
                const float edge = Smoothstep(kEdgeLo, kEdgeHi, hi2[i] - lo2[i]);
                if (edge <= 0.0f) continue;
                const float rgb[3] = {lin(p[0]), lin(p[1]), lin(p[2])};
                float lab[3];
                LinearSrgbToOkLab(rgb, lab);
                const float hue = std::atan2(lab[2], lab[1]) * (180.0f / 3.14159265f);
                const float k = pa * HueWeight(hue, plo, phi) + ga * HueWeight(hue, glo, ghi);
                const float keep = 1.0f - std::clamp(k * edge, 0.0f, 1.0f);
                lab[1] *= keep; lab[2] *= keep;
                float back[3];
                OkLabToLinearSrgb(lab, back);
                for (int c = 0; c < 3; ++c) q[c] = enc(back[c]);
            }
        });
        out.PackInto(dst);
    }

    // --- GPU ----------------------------------------------------------------
    bool HasGPU() const override { return true; }
    int        GpuScratchCount()  const override { return 3; }
    FormatSpec GpuScratchPlanes() const override { return FormatSpec::RG32F; }

    void PrepareGpu(const std::vector<ImageDesc>& inputs) override {
        if (!inputs.empty()) m_linear = inputs[0].linear;
    }

    std::vector<GpuPass> GpuPasses() const override {
        static const std::string pre = std::string(kOkLabHlsl) + kCommon;
        static const std::string light = pre + kLightness, mm = pre + kMinMax, app = pre + kApply;
        return {{light.c_str(), "lightness", {-1}, {0}},
                {mm.c_str(), "minmaxH", {0}, {1}},
                {mm.c_str(), "minmaxV", {1}, {2}},
                {app.c_str(), "apply", {-1, 2}, {-1}}};
    }

    std::vector<uint32_t> GpuPassConstants(int pass) const override {
        return {uint32_t(RadiusPx(GpuScaledPx(float(m_radius)))), pass == 2 ? 1u : 0u,
                m_linear ? 1u : 0u,
                FloatBits(float(m_purple)), FloatBits(float(m_purpleLo)), FloatBits(float(m_purpleHi)),
                FloatBits(float(m_green)), FloatBits(float(m_greenLo)), FloatBits(float(m_greenHi))};
    }

private:
    Param<float> m_purple{this, "purple", 0.6f, 0.0f, 1.0f,
        {.help = "How much colour to take out of purple and magenta fringes "
                 "along edges -- the bright side of a backlit edge, a demosaiced "
                 "blown highlight. 0 leaves them.",
         .step = 0.01}};
    Param<float> m_purpleLo{this, "purple_hue_lo", 270.0f, 0.0f, 360.0f,
        {.help = "Where the purple window starts, as an OkLab hue angle "
                 "(blue ~265, magenta ~330, red ~30).", .step = 1.0}};
    Param<float> m_purpleHi{this, "purple_hue_hi", 350.0f, 0.0f, 360.0f,
        {.help = "Where the purple window ends.", .step = 1.0}};
    Param<float> m_green{this, "green", 0.0f, 0.0f, 1.0f,
        {.help = "The same for green fringes -- the far side of focus. Off by "
                 "default: green edges are far more often foliage than "
                 "aberration.", .step = 0.01}};
    // From 128, not 100: with the 15-degree shoulder that ends the window just
    // above yellow (~110). Starting at 100 took the colour out of sodium-lit
    // girders along every edge of them.
    Param<float> m_greenLo{this, "green_hue_lo", 128.0f, 0.0f, 360.0f,
        {.help = "Where the green window starts (yellow ~110, green ~140). "
                 "Kept above yellow, or warm light along edges goes grey.", .step = 1.0}};
    Param<float> m_greenHi{this, "green_hue_hi", 185.0f, 0.0f, 360.0f,
        {.help = "Where the green window ends (cyan ~195).", .step = 1.0}};
    Param<float> m_radius{this, "radius", 4.0f, 1.0f, 32.0f,
        {.help = "How far from an edge a fringe reaches, in pixels.", .step = 0.5}};

    bool m_linear = false;   // set by PrepareGpu
};

REGISTER_ALGORITHM(Defringe);

}  // namespace
}  // namespace tglab
