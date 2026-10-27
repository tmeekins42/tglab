// color_mixer — hue, saturation and luminance by colour band (an HSL mixer).
//
// EIGHT BANDS around the OkLab hue circle, centred where the sRGB colours they
// are named for fall: red 29, orange 58, yellow 110, green 142, aqua 195,
// blue 264, purple 295, magenta 328 degrees. A pixel's hue lies between two
// neighbouring centres and takes their settings blended by a raised cosine,
// so the bands are a partition of unity: no hue is moved twice, none is
// missed, and nothing bands where one ends and the next begins.
//
// WHY OkLab. Its hue is an angle that stays put as lightness and chroma
// change, so "the blues" are the same pixels in a shadow and a highlight, and
// a hue shift does not drag lightness with it. An HSL mixer built on HSV hue
// does neither: dark blue and light blue sit at different HSV hues, and
// rotating yellow toward green visibly darkens it.
//
// WHAT EACH SLIDER DOES, at +-1:
//   hue          rotates the band's hue by up to 30 degrees
//   saturation   scales its chroma, from grey (-1) to double (+1)
//   luminance    raises or lowers its OkLab lightness by up to 40%
// Each is weighted by how coloured the pixel is (chroma), so greys -- which
// have no hue to belong to a band -- are left alone, and near-greys move a
// little rather than jumping. Encoded images are taken to linear and back.
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

constexpr int kBands = 8;
constexpr float kCentres[kBands] = {29.0f, 58.0f, 110.0f, 142.0f, 195.0f, 264.0f, 295.0f, 328.0f};
const char* const kBandNames[kBands] = {"red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta"};
constexpr float kHueRange = 30.0f;    // degrees at hue = +-1
constexpr float kLumRange = 0.4f;     // fraction of L at luminance = +-1
constexpr float kChromaFull = 0.06f;  // OkLab chroma at which a pixel is fully coloured

// The band settings interpolated at hue h (degrees): out[0] hue, [1] sat, [2] lum.
void Mix(float h, const float (*s)[3], float out[3]) {
    h = std::fmod(h, 360.0f);
    if (h < 0.0f) h += 360.0f;
    int lo = kBands - 1;
    for (int i = 0; i < kBands; ++i)
        if (h >= kCentres[i]) lo = i;
    const int hi = (lo + 1) % kBands;
    float span = kCentres[hi] - kCentres[lo];
    float d = h - kCentres[lo];
    if (span <= 0.0f) span += 360.0f;
    if (d < 0.0f) d += 360.0f;
    const float t = 0.5f - 0.5f * std::cos(3.14159265f * std::clamp(d / span, 0.0f, 1.0f));
    for (int k = 0; k < 3; ++k) out[k] = s[lo][k] + (s[hi][k] - s[lo][k]) * t;
}

const char* const kMixerHead = R"(
Texture2D<float4>   Src : register(t0);
RWTexture2D<float4> Dst : register(u0);
cbuffer Params : register(b0) {
    uint Width;
    uint Height;
    uint Linear;
    uint Pad;        // so S starts on a 16-byte register, as a cbuffer array must
    uint4 S[6];      // per band: hue, saturation, luminance (float bits), packed
};
static const float kCentres[8] = {29.0, 58.0, 110.0, 142.0, 195.0, 264.0, 295.0, 328.0};
)";

const char* const kMixerMain = R"(
float3 ToLinear(float3 c) {
    return Linear != 0 ? c
        : select(c <= 0.04045, c / 12.92, pow((max(c, 0.0) + 0.055) / 1.055, 2.4));
}
float3 FromLinear(float3 c) {
    if (Linear != 0) return max(c, 0.0);
    c = saturate(c);
    return select(c <= 0.0031308, c * 12.92, 1.055 * pow(c, 1.0 / 2.4) - 0.055);
}
uint At(uint k) { return S[k >> 2][k & 3]; }
float3 Band(uint i) { return float3(asfloat(At(i * 3)), asfloat(At(i * 3 + 1)), asfloat(At(i * 3 + 2))); }

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    float4 c = Src[tid.xy];
    float3 lab = LinearSrgbToOkLab(ToLinear(c.rgb));
    float C = length(lab.yz);
    float w = saturate(C / 0.06);
    w = w * w * (3.0 - 2.0 * w);
    if (w <= 0.0) { Dst[tid.xy] = c; return; }
    float h = degrees(atan2(lab.z, lab.y));
    h = h < 0.0 ? h + 360.0 : h;
    uint lo = 7;
    [unroll] for (uint i = 0; i < 8; ++i) if (h >= kCentres[i]) lo = i;
    uint hi = (lo + 1) % 8;
    float span = kCentres[hi] - kCentres[lo];
    float d = h - kCentres[lo];
    if (span <= 0.0) span += 360.0;
    if (d < 0.0) d += 360.0;
    float t = 0.5 - 0.5 * cos(3.14159265 * saturate(d / span));
    float3 m = lerp(Band(lo), Band(hi), t);

    float nh = radians(h + m.x * 30.0 * w);
    float nC = C * max(0.0, 1.0 + m.y * w);
    lab.x *= 1.0 + m.z * 0.4 * w;
    lab.yz = nC * float2(cos(nh), sin(nh));
    Dst[tid.xy] = float4(FromLinear(OkLabToLinearSrgb(lab)), c.a);
}
)";

class ColorMixer : public AlgorithmBase {
public:
    const char* Name()     const override { return "color_mixer"; }
    const char* Category() const override { return "color"; }

    PortList Inputs()  const override { return {{"src", DataType::Image, FormatSpec::Any}}; }
    PortList Outputs() const override { return {{"out", DataType::Image, FormatSpec::SameAsInput}}; }

    bool IsNoOp() const override {
        for (int i = 0; i < kBands; ++i)
            if (float(m_hue[i]) != 0.0f || float(m_sat[i]) != 0.0f || float(m_lum[i]) != 0.0f) return false;
        return true;
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
        float s[kBands][3];
        Settings(s);
        auto lin = [&](float v) { return linear ? v / vs : SrgbToLinear(v / vs); };
        auto enc = [&](float v) {
            return linear ? std::max(v, 0.0f) * vs : LinearToSrgb(std::clamp(v, 0.0f, 1.0f)) * vs;
        };
        ParallelFor(size_t(h), [&](size_t y) {
            for (int x = 0; x < w; ++x) {
                const size_t i = y * size_t(w) + size_t(x);
                const float* p = &px[i * size_t(ch)];
                float* q = &o[i * size_t(ch)];
                for (int c = 0; c < ch; ++c) q[c] = p[c];
                const float rgb[3] = {lin(p[0]), lin(p[1]), lin(p[2])};
                float lab[3];
                LinearSrgbToOkLab(rgb, lab);
                const float C = std::hypot(lab[1], lab[2]);
                float wt = std::clamp(C / kChromaFull, 0.0f, 1.0f);
                wt = wt * wt * (3.0f - 2.0f * wt);
                if (wt <= 0.0f) continue;
                const float hue = std::atan2(lab[2], lab[1]) * (180.0f / 3.14159265f);
                float m[3];
                Mix(hue, s, m);
                const float nh = (hue + m[0] * kHueRange * wt) * (3.14159265f / 180.0f);
                const float nC = C * std::max(0.0f, 1.0f + m[1] * wt);
                lab[0] *= 1.0f + m[2] * kLumRange * wt;
                lab[1] = nC * std::cos(nh);
                lab[2] = nC * std::sin(nh);
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
        static const std::string src = std::string(kMixerHead) + kOkLabHlsl + kMixerMain;
        return src.c_str();
    }

    std::vector<uint32_t> GpuConstants(int) const override {
        float s[kBands][3];
        Settings(s);
        std::vector<uint32_t> k = {m_linear ? 1u : 0u, 0u};
        for (int i = 0; i < kBands; ++i)
            for (int j = 0; j < 3; ++j) k.push_back(FloatBits(s[i][j]));
        return k;
    }

private:
    void Settings(float s[kBands][3]) const {
        for (int i = 0; i < kBands; ++i) {
            s[i][0] = float(m_hue[i]);
            s[i][1] = float(m_sat[i]);
            s[i][2] = float(m_lum[i]);
        }
    }

    // One hue, saturation and luminance slider per band, named "<band>_hue"
    // and so on; built in a loop so the eight stay identical in all but name.
    struct Names {
        std::string hue[kBands], sat[kBands], lum[kBands];
        Names() {
            for (int i = 0; i < kBands; ++i) {
                hue[i] = std::string(kBandNames[i]) + "_hue";
                sat[i] = std::string(kBandNames[i]) + "_saturation";
                lum[i] = std::string(kBandNames[i]) + "_luminance";
            }
        }
    };
    static const Names& N() { static const Names n; return n; }

    Param<float> m_hue[kBands] = {
#define TGLAB_HUE(i) Param<float>{this, N().hue[i].c_str(), 0.0f, -1.0f, 1.0f, \
        {.help = "Turn this band's colours toward the next band round (+) or " \
                 "the previous one (-), by up to 30 degrees of OkLab hue.", .step = 0.01}}
        TGLAB_HUE(0), TGLAB_HUE(1), TGLAB_HUE(2), TGLAB_HUE(3),
        TGLAB_HUE(4), TGLAB_HUE(5), TGLAB_HUE(6), TGLAB_HUE(7)};
#undef TGLAB_HUE
    Param<float> m_sat[kBands] = {
#define TGLAB_SAT(i) Param<float>{this, N().sat[i].c_str(), 0.0f, -1.0f, 1.0f, \
        {.help = "This band's saturation: -1 takes it to grey, +1 doubles its " \
                 "chroma. Lightness is kept.", .step = 0.01}}
        TGLAB_SAT(0), TGLAB_SAT(1), TGLAB_SAT(2), TGLAB_SAT(3),
        TGLAB_SAT(4), TGLAB_SAT(5), TGLAB_SAT(6), TGLAB_SAT(7)};
#undef TGLAB_SAT
    Param<float> m_lum[kBands] = {
#define TGLAB_LUM(i) Param<float>{this, N().lum[i].c_str(), 0.0f, -1.0f, 1.0f, \
        {.help = "This band's lightness, by up to 40% of OkLab L either way: " \
                 "a darker sky, lighter skin. Greys are not moved.", .step = 0.01}}
        TGLAB_LUM(0), TGLAB_LUM(1), TGLAB_LUM(2), TGLAB_LUM(3),
        TGLAB_LUM(4), TGLAB_LUM(5), TGLAB_LUM(6), TGLAB_LUM(7)};
#undef TGLAB_LUM

    bool m_linear = false;   // set by PrepareGpu
};

REGISTER_ALGORITHM(ColorMixer);

}  // namespace
}  // namespace tglab
