// curves — a tone curve: parametric regions, and an optional point curve.
//
// THE PARAMETRIC CURVE is the identity with four smooth bumps added, one per
// tonal region -- shadows [0, split1], darks [split1, split2], lights [split2,
// split3], highlights [split3, 1] -- each a raised cosine over its region,
// lifting (or lowering) its middle by up to 30% of the region's width. The
// endpoints stay put (0 -> 0, 1 -> 1), and the result is forced monotone, so no
// combination of sliders can reverse tones -- the solarised fold a free curve
// produces when a bump overtakes its neighbour. (lightcraft's parametric curve
// is the same idea: crates/pipeline/src/curve.rs.)
//
// THE POINT CURVE is typed, "x,y x,y ...", in the same 0..1 domain, and runs
// after the parametric one. It is a monotone cubic through the points
// (Fritsch-Carlson), so it passes through every point and never overshoots
// between them: a curve that dips below a point on its way up is the classic
// failure of an ordinary spline here.
//
// THE DOMAIN is display values, where curves are drawn and judged: a linear
// image is encoded with the sRGB curve for the lookup and decoded after, so the
// same curve does the same thing to a raw and a JPEG. Above white (a scene-linear
// highlight) the curve does not reach, and the pixel passes unchanged.
//
// On LUMINANCE by default: the curve sets each pixel's luminance and its colour
// is scaled with it, so contrast does not shift hue -- per channel, a steep
// curve pulls a saturated colour's channels apart and orange turns red.
// `per_channel` gives the per-channel behaviour for comparison or for effect.
#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include "../../algo_util/color.h"
#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"
#include "../../core/parallel.h"

namespace tglab {
namespace {

constexpr int kLut = 256;            // entries over 0..1, linearly interpolated
// A region's bump at full strength, as a fraction of its width: the highlights
// region's middle moves 0.075 at +1, and cannot be pushed into white.
constexpr float kBumpMax = 0.3f;

// The four regions' raised cosines, summed, onto the identity.
void BuildParametric(float sh, float dk, float lt, float hl, float s1, float s2, float s3,
                     std::vector<float>* lut) {
    s1 = std::clamp(s1, 0.02f, 0.96f);
    s2 = std::clamp(s2, s1 + 0.01f, 0.97f);
    s3 = std::clamp(s3, s2 + 0.01f, 0.98f);
    const float edges[5] = {0.0f, s1, s2, s3, 1.0f};
    const float amt[4] = {sh, dk, lt, hl};
    lut->resize(kLut);
    for (int i = 0; i < kLut; ++i) {
        const float x = float(i) / float(kLut - 1);
        float y = x;
        for (int r = 0; r < 4; ++r) {
            if (amt[r] == 0.0f || x <= edges[r] || x >= edges[r + 1]) continue;
            const float t = (x - edges[r]) / (edges[r + 1] - edges[r]);
            y += amt[r] * kBumpMax * (edges[r + 1] - edges[r]) * 0.5f *
                 (1.0f - std::cos(6.28318531f * t));
        }
        (*lut)[size_t(i)] = y;
    }
    // Monotone and inside 0..1, endpoints fixed.
    (*lut)[0] = 0.0f;
    for (int i = 1; i < kLut; ++i)
        (*lut)[size_t(i)] = std::clamp((*lut)[size_t(i)], (*lut)[size_t(i - 1)], 1.0f);
    (*lut)[kLut - 1] = 1.0f;
}

// "x,y x,y ..." -> sorted points, or empty when it does not parse.
std::vector<std::pair<float, float>> ParsePoints(const std::string& text) {
    std::vector<std::pair<float, float>> pts;
    std::string t = text;
    for (char& c : t) if (c == ',' || c == ';') c = ' ';
    std::istringstream is(t);
    float x, y;
    while (is >> x >> y) pts.push_back({std::clamp(x, 0.0f, 1.0f), std::clamp(y, 0.0f, 1.0f)});
    std::sort(pts.begin(), pts.end());
    pts.erase(std::unique(pts.begin(), pts.end(),
                          [](const auto& a, const auto& b) { return a.first == b.first; }),
              pts.end());
    if (pts.size() < 2) pts.clear();
    return pts;
}

// Fritsch-Carlson monotone cubic through the points, sampled.
float MonotoneAt(const std::vector<std::pair<float, float>>& p, const std::vector<float>& m, float x) {
    if (x <= p.front().first) return p.front().second;
    if (x >= p.back().first) return p.back().second;
    size_t k = 0;
    while (k + 2 < p.size() && x > p[k + 1].first) ++k;
    const float h = p[k + 1].first - p[k].first, t = (x - p[k].first) / h;
    const float t2 = t * t, t3 = t2 * t;
    return (2 * t3 - 3 * t2 + 1) * p[k].second + (t3 - 2 * t2 + t) * h * m[k] +
           (-2 * t3 + 3 * t2) * p[k + 1].second + (t3 - t2) * h * m[k + 1];
}

void ApplyPoints(const std::vector<std::pair<float, float>>& p, std::vector<float>* lut) {
    const size_t n = p.size();
    std::vector<float> d(n - 1), m(n);
    for (size_t k = 0; k + 1 < n; ++k) d[k] = (p[k + 1].second - p[k].second) / (p[k + 1].first - p[k].first);
    m[0] = d[0]; m[n - 1] = d[n - 2];
    for (size_t k = 1; k + 1 < n; ++k) m[k] = (d[k - 1] * d[k] <= 0.0f) ? 0.0f : 0.5f * (d[k - 1] + d[k]);
    for (size_t k = 0; k + 1 < n; ++k) {
        if (d[k] == 0.0f) { m[k] = m[k + 1] = 0.0f; continue; }
        const float a = m[k] / d[k], b = m[k + 1] / d[k], s = a * a + b * b;
        if (s > 9.0f) { const float tau = 3.0f / std::sqrt(s); m[k] = tau * a * d[k]; m[k + 1] = tau * b * d[k]; }
    }
    for (float& v : *lut) v = MonotoneAt(p, m, v);
}

// See the shader: on an encoded image a channel past white moves toward the
// luminance just enough to fit, rather than clipping and shifting hue.
inline void FitBelowWhite(float* o, float Y, bool linear) {
    if (linear) return;
    const float mx = std::max(o[0], std::max(o[1], o[2]));
    if (mx <= 1.0f || Y >= 1.0f) return;
    const float t = (mx - 1.0f) / (mx - Y);
    for (int c = 0; c < 3; ++c) o[c] += (Y - o[c]) * t;
}

inline float Lookup(const float* lut, float e) {
    const float f = std::clamp(e, 0.0f, 1.0f) * float(kLut - 1);
    const int i = std::min(int(f), kLut - 2);
    return lut[i] + (lut[i + 1] - lut[i]) * (f - float(i));
}

const char* const kCurvesHlsl = R"(
Texture2D<float4>   Src : register(t0);
// The table, 256 x 1: too big for the 32 root constants, so a stage-owned
// texture (GpuExtraInputs), as apply_lut's LUT is.
Texture2D<float>    LutTex : register(t1);
RWTexture2D<float4> Dst : register(u0);
cbuffer Params : register(b0) {
    uint Width;
    uint Height;
    uint Linear;
    uint PerChannel;
};
float LutAt(int i) { return LutTex.Load(int3(i, 0, 0)); }
float Curve(float e) {
    float f = saturate(e) * 255.0;
    int i = min(int(f), 254);
    float a = LutAt(i), b = LutAt(i + 1);
    return a + (b - a) * (f - float(i));
}
float Enc(float v) { return v <= 0.0031308 ? v * 12.92 : 1.055 * pow(max(v, 0.0), 1.0 / 2.4) - 0.055; }
float Dec(float v) { return v <= 0.04045 ? v / 12.92 : pow((max(v, 0.0) + 0.055) / 1.055, 2.4); }
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    float4 c = Src[tid.xy];
    float3 lin = Linear != 0 ? c.rgb : SrgbToLinear(c.rgb);
    float3 o;
    if (PerChannel != 0) {
        [unroll] for (int k = 0; k < 3; ++k)
            o[k] = lin[k] >= 1.0 ? lin[k] : Dec(Curve(Enc(lin[k])));
    } else {
        float Y = Luma(lin);
        float g = (Y <= 0.0 || Y >= 1.0) ? 1.0 : Dec(Curve(Enc(Y))) / Y;
        o = lin * g;
        // An encoded image ends at white: a channel the gain pushed past it
        // moves toward the luminance just enough to fit, rather than clipping
        // -- which would shift the hue luminance mode exists to keep.
        float mx = max(o.r, max(o.g, o.b)), Yn = Y * g;
        if (Linear == 0 && mx > 1.0 && Yn < 1.0) o += (Yn - o) * ((mx - 1.0) / (mx - Yn));
    }
    Dst[tid.xy] = float4(Linear != 0 ? o : LinearToSrgb(o), c.a);
}
)";

class Curves : public AlgorithmBase {
public:
    const char* Name()     const override { return "curves"; }
    const char* Category() const override { return "adjust"; }

    PortList Inputs()  const override { return {{"src", DataType::Image, FormatSpec::Any}}; }
    PortList Outputs() const override { return {{"out", DataType::Image, FormatSpec::SameAsInput}}; }

    bool IsNoOp() const override {
        return float(m_shadows) == 0.0f && float(m_darks) == 0.0f && float(m_lights) == 0.0f &&
               float(m_highlights) == 0.0f && ParsePoints(m_points.get()).empty();
    }

    void RunCPU(RunCtx& ctx) override {
        const ImageView src = ctx.In(0);
        ImageView dst = ctx.Out(0);
        if (!src.Valid() || !dst.Valid()) return;
        PixelBuffer in;
        in.Unpack(src);
        if (!in.Valid()) return;
        const int w = in.Width(), h = in.Height(), ch = in.Channels();
        const float vs = in.ValueScale();
        const bool linear = ctx.InDesc(0).linear, perCh = bool(m_perChannel) || ch < 3;
        std::vector<float> lut;
        Build(&lut);
        std::vector<float>& px = in.Data();
        auto enc = [](float v) { return v <= 0.0031308f ? v * 12.92f : 1.055f * std::pow(std::max(v, 0.0f), 1.0f / 2.4f) - 0.055f; };
        auto dec = [](float v) { return v <= 0.04045f ? v / 12.92f : std::pow((std::max(v, 0.0f) + 0.055f) / 1.055f, 2.4f); };
        ParallelFor(size_t(h), [&](size_t y) {
            for (int x = 0; x < w; ++x) {
                float* p = &px[(y * size_t(w) + size_t(x)) * size_t(ch)];
                const int nc = std::min(ch, 3);
                float lin[3];
                for (int c = 0; c < nc; ++c) lin[c] = linear ? p[c] / vs : SrgbToLinear(p[c] / vs);
                float o[3];
                if (perCh) {
                    for (int c = 0; c < nc; ++c)
                        o[c] = lin[c] >= 1.0f ? lin[c] : dec(Lookup(lut.data(), enc(lin[c])));
                } else {
                    const float Y = Luma(lin);
                    const float g = (Y <= 0.0f || Y >= 1.0f) ? 1.0f : dec(Lookup(lut.data(), enc(Y))) / Y;
                    for (int c = 0; c < 3; ++c) o[c] = lin[c] * g;
                    FitBelowWhite(o, Y * g, linear);
                }
                for (int c = 0; c < nc; ++c) p[c] = (linear ? o[c] : LinearToSrgb(o[c])) * vs;
            }
        });
        in.PackInto(dst);
    }

    bool HasGPU() const override { return true; }
    const char* GpuSource() const override { return kCurvesHlsl; }
    void PrepareGpu(const std::vector<ImageDesc>& inputs) override {
        if (!inputs.empty()) { m_linear = inputs[0].linear; m_grey = inputs[0].format == Format::R32F; }
        // The table as a texture, rebuilt only when the curve changed.
        std::vector<float> lut;
        Build(&lut);
        if (lut != m_lutVals || !m_lutImg.Valid()) {
            m_lutVals = lut;
            m_lutImg.Alloc({kLut, 1, Format::R32F});
            ImageView v = m_lutImg.MapCpuWrite();
            for (int i = 0; i < kLut; ++i) *v.At<float>(i, 0) = lut[size_t(i)];
        }
    }
    std::vector<const Image*> GpuExtraInputs() const override { return {&m_lutImg}; }
    std::vector<uint32_t> GpuConstants(int) const override {
        return {m_linear ? 1u : 0u, (bool(m_perChannel) || m_grey) ? 1u : 0u};
    }

private:
    void Build(std::vector<float>* lut) const {
        BuildParametric(float(m_shadows), float(m_darks), float(m_lights), float(m_highlights),
                        float(m_split1), float(m_split2), float(m_split3), lut);
        const auto pts = ParsePoints(m_points.get());
        if (!pts.empty()) ApplyPoints(pts, lut);
    }

    Param<float> m_highlights{this, "highlights", 0.0f, -1.0f, 1.0f,
        {.help = "The brightest region of the curve, above split3.", .step = 0.01}};
    Param<float> m_lights{this, "lights", 0.0f, -1.0f, 1.0f,
        {.help = "The upper middle, between split2 and split3.", .step = 0.01}};
    Param<float> m_darks{this, "darks", 0.0f, -1.0f, 1.0f,
        {.help = "The lower middle, between split1 and split2.", .step = 0.01}};
    Param<float> m_shadows{this, "shadows", 0.0f, -1.0f, 1.0f,
        {.help = "The darkest region of the curve, below split1.", .step = 0.01}};
    Param<float> m_split1{this, "split1", 0.25f, 0.05f, 0.45f,
        {.help = "Where shadows end and darks begin (display value).", .step = 0.01}};
    Param<float> m_split2{this, "split2", 0.5f, 0.3f, 0.7f,
        {.help = "Where darks end and lights begin.", .step = 0.01}};
    Param<float> m_split3{this, "split3", 0.75f, 0.55f, 0.95f,
        {.help = "Where lights end and highlights begin.", .step = 0.01}};
    Param<std::string> m_points{this, "points", {},
        "A point curve, after the parametric one: \"x,y x,y ...\" in display "
        "values 0..1, e.g. \"0,0 0.25,0.2 0.75,0.85 1,1\". A monotone cubic "
        "through them, so it never overshoots. Empty for none."};
    Param<bool> m_perChannel{this, "per_channel", false,
        "Apply the curve to each channel instead of to luminance. Luminance keeps "
        "hue; per channel, a steep curve pulls a saturated colour's channels "
        "apart and shifts it."};

    bool m_linear = false, m_grey = false;   // set by PrepareGpu
    Image m_lutImg;                          // the table, for the GPU
    std::vector<float> m_lutVals;
};

REGISTER_ALGORITHM(Curves);

}  // namespace
}  // namespace tglab
