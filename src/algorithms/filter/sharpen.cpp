// sharpen — restore the crispness the lens, the anti-aliasing filter and the
// demosaic took away.
//
// AN UNSHARP MASK IN LOG LUMINANCE. The detail is the difference between the
// log of a pixel's luminance and a small Gaussian blur of it:
//
//     detail = log2 L - blur(log2 L)          (in stops)
//     out    = rgb * 2^(amount * clamp(detail, -limit, limit) * mask)
//
// Three choices in that, each against a familiar failure:
//
//   LUMINANCE, applied as one gain to all three channels. Sharpening R, G and
//   B separately sharpens their DIFFERENCES too, and those are colour: every
//   edge between two hues grows a coloured fringe, and chroma noise is
//   amplified along with the detail. One gain per pixel keeps the hue.
//
//   LOG, not linear. In linear light the same edge is a large step in the
//   highlights and a tiny one in the shadows, so a linear mask oversharpens
//   bright edges and does nothing in the dark. In stops an edge is the same
//   size wherever it sits, and the result does not depend on exposure or on
//   whether the image is scene-linear or display-referred. (lightcraft
//   sharpens the same way: crates/pipeline/src/finish.rs.)
//
//   LIMITED, at `limit` stops either way. The halo along a strong edge is the
//   mask's own response to that edge, amplified; a big edge needs no help, so
//   capping the detail removes the halo where it would be worst and leaves
//   fine texture, whose detail is small, untouched.
//
// `masking` keeps the gain to where there is structure: at 0 everything is
// sharpened; rising, only pixels near an edge are. "Near an edge" is the
// gradient of the BLURRED log luminance -- noise averages away in the blur and
// leaves it small, while a real edge, however soft, keeps it large across its
// whole width. Keyed on the detail itself instead, a soft edge and grain both
// have small detail at a pixel's radius, and masking hid the one with the other.
//
// Encoded (non-linear) images are taken to linear for the gain and back, so a
// sharpened edge is the same physical contrast change on either kind.
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

// Below this (relative to white) luminance is held, so black does not read as
// an infinitely deep edge.
constexpr float kFloor = 1e-4f;

inline float Smoothstep(float a, float b, float x) {
    const float t = std::clamp((x - a) / (b - a), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// The weight for a blurred-log gradient `grad` (stops per pixel): none below
// masking * 0.05, all of it at twice that.
inline float MaskWeight(float grad, float masking) {
    if (masking <= 0.0f) return 1.0f;
    const float t = masking * 0.05f;
    return Smoothstep(t, 2.0f * t, grad);
}

int RadiusFor(float sigma) { return std::clamp(int(std::ceil(sigma * 3.0f)), 1, 64); }

class Sharpen : public AlgorithmBase {
public:
    const char* Name()     const override { return "sharpen"; }
    const char* Category() const override { return "filter"; }

    PortList Inputs()  const override { return {{"src", DataType::Image, FormatSpec::Any}}; }
    PortList Outputs() const override { return {{"out", DataType::Image, FormatSpec::SameAsInput}}; }

    bool IsNoOp() const override { return float(m_amount) == 0.0f; }
    int ReachPixels() const override { return RadiusFor(float(m_radius)); }

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
        const int colours = std::min(ch, 3);
        const float* px = in.Data().data();
        const size_t n = size_t(w) * size_t(h);

        auto lin = [&](float v) { return linear ? v / vs : SrgbToLinear(v / vs); };
        std::vector<float> logL(n), tmp(n), blur(n);
        ParallelFor(size_t(h), [&](size_t y) {
            for (int x = 0; x < w; ++x) {
                const size_t i = y * size_t(w) + size_t(x);
                const float* p = &px[i * size_t(ch)];
                const float L = colours == 3 ? Luma(lin(p[0]), lin(p[1]), lin(p[2])) : lin(p[0]);
                logL[i] = std::log2(std::max(L, kFloor));
            }
        });

        const float sigma = std::max(0.05f, ctx.ScaledPx(float(m_radius)));
        const int r = RadiusFor(sigma);
        std::vector<float> k(size_t(2 * r + 1));
        float ks = 0;
        for (int i = -r; i <= r; ++i) ks += k[size_t(i + r)] = std::exp(-float(i * i) / (2 * sigma * sigma));
        for (float& v : k) v /= ks;
        ParallelFor(size_t(h), [&](size_t y) {
            for (int x = 0; x < w; ++x) {
                float s = 0;
                for (int i = -r; i <= r; ++i)
                    s += k[size_t(i + r)] * logL[y * size_t(w) + size_t(std::clamp(x + i, 0, w - 1))];
                tmp[y * size_t(w) + size_t(x)] = s;
            }
        });
        ParallelFor(size_t(h), [&](size_t y) {
            for (int x = 0; x < w; ++x) {
                float s = 0;
                for (int i = -r; i <= r; ++i)
                    s += k[size_t(i + r)] * tmp[size_t(std::clamp(int(y) + i, 0, h - 1)) * size_t(w) + size_t(x)];
                blur[y * size_t(w) + size_t(x)] = s;
            }
        });

        const float amount = float(m_amount), limit = float(m_limit), masking = float(m_masking);
        float* o = out.Data().data();
        ParallelFor(size_t(h), [&](size_t y) {
            for (int x = 0; x < w; ++x) {
                const size_t i = y * size_t(w) + size_t(x);
                const float d = logL[i] - blur[i];
                float m = 1.0f;
                if (masking > 0.0f) {
                    auto b = [&](int xx, int yy) {
                        return blur[size_t(std::clamp(yy, 0, h - 1)) * size_t(w) + size_t(std::clamp(xx, 0, w - 1))];
                    };
                    const int yi = int(y);
                    m = MaskWeight(0.5f * std::hypot(b(x + 1, yi) - b(x - 1, yi), b(x, yi + 1) - b(x, yi - 1)),
                                   masking);
                }
                const float gain = std::exp2(amount * std::clamp(d, -limit, limit) * m);
                const float* p = &px[i * size_t(ch)];
                float* q = &o[i * size_t(ch)];
                for (int c = 0; c < ch; ++c) {
                    if (c >= colours) { q[c] = p[c]; continue; }
                    q[c] = linear ? p[c] * gain : LinearToSrgb(SrgbToLinear(p[c] / vs) * gain) * vs;
                }
            }
        });
        out.PackInto(dst);
    }

    // --- GPU ----------------------------------------------------------------
    // Log luminance into a plane, the blur in two passes, then the gain.
    bool HasGPU() const override { return true; }
    int        GpuScratchCount()  const override { return 3; }
    FormatSpec GpuScratchPlanes() const override { return FormatSpec::R32F; }

    void PrepareGpu(const std::vector<ImageDesc>& inputs) override {
        if (!inputs.empty()) {
            m_linear = inputs[0].linear;
            m_grey = inputs[0].format == Format::R32F;
        }
    }

    std::vector<GpuPass> GpuPasses() const override {
        static const std::string logl = std::string(kCommon) + kLogHlsl;
        static const std::string blr  = std::string(kCommon) + kBlurHlsl;
        static const std::string cmb  = std::string(kCommon) + kCombineHlsl;
        std::vector<GpuPass> p;
        p.push_back({logl.c_str(), "logL",    {-1},    {0}});
        p.push_back({blr.c_str(),  "blurH",   {0},     {1}});
        p.push_back({blr.c_str(),  "blurV",   {1},     {2}});
        p.push_back({cmb.c_str(),  "combine", {-1, 0, 2}, {-1}});
        return p;
    }

    std::vector<uint32_t> GpuPassConstants(int pass) const override {
        return {FloatBits(std::max(0.05f, GpuScaledPx(float(m_radius)))),
                FloatBits(float(m_amount)), FloatBits(float(m_limit)),
                FloatBits(float(m_masking)), pass == 2 ? 1u : 0u,
                m_linear ? 1u : 0u, m_grey ? 1u : 0u};
    }

private:
    static constexpr const char* kCommon = R"(
cbuffer Constants : register(b0) {
    uint Width;
    uint Height;
    uint SigmaBits, AmountBits, LimitBits, MaskingBits;
    uint Vertical, Linear, Grey;
};
float3 ToLinear(float3 c) {
    return Linear != 0 ? c
        : select(c <= 0.04045, c / 12.92, pow((max(c, 0.0) + 0.055) / 1.055, 2.4));
}
float3 FromLinear(float3 c) {
    if (Linear != 0) return c;
    c = saturate(c);
    return select(c <= 0.0031308, c * 12.92, 1.055 * pow(c, 1.0 / 2.4) - 0.055);
}
)";

    static constexpr const char* kLogHlsl = R"(
Texture2D<float4>   Src : register(t0);
RWTexture2D<float>  Dst : register(u0);
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    float3 c = ToLinear(Src[tid.xy].rgb);
    float L = Grey != 0 ? c.r : dot(c, float3(0.2126, 0.7152, 0.0722));
    Dst[tid.xy] = log2(max(L, 1e-4));
}
)";

    static constexpr const char* kBlurHlsl = R"(
Texture2D<float>   Src : register(t0);
RWTexture2D<float> Dst : register(u0);
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    float sigma = asfloat(SigmaBits);
    int r = clamp(int(ceil(sigma * 3.0)), 1, 64);
    float s = 0, ws = 0;
    for (int i = -r; i <= r; ++i) {
        int2 p = Vertical != 0 ? int2(tid.x, clamp(int(tid.y) + i, 0, int(Height) - 1))
                               : int2(clamp(int(tid.x) + i, 0, int(Width) - 1), tid.y);
        float w = exp(-float(i * i) / (2.0 * sigma * sigma));
        s += Src[p] * w; ws += w;
    }
    Dst[tid.xy] = s / ws;
}
)";

    static constexpr const char* kCombineHlsl = R"(
Texture2D<float4>   Src  : register(t0);
Texture2D<float>    LogL : register(t1);
Texture2D<float>    Blur : register(t2);
RWTexture2D<float4> Dst  : register(u0);
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    float d = LogL[tid.xy] - Blur[tid.xy];
    float limit = asfloat(LimitBits), masking = asfloat(MaskingBits);
    float m = 1.0;
    if (masking > 0.0) {
        int2 p = int2(tid.xy), hi = int2(Width, Height) - 1;
        float gx = Blur[clamp(p + int2(1, 0), 0, hi)] - Blur[clamp(p - int2(1, 0), 0, hi)];
        float gy = Blur[clamp(p + int2(0, 1), 0, hi)] - Blur[clamp(p - int2(0, 1), 0, hi)];
        float t = masking * 0.05;
        m = smoothstep(t, 2.0 * t, 0.5 * length(float2(gx, gy)));
    }
    float gain = exp2(asfloat(AmountBits) * clamp(d, -limit, limit) * m);
    float4 c = Src[tid.xy];
    Dst[tid.xy] = float4(FromLinear(ToLinear(c.rgb) * gain), c.a);
}
)";

    Param<float> m_amount{this, "amount", 1.0f, 0.0f, 5.0f,
        {.help = "How strongly fine detail is raised: the local contrast at the "
                 "radius's scale is multiplied by 1 + amount, in stops.",
         .step = 0.01}};
    Param<float> m_radius{this, "radius", 1.0f, 0.3f, 4.0f,
        {.help = "The scale of the detail sharpened, in pixels (the blur's "
                 "sigma). Around 1 for the softness a demosaic leaves; larger "
                 "for a soft lens, at the risk of halos.",
         .step = 0.05}};
    Param<float> m_limit{this, "limit", 0.5f, 0.05f, 3.0f,
        {.help = "The most any pixel's detail counts, in stops. Strong edges "
                 "need no sharpening and halo when they get it; this caps them "
                 "and leaves fine texture alone.",
         .step = 0.01}};
    Param<float> m_masking{this, "masking", 0.5f, 0.0f, 1.0f,
        {.help = "Sharpen only near edges, leaving noise, smooth skin and "
                 "clear sky alone. 0 sharpens everything.",
         .step = 0.01}};

    bool m_linear = false, m_grey = false;   // set by PrepareGpu
};

REGISTER_ALGORITHM(Sharpen);

}  // namespace
}  // namespace tglab
