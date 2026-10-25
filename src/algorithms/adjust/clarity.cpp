// clarity — local contrast that respects edges: clarity at the scale of
// objects, texture at the scale of surfaces.
//
// THE SPLIT. In log luminance, an edge-preserving base layer (a guided filter,
// He, Sun & Tang 2010 -- as tonemap_local uses) and the detail above it:
//
//     detail = log2 L - guided(log2 L)
//
// Raising the detail raises local contrast, and because the base follows any
// edge stronger than the filter's threshold, the detail does not contain the
// edge itself -- so the classic halo along a skyline, which a plain unsharp
// mask at this radius paints in, does not appear. In log, the same local
// contrast is the same number in shadow and highlight (see tonemap_local's
// header for why that matters).
//
//   clarity   detail over a LARGE radius (`radius`, % of the frame's short
//             side): the contrast between neighbouring objects and within
//             them -- the midtone "punch". Negative softens, which is the
//             diffuse look.
//   texture   detail over a radius a tenth of that, its large swings tamed so
//             edges are left to clarity and sharpening: the grain of wood,
//             stone, foliage, skin. Negative smooths it.
//
// Applied as one gain per pixel, 2^(delta), to all three channels: hue is
// kept and colour noise is not amplified with the detail. Deep shadows are
// left out, where log detail is mostly noise. Encoded images are taken to
// linear for the gain and back.
//
// The radii are fractions of the FULL frame, so a proxy and the full image
// match, and a region would not; hence RegionSafe is false.
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

constexpr float kFloor = 1e-6f;
// Guided-filter thresholds, in stops: variation below this is detail, above
// it an edge the base follows.
constexpr float kEpsClarity = 0.5f, kEpsTexture = 0.25f;
// Texture's radius against clarity's.
constexpr float kTextureScale = 0.1f;
// Log luminance where the effect fades in: below 2^-11 it is noise.
constexpr float kShadowLo = -11.0f, kShadowHi = -7.0f;

inline float Smoothstep(float a, float b, float x) {
    const float t = std::clamp((x - a) / (b - a), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// The per-pixel log-luminance change. Shared by the CPU path; the shader
// repeats it.
inline float Delta(float l, float baseC, float baseT, float clarity, float texture) {
    const float dc = std::clamp(l - baseC, -2.0f, 2.0f);
    const float dt = l - baseT;
    const float tame = 1.0f - 0.6f * Smoothstep(0.4f, 1.6f, std::fabs(dt));
    const float shadow = Smoothstep(kShadowLo, kShadowHi, baseC);
    return shadow * (clarity * 0.8f * dc + texture * std::clamp(dt, -1.0f, 1.0f) * tame);
}

void BoxMean(const std::vector<float>& in, std::vector<float>& out, int w, int h, int r) {
    std::vector<float> tmp(in.size());
    const float norm = 1.0f / float(2 * r + 1);
    ParallelFor(size_t(h), [&](size_t y) {
        const size_t row = y * size_t(w);
        float sum = 0.0f;
        for (int i = -r; i <= r; ++i) sum += in[row + size_t(std::clamp(i, 0, w - 1))];
        for (int x = 0; x < w; ++x) {
            tmp[row + size_t(x)] = sum * norm;
            sum += in[row + size_t(std::clamp(x + r + 1, 0, w - 1))] -
                   in[row + size_t(std::clamp(x - r, 0, w - 1))];
        }
    });
    ParallelFor(size_t(w), [&](size_t x) {
        float sum = 0.0f;
        for (int i = -r; i <= r; ++i) sum += tmp[size_t(std::clamp(i, 0, h - 1)) * size_t(w) + x];
        for (int y = 0; y < h; ++y) {
            out[size_t(y) * size_t(w) + x] = sum * norm;
            sum += tmp[size_t(std::clamp(y + r + 1, 0, h - 1)) * size_t(w) + x] -
                   tmp[size_t(std::clamp(y - r, 0, h - 1)) * size_t(w) + x];
        }
    });
}

// The guided filter of `l` by itself.
std::vector<float> Guided(const std::vector<float>& l, int w, int h, int r, float epsStops) {
    const size_t n = l.size();
    std::vector<float> mean(n), meanSq(n), sq(n), a(n), b(n);
    for (size_t i = 0; i < n; ++i) sq[i] = l[i] * l[i];
    BoxMean(l, mean, w, h, r);
    BoxMean(sq, meanSq, w, h, r);
    const float eps = epsStops * epsStops;
    for (size_t i = 0; i < n; ++i) {
        const float var = std::max(0.0f, meanSq[i] - mean[i] * mean[i]);
        a[i] = var / (var + eps);
        b[i] = (1.0f - a[i]) * mean[i];
    }
    BoxMean(a, mean, w, h, r);
    BoxMean(b, meanSq, w, h, r);
    for (size_t i = 0; i < n; ++i) sq[i] = mean[i] * l[i] + meanSq[i];
    return sq;
}

const char* const kCommon = R"(
Texture2D<float4>   T0 : register(t0);
Texture2D<float4>   T1 : register(t1);
Texture2D<float4>   T2 : register(t2);
Texture2D<float4>   T3 : register(t3);
RWTexture2D<float4> U0 : register(u0);

cbuffer Params : register(b0) {
    uint Width;
    uint Height;
    uint Radius;
    uint EpsBits;        // guided-filter eps, squared (a variance)
    uint ClarityBits;
    uint TextureBits;
    uint Linear;
};

int2 ClampXY(int x, int y) {
    return int2(clamp(x, 0, int(Width) - 1), clamp(y, 0, int(Height) - 1));
}
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

const char* const kLogHlsl = R"(
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    float3 c = ToLinear(T0[int2(tid.xy)].rgb);
    float l = log2(max(dot(c, float3(0.2126, 0.7152, 0.0722)), 1e-6));
    U0[tid.xy] = float4(l, l * l, 0, 1);
}
)";

// Running-sum box filters, a run of kBoxRun pixels per thread (as
// tonemap_local's).
const char* const kBoxHHlsl = R"(
static const int kBoxRun = 64;
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    int x0 = int(tid.x) * kBoxRun, y = int(tid.y);
    if (x0 >= int(Width) || y >= int(Height)) return;
    int r = int(Radius);
    float inv = 1.0 / float(2 * r + 1);
    float2 sum = 0.0;
    for (int i = -r; i <= r; ++i) sum += T0[ClampXY(x0 + i, y)].xy;
    int x1 = min(x0 + kBoxRun, int(Width));
    for (int x = x0; x < x1; ++x) {
        if (x > x0) sum += T0[ClampXY(x + r, y)].xy - T0[ClampXY(x - r - 1, y)].xy;
        U0[int2(x, y)] = float4(sum * inv, 0, 1);
    }
}
)";
const char* const kBoxVHlsl = R"(
static const int kBoxRun = 64;
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    int x = int(tid.x), y0 = int(tid.y) * kBoxRun;
    if (x >= int(Width) || y0 >= int(Height)) return;
    int r = int(Radius);
    float inv = 1.0 / float(2 * r + 1);
    float2 sum = 0.0;
    for (int i = -r; i <= r; ++i) sum += T0[ClampXY(x, y0 + i)].xy;
    int y1 = min(y0 + kBoxRun, int(Height));
    for (int y = y0; y < y1; ++y) {
        if (y > y0) sum += T0[ClampXY(x, y + r)].xy - T0[ClampXY(x, y - r - 1)].xy;
        U0[int2(x, y)] = float4(sum * inv, 0, 1);
    }
}
)";
constexpr uint32_t kBoxRun = 64;   // must match the HLSL above

const char* const kAbHlsl = R"(
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    float2 ms = T0[int2(tid.xy)].xy;
    float var = max(0.0, ms.y - ms.x * ms.x);
    float a = var / (var + asfloat(EpsBits));
    U0[tid.xy] = float4(a, (1.0 - a) * ms.x, 0, 1);
}
)";

const char* const kCombineHlsl = R"(
float Smooth(float a, float b, float x) { float t = saturate((x - a) / (b - a)); return t * t * (3 - 2 * t); }
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    float4 src = T0[int2(tid.xy)];
    float  l   = T1[int2(tid.xy)].x;
    float2 abC = T2[int2(tid.xy)].xy;
    float2 abT = T3[int2(tid.xy)].xy;
    float baseC = abC.x * l + abC.y, baseT = abT.x * l + abT.y;
    float dc = clamp(l - baseC, -2.0, 2.0);
    float dt = l - baseT;
    float tame = 1.0 - 0.6 * Smooth(0.4, 1.6, abs(dt));
    float shadow = Smooth(-11.0, -7.0, baseC);
    float delta = shadow * (asfloat(ClarityBits) * 0.8 * dc +
                            asfloat(TextureBits) * clamp(dt, -1.0, 1.0) * tame);
    U0[tid.xy] = float4(FromLinear(ToLinear(src.rgb) * exp2(delta)), src.a);
}
)";

class Clarity : public AlgorithmBase {
public:
    const char* Name()     const override { return "clarity"; }
    const char* Category() const override { return "adjust"; }

    PortList Inputs()  const override { return {{"src", DataType::Image, FormatSpec::Any}}; }
    PortList Outputs() const override { return {{"out", DataType::Image, FormatSpec::SameAsInput}}; }

    bool IsNoOp() const override { return float(m_clarity) == 0.0f && float(m_texture) == 0.0f; }
    bool RegionSafe() const override { return false; }

    void RunCPU(RunCtx& ctx) override {
        const ImageView src = ctx.In(0);
        ImageView dst = ctx.Out(0);
        if (!src.Valid() || !dst.Valid()) return;
        PixelBuffer in, out;
        in.Unpack(src);
        if (!in.Valid()) return;
        out.AllocLike(in);
        const ImageDesc& d = ctx.InDesc(0);
        const int w = in.Width(), h = in.Height(), ch = in.Channels();
        const float vs = in.ValueScale();
        const bool linear = d.linear;
        const int colours = std::min(ch, 3);
        const size_t n = size_t(w) * size_t(h);
        const float* px = in.Data().data();

        auto lin = [&](float v) { return linear ? v / vs : SrgbToLinear(v / vs); };
        std::vector<float> l(n);
        ParallelFor(size_t(h), [&](size_t y) {
            for (int x = 0; x < w; ++x) {
                const size_t i = y * size_t(w) + size_t(x);
                const float* p = &px[i * size_t(ch)];
                const float L = colours == 3 ? Luma(lin(p[0]), lin(p[1]), lin(p[2])) : lin(p[0]);
                l[i] = std::log2(std::max(L, kFloor));
            }
        });
        int rc, rt;
        Radii(d, &rc, &rt);
        const std::vector<float> baseC = Guided(l, w, h, rc, kEpsClarity);
        const std::vector<float> baseT = Guided(l, w, h, rt, kEpsTexture);
        if (ctx.Cancelled()) return;

        const float clarity = float(m_clarity), texture = float(m_texture);
        float* o = out.Data().data();
        ParallelFor(size_t(h), [&](size_t y) {
            for (int x = 0; x < w; ++x) {
                const size_t i = y * size_t(w) + size_t(x);
                const float gain = std::exp2(Delta(l[i], baseC[i], baseT[i], clarity, texture));
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
    // Log and log^2, the two guided filters (box, a/b, box) at their radii,
    // then the gain: twelve passes over five two-channel planes.
    bool HasGPU() const override { return true; }
    int        GpuScratchCount()  const override { return 5; }
    FormatSpec GpuScratchPlanes() const override { return FormatSpec::RG32F; }

    void PrepareGpu(const std::vector<ImageDesc>& inputs) override {
        if (!inputs.empty()) m_desc = inputs[0];
    }

    std::vector<GpuPass> GpuPasses() const override {
        static const std::string logp = std::string(kCommon) + kLogHlsl;
        static const std::string boxh = std::string(kCommon) + kBoxHHlsl;
        static const std::string boxv = std::string(kCommon) + kBoxVHlsl;
        static const std::string ab   = std::string(kCommon) + kAbHlsl;
        static const std::string comb = std::string(kCommon) + kCombineHlsl;
        std::vector<GpuPass> p;
        p.push_back({logp.c_str(), "log", {-1}, {0}});
        for (int k = 0; k < 2; ++k) {           // clarity into plane 3, texture into 4
            const int dstPlane = 3 + k;
            p.push_back({boxh.c_str(), "boxH",   {0}, {1}, kBoxRun, 1});
            p.push_back({boxv.c_str(), "boxV",   {1}, {2}, 1, kBoxRun});
            p.push_back({ab.c_str(),   "ab",     {2}, {1}});
            p.push_back({boxh.c_str(), "boxABH", {1}, {2}, kBoxRun, 1});
            p.push_back({boxv.c_str(), "boxABV", {2}, {dstPlane}, 1, kBoxRun});
        }
        p.push_back({comb.c_str(), "combine", {-1, 0, 3, 4}, {-1}});
        return p;
    }

    std::vector<uint32_t> GpuPassConstants(int pass) const override {
        int rc, rt;
        Radii(m_desc, &rc, &rt);
        // Passes 1-5 are clarity's filter, 6-10 texture's.
        const bool tex = pass >= 6 && pass <= 10;
        const float eps = tex ? kEpsTexture : kEpsClarity;
        return {uint32_t(tex ? rt : rc), FloatBits(eps * eps),
                FloatBits(float(m_clarity)), FloatBits(float(m_texture)),
                m_desc.linear ? 1u : 0u};
    }

private:
    // The two radii, in this raster's pixels, from the FULL frame's short side.
    void Radii(const ImageDesc& d, int* rc, int* rt) const {
        const float side = float(std::max(1, std::min(d.FullWidth(), d.FullHeight())));
        *rc = std::max(2, int(std::lround(float(m_radius) * 0.01f * side)));
        *rt = std::max(1, int(std::lround(float(m_radius) * 0.01f * side * kTextureScale)));
    }

    Param<float> m_clarity{this, "clarity", 0.0f, -1.0f, 1.0f,
        {.help = "Local contrast at the scale of objects, without halos along "
                 "strong edges. Negative softens.",
         .step = 0.01}};
    Param<float> m_texture{this, "texture", 0.0f, -1.0f, 1.0f,
        {.help = "Local contrast at the scale of surfaces -- wood, stone, "
                 "foliage, skin -- leaving edges to clarity. Negative smooths.",
         .step = 0.01}};
    Param<float> m_radius{this, "radius", 2.0f, 0.5f, 8.0f,
        {.help = "Clarity's reach, as a percentage of the frame's short side; "
                 "texture works at a tenth of it.",
         .step = 0.1}};

    ImageDesc m_desc{};   // set by PrepareGpu
};

REGISTER_ALGORITHM(Clarity);

}  // namespace
}  // namespace tglab
