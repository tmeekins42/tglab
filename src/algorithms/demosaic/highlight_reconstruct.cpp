// highlight_reconstruct: rebuilds partly clipped highlights from nearby colour.
//
// WHAT IT FIXES. A demosaiced raw is clamped at the brightest neutral its
// sensor can describe (clip_repair.h): every white-balanced channel at or above
// min(camMul) is cut there. That stops a blown light rendering magenta, and it
// throws away what the unclipped channels knew. Where only SOME channels
// clipped -- the warm halo round a streetlight, a sunlit cloud's edge, skin in
// hard sun -- the pixel is still mostly measured: one or two channels are real,
// and only the clipped one is unknown. Clamped, the pixel goes paler and
// flatter than the scene was, and the halo turns into a pale disc; and every
// darkening control afterwards (highlights, negative exposure) shows it.
//
// HOW. In white-balanced camera RGB, recovered by undoing the camera matrix:
//
//   1. Clipped channels are the ones at the clamp.
//   2. The COLOUR of the surroundings is learned from reliable pixels -- bright
//      enough to describe a highlight (dark foliage says nothing about the
//      colour of a lamp), not on a strong edge (edge pixels mix two surfaces
//      and carry lens fringes), and not next to clipping (those were
//      demosaiced partly from clipped samples). Their chromaticity is spread
//      into the clipped areas by a pull-push pyramid at half resolution, so a
//      clipped region takes the colour of what surrounds it at whatever scale
//      reaches it, and fades to neutral where nothing reliable is near.
//   3. A partly clipped pixel keeps its unclipped channels; the clipped ones
//      are rebuilt at that chromaticity, at the level the unclipped channels
//      imply, and never below the clamp they were cut at.
//   4. A pixel with every channel clipped carries no colour and no level: it
//      becomes neutral, from the clamp at the rim of a clipped area rising to
//      the brightest level the sensor allows at its heart, and the partly
//      clipped pixels around fade into that same level -- so the core of a
//      light is never darker than its halo, and never steps up from it.
//
// Then the camera matrix goes back on. The approach is lightcraft's
// (crates/raw/src/highlight.rs), simplified: no "rim" colour blending, and a
// single trust fade instead of its agreement tests.
//
// TWO PATHS, ONE ALGORITHM. The CPU code below is written as the GPU's passes
// are -- the same pyramid sizes (each level the one above halved, rounded
// down, as the GPU's scratch planes are), the same fixed number of levels, the
// same trust carried down the push -- so the two can be held to agree exactly
// rather than approximately. At 45 MP the CPU path takes ~1.8 s; the GPU path
// is 35 passes, all but three on planes a quarter of the image or smaller.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "../../algo_util/color.h"
#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"
#include "../../core/parallel.h"
#include "clip_repair.h"

namespace tglab {
namespace {

inline float Smoothstep(float a, float b, float x) {
    const float t = std::clamp((x - a) / (b - a), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

bool Invert3(const float* m, float* inv) {
    const double a = m[0], b = m[1], c = m[2], d = m[3], e = m[4], f = m[5],
                 g = m[6], h = m[7], k = m[8];
    const double det = a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g);
    if (std::abs(det) < 1e-9) return false;
    const double r = 1.0 / det;
    inv[0] = float((e * k - f * h) * r); inv[1] = float((c * h - b * k) * r); inv[2] = float((b * f - c * e) * r);
    inv[3] = float((f * g - d * k) * r); inv[4] = float((a * k - c * g) * r); inv[5] = float((c * d - a * f) * r);
    inv[6] = float((d * h - e * g) * r); inv[7] = float((b * g - a * h) * r); inv[8] = float((a * e - b * d) * r);
    return true;
}

// Reliable pixels lend colour in proportion to how bright they are, relative
// to the clamp: from nothing below kSourceLo to fully above kSourceHi.
constexpr float kSourceLo = 0.25f, kSourceHi = 0.6f;
// A source on an edge (a 3x3 neighbour brighter or darker by this many stops)
// fades out between these.
constexpr float kEdgeLo = 0.15f, kEdgeHi = 0.4f;
// Pixels within this many pixels of a clipped one are not sources.
constexpr int kMargin = 2;
// The window, (2 * kDepthR + 1) square, over which "how deep into a clipped
// area" is the fraction of fully clipped pixels.
constexpr int kDepthR = 3;
// How much a coarser estimate counts against a cell's own sources, in units
// of source density.
constexpr float kPrior = 0.01f;
// The rebuilt colour fades to neutral where reliable content covers less than
// kSupport of the surroundings at every scale up to kReach of the long edge.
constexpr float kReach = 0.25f, kSupport = 0.05f;
// Pyramid levels, fixed so the GPU's passes are: level k is the image / 2^(k+1).
// Sixteen reach 1x1 from 131k pixels; past 1x1 a level is a copy of the last.
constexpr int kLevels = 16;

// The channels at the clamp, as bits.
inline int ClipMask(const float* q, float ceiling) {
    int m = 0;
    for (int c = 0; c < 3; ++c) if (q[c] >= 0.995f * ceiling) m |= 1 << c;
    return m;
}

// Which pyramid level first spans kReach of the long edge (level k's cells are
// 2^(k+1) pixels).
int WidestLevel(int w, int h) {
    const float reachPx = kReach * float(std::max(w, h));
    for (int i = 0; i < kLevels; ++i)
        if (float(2 << i) >= reachPx) return i;
    return kLevels - 1;
}

// A plane of four floats per cell.
struct Plane4 {
    int w = 1, h = 1;
    std::vector<float> v;
    void Alloc(int ww, int hh) { w = ww; h = hh; v.assign(size_t(w) * size_t(h) * 4, 0.0f); }
    float* At(int x, int y) { return &v[(size_t(y) * size_t(w) + size_t(x)) * 4]; }
    const float* At(int x, int y) const { return &v[(size_t(y) * size_t(w) + size_t(x)) * 4]; }
};

// Bilinear at a cell of the level BELOW (twice the resolution), clamped --
// exactly as the shader's SampleUp.
void SampleUp(const Plane4& l, int x, int y, float* out) {
    const float fx = std::clamp((float(x) + 0.5f) * 0.5f - 0.5f, 0.0f, float(l.w - 1));
    const float fy = std::clamp((float(y) + 0.5f) * 0.5f - 0.5f, 0.0f, float(l.h - 1));
    const int x0 = int(fx), y0 = int(fy);
    const int x1 = std::min(x0 + 1, l.w - 1), y1 = std::min(y0 + 1, l.h - 1);
    const float tx = fx - float(x0), ty = fy - float(y0);
    for (int c = 0; c < 4; ++c) {
        const float a = l.At(x0, y0)[c], b = l.At(x1, y0)[c];
        const float d = l.At(x0, y1)[c], e = l.At(x1, y1)[c];
        const float top = a + (b - a) * tx, bot = d + (e - d) * tx;
        out[c] = top + (bot - top) * ty;
    }
}

// A partly clipped pixel, rebuilt at chromaticity r (three components): its
// measured channels kept, its clipped ones at the level the measured ones
// imply, never below what they were cut at.
void Rebuild(const float* q, int m, const float* r, float maxMul, float* o) {
    float sum = 0.0f; int k = 0;
    for (int c = 0; c < 3; ++c)
        if (!(m & (1 << c)) && r[c] > 1e-3f) { sum += q[c] / r[c]; ++k; }
    if (k == 0) { for (int c = 0; c < 3; ++c) o[c] = q[c]; return; }
    float floorL = 0.0f, topR = 1e-3f;
    for (int c = 0; c < 3; ++c) {
        topR = std::max(topR, r[c]);
        if (m & (1 << c)) floorL = std::max(floorL, q[c] / std::max(r[c], 1e-3f));
    }
    const float level = std::max(sum / float(k), std::min(floorL, maxMul / topR));
    for (int c = 0; c < 3; ++c)
        o[c] = (m & (1 << c)) ? std::min(maxMul, std::max(q[c], level * r[c])) : q[c];
}

// --- HLSL ----------------------------------------------------------------

const char* const kHrCommon = R"(
Texture2D<float4>   T0 : register(t0);
Texture2D<float4>   T1 : register(t1);
Texture2D<float4>   T2 : register(t2);
RWTexture2D<float4> U0 : register(u0);

cbuffer Params : register(b0) {
    uint Width;
    uint Height;
    uint Mi0, Mi1, Mi2, Mi3, Mi4, Mi5, Mi6, Mi7, Mi8;   // inverse camera matrix
    uint Mc0, Mc1, Mc2, Mc3, Mc4, Mc5, Mc6, Mc7, Mc8;   // camera matrix
    uint MulR, MulG, MulB;
    uint CeilingBits, MaxMulBits;
    uint Widest, Level, Active;
};

static const float kSourceLo = 0.25, kSourceHi = 0.6;
static const float kEdgeLo = 0.15, kEdgeHi = 0.4;
static const float kPrior = 0.01, kSupport = 0.05;

float Smooth(float a, float b, float x) { float t = saturate((x - a) / (b - a)); return t * t * (3.0 - 2.0 * t); }
float Ceiling() { return asfloat(CeilingBits); }
float MaxMul()  { return asfloat(MaxMulBits); }
float3x3 Minv() {
    return float3x3(asfloat(Mi0), asfloat(Mi1), asfloat(Mi2), asfloat(Mi3), asfloat(Mi4),
                    asfloat(Mi5), asfloat(Mi6), asfloat(Mi7), asfloat(Mi8));
}
float3x3 Mcam() {
    return float3x3(asfloat(Mc0), asfloat(Mc1), asfloat(Mc2), asfloat(Mc3), asfloat(Mc4),
                    asfloat(Mc5), asfloat(Mc6), asfloat(Mc7), asfloat(Mc8));
}
float3 Bal(float3 rgb) {
    float3x3 m = Minv();
    return float3(m[0][0] * rgb.r + m[0][1] * rgb.g + m[0][2] * rgb.b,
                  m[1][0] * rgb.r + m[1][1] * rgb.g + m[1][2] * rgb.b,
                  m[2][0] * rgb.r + m[2][1] * rgb.g + m[2][2] * rgb.b);
}
int ClipMask(float3 q) {
    float c = 0.995 * Ceiling();
    return (q.r >= c ? 1 : 0) | (q.g >= c ? 2 : 0) | (q.b >= c ? 4 : 0);
}
int2 Dims(Texture2D<float4> t) { uint w, h; t.GetDimensions(w, h); return int2(w, h); }
// Bilinear at a cell of the level below, clamped -- as SampleUp on the CPU.
float4 SampleUp(Texture2D<float4> t, int x, int y) {
    int2 d = Dims(t);
    float fx = clamp((float(x) + 0.5) * 0.5 - 0.5, 0.0, float(d.x - 1));
    float fy = clamp((float(y) + 0.5) * 0.5 - 0.5, 0.0, float(d.y - 1));
    int x0 = int(fx), y0 = int(fy);
    int x1 = min(x0 + 1, d.x - 1), y1 = min(y0 + 1, d.y - 1);
    float tx = fx - float(x0), ty = fy - float(y0);
    float4 a = t.Load(int3(x0, y0, 0)), b = t.Load(int3(x1, y0, 0));
    float4 c = t.Load(int3(x0, y1, 0)), e = t.Load(int3(x1, y1, 0));
    float4 top = a + (b - a) * tx, bot = c + (e - c) * tx;
    return top + (bot - top) * ty;
}
)";

// Horizontal: near clipping within 2, and the count of fully clipped within 3.
const char* const kHrClipH = R"(
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    float nearH = 0.0, full = 0.0;
    [unroll] for (int k = -3; k <= 3; ++k) {
        int xx = clamp(int(tid.x) + k, 0, int(Width) - 1);
        int m = ClipMask(Bal(T0.Load(int3(xx, tid.y, 0)).rgb));
        if (abs(k) <= 2 && m != 0) nearH = 1.0;
        if (m == 7) full += 1.0;
    }
    U0[tid.xy] = float4(nearH, full, 0, 1);
}
)";

// Vertical: completes both.
const char* const kHrClipV = R"(
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    float nearC = 0.0, full = 0.0;
    [unroll] for (int k = -3; k <= 3; ++k) {
        int yy = clamp(int(tid.y) + k, 0, int(Height) - 1);
        float2 v = T0.Load(int3(tid.x, yy, 0)).xy;
        if (abs(k) <= 2) nearC = max(nearC, v.x);
        full += v.y;
    }
    U0[tid.xy] = float4(nearC, full, 0, 1);
}
)";

// The colour sources, at half resolution: T0 the image, T1 near/depth.
const char* const kHrSource = R"(
float LumAt(int x, int y) {
    int2 d = Dims(T0);
    float3 q = Bal(T0.Load(int3(clamp(x, 0, d.x - 1), clamp(y, 0, d.y - 1), 0)).rgb);
    return max(1e-6, max(0.0, q.r) + max(0.0, q.g) + max(0.0, q.b));
}
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    int2 d = Dims(T0);
    float sr = 0, sb = 0, sw = 0; int k = 0;
    float edgeLo = exp2(kEdgeLo), edgeHi = exp2(kEdgeHi);
    for (int dy = 0; dy < 2; ++dy)
        for (int dx = 0; dx < 2; ++dx) {
            int x = 2 * int(tid.x) + dx, y = 2 * int(tid.y) + dy;
            if (x >= d.x || y >= d.y) continue;
            ++k;
            if (T1.Load(int3(x, y, 0)).x > 0.5) continue;
            float3 q = Bal(T0.Load(int3(x, y, 0)).rgb);
            float top = max(q.r, max(q.g, q.b)) / Ceiling();
            if (top <= kSourceLo) continue;
            float s = LumAt(x, y);
            float lo = s, hi = s;
            for (int ey = -1; ey <= 1; ++ey)
                for (int ex = -1; ex <= 1; ++ex) {
                    float v = LumAt(x + ex, y + ey);
                    lo = min(lo, v); hi = max(hi, v);
                }
            float keep = 1.0 - Smooth(edgeLo, edgeHi, max(hi / s, s / lo));
            float wgt = Smooth(kSourceLo, kSourceHi, top) * keep;
            if (wgt <= 0.0) continue;
            sr += wgt * max(0.0, q.r) / s;
            sb += wgt * max(0.0, q.b) / s;
            sw += wgt;
        }
    float inv = 1.0 / float(max(k, 1));
    U0[tid.xy] = float4(sr * inv, sb * inv, sw * inv, 0);
}
)";

// One level up: the mean of its in-range children.
const char* const kHrPull = R"(
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    int2 d = Dims(T0);
    float4 s = 0; int k = 0;
    for (int dy = 0; dy < 2; ++dy)
        for (int dx = 0; dx < 2; ++dx) {
            int x = 2 * int(tid.x) + dx, y = 2 * int(tid.y) + dy;
            if (x >= d.x || y >= d.y) continue;
            s += T0.Load(int3(x, y, 0)); ++k;
        }
    U0[tid.xy] = s / float(max(k, 1));
}
)";

// The top: its own sources' mean colour, or neutral; its trust.
const char* const kHrTop = R"(
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    float4 l = T0.Load(int3(tid.xy, 0));
    float r = l.z > 0.0 ? l.x / l.z : 1.0 / 3.0;
    float b = l.z > 0.0 ? l.y / l.z : 1.0 / 3.0;
    float t = Level <= Widest ? Smooth(0.0, kSupport, l.z) : 0.0;
    U0[tid.xy] = float4(r, b, t, 0);
}
)";

// One level down: T0 this level's sources, T1 the coarser estimate.
const char* const kHrPush = R"(
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    float4 l = T0.Load(int3(tid.xy, 0));
    float4 e = SampleUp(T1, int(tid.x), int(tid.y));
    float dd = l.z + kPrior;
    float t = Level <= Widest ? max(Smooth(0.0, kSupport, l.z), e.z) : 0.0;
    U0[tid.xy] = float4((l.x + kPrior * e.x) / dd, (l.y + kPrior * e.y) / dd, t, 0);
}
)";

// The rebuild: T0 the image, T1 near/depth, T2 the finest estimate.
const char* const kHrRebuild = R"(
void RebuildPx(float3 q, int m, float3 r, out float3 o) {
    float sum = 0.0; int k = 0;
    [unroll] for (int c = 0; c < 3; ++c)
        if (!(m & (1 << c)) && r[c] > 1e-3) { sum += q[c] / r[c]; ++k; }
    if (k == 0) { o = q; return; }
    float floorL = 0.0, topR = 1e-3;
    [unroll] for (int c2 = 0; c2 < 3; ++c2) {
        topR = max(topR, r[c2]);
        if (m & (1 << c2)) floorL = max(floorL, q[c2] / max(r[c2], 1e-3));
    }
    float level = max(sum / float(k), min(floorL, MaxMul() / topR));
    [unroll] for (int c3 = 0; c3 < 3; ++c3)
        o[c3] = (m & (1 << c3)) ? min(MaxMul(), max(q[c3], level * r[c3])) : q[c3];
}
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    float4 src = T0.Load(int3(tid.xy, 0));
    float3 q = Bal(src.rgb);
    int m = ClipMask(q);
    if (Active == 0 || m == 0) { U0[tid.xy] = src; return; }
    float depth = T1.Load(int3(tid.xy, 0)).y / 49.0;
    float L = Ceiling() + (MaxMul() - Ceiling()) * Smooth(0.0, 1.0, depth);
    float3 o;
    if (m == 7) {
        o = L;
    } else {
        float4 e = SampleUp(T2, int(tid.x), int(tid.y));
        float3 local = float3(e.x, max(0.0, 1.0 - e.x - e.y), e.y);
        float3 a, b;
        RebuildPx(q, m, local, a);
        RebuildPx(q, m, float3(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0), b);
        o = b + (a - b) * e.z;
        float3 mul3 = float3(asfloat(MulR), asfloat(MulG), asfloat(MulB));
        float nearest = 0.0;
        [unroll] for (int c = 0; c < 3; ++c)
            if (!(m & (1 << c))) nearest = max(nearest, q[c] / mul3[c]);
        float fade = max(Smooth(0.75, 1.0, nearest), depth);
        o += (max(L, o) - o) * fade;
    }
    CameraMatrixInGamut(o, Mcam());
    U0[tid.xy] = float4(o, src.a);
}
)";

class HighlightReconstruct : public AlgorithmBase {
public:
    const char* Name()     const override { return "highlight_reconstruct"; }
    const char* Category() const override { return "adjust"; }

    PortList Inputs()  const override { return {{"src", DataType::Image, FormatSpec::Any}}; }
    PortList Outputs() const override { return {{"out", DataType::Image, FormatSpec::SameAsInput}}; }

    // The colour a clipped region takes comes from anywhere within a quarter
    // of the frame: on a region it would be whatever the region holds.
    bool RegionSafe() const override { return false; }

    void RunCPU(RunCtx& ctx) override {
        const ImageView src = ctx.In(0);
        ImageView dst = ctx.Out(0);
        if (!src.Valid() || !dst.Valid()) return;
        PixelBuffer buf;
        buf.Unpack(src);
        if (!buf.Valid()) return;
        const ImageDesc& d = src.desc;
        const int w = buf.Width(), h = buf.Height(), ch = buf.Channels();
        const float scale = buf.ValueScale();

        float minv[9];
        const char* why = Refuse(d, ch, minv);
        if (why) {
            buf.PackInto(dst);
            ctx.SetReport(why);
            return;
        }
        const float ceiling = std::min(d.camMul[0], std::min(d.camMul[1], d.camMul[2]));
        const float maxMul  = std::max(d.camMul[0], std::max(d.camMul[1], d.camMul[2]));

        const size_t n = size_t(w) * size_t(h);
        std::vector<float> bal(n * 3);          // white-balanced camera RGB
        std::vector<uint8_t> clip(n, 0);        // bit c: channel c at the clamp
        std::vector<float>& px = buf.Data();
        ParallelFor(size_t(h), [&](size_t y) {
            for (int x = 0; x < w; ++x) {
                const size_t i = y * size_t(w) + size_t(x);
                const float* p = &px[i * size_t(ch)];
                const float r = p[0] / scale, g = p[1] / scale, b = p[2] / scale;
                float* q = &bal[i * 3];
                q[0] = minv[0] * r + minv[1] * g + minv[2] * b;
                q[1] = minv[3] * r + minv[4] * g + minv[5] * b;
                q[2] = minv[6] * r + minv[7] * g + minv[8] * b;
                clip[i] = uint8_t(ClipMask(q, ceiling));
            }
        });
        size_t clipped = 0, full = 0;
        for (uint8_t m : clip) { clipped += m ? 1 : 0; full += (m == 7) ? 1 : 0; }
        if (clipped == 0) {
            buf.PackInto(dst);
            ctx.SetReport("no clipped pixels");
            return;
        }

        // Near clipping (within kMargin) and the count of fully clipped pixels
        // (within kDepthR), separably: the GPU's clipH and clipV passes.
        std::vector<float> nearH(n), fullH(n), nearC(n), fullC(n);
        ParallelFor(size_t(h), [&](size_t y) {
            for (int x = 0; x < w; ++x) {
                float nv = 0.0f, fv = 0.0f;
                for (int k = -kDepthR; k <= kDepthR; ++k) {
                    const int m = clip[y * size_t(w) + size_t(std::clamp(x + k, 0, w - 1))];
                    if (std::abs(k) <= kMargin && m != 0) nv = 1.0f;
                    if (m == 7) fv += 1.0f;
                }
                nearH[y * size_t(w) + size_t(x)] = nv;
                fullH[y * size_t(w) + size_t(x)] = fv;
            }
        });
        ParallelFor(size_t(h), [&](size_t y) {
            for (int x = 0; x < w; ++x) {
                float nv = 0.0f, fv = 0.0f;
                for (int k = -kDepthR; k <= kDepthR; ++k) {
                    const size_t j = size_t(std::clamp(int(y) + k, 0, h - 1)) * size_t(w) + size_t(x);
                    if (std::abs(k) <= kMargin) nv = std::max(nv, nearH[j]);
                    fv += fullH[j];
                }
                nearC[y * size_t(w) + size_t(x)] = nv;
                fullC[y * size_t(w) + size_t(x)] = fv;
            }
        });

        // The source level, at half resolution, rounded down.
        std::vector<Plane4> L(kLevels), E(kLevels);
        L[0].Alloc(std::max(1, w / 2), std::max(1, h / 2));
        const float edgeLo = std::exp2(kEdgeLo), edgeHi = std::exp2(kEdgeHi);
        auto lumAt = [&](int x, int y) {
            const float* q = &bal[(size_t(std::clamp(y, 0, h - 1)) * size_t(w) +
                                   size_t(std::clamp(x, 0, w - 1))) * 3];
            return std::max(1e-6f, std::max(0.0f, q[0]) + std::max(0.0f, q[1]) + std::max(0.0f, q[2]));
        };
        ParallelFor(size_t(L[0].h), [&](size_t cy) {
            for (int cx = 0; cx < L[0].w; ++cx) {
                float sr = 0, sb = 0, sw = 0; int k = 0;
                for (int dy = 0; dy < 2; ++dy)
                    for (int dx = 0; dx < 2; ++dx) {
                        const int x = 2 * cx + dx, y = 2 * int(cy) + dy;
                        if (x >= w || y >= h) continue;
                        ++k;
                        const size_t i = size_t(y) * size_t(w) + size_t(x);
                        if (nearC[i] > 0.5f) continue;
                        const float* q = &bal[i * 3];
                        const float top = std::max(q[0], std::max(q[1], q[2])) / ceiling;
                        if (top <= kSourceLo) continue;
                        const float s = lumAt(x, y);
                        float lo = s, hi = s;
                        for (int ey = -1; ey <= 1; ++ey)
                            for (int ex = -1; ex <= 1; ++ex) {
                                const float v = lumAt(x + ex, y + ey);
                                lo = std::min(lo, v); hi = std::max(hi, v);
                            }
                        const float keep = 1.0f - Smoothstep(edgeLo, edgeHi, std::max(hi / s, s / lo));
                        const float wgt = Smoothstep(kSourceLo, kSourceHi, top) * keep;
                        if (wgt <= 0.0f) continue;
                        sr += wgt * std::max(0.0f, q[0]) / s;
                        sb += wgt * std::max(0.0f, q[2]) / s;
                        sw += wgt;
                    }
                const float inv = 1.0f / float(std::max(k, 1));
                float* o = L[0].At(cx, int(cy));
                o[0] = sr * inv; o[1] = sb * inv; o[2] = sw * inv; o[3] = 0.0f;
            }
        });

        // Pull, a fixed kLevels deep: each level the one below halved, rounded
        // down, never under 1x1.
        for (int k = 1; k < kLevels; ++k) {
            const Plane4& f = L[size_t(k - 1)];
            Plane4& g = L[size_t(k)];
            g.Alloc(std::max(1, w >> (k + 1)), std::max(1, h >> (k + 1)));
            for (int y = 0; y < g.h; ++y)
                for (int x = 0; x < g.w; ++x) {
                    float s[4] = {0, 0, 0, 0}; int cnt = 0;
                    for (int dy = 0; dy < 2; ++dy)
                        for (int dx = 0; dx < 2; ++dx) {
                            const int sx = 2 * x + dx, sy = 2 * y + dy;
                            if (sx >= f.w || sy >= f.h) continue;
                            for (int c = 0; c < 4; ++c) s[c] += f.At(sx, sy)[c];
                            ++cnt;
                        }
                    float* o = g.At(x, y);
                    for (int c = 0; c < 4; ++c) o[c] = s[c] / float(std::max(cnt, 1));
                }
        }

        // The top, then push down: each level's sources with the coarser
        // estimate, and the trust -- how much reliable content lies around at
        // any scale up to kReach -- carried down with it.
        const int widest = WidestLevel(w, h);
        {
            const Plane4& l = L[kLevels - 1];
            Plane4& e = E[kLevels - 1];
            e.Alloc(l.w, l.h);
            for (int y = 0; y < l.h; ++y)
                for (int x = 0; x < l.w; ++x) {
                    const float* s = l.At(x, y);
                    float* o = e.At(x, y);
                    o[0] = s[2] > 0.0f ? s[0] / s[2] : 1.0f / 3.0f;
                    o[1] = s[2] > 0.0f ? s[1] / s[2] : 1.0f / 3.0f;
                    o[2] = (kLevels - 1) <= widest ? Smoothstep(0.0f, kSupport, s[2]) : 0.0f;
                    o[3] = 0.0f;
                }
        }
        for (int k = kLevels - 2; k >= 0; --k) {
            const Plane4& l = L[size_t(k)];
            const Plane4& coarse = E[size_t(k + 1)];
            Plane4& e = E[size_t(k)];
            e.Alloc(l.w, l.h);
            ParallelFor(size_t(l.h), [&](size_t y) {
                for (int x = 0; x < l.w; ++x) {
                    const float* s = l.At(x, int(y));
                    float c[4];
                    SampleUp(coarse, x, int(y), c);
                    const float dd = s[2] + kPrior;
                    float* o = e.At(x, int(y));
                    o[0] = (s[0] + kPrior * c[0]) / dd;
                    o[1] = (s[1] + kPrior * c[1]) / dd;
                    o[2] = k <= widest ? std::max(Smoothstep(0.0f, kSupport, s[2]), c[2]) : 0.0f;
                    o[3] = 0.0f;
                }
            });
        }

        // Rebuild, and the camera matrix back on.
        std::vector<size_t> partRows(size_t(h), 0);
        ParallelFor(size_t(h), [&](size_t y) {
            for (int x = 0; x < w; ++x) {
                const size_t i = y * size_t(w) + size_t(x);
                const int m = clip[i];
                if (!m) continue;
                const float* q = &bal[i * 3];
                const float depth = fullC[i] / float((2 * kDepthR + 1) * (2 * kDepthR + 1));
                const float L0 = ceiling + (maxMul - ceiling) * Smoothstep(0.0f, 1.0f, depth);
                float out[3];
                if (m == 7) {
                    for (float& v : out) v = L0;      // neutral; see the header, point 4
                } else {
                    float e[4];
                    SampleUp(E[0], x, int(y), e);
                    const float local[3] = {e[0], std::max(0.0f, 1.0f - e[0] - e[1]), e[1]};
                    const float plain[3] = {1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f};
                    float a[3], b[3];
                    Rebuild(q, m, local, maxMul, a);
                    Rebuild(q, m, plain, maxMul, b);
                    for (int c = 0; c < 3; ++c) out[c] = b[c] + (a[c] - b[c]) * e[2];

                    // ON THE WAY TO FULLY CLIPPED, fade into it: as the
                    // channels still measured approach their own clip, and as
                    // the pixel lies deeper in a clipped area, it goes over to
                    // the clipped neutral -- or the core of a light stands out
                    // from the halo round it in a hard, speckled edge.
                    float nearest = 0.0f;
                    for (int c = 0; c < 3; ++c)
                        if (!(m & (1 << c))) nearest = std::max(nearest, q[c] / d.camMul[c]);
                    const float fade = std::max(Smoothstep(0.75f, 1.0f, nearest), depth);
                    for (int c = 0; c < 3; ++c) out[c] += (std::max(L0, out[c]) - out[c]) * fade;
                    ++partRows[y];
                }
                CameraMatrixInGamut(d, out);
                float* p = &px[i * size_t(ch)];
                for (int c = 0; c < 3; ++c) p[c] = out[c] * scale;
            }
        });
        size_t rebuiltPartial = 0;
        for (size_t v : partRows) rebuiltPartial += v;
        buf.PackInto(dst);
        char r[160];
        std::snprintf(r, sizeof r, "%zu partly clipped pixels rebuilt from nearby colour, %zu fully "
                                   "clipped set neutral (%.2f%% of the frame)",
                      rebuiltPartial, full, 100.0 * double(clipped) / double(n));
        ctx.SetReport(r);
    }

    // --- GPU ----------------------------------------------------------------
    bool HasGPU() const override { return true; }

    // Two full-size planes for the near/depth counts (exact in half floats),
    // then the pyramid: kLevels source planes and kLevels estimate planes, level
    // k at 1 / 2^(k+1) of the image.
    int GpuScratchCount() const override { return 2 + 2 * kLevels; }
    FormatSpec GpuScratchPlaneFormat(int i) const override {
        return i < 2 ? FormatSpec::RG16F : FormatSpec::RGBA32F;
    }
    int GpuScratchPlaneDivisor(int i) const override {
        if (i < 2) return 1;
        const int k = (i - 2) % kLevels;
        return 1 << (k + 1);
    }

    void PrepareGpu(const std::vector<ImageDesc>& inputs) override {
        if (inputs.empty()) return;
        const ImageDesc& d = inputs[0];
        m_desc = d;
        m_active = Refuse(d, d.format == Format::R32F ? 1 : 4, m_minv) == nullptr &&
                   d.format != Format::RGBA8;
        m_widest = WidestLevel(d.width, d.height);
        m_gpuRan = true;
    }

    std::vector<GpuPass> GpuPasses() const override {
        static const std::string pre = std::string(kDemosaicHlsl) + kHrCommon;
        static const std::string clipH = pre + kHrClipH, clipV = pre + kHrClipV;
        static const std::string source = pre + kHrSource, pull = pre + kHrPull;
        static const std::string top = pre + kHrTop, push = pre + kHrPush;
        static const std::string rebuild = pre + kHrRebuild;
        auto Lp = [](int k) { return 2 + k; };
        auto Ep = [](int k) { return 2 + kLevels + k; };
        std::vector<GpuPass> p;
        p.push_back({clipH.c_str(), "clipH", {-1}, {0}});
        p.push_back({clipV.c_str(), "clipV", {0}, {1}});
        p.push_back({source.c_str(), "source", {-1, 1}, {Lp(0)}});
        for (int k = 1; k < kLevels; ++k) p.push_back({pull.c_str(), "pull", {Lp(k - 1)}, {Lp(k)}});
        p.push_back({top.c_str(), "top", {Lp(kLevels - 1)}, {Ep(kLevels - 1)}});
        for (int k = kLevels - 2; k >= 0; --k)
            p.push_back({push.c_str(), "push", {Lp(k), Ep(k + 1)}, {Ep(k)}});
        p.push_back({rebuild.c_str(), "rebuild", {-1, 1, Ep(0)}, {-1}});
        return p;
    }

    std::vector<uint32_t> GpuPassConstants(int pass) const override {
        // The level each pass works at: pulls 1..kLevels-1, then the top, then
        // pushes kLevels-2 down to 0.
        int level = 0;
        if (pass >= 3 && pass < 3 + kLevels - 1) level = pass - 2;
        else if (pass == 3 + kLevels - 1) level = kLevels - 1;
        else if (pass > 3 + kLevels - 1 && pass < 3 + 2 * kLevels - 1)
            level = kLevels - 2 - (pass - (3 + kLevels));
        const ImageDesc& d = m_desc;
        std::vector<uint32_t> c;
        for (int i = 0; i < 9; ++i) c.push_back(FloatBits(m_minv[i]));
        for (int i = 0; i < 9; ++i) c.push_back(FloatBits(d.rgbCam[i]));
        for (int i = 0; i < 3; ++i) c.push_back(FloatBits(d.camMul[i]));
        c.push_back(FloatBits(std::min(d.camMul[0], std::min(d.camMul[1], d.camMul[2]))));
        c.push_back(FloatBits(std::max(d.camMul[0], std::max(d.camMul[1], d.camMul[2]))));
        c.push_back(uint32_t(m_widest));
        c.push_back(uint32_t(level));
        c.push_back(m_active ? 1u : 0u);
        return c;
    }

    // The GPU path's note; the CPU path reports per call.
    std::string RunReport() const override {
        if (!m_gpuRan) return {};
        return m_active ? "clipped highlights rebuilt on the GPU (not counted there)"
                        : "not a demosaiced raw, or not straight off a demosaic: passed through";
    }

private:
    // Why this image cannot be reconstructed, or null when it can (and then
    // `minv` holds the inverse camera matrix).
    static const char* Refuse(const ImageDesc& d, int ch, float* minv) {
        const float ceiling = std::min(d.camMul[0], std::min(d.camMul[1], d.camMul[2]));
        const float maxMul  = std::max(d.camMul[0], std::max(d.camMul[1], d.camMul[2]));
        // rawClamp: straight off a demosaic, so "at the clamp" still means
        // clipped. A merge, a stitch or any adjustment in between clears it.
        if (d.rawClamp <= 0.0f && d.linear && maxMul > ceiling * 1.0001f)
            return "not straight off a demosaic (after a merge or an adjustment the "
                   "clamp no longer marks clipping): passed through";
        if (d.rawClamp <= 0.0f || !d.linear || ch < 3 || ceiling <= 0.0f ||
            maxMul <= ceiling * 1.0001f || !Invert3(d.rgbCam, minv))
            return "not a demosaiced raw: nothing to reconstruct";
        return nullptr;
    }

    ImageDesc m_desc{};              // set by PrepareGpu
    float m_minv[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    int m_widest = 0;
    bool m_active = false, m_gpuRan = false;
};

REGISTER_ALGORITHM(HighlightReconstruct);

} // namespace
} // namespace tglab
