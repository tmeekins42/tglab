#include "splat_train_gpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <mutex>

#include "../core/parallel.h"
#include "../gpu/compute.h"
#include "splat_kernels.h"

namespace tglab {
namespace {

constexpr int kTexW = 2048;
constexpr int kTile = 16;

// Texels per Gaussian in the state texture: parameters (14 floats in 4
// texels), Adam's first moment (4), second moment (4), and the densification
// counters (1) -- gradient sum, view count, largest screen radius.
constexpr int kStateTexels = 13;

uint32_t Bits(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, 4);
    return u;
}

// The tiles a Gaussian's screen footprint touches, inclusive; empty (x1 < x0)
// when it is out of view.
struct TileRect { int x0, y0, x1, y1; };

// A depth as an unsigned key that orders the same way the float does, so the
// sort can be a radix sort. -0 is folded into +0 first: the float comparison
// calls them equal, and an equal pair must fall back to index order.
uint32_t DepthKey(float d) {
    const uint32_t b = Bits(d + 0.0f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// Sorts one tile's entries near to far, STABLY. The entries arrive in index
// order, so a stable sort by depth gives exactly the (depth, index) order the
// CPU rasteriser uses -- and therefore the same render, to the bit.
//
// LSD radix, a byte per pass, skipping any byte every key shares (in one tile
// the top byte usually is). `tk`/`ti` are scratch of the same length.
void RadixSortTile(uint32_t* key, uint32_t* idx, uint32_t* tk, uint32_t* ti, size_t m) {
    if (m < 2) return;
    uint32_t hist[4][256] = {};
    for (size_t j = 0; j < m; ++j) {
        const uint32_t k = key[j];
        ++hist[0][k & 255];
        ++hist[1][(k >> 8) & 255];
        ++hist[2][(k >> 16) & 255];
        ++hist[3][k >> 24];
    }
    uint32_t *sk = key, *si = idx, *dk = tk, *di = ti;
    for (int b = 0; b < 4; ++b) {
        const int shift = 8 * b;
        uint32_t* h = hist[b];
        if (h[(sk[0] >> shift) & 255] == m) continue;
        uint32_t sum = 0;
        for (int v = 0; v < 256; ++v) {
            const uint32_t c = h[v];
            h[v] = sum;
            sum += c;
        }
        for (size_t j = 0; j < m; ++j) {
            const uint32_t p = h[(sk[j] >> shift) & 255]++;
            dk[p] = sk[j];
            di[p] = si[j];
        }
        std::swap(sk, dk);
        std::swap(si, di);
    }
    if (sk != key) {
        std::memcpy(key, sk, m * sizeof(uint32_t));
        std::memcpy(idx, si, m * sizeof(uint32_t));
    }
}

ImageDesc FlatDesc(size_t texels, Format f) {
    ImageDesc d;
    d.width  = kTexW;
    d.height = int(std::max<size_t>(1, (texels + kTexW - 1) / kTexW));
    d.format = f;
    return d;
}

using Clock = std::chrono::steady_clock;
double MsSince(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

// Shared by the per-Gaussian kernels: the state layout, and the camera.
constexpr const char* kCommon = R"(
uint2 At(uint i, uint texW) { return uint2(i % texW, i / texW); }

// Fourteen floats from four consecutive texels starting at `base`.
void Load14(Texture2D<float4> tex, uint base, uint texW, out float p[14]) {
    float4 a = tex[At(base, texW)],     b = tex[At(base + 1, texW)];
    float4 c = tex[At(base + 2, texW)], d = tex[At(base + 3, texW)];
    p[0] = a.x; p[1] = a.y; p[2] = a.z; p[3] = a.w;
    p[4] = b.x; p[5] = b.y; p[6] = b.z; p[7] = b.w;
    p[8] = c.x; p[9] = c.y; p[10] = c.z; p[11] = c.w;
    p[12] = d.x; p[13] = d.y;
}

float Sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }

// The Gaussian's rotation from its (normalised) quaternion, row-major, as
// Splat::Rotation and the CPU rasteriser build it.
void Rotation(float4 q, out float R[9]) {
    float w = q.x, x = q.y, y = q.z, z = q.w;
    R[0] = 1 - 2 * (y * y + z * z); R[1] = 2 * (x * y - w * z);     R[2] = 2 * (x * z + w * y);
    R[3] = 2 * (x * y + w * z);     R[4] = 1 - 2 * (x * x + z * z); R[5] = 2 * (y * z - w * x);
    R[6] = 2 * (x * z - w * y);     R[7] = 2 * (y * z + w * x);     R[8] = 1 - 2 * (x * x + y * y);
}
)";

// --- project -------------------------------------------------------------------
//
// One thread per Gaussian: the CPU's Project, in float. Writes the three
// texels the composite and backward kernels read, and one screen texel --
// centre, depth and how far out the Gaussian can matter -- which is all the
// CPU needs to bin it into tiles.
constexpr const char* kProject = R"(
Texture2D<float4>   State  : register(t0);
RWTexture2D<float4> Proj   : register(u0);
RWTexture2D<float4> Screen : register(u1);

cbuffer Params : register(b0) {
    uint D0; uint D1;
    uint N; uint TexW;
    uint R0; uint R1; uint R2; uint R3; uint R4; uint R5; uint R6; uint R7; uint R8;
    uint T0; uint T1; uint T2;
    uint Fx; uint Fy; uint Cx; uint Cy;
    uint ImgW; uint ImgH;
    uint LowPass; uint MinA;
};

void Hide(uint i) {
    Proj[At(i * 3, TexW)]     = float4(0, 0, 0, -1);
    Proj[At(i * 3 + 1, TexW)] = float4(0, 0, 0, 0);
    Proj[At(i * 3 + 2, TexW)] = float4(0, 0, 0, 0);
    Screen[At(i, TexW)]       = float4(0, 0, 0, -1);
}

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    if (i >= N) return;

    float p[14];
    Load14(State, i * 13, TexW, p);

    float W[9] = { asfloat(R0), asfloat(R1), asfloat(R2), asfloat(R3), asfloat(R4),
                   asfloat(R5), asfloat(R6), asfloat(R7), asfloat(R8) };
    float3 t0 = float3(asfloat(T0), asfloat(T1), asfloat(T2));
    float fx = asfloat(Fx), fy = asfloat(Fy);

    float3 tc;
    tc.x = W[0] * p[0] + W[1] * p[1] + W[2] * p[2] + t0.x;
    tc.y = W[3] * p[0] + W[4] * p[1] + W[5] * p[2] + t0.y;
    tc.z = W[6] * p[0] + W[7] * p[1] + W[8] * p[2] + t0.z;
    if (tc.z < 1e-2) { Hide(i); return; }

    float opac = Sigmoid(p[10]);
    float minA = asfloat(MinA);
    if (opac < minA) { Hide(i); return; }

    float4 q = float4(p[6], p[7], p[8], p[9]);
    float len = length(q);
    q = (len < 1e-12) ? float4(1, 0, 0, 0) : q / len;
    float R[9];
    Rotation(q, R);
    float s[3] = { exp(p[3]), exp(p[4]), exp(p[5]) };

    float sig[9];
    [unroll] for (uint a = 0; a < 3; ++a)
        [unroll] for (uint b = 0; b < 3; ++b)
            sig[a * 3 + b] = R[a * 3 + 0] * s[0] * s[0] * R[b * 3 + 0] +
                             R[a * 3 + 1] * s[1] * s[1] * R[b * 3 + 1] +
                             R[a * 3 + 2] * s[2] * s[2] * R[b * 3 + 2];
    float WS[9], M[9];
    [unroll] for (uint a2 = 0; a2 < 3; ++a2)
        [unroll] for (uint b2 = 0; b2 < 3; ++b2)
            WS[a2 * 3 + b2] = W[a2 * 3] * sig[b2] + W[a2 * 3 + 1] * sig[3 + b2] +
                              W[a2 * 3 + 2] * sig[6 + b2];
    [unroll] for (uint a3 = 0; a3 < 3; ++a3)
        [unroll] for (uint b3 = 0; b3 < 3; ++b3)
            M[a3 * 3 + b3] = WS[a3 * 3] * W[b3 * 3] + WS[a3 * 3 + 1] * W[b3 * 3 + 1] +
                             WS[a3 * 3 + 2] * W[b3 * 3 + 2];

    float iz = 1.0 / tc.z, iz2 = iz * iz;
    float J[6] = { fx * iz, 0, -fx * tc.x * iz2, 0, fy * iz, -fy * tc.y * iz2 };
    float JM[6];
    [unroll] for (uint r = 0; r < 2; ++r)
        [unroll] for (uint c = 0; c < 3; ++c)
            JM[r * 3 + c] = J[r * 3] * M[c] + J[r * 3 + 1] * M[3 + c] + J[r * 3 + 2] * M[6 + c];
    float lp = asfloat(LowPass);
    float A = JM[0] * J[0] + JM[1] * J[1] + JM[2] * J[2] + lp;
    float B = JM[0] * J[3] + JM[1] * J[4] + JM[2] * J[5];
    float C = JM[3] * J[3] + JM[4] * J[4] + JM[5] * J[5] + lp;
    float det = A * C - B * B;
    if (!(det > 1e-12)) { Hide(i); return; }

    float u = fx * tc.x * iz + asfloat(Cx);
    float v = fy * tc.y * iz + asfloat(Cy);
    float mid = 0.5 * (A + C);
    float lmax = mid + sqrt(max(0.0, mid * mid - det));
    float reach2 = 2.0 * log(opac / minA);
    if (!(reach2 > 0.0)) { Hide(i); return; }
    float reach = sqrt(reach2 * lmax) + 1.0;
    if (u + reach < 0 || v + reach < 0 ||
        u - reach >= float(ImgW) || v - reach >= float(ImgH)) { Hide(i); return; }

    Proj[At(i * 3, TexW)]     = float4(u, v, opac, 3.0 * sqrt(lmax));
    Proj[At(i * 3 + 1, TexW)] = float4(C / det, -B / det, A / det, 0);
    Proj[At(i * 3 + 2, TexW)] = float4(p[11], p[12], p[13], tc.z);   // depth in .w
    Screen[At(i, TexW)]       = float4(u, v, tc.z, reach);
}
)";

// --- sum ---------------------------------------------------------------------------
//
// One thread per Gaussian, adding up its per-tile-entry gradients through the
// index list the CPU built while binning. The same sum the CPU rasteriser does
// serially; here there is no contention because each thread owns one
// Gaussian's total.
constexpr const char* kSum = R"(
Texture2D<float4>   EGrad : register(t0);   // per entry: 3 texels
Texture2D<float>    GMap  : register(t1);   // N+1 starts into GList
Texture2D<float>    GList : register(t2);   // entry indices
RWTexture2D<float4> G2    : register(u0);   // per Gaussian: 3 texels

cbuffer Params : register(b0) { uint D0; uint D1; uint N; uint TexW; };

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    if (i >= N) return;
    uint start = uint(GMap[At(i, TexW)]), end = uint(GMap[At(i + 1, TexW)]);
    float4 a = 0, b = 0, c = 0;
    for (uint k = start; k < end; ++k) {
        uint e = uint(GList[At(k, TexW)]);
        a += EGrad[At(e * 3, TexW)];
        b += EGrad[At(e * 3 + 1, TexW)];
        c += EGrad[At(e * 3 + 2, TexW)];
    }
    G2[At(i * 3, TexW)] = a;
    G2[At(i * 3 + 1, TexW)] = b;
    G2[At(i * 3 + 2, TexW)] = c;
}
)";

// --- update: the chain rule, then Adam ------------------------------------------
//
// One thread per Gaussian: the CPU rasteriser's per-Gaussian backward pass,
// ported line for line in float, followed by the Adam step train_splats takes
// -- and the densification counters. Reads the old state, writes the new.
constexpr const char* kUpdate = R"(
Texture2D<float4>   Old  : register(t0);
Texture2D<float4>   G2   : register(t1);
Texture2D<float4>   Proj : register(t2);
RWTexture2D<float4> New  : register(u0);

cbuffer Params : register(b0) {
    uint D0; uint D1;
    uint N; uint TexW;
    uint R0; uint R1; uint R2; uint R3; uint R4; uint R5; uint R6; uint R7; uint R8;
    uint T0; uint T1; uint T2;
    uint Fx; uint Fy;
    uint LowPass; uint LrMean; uint C1; uint C2; uint ToNdc; uint ClampColour;
};

void Store14(uint base, float p[14]) {
    New[At(base, TexW)]     = float4(p[0], p[1], p[2], p[3]);
    New[At(base + 1, TexW)] = float4(p[4], p[5], p[6], p[7]);
    New[At(base + 2, TexW)] = float4(p[8], p[9], p[10], p[11]);
    New[At(base + 3, TexW)] = float4(p[12], p[13], 0, 0);
}

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    if (i >= N) return;
    uint base = i * 13;

    float p[14], m1[14], m2[14];
    Load14(Old, base, TexW, p);
    Load14(Old, base + 4, TexW, m1);
    Load14(Old, base + 8, TexW, m2);
    float4 stats = Old[At(base + 12, TexW)];

    float g[14];
    [unroll] for (uint z = 0; z < 14; ++z) g[z] = 0.0;

    float4 pr = Proj[At(i * 3, TexW)];
    if (pr.w >= 0.0) {
        float4 ga = G2[At(i * 3, TexW)], gb = G2[At(i * 3 + 1, TexW)],
               gc = G2[At(i * 3 + 2, TexW)];
        float gu = ga.x, gv = ga.y;
        float gq0 = ga.z, gq1 = ga.w, gq2 = gb.x;
        g[11] = gb.y; g[12] = gb.z; g[13] = gb.w;   // colour
        g[10] = gc.x;                                // opacity logit
        float gz = gc.y;                             // the depth it contributes

        float W[9] = { asfloat(R0), asfloat(R1), asfloat(R2), asfloat(R3), asfloat(R4),
                       asfloat(R5), asfloat(R6), asfloat(R7), asfloat(R8) };
        float3 t0 = float3(asfloat(T0), asfloat(T1), asfloat(T2));
        float fx = asfloat(Fx), fy = asfloat(Fy);

        // Rebuild what projection worked out: cheaper than storing it.
        float3 tc;
        tc.x = W[0] * p[0] + W[1] * p[1] + W[2] * p[2] + t0.x;
        tc.y = W[3] * p[0] + W[4] * p[1] + W[5] * p[2] + t0.y;
        tc.z = W[6] * p[0] + W[7] * p[1] + W[8] * p[2] + t0.z;
        float4 qr = float4(p[6], p[7], p[8], p[9]);
        float qlen = length(qr);
        float4 qn = (qlen < 1e-12) ? float4(1, 0, 0, 0) : qr / qlen;
        if (qlen < 1e-12) qlen = 1.0;
        float R[9];
        Rotation(qn, R);
        float s[3] = { exp(p[3]), exp(p[4]), exp(p[5]) };
        float sig[9];
        [unroll] for (uint a = 0; a < 3; ++a)
            [unroll] for (uint b = 0; b < 3; ++b)
                sig[a * 3 + b] = R[a * 3] * s[0] * s[0] * R[b * 3] +
                                 R[a * 3 + 1] * s[1] * s[1] * R[b * 3 + 1] +
                                 R[a * 3 + 2] * s[2] * s[2] * R[b * 3 + 2];
        float WS[9], M[9];
        [unroll] for (uint a2 = 0; a2 < 3; ++a2)
            [unroll] for (uint b2 = 0; b2 < 3; ++b2)
                WS[a2 * 3 + b2] = W[a2 * 3] * sig[b2] + W[a2 * 3 + 1] * sig[3 + b2] +
                                  W[a2 * 3 + 2] * sig[6 + b2];
        [unroll] for (uint a3 = 0; a3 < 3; ++a3)
            [unroll] for (uint b3 = 0; b3 < 3; ++b3)
                M[a3 * 3 + b3] = WS[a3 * 3] * W[b3 * 3] + WS[a3 * 3 + 1] * W[b3 * 3 + 1] +
                                 WS[a3 * 3 + 2] * W[b3 * 3 + 2];
        float iz = 1.0 / tc.z, iz2 = iz * iz, iz3 = iz2 * iz;
        float J[6] = { fx * iz, 0, -fx * tc.x * iz2, 0, fy * iz, -fy * tc.y * iz2 };
        float JM[6];
        [unroll] for (uint r = 0; r < 2; ++r)
            [unroll] for (uint c = 0; c < 3; ++c)
                JM[r * 3 + c] = J[r * 3] * M[c] + J[r * 3 + 1] * M[3 + c] + J[r * 3 + 2] * M[6 + c];
        float lp = asfloat(LowPass);
        float A = JM[0] * J[0] + JM[1] * J[1] + JM[2] * J[2] + lp;
        float B = JM[0] * J[3] + JM[1] * J[4] + JM[2] * J[5];
        float C = JM[3] * J[3] + JM[4] * J[4] + JM[5] * J[5] + lp;
        float det = A * C - B * B;
        float Q[4] = { C / det, -B / det, -B / det, A / det };

        // conic -> cov2: G_cov = -Q G_Q Q, with G_Q's off-diagonal halved.
        float GQ[4] = { gq0, 0.5 * gq1, 0.5 * gq1, gq2 };
        float QG[4], GS2[4];
        [unroll] for (uint r1 = 0; r1 < 2; ++r1)
            [unroll] for (uint c1 = 0; c1 < 2; ++c1)
                QG[r1 * 2 + c1] = Q[r1 * 2] * GQ[c1] + Q[r1 * 2 + 1] * GQ[2 + c1];
        [unroll] for (uint r2 = 0; r2 < 2; ++r2)
            [unroll] for (uint c2 = 0; c2 < 2; ++c2)
                GS2[r2 * 2 + c2] = -(QG[r2 * 2] * Q[c2] + QG[r2 * 2 + 1] * Q[2 + c2]);

        // cov2 = J M J^T: G_M = J^T G J, G_J = 2 G J M.
        float GM[9];
        [unroll] for (uint a4 = 0; a4 < 3; ++a4)
            [unroll] for (uint b4 = 0; b4 < 3; ++b4) {
                float acc = 0.0;
                [unroll] for (uint r3 = 0; r3 < 2; ++r3)
                    [unroll] for (uint c3 = 0; c3 < 2; ++c3)
                        acc += J[r3 * 3 + a4] * GS2[r3 * 2 + c3] * J[c3 * 3 + b4];
                GM[a4 * 3 + b4] = acc;
            }
        float GJ[6];
        [unroll] for (uint r4 = 0; r4 < 2; ++r4)
            [unroll] for (uint c4 = 0; c4 < 3; ++c4) {
                float acc2 = 0.0;
                [unroll] for (uint s4 = 0; s4 < 2; ++s4)
                    [unroll] for (uint k4 = 0; k4 < 3; ++k4)
                        acc2 += GS2[r4 * 2 + s4] * J[s4 * 3 + k4] * M[k4 * 3 + c4];
                GJ[r4 * 3 + c4] = 2.0 * acc2;
            }

        // M = W Sigma W^T -> G_Sigma = W^T G_M W.
        float WtG[9], GSig[9];
        [unroll] for (uint a5 = 0; a5 < 3; ++a5)
            [unroll] for (uint b5 = 0; b5 < 3; ++b5)
                WtG[a5 * 3 + b5] = W[a5] * GM[b5] + W[3 + a5] * GM[3 + b5] + W[6 + a5] * GM[6 + b5];
        [unroll] for (uint a6 = 0; a6 < 3; ++a6)
            [unroll] for (uint b6 = 0; b6 < 3; ++b6)
                GSig[a6 * 3 + b6] = WtG[a6 * 3] * W[b6] + WtG[a6 * 3 + 1] * W[3 + b6] +
                                    WtG[a6 * 3 + 2] * W[6 + b6];

        // Sigma = (R S)(R S)^T -> scale and rotation.
        float GR[9];
        [unroll] for (uint k7 = 0; k7 < 3; ++k7) {
            float gs = 0.0;
            [unroll] for (uint a7 = 0; a7 < 3; ++a7) {
                float gm3 = 2.0 * (GSig[a7 * 3] * R[k7] * s[k7] +
                                   GSig[a7 * 3 + 1] * R[3 + k7] * s[k7] +
                                   GSig[a7 * 3 + 2] * R[6 + k7] * s[k7]);
                gs += gm3 * R[a7 * 3 + k7];
                GR[a7 * 3 + k7] = gm3 * s[k7];
            }
            g[3 + k7] = gs * s[k7];
        }

        // Rotation matrix -> normalised quaternion -> raw quaternion.
        float w = qn.x, x = qn.y, y = qn.z, z2 = qn.w;
        float4 gqv;
        gqv.x = GR[1] * (-2 * z2) + GR[2] * (2 * y) + GR[3] * (2 * z2) +
                GR[5] * (-2 * x) + GR[6] * (-2 * y) + GR[7] * (2 * x);
        gqv.y = GR[1] * (2 * y) + GR[2] * (2 * z2) + GR[3] * (2 * y) + GR[4] * (-4 * x) +
                GR[5] * (-2 * w) + GR[6] * (2 * z2) + GR[7] * (2 * w) + GR[8] * (-4 * x);
        gqv.z = GR[0] * (-4 * y) + GR[1] * (2 * x) + GR[2] * (2 * w) + GR[3] * (2 * x) +
                GR[5] * (2 * z2) + GR[6] * (-2 * w) + GR[7] * (2 * z2) + GR[8] * (-4 * y);
        gqv.w = GR[0] * (-4 * z2) + GR[1] * (-2 * w) + GR[2] * (2 * x) + GR[3] * (2 * w) +
                GR[4] * (-4 * z2) + GR[5] * (2 * y) + GR[6] * (2 * x) + GR[7] * (2 * y);
        float4 graw = (gqv - qn * dot(gqv, qn)) / qlen;
        g[6] = graw.x; g[7] = graw.y; g[8] = graw.z; g[9] = graw.w;

        // The mean, through the centre and through J.
        float3 gt = float3(0, 0, gz);   // the depth a Gaussian contributes IS tc.z
        gt.x += gu * fx * iz;
        gt.z += gu * (-fx * tc.x * iz2);
        gt.y += gv * fy * iz;
        gt.z += gv * (-fy * tc.y * iz2);
        gt.z += GJ[0] * (-fx * iz2) + GJ[2] * (2.0 * fx * tc.x * iz3) +
                GJ[4] * (-fy * iz2) + GJ[5] * (2.0 * fy * tc.y * iz3);
        gt.x += GJ[2] * (-fx * iz2);
        gt.y += GJ[5] * (-fy * iz2);
        g[0] = W[0] * gt.x + W[3] * gt.y + W[6] * gt.z;
        g[1] = W[1] * gt.x + W[4] * gt.y + W[7] * gt.z;
        g[2] = W[2] * gt.x + W[5] * gt.y + W[8] * gt.z;

        // Densification counters, in the paper's units (see train_splats).
        stats.x += length(float2(gu, gv)) * asfloat(ToNdc);
        stats.y += 1.0;
        stats.z = max(stats.z, pr.w);
    }

    // Adam, with the paper's per-group rates. Every Gaussian steps, in view
    // or not, exactly as the CPU loop does -- momentum carries on.
    const float b1 = 0.9, b2 = 0.999, eps = 1e-15;
    float c1 = asfloat(C1), c2 = asfloat(C2), lrMean = asfloat(LrMean);
    [unroll] for (uint q = 0; q < 14; ++q) {
        float lr = (q < 3) ? lrMean : (q < 6) ? 5e-3 : (q < 10) ? 1e-3
                 : (q < 11) ? 5e-2 : 2.5e-3;
        m1[q] = b1 * m1[q] + (1.0 - b1) * g[q];
        m2[q] = b2 * m2[q] + (1.0 - b2) * g[q] * g[q];
        p[q] -= lr * (m1[q] / c1) / (sqrt(m2[q] / c2) + eps);
    }
    // Colour kept displayable; see train_splats' clamp_colour.
    if (ClampColour != 0)
        [unroll] for (uint k = 11; k < 14; ++k) p[k] = saturate(p[k]);

    Store14(base, p);
    Store14(base + 4, m1);
    Store14(base + 8, m2);
    New[At(base + 12, TexW)] = stats;
}
)";

}  // namespace

struct SplatTrainerGpu::Impl {
    ComputeContext* ctx = nullptr;
    ComputeKernel   project, sum, update, fwd, bwd;
    bool            ready = false;

    GpuImage stateA, stateB;            // ping-pong: A holds the current state
    GpuImage proj, screen, list, offs, gmap, glist, rgbt, last, drgbt, egrad, g2;
    GpuImage depth, ddepth, dshift; // expected depth, dLoss/d(depth), target
    bool     ddepthZero = false;    // ddepth holds zeros: no re-upload needed
    std::vector<float> depthStage, ddStage, tStage, shiftStage;
    size_t   n = 0;
    std::vector<float> stage;

    // Binning scratch, kept between iterations so a step does not allocate
    // and zero tens of megabytes it is about to overwrite.
    std::vector<TileRect> rect;
    std::vector<uint32_t> chunkBase, tileStart, gStart, key, idx, tmpKey, tmpIdx;
    std::vector<float>    listStage, glistStage, gmapStage, offsStage;

    bool Ensure(GpuImage& img, const ImageDesc& d) {
        if (img.Valid() && img.desc == d) return true;
        img.Release();
        return ctx->CreateImage(d, &img);
    }
    bool Put(GpuImage& img, const ImageDesc& d) { return Put(img, d, stage); }
    bool Put(GpuImage& img, const ImageDesc& d, std::vector<float>& from) {
        ImageView v;
        v.desc = d;
        v.data = reinterpret_cast<uint8_t*>(from.data());
        return ctx->Upload(v, &img);
    }
    bool Get(const GpuImage& img, size_t floats) {
        stage.assign(floats, 0.0f);
        ImageView v;
        v.desc = img.desc;
        v.data = reinterpret_cast<uint8_t*>(stage.data());
        return ctx->Readback(img, &v);
    }
    // One dispatch, submitted on its own.
    bool Run(const ComputeKernel& k, const std::vector<const GpuImage*>& in,
             const std::vector<GpuImage*>& out,
             const std::vector<uint32_t>& c, uint32_t gx, uint32_t gy,
             std::string* err) {
        return ctx->Dispatch(k, in, out, c, err, gx, gy) && ctx->Flush(err);
    }
};

SplatTrainerGpu::SplatTrainerGpu(ComputeContext* gpu) : m(std::make_unique<Impl>()) {
    m->ctx = gpu;
}
SplatTrainerGpu::~SplatTrainerGpu() = default;

bool SplatTrainerGpu::Upload(const std::vector<SplatParam>& params,
                             const std::vector<double>& m1,
                             const std::vector<double>& m2,
                             const std::vector<double>& gradAccum,
                             const std::vector<int>& gradCount,
                             const std::vector<double>& maxScreen,
                             std::string* err) {
    Impl& s = *m;
    if (!s.ctx || !s.ctx->Ready()) { *err = "no device"; return false; }
    std::lock_guard<std::mutex> lock(s.ctx->SubmitMutex());

    if (!s.ready) {
        if (!s.ctx->CreateKernel(std::string(kCommon) + kProject, "main",
                                 "splat_project", &s.project, err) ||
            !s.ctx->CreateKernel(std::string(kCommon) + kSum, "main",
                                 "splat_sum", &s.sum, err) ||
            !s.ctx->CreateKernel(std::string(kCommon) + kUpdate, "main",
                                 "splat_update", &s.update, err) ||
            !s.ctx->CreateKernel(splat_kernels::kForward, "main", "splat_forward",
                                 &s.fwd, err) ||
            !s.ctx->CreateKernel(splat_kernels::kBackward, "main", "splat_backward",
                                 &s.bwd, err))
            return false;
        s.ready = true;
    }

    s.n = params.size();
    const ImageDesc d = FlatDesc(std::max<size_t>(1, s.n * kStateTexels), Format::RGBA32F);
    s.stage.assign(size_t(d.width) * size_t(d.height) * 4, 0.0f);
    for (size_t i = 0; i < s.n; ++i) {
        float* t = &s.stage[i * kStateTexels * 4];
        const double* p = params[i].Data();
        for (int q = 0; q < 14; ++q) {
            t[q]      = float(p[q]);
            t[16 + q] = float(m1[i * 14 + size_t(q)]);
            t[32 + q] = float(m2[i * 14 + size_t(q)]);
        }
        t[48] = float(gradAccum[i]);
        t[49] = float(gradCount[i]);
        t[50] = float(maxScreen[i]);
    }
    if (!s.Ensure(s.stateA, d) || !s.Ensure(s.stateB, d) || !s.Put(s.stateA, d)) {
        *err = "could not upload the training state";
        return false;
    }
    return true;
}

bool SplatTrainerGpu::Download(std::vector<SplatParam>* params,
                               std::vector<double>* m1, std::vector<double>* m2,
                               std::vector<double>* gradAccum,
                               std::vector<int>* gradCount,
                               std::vector<double>* maxScreen, std::string* err) {
    Impl& s = *m;
    std::lock_guard<std::mutex> lock(s.ctx->SubmitMutex());
    const ImageDesc& d = s.stateA.desc;
    if (!s.Get(s.stateA, size_t(d.width) * size_t(d.height) * 4)) {
        *err = "could not read back the training state";
        return false;
    }
    params->assign(s.n, SplatParam::Zero());
    m1->assign(s.n * 14, 0.0);
    m2->assign(s.n * 14, 0.0);
    gradAccum->assign(s.n, 0.0);
    gradCount->assign(s.n, 0);
    maxScreen->assign(s.n, 0.0);
    for (size_t i = 0; i < s.n; ++i) {
        const float* t = &s.stage[i * kStateTexels * 4];
        double* p = (*params)[i].Data();
        for (int q = 0; q < 14; ++q) {
            p[q] = double(t[q]);
            (*m1)[i * 14 + size_t(q)] = double(t[16 + q]);
            (*m2)[i * 14 + size_t(q)] = double(t[32 + q]);
        }
        (*gradAccum)[i] = double(t[48]);
        (*gradCount)[i] = int(t[49] + 0.5f);
        (*maxScreen)[i] = double(t[50]);
    }
    return true;
}

bool SplatTrainerGpu::Step(const SplatCam& cam, const RasterOptions& opt,
                           const std::vector<double>& target, double lrMean,
                           double c1, double c2, std::string* err,
                           const std::vector<float>* depthTarget,
                           double depthWeight) {
    Impl& s = *m;
    if (!s.ready || s.n == 0) { *err = "trainer not started"; return false; }
    std::lock_guard<std::mutex> lock(s.ctx->SubmitMutex());

    const size_t n = s.n;
    const uint32_t groupsN = uint32_t((n + 255) / 256);
    const double* W = cam.R.m;
    std::vector<uint32_t> camC;
    for (int k = 0; k < 9; ++k) camC.push_back(Bits(float(W[k])));
    camC.push_back(Bits(float(cam.t.x)));
    camC.push_back(Bits(float(cam.t.y)));
    camC.push_back(Bits(float(cam.t.z)));

    // --- project ------------------------------------------------------------
    auto clk = Clock::now();
    const ImageDesc projD = FlatDesc(std::max<size_t>(1, n * 3), Format::RGBA32F);
    const ImageDesc scrD  = FlatDesc(std::max<size_t>(1, n), Format::RGBA32F);
    if (!s.Ensure(s.proj, projD) || !s.Ensure(s.screen, scrD)) {
        *err = "could not allocate projection textures";
        return false;
    }
    {
        std::vector<uint32_t> c{uint32_t(n), uint32_t(kTexW)};
        c.insert(c.end(), camC.begin(), camC.end());
        c.push_back(Bits(float(cam.fx)));
        c.push_back(Bits(float(cam.fy)));
        c.push_back(Bits(float(cam.cx)));
        c.push_back(Bits(float(cam.cy)));
        c.push_back(uint32_t(cam.w));
        c.push_back(uint32_t(cam.h));
        c.push_back(Bits(float(opt.lowPass)));
        c.push_back(Bits(float(opt.minAlpha)));
        if (!s.Run(s.project, {&s.stateA}, {&s.proj, &s.screen}, c, groupsN, 1, err))
            return false;
    }
    m_time.project += MsSince(clk);

    // --- bin and sort, on the CPU, from the screen texels --------------------
    clk = Clock::now();
    if (!s.Get(s.screen, size_t(scrD.width) * size_t(scrD.height) * 4)) {
        *err = "could not read back screen positions";
        return false;
    }
    const int tilesX = (cam.w + kTile - 1) / kTile;
    const int tilesY = (cam.h + kTile - 1) / kTile;
    const size_t nTiles = size_t(tilesX) * size_t(tilesY);
    const float* scr = s.stage.data();

    // Chunks are contiguous index ranges, counted and filled independently.
    // Each chunk writes its entries into a tile after every earlier chunk's,
    // so each tile's entries still arrive in index order -- exactly what one
    // thread would produce, whatever the thread count.
    const size_t nChunks = std::clamp<size_t>(n / 4096, 1, 64);
    const size_t chunk   = (n + nChunks - 1) / nChunks;
    s.rect.resize(n);
    s.gStart.resize(n + 1);
    s.chunkBase.assign(nChunks * nTiles, 0);
    std::vector<int>      chunkVisible(nChunks, 0);
    std::vector<uint32_t> chunkArea(nChunks, 0);   // entries each chunk makes
    ParallelFor(nChunks, [&](size_t c) {
        uint32_t* cnt = &s.chunkBase[c * nTiles];
        const size_t end = std::min(n, (c + 1) * chunk);
        int vis = 0;        // locals: neighbouring chunks' counters
        uint32_t area = 0;  // would share a cache line
        for (size_t i = c * chunk; i < end; ++i) {
            TileRect& R = s.rect[i];
            const float* sc = scr + i * 4;
            if (sc[3] < 0.0f) {
                R = TileRect{0, 0, -1, -1};
                s.gStart[i + 1] = 0;
                continue;
            }
            ++vis;
            const double u = sc[0], v = sc[1], r = sc[3];
            R.x0 = std::clamp(int(std::floor((u - r) / kTile)), 0, tilesX - 1);
            R.x1 = std::clamp(int(std::floor((u + r) / kTile)), 0, tilesX - 1);
            R.y0 = std::clamp(int(std::floor((v - r) / kTile)), 0, tilesY - 1);
            R.y1 = std::clamp(int(std::floor((v + r) / kTile)), 0, tilesY - 1);
            s.gStart[i + 1] = uint32_t((R.x1 - R.x0 + 1) * (R.y1 - R.y0 + 1));
            area += s.gStart[i + 1];
            for (int ty = R.y0; ty <= R.y1; ++ty)
                for (int tx = R.x0; tx <= R.x1; ++tx)
                    ++cnt[size_t(ty) * size_t(tilesX) + size_t(tx)];
        }
        chunkVisible[c] = vis;
        chunkArea[c]    = area;
    });
    int visible = 0;
    for (int v : chunkVisible) visible += v;
    m_visible = visible;

    // Counts become write positions: tile by tile, chunk by chunk.
    s.tileStart.assign(nTiles + 1, 0);
    for (size_t t = 0; t < nTiles; ++t) {
        uint32_t run = s.tileStart[t];
        for (size_t c = 0; c < nChunks; ++c) {
            uint32_t& k = s.chunkBase[c * nTiles + t];
            const uint32_t count = k;
            k = run;
            run += count;
        }
        s.tileStart[t + 1] = run;
    }
    const size_t entries = s.tileStart[nTiles];

    // Where each Gaussian's own entries start in the per-Gaussian list: its
    // k-th is its k-th tile in row-major order. Written straight into the
    // upload as N+1 starts; the sum kernel takes each count as a difference.
    const ImageDesc listD = FlatDesc(std::max<size_t>(1, entries), Format::R32F);
    const ImageDesc offsD = FlatDesc(std::max<size_t>(1, nTiles), Format::RGBA32F);
    const ImageDesc gmapD = FlatDesc(n + 1, Format::R32F);
    s.gmapStage.resize(size_t(gmapD.width) * size_t(gmapD.height));
    s.gStart[0] = 0;
    s.gmapStage[0] = 0.0f;
    std::vector<uint32_t> chunkStart(nChunks, 0);
    for (size_t c = 1; c < nChunks; ++c) chunkStart[c] = chunkStart[c - 1] + chunkArea[c - 1];

    s.key.resize(entries);
    s.idx.resize(entries);
    s.tmpKey.resize(entries);
    s.tmpIdx.resize(entries);
    ParallelFor(nChunks, [&](size_t c) {
        uint32_t* pos = &s.chunkBase[c * nTiles];
        const size_t end = std::min(n, (c + 1) * chunk);

        // The running sum, in place: slot i+1 holds Gaussian i's entry count
        // and becomes where its entries end, i.e. where i+1's start.
        uint32_t run = chunkStart[c];
        for (size_t i = c * chunk; i < end; ++i) {
            run += s.gStart[i + 1];
            s.gStart[i + 1]    = run;
            s.gmapStage[i + 1] = float(run);
        }

        for (size_t i = c * chunk; i < end; ++i) {
            const TileRect& R = s.rect[i];
            if (R.x1 < R.x0) continue;
            const uint32_t k = DepthKey(scr[i * 4 + 2]);
            for (int ty = R.y0; ty <= R.y1; ++ty)
                for (int tx = R.x0; tx <= R.x1; ++tx) {
                    const uint32_t e = pos[size_t(ty) * size_t(tilesX) + size_t(tx)]++;
                    s.key[e] = k;
                    s.idx[e] = uint32_t(i);
                }
        }
    });

    // Near to far within each tile, then both lists straight from the
    // sorted tile. Every write lands in a slot no other tile touches.
    s.listStage.resize(size_t(listD.width) * size_t(listD.height));
    s.glistStage.resize(s.listStage.size());
    ParallelFor(nTiles, [&](size_t t) {
        const size_t b = s.tileStart[t], m = s.tileStart[t + 1] - b;
        RadixSortTile(&s.key[b], &s.idx[b], &s.tmpKey[b], &s.tmpIdx[b], m);
        const int tx = int(t % size_t(tilesX)), ty = int(t / size_t(tilesX));
        for (size_t e = b; e < b + m; ++e) {
            const uint32_t i = s.idx[e];
            const TileRect& R = s.rect[i];
            const size_t k = size_t(ty - R.y0) * size_t(R.x1 - R.x0 + 1) + size_t(tx - R.x0);
            s.listStage[e] = float(i);
            s.glistStage[s.gStart[i] + k] = float(e);
        }
    });

    s.offsStage.assign(size_t(offsD.width) * size_t(offsD.height) * 4, 0.0f);
    for (size_t t = 0; t < nTiles; ++t) {
        s.offsStage[t * 4 + 0] = float(s.tileStart[t]);
        s.offsStage[t * 4 + 1] = float(s.tileStart[t + 1] - s.tileStart[t]);
    }
    const bool ok = s.Ensure(s.list, listD) && s.Ensure(s.offs, offsD) &&
                    s.Ensure(s.gmap, gmapD) && s.Ensure(s.glist, listD) &&
                    s.Put(s.list, listD, s.listStage) &&
                    s.Put(s.offs, offsD, s.offsStage) &&
                    s.Put(s.gmap, gmapD, s.gmapStage) &&
                    s.Put(s.glist, listD, s.glistStage);
    if (!ok) { *err = "could not upload the tile lists"; return false; }
    m_time.bin += MsSince(clk);

    // --- composite ------------------------------------------------------------
    clk = Clock::now();
    const ImageDesc rgbtD{cam.w, cam.h, Format::RGBA32F};
    const ImageDesc lastD{cam.w, cam.h, Format::R32F};
    if (!s.Ensure(s.rgbt, rgbtD) || !s.Ensure(s.last, lastD)) {
        *err = "could not allocate the render targets";
        return false;
    }
    if (!s.depth.Valid() || !(s.depth.desc == lastD)) {
        s.ddepthZero = false;           // reallocated below: contents unknown
        if (!s.Ensure(s.depth, lastD) || !s.Ensure(s.ddepth, lastD) ||
            !s.Ensure(s.dshift, lastD)) {
            *err = "could not allocate the depth targets";
            return false;
        }
    }
    const std::vector<uint32_t> bgC{
        Bits(float(opt.background.x)), Bits(float(opt.background.y)),
        Bits(float(opt.background.z)), Bits(float(opt.minAlpha)),
        Bits(float(opt.maxAlpha))};
    {
        std::vector<uint32_t> c{uint32_t(tilesX), uint32_t(kTexW)};
        c.insert(c.end(), bgC.begin(), bgC.end());
        c.push_back(Bits(float(opt.tStop)));
        if (!s.Run(s.fwd, {&s.proj, &s.list, &s.offs}, {&s.rgbt, &s.last, &s.depth},
                   c, uint32_t(tilesX), uint32_t(tilesY), err))
            return false;
    }
    const size_t np = size_t(cam.w) * size_t(cam.h);
    const bool useDepth = depthTarget && depthWeight > 0.0;
    if (useDepth) {
        // Read first: Get() reuses the stage the colour lands in next.
        if (!s.Get(s.depth, np)) { *err = "could not read back the depth"; return false; }
        s.depthStage.swap(s.stage);
    }
    if (!s.Get(s.rgbt, np * 4)) { *err = "could not read back the render"; return false; }
    m_time.composite += MsSince(clk);

    // --- loss: L1, as train_splats's CPU loop ------------------------------------
    clk = Clock::now();
    {
        // A negative target is a masked pixel (train_splats' MaskUnmeasured):
        // no gradient, and not counted in the mean.
        size_t used = 0;
        for (size_t i = 0; i < np * 3; ++i) if (target[i] >= 0.0) ++used;
        const float invN = float(1.0 / double(std::max<size_t>(1, used)));
        std::vector<float> d(np * 4);
        for (size_t i = 0; i < np; ++i) {
            for (int ch = 0; ch < 3; ++ch) {
                if (target[i * 3 + size_t(ch)] < 0.0) { d[i * 4 + size_t(ch)] = 0.0f; continue; }
                const double diff = double(s.stage[i * 4 + size_t(ch)]) - target[i * 3 + size_t(ch)];
                d[i * 4 + size_t(ch)] = diff > 0.0 ? invN : (diff < 0.0 ? -invN : 0.0f);
            }
            d[i * 4 + 3] = s.stage[i * 4 + 3];   // final T
        }
        s.stage.swap(d);
        if (!s.Ensure(s.drgbt, rgbtD) || !s.Put(s.drgbt, rgbtD)) {
            *err = "could not upload the loss gradient";
            return false;
        }

        // The depth term, or zeros -- uploaded once and left there while no
        // step has a depth target, since the kernel always reads it.
        if (useDepth) {
            s.depthStage.resize(np);
            s.tStage.resize(np);
            for (size_t i = 0; i < np; ++i) s.tStage[i] = s.stage[i * 4 + 3];
            DepthLossGrad(s.depthStage, s.tStage, *depthTarget, depthWeight,
                          &s.ddStage, &m_depthErr, &m_depthCount);
            // The target doubles as the per-pixel shift; see DepthLossGrad.
            s.shiftStage.assign(depthTarget->begin(), depthTarget->end());
            s.shiftStage.resize(np, 0.0f);
            if (!s.Put(s.ddepth, lastD, s.ddStage) ||
                !s.Put(s.dshift, lastD, s.shiftStage)) {
                *err = "could not upload the depth gradient";
                return false;
            }
            s.ddepthZero = false;
        } else if (!s.ddepthZero) {
            s.ddStage.assign(np, 0.0f);
            if (!s.Put(s.ddepth, lastD, s.ddStage) ||
                !s.Put(s.dshift, lastD, s.ddStage)) {   // zeros for both
                *err = "could not upload the depth gradient";
                return false;
            }
            s.ddepthZero = true;
        }
    }
    m_time.loss += MsSince(clk);

    // --- backward: per pixel, then summed per Gaussian ---------------------------
    clk = Clock::now();
    const ImageDesc eD  = FlatDesc(std::max<size_t>(1, entries * 3), Format::RGBA32F);
    const ImageDesc g2D = FlatDesc(std::max<size_t>(1, n * 3), Format::RGBA32F);
    if (!s.Ensure(s.egrad, eD) || !s.Ensure(s.g2, g2D)) {
        *err = "could not allocate the gradient textures";
        return false;
    }
    {
        std::vector<uint32_t> c{uint32_t(cam.w), uint32_t(cam.h), uint32_t(tilesX),
                                uint32_t(kTexW)};
        c.insert(c.end(), bgC.begin(), bgC.end());
        if (!s.Run(s.bwd, {&s.proj, &s.list, &s.offs, &s.drgbt},
                   {&s.egrad, &s.last, &s.ddepth, &s.dshift},
                   c, uint32_t(tilesX), uint32_t(tilesY), err))
            return false;
    }
    if (!s.Run(s.sum, {&s.egrad, &s.gmap, &s.glist}, {&s.g2},
               {uint32_t(n), uint32_t(kTexW)}, groupsN, 1, err))
        return false;
    m_time.backward += MsSince(clk);

    // --- the chain rule and Adam ------------------------------------------------
    clk = Clock::now();
    {
        std::vector<uint32_t> c{uint32_t(n), uint32_t(kTexW)};
        c.insert(c.end(), camC.begin(), camC.end());
        c.push_back(Bits(float(cam.fx)));
        c.push_back(Bits(float(cam.fy)));
        c.push_back(Bits(float(opt.lowPass)));
        c.push_back(Bits(float(lrMean)));
        c.push_back(Bits(float(c1)));
        c.push_back(Bits(float(c2)));
        c.push_back(Bits(float(0.5 * double(cam.w))));
        c.push_back(m_clampColour ? 1u : 0u);
        if (!s.Run(s.update, {&s.stateA, &s.g2, &s.proj}, {&s.stateB}, c,
                   groupsN, 1, err))
            return false;
    }
    std::swap(s.stateA, s.stateB);
    m_time.update += MsSince(clk);
    return true;
}

}  // namespace tglab
