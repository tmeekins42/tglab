#include "splat_train_gpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <mutex>

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

// The real SH basis beyond Y_0, as splat_sh.h's ShBasis: the first
// (deg+1)^2 - 1 entries are written, the rest left zero.
void ShBasis(float3 d, uint deg, out float Y[15]) {
    [unroll] for (uint k = 0; k < 15; ++k) Y[k] = 0.0;
    if (deg < 1) return;
    float x = d.x, y = d.y, z = d.z;
    Y[0] = -0.4886025119029199 * y;
    Y[1] =  0.4886025119029199 * z;
    Y[2] = -0.4886025119029199 * x;
    if (deg < 2) return;
    float xx = x * x, yy = y * y, zz = z * z;
    Y[3] =  1.0925484305920792 * x * y;
    Y[4] = -1.0925484305920792 * y * z;
    Y[5] =  0.31539156525252005 * (2.0 * zz - xx - yy);
    Y[6] = -1.0925484305920792 * x * z;
    Y[7] =  0.5462742152960396 * (xx - yy);
    if (deg < 3) return;
    Y[8]  = -0.5900435899266435 * y * (3.0 * xx - yy);
    Y[9]  =  2.890611442640554 * x * y * z;
    Y[10] = -0.4570457994644658 * y * (4.0 * zz - xx - yy);
    Y[11] =  0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy);
    Y[12] = -0.4570457994644658 * x * (4.0 * zz - xx - yy);
    Y[13] =  1.445305721320277 * z * (xx - yy);
    Y[14] = -0.5900435899266435 * x * (xx - 3.0 * yy);
}

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
Texture2D<float4>   Sh     : register(t1);   // 12 texels per Gaussian, when ShDeg > 0
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
    uint EyeX; uint EyeY; uint EyeZ; uint ShDeg;
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

    // The colour this camera sees: base plus the spherical-harmonic terms
    // for the direction from the camera to the Gaussian, clamped as base
    // colours are. At degree 0 the base colour goes through untouched, so
    // plain training is exactly what it was.
    float3 col = float3(p[11], p[12], p[13]);
    if (ShDeg > 0) {
        float3 dir = float3(p[0], p[1], p[2]) -
                     float3(asfloat(EyeX), asfloat(EyeY), asfloat(EyeZ));
        float dl = length(dir);
        if (dl > 1e-12) {
            float Y[15];
            ShBasis(dir / dl, ShDeg, Y);
            [unroll] for (uint k2 = 0; k2 < 12; ++k2) {
                float4 cf = Sh[At(i * 12 + k2, TexW)];
                // Texel k2 holds floats 4 k2 .. 4 k2 + 3 of rest[k * 3 + ch].
                float f4[4] = { cf.x, cf.y, cf.z, cf.w };
                [unroll] for (uint j = 0; j < 4; ++j) {
                    uint q = k2 * 4 + j;
                    if (q < 45) col[q % 3] += Y[q / 3] * f4[j];
                }
            }
        }
        col = saturate(col);
    }

    Proj[At(i * 3, TexW)]     = float4(u, v, opac, 3.0 * sqrt(lmax));
    Proj[At(i * 3 + 1, TexW)] = float4(C / det, -B / det, A / det, reach);   // reach: see kForward
    Proj[At(i * 3 + 2, TexW)] = float4(col, tc.z);   // depth in .w
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
Texture2D<float4>   EGrad : register(t0);   // per entry: kTex texels
Texture2D<float>    GMap  : register(t1);   // N+1 starts into GList
Texture2D<float>    GList : register(t2);   // entry indices
RWTexture2D<float4> G2    : register(u0);   // per Gaussian: 3 texels
#ifdef REFL
// With reflections each entry has a fourth texel, dLoss by the Gaussian's
// reflectivity and normal, summed here into RG for the reflection step.
RWTexture2D<float4> RG    : register(u1);
#define kTex 4
#else
#define kTex 3
#endif

cbuffer Params : register(b0) { uint D0; uint D1; uint N; uint TexW; };

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    if (i >= N) return;
    uint start = uint(GMap[At(i, TexW)]), end = uint(GMap[At(i + 1, TexW)]);
    float4 a = 0, b = 0, c = 0;
#ifdef REFL
    float4 x = 0;
#endif
    for (uint k = start; k < end; ++k) {
        uint e = uint(GList[At(k, TexW)]);
        a += EGrad[At(e * kTex, TexW)];
        b += EGrad[At(e * kTex + 1, TexW)];
        c += EGrad[At(e * kTex + 2, TexW)];
#ifdef REFL
        x += EGrad[At(e * kTex + 3, TexW)];
#endif
    }
    G2[At(i * 3, TexW)] = a;
    G2[At(i * 3 + 1, TexW)] = b;
    G2[At(i * 3 + 2, TexW)] = c;
#ifdef REFL
    RG[At(i, TexW)] = x;
#endif
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

// --- sh: the view-dependent colour's step ------------------------------------
//
// One thread per Gaussian, run BEFORE the update. The summed colour gradient
// is dLoss by the colour SHOWN (base + SH, clamped). A channel held at a
// clamp passes nothing on, so it is zeroed here in G2 -- which is how the
// base colour's own step in the update sees the same mask the CPU applies.
// The rest is Adam on each active coefficient, whose gradient is Y_k times
// the colour's, in place: every thread owns its own texels.
constexpr const char* kShStep = R"(
Texture2D<float4>   State : register(t0);   // position, for the direction
Texture2D<float4>   Proj  : register(t1);   // visibility, and the colour shown
RWTexture2D<float4> G2    : register(u0);
RWTexture2D<float4> Sh    : register(u1);   // 12 texels per Gaussian
RWTexture2D<float4> ShM   : register(u2);   // 24: first moments, then second

cbuffer Params : register(b0) {
    uint D0; uint D1; uint N; uint TexW;
    uint EyeX; uint EyeY; uint EyeZ; uint ShDeg;
    uint Lr; uint C1; uint C2;
};

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    if (i >= N) return;
    if (Proj[At(i * 3, TexW)].w < 0.0) return;          // not in view

    float4 gb = G2[At(i * 3 + 1, TexW)];
    float3 gc = gb.yzw;
    float3 shown = Proj[At(i * 3 + 2, TexW)].xyz;
    [unroll] for (uint ch = 0; ch < 3; ++ch)
        if (shown[ch] <= 0.0 || shown[ch] >= 1.0) gc[ch] = 0.0;
    G2[At(i * 3 + 1, TexW)] = float4(gb.x, gc);
    if (all(gc == 0.0)) return;

    float4 a = State[At(i * 13, TexW)];
    float3 dir = a.xyz - float3(asfloat(EyeX), asfloat(EyeY), asfloat(EyeZ));
    float dl = length(dir);
    if (dl < 1e-12) return;
    float Y[15];
    ShBasis(dir / dl, ShDeg, Y);
    uint active = (ShDeg + 1) * (ShDeg + 1) - 1;

    float lr = asfloat(Lr), c1 = asfloat(C1), c2 = asfloat(C2);
    [unroll] for (uint k2 = 0; k2 < 12; ++k2) {
        float4 cf = Sh[At(i * 12 + k2, TexW)];
        float4 m1 = ShM[At(i * 24 + k2, TexW)];
        float4 m2 = ShM[At(i * 24 + 12 + k2, TexW)];
        [unroll] for (uint j = 0; j < 4; ++j) {
            uint q = k2 * 4 + j;
            if (q >= 45 || q / 3 >= active) continue;
            float g = Y[q / 3] * gc[q % 3];
            m1[j] = 0.9 * m1[j] + 0.1 * g;
            m2[j] = 0.999 * m2[j] + 0.001 * g * g;
            cf[j] -= lr * (m1[j] / c1) / (sqrt(m2[j] / c2) + 1e-15);
        }
        Sh[At(i * 12 + k2, TexW)] = cf;
        ShM[At(i * 24 + k2, TexW)] = m1;
        ShM[At(i * 24 + 12 + k2, TexW)] = m2;
    }
}
)";

// --- reflections: the per-Gaussian extras ------------------------------------------
//
// Deferred reflection (splat_reflect.h) composites each Gaussian's
// reflectivity and normal with the same weights as its colour. This writes
// them per Gaussian -- the sigmoid of the logit, and the unit normal turned to
// face the camera, since a disc has no front -- into `Extra`, which the REFL
// builds of the composite and backward kernels read beside the projected
// record. (A first version put them in the projected record's colour slot
// one after another and ran the composite three times; carrying all seven
// channels in one pass does the same work once.)
constexpr const char* kPayload = R"(
Texture2D<float4>   State : register(t0);
Texture2D<float4>   Refl  : register(t1);   // per Gaussian: reflectivity logit, normal
RWTexture2D<float4> Proj  : register(u0);   // unused; kept so the bindings match
RWTexture2D<float4> Extra : register(u1);   // per Gaussian: r, facing normal

cbuffer Params : register(b0) {
    uint D0; uint D1; uint N; uint TexW;
    uint EyeX; uint EyeY; uint EyeZ; uint Mode;
};

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    if (i >= N) return;
    float4 rf = Refl[At(i, TexW)];
    float3 n = rf.yzw;
    float len = length(n);
    n = (len > 1e-12) ? n / len : float3(0, 0, 1);
    float3 mean = State[At(i * 13, TexW)].xyz;
    if (dot(n, float3(asfloat(EyeX), asfloat(EyeY), asfloat(EyeZ)) - mean) < 0.0) n = -n;
    Extra[At(i, TexW)] = float4(Sigmoid(rf.x), n);
}
)";

// --- reflections: the step -----------------------------------------------------------
//
// Adam on each Gaussian's reflectivity logit and normal, from the gradients
// the REFL sum gathered, as train_splats' CPU loop does: r is the sigmoid of the
// logit, and the normal payload was flip * n / |n|.
constexpr const char* kReflStep = R"(
Texture2D<float4>   State : register(t0);
Texture2D<float4>   Proj  : register(t1);
Texture2D<float4>   RG    : register(t2);
RWTexture2D<float4> Refl  : register(u0);
RWTexture2D<float4> RMom  : register(u1);   // 2 per Gaussian: first moments, second

cbuffer Params : register(b0) {
    uint D0; uint D1; uint N; uint TexW;
    uint EyeX; uint EyeY; uint EyeZ;
    uint LrR; uint LrN; uint C1; uint C2;
};

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    if (i >= N) return;
    if (Proj[At(i * 3, TexW)].w < 0.0) return;
    float4 g = RG[At(i, TexW)];
    if (all(g == 0.0)) return;
    float4 rf = Refl[At(i, TexW)];
    float r = Sigmoid(rf.x);
    float3 n = rf.yzw;
    float len = max(length(n), 1e-12);
    float3 u = n / len;
    float3 mean = State[At(i * 13, TexW)].xyz;
    float flip = dot(u, float3(asfloat(EyeX), asfloat(EyeY), asfloat(EyeZ)) - mean) < 0.0 ? -1.0 : 1.0;
    float3 gu = g.yzw * flip;
    float3 gn = (gu - u * dot(u, gu)) / len;
    float4 grad = float4(g.x * r * (1.0 - r), gn);

    float4 m1 = RMom[At(i * 2, TexW)], m2 = RMom[At(i * 2 + 1, TexW)];
    float c1 = asfloat(C1), c2 = asfloat(C2);
    [unroll] for (uint k = 0; k < 4; ++k) {
        m1[k] = 0.9 * m1[k] + 0.1 * grad[k];
        m2[k] = 0.999 * m2[k] + 0.001 * grad[k] * grad[k];
        float lr = (k == 0) ? asfloat(LrR) : asfloat(LrN);
        rf[k] -= lr * (m1[k] / c1) / (sqrt(m2[k] / c2) + 1e-15);
    }
    Refl[At(i, TexW)] = rf;
    RMom[At(i * 2, TexW)] = m1;
    RMom[At(i * 2 + 1, TexW)] = m2;
}
)";

// --- binning and sorting -------------------------------------------------------------
//
// Every (Gaussian, tile) pair its footprint touches becomes one ENTRY, and the
// entries must end up grouped by tile, near to far within each, ties in index
// order -- the CPU rasteriser's order, so the render matches it.
//
//   scan    each Gaussian's entry count, from its screen texel, prefix-summed
//           into where its entries start (gmap, which the sum kernel reads);
//           also the total and how many Gaussians are in view
//   fill    the entries in SLOT order -- Gaussian by Gaussian, each one's
//           tiles row-major -- as (depth, tile, slot, Gaussian)
//   sort    least-significant-digit radix, 8 bits a pass: four passes over
//           the depth, then one or two over the tile. Each pass is stable,
//           and the entries start in index order, so what comes out is
//           ordered by (tile, depth, index) exactly.
//   finish  the per-tile list, where each slot landed (glist), and each
//           tile's range.
//
// Depths of visible Gaussians are positive (projection culls below 0.01),
// and a positive float's bits order as an unsigned integer does, so the
// depth sorts by its own bit pattern and nothing but float textures is
// needed. Indices and counts are float-exact below 2^24.
//
// A RADIX PASS is three dispatches: count each 2048-entry block's digits,
// prefix-sum each digit's counts across blocks, and scatter -- each block
// placing its entries after every earlier block's with the same digit, and
// in its own order among them. That in-block order is what makes the pass
// stable, and is found by sorting each 256 entries on their digit with eight
// stable one-bit splits in group-shared memory.
constexpr const char* kBinCommon = R"(
cbuffer Params : register(b0) {
    uint D0; uint D1; uint N; uint TexW;
    uint TilesX; uint TilesY; uint NB; uint Pass; uint E; uint NT;
};

static const uint kBlock = 2048;   // entries per group: 256 threads, 8 rounds

// The tiles a screen texel's footprint touches, inclusive, as the CPU
// rasteriser bins it; false when out of view.
bool Rect(float4 sc, out int4 r) {
    r = int4(0, 0, -1, -1);
    if (sc.w < 0.0) return false;
    r.x = clamp(int(floor((sc.x - sc.w) / 16.0)), 0, int(TilesX) - 1);
    r.z = clamp(int(floor((sc.x + sc.w) / 16.0)), 0, int(TilesX) - 1);
    r.y = clamp(int(floor((sc.y - sc.w) / 16.0)), 0, int(TilesY) - 1);
    r.w = clamp(int(floor((sc.y + sc.w) / 16.0)), 0, int(TilesY) - 1);
    return true;
}

// Exclusive prefix sum across the group's 256 threads, and the total. Every
// thread must call it: it synchronises the group.
groupshared uint gWave[256];
uint GroupScan(uint v, uint t, out uint total) {
    uint lanes = WaveGetLaneCount();
    uint w = t / lanes;
    uint pre = WavePrefixSum(v);
    uint wsum = WaveActiveSum(v);
    if (WaveIsFirstLane()) gWave[w] = wsum;
    GroupMemoryBarrierWithGroupSync();
    uint add = 0;
    total = 0;
    uint nw = 256 / lanes;
    for (uint k = 0; k < nw; ++k) {
        uint s = gWave[k];
        if (k < w) add += s;
        total += s;
    }
    GroupMemoryBarrierWithGroupSync();
    return pre + add;
}

// This pass's 8-bit digit: depth bits for passes 0-3, tile bits after.
uint Digit(float4 el) {
    return Pass < 4 ? (asuint(el.x) >> (8 * Pass)) & 255u
                    : (uint(el.y) >> (8 * (Pass - 4))) & 255u;
}
)";

// Entry counts per Gaussian, scanned within each block of 2048.
constexpr const char* kBinScan = R"(
Texture2D<float4>   Screen : register(t0);
RWTexture2D<float>  Local  : register(u0);   // start within the block
RWTexture2D<float4> Sums   : register(u1);   // per block: entries, visible

[numthreads(256, 1, 1)]
void main(uint3 g : SV_GroupID, uint3 gt : SV_GroupThreadID) {
    uint t = gt.x, b = g.x;
    uint carry = 0, vis = 0;
    for (uint r = 0; r < 8; ++r) {
        uint i = b * kBlock + r * 256 + t;
        uint v = 0;
        int4 R;
        if (i < N && Rect(Screen[At(i, TexW)], R)) v = uint((R.z - R.x + 1) * (R.w - R.y + 1));
        uint tot, totV;
        uint ex = GroupScan(v, t, tot);
        GroupScan(v > 0 ? 1u : 0u, t, totV);
        if (i < N) Local[At(i, TexW)] = float(carry + ex);
        carry += tot;
        vis += totV;
    }
    if (t == 0) Sums[At(b, TexW)] = float4(carry, vis, 0, 0);
}
)";

// The blocks' totals scanned, by one group; the grand totals into Tot.
constexpr const char* kBinScanTop = R"(
Texture2D<float4>   Sums     : register(t0);
RWTexture2D<float>  SumsScan : register(u0);
RWTexture2D<float4> Tot      : register(u1);   // entries, visible

[numthreads(256, 1, 1)]
void main(uint3 gt : SV_GroupThreadID) {
    uint t = gt.x;
    uint carry = 0, vis = 0;
    for (uint c = 0; c < NB; c += 256) {
        uint b = c + t;
        float4 s = b < NB ? Sums[At(b, TexW)] : float4(0, 0, 0, 0);
        uint tot, totV;
        uint ex = GroupScan(uint(s.x), t, tot);
        GroupScan(uint(s.y), t, totV);
        if (b < NB) SumsScan[At(b, TexW)] = float(carry + ex);
        carry += tot;
        vis += totV;
    }
    if (t == 0) Tot[uint2(0, 0)] = float4(carry, vis, 0, 0);
}
)";

// Each Gaussian's start in the per-Gaussian list: N+1 of them, the last the
// total, as the sum kernel reads them.
constexpr const char* kBinScanAdd = R"(
Texture2D<float>   Local    : register(t0);
Texture2D<float>   SumsScan : register(t1);
Texture2D<float4>  Tot      : register(t2);
RWTexture2D<float> GMap     : register(u0);

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint j = id.x;
    if (j > N) return;
    GMap[At(j, TexW)] = j < N ? Local[At(j, TexW)] + SumsScan[At(j / kBlock, TexW)]
                              : Tot[uint2(0, 0)].x;
}
)";

// The entries, in slot order.
constexpr const char* kBinFill = R"(
Texture2D<float4>   Screen : register(t0);
Texture2D<float>    GMap   : register(t1);
RWTexture2D<float4> Elem   : register(u0);

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    if (i >= N) return;
    float4 sc = Screen[At(i, TexW)];
    int4 R;
    if (!Rect(sc, R)) return;
    uint slot = uint(GMap[At(i, TexW)]);
    for (int ty = R.y; ty <= R.w; ++ty)
        for (int tx = R.x; tx <= R.z; ++tx) {
            Elem[At(slot, TexW)] = float4(sc.z, float(uint(ty) * TilesX + uint(tx)), float(slot), float(i));
            ++slot;
        }
}
)";

// Radix pass, 1 of 3: each block's digit counts, digit-major.
constexpr const char* kBinHist = R"(
Texture2D<float4>  In    : register(t0);
RWTexture2D<float> Table : register(u0);

groupshared uint gHist[256];

[numthreads(256, 1, 1)]
void main(uint3 g : SV_GroupID, uint3 gt : SV_GroupThreadID) {
    uint t = gt.x, b = g.x;
    gHist[t] = 0;
    GroupMemoryBarrierWithGroupSync();
    for (uint r = 0; r < 8; ++r) {
        uint e = b * kBlock + r * 256 + t;
        if (e < E) {
            uint prev;
            InterlockedAdd(gHist[Digit(In[At(e, TexW)])], 1u, prev);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    Table[At(t * NB + b, TexW)] = float(gHist[t]);
}
)";

// Radix pass, 2 of 3: one group per digit scans that digit's counts across
// the blocks, and totals it.
constexpr const char* kBinRowScan = R"(
Texture2D<float>   Table : register(t0);
RWTexture2D<float> TScan : register(u0);
RWTexture2D<float> DTot  : register(u1);

[numthreads(256, 1, 1)]
void main(uint3 g : SV_GroupID, uint3 gt : SV_GroupThreadID) {
    uint t = gt.x, d = g.x;
    uint carry = 0;
    for (uint c = 0; c < NB; c += 256) {
        uint b = c + t;
        uint v = b < NB ? uint(Table[At(d * NB + b, TexW)]) : 0u;
        uint tot;
        uint ex = GroupScan(v, t, tot);
        if (b < NB) TScan[At(d * NB + b, TexW)] = float(carry + ex);
        carry += tot;
    }
    if (t == 0) DTot[At(d, TexW)] = float(carry);
}
)";

// Radix pass, 3 of 3: the stable scatter.
constexpr const char* kBinScatter = R"(
Texture2D<float4>   In    : register(t0);
Texture2D<float>    TScan : register(t1);
Texture2D<float>    DTot  : register(t2);
RWTexture2D<float4> Out   : register(u0);

groupshared uint   gRun[256];    // where this block's next entry of each digit goes
groupshared uint   gFirst[256];  // first sorted position of each digit, this round
groupshared uint   gDig[256], gDig2[256];
groupshared uint   gOrig[256], gOrig2[256];
groupshared float4 gEl[256];

[numthreads(256, 1, 1)]
void main(uint3 g : SV_GroupID, uint3 gt : SV_GroupThreadID) {
    uint t = gt.x, b = g.x;
    uint total;
    uint base = GroupScan(uint(DTot[At(t, TexW)]), t, total);
    gRun[t] = base + uint(TScan[At(t * NB + b, TexW)]);
    GroupMemoryBarrierWithGroupSync();

    for (uint r = 0; r < 8; ++r) {
        uint e0 = b * kBlock + r * 256;
        uint e = e0 + t;
        float4 el = e < E ? In[At(e, TexW)] : float4(0, 0, 0, 0);
        gEl[t] = el;
        // Past the end sorts last: it only occurs in the final round.
        gDig[t] = e < E ? Digit(el) : 255u;
        gOrig[t] = t;
        GroupMemoryBarrierWithGroupSync();

        // Sort the round's 256 on their digit, stably: eight one-bit splits.
        for (uint bit = 0; bit < 8; ++bit) {
            uint dq = gDig[t], oq = gOrig[t];
            uint f = (dq >> bit) & 1u;
            uint zeros;
            uint zb = GroupScan(1u - f, t, zeros);
            uint np = f == 0 ? zb : zeros + (t - zb);
            gDig2[np] = dq;
            gOrig2[np] = oq;
            GroupMemoryBarrierWithGroupSync();
            gDig[t] = gDig2[t];
            gOrig[t] = gOrig2[t];
            GroupMemoryBarrierWithGroupSync();
        }

        uint dq = gDig[t];
        // (Indices clamped: HLSL need not short-circuit ||.)
        if (t == 0 || gDig[max(t, 1u) - 1] != dq) gFirst[dq] = t;
        GroupMemoryBarrierWithGroupSync();
        uint rank = t - gFirst[dq];
        uint o = gOrig[t];
        if (e0 + o < E) Out[At(gRun[dq] + rank, TexW)] = gEl[o];
        GroupMemoryBarrierWithGroupSync();
        if (t == 255 || gDig[min(t + 1, 255u)] != dq) gRun[dq] += rank + 1;
        GroupMemoryBarrierWithGroupSync();
    }
}
)";

// The sorted entries out as the composite and backward kernels read them:
// each tile's list of Gaussians, where each slot landed, and each tile's
// range -- found by binary search on the sorted tiles.
constexpr const char* kBinFinish = R"(
Texture2D<float4>   Sorted : register(t0);
RWTexture2D<float>  List   : register(u0);
RWTexture2D<float>  GList  : register(u1);
RWTexture2D<float4> Offs   : register(u2);

uint LowerBound(uint tile) {
    uint lo = 0, hi = E;
    while (lo < hi) {
        uint mid = (lo + hi) / 2;
        if (uint(Sorted[At(mid, TexW)].y) < tile) lo = mid + 1; else hi = mid;
    }
    return lo;
}

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint j = id.x;
    if (j < E) {
        float4 el = Sorted[At(j, TexW)];
        List[At(j, TexW)] = el.w;
        GList[At(uint(el.z), TexW)] = float(j);
    }
    if (j < NT) {
        uint lo = LowerBound(j), hi = LowerBound(j + 1);
        Offs[At(j, TexW)] = float4(lo, hi - lo, 0, 0);
    }
}
)";

// --- deferred reflection shading -------------------------------------------------------
//
// splat_reflect.cpp's ShadeDeferred and ShadeDeferredBackward, per pixel, so
// the composited colour, reflectivity and normal never leave the device: the
// shaded image comes back for the loss, and the loss's gradient goes up. The
// environment map is uploaded each step (the caller's Adam moves it), a
// texel of it per float4 in cube order, face by face, row by row.
//
// THE ENVIRONMENT'S GRADIENT needs a sum over every pixel whose reflection
// reads a texel -- a scatter, so atomics, and there are none for floats on
// these textures. It is summed in FIXED POINT instead, in an R32U texture, 2^30
// to one: the L1 loss's gradient totals at most 1 over the whole image (each
// of the `used` channels contributes 1/used), and a pixel's share of it times
// R and bilinear weights never exceeds its own, so no texel's sum can pass
// 2 in magnitude. Integer adds commute, so the sum does not depend on the
// order threads arrive in -- the step stays deterministic.
constexpr const char* kShadeCommon = R"(
cbuffer Params : register(b0) {
    uint D0; uint D1; uint W; uint H; uint TexW; uint Res;
    uint R0; uint R1; uint R2; uint R3; uint R4; uint R5; uint R6; uint R7; uint R8;
    uint Cx; uint Cy; uint Fx; uint Fy; uint Sparsity; uint Scale;
};

static const float kUnit = 1073741824.0;   // 2^30: see above

// The pixel's viewing ray, unit length, as splat_reflect.cpp's RayOf.
float3 RayOf(uint x, uint y) {
    float3 c = float3((float(x) - asfloat(Cx)) / asfloat(Fx),
                      (float(y) - asfloat(Cy)) / asfloat(Fy), 1.0);
    float3 r = float3(asfloat(R0) * c.x + asfloat(R3) * c.y + asfloat(R6) * c.z,
                      asfloat(R1) * c.x + asfloat(R4) * c.y + asfloat(R7) * c.z,
                      asfloat(R2) * c.x + asfloat(R5) * c.y + asfloat(R8) * c.z);
    return normalize(r);
}

// Direction -> cube face and texel coordinates, with their derivatives: the
// CPU's FaceUV.
void FaceUV(float3 d, out uint face, out float u, out float v,
            out float3 du, out float3 dv) {
    float c[3] = {d.x, d.y, d.z};
    float ax = abs(d.x), ay = abs(d.y), az = abs(d.z);
    uint as_, at, am;
    float ss, st, sm;
    if (ax >= ay && ax >= az) {
        am = 0; sm = d.x > 0 ? 1.0 : -1.0;
        if (d.x > 0) { face = 0; as_ = 2; ss = -1; at = 1; st = -1; }
        else         { face = 1; as_ = 2; ss =  1; at = 1; st = -1; }
    } else if (ay >= az) {
        am = 1; sm = d.y > 0 ? 1.0 : -1.0;
        if (d.y > 0) { face = 2; as_ = 0; ss =  1; at = 2; st =  1; }
        else         { face = 3; as_ = 0; ss =  1; at = 2; st = -1; }
    } else {
        am = 2; sm = d.z > 0 ? 1.0 : -1.0;
        if (d.z > 0) { face = 4; as_ = 0; ss =  1; at = 1; st = -1; }
        else         { face = 5; as_ = 0; ss = -1; at = 1; st = -1; }
    }
    float ma = max(sm * c[am], 1e-12);
    float sc = ss * c[as_], tc = st * c[at];
    float res = float(Res);
    u = (sc / ma + 1.0) * 0.5 * res;
    v = (tc / ma + 1.0) * 0.5 * res;
    float gu[3] = {0, 0, 0}, gv[3] = {0, 0, 0};
    float k = 0.5 * res / ma;
    gu[as_] += k * ss;
    gu[am] -= k * sc / ma * sm;
    gv[at] += k * st;
    gv[am] -= k * tc / ma * sm;
    du = float3(gu[0], gu[1], gu[2]);
    dv = float3(gv[0], gv[1], gv[2]);
}

// The bilinear read's four texels and weights, clamped to the face: TapsOf.
void Taps(uint face, float u, float v, out uint at[4], out float w[4],
          out float fx, out float fy) {
    float top = float(Res) - 1.0;
    float x = clamp(u - 0.5, 0.0, top), y = clamp(v - 0.5, 0.0, top);
    uint x0 = uint(x), y0 = uint(y);
    uint x1 = min(x0 + 1, Res - 1), y1 = min(y0 + 1, Res - 1);
    fx = x - float(x0);
    fy = y - float(y0);
    uint b = face * Res;
    at[0] = (b + y0) * Res + x0;
    at[1] = (b + y0) * Res + x1;
    at[2] = (b + y1) * Res + x0;
    at[3] = (b + y1) * Res + x1;
    w[0] = (1 - fx) * (1 - fy);
    w[1] = fx * (1 - fy);
    w[2] = (1 - fx) * fy;
    w[3] = fx * fy;
}
)";

constexpr const char* kShade = R"(
Texture2D<float4>   RGBT : register(t0);   // composited colour, final T
Texture2D<float4>   RN   : register(t1);   // composited R, normal
Texture2D<float4>   Env  : register(t2);
RWTexture2D<float4> Out  : register(u0);   // shaded colour, final T

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint x = id.x, y = id.y;
    if (x >= W || y >= H) return;
    float4 cdT = RGBT[uint2(x, y)];
    float4 rn  = RN[uint2(x, y)];
    float R = rn.x;
    float nl = length(rn.yzw);
    float3 e = float3(0, 0, 0);
    if (R > 1e-9 && nl > 1e-9) {
        float3 v = RayOf(x, y), n = rn.yzw / nl;
        uint face; float u, vv; float3 du, dv;
        FaceUV(v - 2.0 * dot(v, n) * n, face, u, vv, du, dv);
        uint at[4]; float w[4]; float fx, fy;
        Taps(face, u, vv, at, w, fx, fy);
        [unroll] for (uint k = 0; k < 4; ++k) e += w[k] * Env[At(at[k], TexW)].rgb;
    }
    Out[uint2(x, y)] = float4((1.0 - R) * cdT.rgb + R * e, cdT.w);
}
)";

constexpr const char* kShadeBwd = R"(
Texture2D<float4>   RGBT  : register(t0);
Texture2D<float4>   RN    : register(t1);
Texture2D<float4>   Env   : register(t2);
Texture2D<float4>   DOut  : register(t3);   // dLoss/d(shaded colour)
RWTexture2D<float4> DRGBT : register(u0);   // dLoss/d(composited colour), final T
RWTexture2D<float4> DRN   : register(u1);   // dLoss/dR, dLoss/d(normal)
RWTexture2D<uint>   EnvG  : register(u2);   // 3 per texel, fixed point

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint x = id.x, y = id.y;
    if (x >= W || y >= H) return;
    float4 cdT = RGBT[uint2(x, y)];
    float4 rn  = RN[uint2(x, y)];
    float3 g   = DOut[uint2(x, y)].rgb;
    float R = rn.x;
    float nl = length(rn.yzw);
    float sp = asfloat(Sparsity);   // reflect_sparsity over the measured pixels
    DRGBT[uint2(x, y)] = float4((1.0 - R) * g, cdT.w);
    if (R <= 1e-9 || nl <= 1e-9) {
        // No reflection drawn here, so only R's own term, e = 0.
        DRN[uint2(x, y)] = float4(-dot(g, cdT.rgb) + sp, 0, 0, 0);
        return;
    }
    float3 n = rn.yzw / nl;
    float3 v = RayOf(x, y);
    float3 om = v - 2.0 * dot(v, n) * n;
    uint face; float u, vv; float3 du, dv;
    FaceUV(om, face, u, vv, du, dv);
    uint at[4]; float w[4]; float fx, fy;
    Taps(face, u, vv, at, w, fx, fy);
    float3 T[4];
    float3 e = float3(0, 0, 0);
    [unroll] for (uint k = 0; k < 4; ++k) {
        T[k] = Env[At(at[k], TexW)].rgb;
        e += w[k] * T[k];
    }
    float dR = dot(g, e - cdT.rgb) + sp;

    // The environment's texels: bilinear weights times R times g.
    float3 gR = g * R;
    [unroll] for (uint k2 = 0; k2 < 4; ++k2)
        [unroll] for (uint ch = 0; ch < 3; ++ch) {
            int q = int(round(w[k2] * gR[ch] * kUnit));
            if (q != 0) {
                uint prev;
                InterlockedAdd(EnvG[At(at[k2] * 3 + ch, TexW)], asuint(q), prev);
            }
        }

    // dLoss/d(reflected direction): the face coordinates' change with
    // direction times the bilinear slope, zero where the read is clamped.
    float top = float(Res) - 1.0;
    float xs = u - 0.5, ys = vv - 0.5;
    float gx = dot(gR, (1 - fy) * (T[1] - T[0]) + fy * (T[3] - T[2]));
    float gy = dot(gR, (1 - fx) * (T[2] - T[0]) + fx * (T[3] - T[1]));
    if (xs < 0.0 || xs > top) gx = 0.0;
    if (ys < 0.0 || ys > top) gy = 0.0;
    float3 gOm = du * gx + dv * gy;
    // om = v - 2 (v.n) n, n = N / |N|.
    float vn = dot(v, n), gn = dot(gOm, n);
    float3 gUnit = -2.0 * (v * gn + gOm * vn);
    float3 gN = (gUnit - n * dot(n, gUnit)) / nl;
    DRN[uint2(x, y)] = float4(dR, gN);
}
)";

constexpr const char* kClearU = R"(
RWTexture2D<uint> Buf : register(u0);
cbuffer Params : register(b0) { uint D0; uint D1; uint N; uint TexW; };
[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x < N) Buf[At(id.x, TexW)] = 0;
}
)";

// --- densification -------------------------------------------------------------------
//
// The summary densification decides from, two texels per Gaussian: the
// counters as they stand (gradient sum, view count, largest screen radius),
// and the opacity logit with the largest log-scale. The CPU divides, so its
// decisions are the ones it would take from a full download.
constexpr const char* kDensifyStats = R"(
Texture2D<float4>   State : register(t0);
RWTexture2D<float4> Out   : register(u0);

cbuffer Params : register(b0) { uint D0; uint D1; uint N; uint TexW; };

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    if (i >= N) return;
    float4 b = State[At(i * 13 + 1, TexW)];   // logScale y z, quat w x
    float4 a = State[At(i * 13, TexW)];       // mean, logScale x
    float4 c = State[At(i * 13 + 2, TexW)];   // quat y z, opacity, colour r
    Out[At(i * 2, TexW)] = State[At(i * 13 + 12, TexW)];
    Out[At(i * 2 + 1, TexW)] = float4(c.z, max(a.w, max(b.x, b.y)), 0, 0);
}
)";

// The state after densification, one thread per NEW Gaussian, from the plan:
// per Gaussian (source index, kind) and the split offset z. Every Gaussian's
// counters start from zero; only a kept one keeps its moments; a split child
// moves by R * (z * scale) and shrinks by 1.6, as train_splats' CPU path does.
constexpr const char* kRebuild = R"(
Texture2D<float4>   Old  : register(t0);
Texture2D<float4>   Plan : register(t1);
RWTexture2D<float4> New  : register(u0);

cbuffer Params : register(b0) { uint D0; uint D1; uint N; uint TexW; };

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint j = id.x;
    if (j >= N) return;
    float4 pl = Plan[At(j * 2, TexW)];
    float3 z = Plan[At(j * 2 + 1, TexW)].xyz;
    uint src = uint(pl.x), kind = uint(pl.y);
    float p[14];
    Load14(Old, src * 13, TexW, p);
    if (kind == 2) {
        float4 q = float4(p[6], p[7], p[8], p[9]);
        float ql = length(q);
        q = (ql > 1e-12) ? q / ql : float4(1, 0, 0, 0);
        float R[9];
        Rotation(q, R);
        float3 v = z * float3(exp(p[3]), exp(p[4]), exp(p[5]));
        [unroll] for (uint a = 0; a < 3; ++a)
            p[a] += R[a * 3 + 0] * v.x + R[a * 3 + 1] * v.y + R[a * 3 + 2] * v.z;
        [unroll] for (uint k = 3; k < 6; ++k) p[k] -= log(1.6);
    }
    uint o = j * 13;
    New[At(o, TexW)]     = float4(p[0], p[1], p[2], p[3]);
    New[At(o + 1, TexW)] = float4(p[4], p[5], p[6], p[7]);
    New[At(o + 2, TexW)] = float4(p[8], p[9], p[10], p[11]);
    New[At(o + 3, TexW)] = float4(p[12], p[13], 0, 0);
    [unroll] for (uint t = 4; t < 12; ++t)
        New[At(o + t, TexW)] = (kind == 0) ? Old[At(src * 13 + t, TexW)] : float4(0, 0, 0, 0);
    New[At(o + 12, TexW)] = float4(0, 0, 0, 0);
}
)";

// Per-Gaussian side data -- view-dependent colour, reflections -- after
// densification: `Values` texels of values and `Moments` of Adam's moments per
// Gaussian, the values inherited by every new Gaussian, the moments only by a
// kept one.
constexpr const char* kRebuildSide = R"(
Texture2D<float4>   Val  : register(t0);
Texture2D<float4>   Mom  : register(t1);
Texture2D<float4>   Plan : register(t2);
RWTexture2D<float4> NVal : register(u0);
RWTexture2D<float4> NMom : register(u1);

cbuffer Params : register(b0) { uint D0; uint D1; uint N; uint TexW; uint Values; uint Moments; };

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint j = id.x;
    if (j >= N) return;
    float4 pl = Plan[At(j * 2, TexW)];
    uint src = uint(pl.x), kind = uint(pl.y);
    for (uint t = 0; t < Values; ++t)
        NVal[At(j * Values + t, TexW)] = Val[At(src * Values + t, TexW)];
    for (uint u = 0; u < Moments; ++u)
        NMom[At(j * Moments + u, TexW)] =
            (kind == 0) ? Mom[At(src * Moments + u, TexW)] : float4(0, 0, 0, 0);
}
)";

// The opacity reset: every opacity logit capped, its moments cleared.
constexpr const char* kResetOpacity = R"(
Texture2D<float4>   Old : register(t0);
RWTexture2D<float4> New : register(u0);

cbuffer Params : register(b0) { uint D0; uint D1; uint N; uint TexW; uint Cap; };

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    if (i >= N) return;
    uint o = i * 13;
    [unroll] for (uint t = 0; t < 13; ++t) {
        float4 v = Old[At(o + t, TexW)];
        if (t == 2) v.z = min(v.z, asfloat(Cap));      // opacity, parameter 10
        if (t == 6 || t == 10) v.z = 0.0;              // its first and second moments
        New[At(o + t, TexW)] = v;
    }
}
)";

}  // namespace

struct SplatTrainerGpu::Impl {
    ComputeContext* ctx = nullptr;
    ComputeKernel   project, sum, update, fwd, bwd, shStep, payload, reflStep;
    ComputeKernel   densifyStats, rebuild, rebuildSide, resetOpacity;
    ComputeKernel   binScan, binScanTop, binScanAdd, binFill, binHist, binRowScan,
                    binScatter, binFinish;
    ComputeKernel   shade, shadeBwd, clearU;   // deferred reflection shading
    // The same three passes compiled with REFL: colour, reflectivity and
    // normal composited and differentiated together in one go.
    ComputeKernel   fwdR, bwdR, sumR;
    bool            ready = false;

    GpuImage stateA, stateB;            // ping-pong: A holds the current state
    GpuImage plan, stats;               // densification: its plan, its summary
    // Binning: the count scan (per Gaussian, per block, totals), the entries
    // ping-ponged through the radix passes, and each pass's digit tables.
    GpuImage binLocal, binSums, binSumsScan, binTot, elemA, elemB, table, tscan, dtot;
    // Deferred shading: the environment (a texel per float4), the shaded
    // image, its loss gradient, and the environment's gradient in fixed point.
    GpuImage envTex, shaded, dshade, envG;
    std::vector<float> envStage;
    // View-dependent colour, when training has it: coefficients (12 texels a
    // Gaussian) and Adam's moments (24), both updated in place by kShStep.
    GpuImage shTex, shMom;
    bool     haveSh = false;
    // Reflections, when training has them: per Gaussian the reflectivity
    // logit and normal (1 texel) and Adam's moments (2); per pass the saved
    // colour payload, the gradient tally and the reflection gradients; and a
    // zero depth-gradient image for the passes that must add no depth term.
    GpuImage reflTex, reflMom, saved, rg, rn, drn;
    bool     haveRefl = false;
    GpuImage proj, screen, list, offs, gmap, glist, rgbt, last, drgbt, egrad, g2;
    GpuImage depth, ddepth, dshift; // expected depth, dLoss/d(depth), target
    bool     ddepthZero = false;    // ddepth holds zeros: no re-upload needed
    std::vector<float> depthStage, ddStage, tStage, shiftStage;
    size_t   n = 0;
    std::vector<float> stage;

    bool Ensure(GpuImage& img, const ImageDesc& d) {
        if (img.Valid() && img.desc == d) return true;
        // A recorded dispatch may still use the old texture: submit first.
        if (img.Valid() && !Sync()) return false;
        img.Release();
        return ctx->CreateImage(d, &img);
    }
    // Submits what is recorded and waits for it.
    bool Sync() {
        std::string e;
        return ctx->Flush(&e);
    }
    // TGLAB_SPLAT_NOBATCH restores a submission per dispatch, for bisecting
    // a hang.
    static bool NoBatch() {
        static const bool no = [] {
            char buf[8] = {};
            return GetEnvironmentVariableA("TGLAB_SPLAT_NOBATCH", buf, sizeof buf) > 0;
        }();
        return no;
    }
    // Room for at least `texels` in the flat layout, grown with headroom and
    // never shrunk: the binning's sizes change every step, and reallocating
    // tens of megabytes each time would cost more than the work.
    bool Reserve(GpuImage& img, size_t texels, Format f) {
        const size_t have = img.Valid() && img.desc.format == f && img.desc.width == kTexW
                                ? size_t(img.desc.height) * size_t(kTexW) : 0;
        if (have >= std::max<size_t>(1, texels)) return true;
        return Ensure(img, FlatDesc(std::max<size_t>(1, texels + texels / 4), f));
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
    // One dispatch, RECORDED: it is submitted with the others around it by
    // the next transfer, Sync, or the framework's own limits (see the header).
    bool Run(const ComputeKernel& k, const std::vector<const GpuImage*>& in,
             const std::vector<GpuImage*>& out,
             const std::vector<uint32_t>& c, uint32_t gx, uint32_t gy,
             std::string* err) {
        return ctx->Dispatch(k, in, out, c, err, gx, gy) && (!NoBatch() || ctx->Flush(err));
    }

    // --- deferred shading (kShadeCommon) ----------------------------------------
    static std::vector<uint32_t> ShadeC(const SplatCam& cam, int res, double sparsity) {
        std::vector<uint32_t> c{uint32_t(cam.w), uint32_t(cam.h), uint32_t(kTexW),
                                uint32_t(res)};
        for (int k = 0; k < 9; ++k) c.push_back(FloatBits(float(cam.R.m[k])));
        for (double v : {cam.cx, cam.cy, cam.fx, cam.fy, sparsity}) c.push_back(FloatBits(float(v)));
        c.push_back(0u);
        return c;
    }
    // The environment as it stands this step: the caller's Adam moves it.
    bool PutEnv(const EnvMap& env) {
        const size_t texels = size_t(6) * size_t(env.res) * size_t(env.res);
        const ImageDesc d = FlatDesc(texels, Format::RGBA32F);
        envStage.assign(size_t(d.width) * size_t(d.height) * 4, 0.0f);
        for (size_t t = 0; t < texels; ++t)
            for (int ch = 0; ch < 3; ++ch)
                envStage[t * 4 + size_t(ch)] = float(env.texels[t * 3 + size_t(ch)]);
        return Ensure(envTex, d) && Put(envTex, d, envStage);
    }
    // rgbt and rn (the composite's) shaded into `shaded`.
    bool ShadeFwd(const SplatCam& cam, const EnvMap& env, std::string* err) {
        const ImageDesc d{cam.w, cam.h, Format::RGBA32F};
        if (!Ensure(shaded, d) || !PutEnv(env)) {
            *err = "could not set up the reflection shading";
            return false;
        }
        return Run(shade, {&rgbt, &rn, &envTex}, {&shaded}, ShadeC(cam, env.res, 0.0),
                   uint32_t((cam.w + 7) / 8), uint32_t((cam.h + 7) / 8), err);
    }
    // dshade (the loss's gradient) back through the shading into drgbt and
    // drn, which the REFL backward reads, and the environment's gradient ADDED
    // into *envGrad. `sparsity` is reflect_sparsity over the measured pixels.
    bool ShadeBwd(const SplatCam& cam, const EnvMap& env, double sparsity,
                  std::vector<double>* envGrad, std::string* err) {
        const ImageDesc d{cam.w, cam.h, Format::RGBA32F};
        const size_t ng = size_t(6) * size_t(env.res) * size_t(env.res) * 3;
        if (!Ensure(drgbt, d) || !Ensure(drn, d) || !Reserve(envG, ng, Format::R32U)) {
            *err = "could not allocate the reflection gradients";
            return false;
        }
        if (!Run(clearU, {}, {&envG}, {uint32_t(ng), uint32_t(kTexW)},
                 uint32_t((ng + 255) / 256), 1, err) ||
            !Run(shadeBwd, {&rgbt, &rn, &envTex, &dshade}, {&drgbt, &drn, &envG},
                 ShadeC(cam, env.res, sparsity), uint32_t((cam.w + 7) / 8),
                 uint32_t((cam.h + 7) / 8), err))
            return false;
        if (envGrad) {
            if (!Get(envG, size_t(envG.desc.width) * size_t(envG.desc.height))) {
                *err = "could not read back the environment's gradient";
                return false;
            }
            if (envGrad->size() != ng) envGrad->assign(ng, 0.0);
            for (size_t i = 0; i < ng; ++i) {
                int32_t q = 0;
                std::memcpy(&q, &stage[i], 4);
                (*envGrad)[i] += double(q) / 1073741824.0;   // kUnit, 2^30
            }
        }
        return true;
    }
};

SplatTrainerGpu::SplatTrainerGpu(ComputeContext* gpu) : m(std::make_unique<Impl>()) {
    m->ctx = gpu;
}
SplatTrainerGpu::~SplatTrainerGpu() {
    // Nothing recorded may outlive the textures it uses.
    if (m->ctx && m->ctx->Ready()) {
        std::lock_guard<std::mutex> lock(m->ctx->SubmitMutex());
        m->Sync();
    }
}

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
            !s.ctx->CreateKernel(std::string(kCommon) + kPayload, "main",
                                 "splat_payload", &s.payload, err) ||
            !s.ctx->CreateKernel(std::string(kCommon) + kReflStep, "main",
                                 "splat_refl_step", &s.reflStep, err) ||
            !s.ctx->CreateKernel(std::string(kCommon) + kShStep, "main",
                                 "splat_sh_step", &s.shStep, err) ||
            !s.ctx->CreateKernel(std::string("#define REFL 1\n") + splat_kernels::kForward,
                                 "main", "splat_forward_refl", &s.fwdR, err) ||
            !s.ctx->CreateKernel(std::string("#define REFL 1\n") + splat_kernels::kBackward,
                                 "main", "splat_backward_refl", &s.bwdR, err) ||
            !s.ctx->CreateKernel(std::string("#define REFL 1\n") + kCommon + kSum,
                                 "main", "splat_sum_refl", &s.sumR, err) ||
            !s.ctx->CreateKernel(std::string(kCommon) + kBinCommon + kBinScan, "main",
                                 "splat_bin_scan", &s.binScan, err) ||
            !s.ctx->CreateKernel(std::string(kCommon) + kBinCommon + kBinScanTop, "main",
                                 "splat_bin_scan_top", &s.binScanTop, err) ||
            !s.ctx->CreateKernel(std::string(kCommon) + kBinCommon + kBinScanAdd, "main",
                                 "splat_bin_scan_add", &s.binScanAdd, err) ||
            !s.ctx->CreateKernel(std::string(kCommon) + kBinCommon + kBinFill, "main",
                                 "splat_bin_fill", &s.binFill, err) ||
            !s.ctx->CreateKernel(std::string(kCommon) + kBinCommon + kBinHist, "main",
                                 "splat_bin_hist", &s.binHist, err) ||
            !s.ctx->CreateKernel(std::string(kCommon) + kBinCommon + kBinRowScan, "main",
                                 "splat_bin_row_scan", &s.binRowScan, err) ||
            !s.ctx->CreateKernel(std::string(kCommon) + kBinCommon + kBinScatter, "main",
                                 "splat_bin_scatter", &s.binScatter, err) ||
            !s.ctx->CreateKernel(std::string(kCommon) + kBinCommon + kBinFinish, "main",
                                 "splat_bin_finish", &s.binFinish, err) ||
            !s.ctx->CreateKernel(std::string(kCommon) + kShadeCommon + kShade, "main",
                                 "splat_shade", &s.shade, err) ||
            !s.ctx->CreateKernel(std::string(kCommon) + kShadeCommon + kShadeBwd, "main",
                                 "splat_shade_bwd", &s.shadeBwd, err) ||
            !s.ctx->CreateKernel(std::string(kCommon) + kClearU, "main",
                                 "splat_clear_u", &s.clearU, err) ||
            !s.ctx->CreateKernel(std::string(kCommon) + kDensifyStats, "main",
                                 "splat_densify_stats", &s.densifyStats, err) ||
            !s.ctx->CreateKernel(std::string(kCommon) + kRebuild, "main",
                                 "splat_rebuild", &s.rebuild, err) ||
            !s.ctx->CreateKernel(std::string(kCommon) + kRebuildSide, "main",
                                 "splat_rebuild_side", &s.rebuildSide, err) ||
            !s.ctx->CreateKernel(std::string(kCommon) + kResetOpacity, "main",
                                 "splat_reset_opacity", &s.resetOpacity, err) ||
            !s.ctx->CreateKernel(splat_kernels::kBackward, "main", "splat_backward",
                                 &s.bwd, err))
            return false;
        s.ready = true;
    }

    s.n = params.size();
    s.haveSh = false;   // the count may have changed: UploadSh follows
    s.haveRefl = false; // ...and UploadRefl
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

// The view-dependent colour, beside the state Upload sends: kShRest
// coefficients per Gaussian and Adam's two moments for each, in the same
// order as the parameters. Called after Upload, since that decides n.
bool SplatTrainerGpu::UploadSh(const std::vector<double>& sh,
                               const std::vector<double>& m1,
                               const std::vector<double>& m2, std::string* err) {
    Impl& s = *m;
    std::lock_guard<std::mutex> lock(s.ctx->SubmitMutex());
    const size_t n = s.n;
    if (sh.size() < n * size_t(kShRest)) { *err = "coefficients do not match"; return false; }
    const ImageDesc dc = FlatDesc(std::max<size_t>(1, n * 12), Format::RGBA32F);
    const ImageDesc dm = FlatDesc(std::max<size_t>(1, n * 24), Format::RGBA32F);
    std::vector<float> a(size_t(dc.width) * size_t(dc.height) * 4, 0.0f);
    std::vector<float> b(size_t(dm.width) * size_t(dm.height) * 4, 0.0f);
    for (size_t i = 0; i < n; ++i)
        for (int q = 0; q < kShRest; ++q) {
            const size_t src = i * size_t(kShRest) + size_t(q);
            a[i * 48 + size_t(q)] = float(sh[src]);
            b[i * 96 + size_t(q)] = float(m1[src]);
            b[i * 96 + 48 + size_t(q)] = float(m2[src]);
        }
    if (!s.Ensure(s.shTex, dc) || !s.Ensure(s.shMom, dm) ||
        !s.Put(s.shTex, dc, a) || !s.Put(s.shMom, dm, b)) {
        *err = "could not upload the view-dependent colour";
        return false;
    }
    s.haveSh = true;
    return true;
}

bool SplatTrainerGpu::DownloadSh(std::vector<double>* sh, std::vector<double>* m1,
                                 std::vector<double>* m2, std::string* err) {
    Impl& s = *m;
    if (!s.haveSh) { *err = "no view-dependent colour on the device"; return false; }
    std::lock_guard<std::mutex> lock(s.ctx->SubmitMutex());
    const size_t n = s.n;
    sh->assign(n * size_t(kShRest), 0.0);
    m1->assign(n * size_t(kShRest), 0.0);
    m2->assign(n * size_t(kShRest), 0.0);
    if (!s.Get(s.shTex, size_t(s.shTex.desc.width) * size_t(s.shTex.desc.height) * 4)) {
        *err = "could not read back the view-dependent colour";
        return false;
    }
    for (size_t i = 0; i < n; ++i)
        for (int q = 0; q < kShRest; ++q)
            (*sh)[i * size_t(kShRest) + size_t(q)] = double(s.stage[i * 48 + size_t(q)]);
    if (!s.Get(s.shMom, size_t(s.shMom.desc.width) * size_t(s.shMom.desc.height) * 4)) {
        *err = "could not read back the view-dependent colour's moments";
        return false;
    }
    for (size_t i = 0; i < n; ++i)
        for (int q = 0; q < kShRest; ++q) {
            (*m1)[i * size_t(kShRest) + size_t(q)] = double(s.stage[i * 96 + size_t(q)]);
            (*m2)[i * size_t(kShRest) + size_t(q)] = double(s.stage[i * 96 + 48 + size_t(q)]);
        }
    return true;
}

// Reflectivity logit and normal per Gaussian, and Adam's moments for the four
// (4 per Gaussian each, in the CPU's order). Called after Upload.
bool SplatTrainerGpu::UploadRefl(const std::vector<ReflParam>& refl,
                                 const std::vector<double>& m1,
                                 const std::vector<double>& m2, std::string* err) {
    Impl& s = *m;
    std::lock_guard<std::mutex> lock(s.ctx->SubmitMutex());
    const size_t n = s.n;
    if (refl.size() < n || m1.size() < n * 4 || m2.size() < n * 4) {
        *err = "reflections do not match";
        return false;
    }
    const ImageDesc d1 = FlatDesc(std::max<size_t>(1, n), Format::RGBA32F);
    const ImageDesc d2 = FlatDesc(std::max<size_t>(1, n * 2), Format::RGBA32F);
    std::vector<float> a(size_t(d1.width) * size_t(d1.height) * 4, 0.0f);
    std::vector<float> b(size_t(d2.width) * size_t(d2.height) * 4, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        a[i * 4 + 0] = float(refl[i].reflLogit);
        for (int k = 0; k < 3; ++k) a[i * 4 + 1 + size_t(k)] = float(refl[i].normal[k]);
        for (int k = 0; k < 4; ++k) {
            b[i * 8 + size_t(k)] = float(m1[i * 4 + size_t(k)]);
            b[i * 8 + 4 + size_t(k)] = float(m2[i * 4 + size_t(k)]);
        }
    }
    if (!s.Ensure(s.reflTex, d1) || !s.Ensure(s.reflMom, d2) ||
        !s.Put(s.reflTex, d1, a) || !s.Put(s.reflMom, d2, b)) {
        *err = "could not upload the reflections";
        return false;
    }
    s.haveRefl = true;
    return true;
}

bool SplatTrainerGpu::DownloadRefl(std::vector<ReflParam>* refl, std::vector<double>* m1,
                                   std::vector<double>* m2, std::string* err) {
    Impl& s = *m;
    if (!s.haveRefl) { *err = "no reflections on the device"; return false; }
    std::lock_guard<std::mutex> lock(s.ctx->SubmitMutex());
    const size_t n = s.n;
    refl->assign(n, ReflParam{});
    m1->assign(n * 4, 0.0);
    m2->assign(n * 4, 0.0);
    if (!s.Get(s.reflTex, size_t(s.reflTex.desc.width) * size_t(s.reflTex.desc.height) * 4)) {
        *err = "could not read back the reflections";
        return false;
    }
    for (size_t i = 0; i < n; ++i) {
        (*refl)[i].reflLogit = double(s.stage[i * 4]);
        for (int k = 0; k < 3; ++k) (*refl)[i].normal[k] = double(s.stage[i * 4 + 1 + size_t(k)]);
    }
    if (!s.Get(s.reflMom, size_t(s.reflMom.desc.width) * size_t(s.reflMom.desc.height) * 4)) {
        *err = "could not read back the reflections' moments";
        return false;
    }
    for (size_t i = 0; i < n; ++i)
        for (int k = 0; k < 4; ++k) {
            (*m1)[i * 4 + size_t(k)] = double(s.stage[i * 8 + size_t(k)]);
            (*m2)[i * 4 + size_t(k)] = double(s.stage[i * 8 + 4 + size_t(k)]);
        }
    return true;
}

bool SplatTrainerGpu::ShadeCheck(const SplatCam& cam, const EnvMap& env,
                                 const std::vector<double>& Cd, const std::vector<double>& Rm,
                                 const std::vector<double>& Nm, const std::vector<double>& dOut,
                                 double sparsity, std::vector<double>* shaded,
                                 std::vector<double>* dCd, std::vector<double>* dRm,
                                 std::vector<double>* dNm, std::vector<double>* envGrad,
                                 std::string* err) {
    Impl& s = *m;
    if (!s.ready) { *err = "no kernels: Upload first"; return false; }
    std::lock_guard<std::mutex> lock(s.ctx->SubmitMutex());
    const size_t np = size_t(cam.w) * size_t(cam.h);
    const ImageDesc d{cam.w, cam.h, Format::RGBA32F};
    auto put = [&](GpuImage& img, const std::vector<double>& a, const std::vector<double>* b) {
        s.stage.assign(np * 4, 0.0f);
        for (size_t i = 0; i < np; ++i) {
            if (b) {   // R from channel 0 of a, the normal from b
                s.stage[i * 4] = float(a[i * 3]);
                for (int ch = 0; ch < 3; ++ch) s.stage[i * 4 + 1 + size_t(ch)] = float((*b)[i * 3 + size_t(ch)]);
            } else {
                for (int ch = 0; ch < 3; ++ch) s.stage[i * 4 + size_t(ch)] = float(a[i * 3 + size_t(ch)]);
            }
        }
        return s.Ensure(img, d) && s.Put(img, d);
    };
    if (!put(s.rgbt, Cd, nullptr) || !put(s.rn, Rm, &Nm) || !put(s.dshade, dOut, nullptr)) {
        *err = "could not upload the maps";
        return false;
    }
    auto get = [&](const GpuImage& img, std::vector<double>* out, int first, int count) {
        if (!s.Get(img, np * 4)) return false;
        out->assign(np * 3, 0.0);
        for (size_t i = 0; i < np; ++i)
            for (int ch = 0; ch < count; ++ch)
                (*out)[i * 3 + size_t(ch)] = double(s.stage[i * 4 + size_t(first + ch)]);
        return true;
    };
    if (!s.ShadeFwd(cam, env, err) || !get(s.shaded, shaded, 0, 3) ||
        !s.ShadeBwd(cam, env, sparsity, envGrad, err) || !get(s.drgbt, dCd, 0, 3) ||
        !get(s.drn, dRm, 0, 1) || !get(s.drn, dNm, 1, 3)) {
        if (err->empty()) *err = "could not read back the shading";
        return false;
    }
    return true;
}

bool SplatTrainerGpu::DensifyStats(std::vector<DensifyIn>* out, std::string* err) {
    Impl& s = *m;
    std::lock_guard<std::mutex> lock(s.ctx->SubmitMutex());
    const size_t n = s.n;
    out->assign(n, DensifyIn{});
    if (n == 0) return true;
    const ImageDesc d = FlatDesc(n * 2, Format::RGBA32F);
    const uint32_t groups = uint32_t((n + 255) / 256);
    if (!s.Ensure(s.stats, d) ||
        !s.Run(s.densifyStats, {&s.stateA}, {&s.stats}, {uint32_t(n), uint32_t(kTexW)},
               groups, 1, err))
        return false;
    if (!s.Get(s.stats, size_t(d.width) * size_t(d.height) * 4)) {
        *err = "could not read back the densification statistics";
        return false;
    }
    for (size_t i = 0; i < n; ++i) {
        const float* t = &s.stage[i * 8];
        DensifyIn& o = (*out)[i];
        const int count = int(t[1] + 0.5f);
        o.gradAvg = count > 0 ? double(t[0]) / double(count) : -1.0;
        o.maxScreen = double(t[2]);
        o.opacityLogit = double(t[4]);
        o.maxLogScale = double(t[5]);
    }
    return true;
}

bool SplatTrainerGpu::ApplyPlan(const std::vector<DensifyPlan>& plan, std::string* err) {
    Impl& s = *m;
    std::lock_guard<std::mutex> lock(s.ctx->SubmitMutex());
    const size_t nn = plan.size();
    if (nn >= (size_t(1) << 24)) { *err = "too many Gaussians for the device"; return false; }
    // Source indices travel as floats: exact below 2^24.
    const ImageDesc pd = FlatDesc(std::max<size_t>(1, nn * 2), Format::RGBA32F);
    s.stage.assign(size_t(pd.width) * size_t(pd.height) * 4, 0.0f);
    for (size_t j = 0; j < nn; ++j) {
        float* t = &s.stage[j * 8];
        t[0] = float(plan[j].src);
        t[1] = float(plan[j].kind);
        t[4] = plan[j].z[0];
        t[5] = plan[j].z[1];
        t[6] = plan[j].z[2];
    }
    if (!s.Ensure(s.plan, pd) || !s.Put(s.plan, pd)) {
        *err = "could not upload the densification plan";
        return false;
    }
    const uint32_t groups = uint32_t(std::max<size_t>(1, (nn + 255) / 256));
    const std::vector<uint32_t> c{uint32_t(nn), uint32_t(kTexW)};

    // The state, into B at its new size, which then becomes A.
    const ImageDesc sd = FlatDesc(std::max<size_t>(1, nn * kStateTexels), Format::RGBA32F);
    if (!s.Ensure(s.stateB, sd) ||
        (nn > 0 && !s.Run(s.rebuild, {&s.stateA, &s.plan}, {&s.stateB}, c, groups, 1, err)))
        return false;
    std::swap(s.stateA, s.stateB);
    if (!s.Ensure(s.stateB, sd)) { *err = "could not resize the training state"; return false; }

    // Side data the same way, through fresh textures.
    auto side = [&](GpuImage& val, GpuImage& mom, uint32_t values, uint32_t moments) {
        const ImageDesc vd = FlatDesc(std::max<size_t>(1, nn * values), Format::RGBA32F);
        const ImageDesc md = FlatDesc(std::max<size_t>(1, nn * moments), Format::RGBA32F);
        GpuImage nv, nm;
        if (!s.ctx->CreateImage(vd, &nv) || !s.ctx->CreateImage(md, &nm)) {
            *err = "could not resize the per-Gaussian data";
            return false;
        }
        std::vector<uint32_t> cs = c;
        cs.push_back(values);
        cs.push_back(moments);
        if (nn > 0 && !s.Run(s.rebuildSide, {&val, &mom, &s.plan}, {&nv, &nm}, cs, groups, 1, err))
            return false;
        // The old textures are released on return: submit first.
        if (!s.Sync()) { *err = "could not rebuild the per-Gaussian data"; return false; }
        std::swap(val, nv);
        std::swap(mom, nm);
        return true;
    };
    if (s.haveSh && !side(s.shTex, s.shMom, 12, 24)) return false;
    if (s.haveRefl && !side(s.reflTex, s.reflMom, 1, 2)) return false;
    s.n = nn;
    return true;
}

bool SplatTrainerGpu::ResetOpacity(double cap, std::string* err) {
    Impl& s = *m;
    std::lock_guard<std::mutex> lock(s.ctx->SubmitMutex());
    if (s.n == 0) return true;
    const uint32_t groups = uint32_t((s.n + 255) / 256);
    if (!s.Run(s.resetOpacity, {&s.stateA}, {&s.stateB},
               {uint32_t(s.n), uint32_t(kTexW), FloatBits(float(cap))}, groups, 1, err))
        return false;
    std::swap(s.stateA, s.stateB);
    return true;
}

bool SplatTrainerGpu::Step(const SplatCam& cam, const RasterOptions& opt,
                           const std::vector<double>& target, double lrMean,
                           double c1, double c2, std::string* err,
                           const std::vector<float>* depthTarget,
                           double depthWeight, int shDegree, double shLr,
                           const ReflStepArgs* ra) {
    return StepImpl(cam, opt, target, lrMean, c1, c2, err, depthTarget, depthWeight,
                    shDegree, shLr, ra, nullptr);
}

// The forward half of a step alone: what training composites and scores,
// view-dependent colour and reflections included, handed back as w*h*3 --
// for render_splats, so a render made to judge a fit is the very picture
// the fit was scored on.
bool SplatTrainerGpu::Render(const SplatCam& cam, const RasterOptions& opt, int shDegree,
                             const EnvMap* env, std::vector<double>* rgb,
                             std::string* err, std::vector<double>* depth,
                             std::vector<double>* finalT) {
    ReflStepArgs ra;
    ra.env = env;
    static const std::vector<double> noTarget;
    m_renderDepth = depth;
    m_renderT = finalT;
    const bool ok = StepImpl(cam, opt, noTarget, 0.0, 1.0, 1.0, err, nullptr, 0.0, shDegree,
                             0.0, env ? &ra : nullptr, rgb);
    m_renderDepth = nullptr;
    m_renderT = nullptr;
    return ok;
}

bool SplatTrainerGpu::StepImpl(const SplatCam& cam, const RasterOptions& opt,
                               const std::vector<double>& target, double lrMean,
                               double c1, double c2, std::string* err,
                               const std::vector<float>* depthTarget,
                               double depthWeight, int shDegree, double shLr,
                               const ReflStepArgs* ra, std::vector<double>* renderOut) {
    Impl& s = *m;
    if (!s.ready || s.n == 0) { *err = "trainer not started"; return false; }
    std::lock_guard<std::mutex> lock(s.ctx->SubmitMutex());

    const size_t n = s.n;
    const uint32_t groupsN = uint32_t((n + 255) / 256);
    const double* W = cam.R.m;
    std::vector<uint32_t> camC;
    for (int k = 0; k < 9; ++k) camC.push_back(FloatBits(float(W[k])));
    camC.push_back(FloatBits(float(cam.t.x)));
    camC.push_back(FloatBits(float(cam.t.y)));
    camC.push_back(FloatBits(float(cam.t.z)));

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
        c.push_back(FloatBits(float(cam.fx)));
        c.push_back(FloatBits(float(cam.fy)));
        c.push_back(FloatBits(float(cam.cx)));
        c.push_back(FloatBits(float(cam.cy)));
        c.push_back(uint32_t(cam.w));
        c.push_back(uint32_t(cam.h));
        c.push_back(FloatBits(float(opt.lowPass)));
        c.push_back(FloatBits(float(opt.minAlpha)));
        // The camera centre and degree, for the view-dependent colour. With
        // none on the device the state stands in for the unused binding.
        const int deg = s.haveSh ? std::clamp(shDegree, 0, kShMaxDegree) : 0;
        const Vec3 eye = CentreOf(cam);
        c.push_back(FloatBits(float(eye.x)));
        c.push_back(FloatBits(float(eye.y)));
        c.push_back(FloatBits(float(eye.z)));
        c.push_back(uint32_t(deg));
        const GpuImage* shIn = deg > 0 ? &s.shTex : &s.stateA;
        if (!s.Run(s.project, {&s.stateA, shIn}, {&s.proj, &s.screen}, c, groupsN, 1, err))
            return false;
    }
    // PHASES END WITH A SUBMISSION -- here, after the backward and after the
    // update -- so that each timer below measures its own phase; the other
    // phases end with a transfer, which submits anyway. Without these, a
    // phase's dispatches would run under whichever later phase first waited.
    if (!s.Sync()) { *err = "the projection did not complete"; return false; }
    m_time.project += MsSince(clk);

    // --- bin and sort, on the device -----------------------------------------
    //
    // See kBinCommon. Only the entry and visible counts come back, to size
    // what follows; everything else stays where the composite reads it.
    clk = Clock::now();
    const int tilesX = (cam.w + kTile - 1) / kTile;
    const int tilesY = (cam.h + kTile - 1) / kTile;
    const size_t nTiles = size_t(tilesX) * size_t(tilesY);
    constexpr size_t kBlock = 2048;   // kBinCommon's
    const size_t nbN = (n + kBlock - 1) / kBlock;
    const ImageDesc gmapD = FlatDesc(n + 1, Format::R32F);
    auto binC = [&](size_t nb, uint32_t pass, size_t e) {
        return std::vector<uint32_t>{uint32_t(n),      uint32_t(kTexW), uint32_t(tilesX),
                                     uint32_t(tilesY), uint32_t(nb),    pass,
                                     uint32_t(e),      uint32_t(nTiles)};
    };
    if (!s.Reserve(s.binLocal, n, Format::R32F) ||
        !s.Reserve(s.binSums, nbN, Format::RGBA32F) ||
        !s.Reserve(s.binSumsScan, nbN, Format::R32F) ||
        !s.Ensure(s.binTot, ImageDesc{1, 1, Format::RGBA32F}) || !s.Ensure(s.gmap, gmapD)) {
        *err = "could not allocate the binning textures";
        return false;
    }
    {
        const std::vector<uint32_t> c = binC(nbN, 0, 0);
        if (!s.Run(s.binScan, {&s.screen}, {&s.binLocal, &s.binSums}, c, uint32_t(nbN), 1,
                   err) ||
            !s.Run(s.binScanTop, {&s.binSums}, {&s.binSumsScan, &s.binTot}, c, 1, 1, err) ||
            !s.Run(s.binScanAdd, {&s.binLocal, &s.binSumsScan, &s.binTot}, {&s.gmap}, c,
                   uint32_t((n + 1 + 255) / 256), 1, err))
            return false;
    }
    if (!s.Get(s.binTot, 4)) {
        *err = "could not read back the entry count";
        return false;
    }
    const size_t entries = size_t(s.stage[0] + 0.5f);
    m_visible = int(s.stage[1] + 0.5f);
    if (entries >= (size_t(1) << 24)) {
        *err = "too many tile entries for the device (" + std::to_string(entries) + ")";
        return false;
    }

    const size_t nbE = (entries + kBlock - 1) / kBlock;
    if (!s.Reserve(s.list, entries, Format::R32F) || !s.Reserve(s.glist, entries, Format::R32F) ||
        !s.Reserve(s.offs, nTiles, Format::RGBA32F) ||
        !s.Reserve(s.elemA, entries, Format::RGBA32F) ||
        !s.Reserve(s.elemB, entries, Format::RGBA32F) ||
        !s.Reserve(s.table, 256 * nbE, Format::R32F) ||
        !s.Reserve(s.tscan, 256 * nbE, Format::R32F) || !s.Reserve(s.dtot, 256, Format::R32F)) {
        *err = "could not allocate the tile lists";
        return false;
    }
    GpuImage* sorted = &s.elemA;
    if (entries > 0) {
        if (!s.Run(s.binFill, {&s.screen, &s.gmap}, {&s.elemA}, binC(nbE, 0, entries), groupsN,
                   1, err))
            return false;
        // Four passes of depth, then as many of tile as its index needs.
        const int tilePasses = nTiles <= 256 ? 1 : (nTiles <= 65536 ? 2 : 3);
        GpuImage* dst = &s.elemB;
        for (int p = 0; p < 4 + tilePasses; ++p) {
            const std::vector<uint32_t> c = binC(nbE, uint32_t(p), entries);
            if (!s.Run(s.binHist, {sorted}, {&s.table}, c, uint32_t(nbE), 1, err) ||
                !s.Run(s.binRowScan, {&s.table}, {&s.tscan, &s.dtot}, c, 256, 1, err) ||
                !s.Run(s.binScatter, {sorted, &s.tscan, &s.dtot}, {dst}, c, uint32_t(nbE), 1,
                       err))
                return false;
            std::swap(sorted, dst);
        }
    }
    if (!s.Run(s.binFinish, {sorted}, {&s.list, &s.glist, &s.offs}, binC(nbE, 0, entries),
               uint32_t((std::max(entries, nTiles) + 255) / 256), 1, err))
        return false;
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
        FloatBits(float(opt.background.x)), FloatBits(float(opt.background.y)),
        FloatBits(float(opt.background.z)), FloatBits(float(opt.minAlpha)),
        FloatBits(float(opt.maxAlpha))};
    // REFLECTIONS IN THE SAME PASS (splat_reflect.h). Each Gaussian's
    // reflectivity and camera-facing normal go into `saved` (kPayload mode
    // 4), and the REFL build of the composite blends them with the very same
    // weights as colour into a second image, R and N. One composite and one
    // backward carry all seven channels, where three of each did before.
    const bool reflNow = ra && ra->env && ra->env->Valid() && s.haveRefl;
    const Vec3 eyeR = CentreOf(cam);
    const ImageDesc rnD{cam.w, cam.h, Format::RGBA32F};
    if (reflNow) {
        const ImageDesc savedD = FlatDesc(std::max<size_t>(1, n), Format::RGBA32F);
        if (!s.Ensure(s.saved, savedD) || !s.Ensure(s.rn, rnD)) {
            *err = "could not allocate the reflection targets";
            return false;
        }
        std::vector<uint32_t> c{uint32_t(n), uint32_t(kTexW), FloatBits(float(eyeR.x)),
                                FloatBits(float(eyeR.y)), FloatBits(float(eyeR.z)), 4u};
        if (!s.Run(s.payload, {&s.stateA, &s.reflTex}, {&s.proj, &s.saved}, c,
                   groupsN, 1, err))
            return false;
    }
    {
        std::vector<uint32_t> c{uint32_t(tilesX), uint32_t(kTexW)};
        c.insert(c.end(), bgC.begin(), bgC.end());
        c.push_back(FloatBits(float(opt.tStop)));
        const bool ok = reflNow
            ? s.Run(s.fwdR, {&s.proj, &s.list, &s.offs, &s.saved},
                    {&s.rgbt, &s.last, &s.depth, &s.rn}, c, uint32_t(tilesX),
                    uint32_t(tilesY), err)
            : s.Run(s.fwd, {&s.proj, &s.list, &s.offs}, {&s.rgbt, &s.last, &s.depth},
                    c, uint32_t(tilesX), uint32_t(tilesY), err);
        if (!ok) return false;
    }
    const size_t np = size_t(cam.w) * size_t(cam.h);
    const bool useDepth = depthTarget && depthWeight > 0.0;
    if (useDepth) {
        // Read first: Get() reuses the stage the colour lands in next.
        if (!s.Get(s.depth, np)) { *err = "could not read back the depth"; return false; }
        s.depthStage.swap(s.stage);
    }
    // With reflections the deferred shading runs here too, and what comes
    // back is the shaded image -- what is shown, so the loss below is taken
    // on it and needs no change of its own.
    if (reflNow) {
        if (!s.ShadeFwd(cam, *ra->env, err)) return false;
        if (!s.Get(s.shaded, np * 4)) { *err = "could not read back the render"; return false; }
    } else if (!s.Get(s.rgbt, np * 4)) {
        *err = "could not read back the render";
        return false;
    }
    m_time.composite += MsSince(clk);

    // RENDER ONLY: the picture, and nothing after it -- no loss, no backward,
    // no step.
    if (renderOut) {
        renderOut->resize(np * 3);
        for (size_t i = 0; i < np; ++i)
            for (int ch = 0; ch < 3; ++ch)
                (*renderOut)[i * 3 + size_t(ch)] = double(s.stage[i * 4 + size_t(ch)]);
        // Depth and coverage too, when asked for: what a depth score needs.
        if (m_renderT) {
            m_renderT->resize(np);
            for (size_t i = 0; i < np; ++i) (*m_renderT)[i] = double(s.stage[i * 4 + 3]);
        }
        if (m_renderDepth) {
            if (!s.Get(s.depth, np)) { *err = "could not read back the depth"; return false; }
            m_renderDepth->assign(s.stage.begin(), s.stage.begin() + long(np));
        }
        return true;
    }

    // --- loss: L1, as train_splats's CPU loop ------------------------------------
    clk = Clock::now();
    size_t usedPx = 0;   // measured pixels, for reflect_sparsity's mean
    {
        // A negative target is a masked pixel (train_splats' MaskUnmeasured):
        // no gradient, and not counted in the mean.
        size_t used = 0;
        for (size_t i = 0; i < np * 3; ++i) if (target[i] >= 0.0) ++used;
        usedPx = used / 3;
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
        // With reflections this is the gradient of the SHADED image, which
        // the shading's backward turns into the composite's.
        GpuImage& dUp = reflNow ? s.dshade : s.drgbt;
        if (!s.Ensure(dUp, rgbtD) || !s.Put(dUp, rgbtD)) {
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
    const ImageDesc eD  = FlatDesc(std::max<size_t>(1, entries * (reflNow ? 4 : 3)),
                                   Format::RGBA32F);
    const ImageDesc g2D = FlatDesc(std::max<size_t>(1, n * 3), Format::RGBA32F);
    if (!s.Ensure(s.egrad, eD) || !s.Ensure(s.g2, g2D)) {
        *err = "could not allocate the gradient textures";
        return false;
    }
    std::vector<uint32_t> bc{uint32_t(cam.w), uint32_t(cam.h), uint32_t(tilesX),
                             uint32_t(kTexW)};
    bc.insert(bc.end(), bgC.begin(), bgC.end());
    if (!reflNow) {
        if (!s.Run(s.bwd, {&s.proj, &s.list, &s.offs, &s.drgbt},
                   {&s.egrad, &s.last, &s.ddepth, &s.dshift},
                   bc, uint32_t(tilesX), uint32_t(tilesY), err))
            return false;
        if (!s.Run(s.sum, {&s.egrad, &s.gmap, &s.glist}, {&s.g2},
                   {uint32_t(n), uint32_t(kTexW)}, groupsN, 1, err))
            return false;
    } else {
        // The shading's gradient into the three maps (and the environment,
        // which the caller steps), then ONE backward over all seven channels:
        // colour and its depth term in DRGBT/DDepth as ever, reflectivity and
        // normal in DRN. The REFL sum adds each Gaussian's r and normal
        // gradients into RG for the reflection step.
        const ImageDesc rgD = FlatDesc(std::max<size_t>(1, n), Format::RGBA32F);
        if (!s.Ensure(s.rg, rgD)) {
            *err = "could not allocate the reflection gradients";
            return false;
        }
        const double sp = ra->sparsity > 0.0
                              ? ra->sparsity / double(std::max<size_t>(1, usedPx)) : 0.0;
        if (!s.ShadeBwd(cam, *ra->env, sp, ra->envGrad, err)) return false;
        if (!s.Run(s.bwdR, {&s.proj, &s.list, &s.offs, &s.drgbt, &s.saved, &s.drn},
                   {&s.egrad, &s.last, &s.ddepth, &s.dshift},
                   bc, uint32_t(tilesX), uint32_t(tilesY), err))
            return false;
        if (!s.Run(s.sumR, {&s.egrad, &s.gmap, &s.glist}, {&s.g2, &s.rg},
                   {uint32_t(n), uint32_t(kTexW)}, groupsN, 1, err))
            return false;
    }
    if (!s.Sync()) { *err = "the backward pass did not complete"; return false; }
    m_time.backward += MsSince(clk);

    // --- the chain rule and Adam ------------------------------------------------
    clk = Clock::now();
    // The view-dependent colour's step first: it also masks the colour
    // gradient where the shown colour sits at a clamp, which the update
    // below then applies to the base colour. See kShStep.
    if (s.haveSh && shDegree > 0) {
        const int deg = std::clamp(shDegree, 0, kShMaxDegree);
        const Vec3 eye = CentreOf(cam);
        std::vector<uint32_t> c{uint32_t(n), uint32_t(kTexW),
                                FloatBits(float(eye.x)), FloatBits(float(eye.y)), FloatBits(float(eye.z)),
                                uint32_t(deg), FloatBits(float(shLr)), FloatBits(float(c1)),
                                FloatBits(float(c2))};
        if (!s.Run(s.shStep, {&s.stateA, &s.proj}, {&s.g2, &s.shTex, &s.shMom}, c,
                   groupsN, 1, err))
            return false;
    }
    // Reflectivity and normal, from the gradients the REFL sum gathered.
    if (reflNow) {
        std::vector<uint32_t> c{uint32_t(n), uint32_t(kTexW),
                                FloatBits(float(eyeR.x)), FloatBits(float(eyeR.y)), FloatBits(float(eyeR.z)),
                                FloatBits(float(ra->lrRefl)), FloatBits(float(ra->lrNormal)),
                                FloatBits(float(c1)), FloatBits(float(c2))};
        if (!s.Run(s.reflStep, {&s.stateA, &s.proj, &s.rg}, {&s.reflTex, &s.reflMom}, c,
                   groupsN, 1, err))
            return false;
    }
    {
        std::vector<uint32_t> c{uint32_t(n), uint32_t(kTexW)};
        c.insert(c.end(), camC.begin(), camC.end());
        c.push_back(FloatBits(float(cam.fx)));
        c.push_back(FloatBits(float(cam.fy)));
        c.push_back(FloatBits(float(opt.lowPass)));
        c.push_back(FloatBits(float(lrMean)));
        c.push_back(FloatBits(float(c1)));
        c.push_back(FloatBits(float(c2)));
        c.push_back(FloatBits(float(0.5 * double(cam.w))));
        c.push_back(m_clampColour ? 1u : 0u);
        if (!s.Run(s.update, {&s.stateA, &s.g2, &s.proj}, {&s.stateB}, c,
                   groupsN, 1, err))
            return false;
    }
    if (!s.Sync()) { *err = "the update did not complete"; return false; }
    std::swap(s.stateA, s.stateB);
    m_time.update += MsSince(clk);
    return true;
}

}  // namespace tglab
