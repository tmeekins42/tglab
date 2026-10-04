#include "nn_gpu.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace tglab {
namespace nn {

namespace {

uint32_t F2U(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

uint32_t Groups(uint32_t n, uint32_t per) { return (n + per - 1) / per; }

// Output channels each thread of the dense convolution computes.
//
// Sixteen, so that every input value a thread loads feeds sixteen
// multiply-adds rather than one: a convolution here is bound by loads, not
// arithmetic. The weights for those sixteen channels are contiguous (see
// UploadConv), so they arrive as four float4 loads, and every thread in a
// group reads the same ones.
constexpr int kOcBlock = 16;

// The dense convolution: one thread per output pixel, kOcBlock output
// channels each, a group's z the block of channels.
const char* kConvHlsl = R"(
ByteAddressBuffer   In  : register(t0);
ByteAddressBuffer   Wt  : register(t1);   // [cout/16][cin][k*k][16]
ByteAddressBuffer   Bs  : register(t2);   // [cout rounded up to 16]
RWByteAddressBuffer Out : register(u0);

cbuffer P : register(b0) {
    uint W, H, Cin, Cout, K, OutOff, Relu;
};

[numthreads(8, 8, 1)]
void main(uint3 g : SV_GroupID, uint3 t : SV_GroupThreadID) {
    const uint x = g.x * 8 + t.x, y = g.y * 8 + t.y, ocb = g.z;
    if (x >= W || y >= H) return;

    float4 acc[4];
    [unroll] for (uint i = 0; i < 4; ++i) acc[i] = asfloat(Bs.Load4((ocb * 16 + i * 4) * 4));

    const int r = int(K) / 2;
    const uint P = W * H, KK = K * K;
    for (uint ic = 0; ic < Cin; ++ic) {
        const uint plane = ic * P;
        const uint wrow  = (ocb * Cin + ic) * KK * 16;
        for (uint ky = 0; ky < K; ++ky) {
            const int sy = int(y) + int(ky) - r;
            if (sy < 0 || sy >= int(H)) continue;
            for (uint kx = 0; kx < K; ++kx) {
                const int sx = int(x) + int(kx) - r;
                if (sx < 0 || sx >= int(W)) continue;
                const float v = asfloat(In.Load((plane + uint(sy) * W + uint(sx)) * 4));
                const uint wa = (wrow + (ky * K + kx) * 16) * 4;
                [unroll] for (uint i = 0; i < 4; ++i)
                    acc[i] += v * asfloat(Wt.Load4(wa + i * 16));
            }
        }
    }

    const uint pix = y * W + x;
    [unroll] for (uint i = 0; i < 4; ++i) {
        [unroll] for (uint j = 0; j < 4; ++j) {
            const uint oc = ocb * 16 + i * 4 + j;
            if (oc >= Cout) continue;
            float v = acc[i][j];
            if (Relu != 0) v = max(v, 0.0);
            Out.Store(((OutOff + oc) * P + pix) * 4, asuint(v));
        }
    }
}
)";

// The dense convolution again, TILED -- what actually runs for k <= 5.
//
// The direct kernel above reads every input value and every weight from
// memory, once per use: each thread reloads the same k x k neighbourhood its
// neighbours just loaded, and only the cache stands between that and the
// memory bus. Measured at a few percent of the card's arithmetic rate.
//
// Here a group of 64 threads owns an output tile of 32 x 8 pixels and 16
// channels. For each slab of 8 input channels it loads, cooperatively, the
// input tile with its k-1 border and the slab's weights into group-shared
// memory -- once -- and then every thread computes FOUR horizontally adjacent
// pixels for all 16 channels from there: per input row of k + 3 values and
// k x 4 float4 weights, 64 k multiply-adds. Compiled per K (1, 3, 5) so every
// loop unrolls and the accumulators stay in registers.
//
// The weights are the same [cout/16][cin][k*k][16] layout as the direct
// kernel, so a slab's are one contiguous run.
const char* kConvTiledHlsl = R"(
ByteAddressBuffer   In  : register(t0);
ByteAddressBuffer   Wt  : register(t1);
ByteAddressBuffer   Bs  : register(t2);
RWByteAddressBuffer Out : register(u0);

cbuffer P : register(b0) {
    uint W, H, Cin, Cout, OutOff, Relu;
};

#define TW  32          // tile width: 8 threads x 4 pixels
#define TH  8           // tile height: 8 threads x 1 row
#define ICB 8           // input channels per slab
#define R   (K / 2)
#define IW  (TW + 2 * R)
#define IH  (TH + 2 * R)
#define KK  (K * K)

groupshared float  sIn[ICB * IH * IW];
groupshared float4 sW[ICB * KK * 4];     // [ic][tap][16 channels as 4 float4]

[numthreads(8, 8, 1)]
void main(uint3 g : SV_GroupID, uint3 t : SV_GroupThreadID, uint gi : SV_GroupIndex) {
    const int tileX = int(g.x * TW), tileY = int(g.y * TH);
    const uint ocb = g.z;

    float4 acc[4][4];   // [pixel][channel quad]
    [unroll] for (uint q = 0; q < 4; ++q) {
        const float4 b = asfloat(Bs.Load4((ocb * 16 + q * 4) * 4));
        [unroll] for (uint p = 0; p < 4; ++p) acc[p][q] = b;
    }

    for (uint ic0 = 0; ic0 < Cin; ic0 += ICB) {
        const uint nic = min(uint(ICB), Cin - ic0);

        // The slab's input tile, border included; zero outside the image,
        // which is the convolution's padding.
        for (uint i = gi; i < ICB * IH * IW; i += 64) {
            const uint c = i / (IH * IW), rem = i % (IH * IW);
            const int sy = tileY + int(rem / IW) - R, sx = tileX + int(rem % IW) - R;
            float v = 0.0;
            if (c < nic && sy >= 0 && sy < int(H) && sx >= 0 && sx < int(W))
                v = asfloat(In.Load((((ic0 + c) * H + uint(sy)) * W + uint(sx)) * 4));
            sIn[i] = v;
        }
        // Its weights: nic * KK * 16 floats, contiguous in the global layout.
        const uint wBase = (ocb * Cin + ic0) * KK * 16;
        for (uint j = gi; j < ICB * KK * 4; j += 64)
            sW[j] = (j < nic * KK * 4) ? asfloat(Wt.Load4((wBase + j * 4) * 4)) : float4(0, 0, 0, 0);
        GroupMemoryBarrierWithGroupSync();

        for (uint c = 0; c < nic; ++c) {
            [unroll] for (uint ky = 0; ky < K; ++ky) {
                float row[4 + 2 * R];
                const uint rowBase = (c * IH + t.y + ky) * IW + t.x * 4;
                [unroll] for (uint j = 0; j < 4 + 2 * R; ++j) row[j] = sIn[rowBase + j];
                [unroll] for (uint kx = 0; kx < K; ++kx) {
                    const uint wi = (c * KK + ky * K + kx) * 4;
                    const float4 w0 = sW[wi], w1 = sW[wi + 1], w2 = sW[wi + 2], w3 = sW[wi + 3];
                    [unroll] for (uint p = 0; p < 4; ++p) {
                        const float v = row[p + kx];
                        acc[p][0] += v * w0;
                        acc[p][1] += v * w1;
                        acc[p][2] += v * w2;
                        acc[p][3] += v * w3;
                    }
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    const int y = tileY + int(t.y);
    if (y >= int(H)) return;
    const uint P = W * H;
    [unroll] for (uint p = 0; p < 4; ++p) {
        const int x = tileX + int(t.x * 4 + p);
        if (x >= int(W)) continue;
        const uint pix = uint(y) * W + uint(x);
        [unroll] for (uint q = 0; q < 4; ++q) {
            [unroll] for (uint j = 0; j < 4; ++j) {
                const uint oc = ocb * 16 + q * 4 + j;
                if (oc >= Cout) continue;
                float v = acc[p][q][j];
                if (Relu != 0) v = max(v, 0.0);
                Out.Store(((OutOff + oc) * P + pix) * 4, asuint(v));
            }
        }
    }
}
)";

// The 1x1 convolution as what it is, a MATRIX MULTIPLY:
//   out[oc][p] = b[oc] + sum over ic of w[oc][ic] * in[ic][p]
//
// The tiled kernel above treats 1x1 as a 3x3 without a border, and so loads
// each input value for only the 16 output channels of its group -- measured
// at a quarter of the 3x3 kernel's rate, and 1x1 convolutions are most of a
// DeDoDe refiner. Here a group of 64 threads owns a 64-channel by 64-pixel
// tile of the output and each thread an 8 x 8 block of it, so every value
// staged into group-shared memory feeds 64 channels (or pixels) and every
// pair of float4 reads per thread feeds 64 multiply-adds.
//
// Same weight layout as the others ([cout/16][cin][16]); pixel tiles past
// 32768 spill into the grid's z, since a 2048 x 2048 map has 65536 of them.
const char* kConv1x1Hlsl = R"(
ByteAddressBuffer   In  : register(t0);
ByteAddressBuffer   Wt  : register(t1);
ByteAddressBuffer   Bs  : register(t2);
RWByteAddressBuffer Out : register(u0);

cbuffer P : register(b0) {
    uint NPix, Cin, Cout, OutOff, Relu, OcPad;   // OcPad: cout rounded up to 16
};

#define TM 64   // output channels per group
#define TN 64   // pixels per group
#define TK 16   // input channels per slab

groupshared float4 sA[TK * TM / 4];   // weights, [k][m / 4]
groupshared float4 sB[TK * TN / 4];   // inputs,  [k][n / 4]

[numthreads(8, 8, 1)]
void main(uint3 g : SV_GroupID, uint3 t : SV_GroupThreadID, uint gi : SV_GroupIndex) {
    const uint pixBase = (g.z * 32768 + g.x) * TN, ocBase = g.y * TM;
    float acc[8][8];
    [unroll] for (uint i = 0; i < 8; ++i) {
        const uint oc = ocBase + t.y * 8 + i;
        const float b = oc < OcPad ? asfloat(Bs.Load(oc * 4)) : 0.0;
        [unroll] for (uint j = 0; j < 8; ++j) acc[i][j] = b;
    }

    for (uint ic0 = 0; ic0 < Cin; ic0 += TK) {
        // 16 x 64 weights and 16 x 64 inputs, four floats per thread per
        // matrix per pass, as float4 to group-shared memory.
        [unroll] for (uint r = 0; r < 4; ++r) {
            const uint e = gi + r * 64;            // 0..255, a float4 slot
            const uint k = e / (TM / 4), m4 = e % (TM / 4);
            const uint ic = ic0 + k;
            float4 a = float4(0, 0, 0, 0);
            if (ic < Cin) {
                [unroll] for (uint q = 0; q < 4; ++q) {
                    const uint oc = ocBase + m4 * 4 + q;
                    if (oc < OcPad) a[q] = asfloat(Wt.Load((((oc / 16) * Cin + ic) * 16 + oc % 16) * 4));
                }
            }
            sA[e] = a;
            float4 v = float4(0, 0, 0, 0);
            const uint p = pixBase + m4 * 4;
            if (ic < Cin) {
                if (p + 3 < NPix) {   // raw loads need only 4-byte alignment
                    v = asfloat(In.Load4((ic * NPix + p) * 4));
                } else {
                    [unroll] for (uint q = 0; q < 4; ++q)
                        if (p + q < NPix) v[q] = asfloat(In.Load((ic * NPix + p + q) * 4));
                }
            }
            sB[e] = v;
        }
        GroupMemoryBarrierWithGroupSync();

        [unroll] for (uint k = 0; k < TK; ++k) {
            const float4 a0 = sA[k * (TM / 4) + t.y * 2], a1 = sA[k * (TM / 4) + t.y * 2 + 1];
            const float4 b0 = sB[k * (TN / 4) + t.x * 2], b1 = sB[k * (TN / 4) + t.x * 2 + 1];
            const float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const float b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
            [unroll] for (uint i = 0; i < 8; ++i)
                [unroll] for (uint j = 0; j < 8; ++j) acc[i][j] += a[i] * b[j];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    [unroll] for (uint i = 0; i < 8; ++i) {
        const uint oc = ocBase + t.y * 8 + i;
        if (oc >= Cout) continue;
        [unroll] for (uint j = 0; j < 8; ++j) {
            const uint p = pixBase + t.x * 8 + j;
            if (p >= NPix) continue;
            float v = acc[i][j];
            if (Relu != 0) v = max(v, 0.0);
            Out.Store(((OutOff + oc) * NPix + p) * 4, asuint(v));
        }
    }
}
)";

// The k x k convolution as a matrix multiply too -- IMPLICIT GEMM:
//   out[oc][p] = b[oc] + sum over (ic, tap) of w[oc][ic][tap] * in[ic][p + tap]
// the 1x1 kernel's tiles and arithmetic exactly, with the inner dimension
// running over input channel AND tap, and the input slab GATHERED: row
// (ic, tap) of the 64-pixel tile is the input shifted by that tap, zero past
// the border. The weight layout [cout/16][cin][k*k][16] is already that
// matrix, cin*k*k long. Nothing is materialised; each input value is read
// once per tap through the cache.
//
// Measured on the DeDoDe descriptor at 784 x 784: the tiled kernel above ran
// its 3x3 convolutions -- 216 billion multiply-adds -- at about 6 TFLOPS,
// 71 ms; this runs them in 28 ms. The 1x1 convolutions use it too, with
// contiguous loads (KS 1). Whole networks, best of ten: the descriptor
// 125 -> 82 ms, the DaD detector at 1024 55 -> 39 ms, outputs unchanged
// against PyTorch. The tile shape (64 or 128 either way) made no measurable
// difference; what did was software pipelining, below. The tiled and 1x1
// kernels stay for TGLAB_NN_TILED, to time against.
const char* kConvGemmHlsl = R"(
ByteAddressBuffer   In  : register(t0);
ByteAddressBuffer   Wt  : register(t1);
ByteAddressBuffer   Bs  : register(t2);
RWByteAddressBuffer Out : register(u0);

cbuffer P : register(b0) {
    uint W, H, Cin, Cout, OutOff, Relu, OcPad;   // OcPad: cout rounded up to 16
};

// KS, TM and TN come from the compile: kernel size, and the output channels
// and pixels a group owns. Each thread owns an 8 x 8 block of them.
#define KK (KS * KS)
#define R  (KS / 2)
#define TK 16                        // rows of the inner dimension per slab
#define TX (TN / 8)                  // threads across pixels
#define TY (TM / 8)                  // threads across channels
#define NT (TX * TY)
#define SA (TK * TM / 4)             // float4 slots of a weight slab
#define SB (TK * TN / 4)             // float4 slots of an input slab
#define RA (SA / NT)                 // of them each thread loads
#define RB (SB / NT)

groupshared float4 sA[SA];   // weights, [k][m / 4]
groupshared float4 sB[SB];   // inputs,  [k][n / 4]

// One slab -- rows k0 .. k0 + TK of the inner dimension -- from memory into
// this thread's registers.
void Fetch(uint k0, uint gi, uint ocBase, uint pixBase, int py[RB], int px[RB],
           out float4 ra[RA], out float4 rb[RB]) {
    const uint NPix = W * H, CinK = Cin * KK;
    [unroll] for (uint r = 0; r < RA; ++r) {
        const uint e = gi + r * NT;
        const uint kr = k0 + e / (TM / 4), m4 = e % (TM / 4);
        float4 a = float4(0, 0, 0, 0);
        if (kr < CinK) {
            [unroll] for (uint q = 0; q < 4; ++q) {
                const uint oc = ocBase + m4 * 4 + q;
                if (oc < OcPad) a[q] = asfloat(Wt.Load((((oc / 16) * CinK + kr) * 16 + oc % 16) * 4));
            }
        }
        ra[r] = a;
    }
    [unroll] for (uint r = 0; r < RB; ++r) {
        const uint e = gi + r * NT;
        const uint kr = k0 + e / (TN / 4), n4 = e % (TN / 4);
        const uint p0 = pixBase + n4 * 4;
        float4 v = float4(0, 0, 0, 0);
        if (kr < CinK) {
#if KS == 1
            // A plain matrix: four neighbouring pixels are contiguous.
            if (p0 + 3 < NPix) {   // raw loads need only 4-byte alignment
                v = asfloat(In.Load4((kr * NPix + p0) * 4));
            } else {
                [unroll] for (uint q = 0; q < 4; ++q)
                    if (p0 + q < NPix) v[q] = asfloat(In.Load((kr * NPix + p0 + q) * 4));
            }
#else
            // Gathered: row (ic, tap) is the input shifted by the tap.
            const uint ic = kr / KK, tap = kr % KK;
            const int dy = int(tap / KS) - R, dx = int(tap % KS) - R;
            int y = py[r], x = px[r];
            [unroll] for (uint q = 0; q < 4; ++q) {
                const int sy = y + dy, sx = x + dx;
                if (p0 + q < NPix && sy >= 0 && sy < int(H) && sx >= 0 && sx < int(W))
                    v[q] = asfloat(In.Load(((ic * H + uint(sy)) * W + uint(sx)) * 4));
                if (++x == int(W)) { x = 0; ++y; }
            }
#endif
        }
        rb[r] = v;
    }
}

[numthreads(TX, TY, 1)]
void main(uint3 g : SV_GroupID, uint3 t : SV_GroupThreadID, uint gi : SV_GroupIndex) {
    const uint NPix = W * H, CinK = Cin * KK;
    const uint pixBase = (g.z * 32768 + g.x) * TN, ocBase = g.y * TM;
    float acc[8][8];
    [unroll] for (uint i = 0; i < 8; ++i) {
        const uint oc = ocBase + t.y * 8 + i;
        const float b = oc < OcPad ? asfloat(Bs.Load(oc * 4)) : 0.0;
        [unroll] for (uint j = 0; j < 8; ++j) acc[i][j] = b;
    }

    // Each thread loads the same pixels of every slab: where they are.
    int py[RB], px[RB];
    [unroll] for (uint r = 0; r < RB; ++r) {
        const uint p = pixBase + ((gi + r * NT) % (TN / 4)) * 4;
        py[r] = int(p / W);
        px[r] = int(p % W);
    }

    // SOFTWARE PIPELINED: the next slab comes from memory into registers
    // while this one is multiplied out of group-shared memory, so the loads'
    // latency -- the gathered input's especially -- hides behind arithmetic
    // instead of stalling the group between barriers.
    float4 ra[RA], rb[RB];
    Fetch(0, gi, ocBase, pixBase, py, px, ra, rb);
    for (uint k0 = 0; k0 < CinK; k0 += TK) {
        [unroll] for (uint r = 0; r < RA; ++r) sA[gi + r * NT] = ra[r];
        [unroll] for (uint r = 0; r < RB; ++r) sB[gi + r * NT] = rb[r];
        GroupMemoryBarrierWithGroupSync();
        if (k0 + TK < CinK) Fetch(k0 + TK, gi, ocBase, pixBase, py, px, ra, rb);

        [unroll] for (uint k = 0; k < TK; ++k) {
            const float4 a0 = sA[k * (TM / 4) + t.y * 2], a1 = sA[k * (TM / 4) + t.y * 2 + 1];
            const float4 b0 = sB[k * (TN / 4) + t.x * 2], b1 = sB[k * (TN / 4) + t.x * 2 + 1];
            const float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const float b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
            [unroll] for (uint i = 0; i < 8; ++i)
                [unroll] for (uint j = 0; j < 8; ++j) acc[i][j] += a[i] * b[j];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    [unroll] for (uint i = 0; i < 8; ++i) {
        const uint oc = ocBase + t.y * 8 + i;
        if (oc >= Cout) continue;
        [unroll] for (uint j = 0; j < 8; ++j) {
            const uint p = pixBase + t.x * 8 + j;
            if (p >= NPix) continue;
            float v = acc[i][j];
            if (Relu != 0) v = max(v, 0.0);
            Out.Store(((OutOff + oc) * NPix + p) * 4, asuint(v));
        }
    }
}
)";

// Depthwise: each channel convolved with its own k x k filter.
const char* kDepthwiseHlsl = R"(
ByteAddressBuffer   In  : register(t0);
ByteAddressBuffer   Wt  : register(t1);   // [c][k*k]
ByteAddressBuffer   Bs  : register(t2);   // [c]
RWByteAddressBuffer Out : register(u0);

cbuffer P : register(b0) {
    uint W, H, C, K, OutOff, Relu;
};

[numthreads(8, 8, 1)]
void main(uint3 g : SV_GroupID, uint3 t : SV_GroupThreadID) {
    const uint x = g.x * 8 + t.x, y = g.y * 8 + t.y, c = g.z;
    if (x >= W || y >= H) return;
    const int r = int(K) / 2;
    const uint P = W * H;
    float acc = asfloat(Bs.Load(c * 4));
    for (uint ky = 0; ky < K; ++ky) {
        const int sy = int(y) + int(ky) - r;
        if (sy < 0 || sy >= int(H)) continue;
        for (uint kx = 0; kx < K; ++kx) {
            const int sx = int(x) + int(kx) - r;
            if (sx < 0 || sx >= int(W)) continue;
            acc += asfloat(In.Load((c * P + uint(sy) * W + uint(sx)) * 4)) *
                   asfloat(Wt.Load((c * K * K + ky * K + kx) * 4));
        }
    }
    if (Relu != 0) acc = max(acc, 0.0);
    Out.Store(((OutOff + c) * P + y * W + x) * 4, asuint(acc));
}
)";

// Depthwise again, TILED: a group of 64 threads owns a 32 x 8 tile of one
// channel, loads it once with its border into group-shared memory, and each
// thread computes four neighbouring pixels from one row of K + 3 values, its
// K x K weights in registers. The plain kernel above reads all K x K
// neighbours of every pixel from memory -- 25 loads a pixel at 5 x 5, with
// only the cache between that and the bus -- and was 16 ms of the DeDoDe
// descriptor's 82. Compiled per K, so everything unrolls.
const char* kDepthwiseTiledHlsl = R"(
ByteAddressBuffer   In  : register(t0);
ByteAddressBuffer   Wt  : register(t1);   // [c][k*k]
ByteAddressBuffer   Bs  : register(t2);   // [c]
RWByteAddressBuffer Out : register(u0);

cbuffer P : register(b0) {
    uint W, H, C, Kunused, OutOff, Relu;
};

#define TW 32
#define TH 8
#define R  (K / 2)
#define IW (TW + 2 * R)
#define IH (TH + 2 * R)

groupshared float sIn[IH * IW];

[numthreads(8, 8, 1)]
void main(uint3 g : SV_GroupID, uint3 t : SV_GroupThreadID, uint gi : SV_GroupIndex) {
    const uint c = g.z;
    const int tileX = int(g.x * TW), tileY = int(g.y * TH);
    const uint P = W * H;
    for (uint i = gi; i < IH * IW; i += 64) {
        const int sy = tileY + int(i / IW) - R, sx = tileX + int(i % IW) - R;
        float v = 0.0;
        if (sy >= 0 && sy < int(H) && sx >= 0 && sx < int(W))
            v = asfloat(In.Load((c * P + uint(sy) * W + uint(sx)) * 4));
        sIn[i] = v;
    }
    float w[K * K];
    [unroll] for (uint j = 0; j < K * K; ++j) w[j] = asfloat(Wt.Load((c * K * K + j) * 4));
    const float b = asfloat(Bs.Load(c * 4));
    GroupMemoryBarrierWithGroupSync();

    float acc[4] = {b, b, b, b};
    [unroll] for (uint ky = 0; ky < K; ++ky) {
        float row[4 + 2 * R];
        const uint base = (t.y + ky) * IW + t.x * 4;
        [unroll] for (uint j = 0; j < 4 + 2 * R; ++j) row[j] = sIn[base + j];
        [unroll] for (uint kx = 0; kx < K; ++kx)
            [unroll] for (uint p = 0; p < 4; ++p) acc[p] += row[p + kx] * w[ky * K + kx];
    }

    const int y = tileY + int(t.y);
    if (y >= int(H)) return;
    [unroll] for (uint p = 0; p < 4; ++p) {
        const int x = tileX + int(t.x * 4 + p);
        if (x >= int(W)) continue;
        float v = acc[p];
        if (Relu != 0) v = max(v, 0.0);
        Out.Store(((OutOff + c) * P + uint(y) * W + uint(x)) * 4, asuint(v));
    }
}
)";

const char* kPoolHlsl = R"(
ByteAddressBuffer   In  : register(t0);
RWByteAddressBuffer Out : register(u0);
cbuffer P : register(b0) { uint IW, IH, OW, OH; };

[numthreads(8, 8, 1)]
void main(uint3 g : SV_GroupID, uint3 t : SV_GroupThreadID) {
    const uint x = g.x * 8 + t.x, y = g.y * 8 + t.y, c = g.z;
    if (x >= OW || y >= OH) return;
    const uint s = (c * IH + 2 * y) * IW + 2 * x;
    const float a = asfloat(In.Load(s * 4)),        b = asfloat(In.Load((s + 1) * 4));
    const float d = asfloat(In.Load((s + IW) * 4)), e = asfloat(In.Load((s + IW + 1) * 4));
    Out.Store(((c * OH + y) * OW + x) * 4, asuint(max(max(a, b), max(d, e))));
}
)";

// Both interpolations, transcribed from nn.cpp's CpuResize: see there.
const char* kResizeHlsl = R"(
ByteAddressBuffer   In  : register(t0);
RWByteAddressBuffer Out : register(u0);
cbuffer P : register(b0) { uint IW, IH, OW, OH, InOff, OutOff, Bicubic; };

float At(uint c, int y, int x) {
    y = clamp(y, 0, int(IH) - 1);
    x = clamp(x, 0, int(IW) - 1);
    return asfloat(In.Load((((InOff + c) * IH + uint(y)) * IW + uint(x)) * 4));
}
float Cubic1(float x, float A) { return ((A + 2) * x - (A + 3)) * x * x + 1; }
float Cubic2(float x, float A) { return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A; }
void Coeffs(float t, out float c[4]) {
    const float A = -0.75;
    c[0] = Cubic2(t + 1.0, A);
    c[1] = Cubic1(t, A);
    c[2] = Cubic1(1.0 - t, A);
    c[3] = Cubic2(2.0 - t, A);
}

[numthreads(8, 8, 1)]
void main(uint3 g : SV_GroupID, uint3 t : SV_GroupThreadID) {
    const uint x = g.x * 8 + t.x, y = g.y * 8 + t.y, c = g.z;
    if (x >= OW || y >= OH) return;
    const float sy = float(IH) / float(OH), sx = float(IW) / float(OW);
    float v;
    if (Bicubic == 0) {
        const float fy = max(0.0, sy * (float(y) + 0.5) - 0.5);
        const float fx = max(0.0, sx * (float(x) + 0.5) - 0.5);
        const int y0 = int(fy), x0 = int(fx);
        const int y1 = y0 + (y0 < int(IH) - 1 ? 1 : 0), x1 = x0 + (x0 < int(IW) - 1 ? 1 : 0);
        const float ly = fy - float(y0), lx = fx - float(x0);
        v = (1 - ly) * ((1 - lx) * At(c, y0, x0) + lx * At(c, y0, x1)) +
            ly * ((1 - lx) * At(c, y1, x0) + lx * At(c, y1, x1));
    } else {
        const float fy = sy * (float(y) + 0.5) - 0.5;
        const float fx = sx * (float(x) + 0.5) - 0.5;
        const int y0 = int(floor(fy)), x0 = int(floor(fx));
        float cy[4], cx[4];
        Coeffs(fy - float(y0), cy);
        Coeffs(fx - float(x0), cx);
        v = 0.0;
        [unroll] for (int j = 0; j < 4; ++j) {
            float row = 0.0;
            [unroll] for (int i = 0; i < 4; ++i) row += cx[i] * At(c, y0 - 1 + j, x0 - 1 + i);
            v += cy[j] * row;
        }
    }
    Out.Store((((OutOff + c) * OH + y) * OW + x) * 4, asuint(v));
}
)";

// Copy or accumulate a channel range; also (a + b) * s when Mode is 2.
const char* kElementHlsl = R"(
ByteAddressBuffer   A   : register(t0);
ByteAddressBuffer   B   : register(t1);
RWByteAddressBuffer Out : register(u0);
cbuffer P : register(b0) { uint W, H, SrcOff, DstOff, Mode; float S; };

[numthreads(8, 8, 1)]
void main(uint3 g : SV_GroupID, uint3 t : SV_GroupThreadID) {
    const uint x = g.x * 8 + t.x, y = g.y * 8 + t.y, c = g.z;
    if (x >= W || y >= H) return;
    const uint pix = y * W + x, P = W * H;
    const uint s = ((SrcOff + c) * P + pix) * 4, d = ((DstOff + c) * P + pix) * 4;
    const float a = asfloat(A.Load(s));
    float v;
    if (Mode == 0)      v = a;
    else if (Mode == 1) v = asfloat(Out.Load(d)) + a;
    else                v = (a + asfloat(B.Load(s))) * S;
    Out.Store(d, asuint(v));
}
)";

// grid_sample at points, one thread per (point, channel): see the CPU path.
const char* kSampleHlsl = R"(
ByteAddressBuffer   Map : register(t0);
ByteAddressBuffer   Pts : register(t1);
RWByteAddressBuffer Out : register(u0);
cbuffer P : register(b0) { uint W, H, C, N; };

float At(uint c, int y, int x) {
    if (x < 0 || x >= int(W) || y < 0 || y >= int(H)) return 0.0;
    return asfloat(Map.Load(((c * H + uint(y)) * W + uint(x)) * 4));
}

[numthreads(64, 1, 1)]
void main(uint3 g : SV_GroupID, uint3 t : SV_GroupThreadID) {
    const uint c = g.x * 64 + t.x, n = g.y;
    if (c >= C || n >= N) return;
    const float2 p = asfloat(Pts.Load2(n * 8));
    const float ix = ((p.x + 1.0) * float(W) - 1.0) * 0.5;
    const float iy = ((p.y + 1.0) * float(H) - 1.0) * 0.5;
    const int x0 = int(floor(ix)), y0 = int(floor(iy));
    const float fx = ix - float(x0), fy = iy - float(y0);
    const float v = (1 - fx) * (1 - fy) * At(c, y0, x0) + fx * (1 - fy) * At(c, y0, x0 + 1) +
                    (1 - fx) * fy * At(c, y0 + 1, x0) + fx * fy * At(c, y0 + 1, x0 + 1);
    Out.Store((n * C + c) * 4, asuint(v));
}
)";

// --- transformer kernels -------------------------------------------------------

// GELU(LayerNorm) across each point's channels: one thread per point. Two
// passes for the variance, as PyTorch's reduction is effectively exact.
// HLSL has no erf, so the exact GELU uses Abramowitz & Stegun 7.1.26 (error
// below 1.5e-7, under float's own rounding at these magnitudes).
const char* kLayerNormGeluHlsl = R"(
ByteAddressBuffer   In  : register(t0);
ByteAddressBuffer   G   : register(t1);
ByteAddressBuffer   B   : register(t2);
RWByteAddressBuffer Out : register(u0);
cbuffer P : register(b0) { uint C, NP; };

float Erf(float x) {
    const float t = 1.0 / (1.0 + 0.3275911 * abs(x));
    const float y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                            - 0.284496736) * t + 0.254829592) * t * exp(-x * x);
    return x < 0.0 ? -y : y;
}

[numthreads(64, 1, 1)]
void main(uint3 g : SV_GroupID, uint gi : SV_GroupIndex) {
    const uint p = (g.y * 65535 + g.x) * 64 + gi;
    if (p >= NP) return;
    float mean = 0.0;
    for (uint c = 0; c < C; ++c) mean += asfloat(In.Load((c * NP + p) * 4));
    mean /= float(C);
    float var = 0.0;
    for (uint c = 0; c < C; ++c) {
        const float d = asfloat(In.Load((c * NP + p) * 4)) - mean;
        var += d * d;
    }
    const float inv = rsqrt(var / float(C) + 1e-5);
    for (uint c = 0; c < C; ++c) {
        const float x = (asfloat(In.Load((c * NP + p) * 4)) - mean) * inv *
                        asfloat(G.Load(c * 4)) + asfloat(B.Load(c * 4));
        Out.Store((c * NP + p) * 4, asuint(0.5 * x * (1.0 + Erf(x * 0.70710678))));
    }
}
)";

// Rotary encoding in place: one thread per (point, channel pair), all heads.
const char* kRotaryHlsl = R"(
ByteAddressBuffer   Pos : register(t0);
ByteAddressBuffer   Fr  : register(t1);
RWByteAddressBuffer T   : register(u0);
cbuffer P : register(b0) { uint NP, Off, Count, HeadDim; };

[numthreads(64, 1, 1)]
void main(uint3 g : SV_GroupID, uint gi : SV_GroupIndex) {
    const uint p = g.x * 64 + gi, j = g.y;
    if (p >= NP) return;
    const float x = asfloat(Pos.Load(p * 4)), y = asfloat(Pos.Load((NP + p) * 4));
    const float a = asfloat(Fr.Load((2 * j) * 4)) * x + asfloat(Fr.Load((2 * j + 1) * 4)) * y;
    float sn, cs;
    sincos(a, sn, cs);
    for (uint c = Off + 2 * j; c < Off + Count; c += HeadDim) {
        const float u = asfloat(T.Load((c * NP + p) * 4));
        const float v = asfloat(T.Load(((c + 1) * NP + p) * 4));
        T.Store((c * NP + p) * 4, asuint(u * cs - v * sn));
        T.Store(((c + 1) * NP + p) * 4, asuint(v * cs + u * sn));
    }
}
)";

// Attention, never materialising the score matrix: one thread per query
// point and head, its query and running output in registers; the keys and
// values stream through group-shared memory TK at a time, and the softmax is
// kept ONLINE -- a running maximum and sum, the output rescaled whenever the
// maximum rises -- so memory is linear in the points, not quadratic.
const char* kAttentionHlsl = R"(
ByteAddressBuffer   Qb  : register(t0);
ByteAddressBuffer   Kb  : register(t1);
ByteAddressBuffer   Vb  : register(t2);
RWByteAddressBuffer Out : register(u0);
cbuffer P : register(b0) { uint NQ, NK, QOff, KOff, VOff, OOff; float Scale; };

#define HD HEADDIM
#define TK 16

groupshared float sK[TK * HD];   // [key][channel]
groupshared float sV[TK * HD];

[numthreads(64, 1, 1)]
void main(uint3 g : SV_GroupID, uint gi : SV_GroupIndex) {
    const uint i = g.x * 64 + gi, h = g.y;
    const bool live = i < NQ;
    float q[HD], o[HD];
    [unroll] for (uint c = 0; c < HD; ++c) {
        q[c] = live ? asfloat(Qb.Load(((QOff + h * HD + c) * NQ + i) * 4)) * Scale : 0.0;
        o[c] = 0.0;
    }
    float m = -1e30, l = 0.0;

    for (uint j0 = 0; j0 < NK; j0 += TK) {
        for (uint e = gi; e < TK * HD; e += 64) {
            const uint c = e / TK, t = e % TK, j = j0 + t;
            sK[t * HD + c] = j < NK ? asfloat(Kb.Load(((KOff + h * HD + c) * NK + j) * 4)) : 0.0;
            sV[t * HD + c] = j < NK ? asfloat(Vb.Load(((VOff + h * HD + c) * NK + j) * 4)) : 0.0;
        }
        GroupMemoryBarrierWithGroupSync();

        const uint cnt = min(uint(TK), NK - j0);
        float s[TK];
        float tmax = -1e30;
        [unroll] for (uint t = 0; t < TK; ++t) {
            float d = 0.0;
            [unroll] for (uint c = 0; c < HD; ++c) d += q[c] * sK[t * HD + c];
            s[t] = t < cnt ? d : -1e30;
            tmax = max(tmax, s[t]);
        }
        const float mNew = max(m, tmax), corr = exp(m - mNew);
        l *= corr;
        [unroll] for (uint c = 0; c < HD; ++c) o[c] *= corr;
        [unroll] for (uint t = 0; t < TK; ++t) {
            const float p = t < cnt ? exp(s[t] - mNew) : 0.0;
            l += p;
            [unroll] for (uint c = 0; c < HD; ++c) o[c] += p * sV[t * HD + c];
        }
        m = mNew;
        GroupMemoryBarrierWithGroupSync();
    }

    if (!live) return;
    const float inv = 1.0 / l;
    [unroll] for (uint c = 0; c < HD; ++c)
        Out.Store(((OOff + h * HD + c) * NQ + i) * 4, asuint(o[c] * inv));
}
)";

// C (m x n) = scale * A^T B for channel-major point sets A (k x m) and B
// (k x n): the 1x1 kernel's tiling with both operands read from tensors.
const char* kMatMulTNHlsl = R"(
ByteAddressBuffer   A : register(t0);
ByteAddressBuffer   B : register(t1);
RWByteAddressBuffer C : register(u0);
cbuffer P : register(b0) { uint M, N, K; float Scale; };

#define TM 64
#define TN 64
#define TK 16

groupshared float4 sA[TK * TM / 4];
groupshared float4 sB[TK * TN / 4];

float4 Load4(ByteAddressBuffer buf, uint row, uint col, uint width) {
    float4 v = float4(0, 0, 0, 0);
    if (col + 3 < width) return asfloat(buf.Load4((row * width + col) * 4));
    [unroll] for (uint q = 0; q < 4; ++q)
        if (col + q < width) v[q] = asfloat(buf.Load((row * width + col + q) * 4));
    return v;
}

[numthreads(8, 8, 1)]
void main(uint3 g : SV_GroupID, uint3 t : SV_GroupThreadID, uint gi : SV_GroupIndex) {
    const uint jBase = g.x * TN, iBase = g.y * TM;
    float acc[8][8];
    [unroll] for (uint a = 0; a < 8; ++a)
        [unroll] for (uint b = 0; b < 8; ++b) acc[a][b] = 0.0;

    for (uint k0 = 0; k0 < K; k0 += TK) {
        [unroll] for (uint r = 0; r < 4; ++r) {
            const uint e = gi + r * 64, k = e / (TM / 4), c4 = (e % (TM / 4)) * 4;
            const bool valid = k0 + k < K;
            sA[e] = valid ? Load4(A, k0 + k, iBase + c4, M) : float4(0, 0, 0, 0);
            sB[e] = valid ? Load4(B, k0 + k, jBase + c4, N) : float4(0, 0, 0, 0);
        }
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint k = 0; k < TK; ++k) {
            const float4 a0 = sA[k * (TM / 4) + t.y * 2], a1 = sA[k * (TM / 4) + t.y * 2 + 1];
            const float4 b0 = sB[k * (TN / 4) + t.x * 2], b1 = sB[k * (TN / 4) + t.x * 2 + 1];
            const float av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const float bv[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
            [unroll] for (uint a = 0; a < 8; ++a)
                [unroll] for (uint b = 0; b < 8; ++b) acc[a][b] += av[a] * bv[b];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    [unroll] for (uint a = 0; a < 8; ++a) {
        const uint i = iBase + t.y * 8 + a;
        if (i >= M) continue;
        [unroll] for (uint b = 0; b < 8; ++b) {
            const uint j = jBase + t.x * 8 + b;
            if (j < N) C.Store((i * N + j) * 4, asuint(acc[a][b] * Scale));
        }
    }
}
)";

// The dual softmax's pieces, in four passes over sim (m x n):
//   ROW (Mode 0): a group per row, its log-sum-exp (rowLse).
//   COL (Mode 1): a thread per column, its log-sum-exp (colLse).
//   ROWBEST (Mode 2): a group per row, the j maximising 2 s - colLse[j],
//            lowest j on a tie, and P there.
//   COLBEST (Mode 3): a thread per column, the i maximising 2 s - rowLse[i].
// Rows are reduced across a group because a row is contiguous -- 64 threads
// read it coalesced -- and columns per thread because adjacent threads then
// read adjacent columns of the same row.
const char* kDualSoftmaxHlsl = R"(
ByteAddressBuffer   S      : register(t0);
ByteAddressBuffer   RowLse : register(t1);
ByteAddressBuffer   ColLse : register(t2);
RWByteAddressBuffer Out    : register(u0);   // per mode: rowLse, colLse, best0 (idx, p), best1
cbuffer P : register(b0) { uint M, N, Mode; };

groupshared float sVal[64];
groupshared uint  sIdx[64];

float At(uint i, uint j) { return asfloat(S.Load((i * N + j) * 4)); }

[numthreads(64, 1, 1)]
void main(uint3 g : SV_GroupID, uint gi : SV_GroupIndex) {
    if (Mode == 1 || Mode == 3) {
        const uint j = g.x * 64 + gi;
        if (j >= N) return;
        if (Mode == 1) {
            float mx = -1e30;
            for (uint i = 0; i < M; ++i) mx = max(mx, At(i, j));
            float s = 0.0;
            for (uint i = 0; i < M; ++i) s += exp(At(i, j) - mx);
            Out.Store(j * 4, asuint(mx + log(s)));
        } else {
            float bv = -1e30;
            uint bi = 0;
            for (uint i = 0; i < M; ++i) {
                const float e = 2.0 * At(i, j) - asfloat(RowLse.Load(i * 4));
                if (e > bv) { bv = e; bi = i; }
            }
            Out.Store(j * 4, bi);
        }
        return;
    }

    const uint i = g.y * 65535 + g.x;   // a group per row
    if (i >= M) return;   // uniform across the group
    if (Mode == 0) {
        float mx = -1e30;
        for (uint j = gi; j < N; j += 64) mx = max(mx, At(i, j));
        sVal[gi] = mx;
        GroupMemoryBarrierWithGroupSync();
        for (uint w = 32; w > 0; w >>= 1) {
            if (gi < w) sVal[gi] = max(sVal[gi], sVal[gi + w]);
            GroupMemoryBarrierWithGroupSync();
        }
        mx = sVal[0];
        GroupMemoryBarrierWithGroupSync();
        float s = 0.0;
        for (uint j = gi; j < N; j += 64) s += exp(At(i, j) - mx);
        sVal[gi] = s;
        GroupMemoryBarrierWithGroupSync();
        for (uint w = 32; w > 0; w >>= 1) {
            if (gi < w) sVal[gi] += sVal[gi + w];
            GroupMemoryBarrierWithGroupSync();
        }
        if (gi == 0) Out.Store(i * 4, asuint(mx + log(sVal[0])));
    } else {
        float bv = -1e30;
        uint bi = 0xFFFFFFFF;
        for (uint j = gi; j < N; j += 64) {
            const float e = 2.0 * At(i, j) - asfloat(ColLse.Load(j * 4));
            if (e > bv) { bv = e; bi = j; }
        }
        sVal[gi] = bv;
        sIdx[gi] = bi;
        GroupMemoryBarrierWithGroupSync();
        for (uint w = 32; w > 0; w >>= 1) {
            if (gi < w) {
                const float ov = sVal[gi + w];
                const uint oi = sIdx[gi + w];
                if (ov > sVal[gi] || (ov == sVal[gi] && oi < sIdx[gi])) { sVal[gi] = ov; sIdx[gi] = oi; }
            }
            GroupMemoryBarrierWithGroupSync();
        }
        if (gi == 0) {
            Out.Store(i * 8, sIdx[0]);
            Out.Store(i * 8 + 4, asuint(exp(sVal[0] - asfloat(RowLse.Load(i * 4)))));
        }
    }
}
)";

} // namespace

struct GpuKernels::Impl {
    ComputeContext* gpu = nullptr;
    // From the context's SharedKernel cache: compiled once per device, not
    // once per engine -- engines are made per frame.
    const ComputeKernel *conv = nullptr, *depthwise = nullptr, *pool = nullptr,
                        *resize = nullptr, *element = nullptr, *sample = nullptr,
                        *conv1x1 = nullptr;
    const ComputeKernel* tiled[6] = {};   // kConvTiledHlsl per odd K; [1], [3], [5] used

    // Set by TGLAB_NN_DIRECT, to time or bisect against the direct kernel.
    bool forceDirect = GetEnvironmentVariableA("TGLAB_NN_DIRECT", nullptr, 0) > 0;
    // And TGLAB_NN_TILED, the tiled kernel instead of the implicit GEMM.
    bool forceTiled = GetEnvironmentVariableA("TGLAB_NN_TILED", nullptr, 0) > 0;
    // kConvGemmHlsl by [K][TM is 128][TN is 128].
    const ComputeKernel* gemm[6][2][2] = {};
    const ComputeKernel* dwTiled[8] = {};   // kDepthwiseTiledHlsl per odd K
    int tmOverride = EnvInt("TGLAB_NN_TM"), tnOverride = EnvInt("TGLAB_NN_TN");

    static int EnvInt(const char* name) {
        char buf[16] = {};
        return GetEnvironmentVariableA(name, buf, sizeof(buf)) > 0 ? std::atoi(buf) : 0;
    }

    // A dense k x k convolution, k odd and at most 5, by implicit GEMM; see
    // kConvGemmHlsl. The tile: 128 output channels by 128 pixels, each of
    // 256 threads an 8 x 8 block, unless the layer has only 64 channels to
    // give, when half the tile would compute nothing.
    bool Gemm(const Tensor& in, const Conv& cv, bool relu, Tensor* out, int outOff,
              std::string* err) {
        const int tm = tmOverride == 64 || tmOverride == 128 ? tmOverride : (cv.cout <= 64 ? 64 : 128);
        const int tn = tnOverride == 64 || tnOverride == 128 ? tnOverride : 128;
        const ComputeKernel*& k = gemm[cv.k][tm == 128][tn == 128];
        if (!k) {
            const std::string src = "#define KS " + std::to_string(cv.k) + "\n#define TM " +
                                    std::to_string(tm) + "\n#define TN " + std::to_string(tn) +
                                    "\n" + kConvGemmHlsl;
            const std::string name = "nn_conv_gemm_k" + std::to_string(cv.k) + "_" +
                                     std::to_string(tm) + "x" + std::to_string(tn);
            if (!Kernel(&k, src, name, err)) return false;
        }
        const uint32_t W = uint32_t(in.w), H = uint32_t(in.h);
        const uint32_t tiles = Groups(W * H, uint32_t(tn));
        return gpu->DispatchBuffers(
            *k, {&in.gpu, &cv.gw, &cv.gb}, {&out->gpu},
            {W, H, uint32_t(cv.cin), uint32_t(cv.cout), uint32_t(outOff), relu ? 1u : 0u,
             Groups(uint32_t(cv.cout), kOcBlock) * kOcBlock},
            std::min(tiles, 32768u), Groups(uint32_t(cv.cout), uint32_t(tm)), Groups(tiles, 32768), err);
    }

    bool Kernel(const ComputeKernel** k, const std::string& src, const std::string& name,
                std::string* err) {
        if (*k) return true;
        std::string e;
        *k = gpu->SharedKernel(name, src, &e);
        if (!*k) *err = name + ": " + e;
        return *k != nullptr;
    }

    const ComputeKernel *layerNorm = nullptr, *rotary = nullptr, *matmul = nullptr,
                        *dualSoftmax = nullptr, *attention[257] = {};   // by head dim

    // A parameter vector's device copy, made once.
    bool UploadVec(const Vec& v, std::string* err) {
        if (v.g.Valid()) return true;
        if (!gpu->CreateBuffer(v.v.size() * 4, &v.g) ||
            !gpu->UploadBuffer(v.v.data(), v.v.size() * 4, &v.g)) {
            v.g.Release();
            *err = "could not upload a parameter vector";
            return false;
        }
        return true;
    }

    // The weights in the layout kConvHlsl reads, uploaded once per Conv.
    bool UploadConv(const Conv& cv, std::string* err) {
        if (cv.gw.Valid()) return true;
        std::vector<float> w, b;
        if (cv.groups == 1) {
            const int blocks = (cv.cout + kOcBlock - 1) / kOcBlock, kk = cv.k * cv.k;
            w.assign(size_t(blocks) * size_t(cv.cin) * size_t(kk) * kOcBlock, 0.0f);
            for (int oc = 0; oc < cv.cout; ++oc)
                for (int ic = 0; ic < cv.cin; ++ic)
                    for (int t = 0; t < kk; ++t)
                        w[((size_t(oc / kOcBlock) * size_t(cv.cin) + size_t(ic)) * size_t(kk) +
                           size_t(t)) * kOcBlock + size_t(oc % kOcBlock)] =
                            cv.w[(size_t(oc) * size_t(cv.cin) + size_t(ic)) * size_t(kk) + size_t(t)];
            b.assign(size_t(blocks) * kOcBlock, 0.0f);
        } else {
            w = cv.w;
            b.assign(size_t(cv.cout), 0.0f);
        }
        std::copy(cv.b.begin(), cv.b.end(), b.begin());
        if (!gpu->CreateBuffer(w.size() * 4, &cv.gw) || !gpu->CreateBuffer(b.size() * 4, &cv.gb) ||
            !gpu->UploadBuffer(w.data(), w.size() * 4, &cv.gw) ||
            !gpu->UploadBuffer(b.data(), b.size() * 4, &cv.gb)) {
            cv.gw.Release();
            *err = "could not upload convolution weights";
            return false;
        }
        return true;
    }
};

GpuKernels::GpuKernels(ComputeContext* gpu) : m(new Impl) { m->gpu = gpu; }
GpuKernels::~GpuKernels() { delete m; }

bool GpuKernels::Conv2d(const Tensor& in, const Conv& cv, bool relu, Tensor* out, int outOff,
                        std::string* err) {
    if (!m->UploadConv(cv, err)) return false;
    const uint32_t W = uint32_t(in.w), H = uint32_t(in.h);
    if (cv.groups == 1 && cv.k <= 5 && (cv.k & 1) && !m->forceDirect && !m->forceTiled)
        return m->Gemm(in, cv, relu, out, outOff, err);
    if (cv.groups == 1 && cv.k == 1 && !m->forceDirect) {
        if (!m->Kernel(&m->conv1x1, kConv1x1Hlsl, "nn_conv1x1", err)) return false;
        const uint32_t tiles = Groups(W * H, 64);
        return m->gpu->DispatchBuffers(
            *m->conv1x1, {&in.gpu, &cv.gw, &cv.gb}, {&out->gpu},
            {W * H, uint32_t(cv.cin), uint32_t(cv.cout), uint32_t(outOff), relu ? 1u : 0u,
             Groups(uint32_t(cv.cout), kOcBlock) * kOcBlock},
            std::min(tiles, 32768u), Groups(uint32_t(cv.cout), 64), Groups(tiles, 32768), err);
    }
    if (cv.groups == 1 && cv.k <= 5 && (cv.k & 1) && !m->forceDirect) {
        const ComputeKernel*& k = m->tiled[cv.k];
        if (!k) {
            const std::string src = "#define K " + std::to_string(cv.k) + "\n" + kConvTiledHlsl;
            if (!m->Kernel(&k, src, "nn_conv_tiled_k" + std::to_string(cv.k), err)) return false;
        }
        return m->gpu->DispatchBuffers(
            *k, {&in.gpu, &cv.gw, &cv.gb}, {&out->gpu},
            {W, H, uint32_t(cv.cin), uint32_t(cv.cout), uint32_t(outOff), relu ? 1u : 0u},
            Groups(W, 32), Groups(H, 8), Groups(uint32_t(cv.cout), kOcBlock), err);
    }
    if (cv.groups == 1) {
        if (!m->Kernel(&m->conv, kConvHlsl, "nn_conv", err)) return false;
        return m->gpu->DispatchBuffers(
            *m->conv, {&in.gpu, &cv.gw, &cv.gb}, {&out->gpu},
            {W, H, uint32_t(cv.cin), uint32_t(cv.cout), uint32_t(cv.k), uint32_t(outOff),
             relu ? 1u : 0u},
            Groups(W, 8), Groups(H, 8), Groups(uint32_t(cv.cout), kOcBlock), err);
    }
    if (cv.groups != cv.cin || cv.cin != cv.cout) {
        *err = "only dense and depthwise convolutions are supported";
        return false;
    }
    if (cv.k <= 7 && (cv.k & 1) && !m->forceDirect && !m->forceTiled) {
        const ComputeKernel*& k = m->dwTiled[cv.k];
        if (!k) {
            const std::string src = "#define K " + std::to_string(cv.k) + "\n" + kDepthwiseTiledHlsl;
            if (!m->Kernel(&k, src, "nn_depthwise_tiled_k" + std::to_string(cv.k), err)) return false;
        }
        return m->gpu->DispatchBuffers(
            *k, {&in.gpu, &cv.gw, &cv.gb}, {&out->gpu},
            {W, H, uint32_t(cv.cout), uint32_t(cv.k), uint32_t(outOff), relu ? 1u : 0u},
            Groups(W, 32), Groups(H, 8), uint32_t(cv.cout), err);
    }
    if (!m->Kernel(&m->depthwise, kDepthwiseHlsl, "nn_depthwise", err)) return false;
    return m->gpu->DispatchBuffers(
        *m->depthwise, {&in.gpu, &cv.gw, &cv.gb}, {&out->gpu},
        {W, H, uint32_t(cv.cout), uint32_t(cv.k), uint32_t(outOff), relu ? 1u : 0u},
        Groups(W, 8), Groups(H, 8), uint32_t(cv.cout), err);
}

bool GpuKernels::MaxPool2(const Tensor& in, Tensor* out, std::string* err) {
    if (!m->Kernel(&m->pool, kPoolHlsl, "nn_pool", err)) return false;
    return m->gpu->DispatchBuffers(
        *m->pool, {&in.gpu}, {&out->gpu},
        {uint32_t(in.w), uint32_t(in.h), uint32_t(out->w), uint32_t(out->h)},
        Groups(uint32_t(out->w), 8), Groups(uint32_t(out->h), 8), uint32_t(out->c), err);
}

bool GpuKernels::Resize(const Tensor& in, int inOff, int count, Interp mode, Tensor* out,
                        int outOff, std::string* err) {
    if (!m->Kernel(&m->resize, kResizeHlsl, "nn_resize", err)) return false;
    return m->gpu->DispatchBuffers(
        *m->resize, {&in.gpu}, {&out->gpu},
        {uint32_t(in.w), uint32_t(in.h), uint32_t(out->w), uint32_t(out->h), uint32_t(inOff),
         uint32_t(outOff), mode == Interp::Bicubic ? 1u : 0u},
        Groups(uint32_t(out->w), 8), Groups(uint32_t(out->h), 8), uint32_t(count), err);
}

bool GpuKernels::CopyAcc(const Tensor& src, int srcOff, int count, Tensor* dst, int dstOff,
                         bool accumulate, std::string* err) {
    if (!m->Kernel(&m->element, kElementHlsl, "nn_element", err)) return false;
    return m->gpu->DispatchBuffers(
        *m->element, {&src.gpu, nullptr}, {&dst->gpu},
        {uint32_t(src.w), uint32_t(src.h), uint32_t(srcOff), uint32_t(dstOff),
         accumulate ? 1u : 0u, 0u},
        Groups(uint32_t(src.w), 8), Groups(uint32_t(src.h), 8), uint32_t(count), err);
}

bool GpuKernels::AddScaled(const Tensor& a, const Tensor& b, float s, Tensor* out,
                           std::string* err) {
    if (!m->Kernel(&m->element, kElementHlsl, "nn_element", err)) return false;
    return m->gpu->DispatchBuffers(
        *m->element, {&a.gpu, &b.gpu}, {&out->gpu},
        {uint32_t(a.w), uint32_t(a.h), 0u, 0u, 2u, F2U(s)},
        Groups(uint32_t(a.w), 8), Groups(uint32_t(a.h), 8), uint32_t(a.c), err);
}

bool GpuKernels::LayerNormGelu(const Tensor& in, const Vec& gamma, const Vec& beta, Tensor* out,
                               std::string* err) {
    if (!m->UploadVec(gamma, err) || !m->UploadVec(beta, err)) return false;
    if (!m->Kernel(&m->layerNorm, kLayerNormGeluHlsl, "nn_layernorm_gelu", err)) return false;
    const uint32_t np = uint32_t(in.Plane()), groups = Groups(np, 64);
    return m->gpu->DispatchBuffers(*m->layerNorm, {&in.gpu, &gamma.g, &beta.g}, {&out->gpu},
                                   {uint32_t(in.c), np}, std::min(groups, 65535u),
                                   Groups(groups, 65535), 1, err);
}

bool GpuKernels::Rotary(Tensor* t, int off, int count, int headDim, const Tensor& pos,
                        const Vec& freq, std::string* err) {
    if (!m->UploadVec(freq, err)) return false;
    if (!m->Kernel(&m->rotary, kRotaryHlsl, "nn_rotary", err)) return false;
    const uint32_t np = uint32_t(t->Plane());
    return m->gpu->DispatchBuffers(*m->rotary, {&pos.gpu, &freq.g}, {&t->gpu},
                                   {np, uint32_t(off), uint32_t(count), uint32_t(headDim)},
                                   Groups(np, 64), uint32_t(headDim / 2), 1, err);
}

bool GpuKernels::Attention(const Tensor& q, int qOff, const Tensor& k, int kOff, const Tensor& v,
                           int vOff, int heads, int headDim, Tensor* out, int outOff,
                           std::string* err) {
    if (headDim < 4 || headDim > 256 || headDim % 4) {
        *err = "unsupported attention head size";
        return false;
    }
    const ComputeKernel*& kern = m->attention[headDim];
    if (!kern) {
        const std::string src = "#define HEADDIM " + std::to_string(headDim) + "\n" + kAttentionHlsl;
        if (!m->Kernel(&kern, src, "nn_attention_hd" + std::to_string(headDim), err)) return false;
    }
    const uint32_t nq = uint32_t(q.Plane()), nk = uint32_t(k.Plane());
    return m->gpu->DispatchBuffers(
        *kern, {&q.gpu, &k.gpu, &v.gpu}, {&out->gpu},
        {nq, nk, uint32_t(qOff), uint32_t(kOff), uint32_t(vOff), uint32_t(outOff),
         F2U(1.0f / std::sqrt(float(headDim)))},
        Groups(nq, 64), uint32_t(heads), 1, err);
}

bool GpuKernels::MatMulTN(const Tensor& a, const Tensor& b, float scale, Tensor* out,
                          std::string* err) {
    if (!m->Kernel(&m->matmul, kMatMulTNHlsl, "nn_matmul_tn", err)) return false;
    const uint32_t M = uint32_t(a.Plane()), N = uint32_t(b.Plane());
    return m->gpu->DispatchBuffers(*m->matmul, {&a.gpu, &b.gpu}, {&out->gpu},
                                   {M, N, uint32_t(a.c), F2U(scale)}, Groups(N, 64),
                                   Groups(M, 64), 1, err);
}

bool GpuKernels::DualSoftmaxBest(const Tensor& sim, std::vector<int>* best0,
                                 std::vector<float>* p0, std::vector<int>* best1,
                                 std::string* err) {
    if (!m->Kernel(&m->dualSoftmax, kDualSoftmaxHlsl, "nn_dual_softmax", err)) return false;
    const uint32_t M = uint32_t(sim.h), N = uint32_t(sim.w);
    GpuBuffer rowLse, colLse, b0, b1;
    if (!m->gpu->AcquireBuffer(M * 4, &rowLse) || !m->gpu->AcquireBuffer(N * 4, &colLse) ||
        !m->gpu->AcquireBuffer(uint64_t(M) * 8, &b0) || !m->gpu->AcquireBuffer(N * 4, &b1)) {
        *err = "could not allocate dual-softmax scratch";
        return false;
    }
    const ComputeKernel& k = *m->dualSoftmax;
    const uint32_t rowsX = std::min(M, 65535u), rowsY = Groups(M, 65535);
    bool ok = m->gpu->DispatchBuffers(k, {&sim.gpu}, {&rowLse}, {M, N, 0u}, rowsX, rowsY, 1, err) &&
              m->gpu->DispatchBuffers(k, {&sim.gpu}, {&colLse}, {M, N, 1u}, Groups(N, 64), 1, 1, err) &&
              m->gpu->DispatchBuffers(k, {&sim.gpu, &rowLse, &colLse}, {&b0}, {M, N, 2u}, rowsX,
                                      rowsY, 1, err) &&
              m->gpu->DispatchBuffers(k, {&sim.gpu, &rowLse}, {&b1}, {M, N, 3u}, Groups(N, 64), 1,
                                      1, err);
    std::vector<uint32_t> r0(size_t(M) * 2), r1(N);
    ok = ok && m->gpu->ReadbackBuffer(b0, r0.data(), uint64_t(M) * 8) &&
         m->gpu->ReadbackBuffer(b1, r1.data(), uint64_t(N) * 4);
    for (GpuBuffer* b : {&rowLse, &colLse, &b0, &b1}) m->gpu->RecycleBuffer(std::move(*b));
    if (!ok) {
        if (err->empty()) *err = "dual softmax failed";
        return false;
    }
    best0->resize(M);
    p0->resize(M);
    best1->resize(N);
    for (uint32_t i = 0; i < M; ++i) {
        (*best0)[i] = int(r0[2 * i]);
        std::memcpy(&(*p0)[i], &r0[2 * i + 1], 4);
    }
    for (uint32_t j = 0; j < N; ++j) (*best1)[j] = int(r1[j]);
    return true;
}

bool GpuKernels::SampleBilinear(const Tensor& map, const Tensor& pts, int n, Tensor* out,
                                std::string* err) {
    if (!m->Kernel(&m->sample, kSampleHlsl, "nn_sample", err)) return false;
    // The point index rides in y, which caps a dispatch at 65535 points.
    if (n > 65535) { *err = "too many points to sample in one dispatch"; return false; }
    return m->gpu->DispatchBuffers(
        *m->sample, {&map.gpu, &pts.gpu}, {&out->gpu},
        {uint32_t(map.w), uint32_t(map.h), uint32_t(map.c), uint32_t(n)},
        Groups(uint32_t(map.c), 64), uint32_t(n), 1, err);
}

} // namespace nn
} // namespace tglab
