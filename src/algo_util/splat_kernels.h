// The splat rasteriser's per-pixel kernels, shared by SplatRaster's GPU path
// and the GPU trainer so there is one copy of each, not two that can drift.
//
// Layout conventions both callers follow: flat arrays in rows of TexW
// texels, index i at (i % TexW, i / TexW); a projected Gaussian is three
// texels -- (u, v, opacity, radius), (conic a, b, c, -), (r, g, b, depth),
// depth being the centre's camera-space z; a tile
// list entry is one R32F texel holding the Gaussian's index; a tile's offset
// texel holds (start, count).
#pragma once

namespace tglab {
namespace splat_kernels {

// --- forward: composite -------------------------------------------------------
//
// One 16x16 group per screen tile, one thread per pixel, each walking its
// tile's list near to far exactly as the CPU loop does. Writes the colour
// with the final transmittance in .w, and how many list entries were used --
// both of which the backward pass needs.
inline constexpr const char* kForward = R"(
Texture2D<float4>   Proj    : register(t0);   // per Gaussian: 3 texels
Texture2D<float>    List    : register(t1);   // tile lists, flattened
Texture2D<float4>   Offs    : register(t2);   // per tile: start, count
RWTexture2D<float4> OutRGBT : register(u0);
RWTexture2D<float>  OutLast : register(u1);
RWTexture2D<float>  OutDepth: register(u2);   // expected depth, background 0

cbuffer Params : register(b0) {
    uint Width; uint Height;
    uint TilesX; uint TexW;
    uint BgR; uint BgG; uint BgB;
    uint MinA; uint MaxA; uint TStop;
};

uint2 At(uint i) { return uint2(i % TexW, i / TexW); }

[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 gt : SV_GroupThreadID) {
    uint px = gid.x * 16 + gt.x, py = gid.y * 16 + gt.y;
    if (px >= Width || py >= Height) return;

    float4 o = Offs[At(gid.y * TilesX + gid.x)];
    uint start = uint(o.x), count = uint(o.y);
    float minA = asfloat(MinA), maxA = asfloat(MaxA), tStop = asfloat(TStop);

    float T = 1.0;
    float3 C = float3(0, 0, 0);
    float D = 0.0;
    uint last = 0;
    for (uint k = 0; k < count; ++k) {
        uint j = uint(List[At(start + k)]);
        float4 a = Proj[At(j * 3)];        // u, v, opacity
        float4 c = Proj[At(j * 3 + 1)];    // conic a, b, c
        float dx = float(px) - a.x, dy = float(py) - a.y;
        float power = -0.5 * (c.x * dx * dx + c.z * dy * dy) - c.y * dx * dy;
        if (power > 0.0) continue;
        float alpha = min(maxA, a.z * exp(power));
        if (alpha < minA) continue;
        float nextT = T * (1.0 - alpha);
        if (nextT < tStop) break;          // before blending, as the CPU
        float4 col = Proj[At(j * 3 + 2)];
        C += col.rgb * alpha * T;
        D += col.w * alpha * T;
        T = nextT;
        last = k + 1;
    }
    float3 bg = float3(asfloat(BgR), asfloat(BgG), asfloat(BgB));
    OutRGBT[uint2(px, py)] = float4(C + T * bg, T);
    OutLast[uint2(px, py)] = float(last);
    OutDepth[uint2(px, py)] = D;
}
)";

// --- backward: per-pixel gradients, reduced per list entry ----------------------
//
// Same one-group-per-tile layout. The group walks its list BACK TO FRONT in
// lockstep -- every thread at the same entry at once -- and each thread works
// out its own pixel's contribution to that entry's gradient exactly as the
// CPU does. The group then sums those contributions (wave intrinsics, then
// group-shared memory across waves) and the group writes the entry.
//
// Summing within the group rather than with atomics: there are no float
// atomics on the texture formats the framework binds, and a per-entry sum
// across a tile is precisely the reduction the CPU does serially.
//
// IT RUNS IN BATCHES of up to 32 entries, which is where its speed comes
// from. Done one entry at a time it cost two group barriers per entry, one
// thread writing each result alone, and in every thread a dependent pair of
// reads (the list, then the Gaussian it names) with nothing to hide their
// latency. Per batch instead:
//
//   1. one thread per entry fetches the Gaussian into group-shared memory;
//   2. every thread walks the batch, each wave summing its pixels'
//      contributions into group-shared memory -- and skipping the nine sums
//      outright when no pixel in the wave touches that Gaussian, which in a
//      16x16 tile is most of the time;
//   3. after one barrier, the group adds the waves' sums, a texel per thread.
//
// The additions happen in the same order as before -- the same wave sums,
// then waves 0, 1, 2... -- but the result is not the same to the bit: the
// compiler fuses the per-pixel arithmetic differently once the code around
// it changes. On fountain-P11 that moved the final L1 in its fourth digit
// and densification's thresholds by six Gaussians in half a million.
// Measured there: 31 ms an iteration one entry at a time, 12 batched.
inline constexpr const char* kBackward = R"(
Texture2D<float4>   Proj  : register(t0);
Texture2D<float>    List  : register(t1);
Texture2D<float4>   Offs  : register(t2);
Texture2D<float4>   DRGBT : register(t3);   // dLoss/dRGB, final T in .w
RWTexture2D<float4> EGrad : register(u0);   // per entry: 3 texels
RWTexture2D<float>  Last  : register(u1);   // from the forward pass
RWTexture2D<float>  DDepth: register(u2);   // dLoss/d(expected depth); 0 = no depth loss
RWTexture2D<float>  DShift: register(u3);   // per-pixel depth shift: see DepthLossGrad

cbuffer Params : register(b0) {
    uint D0; uint D1;                        // EGrad's size; unused
    uint ImgW; uint ImgH;
    uint TilesX; uint TexW;
    uint BgR; uint BgG; uint BgB;
    uint MinA; uint MaxA;
};

// Per-wave partial sums for a batch: [entry][wave][10]. Sized for the batch
// at the smallest wave this might meet; the batch shrinks to fit narrower
// waves (more of them) rather than the array growing.
#define kPart  4608
#define kBatch 32

groupshared uint   gMax;
groupshared float4 sA[kBatch];     // u, v, opacity
groupshared float4 sC[kBatch];     // conic
groupshared float4 sCol[kBatch];   // colour, depth in .w
groupshared float  part[kPart];

uint2 At(uint i) { return uint2(i % TexW, i / TexW); }

[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 gt : SV_GroupThreadID,
          uint gi : SV_GroupIndex) {
    uint px = gid.x * 16 + gt.x, py = gid.y * 16 + gt.y;
    bool inside = px < ImgW && py < ImgH;

    float4 o = Offs[At(gid.y * TilesX + gid.x)];
    uint start = uint(o.x), count = uint(o.y);
    float minA = asfloat(MinA), maxA = asfloat(MaxA);

    uint last = inside ? uint(Last[uint2(px, py)]) : 0;
    float4 d  = inside ? DRGBT[uint2(px, py)] : float4(0, 0, 0, 0);
    float  T  = d.w;
    float3 dC = d.xyz;
    float3 acc = float3(asfloat(BgR), asfloat(BgG), asfloat(BgB));
    // Depth is a fourth channel: each Gaussian's "colour" in it is its own
    // depth, and the background's is 0.
    float  dD  = inside ? DDepth[uint2(px, py)] : 0.0;
    float  shift = inside ? DShift[uint2(px, py)] : 0.0;
    float  accD = 0.0;

    // How far back ANY pixel in the tile needs to go.
    if (gi == 0) gMax = 0;
    GroupMemoryBarrierWithGroupSync();
    InterlockedMax(gMax, last);
    GroupMemoryBarrierWithGroupSync();
    uint kmax = gMax;

    // Entries no pixel reached get zero, written here because the loop
    // below never visits them.
    for (uint e = kmax + gi; e < count; e += 256) {
        EGrad[At((start + e) * 3)]     = float4(0, 0, 0, 0);
        EGrad[At((start + e) * 3 + 1)] = float4(0, 0, 0, 0);
        EGrad[At((start + e) * 3 + 2)] = float4(0, 0, 0, 0);
    }

    uint lanes = WaveGetLaneCount();
    uint wid = gi / lanes;
    uint nw = (256 + lanes - 1) / lanes;
    uint batch = min(uint(kBatch), uint(kPart) / (nw * 10));

    for (uint top = kmax; top > 0; ) {
        uint nb = min(batch, top);          // entries top-1 down to top-nb

        // 1. Fetch the batch. The barrier also orders this batch's writes to
        //    `part` after the previous batch's reads of it.
        if (gi < nb) {
            uint j = uint(List[At(start + top - 1 - gi)]);
            sA[gi]   = Proj[At(j * 3)];
            sC[gi]   = Proj[At(j * 3 + 1)];
            sCol[gi] = Proj[At(j * 3 + 2)];
        }
        GroupMemoryBarrierWithGroupSync();

        // 2. Walk it back to front, as one entry at a time did.
        for (uint b = 0; b < nb; ++b) {
            uint k = top - 1 - b;
            float g[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            bool hit = false;

            if (inside && k < last) {
                float4 a = sA[b];
                float4 c = sC[b];
                float dx = float(px) - a.x, dy = float(py) - a.y;
                float power = -0.5 * (c.x * dx * dx + c.z * dy * dy) - c.y * dx * dy;
                if (power <= 0.0) {
                    float G = exp(power);
                    float raw = a.z * G;
                    float alpha = min(maxA, raw);
                    if (alpha >= minA) {
                        hit = true;
                        float Ti = T / (1.0 - alpha);
                        float3 col = sCol[b].rgb;
                        g[5] = alpha * Ti * dC.x;
                        g[6] = alpha * Ti * dC.y;
                        g[7] = alpha * Ti * dC.z;
                        float dAlpha = Ti * dot(col - acc, dC);
                        if (dD != 0.0) {   // no depth loss: skip it exactly
                            float z = sCol[b].w - shift;   // the channel's value
                            g[9] = alpha * Ti * dD;
                            dAlpha += Ti * (z - accD) * dD;
                            accD = alpha * z + (1.0 - alpha) * accD;
                        }
                        acc = alpha * col + (1.0 - alpha) * acc;
                        T = Ti;
                        if (raw <= maxA) {
                            g[8] = dAlpha * G * a.z * (1.0 - a.z);
                            float dP = dAlpha * alpha;
                            g[2] = dP * (-0.5 * dx * dx);
                            g[3] = dP * (-dx * dy);
                            g[4] = dP * (-0.5 * dy * dy);
                            g[0] = dP * (c.x * dx + c.y * dy);
                            g[1] = dP * (c.y * dx + c.z * dy);
                        }
                    }
                }
            }

            uint p = (b * nw + wid) * 10;
            if (WaveActiveAnyTrue(hit)) {
                [unroll] for (uint q = 0; q < 10; ++q) {
                    float s = WaveActiveSum(g[q]);
                    if (WaveIsFirstLane()) part[p + q] = s;
                }
            } else if (WaveIsFirstLane()) {
                [unroll] for (uint q = 0; q < 10; ++q) part[p + q] = 0;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // 3. Add the waves' sums and write: one texel per thread.
        for (uint t = gi; t < nb * 3; t += 256) {
            uint b = t / 3, r = t % 3;
            uint n = (r == 2) ? 2 : 4;   // texel 2: opacity, depth
            float4 v = float4(0, 0, 0, 0);
            for (uint w = 0; w < nw; ++w) {
                uint p = (b * nw + w) * 10 + r * 4;
                v.x += part[p];
                v.y += part[p + 1];
                if (n == 4) { v.z += part[p + 2]; v.w += part[p + 3]; }
            }
            EGrad[At((start + top - 1 - b) * 3 + r)] = v;
        }
        top -= nb;
    }
}
)";

}  // namespace splat_kernels
}  // namespace tglab
