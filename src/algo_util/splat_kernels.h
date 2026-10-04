// The splat rasteriser's per-pixel kernels, shared by SplatRaster's GPU path
// and the GPU trainer so there is one copy of each, not two that can drift.
//
// Layout conventions both callers follow: flat arrays in rows of TexW
// texels, index i at (i % TexW, i / TexW); a projected Gaussian is three
// texels -- (u, v, opacity, radius), (conic a, b, c, reach), (r, g, b, depth),
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
//
// The list is fetched 256 entries at a time by the whole group, a thread an
// entry, rather than by every thread reading each entry and then the
// Gaussian it names -- a dependent pair of reads per entry with nothing to
// hide them. The group stops once every pixel is opaque, and a wave skips a
// Gaussian whose reach misses all its pixels. Measured on a 100-frame video
// (720k Gaussians in view at 180x320, 240 tiles): 10.1 ms a step before,
// 2.9 after.
inline constexpr const char* kForward = R"(
Texture2D<float4>   Proj    : register(t0);   // per Gaussian: 3 texels
Texture2D<float>    List    : register(t1);   // tile lists, flattened
Texture2D<float4>   Offs    : register(t2);   // per tile: start, count
RWTexture2D<float4> OutRGBT : register(u0);
RWTexture2D<float>  OutLast : register(u1);
RWTexture2D<float>  OutDepth: register(u2);   // expected depth, background 0
#ifdef REFL
// Reflections in the same pass (splat_reflect.h): each Gaussian's
// reflectivity and camera-facing normal, composited with the same weights
// as its colour over a background of 0.
Texture2D<float4>   Extra   : register(t3);   // per Gaussian: r, normal
RWTexture2D<float4> OutRN   : register(u3);   // R, normal
#endif

cbuffer Params : register(b0) {
    uint Width; uint Height;
    uint TilesX; uint TexW;
    uint BgR; uint BgG; uint BgB;
    uint MinA; uint MaxA; uint TStop;
};

uint2 At(uint i) { return uint2(i % TexW, i / TexW); }

// A batch of the tile's list, fetched by the whole group at once.
#define kFB 256
groupshared float4 fA[kFB];     // u, v, opacity
groupshared float4 fC[kFB];     // conic
groupshared float4 fCol[kFB];   // colour, depth
#ifdef REFL
groupshared float4 fX[kFB];     // reflectivity, normal
#endif
groupshared uint   fAlive;      // any pixel still compositing

[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 gt : SV_GroupThreadID,
          uint gi : SV_GroupIndex) {
    uint px = gid.x * 16 + gt.x, py = gid.y * 16 + gt.y;
    // No early return: every thread takes part in fetching and the barriers.
    bool inside = px < Width && py < Height;

    float4 o = Offs[At(gid.y * TilesX + gid.x)];
    uint start = uint(o.x), count = uint(o.y);
    float minA = asfloat(MinA), maxA = asfloat(MaxA), tStop = asfloat(TStop);

    float T = 1.0;
    float3 C = float3(0, 0, 0);
    float D = 0.0;
#ifdef REFL
    float4 RN = float4(0, 0, 0, 0);
#endif
    uint last = 0;
    bool done = !inside;
    // The wave's pixels as a rectangle. A Gaussian whose reach -- the
    // radius past which its alpha is below minA -- misses the rectangle is
    // skipped by every pixel in it anyway, so the whole wave skips it
    // without the arithmetic. Most Gaussians in a tile's list touch a few of
    // its waves at most.
    float wx0 = float(WaveActiveMin(inside ? px : 0xffffffffu));
    float wx1 = float(WaveActiveMax(inside ? px : 0u));
    float wy0 = float(WaveActiveMin(inside ? py : 0xffffffffu));
    float wy1 = float(WaveActiveMax(inside ? py : 0u));
    for (uint base = 0; base < count; base += kFB) {
        // Fetch the batch, one entry a thread, and find out whether any
        // pixel still needs it: once all are opaque the group stops.
        if (gi == 0) fAlive = 0;
        GroupMemoryBarrierWithGroupSync();
        if (!done) fAlive = 1;
        if (base + gi < count) {
            uint j = uint(List[At(start + base + gi)]);
            fA[gi]   = Proj[At(j * 3)];
            fC[gi]   = Proj[At(j * 3 + 1)];
            fCol[gi] = Proj[At(j * 3 + 2)];
#ifdef REFL
            fX[gi] = Extra[At(j)];
#endif
        }
        GroupMemoryBarrierWithGroupSync();
        if (fAlive == 0) break;

        // Walk it near to far, exactly as one entry at a time did.
        uint nb = min(uint(kFB), count - base);
        for (uint b = 0; b < nb && !done; ++b) {
            float4 a = fA[b];                  // u, v, opacity
            float4 c = fC[b];                  // conic a, b, c, reach
            float ex = clamp(a.x, wx0, wx1) - a.x, ey = clamp(a.y, wy0, wy1) - a.y;
            if (ex * ex + ey * ey > c.w * c.w) continue;
            float dx = float(px) - a.x, dy = float(py) - a.y;
            float power = -0.5 * (c.x * dx * dx + c.z * dy * dy) - c.y * dx * dy;
            if (power > 0.0) continue;
            float alpha = min(maxA, a.z * exp(power));
            if (alpha < minA) continue;
            float nextT = T * (1.0 - alpha);
            if (nextT < tStop) { done = true; break; }   // before blending, as the CPU
            float4 col = fCol[b];
            C += col.rgb * alpha * T;
            D += col.w * alpha * T;
#ifdef REFL
            RN += fX[b] * (alpha * T);
#endif
            T = nextT;
            last = base + b + 1;
        }
        // The next fetch overwrites the batch: wait for everyone to finish it.
        GroupMemoryBarrierWithGroupSync();
    }
    if (!inside) return;
    float3 bg = float3(asfloat(BgR), asfloat(BgG), asfloat(BgB));
    OutRGBT[uint2(px, py)] = float4(C + T * bg, T);
    OutLast[uint2(px, py)] = float(last);
    OutDepth[uint2(px, py)] = D;
#ifdef REFL
    OutRN[uint2(px, py)] = RN;
#endif
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
//
// THE WAVE SUMS were then most of what was left. Summing each of the ten to
// fourteen values on its own is five exchanges apiece; halving -- each lane
// keeping half its values and adding its partner's copy of that half -- sums
// all sixteen in sixteen exchanges, and leaves them spread across the lanes
// to be written at once. On NVIDIA it pairs lanes in the same order the
// built-in sum does, so the result is the same to the bit. With the per-wave
// skip of Gaussians out of reach (see kForward) and waves that miss writing
// nothing: 15.8 ms a step on the video above, 6.7 after, same training
// result to the last digit.
inline constexpr const char* kBackward = R"(
Texture2D<float4>   Proj  : register(t0);
Texture2D<float>    List  : register(t1);
Texture2D<float4>   Offs  : register(t2);
Texture2D<float4>   DRGBT : register(t3);   // dLoss/dRGB, final T in .w
RWTexture2D<float4> EGrad : register(u0);   // per entry: 3 texels
RWTexture2D<float>  Last  : register(u1);   // from the forward pass
RWTexture2D<float>  DDepth: register(u2);   // dLoss/d(expected depth); 0 = no depth loss
RWTexture2D<float>  DShift: register(u3);   // per-pixel depth shift: see DepthLossGrad
#ifdef REFL
// Reflections in the same pass: each Gaussian's reflectivity and normal as
// the forward composited them, and dLoss by the composited R and normal per
// pixel. Four more values per entry -- dLoss by the Gaussian's r and normal
// -- in a fourth EGrad texel, and four more running "behind" sums feeding
// dAlpha, exactly as colour's three do. Background 0 in all four.
Texture2D<float4>   Extra : register(t4);   // per Gaussian: r, normal
Texture2D<float4>   DRN   : register(t5);   // per pixel: dLoss/dR, dLoss/dnormal
#define kQ   14                             // gradient values per entry
#define kTex 4                              // EGrad texels per entry
#else
#define kQ   10
#define kTex 3
#endif

cbuffer Params : register(b0) {
    uint D0; uint D1;                        // EGrad's size; unused
    uint ImgW; uint ImgH;
    uint TilesX; uint TexW;
    uint BgR; uint BgG; uint BgB;
    uint MinA; uint MaxA;
    uint Gate;   // RasterOptions::colourGate; 0 is off
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
#ifdef REFL
groupshared float4 sX[kBatch];     // reflectivity, normal
#endif
groupshared float  part[kPart];
groupshared uint   hitW[kBatch];   // per entry: which waves wrote a sum

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
    float  gate = asfloat(Gate);
#ifdef REFL
    float4 dX   = inside ? DRN[uint2(px, py)] : float4(0, 0, 0, 0);
    float4 accX = float4(0, 0, 0, 0);
#endif

    // How far back ANY pixel in the tile needs to go.
    if (gi == 0) gMax = 0;
    GroupMemoryBarrierWithGroupSync();
    InterlockedMax(gMax, last);
    GroupMemoryBarrierWithGroupSync();
    uint kmax = gMax;

    // Entries no pixel reached get zero, written here because the loop
    // below never visits them.
    for (uint e = kmax + gi; e < count; e += 256)
        [unroll] for (uint zt = 0; zt < kTex; ++zt)
            EGrad[At((start + e) * kTex + zt)] = float4(0, 0, 0, 0);

    uint lanes = WaveGetLaneCount();
    uint wid = gi / lanes;
    uint nw = (256 + lanes - 1) / lanes;
    uint batch = min(uint(kBatch), uint(kPart) / (nw * kQ));
    // A wave that no pixel of hits writes nothing, and the sum skips it --
    // with a bit per wave, so up to 32 of them; narrower waves than 8 lanes
    // fall back to writing zeros.
    bool useMask = nw <= 32;
    // The halving reduction below needs at least 16 lanes, a power of two.
    bool useXp = lanes >= 16 && (lanes & (lanes - 1)) == 0;

    // The wave's pixels as a rectangle, for skipping Gaussians that cannot
    // reach any of them; see kForward.
    float wx0 = float(WaveActiveMin(inside ? px : 0xffffffffu));
    float wx1 = float(WaveActiveMax(inside ? px : 0u));
    float wy0 = float(WaveActiveMin(inside ? py : 0xffffffffu));
    float wy1 = float(WaveActiveMax(inside ? py : 0u));

    for (uint top = kmax; top > 0; ) {
        uint nb = min(batch, top);          // entries top-1 down to top-nb

        // 1. Fetch the batch. The barrier also orders this batch's writes to
        //    `part` after the previous batch's reads of it.
        if (gi < nb) {
            hitW[gi] = 0;
            uint j = uint(List[At(start + top - 1 - gi)]);
            sA[gi]   = Proj[At(j * 3)];
            sC[gi]   = Proj[At(j * 3 + 1)];
            sCol[gi] = Proj[At(j * 3 + 2)];
#ifdef REFL
            sX[gi] = Extra[At(j)];
#endif
        }
        GroupMemoryBarrierWithGroupSync();

        // 2. Walk it back to front, as one entry at a time did.
        for (uint b = 0; b < nb; ++b) {
            uint k = top - 1 - b;
            float g[kQ];
            [unroll] for (uint gz = 0; gz < kQ; ++gz) g[gz] = 0.0;
            bool hit = false;

            float4 a = sA[b];
            float4 c = sC[b];
            float ex = clamp(a.x, wx0, wx1) - a.x, ey = clamp(a.y, wy0, wy1) - a.y;
            if (inside && k < last && ex * ex + ey * ey <= c.w * c.w) {
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
                        // Depth-gated colour: see RasterOptions::colourGate.
                        // `shift` is the pixel's measured depth, 0 for none.
                        bool gated = gate > 0.0 && shift > 0.0 &&
                                     abs(sCol[b].w - shift) > gate * shift;
                        if (!gated) {
                            g[5] = alpha * Ti * dC.x;
                            g[6] = alpha * Ti * dC.y;
                            g[7] = alpha * Ti * dC.z;
                        }
                        float dAlpha = Ti * dot(col - acc, dC);
                        if (dD != 0.0) {   // no depth loss: skip it exactly
                            float z = sCol[b].w - shift;   // the channel's value
                            g[9] = alpha * Ti * dD;
                            dAlpha += Ti * (z - accD) * dD;
                            accD = alpha * z + (1.0 - alpha) * accD;
                        }
#ifdef REFL
                        // Reflectivity and normal: four more channels, as
                        // colour's, over a background of 0.
                        float4 xv = sX[b];
                        if (!gated) {   // appearance too, as colour
                            g[10] = alpha * Ti * dX.x;
                            g[11] = alpha * Ti * dX.y;
                            g[12] = alpha * Ti * dX.z;
                            g[13] = alpha * Ti * dX.w;
                        }
                        dAlpha += Ti * dot(xv - accX, dX);
                        accX = alpha * xv + (1.0 - alpha) * accX;
#endif
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

            uint p = (b * nw + wid) * kQ;
            if (WaveActiveAnyTrue(hit)) {
                if (useXp) {
                    // ALL SIXTEEN SUMS AT ONCE, by halving: at each step a
                    // lane keeps half its values and adds its partner's
                    // copy of that half, so after four steps each lane holds
                    // one value's partial sum over sixteen lanes -- 16
                    // exchanges where one sum per value took 5 each, 70 --
                    // and the waves' results are written by 16 lanes at once.
                    float v[16];
                    [unroll] for (uint q0 = 0; q0 < 16; ++q0) v[q0] = q0 < kQ ? g[q0] : 0.0;
                    uint l = WaveGetLaneIndex();
                    [unroll] for (uint s = 0; s < 4; ++s) {
                        uint o = lanes >> (s + 1);
                        bool up = (l & o) != 0;
                        [unroll] for (uint i = 0; i < (8u >> s); ++i) {
                            float send = up ? v[i] : v[i + (8u >> s)];
                            float keep = up ? v[i + (8u >> s)] : v[i];
                            v[i] = keep + WaveReadLaneAt(send, l ^ o);
                        }
                    }
                    float r = v[0];
                    for (uint o2 = lanes >> 5; o2 > 0; o2 >>= 1) r += WaveReadLaneAt(r, l ^ o2);
                    if ((l & ((lanes >> 4) - 1)) == 0) {
                        uint idx = ((l & (lanes >> 1)) ? 8u : 0u) + ((l & (lanes >> 2)) ? 4u : 0u) +
                                   ((l & (lanes >> 3)) ? 2u : 0u) + ((l & (lanes >> 4)) ? 1u : 0u);
                        if (idx < kQ) part[p + idx] = r;
                    }
                } else
                [unroll] for (uint q = 0; q < kQ; ++q) {
                    float s = WaveActiveSum(g[q]);
                    if (WaveIsFirstLane()) part[p + q] = s;
                }
                if (useMask && WaveIsFirstLane()) {
                    uint prev;
                    InterlockedOr(hitW[b], 1u << wid, prev);
                }
            } else if (!useMask && WaveIsFirstLane()) {
                [unroll] for (uint q = 0; q < kQ; ++q) part[p + q] = 0;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // 3. Add the waves' sums and write: one texel per thread. Texels 0
        //    and 1 hold values 0-7, texel 2 values 8-9 (opacity, depth), and
        //    with reflections texel 3 values 10-13.
        for (uint t = gi; t < nb * kTex; t += 256) {
            uint b = t / kTex, r = t % kTex;
            uint n = (r == 2) ? 2 : 4;
            uint off = (r < 3) ? r * 4 : 10;
            float4 v = float4(0, 0, 0, 0);
            uint mask = useMask ? hitW[b] : 0xffffffffu;
            for (uint w = 0; w < nw; ++w) {
                if (useMask && ((mask >> w) & 1u) == 0) continue;
                uint p = (b * nw + w) * kQ + off;
                v.x += part[p];
                v.y += part[p + 1];
                if (n == 4) { v.z += part[p + 2]; v.w += part[p + 3]; }
            }
            EGrad[At((start + top - 1 - b) * kTex + r)] = v;
        }
        top -= nb;
    }
}
)";

}  // namespace splat_kernels
}  // namespace tglab
