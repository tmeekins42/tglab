// The splat rasteriser's per-pixel passes, on the GPU.
//
// See SplatRaster::SetGpu for why only these two. This file is the device
// half: two compute kernels, and the packing that gets the CPU's projected
// Gaussians and tile lists into textures and the results back out.
//
// SHAPED BY THIS MACHINE'S HISTORY. The GPU plane sweep hung the device until
// it was reduced to one dispatch per submission, and ComputeContext records
// the same failure from the develop chains. So each pass here is ONE
// dispatch, and every Upload and Readback around it flushes on its own --
// the pattern that has never hung this hardware.
//
// EVERYTHING IS A TEXTURE, because that is all the framework binds: four
// SRVs, four UAVs, thirty-two root constants. Flat arrays are laid out in
// rows of kTexW texels; an index i lives at (i % kTexW, i / kTexW). Indices
// travel as floats, which is exact below 2^24 -- sixteen million Gaussians or
// list entries, far past what a CPU-prepared frame will hold.
#include "splat_raster.h"

#include <algorithm>
#include <cstring>
#include <mutex>

#include "../gpu/compute.h"

namespace tglab {
namespace {

constexpr int kTexW = 2048;

uint32_t Bits(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, 4);
    return u;
}

ImageDesc FlatDesc(size_t texels, Format f) {
    ImageDesc d;
    d.width  = kTexW;
    d.height = int(std::max<size_t>(1, (texels + kTexW - 1) / kTexW));
    d.format = f;
    return d;
}

// --- forward: composite -------------------------------------------------------
//
// One 16x16 group per screen tile, one thread per pixel, each walking its
// tile's list near to far exactly as the CPU loop does. Writes the colour
// with the final transmittance in .w, and how many list entries were used --
// both of which the backward pass needs.
constexpr const char* kForwardHlsl = R"(
Texture2D<float4>   Proj    : register(t0);   // per Gaussian: 3 texels
Texture2D<float>    List    : register(t1);   // tile lists, flattened
Texture2D<float4>   Offs    : register(t2);   // per tile: start, count
RWTexture2D<float4> OutRGBT : register(u0);
RWTexture2D<float>  OutLast : register(u1);

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
        T = nextT;
        last = k + 1;
    }
    float3 bg = float3(asfloat(BgR), asfloat(BgG), asfloat(BgB));
    OutRGBT[uint2(px, py)] = float4(C + T * bg, T);
    OutLast[uint2(px, py)] = float(last);
}
)";

// --- backward: per-pixel gradients, reduced per list entry ----------------------
//
// Same one-group-per-tile layout. The group walks its list BACK TO FRONT in
// lockstep -- every thread at the same entry at once -- and each thread works
// out its own pixel's contribution to that entry's gradient exactly as the
// CPU does. The group then sums those contributions (wave intrinsics, then
// group-shared memory across waves) and one thread writes the entry.
//
// Summing within the group rather than with atomics: there are no float
// atomics on the texture formats the framework binds, and a per-entry sum
// across a tile is precisely the reduction the CPU does serially.
constexpr const char* kBackwardHlsl = R"(
Texture2D<float4>   Proj  : register(t0);
Texture2D<float>    List  : register(t1);
Texture2D<float4>   Offs  : register(t2);
Texture2D<float4>   DRGBT : register(t3);   // dLoss/dRGB, final T in .w
RWTexture2D<float4> EGrad : register(u0);   // per entry: 3 texels
RWTexture2D<float>  Last  : register(u1);   // from the forward pass

cbuffer Params : register(b0) {
    uint D0; uint D1;                        // EGrad's size; unused
    uint ImgW; uint ImgH;
    uint TilesX; uint TexW;
    uint BgR; uint BgG; uint BgB;
    uint MinA; uint MaxA;
};

groupshared uint  gMax;
groupshared float part[64][9];

uint2 At(uint i) { return uint2(i % TexW, i / TexW); }

void WriteEntry(uint e, float g[9]) {
    EGrad[At(e * 3)]     = float4(g[0], g[1], g[2], g[3]);
    EGrad[At(e * 3 + 1)] = float4(g[4], g[5], g[6], g[7]);
    EGrad[At(e * 3 + 2)] = float4(g[8], 0, 0, 0);
}

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

    // How far back ANY pixel in the tile needs to go.
    if (gi == 0) gMax = 0;
    GroupMemoryBarrierWithGroupSync();
    InterlockedMax(gMax, last);
    GroupMemoryBarrierWithGroupSync();
    uint kmax = gMax;

    // Entries no pixel reached get zero, written here because the loop
    // below never visits them.
    float zero[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (uint e = kmax + gi; e < count; e += 256) WriteEntry(start + e, zero);

    uint lanes = WaveGetLaneCount();
    uint wid = gi / lanes;
    uint nw = (256 + lanes - 1) / lanes;

    for (uint kk = kmax; kk > 0; --kk) {
        uint k = kk - 1;
        float g[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

        if (inside && k < last) {
            uint j = uint(List[At(start + k)]);
            float4 a = Proj[At(j * 3)];
            float4 c = Proj[At(j * 3 + 1)];
            float dx = float(px) - a.x, dy = float(py) - a.y;
            float power = -0.5 * (c.x * dx * dx + c.z * dy * dy) - c.y * dx * dy;
            if (power <= 0.0) {
                float G = exp(power);
                float raw = a.z * G;
                float alpha = min(maxA, raw);
                if (alpha >= minA) {
                    float Ti = T / (1.0 - alpha);
                    float3 col = Proj[At(j * 3 + 2)].rgb;
                    g[5] = alpha * Ti * dC.x;
                    g[6] = alpha * Ti * dC.y;
                    g[7] = alpha * Ti * dC.z;
                    float dAlpha = Ti * dot(col - acc, dC);
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

        // Sum across the group: each wave first, then the waves.
        [unroll] for (uint q = 0; q < 9; ++q) {
            float s = WaveActiveSum(g[q]);
            if (WaveIsFirstLane()) part[wid][q] = s;
        }
        GroupMemoryBarrierWithGroupSync();
        if (gi == 0) {
            float t[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
            for (uint w = 0; w < nw; ++w)
                [unroll] for (uint q2 = 0; q2 < 9; ++q2) t[q2] += part[w][q2];
            WriteEntry(start + k, t);
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
)";

}  // namespace

struct SplatRaster::Gpu {
    ComputeContext* ctx = nullptr;
    ComputeKernel   fwd, bwd;
    bool            ready = false, failed = false;

    // Kept between calls and reallocated only when a size changes: a
    // training run reuses the same dimensions for thousands of iterations.
    GpuImage proj, list, offs, rgbt, last, drgbt, egrad;
    std::vector<float> stage;
    int tilesX = 0;

    bool Ensure(GpuImage& img, const ImageDesc& d) {
        if (img.Valid() && img.desc == d) return true;
        img.Release();
        return ctx->CreateImage(d, &img);
    }
    bool Put(GpuImage& img, const ImageDesc& d) {
        ImageView v;
        v.desc = d;
        v.data = reinterpret_cast<uint8_t*>(stage.data());
        return ctx->Upload(v, &img);
    }
};

SplatRaster::SplatRaster() = default;
SplatRaster::~SplatRaster() = default;

void SplatRaster::SetGpu(ComputeContext* gpu) {
    if (!gpu) { m_gpu.reset(); return; }
    if (!m_gpu) m_gpu = std::make_unique<Gpu>();
    m_gpu->ctx = gpu;
}

bool SplatRaster::CompositeGpu(const std::vector<SplatParam>& splats,
                               const SplatCam& cam, const RasterOptions& opt,
                               std::vector<double>* rgb) {
    Gpu& g = *m_gpu;
    if (g.failed || !g.ctx || !g.ctx->Ready()) return false;

    // ComputeContext has one command queue and no locking of its own.
    std::lock_guard<std::mutex> lock(g.ctx->SubmitMutex());

    if (!g.ready) {
        std::string err;
        if (!g.ctx->CreateKernel(kForwardHlsl, "main", "splat_forward", &g.fwd, &err) ||
            !g.ctx->CreateKernel(kBackwardHlsl, "main", "splat_backward", &g.bwd, &err)) {
            g.failed = true;
            m_gpuNote = "splat kernels did not compile: " + err;
            return false;
        }
        g.ready = true;
    }

    const size_t n = m_proj.size();
    const size_t nTiles = m_tiles.size();
    size_t entries = 0;
    for (const auto& t : m_tiles) entries += t.size();

    // Projected Gaussians: three texels each.
    const ImageDesc projD = FlatDesc(std::max<size_t>(1, n * 3), Format::RGBA32F);
    g.stage.assign(size_t(projD.width) * size_t(projD.height) * 4, 0.0f);
    for (size_t j = 0; j < n; ++j) {
        const Proj& P = m_proj[j];
        if (!P.ok) continue;
        float* t = &g.stage[j * 12];
        t[0] = float(P.u); t[1] = float(P.v); t[2] = float(P.opac);
        t[4] = float(P.conic[0]); t[5] = float(P.conic[1]); t[6] = float(P.conic[2]);
        t[8] = float(splats[j].color[0]);
        t[9] = float(splats[j].color[1]);
        t[10] = float(splats[j].color[2]);
    }
    if (!g.Ensure(g.proj, projD) || !g.Put(g.proj, projD)) return false;

    // Tile lists, flattened, and each tile's start and count.
    const ImageDesc listD = FlatDesc(std::max<size_t>(1, entries), Format::R32F);
    const ImageDesc offsD = FlatDesc(std::max<size_t>(1, nTiles), Format::RGBA32F);
    {
        g.stage.assign(size_t(listD.width) * size_t(listD.height), 0.0f);
        size_t e = 0;
        for (const auto& t : m_tiles)
            for (int j : t) g.stage[e++] = float(j);
        if (!g.Ensure(g.list, listD) || !g.Put(g.list, listD)) return false;

        g.stage.assign(size_t(offsD.width) * size_t(offsD.height) * 4, 0.0f);
        size_t start = 0;
        for (size_t t = 0; t < nTiles; ++t) {
            g.stage[t * 4 + 0] = float(start);
            g.stage[t * 4 + 1] = float(m_tiles[t].size());
            start += m_tiles[t].size();
        }
        if (!g.Ensure(g.offs, offsD) || !g.Put(g.offs, offsD)) return false;
    }

    const ImageDesc rgbtD{cam.w, cam.h, Format::RGBA32F};
    const ImageDesc lastD{cam.w, cam.h, Format::R32F};
    if (!g.Ensure(g.rgbt, rgbtD) || !g.Ensure(g.last, lastD)) return false;

    g.tilesX = m_tilesX;
    std::string err;
    const std::vector<uint32_t> c{
        uint32_t(m_tilesX), uint32_t(kTexW),
        Bits(float(opt.background.x)), Bits(float(opt.background.y)),
        Bits(float(opt.background.z)), Bits(float(opt.minAlpha)),
        Bits(float(opt.maxAlpha)), Bits(float(opt.tStop))};
    if (!g.ctx->Dispatch(g.fwd, {&g.proj, &g.list, &g.offs}, {&g.rgbt, &g.last},
                         c, &err, uint32_t(m_tilesX), uint32_t(m_tilesY))) {
        m_gpuNote = "splat forward dispatch failed: " + err;
        return false;
    }

    // Colour and transmittance back, and the last-used index -- the CPU
    // backward pass needs both if the GPU one later fails.
    const size_t np = size_t(cam.w) * size_t(cam.h);
    g.stage.assign(np * 4, 0.0f);
    {
        ImageView v;
        v.desc = rgbtD;
        v.data = reinterpret_cast<uint8_t*>(g.stage.data());
        if (!g.ctx->Readback(g.rgbt, &v)) { m_gpuNote = "readback failed"; return false; }
    }
    rgb->assign(np * 3, 0.0);
    m_finalT.assign(np, 1.0);
    for (size_t i = 0; i < np; ++i) {
        (*rgb)[i * 3 + 0] = double(g.stage[i * 4 + 0]);
        (*rgb)[i * 3 + 1] = double(g.stage[i * 4 + 1]);
        (*rgb)[i * 3 + 2] = double(g.stage[i * 4 + 2]);
        m_finalT[i] = double(g.stage[i * 4 + 3]);
    }
    g.stage.assign(np, 0.0f);
    {
        ImageView v;
        v.desc = lastD;
        v.data = reinterpret_cast<uint8_t*>(g.stage.data());
        if (!g.ctx->Readback(g.last, &v)) { m_gpuNote = "readback failed"; return false; }
    }
    m_lastIdx.assign(np, 0);
    for (size_t i = 0; i < np; ++i) m_lastIdx[i] = int(g.stage[i]);
    return true;
}

bool SplatRaster::BackPixelGpu(const SplatCam& cam, const RasterOptions& opt,
                               const std::vector<double>& dRgb,
                               const std::vector<size_t>& offset,
                               std::vector<Grad2>* entry) {
    Gpu& g = *m_gpu;
    if (g.failed || !g.ready || !g.ctx || !g.ctx->Ready()) return false;
    std::lock_guard<std::mutex> lock(g.ctx->SubmitMutex());

    // The loss gradient, with the forward pass's transmittance beside it.
    const size_t np = size_t(cam.w) * size_t(cam.h);
    const ImageDesc dD{cam.w, cam.h, Format::RGBA32F};
    g.stage.assign(np * 4, 0.0f);
    for (size_t i = 0; i < np; ++i) {
        g.stage[i * 4 + 0] = float(dRgb[i * 3 + 0]);
        g.stage[i * 4 + 1] = float(dRgb[i * 3 + 1]);
        g.stage[i * 4 + 2] = float(dRgb[i * 3 + 2]);
        g.stage[i * 4 + 3] = float(m_finalT[i]);
    }
    if (!g.Ensure(g.drgbt, dD) || !g.Put(g.drgbt, dD)) return false;

    const size_t entries = offset.back();
    const ImageDesc eD = FlatDesc(std::max<size_t>(1, entries * 3), Format::RGBA32F);
    if (!g.Ensure(g.egrad, eD)) return false;

    std::string err;
    const std::vector<uint32_t> c{
        uint32_t(cam.w), uint32_t(cam.h), uint32_t(g.tilesX), uint32_t(kTexW),
        Bits(float(opt.background.x)), Bits(float(opt.background.y)),
        Bits(float(opt.background.z)), Bits(float(opt.minAlpha)),
        Bits(float(opt.maxAlpha))};
    if (!g.ctx->Dispatch(g.bwd, {&g.proj, &g.list, &g.offs, &g.drgbt},
                         {&g.egrad, &g.last}, c, &err, uint32_t(m_tilesX),
                         uint32_t(m_tilesY))) {
        m_gpuNote = "splat backward dispatch failed: " + err;
        return false;
    }

    g.stage.assign(size_t(eD.width) * size_t(eD.height) * 4, 0.0f);
    ImageView v;
    v.desc = eD;
    v.data = reinterpret_cast<uint8_t*>(g.stage.data());
    if (!g.ctx->Readback(g.egrad, &v)) { m_gpuNote = "readback failed"; return false; }

    for (size_t e = 0; e < entries; ++e) {
        const float* s = &g.stage[e * 12];
        Grad2& o = (*entry)[e];
        o.u = s[0]; o.v = s[1];
        o.conic[0] = s[2]; o.conic[1] = s[3]; o.conic[2] = s[4];
        o.color[0] = s[5]; o.color[1] = s[6]; o.color[2] = s[7];
        o.opacity = s[8];
    }
    return true;
}

}  // namespace tglab
