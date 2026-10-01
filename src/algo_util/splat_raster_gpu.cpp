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
#include "splat_kernels.h"

#include <algorithm>
#include <cstring>
#include <mutex>

#include "../gpu/compute.h"

namespace tglab {
namespace {

constexpr int kTexW = 2048;

ImageDesc FlatDesc(size_t texels, Format f) {
    ImageDesc d;
    d.width  = kTexW;
    d.height = int(std::max<size_t>(1, (texels + kTexW - 1) / kTexW));
    d.format = f;
    return d;
}


}  // namespace

struct SplatRaster::Gpu {
    ComputeContext* ctx = nullptr;
    ComputeKernel   fwd, bwd;
    bool            ready = false, failed = false;

    // Kept between calls and reallocated only when a size changes: a
    // training run reuses the same dimensions for thousands of iterations.
    GpuImage proj, list, offs, rgbt, last, drgbt, egrad, depth, ddepth, dshift;
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
                               std::vector<double>* rgb,
                               std::vector<double>* depth) {
    Gpu& g = *m_gpu;
    if (g.failed || !g.ctx || !g.ctx->Ready()) return false;

    // ComputeContext has one command queue and no locking of its own.
    std::lock_guard<std::mutex> lock(g.ctx->SubmitMutex());

    if (!g.ready) {
        std::string err;
        if (!g.ctx->CreateKernel(splat_kernels::kForward, "main", "splat_forward", &g.fwd, &err) ||
            !g.ctx->CreateKernel(splat_kernels::kBackward, "main", "splat_backward", &g.bwd, &err)) {
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
        t[4] = float(P.conic[0]); t[5] = float(P.conic[1]); t[6] = float(P.conic[2]); t[7] = float(P.reach);
        t[8] = float(splats[j].color[0]);
        t[9] = float(splats[j].color[1]);
        t[10] = float(splats[j].color[2]);
        t[11] = float(P.depth);
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
    if (!g.Ensure(g.rgbt, rgbtD) || !g.Ensure(g.last, lastD) ||
        !g.Ensure(g.depth, lastD))
        return false;

    g.tilesX = m_tilesX;
    std::string err;
    const std::vector<uint32_t> c{
        uint32_t(m_tilesX), uint32_t(kTexW),
        FloatBits(float(opt.background.x)), FloatBits(float(opt.background.y)),
        FloatBits(float(opt.background.z)), FloatBits(float(opt.minAlpha)),
        FloatBits(float(opt.maxAlpha)), FloatBits(float(opt.tStop))};
    if (!g.ctx->Dispatch(g.fwd, {&g.proj, &g.list, &g.offs},
                         {&g.rgbt, &g.last, &g.depth},
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

    if (depth) {
        g.stage.assign(np, 0.0f);
        ImageView v;
        v.desc = lastD;
        v.data = reinterpret_cast<uint8_t*>(g.stage.data());
        if (!g.ctx->Readback(g.depth, &v)) { m_gpuNote = "readback failed"; return false; }
        depth->assign(np, 0.0);
        for (size_t i = 0; i < np; ++i) (*depth)[i] = double(g.stage[i]);
    }
    return true;
}

bool SplatRaster::BackPixelGpu(const SplatCam& cam, const RasterOptions& opt,
                               const std::vector<double>& dRgb,
                               const std::vector<double>* dDepth,
                               const std::vector<float>* depthShift,
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

    // The depth loss's gradient, zero when there is none: the kernel always
    // reads it, and zero makes the depth terms vanish exactly.
    const ImageDesc ddD{cam.w, cam.h, Format::R32F};
    g.stage.assign(np, 0.0f);
    if (dDepth)
        for (size_t i = 0; i < np; ++i) g.stage[i] = float((*dDepth)[i]);
    if (!g.Ensure(g.ddepth, ddD) || !g.Put(g.ddepth, ddD)) return false;
    g.stage.assign(np, 0.0f);
    if (dDepth && depthShift)
        for (size_t i = 0; i < np; ++i) g.stage[i] = (*depthShift)[i];
    if (!g.Ensure(g.dshift, ddD) || !g.Put(g.dshift, ddD)) return false;

    const size_t entries = offset.back();
    const ImageDesc eD = FlatDesc(std::max<size_t>(1, entries * 3), Format::RGBA32F);
    if (!g.Ensure(g.egrad, eD)) return false;

    std::string err;
    const std::vector<uint32_t> c{
        uint32_t(cam.w), uint32_t(cam.h), uint32_t(g.tilesX), uint32_t(kTexW),
        FloatBits(float(opt.background.x)), FloatBits(float(opt.background.y)),
        FloatBits(float(opt.background.z)), FloatBits(float(opt.minAlpha)),
        FloatBits(float(opt.maxAlpha))};
    if (!g.ctx->Dispatch(g.bwd, {&g.proj, &g.list, &g.offs, &g.drgbt},
                         {&g.egrad, &g.last, &g.ddepth, &g.dshift}, c, &err,
                         uint32_t(m_tilesX),
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
        o.depth = s[9];
    }
    return true;
}

}  // namespace tglab
