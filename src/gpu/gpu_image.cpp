#include "gpu_image.h"

#include <cstdio>

namespace tglab {

// Declared in core/image.cpp.
extern bool (*g_readbackFromGpu)(Image&);

void GpuResidencyDeleter::operator()(GpuResidency* p) const noexcept {
    delete p;
}

namespace {

// Ensures the image has a GPU resource of the right description, creating one
// if it is missing or the size/format changed.
GpuResidency* EnsureGpuStorage(Image& img, ComputeContext& ctx) {
    GpuResidency* g = img.RawGpu();
    if (g && g->image.Valid() && g->image.desc == img.Desc() && g->owner == &ctx)
        return g;
    return nullptr;   // caller allocates through Image, which owns the pointer
}

} // namespace

GpuResidency* Image::AcquireGpuWrite(ComputeContext& ctx) {
    if (!m_desc.Valid()) return nullptr;

    if (!EnsureGpuStorage(*this, ctx)) {
        auto fresh = std::unique_ptr<GpuResidency, GpuResidencyDeleter>(new GpuResidency());
        if (!ctx.CreateImage(m_desc, &fresh->image)) return nullptr;
        fresh->owner = &ctx;
        m_gpu = std::move(fresh);
    }

    // Writing on the GPU makes the CPU copy stale. Marking it here — before
    // the dispatch — is what lets a chain of GPU stages skip readback
    // entirely: nothing asks for CPU pixels in between.
    m_res = Residency::Gpu;
    return m_gpu.get();
}

GpuResidency* Image::AcquireGpuRead(ComputeContext& ctx) {
    if (!m_desc.Valid()) return nullptr;

    const bool needUpload = !HasGpu();
    if (!EnsureGpuStorage(*this, ctx)) {
        auto fresh = std::unique_ptr<GpuResidency, GpuResidencyDeleter>(new GpuResidency());
        if (!ctx.CreateImage(m_desc, &fresh->image)) {
            // CreateImage already reported the HRESULT and the size.
            std::fprintf(stderr, "[gpu] residency: allocation failed\n");
            return nullptr;
        }
        fresh->owner = &ctx;
        m_gpu = std::move(fresh);
    }

    // Transfer only when the GPU copy is missing or stale. This is the rule
    // that keeps a chain of GPU stages free of intermediate uploads.
    if (needUpload) {
        if (!HasCpu()) {
            // SAYS WHICH CONDITION FIRED. "could not make an input
            // GPU-resident" covers three unrelated failures -- allocation, an
            // image with pixels nowhere, and a failed upload -- and they want
            // different fixes. Reported here because the caller cannot tell
            // them apart, and chasing the wrong one costs an afternoon.
            std::fprintf(stderr,
                         "[gpu] residency: %dx%d image has pixels nowhere "
                         "(res=%d)\n",
                         m_desc.width, m_desc.height, int(m_res));
            return nullptr;   // nothing anywhere to upload from
        }
        ImageView v = MapCpuRead();
        if (!v.Valid() || !ctx.Upload(v, &m_gpu->image)) {
            std::fprintf(stderr,
                         "[gpu] residency: upload of a %dx%d image failed "
                         "(view %s)\n",
                         m_desc.width, m_desc.height,
                         v.Valid() ? "valid" : "INVALID");
            return nullptr;
        }
        m_res = Residency::Both;
    }
    return m_gpu.get();
}


void SharedGpuTextureDeleter::operator()(SharedGpuTexture* p) const noexcept {
    delete p;
}

std::shared_ptr<SharedGpuTexture> ShareGpuTexture(const Image& img) {
    const GpuResidency* g = img.RawGpu();
    if (!g || !g->image.Valid()) return nullptr;
    return std::make_shared<SharedGpuTexture>(g->image.res, g->image.desc);
}
void InstallGpuResidencyHooks() {
    g_readbackFromGpu = [](Image& img) -> bool {
        GpuResidency* g = img.RawGpu();
        if (!g || !g->image.Valid() || !g->owner) return false;
        ImageView v = img.CpuBufferForFill();
        if (!v.Valid()) return false;
        return g->owner->Readback(g->image, &v);
    };
}

} // namespace tglab
