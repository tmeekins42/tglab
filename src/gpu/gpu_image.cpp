#include "gpu_image.h"

#include "gpu_budget.h"

#include <cstdio>
#include <cstring>

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
    MarkGpuResident();
    TouchGpuUse(*this);
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

    // Stamped whether or not an upload happened: what the collector wants to
    // know is when this copy was last NEEDED, and a cache hit is exactly the
    // case worth protecting from eviction.
    TouchGpuUse(*this);
    return m_gpu.get();
}


void SharedGpuTextureDeleter::operator()(SharedGpuTexture* p) const noexcept {
    delete p;
}

void DiscardGpuResidency(GpuResidency* g) {
    if (!g) return;
    g->owner = nullptr;   // so its destructor releases rather than recycles
    delete g;
}

bool ReadImageRegion(const Image& img, int x, int y, int w, int h, std::vector<uint8_t>* out) {
    const ImageDesc& d = img.Desc();
    if (w <= 0 || h <= 0 || x < 0 || y < 0 || x + w > d.width || y + h > d.height) return false;
    const size_t bpp = size_t(BytesPerPixel(d.format)), row = size_t(w) * bpp;
    out->resize(row * size_t(h));
    if (img.HasCpu()) {
        ImageView v = const_cast<Image&>(img).MapCpuRead();   // no sync: already here
        if (!v.Valid()) return false;
        for (int r = 0; r < h; ++r)
            std::memcpy(out->data() + size_t(r) * row,
                        v.data + size_t(y + r) * size_t(v.Pitch()) + size_t(x) * bpp, row);
        return true;
    }
    const GpuResidency* g = img.RawGpu();
    if (!g || !g->image.Valid() || !g->owner) return false;
    return g->owner->ReadbackRegion(g->image, x, y, w, h, out->data(), row);
}

bool ReadImageRegions(const Image& img, const std::vector<std::array<int, 4>>& rects,
                      std::vector<std::vector<uint8_t>>* outs) {
    if (img.HasCpu() || !img.RawGpu() || !img.RawGpu()->owner) {
        outs->assign(rects.size(), {});
        for (size_t i = 0; i < rects.size(); ++i)
            if (!ReadImageRegion(img, rects[i][0], rects[i][1], rects[i][2], rects[i][3], &(*outs)[i]))
                return false;
        return true;
    }
    const GpuResidency* g = img.RawGpu();
    return g->image.Valid() && g->owner->ReadbackRegions(g->image, rects, outs);
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
