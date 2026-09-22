#include "gpu_budget.h"

#include <algorithm>
#include <atomic>
#include <cstdio>

#include "../core/image.h"

namespace tglab {

namespace {
// Starts at 1, because 0 on an image means "its GPU copy was never used" and
// has to stay distinguishable from "used at the very first tick".
std::atomic<uint64_t> g_gpuTick{1};
}  // namespace

uint64_t GpuTickNow() { return g_gpuTick.load(std::memory_order_relaxed); }
void     GpuTickAdvance() { g_gpuTick.fetch_add(1, std::memory_order_relaxed); }

void TouchGpuUse(Image& img) {
    img.TouchGpu(GpuTickNow());
}

std::string GpuCollectReport::Summary() const {
    if (!ran || imagesFreed == 0) return {};
    char buf[96];
    std::snprintf(buf, sizeof buf, "freed %.0f MB of video memory (%d image%s)",
                  double(bytesFreed) / (1024.0 * 1024.0), imagesFreed,
                  imagesFreed == 1 ? "" : "s");
    return buf;
}

namespace {

// One eviction candidate: an image holding a GPU copy that is safe to free.
struct Candidate {
    Image*   img  = nullptr;
    uint64_t used = 0;     // when its GPU copy was last touched
    size_t   bytes = 0;
};

// Collects candidates out of one Data, which may be a single image or a whole
// group. A group is the interesting case: it is N images in one palette slot,
// and a bracket of seven raws is most of what fills a card.
void Gather(Data* d, std::vector<Candidate>* out) {
    if (!d) return;

    auto consider = [&](Image& im) {
        // Only a genuine cache copy. HasCpu is the safety condition -- see
        // Image::DropGpuCopy, which refuses the rest anyway; checking here as
        // well keeps the candidate list honest, so `imagesFreed` counts what
        // was actually freed rather than what was attempted.
        if (!im.HasGpu() || !im.HasCpu()) return;
        const size_t bytes = im.Desc().SizeInBytes();
        if (bytes == 0) return;
        out->push_back(Candidate{&im, im.GpuLastUsed(), bytes});
    };

    if (Image* im = std::get_if<Image>(d)) {
        consider(*im);
    } else if (ImageSet* set = std::get_if<ImageSet>(d)) {
        for (Image& im : set->images) consider(im);
    }
}

}  // namespace

GpuCollectReport GpuBudget::Collect(const std::vector<Data*>& pool,
                                    uint64_t used, uint64_t budget,
                                    const GpuBudgetPolicy& policy) const {
    GpuCollectReport r;
    if (budget == 0) return r;   // the driver told us nothing

    const double pct = 100.0 * double(used) / double(budget);
    if (pct < policy.startPct) return r;   // comfortable; do nothing
    r.ran = true;

    // How far under to get. Computed in BYTES rather than by re-measuring the
    // percentage as we go: querying the driver mid-loop reports its own
    // bookkeeping, which lags a Release by an unspecified amount, so a
    // measure-and-check loop either stops early or frees the whole pool.
    const uint64_t target = uint64_t(double(budget) * policy.targetPct / 100.0);
    if (used <= target) return r;
    const uint64_t want = used - target;

    std::vector<Candidate> cands;
    for (Data* d : pool) Gather(d, &cands);

    // LEAST RECENTLY USED FIRST. The image being worked on keeps its copy;
    // the ones from three scripts ago go. Freeing by size instead would evict
    // the 45 MP raw the user is actually dragging, which is both the biggest
    // win and the worst choice.
    std::sort(cands.begin(), cands.end(),
              [](const Candidate& a, const Candidate& b) {
                  if (a.used != b.used) return a.used < b.used;
                  // Same age: take the larger one, so a pass frees fewer,
                  // bigger textures and does less work per byte reclaimed.
                  return a.bytes > b.bytes;
              });

    for (const Candidate& c : cands) {
        if (uint64_t(r.bytesFreed) >= want) break;

        // Too recent to touch, and STOP rather than skip: the list is sorted
        // oldest first, so everything after this is younger still. Protecting
        // the recent ones is what keeps the collector from freeing the image
        // the user is dragging and paying to upload it again next frame.
        if (c.used != 0 && GpuTickNow() - c.used < policy.minAgeTicks) break;

        const size_t freed = c.img->DropGpuCopy();
        if (freed == 0) continue;   // refused: GPU-only, so not a cache
        r.bytesFreed += freed;
        ++r.imagesFreed;
    }
    return r;
}

} // namespace tglab
