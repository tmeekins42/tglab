// Reclaiming video memory when a session approaches the card's budget.
//
// THE PROBLEM, which is retention rather than a leak.
//
// gpu_leak and gpu_leak_scripts both report flat VRAM: re-running a pipeline
// reuses its kernels, scratch and outputs, and switching scripts releases the
// old pipeline's resources. Nothing is leaking. What grows is legitimate: every
// palette image a pipeline touches gains a GPU copy on its first use and keeps
// it forever, because nothing has ever had a reason to let one go. A 45 MP raw
// is ~358 MB in RGBA16F, so half a dozen loaded frames is two gigabytes of
// source copies before a single intermediate exists.
//
// Past the card's budget the driver starts paging, and the symptom is not an
// error -- it is everything becoming slow for a reason that is invisible from
// inside the app. Worse, an allocation can simply fail, and a failed
// CreateImage turns into a CPU fallback several layers away from the cause.
//
// WHAT IS SAFE TO FREE, and it is a narrow set.
//
// An image resident on BOTH sides holds its GPU copy as a cache: the pixels
// also exist in system memory, so releasing the texture loses nothing and the
// next AcquireGpuRead uploads again. That upload is the entire cost, which is
// why this evicts least-recently-used rather than whatever it finds first --
// the image being dragged must keep its copy, or the collector turns a memory
// problem into a bandwidth problem.
//
// An image resident ONLY on the GPU is not a cache. Its pixels exist nowhere
// else, and Image::DropGpuCopy refuses it for that reason.
//
// THREADING, which decides where this can run at all.
//
// The palette's sources are shared with the worker by shared_ptr, and the
// worker touches their residency while a run is in flight. Freeing a texture
// from the UI thread at the same moment is a data race on the same pointer the
// worker is reading, and the failure would be a use-after-free rather than
// anything as friendly as a wrong picture. So collection happens on the UI
// thread ONLY when the worker is idle, which is also the only moment the
// answer to "what is still needed" is stable.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "../core/data.h"

namespace tglab {

class Image;

// What one collection pass did, for the status line and the tests.
struct GpuCollectReport {
    size_t bytesFreed = 0;
    int    imagesFreed = 0;
    bool   ran = false;      // false when the budget was comfortable

    std::string Summary() const;
};

// Decides when to collect and how much to free.
//
// A HIGH-WATER RULE rather than a fixed cap: collection starts when usage
// crosses `startPct` of the driver's reported budget and frees until it is
// back under `targetPct`. One threshold would thrash -- freeing a single
// texture drops usage just below the line, the next allocation crosses it
// again, and the collector runs every frame while the picture reuploads.
struct GpuBudgetPolicy {
    double startPct  = 80.0;   // begin collecting above this
    double targetPct = 65.0;   // free until back under this

    // Never evict an image used within this many ticks. A tick is one
    // collection opportunity (one idle frame), so this protects whatever the
    // user is actually working on from being freed and immediately re-uploaded.
    uint64_t minAgeTicks = 2;
};

// The clock the age stamps are measured in.
//
// A free function over one atomic rather than a member of GpuBudget, because
// the two ends live on different threads: the WORKER stamps images as it
// acquires them, while the UI thread advances the clock and collects. An
// atomic counter is the whole of what they need to share -- and it is a
// counter, so a stamp racing an increment is off by one tick, which changes
// nothing about which image is oldest.
uint64_t GpuTickNow();
void     GpuTickAdvance();

// Records that `img`'s GPU copy was just used, so the collector treats it as
// recent. Called from the acquire path on the worker thread.
void TouchGpuUse(Image& img);

class GpuBudget {
public:
    // Advances the clock. Called once per idle frame, whether or not anything
    // is collected, so "recently used" means recently in wall-clock terms
    // rather than in collections.
    void Tick() const { GpuTickAdvance(); }
    uint64_t Now() const { return GpuTickNow(); }

    // Frees GPU copies from `pool` until usage is back under the target, or
    // until nothing more can safely be freed.
    //
    // `used` and `budget` come from the device. Passing them in rather than
    // querying keeps this testable without a D3D12 device, which matters
    // because the eviction ORDER is the part worth testing and it has nothing
    // to do with the driver.
    GpuCollectReport Collect(const std::vector<Data*>& pool,
                             uint64_t used, uint64_t budget,
                             const GpuBudgetPolicy& policy = {}) const;
};

} // namespace tglab
