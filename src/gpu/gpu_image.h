// The GPU half of Image residency.
//
// core/image.h holds only an opaque pointer to GpuResidency, so the dependency
// points one way: gpu -> core. This file supplies the definition and the
// acquire/readback implementations.
#pragma once

#include "../core/image.h"
#include "compute.h"

namespace tglab {

struct GpuResidency {
    GpuImage        image;
    ComputeContext* owner = nullptr;   // context that created `image`

    ~GpuResidency() { image.Release(); }
};

// A GPU texture handed from the worker to the UI for display.
//
// The worker's compute context and ImGui share one ID3D12Device, and compute
// images are created with ALLOW_SIMULTANEOUS_ACCESS, so the UI's direct queue
// can read one while the compute queue owns it -- no readback, no cross-queue
// transition. That is what the residency design was for; it was simply never
// wired through to the viewers.
//
// Lifetime is the whole difficulty. The worker frees a stage's outputs whenever
// the cache is replaced, on its own thread, while the UI may still be drawing
// the previous frame from that resource. So this holds its own reference:
// AddRef on construction, Release on destruction, and the UI keeps it alive as
// long as it needs it. Refcounting rather than the device's deferred-release
// list, because that list is a UI-thread facility and this free happens on the
// worker.
struct SharedGpuTexture {
    ID3D12Resource* res = nullptr;
    ImageDesc       desc{};

    SharedGpuTexture(ID3D12Resource* r, const ImageDesc& d) : res(r), desc(d) {
        if (res) res->AddRef();
    }
    ~SharedGpuTexture() { if (res) res->Release(); }

    // Non-copyable: it is always held through the shared_ptr that owns it.
    SharedGpuTexture(const SharedGpuTexture&)            = delete;
    SharedGpuTexture& operator=(const SharedGpuTexture&) = delete;
};

// Installs the readback hook that Image::MapCpuRead() uses when an image is
// GPU-resident only. Call once at startup, on the thread that owns the
// ComputeContext.
void InstallGpuResidencyHooks();

} // namespace tglab
