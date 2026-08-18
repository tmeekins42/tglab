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

// Installs the readback hook that Image::MapCpuRead() uses when an image is
// GPU-resident only. Call once at startup, on the thread that owns the
// ComputeContext.
void InstallGpuResidencyHooks();

} // namespace tglab
