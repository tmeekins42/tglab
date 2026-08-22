// View — a dockable viewer panel.
//
// M1 ships only ImageViewPanel. The interface exists so that a Viewport3D
// (camera, depth buffer, point/splat pipeline) can dock alongside 2D panels
// later without special-casing anything in the app loop.
#pragma once

#include <memory>
#include <string>

#include "../core/data.h"

namespace tglab {

class Device;

class View {
public:
    virtual ~View() = default;

    virtual const std::string& Name() const = 0;

    // Called once per frame with this view's image, or null when the pipeline
    // has not produced one yet (a newly declared viewer, or a failed run).
    virtual void Draw(Device& dev, Image* image) = 0;

    // Per-frame hints, both optional: a view that ignores them still draws
    // correctly, just without skipping redundant work.
    //
    // The version changes only when this view's own pixels do, so an unchanged
    // view can skip its texture upload. The GPU source, when present, is the
    // result still resident on the device -- converting from it needs no
    // readback. A 3D viewport, for instance, wants neither.
    virtual void SetContentVersion(uint64_t) {}
    virtual void SetGpuSource(std::shared_ptr<SharedGpuTexture>) {}
};

} // namespace tglab
