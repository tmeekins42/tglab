// View — a dockable viewer panel.
//
// M1 ships only ImageViewPanel. The interface exists so that a Viewport3D
// (camera, depth buffer, point/splat pipeline) can dock alongside 2D panels
// later without special-casing anything in the app loop.
#pragma once

#include <string>

#include "../core/data.h"

namespace tglab {

class Device;

class View {
public:
    virtual ~View() = default;

    virtual const std::string& Name() const = 0;

    // Called once per frame with the data this view was declared on
    // (null when the pipeline produced nothing for it).
    virtual void Draw(Device& dev, Data* data) = 0;
};

} // namespace tglab
