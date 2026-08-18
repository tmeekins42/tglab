#pragma once

#include "../gpu/texture.h"
#include "view.h"

namespace tglab {

// Pan/zoom state shared between image views, so comparing two panels keeps
// them aligned (M2 turns this into an explicit sync toggle).
struct ViewCamera {
    float zoom = 1.0f;
    float panX = 0.0f;
    float panY = 0.0f;
    bool  fit  = true;
};

class ImageViewPanel : public View {
public:
    explicit ImageViewPanel(std::string name) : m_name(std::move(name)) {}

    const std::string& Name() const override { return m_name; }
    void Draw(Device& dev, Data* data) override;

    void SetSharedCamera(ViewCamera* cam) { m_shared = cam; }

    // Bumped by the app whenever the pipeline produced new pixels, so the
    // texture re-uploads only then rather than every frame.
    void SetContentVersion(uint64_t v) { m_version = v; }

private:
    std::string m_name;
    GpuTexture  m_tex;
    ViewCamera  m_own;
    ViewCamera* m_shared  = nullptr;
    uint64_t    m_version = 0;
};

} // namespace tglab
