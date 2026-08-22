#pragma once

#include "imgui.h"

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

    // True when this viewer had focus at its last Draw(). Lets the info panel
    // describe what the user is actually looking at rather than a fixed image.
    bool Focused() const { return m_focused; }

    // True when the panel was actually on screen at its last Draw(): its tab is
    // on top and it is not collapsed. Distinct from Focused(), which needs a
    // click -- a tab can be the one you are looking at without having focus.
    bool Visible() const { return m_visible; }
    void Draw(Device& dev, Image* image) override;

    void SetSharedCamera(ViewCamera* cam) { m_shared = cam; }

    // Bumped by the app whenever the pipeline produced new pixels, so the
    // texture re-uploads only then rather than every frame.
    void SetContentVersion(uint64_t v) override { m_version = v; }

    // The result still resident on the GPU, when there is one. Set alongside
    // the version each frame; drawing prefers it over the Image, because
    // converting from it needs no readback.
    void SetGpuSource(std::shared_ptr<SharedGpuTexture> g) override { m_gpuSrc = std::move(g); }

    // The 1:1 loupe, for inspecting individual pixels.
    void SetLoupe(bool on) { m_loupe = on; }
    bool Loupe() const { return m_loupe; }

private:
    // Drawn from CPU pixels rather than the GPU texture, so the magnified view
    // is truly point-sampled. See the definition for why.
    void DrawLoupe(Device& dev, Image& img, const ImVec2& mouse, const ImVec2& imgOrigin,
                   float zoom, ImDrawList* dl);
    static void SampleRgb(const ImageView& v, int x, int y, float* rgb);

    std::string m_name;
    GpuTexture  m_tex;

    // Kept alive for as long as this view might draw from it -- the worker can
    // free the pipeline's outputs on its own thread at any point.
    std::shared_ptr<SharedGpuTexture> m_gpuSrc;
    ViewCamera  m_own;
    ViewCamera* m_shared  = nullptr;
    uint64_t    m_version = 0;
    bool        m_focused = false;
    bool        m_visible = false;
    bool        m_loupe   = false;
};

} // namespace tglab
