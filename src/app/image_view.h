#pragma once

#include <functional>
#include <string>

#include "imgui.h"

#include "../gpu/texture.h"
#include "view.h"
#include "../core/image.h"

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

    // Screen pixels per FULL-RESOLUTION image pixel, as of the last Draw().
    //
    // This is what the proxy scale must be derived from, and deriving it from
    // the panel width instead was wrong: that assumes the image is fitted, so
    // zooming to 1:1 still asked for a 20% proxy and the picture went soft
    // during every drag. Zoom answers the question directly -- at 1:1 it is
    // 1.0 and no proxy is wanted; fitted it is small and a proxy is free.
    //
    // 0 until the panel has drawn once.
    float LastZoom() const { return m_lastZoom; }

    // The part of the full-resolution image this panel could see at its last
    // Draw(), or an empty rect when it drew nothing.
    //
    // The app takes the UNION over visible panels before asking the pipeline
    // for it: a second viewer showing the whole frame means the whole frame is
    // needed however far this one is zoomed in.
    ImageRect VisibleRect() const { return m_visRect; }

    // True when the panel was actually on screen at its last Draw(): its tab is
    // on top and it is not collapsed. Distinct from Focused(), which needs a
    // click -- a tab can be the one you are looking at without having focus.
    bool Visible() const { return m_visible; }
    void Draw(Device& dev, Image* image) override;

    void SetSharedCamera(ViewCamera* cam) { m_shared = cam; }

    // Called when the user picks "Save image..." from this panel's right-click
    // menu. A callback rather than a save here: the panel has a texture and a
    // name, while the pixels, the file dialog and the error reporting all live
    // in the app.
    void SetSaveHandler(std::function<void(const std::string&)> f) {
        m_onSave = std::move(f);
    }

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
    std::function<void(const std::string&)> m_onSave;
    ViewCamera  m_own;
    ViewCamera* m_shared  = nullptr;
    uint64_t    m_version = 0;
    bool        m_focused = false;
    float       m_lastZoom = 0.0f;   // see LastZoom
    ImageRect   m_visRect{};        // see VisibleRect
    bool        m_visible = false;
    bool        m_loupe   = false;
};

} // namespace tglab
