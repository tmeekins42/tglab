#include "image_view.h"

#include <algorithm>

#include "../gpu/device.h"
#include "imgui.h"

namespace tglab {

void ImageViewPanel::Draw(Device& dev, Image* img) {
    // Keep the panel identity stable so ImGui remembers docking across runs.
    if (!ImGui::Begin(m_name.c_str())) {
        // Begin() returns false when the panel is collapsed or its tab is
        // hidden behind another -- either way the user cannot see it.
        m_focused = false;
        m_visible = false;
        ImGui::End();
        return;
    }
    // Visible and focused are different questions, and the info panel needs
    // both. A tab that is on top is *visible* but does not have keyboard focus
    // until it is clicked, so a focus test alone reports nothing at startup --
    // the histogram then described the source image while the user was plainly
    // looking at a result.
    m_visible = true;
    m_focused = ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows);

    if (!img || !img->Valid()) {
        ImGui::TextDisabled("computing...");
        ImGui::End();
        return;
    }

    if (!m_tex.Update(dev, *img, m_version)) {
        ImGui::TextDisabled("could not upload image");
        ImGui::End();
        return;
    }

    ViewCamera& cam = m_shared ? *m_shared : m_own;

    const float iw = float(m_tex.Width());
    const float ih = float(m_tex.Height());

    ImGui::Text("%.0f x %.0f  %s", iw, ih, FormatName(img->Desc().format));
    ImGui::SameLine();
    if (ImGui::SmallButton("Fit")) cam.fit = true;
    ImGui::SameLine();
    if (ImGui::SmallButton("1:1")) { cam.fit = false; cam.zoom = 1.0f; cam.panX = cam.panY = 0.0f; }
    ImGui::SameLine();
    // Sub-1% zooms need a decimal, or the readout just says "0%".
    const float zoomPct = cam.zoom * 100.0f;
    ImGui::Text(zoomPct < 10.0f ? "%.1f%%" : "%.0f%%", zoomPct);
    ImGui::SameLine();
    // Per-viewer rather than global: comparing two panels usually means
    // inspecting one of them.
    if (ImGui::SmallButton(m_loupe ? "[loupe]" : "loupe")) m_loupe = !m_loupe;
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("1:1 magnifier at the cursor, point-sampled, with the "
                          "pixel's actual values");

    // End the toolbar row, then take the whole remaining content rect as the
    // canvas. GetContentRegion* is measured from the window's content edges,
    // so it stays correct regardless of where the toolbar left the cursor.
    ImGui::NewLine();

    const ImVec2 origin = ImVec2(ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMin().x,
                                 ImGui::GetCursorScreenPos().y);
    const ImVec2 contentMax = ImGui::GetWindowContentRegionMax();
    const ImVec2 region(
        std::max(contentMax.x - ImGui::GetWindowContentRegionMin().x, 1.0f),
        std::max(ImGui::GetWindowPos().y + contentMax.y - origin.y, 1.0f));

    if (cam.fit && iw > 0 && ih > 0) {
        // Clamp: a panel that is briefly tiny (undocked, or mid-resize) would
        // otherwise fit to ~0% and show nothing at all.
        cam.zoom = std::clamp(std::min(region.x / iw, region.y / ih), 0.01f, 64.0f);
        cam.panX = cam.panY = 0.0f;
    }

    const ImVec2 size(iw * cam.zoom, ih * cam.zoom);

    // Draw through the draw list rather than ImGui::Image() so the image never
    // advances the cursor past the panel, and clip it to the visible region —
    // otherwise a zoomed image spills over neighbouring panels.
    const ImVec2 offset(std::max(0.0f, (region.x - size.x) * 0.5f) + cam.panX,
                        std::max(0.0f, (region.y - size.y) * 0.5f) + cam.panY);
    const ImVec2 p0(origin.x + offset.x, origin.y + offset.y);
    const ImVec2 p1(p0.x + size.x, p0.y + size.y);

    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->PushClipRect(origin, ImVec2(origin.x + region.x, origin.y + region.y), true);
    dl->AddImage(ImTextureRef(static_cast<ImTextureID>(m_tex.Handle().ptr)), p0, p1);
    dl->PopClipRect();

    // Wheel zooms, left-drag pans — both disable fit so the user stays in control.
    ImGui::SetCursorScreenPos(origin);
    ImGui::InvisibleButton("##canvas", region,
                           ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
    if (ImGui::IsItemHovered()) {
        const float wheel = ImGui::GetIO().MouseWheel;
        if (wheel != 0.0f) {
            // Zoom about the cursor, not the panel centre.
            //
            // Scaling alone moves whatever you were looking at off-screen, so
            // inspecting a detail means zoom, pan back, zoom, pan back. The fix
            // is to work out which image pixel is under the cursor, apply the
            // zoom, then shift the pan so that same pixel lands back under it.
            const ImVec2 mouse = ImGui::GetIO().MousePos;
            const float oldZoom = cam.zoom;

            cam.fit = false;
            cam.zoom = std::clamp(cam.zoom * (1.0f + wheel * 0.1f), 0.02f, 64.0f);

            if (oldZoom > 1e-6f && cam.zoom != oldZoom) {
                // Where the cursor sits within the drawn image, in image pixels.
                const float imgX = (mouse.x - p0.x) / oldZoom;
                const float imgY = (mouse.y - p0.y) / oldZoom;
                // What that costs in screen space at the new zoom, minus the
                // centring term, which also shifts as the image resizes.
                const ImVec2 newSize(iw * cam.zoom, ih * cam.zoom);
                const float newCentreX = std::max(0.0f, (region.x - newSize.x) * 0.5f);
                const float newCentreY = std::max(0.0f, (region.y - newSize.y) * 0.5f);
                cam.panX = mouse.x - origin.x - newCentreX - imgX * cam.zoom;
                cam.panY = mouse.y - origin.y - newCentreY - imgY * cam.zoom;
            }
        }
    }
    if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
        const ImVec2 d = ImGui::GetIO().MouseDelta;
        cam.fit = false;
        cam.panX += d.x;
        cam.panY += d.y;
    }

    if (m_loupe && ImGui::IsItemHovered())
        DrawLoupe(*img, ImGui::GetIO().MousePos, p0, cam.zoom, dl);

    ImGui::End();
}

// A 1:1 magnifier at the cursor, drawn pixel by pixel from the CPU image.
//
// Deliberately not a scaled draw of the GPU texture: ImGui's DX12 backend
// picks its sampler per pipeline state, with no public per-draw override, so a
// magnified texture would come out bilinear -- which is exactly wrong for
// pixel snooping, since it invents values between the ones you are trying to
// read. Reading the CPU pixels directly gives true point sampling and, as a
// bonus, the real numbers to print underneath.
void ImageViewPanel::DrawLoupe(Image& img, const ImVec2& mouse, const ImVec2& imgOrigin,
                               float zoom, ImDrawList* dl) {
    if (zoom <= 1e-6f) return;

    ImageView v = img.MapCpuRead();
    if (!v.Valid()) return;

    // Which image pixel the cursor is over.
    const int cx = int((mouse.x - imgOrigin.x) / zoom);
    const int cy = int((mouse.y - imgOrigin.y) / zoom);
    if (cx < 0 || cy < 0 || cx >= v.desc.width || cy >= v.desc.height) return;

    // Odd, so there is a true centre pixel to put the crosshair on.
    constexpr int kSamples = 15;
    constexpr float kCell  = 12.0f;
    constexpr int kHalf    = kSamples / 2;
    const float side = kSamples * kCell;

    // Placed away from the cursor so it does not cover what is being inspected,
    // and flipped near the panel edges so it stays on screen.
    const ImVec2 wmin = ImGui::GetWindowPos();
    const ImVec2 wmax(wmin.x + ImGui::GetWindowSize().x, wmin.y + ImGui::GetWindowSize().y);
    float lx = mouse.x + 24.0f;
    float ly = mouse.y + 24.0f;
    if (lx + side > wmax.x) lx = mouse.x - 24.0f - side;
    if (ly + side + 34.0f > wmax.y) ly = mouse.y - 24.0f - side - 34.0f;

    dl->AddRectFilled(ImVec2(lx - 2, ly - 2), ImVec2(lx + side + 2, ly + side + 2),
                      IM_COL32(20, 20, 24, 240));

    for (int j = 0; j < kSamples; ++j) {
        for (int i = 0; i < kSamples; ++i) {
            const int sx = std::clamp(cx - kHalf + i, 0, v.desc.width - 1);
            const int sy = std::clamp(cy - kHalf + j, 0, v.desc.height - 1);

            float rgb[3];
            SampleRgb(v, sx, sy, rgb);

            const ImU32 col = IM_COL32(int(std::clamp(rgb[0], 0.0f, 1.0f) * 255.0f),
                                       int(std::clamp(rgb[1], 0.0f, 1.0f) * 255.0f),
                                       int(std::clamp(rgb[2], 0.0f, 1.0f) * 255.0f), 255);
            const ImVec2 a(lx + i * kCell, ly + j * kCell);
            dl->AddRectFilled(a, ImVec2(a.x + kCell, a.y + kCell), col);
        }
    }

    // Centre marker, drawn as an outline so it never hides the pixel it marks.
    const ImVec2 c0(lx + kHalf * kCell, ly + kHalf * kCell);
    dl->AddRect(c0, ImVec2(c0.x + kCell, c0.y + kCell), IM_COL32(255, 255, 0, 255), 0, 0, 2.0f);

    // The actual values. Float formats keep three decimals -- a raw image lives
    // in 0..1 and beyond, where "0.5" and "0.512" are meaningfully different.
    float centre[3];
    SampleRgb(v, cx, cy, centre);
    char buf[128];
    if (v.desc.format == Format::RGBA8)
        std::snprintf(buf, sizeof buf, "(%d, %d)  %d %d %d", cx, cy,
                      int(centre[0] * 255.0f + 0.5f), int(centre[1] * 255.0f + 0.5f),
                      int(centre[2] * 255.0f + 0.5f));
    else
        std::snprintf(buf, sizeof buf, "(%d, %d)  %.3f %.3f %.3f", cx, cy,
                      centre[0], centre[1], centre[2]);

    dl->AddRectFilled(ImVec2(lx - 2, ly + side + 2), ImVec2(lx + side + 2, ly + side + 22),
                      IM_COL32(20, 20, 24, 240));
    dl->AddText(ImVec2(lx + 2, ly + side + 5), IM_COL32(230, 230, 230, 255), buf);
}

// One pixel as linear RGB, whatever the storage format. Mirrors what the
// viewer's own upload does, so the loupe shows the same colours as the image.
void ImageViewPanel::SampleRgb(const ImageView& v, int x, int y, float* rgb) {
    switch (v.desc.format) {
        case Format::R32F: {
            // A mosaic holds raw sensor counts, so scale by its own levels --
            // otherwise every sample reads as pure white.
            const float black = v.desc.blackLevel;
            const float range = std::max(v.desc.whiteLevel - black, 1e-6f);
            const float s = (*v.At<float>(x, y) - black) / range;
            rgb[0] = rgb[1] = rgb[2] = s;
            break;
        }
        case Format::RGBA32F: {
            const float* p = v.At<float>(x, y);
            for (int c = 0; c < 3; ++c) rgb[c] = p[c];
            break;
        }
        case Format::RGBA16F: {
            const uint16_t* p = v.At<uint16_t>(x, y);
            for (int c = 0; c < 3; ++c) rgb[c] = HalfToFloat(p[c]);
            break;
        }
        default: {   // RGBA8
            const uint8_t* p = v.At<uint8_t>(x, y);
            for (int c = 0; c < 3; ++c) rgb[c] = float(p[c]) / 255.0f;
            break;
        }
    }
}

} // namespace tglab
