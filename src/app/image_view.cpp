#include "image_view.h"

#include <algorithm>

#include "../gpu/device.h"
#include "imgui.h"

namespace tglab {

void ImageViewPanel::Draw(Device& dev, Image* img) {
    // Keep the panel identity stable so ImGui remembers docking across runs.
    if (!ImGui::Begin(m_name.c_str())) {
        m_focused = false;
        ImGui::End();
        return;
    }
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
            cam.fit = false;
            cam.zoom = std::clamp(cam.zoom * (1.0f + wheel * 0.1f), 0.02f, 64.0f);
        }
    }
    if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
        const ImVec2 d = ImGui::GetIO().MouseDelta;
        cam.fit = false;
        cam.panX += d.x;
        cam.panY += d.y;
    }

    ImGui::End();
}

} // namespace tglab
