#include "image_view.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

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

    // "Is there something to draw?" -- which is not the same as
    // Image::Valid(), and the difference matters now that a result can live
    // only on the GPU. Such an image holds a descriptor and no pixels, so
    // Image::Valid() is false (it asks whether pixels exist *somewhere*), yet
    // there is a perfectly good texture to convert from.
    //
    // Getting this wrong showed every GPU-resident viewer as "computing..."
    // forever, on a run the status bar reported as finished.
    const bool haveSomething = m_gpuSrc || (img && img->Valid());
    if (!haveSomething || !img || !img->Desc().Valid()) {
        ImGui::TextDisabled("computing...");
        ImGui::End();
        return;
    }

    // Prefer the GPU source when the result is still resident there: converting
    // from it is a dispatch, where the CPU path is a readback plus a scalar
    // conversion plus an upload. Falls back for a CPU-only stage, which has no
    // shared texture -- a script can mix the two.
    // Draw straight from the GPU result unless TGLAB_CPUDISPLAY=1 forces the
    // readback path (kept as an escape hatch for comparison and debugging).
    //
    // This path previously hung the device. The cause was a second
    // shader-visible descriptor heap: only one can be bound at a time, and
    // descriptor heaps resolve when the command list EXECUTES rather than when
    // it records, so rebinding ImGui's heap at the end of the frame left the
    // conversion's table pointing into the wrong heap. GPU-based validation
    // named it ("Invalid resource pointed to by descriptor") and DRED showed
    // both queues stalled on a dispatch. The conversion's descriptors now live
    // in ImGui's heap, so one heap stays bound throughout. See texture.cpp.
    static const bool kCpuDisplay = [] {
        size_t len = 0;
        return getenv_s(&len, nullptr, 0, "TGLAB_CPUDISPLAY") == 0 && len > 0;
    }();
    const bool useGpu = m_gpuSrc && !kCpuDisplay;
    const bool ok = useGpu ? m_tex.UpdateFromGpu(dev, *m_gpuSrc, m_version)
                           : m_tex.Update(dev, *img, m_version);
    if (!ok) {
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

    // Clipping and gamut toggles, each with a swatch of the colour it paints.
    //
    // The swatch is the point: three overlays in three colours is more than a
    // label can carry, and a user should not have to remember that yellow means
    // gamut rather than clipping. Drawn as a filled square in the button's own
    // colour slots so it reads as a swatch rather than a coloured button.
    //
    // Global rather than per-viewer, unlike the loupe. The loupe is for
    // inspecting ONE panel while comparing; these answer "where is this frame
    // losing data", which is a property of the image and wants the same answer
    // everywhere.
    {
        uint32_t mask = GpuTexture::GetClipOverlay();
        const struct { uint32_t bit; ImVec4 col; const char* label; const char* tip; } kOverlays[] = {
            {GpuTexture::kOverlayWhite,      ImVec4(1.0f, 0.15f, 0.15f, 1.0f), "hi",
             "Blown highlights: at or above the sensor's white level. The detail "
             "is gone and no adjustment recovers it."},
            {GpuTexture::kOverlayBlack,      ImVec4(0.15f, 0.35f, 1.0f, 1.0f), "lo",
             "Crushed shadows: at or below zero."},
            {GpuTexture::kOverlayOutOfGamut, ImVec4(1.0f, 0.85f, 0.10f, 1.0f), "gamut",
             "Out of gamut: a real colour the sensor captured that sRGB cannot "
             "represent. NOT a loss -- it survives editing and returns to range "
             "if you desaturate."},
        };
        for (const auto& o : kOverlays) {
            ImGui::SameLine();
            const bool on = (mask & o.bit) != 0;

            // The swatch. Filled when the overlay is on, outlined when off, so
            // the state is legible without reading the label.
            const ImVec2 p = ImGui::GetCursorScreenPos();
            const float  sz = ImGui::GetFontSize() * 0.75f;
            const float  yOff = (ImGui::GetFrameHeight() - sz) * 0.5f;
            ImDrawList* dl = ImGui::GetWindowDrawList();
            const ImU32 col = ImGui::ColorConvertFloat4ToU32(o.col);
            if (on) dl->AddRectFilled(ImVec2(p.x, p.y + yOff),
                                      ImVec2(p.x + sz, p.y + yOff + sz), col, 2.0f);
            else    dl->AddRect(ImVec2(p.x, p.y + yOff),
                                ImVec2(p.x + sz, p.y + yOff + sz), col, 2.0f);
            ImGui::Dummy(ImVec2(sz + 3.0f, sz));

            ImGui::SameLine(0.0f, 0.0f);
            char btn[32];
            std::snprintf(btn, sizeof btn, on ? "[%s]" : "%s", o.label);
            if (ImGui::SmallButton(btn)) GpuTexture::SetClipOverlay(mask ^ o.bit);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", o.tip);
        }
    }

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
        DrawLoupe(dev, *img, ImGui::GetIO().MousePos, p0, cam.zoom, dl);

    ImGui::End();
}

// A 1:1 magnifier at the cursor, drawn pixel by pixel.
//
// Deliberately not a scaled draw of the GPU texture: ImGui's DX12 backend picks
// its sampler per pipeline state, with no public per-draw override, so a
// magnified texture would come out bilinear -- exactly wrong for pixel
// snooping, since it invents values between the ones you are trying to read.
// Drawing one filled rect per sample gives true point sampling and, as a bonus,
// the real numbers to print underneath.
//
// It needs actual pixel values, which the display path no longer brings back:
// a GPU-resident result is converted on the device and never read. So the loupe
// fetches just its own window -- 15x15 samples, a few kilobytes -- rather than
// the whole image, and only while it is switched on.
void ImageViewPanel::DrawLoupe(Device& dev, Image& img, const ImVec2& mouse,
                               const ImVec2& imgOrigin, float zoom, ImDrawList* dl) {
    if (zoom <= 1e-6f) return;

    const ImageDesc& d = img.Desc();
    if (!d.Valid()) return;

    // Which image pixel the cursor is over.
    const int cx = int((mouse.x - imgOrigin.x) / zoom);
    const int cy = int((mouse.y - imgOrigin.y) / zoom);
    if (cx < 0 || cy < 0 || cx >= d.width || cy >= d.height) return;

    // Odd, so there is a true centre pixel to put the crosshair on.
    constexpr int kSamples = 15;
    constexpr float kCell  = 12.0f;
    constexpr int kHalf    = kSamples / 2;
    const float side = kSamples * kCell;

    // The window, clamped so it stays inside the image. `ox`/`oy` are its
    // top-left in image space, which the sampling loop below indexes from.
    const int ox = std::clamp(cx - kHalf, 0, std::max(0, d.width  - kSamples));
    const int oy = std::clamp(cy - kHalf, 0, std::max(0, d.height - kSamples));
    const int ow = std::min(kSamples, d.width  - ox);
    const int oh = std::min(kSamples, d.height - oy);

    // Either the region comes back from the GPU, or the image already has CPU
    // pixels (a CPU-only stage). One of the two must work.
    ImageView v{};
    std::vector<uint8_t> region;
    ImageDesc regionDesc = d;
    if (m_gpuSrc) {
        if (!GpuTexture::ReadRegion(dev, *m_gpuSrc, ox, oy, ow, oh, &region)) return;
        regionDesc.width  = ow;
        regionDesc.height = oh;
        v.data = region.data();
        v.desc = regionDesc;
    } else {
        v = img.MapCpuRead();
        if (!v.Valid()) return;
    }

    // Sampling is relative to the region when one was fetched, and absolute
    // otherwise -- this is the only difference between the two paths.
    const int sampleOx = m_gpuSrc ? 0 : ox;
    const int sampleOy = m_gpuSrc ? 0 : oy;

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
            const int sx = std::clamp(sampleOx + i, 0, v.desc.width - 1);
            const int sy = std::clamp(sampleOy + j, 0, v.desc.height - 1);

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
    // Where the cursor pixel actually landed in the window: near an edge the
    // window is clamped inward, so it is no longer the middle cell.
    const int markI = std::clamp(cx - ox, 0, kSamples - 1);
    const int markJ = std::clamp(cy - oy, 0, kSamples - 1);
    const ImVec2 c0(lx + markI * kCell, ly + markJ * kCell);
    dl->AddRect(c0, ImVec2(c0.x + kCell, c0.y + kCell), IM_COL32(255, 255, 0, 255), 0, 0, 2.0f);

    // The actual values. Float formats keep three decimals -- a raw image lives
    // in 0..1 and beyond, where "0.5" and "0.512" are meaningfully different.
    float centre[3];
    SampleRgb(v, std::clamp(sampleOx + markI, 0, v.desc.width - 1),
              std::clamp(sampleOy + markJ, 0, v.desc.height - 1), centre);
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
