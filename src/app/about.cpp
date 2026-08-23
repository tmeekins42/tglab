// about.cpp — the About dialog and its third-party attribution.
//
// One dialog rather than two. The third-party list is not a courtesy here:
// LibRaw is LGPL-2.1 / CDDL-1.0 and is statically linked, so the notice and the
// offer of source are distribution obligations, not decoration. Putting them
// behind a second "Licences..." click makes them easier to forget when the
// build is packaged, and gains nothing -- a collapsing section costs one line
// of screen when closed.
//
// The library versions come from each library's own headers rather than from
// strings typed here, so they cannot drift when one is updated.
#include "about.h"

#include <cstdio>

#include "imgui.h"

#include "libraw/libraw_version.h"

namespace tglab {
namespace {

// Everything the licences require us to state, per library.
//
// `notice` is the copyright line the licence asks to be reproduced; `terms`
// names the licence itself. Kept as data so adding a dependency is one entry
// rather than a new block of layout code.
struct Attribution {
    const char* name;
    const char* version;
    const char* url;
    const char* terms;
    const char* notice;
};

// stb ships two files under the same terms; listed separately because the
// versions move independently.
const Attribution kLibraries[] = {
    {"Dear ImGui", IMGUI_VERSION, "https://github.com/ocornut/imgui",
     "MIT License",
     "Copyright (c) 2014-2026 Omar Cornut"},

    {"LibRaw", LIBRAW_VERSION_STR, "https://www.libraw.org",
     "LGPL-2.1 or CDDL-1.0 (see below)",
     "Copyright (C) 2008-2025 LibRaw LLC\n"
     "Uses code from dcraw.c, copyright 1997-2018 Dave Coffin\n"
     "DCB demosaic and FBDD denoise, copyright (C) 2010 Jacek Gozdz (BSD-3-Clause)"},

    {"stb_image", "2.30", "https://github.com/nothings/stb",
     "MIT License or public domain",
     "Copyright (c) 2017 Sean Barrett"},

    {"stb_image_write", "1.16", "https://github.com/nothings/stb",
     "MIT License or public domain",
     "Copyright (c) 2017 Sean Barrett"},
};

// The LGPL's practical obligation for a statically linked library: say which
// library it is, where its source is, and how the user can relink. Stating it
// plainly is both the honest answer and the cheapest one -- the sources are
// already vendored in the repository.
const char* kLibRawNotice =
    "tglab links LibRaw statically. LibRaw is licensed under the GNU LGPL "
    "version 2.1 or the CDDL version 1.0, at your option.\n\n"
    "LibRaw's complete source is included in this project under "
    "third_party/libraw/, with the full licence texts in LICENSE.LGPL and "
    "LICENSE.CDDL alongside it. Building tglab from source rebuilds LibRaw "
    "from that source, so a modified LibRaw can be substituted by editing it "
    "there and rebuilding.\n\n"
    "LibRaw is distributed in the hope that it will be useful, but WITHOUT ANY "
    "WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS "
    "FOR A PARTICULAR PURPOSE.";

// Selectable rather than a hyperlink: opening a browser from a dialog is a
// surprise, and a URL the user can copy costs nothing and works everywhere.
void UrlRow(const char* url) {
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.55f, 0.70f, 0.95f, 1.0f));
    ImGui::TextUnformatted(url);
    ImGui::PopStyleColor();
    if (ImGui::IsItemHovered()) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        ImGui::SetTooltip("Click to copy");
    }
    if (ImGui::IsItemClicked()) ImGui::SetClipboardText(url);
}

} // namespace

void DrawAboutDialog(bool* open) {
    if (!*open) return;

    // Opened here rather than at the menu item, so the dialog owns its own
    // lifecycle and the caller only has to flip a bool.
    if (!ImGui::IsPopupOpen("About tglab")) ImGui::OpenPopup("About tglab");

    ImGui::SetNextWindowSize(ImVec2(540, 560), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(),
                            ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

    if (!ImGui::BeginPopupModal("About tglab", open,
                                ImGuiWindowFlags_NoSavedSettings)) {
        return;
    }

    ImGui::PushFont(nullptr, ImGui::GetFontSize() * 1.4f);
    ImGui::TextUnformatted("tglab");
    ImGui::PopFont();

    ImGui::Text("Version %s", TGLAB_VERSION);
    ImGui::Spacing();
    ImGui::TextWrapped(
        "A lab bench for computer vision and computer graphics algorithms. "
        "Drop images in, wire algorithms together with a short script, and "
        "watch the result change as you drag the sliders.");

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::TextUnformatted("Copyright (c) 2026 Tim Meekins");
    ImGui::TextUnformatted("MIT License");
    UrlRow("https://github.com/tmeekins42/tglab");

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Open by default: an attribution nobody sees is not an attribution.
    if (ImGui::CollapsingHeader("Third-party software",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
        // Bounded so the dialog cannot outgrow the screen as libraries are
        // added; the list scrolls instead.
        ImGui::BeginChild("licences", ImVec2(0, 250), ImGuiChildFlags_Borders);
        for (const Attribution& a : kLibraries) {
            ImGui::PushID(a.name);
            ImGui::Text("%s %s", a.name, a.version);
            ImGui::Indent();
            ImGui::TextDisabled("%s", a.terms);
            ImGui::TextWrapped("%s", a.notice);
            UrlRow(a.url);
            ImGui::Unindent();
            ImGui::Spacing();
            ImGui::PopID();
        }

        ImGui::Separator();
        ImGui::Spacing();
        ImGui::TextUnformatted("LibRaw (LGPL) — written offer");
        ImGui::Indent();
        ImGui::TextWrapped("%s", kLibRawNotice);
        ImGui::Unindent();
        ImGui::EndChild();
    }

    ImGui::Spacing();
    if (ImGui::Button("Close", ImVec2(120, 0))) {
        *open = false;
        ImGui::CloseCurrentPopup();
    }

    ImGui::EndPopup();
}

} // namespace tglab
