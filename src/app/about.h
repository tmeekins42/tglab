// The About dialog: version, licence, and third-party attribution.
#pragma once

namespace tglab {

// Draws the modal when *open is true, and clears it when the user closes.
// Call once per frame, unconditionally -- it returns immediately when closed.
void DrawAboutDialog(bool* open);

} // namespace tglab
