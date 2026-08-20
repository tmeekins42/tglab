// Minimal EXIF reader for the info panel.
//
// stb_image decodes pixels and discards everything else, so the capture
// settings are simply gone by the time an Image exists. They are worth having
// -- knowing a scan was shot at ISO 3200 explains the noise a filter is being
// asked to remove -- so this re-reads the file's APP1 segment separately.
//
// Deliberately small: the handful of tags a photographer actually looks at,
// not a general TIFF parser. Anything unrecognised is skipped rather than
// guessed at, and a malformed file yields an empty result rather than a crash
// -- every read is bounds-checked, since this parses untrusted input.
#pragma once

#include <string>

namespace tglab {

struct ExifData {
    bool present = false;      // true when an APP1/EXIF block was parsed

    std::string cameraMake;
    std::string cameraModel;
    std::string lens;

    // Empty when the tag was absent. Kept as formatted strings because that is
    // all the panel does with them, and it avoids inventing sentinel values for
    // "not present".
    std::string exposureTime;   // "1/250 s"
    std::string aperture;       // "f/2.8"
    std::string iso;            // "ISO 400"
    std::string focalLength;    // "50 mm"
    std::string dateTaken;      // "2026:08:19 14:32:07"

    bool Any() const {
        return !cameraMake.empty() || !cameraModel.empty() || !lens.empty() ||
               !exposureTime.empty() || !aperture.empty() || !iso.empty() ||
               !focalLength.empty() || !dateTaken.empty();
    }
};

// Reads EXIF from an image file. Returns a default-constructed ExifData when
// the file has none, is not a JPEG, or cannot be read -- absence of metadata is
// normal, not an error worth reporting.
ExifData ReadExif(const std::string& path);

} // namespace tglab
