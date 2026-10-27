#pragma once
// Lens profiles from the lensfun database (https://lensfun.github.io).
//
// The database is a folder of XML files: per lens, the distortion, lateral
// chromatic aberration (TCA) and vignetting measured at a set of focal lengths
// (and, for vignetting, apertures and focus distances). This reads those files
// itself -- the format is a few element types -- rather than linking lensfun,
// which would bring GLib with it.
//
// WHAT COMES BACK IS IN lens_correct's UNITS, not lensfun's. lensfun keeps its
// coefficients in three coordinate systems (r = 1 at half the short side of the
// CALIBRATION sensor for distortion and TCA, at its half diagonal for
// vignetting, and focal-length units internally); every term here has been
// rescaled to r = 1 at the half diagonal of the image being corrected, so the
// caller needs no camera geometry at all. The real focal length lensfun
// carries cancels out of that conversion, and the PanoTools "d" term is
// divided out as lensfun does, so the centre of the frame keeps its scale.
//
// The database is CC BY-SA 3.0 (third_party/lensfun/README.md) and is data,
// not code: it is looked for at run time (LensfunDirectory) and, absent,
// every lookup simply finds nothing.
#include <string>

namespace tglab {

struct LensProfile {
    bool valid = false;          // a lens matched (any of the three may still be missing)
    std::string lens;            // the database's name for it
    std::string camera;          // ... and for the camera, when that matched
    float focal = 0.0f;          // the focal length it was evaluated at, mm
    float aperture = 0.0f;       // ... and the aperture, for vignetting
    float cropFactor = 0.0f;     // the image's, as used
    float calibCrop = 0.0f;      // the calibration's

    // Where an output pixel at radius R takes its light from, as a multiple of
    // R: f(R) = dist[0] + dist[1] R + dist[2] R^2 + dist[3] R^3 + dist[4] R^4.
    // Below 1 at the corners is barrel, as lens_correct's k1 < 0.
    bool hasDistortion = false;
    double dist[5] = {1, 0, 0, 0, 0};

    // Red's and blue's extra magnification against green, at green's source
    // radius r: tca[c][0] + tca[c][1] r + tca[c][2] r^2 (c = 0 red, 1 blue).
    bool hasTca = false;
    double tca[2][3] = {{1, 0, 0}, {1, 0, 0}};

    // The light the lens kept at radius r is 1 + k1 r^2 + k2 r^4 + k3 r^6;
    // the correction divides by it, in linear light.
    bool hasVignetting = false;
    double vig[3] = {0, 0, 0};
};

// Looks the lens up, by fuzzy name, and evaluates its profile at the shot's
// focal length and aperture. `crop35` is the 35 mm equivalent focal over the
// real one when the file says (0 when not); the camera's own entry in the
// database is preferred to it. An empty lens name is a compact's fixed lens,
// found by the body's mount; a focal of 0 finds nothing.
LensProfile FindLensProfile(const char* cameraMake, const char* cameraModel,
                            const char* lens, float focalMm, float aperture,
                            float crop35);

// The folder the database was read from, or empty if none was found: the
// environment's TGLAB_LENSFUN_DIR, else a `lensfun` folder (as a release
// ships it) or `third_party/lensfun/db` (the repository's copy) beside the
// executable or up to four levels above it.
std::string LensfunDirectory();

// Points the database at `dir` and drops what was loaded, for tests. An empty
// string goes back to searching.
void SetLensfunDirectory(const std::string& dir);

// How many lenses and cameras are loaded (loading them if need be).
int LensfunLensCount();
int LensfunCameraCount();

} // namespace tglab
