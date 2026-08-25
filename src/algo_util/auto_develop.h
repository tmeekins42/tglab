// auto_develop — deriving sensible starting adjustments from the image itself.
//
// The counterpart to reading the camera's white balance out of EXIF: that
// tells you what the camera THOUGHT, and this measures what it actually
// captured. An under-exposed frame carries no metadata saying "push me two
// stops"; the only way to know is to look at the pixels.
//
// Deliberately a set of measurements plus a suggestion, not a filter. The
// values it produces become parameter DEFAULTS (see
// AlgorithmBase::PrepareDefaults), so an untouched slider opens somewhere
// useful and a touched one is left alone. That is what makes "auto" a starting
// point the user can argue with rather than a mode they have to turn off.
#pragma once

#include "../core/image.h"

namespace tglab {

// What a measurement of the source suggests.
//
// Expressed in the units of basic_adjust's own controls, so applying it is
// assignment rather than translation -- and so a wrong suggestion is visible
// on the slider rather than hidden in a transform.
struct AutoDevelopSuggestion {
    bool  valid    = false;   // false when the image could not be measured

    float exposure   = 0.0f;    // stops
    float blacks     = 0.0f;    // -1..1, basic_adjust's own scale
    float shadows    = 0.0f;
    float highlights = 0.0f;    // negative recovers

    // What the measurement actually saw, so the suggestion can be explained
    // rather than just obeyed. Reported in 0..1 of the sensor's range.
    float blackPoint  = 0.0f;   // 1st percentile
    float midtone     = 0.0f;   // median
    float highlight   = 0.0f;   // brightest unclipped detail

    // Clipping at each end, as a fraction of the frame.
    //
    // `clippedFrac` is measured as captured; `clippedAfter` is what the
    // suggested exposure push would produce, which is the figure the highlight
    // recovery actually has to answer for. On an under-exposed frame those are
    // very different numbers -- measured 0.67% before a +1.82 stop push and
    // 4.32% after.
    float clippedFrac  = 0.0f;   // at or above the white level, as captured
    float clippedAfter = 0.0f;   // ... after the suggested exposure
    float crushedFrac  = 0.0f;   // at or below black

    // The scene's range from the 10th to the 99th percentile, in stops.
    //
    // What drives the highlight and shadow suggestions PROACTIVELY: a scene
    // wider than the display can show needs compressing whether or not it has
    // clipped yet, and clipping alone can only be reacted to after detail is
    // already gone.
    float dynamicRange = 0.0f;
};

// Measures a Bayer mosaic and suggests exposure adjustments.
//
// Works on the MOSAIC rather than a developed image, for the same reason the
// hot-pixel repair does: this is the only point where the numbers are the
// sensor's own, before a demosaic interpolates them and a tone curve reshapes
// them. Measuring after development would be measuring the previous
// suggestion's effect, which does not converge.
//
// Green sensels only. Green is half of all sites so it is the best-sampled
// channel, and it carries most of the luminance -- using all three would fold
// the white-balance imbalance into the exposure estimate, so a warm image
// would read as brighter than it is.
AutoDevelopSuggestion SuggestExposure(const Image& mosaic);

} // namespace tglab
