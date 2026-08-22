// White balance — colour temperature to channel gains, and back.
//
// In algo_util/ rather than inside basic_adjust because two callers need the
// same answer and must not disagree: the algorithm, which applies a correction
// relative to what the camera did, and the app, which opens the kelvin slider at
// the value the camera used.
#pragma once

#include "../core/image.h"

namespace tglab {

// The white point the sRGB primaries assume, and therefore the reference the
// camera's daylight gains are expressed against.
inline constexpr float kD65Kelvin = 6504.0f;

// The usable range.
//
// The lower bound is not arbitrary: below about 1900 K the Planckian locus
// leaves the sRGB gamut and the blue component goes NEGATIVE -- measured,
// -0.036 at 1700 K. Dividing by a clamped epsilon there produced a blue gain of
// 32x at 2000 K and broke monotonicity, so dragging the slider one way moved the
// colour back the other.
inline constexpr float kKelvinMin = 2000.0f;
inline constexpr float kKelvinMax = 25000.0f;

// The chromaticity of a black body at `kelvin`, as CIE xy.
//
// Kim et al.'s cubic fit to the Planckian locus, the same one every raw
// developer uses. Checked against Illuminant A to within 0.0005. D50 and D65 sit
// about 0.006 BELOW the locus, which is correct rather than error -- daylight
// illuminants are not black bodies -- and worth knowing before someone "fixes"
// it.
void PlanckianXy(float kelvin, float* x, float* y);

// The per-channel gains that render an illuminant at `kelvin`, offset along the
// green/magenta axis by `tint`, as neutral. Normalised on green, so the control
// does not double as an exposure slider.
void KelvinGains(float kelvin, float tint, float* gr, float* gg, float* gb);

// The temperature and tint the camera chose for a shot, recovered from its
// white-balance metadata.
//
// Inverted by bisection: the forward direction is monotonic in R/B above
// kKelvinMin, so a fixed 40 steps finds it with no convergence to tune.
//
// Both are 0 when `d` carries no daylight reference (a JPEG, or a raw whose
// profile lacks one), which correctly means "nothing to measure against".
//
// Note that the pair is needed, not just the temperature: measured on a real
// file, matching the as-shot temperature alone still left red and blue BOTH off
// by 1.078 -- equal in both, which is a pure green shift by definition, and
// exactly what tint exists to express.
void AsShotWhiteBalance(const ImageDesc& d, float* kelvin, float* tint);

} // namespace tglab
