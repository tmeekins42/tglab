// Highlight handling shared by every demosaicer.
//
// WHY THIS EXISTS AT ALL. Demosaic itself is simple -- interpolate the two
// missing colours at each photosite, apply the camera's white balance gains,
// rotate through the camera matrix into sRGB. Nothing here is part of that.
// This file deals with one specific way the simple pipeline goes visibly
// wrong, and it is worth stating the physics before the code.
//
// A blown highlight pins every photosite at the sensor ceiling, so the raw
// values read equal -- say 1.0, 1.0, 1.0. White balance then multiplies them
// APART: on a 3313 K frame the gains are (1.1729, 1.0000, 3.0840), so the
// pixel becomes 1.17 / 1.00 / 3.08. The camera matrix, which subtracts about
// 1.12*green from red by design, turns that into vivid magenta. A blown white
// streetlight renders bright pink.
//
// That is not a bug in white balance. A neutral SCENE colour does not read
// equal on the sensor -- it reads inversely to the gains (0.853 / 1.000 /
// 0.324), and white balance turns that into 1.0 / 1.0 / 1.0, correctly. The
// real problem is headroom: for the scene to stay neutral at the brightness
// blue reaches at its ceiling, red would need a raw value of 2.63 and green
// 3.08. Both are impossible. Red and green ran out of range first, and the
// colour information is genuinely gone.
//
// So a blown highlight cannot be recovered, only replaced with something less
// wrong than magenta. The question is what.
//
// WHAT THIS DOES: clamp each channel, after white balance, to the brightest
// NEUTRAL the sensor can represent. Channel i can reach a balanced value V
// only if its raw sample V/camMul[i] fits under 1.0, so the binding limit is
// V <= min(camMul). Above that the sensor cannot describe a neutral colour at
// all, and anything it reports there is an artefact of the gains.
//
// One line, no mask, no per-pixel branch. Measured on _L0A0738.CR2:
//
//                        blown px tinted   negatives   blown px level
//   no handling at all        6663           4130      1.17/1.00/3.08
//   this clamp                   0           2628      1.000
//
// WHAT WAS TRIED AND REMOVED. An earlier version built a per-channel
// saturation mask from the raw samples and lifted every clipped channel to the
// pixel's brightest channel. It also drove the tint to zero, and on negatives
// it scored slightly better (2154). It was removed anyway, for two reasons:
//
//  * It lifted blown pixels to 3.084 -- THREE TIMES full white. Tonemapping a
//    region that far over range makes it swamp its surroundings, which is the
//    bright blob visible in stage 5 of scripts/demosaic_stages.tgl. The clamp
//    puts the same pixels at 1.000.
//  * The mask was built from the neighbours the interpolation reads, which for
//    red and blue is 2 samples at a green site and 4 at a red site. The clip
//    decision therefore alternated with CFA parity: 45.5% of horizontally
//    adjacent pixels disagreed INSIDE a uniformly blown light. Half the pixels
//    got repaired and half did not, and the camera matrix turned that
//    checkerboard into the magenta and cyan speckles. Aligning the window to
//    the CFA cell cut it to 20.6%, but the artefact only changed colour.
//
// The lesson worth keeping: the machinery was patching an artefact its own
// per-pixel decisions created. Clamping needs no decision, so it has no
// checkerboard to produce.
// WHY WHITE BALANCE COMES FIRST, AND NOT THE MATRIX.
//
// The natural objection: the raw values came through colour filters, so surely
// the first thing to do is map those filters into our colour space, and only
// then correct the illuminant. And there is a real observation behind it --
// running white balance first INFLATES the very channels the matrix
// subtracts. On a blown pixel blue arrives at 3.084 and green's -0.4326
// coefficient removes 1.334 of it; matrix-first it would remove only 0.433.
//
// But rgb_cam is not a neutral filter-to-sRGB rotation. LibRaw derives it
// assuming its input is ALREADY WHITE BALANCED -- which is visible in the
// matrix itself, since all three rows sum to exactly 1.0. That is precisely
// the property that maps a balanced neutral (1,1,1) to an sRGB neutral
// (1,1,1). A matrix expecting raw sensor values would need different row sums
// to absorb the gains.
//
// So reversing the order fixes the clipped pixels and breaks every other one.
// A neutral scene grey reads raw 0.853 / 1.000 / 0.324 (inverse to the gains,
// which is what white balance exists to undo):
//
//   wb then matrix   1.0008  1.0002  0.9988   neutral, correct
//   matrix then wb   0.7111  1.3279 -0.1243   green-cyan cast, blue negative
//
// Clipped pixels are about 0.2% of a frame. Trading a correct 99.8% for them
// is the wrong way round, and the clamp fixes the 0.2% without touching the
// rest -- it only ever affects values that were already impossible.
//
// Note also what this means for the "grey goes magenta" worry: it does not.
// A neutral scene develops perfectly neutral at every luminance, because its
// raw values are inverse to the gains all the way up. Magenta appears only
// when clipping breaks that inverse relationship -- the sensor pins all three
// channels equal, white balance multiplies them apart, and the matrix reads
// the result as an intensely blue colour and subtracts green accordingly.
#pragma once

#include <algorithm>

#include "../../core/image.h"

namespace tglab {

// White balance and the highlight clamp together -- they belong in one place
// because the clamp is meaningless except in balanced units.
inline void BalanceAndClamp(const ImageDesc& d, float* rgb) {
    // The brightest neutral this sensor can describe. Below it the data is
    // real; above it, only the gains are talking.
    const float ceiling =
        std::min(d.camMul[0], std::min(d.camMul[1], d.camMul[2]));

    for (int i = 0; i < 3; ++i)
        rgb[i] = std::min(rgb[i] * d.camMul[i], ceiling);
}


// The camera matrix, with the smallest desaturation that keeps the result in
// gamut. Input is white-balanced camera RGB; output is sRGB primaries.
//
// WHY THIS EXISTS. The matrix maps a wider gamut inward and has negative
// coefficients by design -- green's row is -0.2412*R + 1.6738*G - 0.4326*B.
// For an ordinary colour that is a small rotation, bounded and hue dependent,
// measured between -6.9 and +18.4 degrees. But bilinear interpolation across
// the steep edge of a blown highlight produces channel ratios the SCENE never
// had, and the matrix faithfully rotates that false direction into a channel
// below zero. Traced on _L0A0738.CR2 at the shoulder of a bridge light:
//
//     raw neighbourhood   R0.9999 G0.7001 R0.9999 G0.1846
//                         G0.9999 B0.9999 G0.8527 B0.1530
//     interpolated        0.658  0.154  0.062   red averaged from two
//                                               saturated neighbours, green
//                                               taken from one dim centre
//     after wb            0.772  0.154  0.190
//     after matrix        1.374 -0.011  0.218   green through zero -> magenta
//
// 16884 pixels on that frame develop with green suppressed although their
// green photosites were the BRIGHTEST of the three -- the colour is
// manufactured by the pipeline, not measured.
//
// WHAT THIS DOES. Rather than detecting those pixels, exploit the fact that
// both the desaturation and the matrix are LINEAR, so the output is linear in
// the desaturation amount t:
//
//     out(t) = out(0) + t * (out(1) - out(0))
//
// where out(1) is the fully neutral result. Solve out_i(t) = 0 per channel and
// take the largest root in [0,1]. That is the SMALLEST desaturation keeping
// every channel non-negative, and it is exact -- no threshold, no detector, no
// false positives. On the pixel above t comes out at 0.028: under 3% of
// desaturation. On a pixel already in gamut t is exactly 0 and the whole thing
// is inert.
//
// NOTE this is not "scale the pixel down and back up". Pure scaling is a
// no-op: the matrix is linear, so M(k*v)/k == M(v). Only moving TOWARD THE
// NEUTRAL AXIS changes the direction the matrix sees, which is what matters.
inline void CameraMatrixInGamut(const ImageDesc& d, float* rgb) {
    const float* m = d.rgbCam;
    const float r = rgb[0], g = rgb[1], b = rgb[2];

    float out[3] = {m[0] * r + m[1] * g + m[2] * b,
                    m[3] * r + m[4] * g + m[5] * b,
                    m[6] * r + m[7] * g + m[8] * b};

    if (out[0] >= 0.0f && out[1] >= 0.0f && out[2] >= 0.0f) {
        rgb[0] = out[0];
        rgb[1] = out[1];
        rgb[2] = out[2];
        return;
    }

    // The fully desaturated result. The matrix's rows sum to 1.0 (that is what
    // maps a balanced neutral to an sRGB neutral), so a neutral input passes
    // straight through and every channel equals the luminance.
    const float lum = (r + g + b) * (1.0f / 3.0f);

    float t = 0.0f;
    for (int i = 0; i < 3; ++i) {
        if (out[i] >= 0.0f) continue;
        const float span = lum - out[i];        // out_i(1) - out_i(0)
        if (span <= 1e-9f) continue;            // cannot be fixed by blending
        t = std::max(t, -out[i] / span);
    }
    t = std::min(t, 1.0f);

    for (int i = 0; i < 3; ++i)
        rgb[i] = out[i] + t * (lum - out[i]);
}
// The same rule in HLSL. Pasted into each demosaicer's shader source; keep it
// in step with BalanceAndClamp above.
inline const char* kClipRepairHlsl = R"(
void BalanceAndClamp(inout float3 rgb, float3 camMul) {
    float ceiling = min(camMul.r, min(camMul.g, camMul.b));
    rgb = min(rgb * camMul, ceiling);
}
)";

}  // namespace tglab
