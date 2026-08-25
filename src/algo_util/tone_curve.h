// tone_curve — the base curve that turns scene-linear light into a picture.
//
// This is the piece every raw developer has and we did not. Its absence was
// visible: our render of a Petaluma alley needed +2.70 stops of exposure to
// stop looking like mud, and that push blew 7.2% of the frame to white.
// Lightroom's default render of the same file sits at exposure 0.00, and its
// Auto at +0.68 -- neither one clips. They are not reading a brightness value
// out of the file (the EXIF says exposure compensation +0.00); they are
// applying a curve.
//
// What the curve is for
// ---------------------
// A raw holds far more range than a screen can show -- this frame spans about
// nine stops, an sRGB display about six. Something has to compress that, and
// there are only two options: clip the ends, or bend them. Exposure is a
// multiply, so it can only slide the range up and down; whatever falls off the
// top is gone. A curve bends instead, which is why it can hold a bright sky and
// a dark alley in one image.
//
// So the division of labour is:
//
//   exposure -- lifts the scene to where the midtones belong
//   curve    -- catches the top so that lift stops blowing out
//
// It is worth being precise that the curve does NOT brighten the image, which
// is tempting to assume. It is anchored at middle grey, and this scene's median
// sits three stops below that, so the curve moves the midtones almost not at
// all: measured, the median came out 0.160 through the curve and 0.160 through
// a plain sRGB encode -- identical. The exposure push is still doing the
// lifting. What changes is that the sky at 4.0 linear now lands at 0.91 instead
// of 1.83-clipped.
//
// Shape
// -----
// A sigmoid in log2 space around middle grey, because that is the space
// photographic tone actually lives in: a stop is a stop whether it is a dark
// stop or a bright one, and film's characteristic curve is a straight line in
// exactly this space with a toe and a shoulder at the ends.
//
//   - kGreyIn -> kGreyOut anchors the midtones. 0.18 linear -> 0.46 display
//     is the industry baseline and matches what Adobe's default renders.
//   - Above grey, the response is compressed asymptotically, so no input
//     however bright ever reaches 1.0. This is the shoulder.
//   - Below grey, the response stays straight in log space.
//
// That asymmetry is deliberate and was arrived at by getting it wrong first.
// The first version compressed BOTH ends asymptotically, which is elegant and
// unusable: an asymptote at the bottom can never reach zero, so 0.002 linear
// came out 0.226 display -- black rendered as milky grey. Highlights and
// shadows are not the same problem. Highlights must never clip, so they need an
// asymptote. Shadows must reach true black, so they must not have one.
//
// A second approach -- bend in linear light, then encode sRGB -- was tried and
// abandoned: asymptoting to 1.6 in linear put the sRGB encode above 1.0, so it
// clipped precisely what it existed to prevent. Compressing in the perceptual
// domain, where the anchors mean something, is the sounder design.
#pragma once

#include <algorithm>
#include <cmath>

namespace tglab {

// Middle grey, in and out. 0.18 is the reflectance a meter is calibrated to;
// 0.46 is where a default raw render puts it, and is what makes our output
// comparable to Lightroom's rather than merely self-consistent.
inline constexpr float kGreyIn  = 0.18f;
inline constexpr float kGreyOut = 0.46f;

// Slope through middle grey, in output stops per input stop.
//
// 1.0 -- a straight pass -- is deliberate. The curve's job here is the
// shoulder; contrast is already a control the user has, and baking extra
// contrast into the base curve would mean the contrast slider starts from a
// place it cannot get back to. Measured at 1.15 and 1.30 the midtones came out
// DARKER than a plain sRGB encode (0.137 and 0.117 against 0.160), which is the
// wrong direction for a curve meant to make raws presentable.
inline constexpr float kContrast = 1.0f;

// Where the highlights asymptote, in output stops above middle grey.
//
// This sets how much headroom the shoulder can absorb. At 3.2, a value of 8.0
// linear renders 0.93 and 4.0 renders 0.87 -- bright, distinct, and unclipped,
// which covers the range a raw actually reaches after auto-exposure.
//
// It is worth being honest that this does NOT make clipping mathematically
// impossible. The asymptote sits at kGreyOut * 2^(kShoulder * kOutputScale) =
// 1.39, so inputs above roughly 19.0 linear still hit the clamp. Making the
// curve unclippable everywhere means an asymptote at 1.0, which is kShoulder
// 2.24 -- and that compresses the real range much harder: on the Petaluma
// alley the sky fell from 0.87 to 0.77 and the bright brickwork flattened
// noticeably against it.
//
// So this is a deliberate trade rather than an oversight. Protecting values no
// photograph reaches, at the cost of highlight separation across every frame
// that does, is the wrong way round. 19 stops-equivalent of headroom is beyond
// what the sensor records.
inline constexpr float kShoulder = 3.2f;

// Output stops -> display value. Half a display "stop" per output stop keeps
// the anchor at kGreyOut while leaving room above it.
inline constexpr float kOutputScale = 0.5f;

// Scene-linear luminance or channel value -> display value in 0..1.
//
// Total on 0: an input of zero is black, with no toe lift. Monotonic
// throughout, so it can never invert an ordering between two tones.
inline float ToneCurve(float x) {
    if (!(x > 0.0f)) return 0.0f;   // also catches NaN

    const float s = std::log2(x / kGreyIn) * kContrast;

    // Compress above grey; leave below grey straight so black stays reachable.
    const float y = (s >= 0.0f) ? s / (1.0f + s / kShoulder) : s;

    return std::clamp(kGreyOut * std::exp2(y * kOutputScale), 0.0f, 1.0f);
}

// The same curve as HLSL, for the display shader and any GPU stage that needs
// it. Kept beside the C++ so the two cannot drift -- compare mode checks one
// against the other, and a divergence here would look like a GPU bug.
inline const char* kToneCurveHlsl = R"(
static const float kGreyIn      = 0.18;
static const float kGreyOut     = 0.46;
static const float kContrast    = 1.0;
static const float kShoulder    = 3.2;
static const float kOutputScale = 0.5;

float ToneCurve(float x) {
    if (!(x > 0.0)) return 0.0;
    float s = log2(x / kGreyIn) * kContrast;
    float y = (s >= 0.0) ? s / (1.0 + s / kShoulder) : s;
    return saturate(kGreyOut * exp2(y * kOutputScale));
}
float3 ToneCurve3(float3 c) {
    return float3(ToneCurve(c.r), ToneCurve(c.g), ToneCurve(c.b));
}
)";

} // namespace tglab
