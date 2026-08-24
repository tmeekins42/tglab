#include "auto_develop.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace tglab {
namespace {

// Where the median of a well-exposed frame should sit, in linear light.
//
// 0.18 is the photographic middle grey -- the reflectance a meter assumes, and
// what an 18% grey card is named for. Working in LINEAR light is what makes it
// usable as a target: after a display gamma the same grey sits near 0.46, and
// aiming a linear measurement at that would over-expose by well over a stop.
constexpr double kMiddleGrey = 0.18;

// Never suggest more than this. A very dark frame can compute an enormous
// push, and applying it would produce noise rather than a photograph -- the
// information is not there to recover. Four stops is already aggressive.
constexpr double kMaxStops = 4.0;

// How much of the push BEYOND the highlight headroom to still apply.
//
// 0.85 rather than 0 (a hard ceiling) or 1 (ignore highlights entirely).
// Calibrated against Tim settling on +1.85 stops for a frame whose midtones
// asked +2.05 and whose highlights allowed +0.53: he took most of what the
// midtones wanted and accepted the clipping. This lands at +1.82.
constexpr double kOverdrive = 0.85;

// Highlights are allowed to reach this before the push is held back.
//
// Not 1.0: a suggestion that puts the brightest real detail exactly at
// saturation leaves no headroom for the tone curve, and highlight recovery
// then has nothing to work with.
constexpr double kHighlightCeiling = 0.85;


} // namespace

AutoDevelopSuggestion SuggestExposure(const Image& mosaic) {
    AutoDevelopSuggestion out;

    const ImageDesc& d = mosaic.Desc();
    if (!d.Valid() || !d.IsMosaic()) return out;

    ImageView v = const_cast<Image&>(mosaic).MapCpuRead();
    if (!v.Valid()) return out;

    const double black = double(d.blackLevel);
    const double range = std::max(double(d.whiteLevel) - black, 1.0);

    // Green sensels on a sampled grid.
    //
    // Sampling rather than reading every pixel: this runs once per image load
    // on the UI thread, and at 24 MP a full scan is tens of milliseconds for a
    // number that a few hundred thousand samples already pins down. The stride
    // is odd so it does not lock onto one CFA phase.
    std::vector<float> g;
    g.reserve(400000);
    const int stride = std::max(1, std::min(d.width, d.height) / 700) | 1;
    for (int y = 0; y < d.height; y += stride) {
        for (int x = 0; x < d.width; x += stride) {
            if (CfaColorAt(d.cfa, x, y) != 1) continue;      // green only
            const double s = (double(*v.At<float>(x, y)) - black) / range;
            g.push_back(float(std::clamp(s, 0.0, 4.0)));
        }
    }
    if (g.size() < 1000) return out;                          // too little to judge

    std::sort(g.begin(), g.end());
    auto pct = [&](double f) {
        const size_t i = std::min(g.size() - 1,
                                  size_t(f * double(g.size() - 1) + 0.5));
        return double(g[i]);
    };

    out.blackPoint = float(pct(0.01));
    out.midtone    = float(pct(0.50));

    size_t clipped = 0;
    for (float s : g) if (s >= 0.99f) ++clipped;
    out.clippedFrac = float(double(clipped) / double(g.size()));

    // The brightest UNCLIPPED detail, not a fixed percentile.
    //
    // A fixed 99.5th percentile lands inside the clipped region as soon as
    // more than 0.5% of the frame is saturated, and then "protect the
    // highlights" is protecting saturation -- which cannot be protected and
    // which vetoes any push at all. Measured on an ISO 6400 frame with 0.67%
    // clipped: the 99.5th percentile read 0.9993 and suppressed a needed
    // +2 stops down to zero.
    //
    // Stepping below the clipped fraction makes the number mean what its name
    // says. The floor keeps it away from the midtones on a frame that is
    // mostly blown, where there is little real highlight detail to speak of.
    const double hiPct = std::max(0.50, 0.995 - double(out.clippedFrac));
    out.highlight = float(pct(hiPct));

    // The exposure that would put the median at middle grey.
    //
    // The median rather than the mean: a frame with a bright sky and a dark
    // subject has a mean pulled around by the area split, while the median
    // tracks where most of the picture actually sits. It is also what survives
    // the long dark tail of an under-exposed raw.
    const double mid = std::max(double(out.midtone), 1e-5);
    double stops = std::log2(kMiddleGrey / mid);

    // Highlight protection, as a brake rather than a veto.
    //
    // Clamping hard to the headroom is what a light meter would do, and it is
    // wrong for a rescue: on Tim's ISO 6400 frame the midtones wanted +2.05
    // stops while the highlights allowed only +0.53, and he settled on +1.85 by
    // eye -- accepting blown highlights to get a usable picture. A photographer
    // makes that trade constantly; a ceiling that refuses to clip anything can
    // never reach a usable image from an under-exposed one.
    //
    // So the headroom is honoured in full up to where it runs out, and beyond
    // that the remaining push is applied at a reduced rate. The suggestion
    // still lands short of what the midtones alone would ask, which is the
    // right bias for a starting point the user will adjust.
    const double hi = double(out.highlight);
    if (hi > 1e-5) {
        const double headroom = std::max(std::log2(kHighlightCeiling / hi), 0.0);
        if (stops > headroom) {
            stops = headroom + (stops - headroom) * kOverdrive;
        }
    }

    // Only ever suggest a push. A raw that needs pulling down is usually one
    // with blown highlights, and the fix there is highlight recovery rather
    // than a global darkening that muddies everything else.
    stops = std::clamp(stops, 0.0, kMaxStops);
    out.exposure = float(stops);

    // Lift shadows when the frame needed a big push: the same push that brings
    // the midtones up leaves the darkest quarter compressed against black.
    // Scaled by the push rather than fixed, so a frame that needed nothing gets
    // nothing.
    if (stops > 0.5) {
        out.shadows = float(std::clamp((stops - 0.5) * 0.25, 0.0, 0.6));
    }

    // Set the black point where the data actually starts, so a raw with a
    // lifted floor (haze, or a sensor whose black level is conservative) does
    // not open looking washed out. Only when the floor is meaningfully above
    // zero -- otherwise this would crush real shadow detail.
    if (out.blackPoint > 0.02f) {
        out.blacks = float(-std::clamp(double(out.blackPoint) * 2.0, 0.0, 0.5));
    }

    out.valid = true;
    return out;
}

} // namespace tglab
