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

// What the median is actually aimed at, as a fraction of middle grey.
//
// Slightly below it, and measured rather than chosen. Aiming the median at a
// full 0.18 put the DISPLAYED median at 0.465 to 0.501 across three cameras --
// consistently above the 0.35-0.45 a default render should sit at, on frames
// with and without a shadow lift, so it is the target itself rather than the
// recovery terms.
//
// The gap exists because the median is measured on the green sensels of the
// mosaic, while what the eye judges is luminance after demosaic and white
// balance -- and those lift it. Correcting the target is more honest than
// bending the curve, which would move every tone to fix one.
constexpr double kMedianAim = 0.78;

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
// Raised from 0.85 once the base tone curve landed. The old value braked hard
// against a ceiling that meant "pixels start clipping"; with the shoulder
// absorbing the top, crossing it costs compression rather than destruction, so
// the brake should be lighter. Measured on the Petaluma alley: at 0.95 the
// suggestion comes to +2.75 stops, which puts the median at 0.410 display and
// the sky at 0.896 -- nothing clipped.
constexpr double kOverdrive = 0.95;

// Clipping below this is not worth answering for.
//
// Every photograph has a few saturated pixels -- a specular glint on water, a
// filament, a sun through leaves -- and reshaping the whole upper range to
// chase them costs contrast where the picture is brightest. 1% of the frame is
// where blown area stops reading as sparkle and starts reading as loss.
constexpr double kClipNotice = 0.01;

// ... and the same at the black end. A frame with a genuinely black background
// is not a fault to be corrected.
constexpr double kCrushNotice = 0.02;

// How hard to answer, per unit of clipped area beyond the threshold.
//
// Both deliberately gentle. These land on sliders the user can see and move, so
// the suggestion should be visibly helping rather than making a decision they
// have to undo -- and both controls cost contrast in the range they reshape.
//
// Calibrated on the ISO 6400 rescue: 4.3% blown after its push gives -0.17
// highlights, and 7.3% crushed gives +0.21 shadows. Both are a nudge rather
// than a rescue, which is the right weight for something applied unasked.
constexpr double kHighlightGain = 5.0;
constexpr double kShadowGain    = 4.0;
constexpr double kMaxHighlights = 0.45;
constexpr double kMaxShadows    = 0.50;

// Where a scene stops fitting the display, in stops from p10 to p99.
//
// An sRGB display shows roughly six stops of usable range. Below that a scene
// fits and wants no compression -- suggesting some would only flatten a frame
// that was already fine. Above it, something has to give.
constexpr double kRangeNotice = 5.0;

// How hard to compress, per stop of excess range.
//
// Calibrated against Lightroom's Auto on the Petaluma alley, which is the only
// external reference available for what a good answer looks like. That frame
// measures 6.2 stops, so 1.2 stops of excess; Lightroom chose highlights -69
// and shadows +50 on its own -100..100 scale, which is -0.69 and +0.50 on
// ours. Matching that exactly would need gains of 0.58 and 0.42.
//
// Deliberately pitched below that, at roughly half. Lightroom's Auto is a
// finished opinion; ours is a starting point on sliders the user can see and
// move, and an over-eager default is worse than a mild one because it has to
// be undone before anything else can be judged. This lands at -0.36 and +0.24
// on the alley -- clearly helping, still conservative.
constexpr double kRangeHighlightGain = 0.30;

// The shadow half is gentler still, and for a measured reason rather than
// symmetry. A shadow lift brightens the lower midtones, and on a dark scene the
// median sits inside exactly that band -- so it moves the median as well as the
// shadows. End to end on the Petaluma alley, +0.24 of range-driven shadows
// carried the displayed median from the 0.443 the exposure solve intended up to
// 0.490, which is above where a default render should sit.
//
// Halving it keeps the shadow separation the wide scene needs without the
// exposure target being quietly overridden by a control meant to act below it.
constexpr double kRangeShadowGain    = 0.10;

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

    // The scene's own dynamic range, in stops, from the 10th to the 99th
    // percentile.
    //
    // Not the true min-to-max: both ends of that are noise on one side and a
    // specular glint on the other, so it measures the outliers rather than the
    // picture. p10 to p99 covers what the frame is actually made of -- on the
    // Petaluma alley 0.0066 to 0.4881, which is 6.2 stops against a display's
    // roughly 6, so it is right at the edge of what fits.
    //
    // This is what makes highlight and shadow suggestions PROACTIVE. Derived
    // from clipping alone they can only react once detail is already being
    // lost; derived from range they can tell that a nine-stop scene needs
    // compressing before anything is destroyed. It is also what Lightroom's
    // Auto is evidently doing: on this frame it chose highlights -69 and
    // shadows +50 at exposure +0.68, which is a lot of local compression on a
    // frame with essentially no clipping to react to.
    {
        const double lo = std::max(pct(0.10), 1e-5);
        const double hi = std::max(pct(0.99), lo * 1.0001);
        out.dynamicRange = float(std::log2(hi / lo));
    }

    size_t clipped = 0, crushed = 0;
    for (float s : g) {
        if (s >= 0.99f)  ++clipped;
        if (s <= 0.002f) ++crushed;
    }
    out.clippedFrac = float(double(clipped) / double(g.size()));
    out.crushedFrac = float(double(crushed) / double(g.size()));

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
    double stops = std::log2(kMiddleGrey * kMedianAim / mid);

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
    //
    // The brake is much gentler than it used to be, because the base tone curve
    // now catches the top. Before it existed, exceeding the headroom meant
    // pixels hitting 1.0 and staying there -- detail destroyed. With the
    // shoulder in place a bright pixel is compressed rather than clipped: on
    // the Petaluma alley the sky at 4.0 linear lands at 0.91 display instead of
    // 1.83-clipped. So the ceiling being crossed is no longer a cliff, and
    // braking hard against it only makes the picture darker than it needs to
    // be. See tone_curve.h.
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

    // How much of the frame the push will drive into saturation.
    //
    // Measured rather than inferred, and measured AFTER the push, because that
    // is what the highlight recovery has to answer for. On the ISO 6400 frame
    // the two differ by more than six times: 0.67% clipped as captured, 4.32%
    // once +1.82 stops is applied.
    {
        const double gain = std::pow(2.0, stops);
        size_t after = 0;
        for (float s : g) if (double(s) * gain >= 1.0) ++after;
        out.clippedAfter = float(double(after) / double(g.size()));
    }

    // Highlight recovery, proportional to what the push actually blows out.
    //
    // Negative, because that is the recovering direction. Nothing is suggested
    // below a threshold: a frame with a few specular glints does not want its
    // upper tones reshaped, and a recovery applied to an image with nothing to
    // recover only flattens it.
    //
    // Capped well short of the slider's own range. This is a starting point on
    // a control the user can see, so it should be visibly helping rather than
    // making a decision they then have to undo -- and the deeper the recovery,
    // the more it costs in contrast up there.
    if (out.clippedAfter > kClipNotice) {
        const double over = double(out.clippedAfter) - kClipNotice;
        out.highlights = float(-std::clamp(over * kHighlightGain, 0.0, kMaxHighlights));
    }

    // ... and proportional to how much wider the scene is than the display.
    //
    // The reactive term above answers for what the push destroys. This one
    // answers for the scene itself, which is the case it could never see: the
    // Petaluma alley spans 6.2 stops with essentially nothing clipped, so
    // clipping-driven recovery suggests nothing at all, while Lightroom's Auto
    // chose highlights -69 on the same frame. A wide scene wants its top
    // compressed on its own account.
    //
    // The stronger of the two wins rather than the sum: they are two ways of
    // detecting the same need, and adding them double-counts a frame that is
    // both wide and clipped.
    if (out.dynamicRange > kRangeNotice) {
        const double over = double(out.dynamicRange) - kRangeNotice;
        const double fromRange = std::clamp(over * kRangeHighlightGain, 0.0, kMaxHighlights);
        out.highlights = float(-std::max(double(-out.highlights), fromRange));
    }

    // Shadow lift, from measured crushing rather than from the push.
    //
    // The earlier version derived this from the number of stops, which is a
    // proxy: a frame can need a big push and still have no shadow detail
    // against the floor, and another can be crushed without needing any push at
    // all. Measuring how much of the frame actually sits at black answers the
    // question directly.
    //
    // The push itself is still accounted for, because lifting the midtones
    // lifts the shadows with them -- so what matters is what remains crushed
    // afterwards, which the push has already reduced.
    if (out.crushedFrac > kCrushNotice) {
        const double over = double(out.crushedFrac) - kCrushNotice;
        out.shadows = float(std::clamp(over * kShadowGain, 0.0, kMaxShadows));
    }

    // ... and from the scene's range, for the same reason as the highlights:
    // compressing a wide scene means lifting the bottom as well as holding the
    // top, and a frame can be wide without anything sitting against the floor.
    if (out.dynamicRange > kRangeNotice) {
        const double over = double(out.dynamicRange) - kRangeNotice;
        const double fromRange = std::clamp(over * kRangeShadowGain, 0.0, kMaxShadows);
        out.shadows = float(std::max(double(out.shadows), fromRange));
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
