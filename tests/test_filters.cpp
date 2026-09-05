// Behavioural checks for the blur / edge-preserving filter family.
//
// Compiling and running is not evidence a filter is correct: a smoother that
// returns its input unchanged, or that quietly averages across an edge, does
// both. These tests assert the properties that actually distinguish the
// algorithms from one another.
//
// The fixture is a vertical step edge (dark left, bright right) with optional
// impulse noise, which is the case every filter here claims to handle and the
// one where a plain Gaussian visibly fails.
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include "../src/algo_util/pixel_buffer.h"
#include "../src/core/algorithm.h"
#include "../src/core/pipeline.h"
#include "../src/gpu/compute.h"
#include <d3d12.h>

#include "../src/core/lut.h"
#include "../src/script/value.h"

using namespace tglab;

static int g_fail = 0;

static void Check(bool cond, const std::string& what) {
    std::printf("%s  %s\n", cond ? "[ ok ]" : "[FAIL]", what.c_str());
    if (!cond) ++g_fail;
}

namespace {

constexpr int kW = 32;
constexpr int kH = 32;
constexpr uint8_t kDark = 40;
constexpr uint8_t kBright = 210;

// Vertical step edge down the middle. `impulses` scatters salt-and-pepper
// spikes on a fixed grid -- fixed, not random, so failures reproduce exactly.
Image MakeEdgeImage(bool impulses) {
    Image img;
    img.Alloc({kW, kH, Format::RGBA8});
    ImageView v = img.MapCpuWrite();

    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < kW; ++x) {
            uint8_t value = (x < kW / 2) ? kDark : kBright;

            // Every 7th pixel on every 5th row flips to an extreme, giving
            // impulses on both sides of the edge.
            if (impulses && (y % 5 == 0) && (x % 7 == 3))
                value = (x < kW / 2) ? 255 : 0;

            uint8_t* p = v.At<uint8_t>(x, y);
            p[0] = p[1] = p[2] = value;
            p[3] = 255;
        }
    }
    return img;
}

// Runs one registered algorithm over the fixture and returns its output.
// As RunFilter, but the parameter values are script Values rather than plain
// doubles -- needed for apply_lut, whose `file` is a string. Kept separate
// rather than widening RunFilter, which every other test calls with doubles.
bool RunFilterV(const std::string& name, Image&& input,
                const std::vector<std::pair<std::string, Value>>& params,
                Image* out, std::string* err) {
    auto algo = Registry::Get().Create(name);
    if (!algo) { *err = "no such algorithm: " + name; return false; }

    for (const auto& [key, value] : params) {
        bool found = false;
        for (ParamBase* p : algo->Params()) {
            if (key != p->Name()) continue;
            found = true;
            if (!p->SetFromScript(value, err)) return false;
        }
        if (!found) { *err = name + " has no parameter '" + key + "'"; return false; }
    }

    Pipeline pipe;
    std::vector<Data> sources;
    sources.push_back(Data{std::move(input)});
    pipe.AddStage(std::move(algo), name, {{-1, 0}}, 1, 1);

    if (!pipe.Execute(&sources, nullptr, err)) return false;

    const Data* d = pipe.Resolve({0, 0}, &sources);
    if (!d || !std::holds_alternative<Image>(*d)) { *err = "no output"; return false; }
    *out = const_cast<Image&>(std::get<Image>(*d)).Clone();
    return true;
}

bool RunFilter(const std::string& name, Image&& input,
               const std::vector<std::pair<std::string, double>>& params,
               Image* out, std::string* err) {
    auto algo = Registry::Get().Create(name);
    if (!algo) { *err = "no such algorithm: " + name; return false; }

    for (const auto& [key, value] : params) {
        bool found = false;
        for (ParamBase* p : algo->Params()) {
            if (key != p->Name()) continue;
            found = true;
            if (!p->SetFromScript(Value(value), err)) return false;
        }
        if (!found) { *err = name + " has no parameter '" + key + "'"; return false; }
    }

    Pipeline pipe;
    std::vector<Data> sources;
    sources.push_back(Data{std::move(input)});
    pipe.AddStage(std::move(algo), name, {{-1, 0}}, 1, 1);

    if (!pipe.Execute(&sources, nullptr, err)) return false;

    const Data* d = pipe.Resolve({0, 0}, &sources);
    if (!d || !std::holds_alternative<Image>(*d)) { *err = "no output"; return false; }
    *out = const_cast<Image&>(std::get<Image>(*d)).Clone();
    return true;
}

double PixelAt(Image& img, int x, int y) {
    ImageView v = img.MapCpuRead();
    return double(v.At<uint8_t>(x, y)[0]);
}

// Mean absolute deviation from the ideal step, away from the edge itself.
// A filter that blurs across the edge raises this; one that preserves it
// leaves it near zero.
double EdgeBleed(Image& img) {
    ImageView v = img.MapCpuRead();
    double sum = 0.0;
    int    n   = 0;
    for (int y = 2; y < kH - 2; ++y) {
        for (int x = 2; x < kW - 2; ++x) {
            // Skip the two columns either side of the transition: every filter
            // legitimately transitions somewhere.
            if (std::abs(x - kW / 2) <= 2) continue;
            const double ideal = (x < kW / 2) ? kDark : kBright;
            sum += std::abs(double(v.At<uint8_t>(x, y)[0]) - ideal);
            ++n;
        }
    }
    return n ? sum / n : 0.0;
}

// Standard deviation within one flat region, which is what "did it actually
// smooth anything" means for a noisy input.
double FlatRegionStdDev(Image& img) {
    ImageView v = img.MapCpuRead();
    double sum = 0.0, sumSq = 0.0;
    int    n   = 0;
    for (int y = 2; y < kH - 2; ++y) {
        for (int x = 2; x < kW / 2 - 3; ++x) {
            const double s = double(v.At<uint8_t>(x, y)[0]);
            sum += s; sumSq += s * s; ++n;
        }
    }
    if (!n) return 0.0;
    const double mean = sum / n;
    return std::sqrt(std::max(0.0, sumSq / n - mean * mean));
}

} // namespace


// The Kelvin white-balance control.
//
// Absolute, unlike `temperature`: it names the illuminant the scene was under,
// and setting it to the actual light neutralises the cast. That needs two
// references from the raw file -- the gains the camera chose for the shot
// (camMul) and the camera's daylight reference (preMul) -- and the ratio
// between them is what says where neutral actually is.
//
// The relative control could never do this. It scaled red and blue by at most
// +-40%, while a tungsten frame measured here sits a factor of 0.65 in red and
// 1.67 in blue away from daylight: outside its reach entirely, which is why a
// warm image could not be corrected.
static void TestKelvinWhiteBalance() {
    std::printf("\n--- kelvin white balance ---\n");

    // A camera whose daylight reference and as-shot gains differ, i.e. a shot
    // taken under something other than daylight.
    auto make = [](bool withDaylightRef) {
        ImageDesc d{16, 16, Format::RGBA16F};
        d.linear = true;
        d.camMul[0] = 1.4541f; d.camMul[1] = 1.0f; d.camMul[2] = 2.3701f;  // as shot
        d.preMul[0] = 2.2513f; d.preMul[1] = 1.0f; d.preMul[2] = 1.4180f;  // daylight
        d.hasDaylightWb = withDaylightRef;
        Image img;
        img.Alloc(d);
        ImageView v = img.MapCpuWrite();
        for (int y = 0; y < 16; ++y)
            for (int x = 0; x < 16; ++x) {
                uint16_t* p = v.At<uint16_t>(x, y);
                for (int c = 0; c < 3; ++c) p[c] = FloatToHalf(0.40f);
                p[3] = FloatToHalf(1.0f);
            }
        return img;
    };

    auto ratioOf = [](Image& img) {
        ImageView v = img.MapCpuRead();
        const uint16_t* p = v.At<uint16_t>(8, 8);
        const float r = HalfToFloat(p[0]);
        const float b = HalfToFloat(p[2]);
        return b > 1e-6f ? double(r / b) : 0.0;
    };

    Image out;
    std::string err;

    // kelvin = 0 means "leave the camera's white balance alone", so the image
    // must come through untouched. A control that jumped to daylight on load
    // would change every picture before the user touched anything.
    Image asShot;
    if (!RunFilter("basic_adjust", make(true), {}, &asShot, &err)) {
        Check(false, "basic_adjust runs on a raw-like image: " + err);
        return;
    }
    Check(std::abs(ratioOf(asShot) - 1.0) < 0.01,
          "kelvin 0 leaves the camera's own white balance alone (R/B " +
              std::to_string(ratioOf(asShot)) + ")");

    // Warmer request -> warmer render, cooler -> cooler, and monotonic between.
    double prev = -1.0;
    bool monotonic = true;
    for (float k : {2500.0f, 3500.0f, 5000.0f, 6500.0f, 9000.0f}) {
        if (!RunFilter("basic_adjust", make(true), {{"kelvin", double(k)}}, &out, &err)) {
            Check(false, "basic_adjust at " + std::to_string(k) + " K: " + err);
            return;
        }
        const double ratio = ratioOf(out);
        if (prev >= 0.0 && ratio <= prev) { monotonic = false; break; }
        prev = ratio;
    }
    Check(monotonic,
          "raising kelvin renders progressively warmer (R/B rises monotonically)");

    // The round trip: asking for the illuminant the shot was ALREADY balanced
    // for must be a no-op. This is what proves the control is anchored to the
    // file rather than applying an arbitrary curve -- the as-shot and requested
    // corrections have to cancel.
    //
    // The fixture's camMul/preMul ratio corresponds to roughly 3800 K, so a
    // request near there should land close to unchanged, and much closer than
    // daylight does.
    Image atShot, atDaylight;
    if (RunFilter("basic_adjust", make(true), {{"kelvin", 3800.0}}, &atShot, &err) &&
        RunFilter("basic_adjust", make(true), {{"kelvin", 6504.0}}, &atDaylight, &err)) {
        const double dShot     = std::abs(ratioOf(atShot) - 1.0);
        const double dDaylight = std::abs(ratioOf(atDaylight) - 1.0);
        Check(dShot < dDaylight,
              "requesting the shot's own temperature changes it less than requesting "
              "daylight (" + std::to_string(dShot) + " vs " + std::to_string(dDaylight) + ")");
    } else {
        Check(false, "round-trip runs: " + err);
    }

    // No daylight reference -- a JPEG, or a raw whose profile lacks one. The
    // control has nothing to anchor to, so it must do nothing rather than guess.
    Image noRef;
    if (RunFilter("basic_adjust", make(false), {{"kelvin", 2500.0}}, &noRef, &err)) {
        Check(std::abs(ratioOf(noRef) - 1.0) < 0.01,
              "without a daylight reference, kelvin does nothing rather than guessing");
    } else {
        Check(false, "basic_adjust without a daylight reference: " + err);
    }

    // Tint is meaningful with or without a daylight reference: it is a plain
    // green/magenta push, not something that needs to know where neutral was.
    // So it stays available on a JPEG, where kelvin cannot be.
    auto greenOf = [](Image& img) {
        ImageView v = img.MapCpuRead();
        return double(HalfToFloat(v.At<uint16_t>(8, 8)[1]));
    };
    Image neutralTint, greener;
    if (RunFilter("basic_adjust", make(false), {}, &neutralTint, &err) &&
        RunFilter("basic_adjust", make(false), {{"tint", -0.5}}, &greener, &err)) {
        Check(greenOf(greener) > greenOf(neutralTint),
              "tint still works without a daylight reference");
    } else {
        Check(false, "tint without a daylight reference: " + err);
    }

    // The as-shot recovery: the file knows what temperature it was taken at, so
    // the slider should open there rather than at a sentinel.
    //
    // Tim reported the old "0 means leave it alone" default as confusing on
    // sight, and he was right: it is a poor answer to "what temperature is this
    // photograph?" when the answer is in the metadata.
    {
        ImageDesc d{8, 8, Format::RGBA16F};
        d.linear = true;
        d.camMul[0] = 1.4541f; d.camMul[1] = 1.0f; d.camMul[2] = 2.3701f;
        d.preMul[0] = 2.2513f; d.preMul[1] = 1.0f; d.preMul[2] = 1.4180f;
        d.hasDaylightWb = true;

        float k = 0.0f, t = 0.0f;
        AsShotWhiteBalance(d, &k, &t);

        // Measured on the real file these numbers come from: ~4027 K. A
        // tungsten-leaning shot, which is what that cam_mul says.
        Check(k > 3500.0f && k < 4500.0f,
              "as-shot temperature recovered from the metadata (" +
                  std::to_string(k) + " K, expected ~4000)");

        // Feeding it back must be a no-op: that round trip is what makes the
        // recovered number trustworthy rather than merely plausible.
        auto make = [&]() {
            Image img;
            img.Alloc(d);
            ImageView v = img.MapCpuWrite();
            for (int y = 0; y < 8; ++y)
                for (int x = 0; x < 8; ++x) {
                    uint16_t* p = v.At<uint16_t>(x, y);
                    for (int c = 0; c < 3; ++c) p[c] = FloatToHalf(0.40f);
                    p[3] = FloatToHalf(1.0f);
                }
            return img;
        };
        Image back;
        std::string e;
        if (RunFilter("basic_adjust", make(),
                      {{"kelvin", double(k)}, {"tint", double(t)}}, &back, &e)) {
            ImageView v = back.MapCpuRead();
            const float r = HalfToFloat(v.At<uint16_t>(4, 4)[0]);
            const float g = HalfToFloat(v.At<uint16_t>(4, 4)[1]);
            const float b = HalfToFloat(v.At<uint16_t>(4, 4)[2]);
            Check(std::abs(r / b - 1.0f) < 0.02f && std::abs(g / b - 1.0f) < 0.02f,
                  "feeding the recovered values back leaves the image neutral (R/B " +
                      std::to_string(r / b) + ", G/B " + std::to_string(g / b) + ")");
        } else {
            Check(false, "round-trip through the recovered values: " + e);
        }

        // A JPEG has nothing to measure against and must say so rather than
        // inventing a number.
        ImageDesc plain{8, 8, Format::RGBA8};
        float k2 = -1.0f, t2 = -1.0f;
        AsShotWhiteBalance(plain, &k2, &t2);
        Check(k2 == 0.0f && t2 == 0.0f,
              "no daylight reference yields no as-shot temperature rather than a guess");
    }
}

// --- vignette -------------------------------------------------------------
//
// The four properties worth pinning, because each has a plausible wrong
// implementation that looks fine on a casual glance:
//
//   * the centre is untouched, whatever the amount
//   * the sign convention matches Lightroom -- negative darkens
//   * the two directions are NOT the same operation mirrored: darkening
//     scales (a transmission loss, which preserves black), lightening lerps
//     toward white (multiplying up instead blows the corner highlights while
//     barely moving its shadows, which reads as a lighting error)
//   * at roundness 0 the falloff follows the frame's aspect, so on a
//     non-square image the mid-edges darken equally. Normalising to the
//     shorter axis -- the obvious mistake -- would clip the long edges flat
//     while the corners were still untouched.
static void TestVignette() {
    const int W = 300, H = 200;    // 3:2, so aspect errors show
    auto flat = [&] {
        Image img;
        img.Alloc({W, H, Format::RGBA32F});
        ImageView v = img.MapCpuWrite();
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                float* p = v.At<float>(x, y);
                p[0] = p[1] = p[2] = 0.5f; p[3] = 1.0f;
            }
        return img;
    };
    auto at = [](const Image& im, int x, int y) {
        ImageView v = const_cast<Image&>(im).MapCpuRead();
        return v.At<float>(x, y)[0];
    };

    std::string err;
    Image dark, light, off, round0, round1;

    if (RunFilter("vignette", flat(), {{"amount", -1.0}}, &dark, &err) &&
        RunFilter("vignette", flat(), {{"amount",  1.0}}, &light, &err) &&
        RunFilter("vignette", flat(), {{"amount",  0.0}}, &off, &err)) {

        Check(std::fabs(at(dark, W / 2, H / 2) - 0.5f) < 1e-4f,
              "vignette leaves the centre untouched (" +
                  std::to_string(at(dark, W / 2, H / 2)) + ")");

        Check(at(dark, 0, 0) < 0.05f,
              "negative amount darkens the corners, as in Lightroom (" +
                  std::to_string(at(dark, 0, 0)) + ")");

        Check(at(light, 0, 0) > 0.95f,
              "positive amount lightens them (" +
                  std::to_string(at(light, 0, 0)) + ")");

        // Not merely "brighter than the centre": a multiply would also be
        // brighter. What distinguishes a lerp toward white is that it
        // APPROACHES white and stops, rather than scaling past it.
        Check(at(light, 0, 0) <= 1.0f + 1e-4f,
              "...by lifting toward white rather than scaling up (" +
                  std::to_string(at(light, 0, 0)) + ", must not exceed 1)");

        Check(std::fabs(at(off, 0, 0) - 0.5f) < 1e-6f,
              "amount 0 is exactly the identity (" +
                  std::to_string(at(off, 0, 0)) + ")");
    } else {
        Check(false, "vignette runs: " + err);
    }

    // Aspect. At roundness 0 the shape follows the frame, so the middle of the
    // long edge and the middle of the short edge sit at the same normalised
    // distance and must darken identically. At roundness 1 the shape is a
    // circle, so they must not.
    if (RunFilter("vignette", flat(),
                  {{"amount", -1.0}, {"midpoint", 0.3}, {"feather", 0.4},
                   {"roundness", 0.0}}, &round0, &err) &&
        RunFilter("vignette", flat(),
                  {{"amount", -1.0}, {"midpoint", 0.3}, {"feather", 0.4},
                   {"roundness", 1.0}}, &round1, &err)) {

        // Sampled at the true edges, not a few pixels in: on a 300x200 frame
        // "2 px from the edge" is a different fraction of each axis, which
        // would make the two differ for a reason that is not the aspect.
        const float l0 = at(round0, 0, H / 2), t0 = at(round0, W / 2, 0);
        const float l1 = at(round1, 0, H / 2), t1 = at(round1, W / 2, 0);

        Check(std::fabs(l0 - t0) < 1e-4f,
              "roundness 0 follows the frame, so both mid-edges match (" +
                  std::to_string(l0) + " vs " + std::to_string(t0) + ")");
        Check(std::fabs(l1 - t1) > 1e-3f,
              "...and roundness 1 is a circle, so they differ (" +
                  std::to_string(l1) + " vs " + std::to_string(t1) + ")");
    } else {
        Check(false, "vignette roundness runs: " + err);
    }

    // Opposite corners must match. An off-by-one in the pixel-centre offset
    // shows here and nowhere else.
    Image sym;
    if (RunFilter("vignette", flat(),
                  {{"amount", -0.8}, {"midpoint", 0.4}}, &sym, &err)) {
        const float tl = at(sym, 0, 0),         tr = at(sym, W - 1, 0);
        const float bl = at(sym, 0, H - 1),     br = at(sym, W - 1, H - 1);
        Check(std::fabs(tl - tr) < 1e-5f && std::fabs(tl - bl) < 1e-5f &&
                  std::fabs(tl - br) < 1e-5f,
              "the four corners are symmetric (" + std::to_string(tl) + " " +
                  std::to_string(tr) + " " + std::to_string(bl) + " " +
                  std::to_string(br) + ")");
    } else {
        Check(false, "vignette symmetry runs: " + err);
    }
}

// --- film grain ------------------------------------------------------------
//
// Four properties, each of which has a plausible wrong version:
//
//   * STRENGTH IS INDEPENDENT OF SIZE. Interpolating a coarse lattice reduces
//     variance, so without compensation the size slider quietly changes
//     loudness too and the two controls fight. This is the one that was
//     actually wrong first time: at size exactly 1 every pixel lands on a
//     lattice point, nothing is interpolated, and applying the compensation
//     anyway made size 1 measure 35% louder than every other size -- a visible
//     step right where the slider starts.
//   * The grain is MULTIPLICATIVE and midtone-weighted, so black stays black.
//     Additive noise would lift the shadows into a grey haze, which is the
//     digital-sensor look rather than film.
//   * At colour 0 the three channels share one field exactly, so the grain
//     moves along luminance and has no hue at all.
//   * It does not shift exposure: the mean survives.
static void TestFilmGrain() {
    const int W = 128, H = 128;
    auto flat = [&](float level) {
        Image img;
        img.Alloc({W, H, Format::RGBA32F});
        ImageView v = img.MapCpuWrite();
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                float* p = v.At<float>(x, y);
                p[0] = p[1] = p[2] = level; p[3] = 1.0f;
            }
        return img;
    };
    auto sigma = [](const Image& im, int c) {
        ImageView v = const_cast<Image&>(im).MapCpuRead();
        double s = 0, s2 = 0; long long n = 0;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const double p = v.At<float>(x, y)[c];
                s += p; s2 += p * p; ++n;
            }
        const double m = s / double(n);
        return std::sqrt(std::max(s2 / double(n) - m * m, 0.0));
    };
    auto mean = [](const Image& im) {
        ImageView v = const_cast<Image&>(im).MapCpuRead();
        double s = 0; long long n = 0;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) { s += v.At<float>(x, y)[0]; ++n; }
        return s / double(n);
    };

    std::string err;

    // Strength holds across sizes, INCLUDING size 1.
    double lo = 1e9, hi = 0.0;
    bool ok = true;
    for (double sz : {1.0, 2.0, 4.0, 8.0, 16.0}) {
        Image out;
        if (!RunFilter("film_grain", flat(0.5f),
                       {{"strength", 0.2}, {"size", sz}}, &out, &err)) {
            ok = false; break;
        }
        const double s = sigma(out, 0);
        lo = std::min(lo, s);
        hi = std::max(hi, s);
    }
    if (ok) {
        Check(hi < lo * 1.1,
              "grain strength is independent of size (sigma " +
                  std::to_string(lo) + ".." + std::to_string(hi) + ")");
    } else {
        Check(false, "film_grain runs: " + err);
    }

    // Multiplicative: black stays black, midtones carry the texture.
    Image dark, mid;
    if (RunFilter("film_grain", flat(0.02f),
                  {{"strength", 0.2}, {"size", 4.0}}, &dark, &err) &&
        RunFilter("film_grain", flat(0.5f),
                  {{"strength", 0.2}, {"size", 4.0}}, &mid, &err)) {
        Check(sigma(dark, 0) < sigma(mid, 0) * 0.1,
              "grain is multiplicative, so deep shadows stay clean (" +
                  std::to_string(sigma(dark, 0)) + " vs " +
                  std::to_string(sigma(mid, 0)) + " at midtone)");
        Check(std::fabs(mean(mid) - 0.5) < 0.01,
              "...and it does not shift exposure (mean " +
                  std::to_string(mean(mid)) + " from 0.5)");
    } else {
        Check(false, "film_grain tonal test runs: " + err);
    }

    // Monochrome at colour 0, chromatic at colour 1.
    Image mono, chroma;
    if (RunFilter("film_grain", flat(0.5f),
                  {{"strength", 0.2}, {"size", 4.0}, {"colour", 0.0}},
                  &mono, &err) &&
        RunFilter("film_grain", flat(0.5f),
                  {{"strength", 0.2}, {"size", 4.0}, {"colour", 1.0}},
                  &chroma, &err)) {
        auto worstRB = [](const Image& im) {
            ImageView v = const_cast<Image&>(im).MapCpuRead();
            double w = 0;
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x)
                    w = std::max(w, std::fabs(double(v.At<float>(x, y)[0]) -
                                              double(v.At<float>(x, y)[2])));
            return w;
        };
        Check(worstRB(mono) < 1e-6,
              "colour 0 gives monochrome grain, identical in every channel (" +
                  std::to_string(worstRB(mono)) + ")");
        Check(worstRB(chroma) > 0.05,
              "...and colour 1 gives an independent field per channel (" +
                  std::to_string(worstRB(chroma)) + ")");
    } else {
        Check(false, "film_grain colour test runs: " + err);
    }

    // Strength 0 is exactly the identity.
    Image off;
    if (RunFilter("film_grain", flat(0.5f), {{"strength", 0.0}}, &off, &err)) {
        Check(sigma(off, 0) < 1e-9,
              "strength 0 is exactly the identity (sigma " +
                  std::to_string(sigma(off, 0)) + ")");
    } else {
        Check(false, "film_grain identity runs: " + err);
    }
}

// --- orton -----------------------------------------------------------------
//
// The effect is a darkroom sandwich: a sharp frame screened over a blurred,
// brightened copy. The properties that make it that rather than a blurry
// overlay:
//
//   * the glow SPILLS OUT of a bright area into the surrounding dark. If the
//     brightening happened after the blur instead of before, the whole frame
//     would lift uniformly and there would be no spill.
//   * screen only ever BRIGHTENS. Averaging the layers instead -- the usual
//     wrong implementation -- pulls highlights down to meet the shadows and
//     gives a flat haze with no glow.
//   * black stays black, because screen leaves 0 alone.
//   * it saturates toward white rather than running away. This is the one that
//     was wrong first time: compressing both operands through x/(1+x),
//     screening, and expanding with the inverse is not a round trip, because
//     the screen of two compressed values is not itself a compressed value.
//     A 0.9 square came out at 2.79 and a 6.0 highlight at 51.75.
static void TestOrton() {
    const int N = 128;
    auto spot = [&](float bg, float fg) {
        Image img;
        img.Alloc({N, N, Format::RGBA32F});
        ImageView v = img.MapCpuWrite();
        for (int y = 0; y < N; ++y)
            for (int x = 0; x < N; ++x) {
                const bool box = std::abs(x - N / 2) < N / 8 &&
                                 std::abs(y - N / 2) < N / 8;
                float* p = v.At<float>(x, y);
                p[0] = p[1] = p[2] = box ? fg : bg;
                p[3] = 1.0f;
            }
        return img;
    };
    auto at = [](const Image& im, int x, int y) {
        ImageView v = const_cast<Image&>(im).MapCpuRead();
        return v.At<float>(x, y)[0];
    };

    std::string err;
    Image glow, off, hdr;
    const std::vector<std::pair<std::string, double>> full = {
        {"blur", 10.0}, {"strength", 1.0}, {"brightness", 1.4}, {"contrast", 0.0}};

    if (RunFilter("orton", spot(0.05f, 0.9f), full, &glow, &err)) {
        // Just outside the square: input 0.05, and the glow must lift it well
        // clear of that. The farCorner corner should be much less affected.
        const float nearBox = at(glow, N / 2, N / 2 + 20);
        const float farCorner     = at(glow, 4, 4);
        Check(nearBox > 0.2f,
              "the glow spills out of a bright area into the dark (" +
                  std::to_string(nearBox) + " from an input of 0.05)");
        Check(nearBox > farCorner * 2.0f,
              "...and falls off with distance (" + std::to_string(nearBox) +
                  " beside the square vs " + std::to_string(farCorner) + " far off)");

        // Screen saturates toward white. Before the fix this read 2.79.
        const float centre = at(glow, N / 2, N / 2);
        Check(centre <= 1.05f,
              "a bright area saturates toward white rather than running away (" +
                  std::to_string(centre) + " from an input of 0.9)");
        Check(centre > 0.9f,
              "...while still brightening it (" + std::to_string(centre) + ")");

        // Screen can only brighten: nothing may come out below its input.
        ImageView v = glow.MapCpuRead();
        bool darkened = false;
        for (int y = 0; y < N && !darkened; ++y)
            for (int x = 0; x < N; ++x) {
                const bool box = std::abs(x - N / 2) < N / 8 &&
                                 std::abs(y - N / 2) < N / 8;
                if (v.At<float>(x, y)[0] < (box ? 0.9f : 0.05f) - 1e-4f) {
                    darkened = true;
                    break;
                }
            }
        Check(!darkened,
              "the combine is a screen, so no pixel is darkened");
    } else {
        Check(false, "orton runs: " + err);
    }

    if (RunFilter("orton", spot(0.05f, 0.9f),
                  {{"strength", 0.0}, {"blur", 10.0}}, &off, &err)) {
        Check(std::fabs(at(off, N / 2, N / 2) - 0.9f) < 1e-4f &&
                  std::fabs(at(off, 4, 4) - 0.05f) < 1e-4f,
              "strength 0 is exactly the identity (" +
                  std::to_string(at(off, N / 2, N / 2)) + ", " +
                  std::to_string(at(off, 4, 4)) + ")");
    } else {
        Check(false, "orton identity runs: " + err);
    }

    // Black stays black -- screen leaves 0 alone whatever the brightness.
    Image black;
    if (RunFilter("orton", spot(0.0f, 0.0f),
                  {{"blur", 10.0}, {"strength", 1.0}, {"brightness", 3.0}},
                  &black, &err)) {
        Check(at(black, N / 2, N / 2) < 1e-6f,
              "black stays black (" + std::to_string(at(black, N / 2, N / 2)) + ")");
    } else {
        Check(false, "orton black runs: " + err);
    }

    // Scene-linear data above 1.0 must keep its headroom rather than being
    // amplified. Before the fix a 6.0 highlight came out at 51.75.
    if (RunFilter("orton", spot(0.05f, 6.0f), full, &hdr, &err)) {
        const float centre = at(hdr, N / 2, N / 2);
        Check(centre > 6.0f && centre < 8.0f,
              "a scene-linear highlight keeps its headroom without running "
              "away (" + std::to_string(centre) + " from an input of 6.0)");
    } else {
        Check(false, "orton HDR runs: " + err);
    }

    // THE S-CURVE MUST NOT TURN OVER ON A BLOWN HIGHLIGHT.
    //
    // Reported from a real landscape: overexposed cloud came out BLACK at every
    // Orton strength. The cause was smoothstep, which is only a step on 0..1 --
    // past 1 its cubic dives (t=1.5 gives exactly 0, t=2 gives -4, t=3 gives
    // -27). At the default contrast of 0.2 that put anything above about 2.4x
    // white below zero, and scene-linear highlights are routinely 4-8x.
    //
    // The existing HDR check above missed it entirely because it passes
    // contrast = 0, which switches the S-curve off. Any test of an HDR path has
    // to exercise the DEFAULT settings, not a configuration that happens to
    // avoid the arithmetic under suspicion.
    {
        const std::vector<std::pair<std::string, double>> withCurve = {
            {"blur", 10.0}, {"strength", 1.0}, {"brightness", 1.0},
            {"contrast", 0.2}};

        // Several brightnesses, because the failure was not at one value: the
        // cubic crosses zero at t=1.5 and gets worse from there, so a single
        // sample could sit either side of it by luck.
        for (float fg : {1.2f, 2.0f, 4.0f, 8.0f}) {
            Image out;
            if (!RunFilter("orton", spot(0.05f, fg), withCurve, &out, &err)) {
                Check(false, "orton with contrast runs: " + err);
                continue;
            }
            const float centre = at(out, N / 2, N / 2);
            Check(centre > 0.5f * fg,
                  "a blown highlight survives the S-curve at " +
                      std::to_string(fg) + "x white (" +
                      std::to_string(centre) + ")");
        }

        // MONOTONIC through white: brighter in must stay brighter out. This is
        // the property the cubic broke, and checking it directly is what makes
        // the test about the shape of the curve rather than about one value.
        float prev = -1e9f;
        bool rising = true;
        for (float fg : {0.5f, 0.9f, 1.0f, 1.1f, 1.5f, 2.0f, 3.0f, 6.0f}) {
            Image out;
            if (!RunFilter("orton", spot(0.05f, fg), withCurve, &out, &err))
                continue;
            const float centre = at(out, N / 2, N / 2);
            if (centre <= prev) rising = false;
            prev = centre;
        }
        Check(rising, "orton's S-curve stays monotonic across the white point");
    }
}

// --- 3D LUTs ---------------------------------------------------------------
//
// The table is written out and read back rather than shipped as a fixture, so
// the parser is tested against a file it did not produce in memory.
//
// The properties worth pinning:
//
//   * an IDENTITY table is the identity. This is the one that catches an axis
//     transposition: .cube stores blue slowest, and getting that backwards
//     still round-trips on a symmetric table but swaps red and blue on a real
//     one -- which looks like a plausible grade rather than a bug.
//   * grid points reproduce their stored entries exactly.
//   * a missing or malformed file passes the image through rather than
//     blanking it, and says so.
//   * strength blends, and 0 is the identity.
static void TestLut() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "tglab_lut_test";
    std::error_code ec;
    fs::create_directories(dir, ec);

    // An identity 5^3 table, in .cube's own blue-slowest order.
    const int N = 5;
    const fs::path idPath = dir / "identity.cube";
    {
        std::ofstream f(idPath);
        f << "# a deliberately small identity table\n";
        f << "TITLE \"identity\"\n";
        f << "LUT_3D_SIZE " << N << "\n\n";
        for (int b = 0; b < N; ++b)
            for (int g = 0; g < N; ++g)
                for (int r = 0; r < N; ++r)
                    f << float(r) / float(N - 1) << " "
                      << float(g) / float(N - 1) << " "
                      << float(b) / float(N - 1) << "\n";
    }

    // A table that ONLY swaps red and blue. On this one an axis-order mistake
    // is unmissable, where on the identity it would be invisible.
    const fs::path swapPath = dir / "swap.cube";
    {
        std::ofstream f(swapPath);
        f << "LUT_3D_SIZE " << N << "\n";
        for (int b = 0; b < N; ++b)
            for (int g = 0; g < N; ++g)
                for (int r = 0; r < N; ++r)
                    f << float(b) / float(N - 1) << " "
                      << float(g) / float(N - 1) << " "
                      << float(r) / float(N - 1) << "\n";
    }

    Lut3D lut;
    std::string err;

    if (lut.Load(idPath.string(), &err)) {
        Check(lut.Size() == N && !lut.Is1D(),
              "a .cube header is parsed (size " + std::to_string(lut.Size()) + ")");
        Check(lut.Title() == "identity",
              "...including its title (\"" + lut.Title() + "\")");

        double worst = 0.0;
        for (int i = 0; i <= 20; ++i)
            for (int j = 0; j <= 20; ++j) {
                const float r = float(i) / 20.0f, g = float(j) / 20.0f;
                const float b = float((i + j) % 21) / 20.0f;
                float o[3];
                lut.Sample(r, g, b, o);
                worst = std::max({worst, std::fabs(double(o[0] - r)),
                                  std::fabs(double(o[1] - g)),
                                  std::fabs(double(o[2] - b))});
            }
        Check(worst < 1e-5,
              "an identity table is the identity, so the axis order is right "
              "(worst error " + std::to_string(worst) + ")");
    } else {
        Check(false, "identity LUT loads: " + err);
    }

    // The swap table proves the axis order rather than merely being consistent
    // with it.
    if (lut.Load(swapPath.string(), &err)) {
        float o[3];
        lut.Sample(1.0f, 0.0f, 0.0f, o);
        Check(o[2] > 0.99f && o[0] < 0.01f,
              "a red-to-blue table really does swap them, so blue is the "
              "slowest axis as .cube specifies (" + std::to_string(o[0]) + " " +
                  std::to_string(o[1]) + " " + std::to_string(o[2]) + ")");
    } else {
        Check(false, "swap LUT loads: " + err);
    }

    // A malformed file must fail with a message rather than half-loading.
    const fs::path badPath = dir / "bad.cube";
    {
        std::ofstream f(badPath);
        f << "LUT_3D_SIZE 5\n0.1 0.2 0.3\n";   // far too few entries
    }
    Lut3D bad;
    std::string badErr;
    const bool rejected = !bad.Load(badPath.string(), &badErr);
    // Captured before the next Load overwrites it -- reporting `badErr` after
    // reusing the variable printed an empty string and made a passing check
    // look like a broken one.
    Check(rejected && !badErr.empty(),
          "a truncated table is rejected with a message (" + badErr + ")");

    Lut3D absent;
    std::string missingErr;
    const bool refused = !absent.Load((dir / "nope.cube").string(), &missingErr);
    Check(refused && !missingErr.empty(),
          "a missing file is rejected rather than crashing (" + missingErr + ")");

    // Through the algorithm: identity in, identity out, and strength blends.
    auto grey = [] {
        Image img;
        img.Alloc({16, 16, Format::RGBA32F});
        ImageView v = img.MapCpuWrite();
        for (int y = 0; y < 16; ++y)
            for (int x = 0; x < 16; ++x) {
                float* p = v.At<float>(x, y);
                p[0] = 0.8f; p[1] = 0.5f; p[2] = 0.2f; p[3] = 1.0f;
            }
        return img;
    };
    auto at = [](const Image& im, int c) {
        ImageView v = const_cast<Image&>(im).MapCpuRead();
        return v.At<float>(8, 8)[c];
    };

    Image out;
    if (RunFilterV("apply_lut", grey(), {{"file", Value(idPath.string())}}, &out, &err)) {
        Check(std::fabs(at(out, 0) - 0.8f) < 1e-4f &&
                  std::fabs(at(out, 2) - 0.2f) < 1e-4f,
              "apply_lut with an identity table changes nothing (" +
                  std::to_string(at(out, 0)) + ", " + std::to_string(at(out, 2)) + ")");
    } else {
        Check(false, "apply_lut runs: " + err);
    }

    Image swapped, half;
    if (RunFilterV("apply_lut", grey(), {{"file", Value(swapPath.string())}},
                   &swapped, &err) &&
        RunFilterV("apply_lut", grey(),
                   {{"file", Value(swapPath.string())}, {"strength", Value(0.5)}},
                   &half, &err)) {
        Check(std::fabs(at(swapped, 0) - 0.2f) < 1e-3f,
              "...and a real table is applied (red " +
                  std::to_string(at(swapped, 0)) + ", was 0.8)");
        // Half strength must land midway between the two.
        Check(std::fabs(at(half, 0) - 0.5f) < 1e-2f,
              "strength blends toward the original (" +
                  std::to_string(at(half, 0)) + ", midway between 0.8 and 0.2)");
    } else {
        Check(false, "apply_lut swap runs: " + err);
    }

    // A missing file must pass the image through, not blank it.
    Image missing;
    if (RunFilterV("apply_lut", grey(),
                   {{"file", Value((dir / "nope.cube").string())}}, &missing, &err)) {
        Check(std::fabs(at(missing, 0) - 0.8f) < 1e-4f,
              "a missing LUT passes the image through rather than blanking it (" +
                  std::to_string(at(missing, 0)) + ")");
    } else {
        Check(false, "apply_lut missing-file runs: " + err);
    }

    // THE GPU PATH, WITH A TABLE ACTUALLY LOADED.
    //
    // gpu_audit covers apply_lut, but it runs every algorithm at its defaults
    // -- and this one defaults to no LUT, where both paths simply pass through.
    // "Clean" there says nothing about the shader. The table has to be loaded
    // for the comparison to mean anything, which is what this does.
    //
    // The swap table is the right fixture again: an identity would agree even
    // if the shader's row addressing were transposed.
    {
        ID3D12Device* dev = nullptr;
        if (SUCCEEDED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
                                        IID_PPV_ARGS(&dev)))) {
            ComputeContext gpu;
            if (gpu.Init(dev)) {
                auto run = [&](ExecMode mode, Image* out) {
                    auto algo = Registry::Get().Create("apply_lut");
                    std::string e;
                    if (ParamBase* p = algo->FindParam("file"))
                        p->SetFromScript(Value(swapPath.string()), &e);
                    std::vector<Data> src;
                    src.push_back(Data{grey()});
                    Pipeline pipe;
                    pipe.AddStage(std::move(algo), "apply_lut", {{-1, 0}}, 1, 1);
                    if (!pipe.Execute(&src, nullptr, &e, &gpu, mode)) return false;
                    const Data* d = pipe.Resolve({0, 0}, &src);
                    if (!d || !std::holds_alternative<Image>(*d)) return false;
                    *out = const_cast<Image&>(std::get<Image>(*d)).Clone();
                    return true;
                };
                Image onCpu, onGpu;
                if (run(ExecMode::ForceCPU, &onCpu) && run(ExecMode::ForceGPU, &onGpu)) {
                    double worst = 0.0;
                    for (int c = 0; c < 3; ++c)
                        worst = std::max(worst,
                                         std::fabs(double(at(onCpu, c)) -
                                                   double(at(onGpu, c))));
                    Check(worst < 1e-4,
                          "the GPU LUT matches the CPU with a table loaded, so "
                          "the packed 2D layout and the tetrahedral shader are "
                          "right (worst " + std::to_string(worst) + ")");
                    Check(std::fabs(at(onGpu, 0) - 0.2f) < 1e-3f,
                          "...and the GPU really applied it (red " +
                              std::to_string(at(onGpu, 0)) + ", was 0.8)");
                } else {
                    Check(false, "apply_lut ran on both paths");
                }
            }
            dev->Release();
        }
    }

    fs::remove_all(dir, ec);
}

// Dehaze, against a scene hazed by the model it inverts.
//
// The fixture APPLIES Koschmieder -- I = J*t + A*(1-t) -- with a known airlight
// and a known depth ramp, so there is a real answer to compare against rather
// than "it looks clearer". That also means the test can tell the difference
// between removing haze and merely raising contrast, which is what a naive
// implementation does and what a contrast metric alone would happily accept.
static void TestDehaze() {
    const int W = 240, H = 160;
    const float A[3] = {0.85f, 0.88f, 0.95f};   // pale blue airlight

    // The haze-free scene, built to SATISFY the dark channel prior rather than
    // to look interesting -- the prior's assumption is that every small patch
    // contains a near-black pixel, and a fixture that violates it is testing the
    // prior's known failure mode instead of the implementation.
    //
    // A fine checker does that: whatever window the algorithm minimises over, it
    // contains dark cells. An early version used 20 px cells against a patch
    // radius that scaled down to 3, so the bright cells had no dark pixel within
    // reach, the transmission came out too low there, and the near end was
    // over-corrected by about 2x. That was the fixture breaking the prior's
    // premise, not the code getting it wrong.
    auto sceneAt = [&](int x, int y, float* out) {
        const float v = ((x / 3 + y / 3) % 2 == 0) ? 0.03f : 0.55f;
        out[0] = v;
        out[1] = v * 0.9f;
        out[2] = v * 0.8f;
    };

    // Transmission falls left to right: near on the left, distant on the right.
    auto transAt = [&](int x) {
        return 0.9f - 0.6f * (float(x) / float(W - 1));
    };

    Image hazy;
    hazy.Alloc({W, H, Format::RGBA32F});
    {
        ImageView v = hazy.MapCpuWrite();
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                float j[3];
                sceneAt(x, y, j);
                const float t = transAt(x);
                float* p = v.At<float>(x, y);
                for (int c = 0; c < 3; ++c) p[c] = j[c] * t + A[c] * (1.0f - t);
                p[3] = 1.0f;
            }
    }

    auto at = [](const Image& im, int x, int y, int c) {
        ImageView v = const_cast<Image&>(im).MapCpuRead();
        return v.At<float>(x, y)[c];
    };

    std::string err;
    Image out, off;

    // sky_protect off: the fixture has no sky, and the protection would floor
    // the transmission exactly where the test wants it measured.
    const bool ran =
        RunFilter("dehaze", hazy.Clone(),
                  {{"strength", 0.95},
                   {"sky_protect", 0.0}}, &out, &err);
    Check(ran, "dehaze runs" + (ran ? "" : ": " + err));
    if (!ran) return;

    // Strength 0 must be a true pass-through: the control has to be able to
    // turn the algorithm off completely.
    if (RunFilter("dehaze", hazy.Clone(), {{"strength", 0.0}}, &off, &err)) {
        double worst = 0.0;
        for (int y = 0; y < H; y += 7)
            for (int x = 0; x < W; x += 7)
                for (int c = 0; c < 3; ++c)
                    worst = std::max(worst,
                        double(std::fabs(at(off, x, y, c) - at(hazy, x, y, c))));
        Check(worst < 1e-5, "dehaze at strength 0 leaves the image alone (" +
                                std::to_string(worst) + ")");
    }

    // THE REAL TEST: the recovered image must be closer to the true scene than
    // the hazy one was. Measured on the HAZY HALF, where there is something to
    // remove -- the near edge starts at t=0.9 and has almost no haze in it.
    //
    // Error rather than contrast, deliberately. Any contrast stretch improves a
    // contrast score; only an actual inversion of the haze model moves the
    // pixels toward where they belong.
    {
        double before = 0.0, after = 0.0;
        int n = 0;
        for (int y = 4; y < H - 4; y += 3)
            for (int x = W / 2; x < W - 4; x += 3) {
                float j[3];
                sceneAt(x, y, j);
                for (int c = 0; c < 3; ++c) {
                    const double e0 = double(at(hazy, x, y, c)) - double(j[c]);
                    const double e1 = double(at(out,  x, y, c)) - double(j[c]);
                    before += e0 * e0;
                    after  += e1 * e1;
                    ++n;
                }
            }
        const double rmsBefore = std::sqrt(before / double(n));
        const double rmsAfter  = std::sqrt(after  / double(n));
        Check(rmsAfter < rmsBefore * 0.6,
              "dehaze recovers the scene it was hazed from (RMS " +
                  std::to_string(rmsBefore) + " -> " + std::to_string(rmsAfter) + ")");
    }

    // The correction must GROW with distance. A uniform brightening would pass
    // an average-error test on a ramp while being the wrong operation entirely,
    // so compare how much each end moved.
    {
        auto moved = [&](int x) {
            double s = 0.0;
            int n = 0;
            for (int y = 4; y < H - 4; y += 3)
                for (int c = 0; c < 3; ++c) {
                    s += std::fabs(double(at(out, x, y, c)) -
                                   double(at(hazy, x, y, c)));
                    ++n;
                }
            return n ? s / double(n) : 0.0;
        };
        const double nearEnd = moved(10);
        const double farEnd  = moved(W - 11);
        Check(farEnd > nearEnd * 2.0,
              "dehaze corrects the distant end more than the near one (" +
                  std::to_string(nearEnd) + " vs " + std::to_string(farEnd) + ")");
    }

    // Greyscale has no cross-channel minimum, so the prior cannot work and the
    // algorithm must pass through rather than destroy the image.
    {
        Image grey;
        grey.Alloc({W, H, Format::R32F});
        {
            ImageView v = grey.MapCpuWrite();
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x) *v.At<float>(x, y) = 0.4f;
        }
        Image gout;
        if (RunFilter("dehaze", grey.Clone(), {{"strength", 0.9}}, &gout, &err)) {
            ImageView v = gout.MapCpuRead();
            Check(std::fabs(*v.At<float>(W / 2, H / 2) - 0.4f) < 1e-5f,
                  "dehaze passes greyscale through rather than mangling it");
        }
    }

    // AN OVER-STRONG CORRECTION MUST DARKEN, NOT CHANGE COLOUR.
    //
    // Where a pixel sits below the airlight, (I - A) is negative and a hard
    // correction drives the recovery negative. Clamping each channel at zero
    // on its own then removes the channels at different rates -- red first,
    // blue last -- and a neutral cloud comes out PURPLE. That is much worse
    // than the darkening it replaces, because it does not read as too much
    // dehaze, it reads as a broken image.
    //
    // A scene-linear frame is what exposes it: values above 1.0 in the
    // highlights let the airlight search land on a specular edge, after which
    // every pixel in the sky is below the airlight.
    {
        Image sky;
        sky.Alloc({W, H, Format::RGBA32F});
        {
            ImageView v = sky.MapCpuWrite();
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x) {
                    float* p = v.At<float>(x, y);
                    // A neutral grey cloud field, plus blown highlights well
                    // above 1.0 for the airlight search to land on.
                    //
                    // The highlights are BLUE-TINTED, which is the part that
                    // matters. A neutral airlight cannot shift a hue however
                    // badly the clamping behaves -- all three channels recover
                    // identically -- so a neutral fixture would pass whether
                    // the bug were present or not. The real image's airlight
                    // came from a sunlit cloud edge and was not neutral, and
                    // it is the GAP between channels that decides which one
                    // reaches the clamp first.
                    const bool hot = (x > W - 12 && y < 12);
                    if (hot) { p[0] = 2.4f; p[1] = 2.7f; p[2] = 3.0f; }
                    else     { p[0] = p[1] = p[2] = 0.72f; }
                    p[3] = 1.0f;
                }
        }

        Image out;
        std::string e2;
        if (RunFilter("dehaze", sky.Clone(),
                      {{"strength", 1.0}, {"sky_protect", 0.0}}, &out, &e2)) {
            ImageView v = out.MapCpuRead();

            // Worst hue departure over the neutral field. The input is exactly
            // neutral, so any spread between channels is introduced by the
            // recovery -- there is nothing else it could come from.
            float worst = 0.0f;
            for (int y = 20; y < H; ++y)
                for (int x = 0; x < W - 20; ++x) {
                    const float* p = v.At<float>(x, y);
                    const float hi = std::max(p[0], std::max(p[1], p[2]));
                    const float lo = std::min(p[0], std::min(p[1], p[2]));
                    worst = std::max(worst, hi - lo);
                }
            Check(worst < 0.02f,
                  "a neutral sky stays neutral however hard it is dehazed "
                  "(worst channel spread " + std::to_string(worst) + ")");
        } else {
            Check(false, "dehaze runs on a scene-linear sky: " + e2);
        }
    }
}

// resize, and the proxy-scale bookkeeping the interactive path rests on.
static void TestResize() {
    const int W = 240, H = 160;

    // A gradient plus fine detail. The gradient checks that resampling does not
    // shift or bias; the fine detail is what ALIASES if a minify samples rather
    // than averaging, which is the failure this is really guarding.
    Image src;
    src.Alloc({W, H, Format::RGBA32F});
    {
        ImageView v = src.MapCpuWrite();
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                float* p = v.At<float>(x, y);
                p[0] = float(x) / float(W - 1);
                p[1] = float(y) / float(H - 1);
                p[2] = ((x + y) % 2 == 0) ? 1.0f : 0.0f;   // 1-px checker
                p[3] = 1.0f;
            }
    }

    auto at = [](const Image& im, int x, int y, int c) {
        ImageView v = const_cast<Image&>(im).MapCpuRead();
        return v.At<float>(x, y)[c];
    };

    std::string err;

    // MINIFY averages rather than samples.
    //
    // The blue channel is a one-pixel checker, so its true local mean is 0.5
    // everywhere. An area average returns that; point or bilinear sampling
    // returns values near 0 or 1 depending which pixel it lands on -- and those
    // MOVE as the scale changes, which is what makes a sampled proxy shimmer
    // while a slider drags.
    {
        Image tiny;
        if (RunFilter("resize", src.Clone(), {{"scale", 0.25}}, &tiny, &err)) {
            Check(tiny.Desc().width == 60 && tiny.Desc().height == 40,
                  "resize scales the raster (" +
                      std::to_string(tiny.Desc().width) + "x" +
                      std::to_string(tiny.Desc().height) + ")");

            double worst = 0.0;
            for (int y = 2; y < 38; ++y)
                for (int x = 2; x < 58; ++x)
                    worst = std::max(worst,
                        std::fabs(double(at(tiny, x, y, 2)) - 0.5));
            Check(worst < 0.1,
                  "minify AREA-AVERAGES rather than sampling (checker mean "
                  "off by at most " + std::to_string(worst) + ")");

            // And the gradient survives: the first and last columns should read
            // close to their source positions, not shifted by half a pixel.
            Check(at(tiny, 0, 20, 0) < 0.05f && at(tiny, 59, 20, 0) > 0.95f,
                  "minify keeps the gradient aligned (" +
                      std::to_string(at(tiny, 0, 20, 0)) + ".." +
                      std::to_string(at(tiny, 59, 20, 0)) + ")");

            // THE SCALE FACTOR IS RECORDED. Everything downstream depends on
            // this: an algorithm with a pixel-unit parameter reads it to know
            // how much to shrink its radius by.
            Check(std::fabs(tiny.Desc().proxyScale - 0.25f) < 1e-5f,
                  "resize records proxyScale (" +
                      std::to_string(tiny.Desc().proxyScale) + ")");

            // ...and it COMPOUNDS. Half of a half is a quarter of the original,
            // and a downstream stage needs the total rather than the last step.
            Image tinier;
            if (RunFilter("resize", tiny.Clone(), {{"scale", 0.5}}, &tinier, &err))
                Check(std::fabs(tinier.Desc().proxyScale - 0.125f) < 1e-5f,
                      "proxyScale compounds across resizes (" +
                          std::to_string(tinier.Desc().proxyScale) + ")");
        } else {
            Check(false, "resize minify runs: " + err);
        }
    }

    // MAGNIFY interpolates, and does not shift.
    //
    // A half-pixel offset error is invisible on a single resize and obvious on
    // a proxy ROUND TRIP, where it shows as the preview sliding when the scale
    // changes. Checking the gradient's endpoints catches it directly.
    {
        Image big;
        if (RunFilter("resize", src.Clone(), {{"scale", 2.0}}, &big, &err)) {
            Check(big.Desc().width == 480 && big.Desc().height == 320,
                  "resize magnifies (" + std::to_string(big.Desc().width) + "x" +
                      std::to_string(big.Desc().height) + ")");
            Check(at(big, 0, 160, 0) < 0.05f && at(big, 479, 160, 0) > 0.95f,
                  "magnify keeps the gradient aligned (" +
                      std::to_string(at(big, 0, 160, 0)) + ".." +
                      std::to_string(at(big, 479, 160, 0)) + ")");
        } else {
            Check(false, "resize magnify runs: " + err);
        }
    }

    // THE ROUND TRIP, which is what the proxy path actually does: shrink, work,
    // grow back. The result must land where it started -- softer, certainly,
    // but not shifted and not differently exposed.
    {
        Image tiny, back;
        if (RunFilter("resize", src.Clone(), {{"scale", 0.25}}, &tiny, &err) &&
            RunFilter("resize", tiny.Clone(), {{"scale", 4.0}}, &back, &err)) {

            Check(back.Desc().width == W && back.Desc().height == H,
                  "the round trip returns to the original size");

            // Scale back to 1.0, because a full-resolution run must not inherit
            // a proxy's scale -- that is the cache-poisoning failure.
            Check(std::fabs(back.Desc().proxyScale - 1.0f) < 1e-5f,
                  "the round trip restores proxyScale to 1.0 (" +
                      std::to_string(back.Desc().proxyScale) + ")");

            // MEAN preserved: the round trip may soften but must not darken or
            // brighten. Measured on the gradient channel, away from the border.
            double a = 0.0, b = 0.0;
            int n = 0;
            for (int y = 8; y < H - 8; ++y)
                for (int x = 8; x < W - 8; ++x) {
                    a += double(at(src,  x, y, 0));
                    b += double(at(back, x, y, 0));
                    ++n;
                }
            const double da = a / n, db = b / n;
            Check(std::fabs(da - db) < 0.02,
                  "the round trip preserves the mean (" + std::to_string(da) +
                      " -> " + std::to_string(db) + ")");
        } else {
            Check(false, "resize round trip runs: " + err);
        }
    }

    // scale = 1 is a genuine pass-through, so leaving a resize in a chain costs
    // nothing when proxying is off.
    {
        Image same;
        if (RunFilter("resize", src.Clone(), {{"scale", 1.0}}, &same, &err)) {
            double worst = 0.0;
            for (int y = 0; y < H; y += 7)
                for (int x = 0; x < W; x += 7)
                    for (int c = 0; c < 3; ++c)
                        worst = std::max(worst,
                            std::fabs(double(at(same, x, y, c)) -
                                      double(at(src, x, y, c))));
            Check(worst < 1e-6, "resize at scale 1 changes nothing (" +
                                    std::to_string(worst) + ")");
        }
    }
}

// Which algorithms may run on a reduced-resolution proxy.
//
// Checked over the WHOLE REGISTRY rather than a handful of names, because the
// failure this guards is an algorithm added later that quietly inherits the
// wrong default. A demosaic run on a downscaled mosaic produces mush and looks
// like a demosaic bug; a detector run on one produces keypoints in the wrong
// coordinate system and looks like an alignment bug. Neither points at the
// proxy machinery that caused it.
static void TestProxyBehaviour() {
    using PB = AlgorithmBase::ProxyBehaviour;

    // Everything that produces or consumes a SIDECAR. Listed by name rather
    // than derived from the category, because the categories do not line up:
    // detect_* and draw_* say "features", match_* says "match", and
    // align_features and bundle_adjust say "merge". A category test reads
    // correctly and silently misses two thirds of them -- which is why this
    // list is written out, and why the test exists at all.
    const std::set<std::string> sidecar = {
        "detect_sift", "detect_surf", "detect_akaze", "detect_orb",
        "detect_brisk", "match_brute", "match_ann", "align_features",
        "bundle_adjust", "draw_features", "draw_matches",
    };

    int never = 0, other = 0;
    for (const std::string& name : Registry::Get().Names()) {
        auto a = Registry::Get().Create(name);
        if (!a) continue;
        const std::string cat = a->Category();
        const PB pb = a->Proxy();

        // A CFA mosaic cannot survive being downscaled -- the pattern IS the
        // data. Sidecar coordinates are in image pixels and nothing rescales
        // them.
        if (cat == "demosaic" || sidecar.count(name)) {
            Check(pb == PB::Never,
                  name + " (" + cat + ") must not run on a proxy");
            ++never;
        } else {
            Check(pb != PB::Never,
                  name + " (" + cat + ") should be proxyable");
            ++other;
        }
    }

    Check(never > 10 && other > 20,
          "the proxy audit covered the registry (" + std::to_string(never) +
              " never, " + std::to_string(other) + " proxyable)");

    // resize is Exact rather than Scaled: it IS the scale change, so scaling it
    // by the current factor as well would compound it twice.
    {
        auto r = Registry::Get().Create("resize");
        if (r) Check(r->Proxy() == PB::Exact, "resize is Exact, not Scaled");
    }

    // --- region declarations ------------------------------------------------
    //
    // An algorithm that reads neighbours must SAY how far, or a region-limited
    // run puts a seam at every tile edge. Checked against the set that already
    // scales a pixel-unit parameter, because the two are the same property
    // seen from different sides: a radius worth scaling is a radius worth
    // declaring.
    {
        const std::set<std::string> neighbourhood = {
            "gaussian_blur", "box_blur", "median_blur", "bilateral",
            "kuwahara", "kuwahara_generalized", "symmetric_nearest",
            "nonlocal_means", "guided_filter", "bloom", "orton",
            "threshold_niblack", "threshold_sauvola", "dehaze",
        };
        for (const std::string& n : neighbourhood) {
            auto a = Registry::Get().Create(n);
            if (!a) continue;
            Check(a->ReachPixels() > 0,
                  n + " declares how far it reads (" +
                      std::to_string(a->ReachPixels()) + " px)");
        }

        // A per-pixel algorithm must declare zero, or every region would be
        // grown for no reason.
        for (const char* n : {"brightness", "grayscale", "apply_lut",
                              "vignette", "basic_adjust"}) {
            auto a = Registry::Get().Create(n);
            if (a)
                Check(a->ReachPixels() == 0,
                      std::string(n) + " is per-pixel and needs no margin");
        }

        // Whole-image statistics are safe to SHRINK and unsafe to CROP -- a
        // distinct question from ProxyBehaviour, which is why it is a separate
        // declaration rather than a reuse of it.
        for (const char* n : {"threshold_otsu", "tonemap", "dehaze"}) {
            auto a = Registry::Get().Create(n);
            if (a) {
                Check(!a->RegionSafe(),
                      std::string(n) + " measures the whole frame, so it "
                      "cannot run on a region");
                Check(a->Proxy() != PB::Never,
                      std::string(n) + " ...but is still fine at reduced scale");
            }
        }
    }
}

int main() {
    std::string err;
    TestKelvinWhiteBalance();
    TestVignette();
    TestFilmGrain();
    TestOrton();
    TestLut();
    TestDehaze();
    TestResize();
    TestProxyBehaviour();


// --- brightness handles every pixel format ---------------------------------
//
// It branched on RGBA8 and R32F by hand and fell out of the bottom for anything
// else, so a demosaiced raw (RGBA16F) came out as ZEROS -- a black image, no
// error anywhere. Measured before the fix: 0.5 in, 0.0000 out.
//
// The general form of this is worth stating: an `if` chain that handles the
// formats it knows and silently does nothing otherwise looks correct until a
// new format arrives, and then fails in the way that is hardest to trace. Every
// other algorithm goes through PixelBuffer, whose switch now names each format
// and complains about anything else.
{
    struct Case { const char* label; Format fmt; };
    const Case cases[] = {
        {"RGBA8",   Format::RGBA8},
        {"R32F",    Format::R32F},
        {"RGBA16F", Format::RGBA16F},
        {"RGBA32F", Format::RGBA32F},
    };

    for (const Case& tc : cases) {
        Image img;
        img.Alloc({8, 8, tc.fmt});
        {
            ImageView v = img.MapCpuWrite();
            for (int y = 0; y < 8; ++y)
                for (int x = 0; x < 8; ++x) {
                    switch (tc.fmt) {
                        case Format::RGBA8: {
                            uint8_t* p = v.At<uint8_t>(x, y);
                            p[0] = p[1] = p[2] = 128; p[3] = 255;
                            break;
                        }
                        case Format::R32F:
                            *v.At<float>(x, y) = 0.5f;
                            break;
                        case Format::RGBA16F: {
                            uint16_t* p = v.At<uint16_t>(x, y);
                            for (int c = 0; c < 3; ++c) p[c] = FloatToHalf(0.5f);
                            p[3] = FloatToHalf(1.0f);
                            break;
                        }
                        default: {
                            float* p = v.At<float>(x, y);
                            p[0] = p[1] = p[2] = 0.5f; p[3] = 1.0f;
                            break;
                        }
                    }
                }
        }

        // gain 2 doubles the value, whatever the format's units.
        Image out;
        std::string e;
        if (!RunFilter("brightness", std::move(img), {{"gain", 2.0}}, &out, &e)) {
            Check(false, std::string(tc.label) + ": brightness runs: " + e);
            continue;
        }

        ImageView v = out.MapCpuRead();
        double got = 0.0;
        switch (tc.fmt) {
            case Format::RGBA8:   got = double(v.At<uint8_t>(4, 4)[0]) / 255.0; break;
            case Format::R32F:    got = double(*v.At<float>(4, 4));             break;
            case Format::RGBA16F: got = double(HalfToFloat(v.At<uint16_t>(4, 4)[0])); break;
            default:              got = double(v.At<float>(4, 4)[0]);           break;
        }

        // 0.5 doubled is 1.0, which RGBA8 clamps to exactly 1.0 (255).
        Check(got > 0.9,
              std::string(tc.label) + ": gain 2 doubles the value (0.5 -> " +
                  std::to_string(got) + ", NOT zeros)");
    }

    // The parameters say what they do. `amount` said nothing -- an amount of
    // what? -- and sat next to `gain`, which does.
    auto probe = Registry::Get().Create("brightness");
    bool hasBrightness = false, hasGain = false, hasAmount = false;
    for (ParamBase* p : probe->Params()) {
        const std::string n = p->Name();
        if (n == "brightness") hasBrightness = true;
        if (n == "gain")       hasGain = true;
        if (n == "amount")     hasAmount = true;
    }
    Check(hasBrightness && hasGain, "brightness exposes 'brightness' and 'gain'");
    Check(!hasAmount, "...and no longer exposes 'amount'");
}

// --- every single-input algorithm handles every format ---------------------
//
// Two format bugs were found one at a time by hand, both AFTER a grep-based
// survey said the code was clean: brightness produced zeros on the float
// formats, and grayscale bailed out on anything but RGBA8 -- `if (format !=
// RGBA8) return;` -- leaving its output as allocated, i.e. black. Since
// grayscale is the first stage in hello.tgl, everything downstream went black
// too, which read as a broken brightness slider.
//
// Enumerating the registry is the only way to be sure there is not a third.
// Anything that produces an all-zero output from a non-zero input has not run
// at all: outputs are allocated zero-filled.
{
    struct FmtCase { const char* label; Format fmt; };
    const FmtCase fmts[] = {
        {"RGBA8",   Format::RGBA8},
        {"R32F",    Format::R32F},
        {"RGBA16F", Format::RGBA16F},
        {"RGBA32F", Format::RGBA32F},
    };

    // A step edge, so an edge detector has something to find -- on a flat or
    // smoothly-varying image it legitimately outputs zeros and would look
    // broken.
    auto makeInput = [](Format fmt) {
        Image img;
        img.Alloc({24, 24, fmt});
        ImageView v = img.MapCpuWrite();
        for (int y = 0; y < 24; ++y)
            for (int x = 0; x < 24; ++x) {
                const float t = (x < 12) ? 0.2f : 0.8f;
                switch (fmt) {
                    case Format::RGBA8: {
                        uint8_t* p = v.At<uint8_t>(x, y);
                        p[0] = uint8_t(t * 255); p[1] = uint8_t(t * 200);
                        p[2] = uint8_t(t * 150); p[3] = 255;
                        break;
                    }
                    case Format::R32F:
                        *v.At<float>(x, y) = t;
                        break;
                    case Format::RGBA16F: {
                        uint16_t* p = v.At<uint16_t>(x, y);
                        p[0] = FloatToHalf(t); p[1] = FloatToHalf(t * 0.8f);
                        p[2] = FloatToHalf(t * 0.6f); p[3] = FloatToHalf(1.0f);
                        break;
                    }
                    default: {
                        float* p = v.At<float>(x, y);
                        p[0] = t; p[1] = t * 0.8f; p[2] = t * 0.6f; p[3] = 1.0f;
                        break;
                    }
                }
            }
        return img;
    };

    auto meanOf = [](const ImageView& v) {
        double sum = 0; long long n = 0;
        for (int y = 0; y < v.desc.height; ++y)
            for (int x = 0; x < v.desc.width; ++x) {
                switch (v.desc.format) {
                    case Format::RGBA8:   sum += v.At<uint8_t>(x, y)[0] / 255.0; break;
                    case Format::R32F:    sum += *v.At<float>(x, y);             break;
                    case Format::RGBA16F: sum += HalfToFloat(v.At<uint16_t>(x, y)[0]); break;
                    default:              sum += v.At<float>(x, y)[0];           break;
                }
                ++n;
            }
        return n ? sum / n : -1.0;
    };

    // Excluded, with reasons -- not blanket exemptions.
    //
    // The threshold family maps values to 0 or 1 against a level whose range is
    // fixed at 0..255, so on a float image everything falls below it and black
    // is the correct answer for that level. The level not following the format
    // is a real issue, tracked in todo.txt alongside basic_adjust's ranges, but
    // it is a different bug from "the algorithm never ran".
    auto excluded = [](const std::string& n) {
        return n.rfind("threshold", 0) == 0 || n.rfind("mosaic", 0) == 0;
    };

    std::string offenders;
    for (const std::string& name : Registry::Get().Names()) {
        if (excluded(name)) continue;

        auto probe = Registry::Get().Create(name);
        if (!probe) continue;
        // Wiring one source into a multi-input stage would read unset ports.
        if (probe->Inputs().size() != 1) continue;
        const size_t outs = probe->Outputs().size();

        for (const FmtCase& f : fmts) {
            auto algo = Registry::Get().Create(name);
            Pipeline pipe;
            std::vector<Data> sources;
            sources.push_back(Data{makeInput(f.fmt)});
            pipe.AddStage(std::move(algo), name, {{-1, 0}}, outs, 1);

            std::string e;
            if (!pipe.Execute(&sources, nullptr, &e)) continue;   // a real refusal

            const Data* out = pipe.Resolve({0, 0}, &sources);
            if (!out || !std::holds_alternative<Image>(*out)) continue;
            ImageView v = const_cast<Image&>(std::get<Image>(*out)).MapCpuRead();
            if (!v.Valid()) continue;

            if (meanOf(v) < 1e-6) {
                offenders += (offenders.empty() ? "" : ", ");
                offenders += name + " on " + f.label;
            }
        }
    }

    Check(offenders.empty(),
          "no algorithm leaves its output untouched on any format" +
              (offenders.empty() ? std::string() : " (" + offenders + ")"));
}

    // --- local thresholds recover text under uneven lighting -----------------
    //
    // The point of a local method, and the one thing no global threshold can
    // do: a page lit brightly at one end and dimly at the other has no single
    // level that works across it.
    //
    // Worth having because the failure mode is confusing rather than obvious.
    // Run on a smooth gradient with no local structure, every local method
    // correctly returns nearly uniform white -- which looks broken. Tim reported
    // exactly that from thresholds.tgl on test.png, and the algorithms were
    // fine; the fixture had nothing for them to find.
    {
        const int W = 192, H = 192;
        auto strokeAt = [](int x, int y) { return ((y / 4) % 3 == 0) && (x % 16 < 11); };

        Image page;
        page.Alloc({W, H, Format::RGBA8});
        {
            ImageView v = page.MapCpuWrite();
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x) {
                    // Paper falling from 220 to 90 across the frame. Text sits a
                    // constant 60 below the LOCAL paper level, so it stays
                    // equally legible at both ends -- and no global threshold
                    // can separate them, because dark paper on the right is
                    // darker than bright text on the left.
                    const double paper = 220.0 - 130.0 * (double(x) / (W - 1));
                    const double val = strokeAt(x, y) ? paper - 60.0 : paper;
                    uint8_t* p = v.At<uint8_t>(x, y);
                    p[0] = p[1] = p[2] = uint8_t(std::clamp(val, 0.0, 255.0));
                    p[3] = 255;
                }
        }

        auto agreement = [&](const char* name) -> double {
            Image out;
            std::string e;
            if (!RunFilter(name, page.Clone(), {}, &out, &e)) return -1.0;
            ImageView v = out.MapCpuRead();
            int correct = 0, total = 0;
            for (int y = 2; y < H - 2; ++y)
                for (int x = 2; x < W - 2; ++x) {
                    const bool isText  = strokeAt(x, y);
                    const bool sawText = *v.At<float>(x, y) < 0.5f;
                    if (isText == sawText) ++correct;
                    ++total;
                }
            return total ? 100.0 * double(correct) / double(total) : -1.0;
        };

        for (const char* name : {"threshold_niblack", "threshold_sauvola",
                                 "threshold_bernsen", "threshold_adaptive_mean",
                                 "threshold_adaptive_gaussian"}) {
            const double pct = agreement(name);
            Check(pct > 90.0,
                  std::string(name) + " recovers text under a lighting gradient (" +
                      std::to_string(pct) + "%)");
        }

        // And the contrast that justifies them existing: a global threshold
        // cannot do this, so if one ever scores as well the fixture has stopped
        // testing what it claims to.
        const double otsu = agreement("threshold_otsu");
        Check(otsu < 80.0,
              "a global threshold cannot (" + std::to_string(otsu) +
                  "%), which is why the local ones exist");
    }
    // --- the shared plumbing ------------------------------------------------
    {
        Image img = MakeEdgeImage(false);
        ImageView v = img.MapCpuRead();

        PixelBuffer buf;
        buf.Unpack(v);
        Check(buf.Width() == kW && buf.Height() == kH && buf.Channels() == 4,
              "PixelBuffer unpacks an RGBA8 image");
        Check(std::abs(buf.Get(0, 0, 0) - double(kDark)) < 1e-4,
              "unpacked values keep the source's 0..255 scale");
        Check(std::abs(buf.ValueScale() - 255.0f) < 1e-4,
              "ValueScale reports the 8-bit range");

        // Edge clamping, not zero padding: a border read must not darken.
        Check(std::abs(buf.AtClamped(-5, -5)[0] - buf.Get(0, 0, 0)) < 1e-4,
              "out-of-bounds reads clamp to the edge");

        Image round;
        round.Alloc({kW, kH, Format::RGBA8});
        ImageView rv = round.MapCpuWrite();
        buf.PackInto(rv);
        Check(std::abs(PixelAt(round, 5, 5) - double(kDark)) < 1e-4 &&
              std::abs(PixelAt(round, kW - 5, 5) - double(kBright)) < 1e-4,
              "unpack then pack round-trips unchanged");
    }

    // --- every filter must actually smooth ----------------------------------
    //
    // Run each over the noisy fixture and require the flat-region deviation to
    // drop. This is the check that catches a filter which silently does
    // nothing -- the failure mode a "does it compile" test cannot see.
    {
        Image noisy = MakeEdgeImage(true);
        const double before = FlatRegionStdDev(noisy);
        Check(before > 10.0, "the noisy fixture really is noisy");

        const std::vector<std::pair<std::string, std::vector<std::pair<std::string, double>>>>
            filters = {
                {"box_blur",              {{"radius", 2}}},
                {"gaussian_blur",         {{"sigma", 1.5}}},
                {"median_blur",           {{"radius", 2}}},
                {"bilateral",             {{"sigma_space", 3.0}, {"sigma_range", 0.15}}},
                // Large k, because Perona-Malik treats a ±255 impulse as a
                // strong edge and preserves it at small k -- see the dedicated
                // conductance check below.
                {"anisotropic_diffusion", {{"iterations", 15}, {"k", 0.9}}},
                {"kuwahara",              {{"radius", 3}}},
                {"kuwahara_generalized",  {{"radius", 4}, {"sectors", 8}}},
                {"guided_filter",         {{"radius", 3}, {"eps", 0.1}}},
                {"symmetric_nearest",     {{"radius", 3}}},
                // nonlocal_means is checked separately: patch distances around
                // a ±255 impulse are enormous, so like Perona-Malik it treats
                // them as structure to keep. It is a denoiser for broad-band
                // noise, not an impulse filter.
            };

        for (const auto& [name, params] : filters) {
            Image out;
            if (!RunFilter(name, MakeEdgeImage(true), params, &out, &err)) {
                Check(false, name + " ran: " + err);
                continue;
            }
            const double after = FlatRegionStdDev(out);
            Check(after < before,
                  name + " reduces noise in a flat region (" +
                      std::to_string(int(before)) + " -> " + std::to_string(int(after)) + ")");
        }
    }

    // --- edge preservation is what separates them ---------------------------
    //
    // On a clean step edge, a Gaussian bleeds across it and the edge-preserving
    // filters do not. Asserting each is *better than the Gaussian* states the
    // property that matters without hard-coding numbers that would drift.
    {
        Image gauss;
        if (RunFilter("gaussian_blur", MakeEdgeImage(false), {{"sigma", 3.0}}, &gauss, &err)) {
            const double gaussBleed = EdgeBleed(gauss);
            Check(gaussBleed > 2.0, "a Gaussian visibly bleeds across the edge");

            const std::vector<std::pair<std::string, std::vector<std::pair<std::string, double>>>>
                preserving = {
                    {"median_blur",           {{"radius", 3}}},
                    {"bilateral",             {{"sigma_space", 3.0}, {"sigma_range", 0.08}}},
                    {"anisotropic_diffusion", {{"iterations", 20}, {"k", 0.03}}},
                    {"kuwahara",              {{"radius", 3}}},
                    {"kuwahara_generalized",  {{"radius", 4}, {"sharpness", 8.0}}},
                    {"guided_filter",         {{"radius", 3}, {"eps", 0.05}}},
                    {"symmetric_nearest",     {{"radius", 3}}},
                };

            for (const auto& [name, params] : preserving) {
                Image out;
                if (!RunFilter(name, MakeEdgeImage(false), params, &out, &err)) {
                    Check(false, name + " ran: " + err);
                    continue;
                }
                Check(EdgeBleed(out) < gaussBleed,
                      name + " preserves the edge better than a Gaussian");
            }
        } else {
            Check(false, "gaussian_blur ran: " + err);
        }
    }

    // --- median is the one that must reject impulses outright ---------------
    {
        Image out;
        if (RunFilter("median_blur", MakeEdgeImage(true), {{"radius", 2}}, &out, &err)) {
            // (3, 0) is an impulse set to 255 in a region whose true value is
            // kDark. A mean would be dragged upwards; a median discards it.
            const double v = PixelAt(out, 3, 0);
            Check(std::abs(v - double(kDark)) < 12.0,
                  "median_blur removes an impulse rather than smearing it");
        } else {
            Check(false, "median_blur ran: " + err);
        }
    }

    // --- nonlocal_means on the noise it is designed for ---------------------
    //
    // Broad-band perturbation rather than impulses. NLM finds matching patches
    // elsewhere in the image and averages them, so a repeating pattern with
    // added wobble is exactly its strong case -- and the closest small fixture
    // to the scanned-texture work it is meant for.
    {
        auto makeTexture = []() {
            Image img;
            img.Alloc({kW, kH, Format::RGBA8});
            ImageView v = img.MapCpuWrite();
            for (int y = 0; y < kH; ++y)
                for (int x = 0; x < kW; ++x) {
                    // A repeating base pattern NLM can match across the image,
                    // plus a deterministic ±20 perturbation to remove.
                    const int base   = ((x / 4 + y / 4) % 2) ? 90 : 170;
                    const int wobble = (((x * 5 + y * 11) % 9) - 4) * 5;
                    const uint8_t s  = uint8_t(std::clamp(base + wobble, 0, 255));
                    uint8_t* p = v.At<uint8_t>(x, y);
                    p[0] = p[1] = p[2] = s;
                    p[3] = 255;
                }
            return img;
        };

        // Deviation from the clean repeating pattern.
        auto patternError = [](Image& img) {
            ImageView v = img.MapCpuRead();
            double sum = 0.0;
            int    n   = 0;
            for (int y = 4; y < kH - 4; ++y)
                for (int x = 4; x < kW - 4; ++x) {
                    const double ideal = ((x / 4 + y / 4) % 2) ? 90.0 : 170.0;
                    sum += std::abs(double(v.At<uint8_t>(x, y)[0]) - ideal);
                    ++n;
                }
            return n ? sum / n : 0.0;
        };

        Image noisy = makeTexture();
        Image out;
        if (RunFilter("nonlocal_means", makeTexture(),
                      {{"patch", 1}, {"search", 6}, {"strength", 0.12}}, &out, &err)) {
            Check(patternError(out) < patternError(noisy),
                  "nonlocal_means denoises a repeating texture");
        } else {
            Check(false, "nonlocal_means ran: " + err);
        }
    }

    // --- Perona-Malik: k is the edge/noise decision boundary ----------------
    //
    // The defining property of the algorithm. k sets what counts as an edge:
    // below it, gradients diffuse away; above it, they are preserved. A ±255
    // impulse is a very strong gradient, so small k *keeps* it and large k
    // removes it -- the opposite of how a linear blur responds to its radius,
    // and the thing to understand before tuning it on real scans.
    {
        Image sharp, smooth;
        const bool okSharp = RunFilter("anisotropic_diffusion", MakeEdgeImage(true),
                                       {{"iterations", 15}, {"k", 0.02}}, &sharp, &err);
        const bool okSmooth = RunFilter("anisotropic_diffusion", MakeEdgeImage(true),
                                        {{"iterations", 15}, {"k", 0.9}}, &smooth, &err);

        if (okSharp && okSmooth) {
            Check(FlatRegionStdDev(smooth) < FlatRegionStdDev(sharp),
                  "anisotropic_diffusion: raising k diffuses what small k preserves");
            Check(EdgeBleed(sharp) < EdgeBleed(smooth),
                  "anisotropic_diffusion: small k preserves edges more strongly");
        } else {
            Check(false, "anisotropic_diffusion ran at both k values: " + err);
        }
    }

    // --- what the generalized Kuwahara actually buys ------------------------
    //
    // Not diagonal accuracy: a square quadrant fits perfectly on one side of a
    // clean 45-degree step, so classic Kuwahara scores exactly 0 there. What
    // Papari et al. fix is the *hard* quadrant choice, which on noisy input
    // flips between quadrants and leaves blocky, directional artefacts. The
    // smooth variance weighting has no such discontinuity, so on a noisy flat
    // field it lands closer to the true value.
    {
        auto makeNoisyFlat = []() {
            Image img;
            img.Alloc({kW, kH, Format::RGBA8});
            ImageView v = img.MapCpuWrite();
            for (int y = 0; y < kH; ++y)
                for (int x = 0; x < kW; ++x) {
                    // Deterministic ±25 checker-ish perturbation around 128.
                    const int wobble = (((x * 7 + y * 13) % 11) - 5) * 5;
                    const uint8_t s = uint8_t(std::clamp(128 + wobble, 0, 255));
                    uint8_t* p = v.At<uint8_t>(x, y);
                    p[0] = p[1] = p[2] = s;
                    p[3] = 255;
                }
            return img;
        };

        auto deviationFrom128 = [](Image& img) {
            ImageView v = img.MapCpuRead();
            double sum = 0.0;
            int    n   = 0;
            for (int y = 5; y < kH - 5; ++y)
                for (int x = 5; x < kW - 5; ++x) {
                    sum += std::abs(double(v.At<uint8_t>(x, y)[0]) - 128.0);
                    ++n;
                }
            return n ? sum / n : 0.0;
        };

        Image plain, general;
        const bool a = RunFilter("kuwahara", makeNoisyFlat(), {{"radius", 4}}, &plain, &err);
        const bool b = RunFilter("kuwahara_generalized", makeNoisyFlat(),
                                 {{"radius", 4}, {"sectors", 8}, {"sharpness", 8.0}},
                                 &general, &err);
        if (a && b) {
            Check(deviationFrom128(general) < deviationFrom128(plain),
                  "kuwahara_generalized's smooth weighting beats hard quadrant "
                  "selection on noise");
        } else {
            Check(false, "both kuwahara variants ran: " + err);
        }
    }

    // --- a flat image must come back unchanged ------------------------------
    //
    // Every one of these is an average of nearby values, so a constant input
    // must be a fixed point. This catches normalisation errors -- notably the
    // sector weighting in kuwahara_generalized, which brightens or darkens if
    // the weights are not normalised across sectors.
    {
        const std::vector<std::pair<std::string, std::vector<std::pair<std::string, double>>>>
            filters = {
                {"box_blur",              {{"radius", 3}}},
                {"gaussian_blur",         {{"sigma", 2.0}}},
                {"median_blur",           {{"radius", 2}}},
                {"bilateral",             {{"sigma_space", 3.0}}},
                {"anisotropic_diffusion", {{"iterations", 10}}},
                {"kuwahara",              {{"radius", 3}}},
                {"kuwahara_generalized",  {{"radius", 4}, {"sectors", 8}}},
                {"guided_filter",         {{"radius", 3}}},
                {"nonlocal_means",        {{"patch", 1}, {"search", 3}}},
                {"symmetric_nearest",     {{"radius", 3}}},
            };

        for (const auto& [name, params] : filters) {
            Image flat;
            flat.Alloc({kW, kH, Format::RGBA8});
            ImageView fv = flat.MapCpuWrite();
            for (int y = 0; y < kH; ++y)
                for (int x = 0; x < kW; ++x) {
                    uint8_t* p = fv.At<uint8_t>(x, y);
                    p[0] = p[1] = p[2] = 128; p[3] = 255;
                }

            Image out;
            if (!RunFilter(name, std::move(flat), params, &out, &err)) {
                Check(false, name + " ran on a flat image: " + err);
                continue;
            }

            double worst = 0.0;
            for (int y = 0; y < kH; ++y)
                for (int x = 0; x < kW; ++x)
                    worst = std::max(worst, std::abs(PixelAt(out, x, y) - 128.0));

            Check(worst <= 1.0,
                  name + " leaves a flat image unchanged (max drift " +
                      std::to_string(worst) + ")");
        }
    }


    // --- basic_adjust on scene-linear input ---------------------------------
    //
    // A demosaiced raw is scene-linear with real headroom above 1.0, which is
    // a different kind of image from the RGBA8 the controls were written
    // against. Two things went wrong and neither is visible on 8-bit input:
    //
    //   1. The sRGB transfer functions were applied unconditionally, decoding
    //      already-linear data as though it were gamma-encoded, and the encode
    //      on the way out clamped to 1.0 -- discarding every value above it.
    //      Measured on a real CR2, a peak of 1.77 came out as 0.9995.
    //
    //   2. The highlight band ran to a hardcoded 1.0. Measured across six raws
    //      from two bodies, peak luminance after demosaic ran 0.25 to 0.75, so
    //      on every one of them the band never engaged and the highlights
    //      slider did nothing. This is the "ranges are too small" report: the
    //      control was fine, the band was pitched for gamma-encoded data.
    {
        // Scene-linear, flagged as such, with a highlight above 1.0 that only
        // survives if nothing clamps -- the case that matters and that an
        // RGBA8 fixture structurally cannot express.
        auto makeLinear = []() {
            ImageDesc d{32, 32, Format::RGBA16F};
            d.linear = true;
            Image img;
            img.Alloc(d);
            ImageView v = img.MapCpuWrite();
            for (int y = 0; y < 32; ++y)
                for (int x = 0; x < 32; ++x) {
                    // Top third genuinely blown (1.4), middle a highlight
                    // inside the band (0.6), bottom a midtone (0.2).
                    const float lum = (y < 11) ? 1.4f : (y < 22 ? 0.6f : 0.2f);
                    uint16_t* p = v.At<uint16_t>(x, y);
                    for (int c = 0; c < 3; ++c) p[c] = FloatToHalf(lum);
                    p[3] = FloatToHalf(1.0f);
                }
            return img;
        };

        auto bandMax = [](Image& img, int y0, int y1) {
            ImageView v = img.MapCpuRead();
            float hi = 0.0f;
            for (int y = y0; y < y1; ++y)
                for (int x = 0; x < 32; ++x)
                    hi = std::max(hi, HalfToFloat(v.At<uint16_t>(x, y)[0]));
            return double(hi);
        };

        Image out;
        std::string err;

        // Defaults must be identity on a linear image, headroom included. This
        // is the check that fails when the transfer functions are applied to
        // linear data: 1.4 came back as 0.9995.
        if (RunFilter("basic_adjust", makeLinear(), {}, &out, &err)) {
            Check(std::abs(bandMax(out, 0, 11) - 1.4) < 0.01,
                  "linear input: defaults preserve headroom above 1.0 (got " +
                      std::to_string(bandMax(out, 0, 11)) + ", want 1.4)");
            Check(std::abs(bandMax(out, 22, 32) - 0.2) < 0.01,
                  "linear input: defaults leave midtones alone");
        } else {
            Check(false, "basic_adjust runs on linear input: " + err);
        }

        // Pulling highlights down must actually reach the blown region. This
        // is the check that fails when the band tops out at 1.0.
        if (RunFilter("basic_adjust", makeLinear(), {{"highlights", -1.0}}, &out, &err)) {
            const double blown = bandMax(out, 0, 11);
            // Bounded on BOTH sides: it must come down, but a control that
            // drives the region to black is not recovering anything. The first
            // version of this check only tested blown < 1.0, and passed while
            // the value was exactly 0.
            Check(blown < 1.0 && blown > 0.1,
                  "linear input: highlights -1 recovers detail rather than erasing it"
                  " (1.4 -> " + std::to_string(blown) + ")");
            Check(std::abs(bandMax(out, 22, 32) - 0.2) < 0.02,
                  "linear input: highlights -1 leaves midtones alone (got " +
                      std::to_string(bandMax(out, 22, 32)) + ")");
        } else {
            Check(false, "basic_adjust highlights on linear input: " + err);
        }
        // Recovery must PRESERVE VARIATION in the highlights, not flatten it.
        //
        // Tim's report: on a tonemapped sky, pulling highlights down turned the
        // whole sky into a solid grey mass -- "I've never seen that behavior in
        // other photo editing apps". He was right, and the cause was that
        // recovery subtracted the ENTIRE excess above the band top, landing
        // every bright pixel on exactly that value.
        //
        // The tests above could not see it: their blown region is UNIFORM, so a
        // control that maps every bright pixel to one number passes happily. It
        // takes a gradient to tell compression from collapse.
        //
        // This matters far more for scene-linear data than it ever did for
        // sRGB. A gamma-encoded image has little above the band top and the
        // window still varies across it; a tonemapped sky sits entirely above
        // it, where SmoothBand saturates at 1.0 and every pixel gets the same
        // full-strength correction.
        {
            auto makeSky = []() {
                ImageDesc d{32, 32, Format::RGBA16F};
                d.linear = true;
                Image img;
                img.Alloc(d);
                ImageView v = img.MapCpuWrite();
                for (int y = 0; y < 32; ++y)
                    for (int x = 0; x < 32; ++x) {
                        // A bright gradient, all of it above kLinearShoulder
                        // (0.70) -- 1.0 at the top to 3.0 at the bottom, like
                        // cloud detail in a tonemapped sky.
                        const float lum = 1.0f + 2.0f * (float(y) / 31.0f);
                        uint16_t* p = v.At<uint16_t>(x, y);
                        for (int c = 0; c < 3; ++c) p[c] = FloatToHalf(lum);
                        p[3] = FloatToHalf(1.0f);
                    }
                return img;
            };

            Image sky;
            std::string skyErr;
            if (RunFilter("basic_adjust", makeSky(), {{"highlights", -1.0}}, &sky, &skyErr)) {
                const double top = bandMax(sky, 0, 4);     // was 1.0 in
                const double bot = bandMax(sky, 28, 32);   // was 3.0 in
                Check(bot > top * 1.15,
                      "highlights -1 keeps a bright gradient ORDERED rather than "
                      "flattening it (" + std::to_string(top) + " .. " +
                      std::to_string(bot) + ")");
                Check(bot < 3.0,
                      "and still brings the brightest end down");
            } else {
                Check(false, "basic_adjust on a bright gradient: " + skyErr);
            }
        }

        // The SHADOW band must stay in the shadows on linear input.
        //
        // Tim: "the shadows slider seems to brighten the entire scene except
        // clouds". The band was hardcoded 0.0 to 0.5, which is a gamma-encoded
        // assumption -- in sRGB 0.5 is roughly a midtone. In linear light
        // middle grey is 0.18, so that band handed middle grey 70% of the full
        // lift and only faded out near the highlights. It was a global
        // brightener with the top masked off, not a shadow control.
        //
        // Checked as a RATIO between what a shadow gets and what a midtone
        // gets. An absolute threshold would need re-tuning whenever the band
        // moves; the property that matters is that the shadows get much more.
        {
            auto makeTones = []() {
                ImageDesc d{32, 32, Format::RGBA16F};
                d.linear = true;
                Image img;
                img.Alloc(d);
                ImageView v = img.MapCpuWrite();
                for (int y = 0; y < 32; ++y)
                    for (int x = 0; x < 32; ++x) {
                        // A shadow (0.02), a midtone (0.18 = middle grey), and
                        // a highlight (0.60).
                        // The middle band is 0.10, NOT middle grey.
                        //
                        // 0.18 was the first choice and it could not fail: the
                        // band topped out there, so a midtone at exactly 0.18
                        // got zero lift by construction and the check passed
                        // against a band that was still far too wide. 0.10 is
                        // where Tim's valley merge actually puts its median --
                        // just under a stop below grey -- which is the tone that
                        // decides whether this reads as a shadow control or as a
                        // global brightener.
                        const float lum = (y < 11) ? 0.02f : (y < 22 ? 0.10f : 0.60f);
                        uint16_t* p = v.At<uint16_t>(x, y);
                        for (int c = 0; c < 3; ++c) p[c] = FloatToHalf(lum);
                        p[3] = FloatToHalf(1.0f);
                    }
                return img;
            };

            Image lifted;
            std::string sErr;
            if (RunFilter("basic_adjust", makeTones(), {{"shadows", 1.0}}, &lifted, &sErr)) {
                const double sh  = bandMax(lifted, 0, 11);    // 0.02 in
                const double mid = bandMax(lifted, 11, 22);   // 0.18 in
                const double hl  = bandMax(lifted, 22, 32);   // 0.60 in

                const double shGain  = sh  / 0.02;
                const double midGain = mid / 0.10;
                const double hlGain  = hl  / 0.60;

                Check(shGain > midGain * 1.5,
                      "shadows +1 lifts a shadow far more than a midtone (" +
                      std::to_string(shGain) + "x vs " + std::to_string(midGain) + "x)");
                Check(hlGain < 1.05,
                      "and leaves the highlights essentially alone (" +
                      std::to_string(hlGain) + "x)");
                Check(shGain > 1.3, "while actually lifting the shadow");
            } else {
                Check(false, "basic_adjust shadows on linear input: " + sErr);
            }
        }


        // The band pitch, isolated. Real raws mostly do NOT peak above 1.0 --
        // measured across six files, peak luminance ran 0.25 to 0.75 -- so the
        // case that matters is a highlight *below* 1.0 that the old band, which
        // ran from 0.35 to a hardcoded 1.0, barely engaged with. A fixture that
        // peaks above 1.0 cannot tell the two bands apart, because it clears
        // the old shoulder as well.
        auto makeDim = []() {
            ImageDesc d{32, 32, Format::RGBA16F};
            d.linear = true;
            Image img;
            img.Alloc(d);
            ImageView v = img.MapCpuWrite();
            for (int y = 0; y < 32; ++y)
                for (int x = 0; x < 32; ++x) {
                    // 0.7 is a highlight in a scene-linear raw, and is exactly
                    // where the old band's weight was still only ~0.4.
                    const float lum = (y < 16) ? 0.7f : 0.2f;
                    uint16_t* p = v.At<uint16_t>(x, y);
                    for (int c = 0; c < 3; ++c) p[c] = FloatToHalf(lum);
                    p[3] = FloatToHalf(1.0f);
                }
            return img;
        };

        if (RunFilter("basic_adjust", makeDim(), {{"highlights", -1.0}}, &out, &err)) {
            const double hl = bandMax(out, 0, 16);
            // Computed, not guessed, and recomputed when the recovery formula
            // changed from a uniform multiply to a pull-toward-the-top.
            //
            // At lum 0.7, band toe 0.35: with the 0.70 shoulder the weight is
            // 1.0, nothing sits above the top so the pull is a no-op, and the
            // residual darkening (1 - 0.35) gives 0.455. With the old 1.0
            // shoulder the weight would be 0.558, and the same arithmetic gives
            // 0.563. 0.50 separates them.
            Check(hl < 0.50,
                  "linear input: highlights reach a sub-1.0 highlight (0.7 -> " +
                      std::to_string(hl) + ")");
            Check(std::abs(bandMax(out, 16, 32) - 0.2) < 0.02,
                  "linear input: a sub-1.0 pull still leaves midtones alone");
        } else {
            Check(false, "basic_adjust on a dim linear image: " + err);
        }

        // Highlight recovery must DESATURATE a saturated highlight, not merely
        // darken it.
        //
        // A blown highlight is blown because a channel clipped, so recovering it
        // means bringing that channel back toward the others. Scaling all three
        // by one factor preserves their ratios exactly, so the region comes out
        // dark and just as colour-dense as before -- more so relative to its
        // surroundings, which is what reads as colour crushing.
        //
        // Measured with the uniform multiply: a 1.20/0.35/0.20 red held
        // saturation 0.833 at every slider position from 0 to -1.
        {
            auto makeColoured = []() {
                ImageDesc d{16, 16, Format::RGBA16F};
                d.linear = true;
                Image img;
                img.Alloc(d);
                ImageView v = img.MapCpuWrite();
                for (int y = 0; y < 16; ++y)
                    for (int x = 0; x < 16; ++x) {
                        uint16_t* p = v.At<uint16_t>(x, y);
                        p[0] = FloatToHalf(1.20f);   // red clipped past the top
                        p[1] = FloatToHalf(0.35f);
                        p[2] = FloatToHalf(0.20f);
                        p[3] = FloatToHalf(1.0f);
                    }
                return img;
            };

            auto saturationOf = [](Image& img) {
                ImageView v = img.MapCpuRead();
                const uint16_t* p = v.At<uint16_t>(8, 8);
                const float r = HalfToFloat(p[0]);
                const float g = HalfToFloat(p[1]);
                const float b = HalfToFloat(p[2]);
                const float mx = std::max({r, g, b});
                const float mn = std::min({r, g, b});
                return mx > 1e-6f ? double((mx - mn) / mx) : 0.0;
            };
            auto redOf = [](Image& img) {
                ImageView v = img.MapCpuRead();
                return double(HalfToFloat(v.At<uint16_t>(8, 8)[0]));
            };

            Image before, after;
            std::string e;
            if (RunFilter("basic_adjust", makeColoured(), {}, &before, &e) &&
                RunFilter("basic_adjust", makeColoured(), {{"highlights", -1.0}}, &after, &e)) {

                Check(redOf(after) < redOf(before) * 0.9,
                      "coloured highlight: recovery brings the clipped channel down (" +
                          std::to_string(redOf(before)) + " -> " +
                          std::to_string(redOf(after)) + ")");

                // The check that fails for a uniform multiply, where saturation
                // is identical to six decimal places.
                Check(saturationOf(after) < saturationOf(before) - 0.01,
                      "coloured highlight: recovery DESATURATES rather than only darkening"
                      " (" + std::to_string(saturationOf(before)) + " -> " +
                          std::to_string(saturationOf(after)) + ")");
            } else {
                Check(false, "basic_adjust on a coloured highlight: " + e);
            }
        }

        // And it must not be brutal on a neutral one. Recovery is meant to
        // reveal detail, not to dim the whole region towards black: the first
        // version took 0.90 down to 0.135, losing 85% of the brightness.
        {
            auto makeNeutral = []() {
                ImageDesc d{16, 16, Format::RGBA16F};
                d.linear = true;
                Image img;
                img.Alloc(d);
                ImageView v = img.MapCpuWrite();
                for (int y = 0; y < 16; ++y)
                    for (int x = 0; x < 16; ++x) {
                        uint16_t* p = v.At<uint16_t>(x, y);
                        for (int c = 0; c < 3; ++c) p[c] = FloatToHalf(0.90f);
                        p[3] = FloatToHalf(1.0f);
                    }
                return img;
            };

            Image out;
            std::string e;
            if (RunFilter("basic_adjust", makeNeutral(), {{"highlights", -1.0}}, &out, &e)) {
                ImageView v = out.MapCpuRead();
                const double got = double(HalfToFloat(v.At<uint16_t>(8, 8)[0]));
                Check(got < 0.85 && got > 0.30,
                      "neutral highlight: -1 darkens meaningfully but does not crush "
                      "(0.90 -> " + std::to_string(got) + ")");
            } else {
                Check(false, "basic_adjust on a neutral highlight: " + e);
            }
        }

        // Gamma-encoded input must be unaffected by all of the above: the sRGB
        // path still applies and the band still tops out at 1.0.
        Image srgb;
        if (RunFilter("basic_adjust", MakeEdgeImage(false), {}, &srgb, &err)) {
            Image again;
            if (RunFilter("basic_adjust", MakeEdgeImage(false), {}, &again, &err)) {
                ImageView a = srgb.MapCpuRead(), b = again.MapCpuRead();
                double worst = 0.0;
                for (int y = 0; y < a.desc.height; ++y)
                    for (int x = 0; x < a.desc.width; ++x)
                        worst = std::max(worst, std::abs(double(a.At<uint8_t>(x, y)[0]) -
                                                         double(b.At<uint8_t>(x, y)[0])));
                Check(worst < 1.0, "gamma-encoded input still takes the sRGB path unchanged");
            }
        }
    }
    // --- basic_adjust: each control does what its help text claims -----------
    //
    // Ten controls fused into one kernel means a mistake in any of them is
    // invisible in the others' output. Each is checked in isolation, in the
    // direction a photographer would expect.
    {
        // Mid-grey with a colour cast and a bright and dark patch, so tonal
        // controls have something in each band to act on.
        auto makeTest = []() {
            Image img;
            img.Alloc({64, 64, Format::RGBA8});
            ImageView v = img.MapCpuWrite();
            for (int y = 0; y < 64; ++y)
                for (int x = 0; x < 64; ++x) {
                    uint8_t* p = v.At<uint8_t>(x, y);
                    if (y < 16)      { p[0] = 230; p[1] = 230; p[2] = 230; }  // highlights
                    else if (y >= 48){ p[0] = 25;  p[1] = 25;  p[2] = 25;  }  // shadows
                    else             { p[0] = 140; p[1] = 120; p[2] = 100; }  // warm midtone
                    p[3] = 255;
                }
            return img;
        };

        // Mean over a band of rows, as a proxy for "did that band get brighter".
        auto bandMean = [](Image& img, int y0, int y1) {
            ImageView v = img.MapCpuRead();
            double sum = 0; int n = 0;
            for (int y = y0; y < y1; ++y)
                for (int x = 0; x < 64; ++x) {
                    const uint8_t* p = v.At<uint8_t>(x, y);
                    sum += (p[0] + p[1] + p[2]) / 3.0;
                    ++n;
                }
            return n ? sum / n : 0.0;
        };
        // Mean channel spread, as a proxy for saturation.
        auto chroma = [](Image& img, int y0, int y1) {
            ImageView v = img.MapCpuRead();
            double sum = 0; int n = 0;
            for (int y = y0; y < y1; ++y)
                for (int x = 0; x < 64; ++x) {
                    const uint8_t* p = v.At<uint8_t>(x, y);
                    const int mx = std::max(p[0], std::max(p[1], p[2]));
                    const int mn = std::min(p[0], std::min(p[1], p[2]));
                    sum += mx - mn;
                    ++n;
                }
            return n ? sum / n : 0.0;
        };

        Image base;
        if (!RunFilter("basic_adjust", makeTest(), {}, &base, &err)) {
            Check(false, "basic_adjust runs with defaults: " + err);
        } else {
            // Defaults must be a no-op, or every other control is measured
            // against a moving baseline -- and the user's untouched image
            // would silently change just by adding the algorithm.
            double worst = 0.0;
            // Held in a named Image: a view into a temporary would dangle the
            // moment the statement ended.
            Image pristine = makeTest();
            ImageView a = pristine.MapCpuRead();
            ImageView b = base.MapCpuRead();
            for (int y = 0; y < 64; ++y)
                for (int x = 0; x < 64; ++x)
                    for (int c = 0; c < 3; ++c)
                        worst = std::max(worst,
                                         std::abs(double(a.At<uint8_t>(x, y)[c]) -
                                                  double(b.At<uint8_t>(x, y)[c])));
            Check(worst <= 1.0,
                  "defaults leave the image unchanged (max drift " +
                      std::to_string(worst) + ")");
        }

        struct Case {
            const char* param;
            double      value;
            const char* claim;
        };

        // Each claim is the direction the help text promises.
        Image out;
        auto run = [&](const char* p, double v) {
            return RunFilter("basic_adjust", makeTest(), {{p, v}}, &out, &err);
        };

        if (run("exposure", 1.0))
            Check(bandMean(out, 16, 48) > bandMean(base, 16, 48) + 20,
                  "exposure +1 stop brightens the midtones");
        if (run("exposure", -1.0))
            Check(bandMean(out, 16, 48) < bandMean(base, 16, 48) - 20,
                  "exposure -1 stop darkens the midtones");

        // Highlights and shadows must act on their own band and largely leave
        // the other alone -- that is the whole point of them over exposure.
        if (run("highlights", -0.8)) {
            const double hi = bandMean(out, 0, 16), sh = bandMean(out, 48, 64);
            Check(hi < bandMean(base, 0, 16) - 10,
                  "negative highlights recovers the bright band");
            Check(std::abs(sh - bandMean(base, 48, 64)) < 8,
                  "...without disturbing the shadows");
        }
        if (run("shadows", 0.8)) {
            const double hi = bandMean(out, 0, 16), sh = bandMean(out, 48, 64);
            Check(sh > bandMean(base, 48, 64) + 5,
                  "positive shadows opens up the dark band");
            Check(std::abs(hi - bandMean(base, 0, 16)) < 8,
                  "...without disturbing the highlights");
        }

        if (run("contrast", 0.5))
            Check(bandMean(out, 0, 16) - bandMean(out, 48, 64) >
                  bandMean(base, 0, 16) - bandMean(base, 48, 64),
                  "positive contrast widens the gap between bands");

        if (run("saturation", -1.0))
            Check(chroma(out, 16, 48) < 2.0,
                  "saturation -1 is greyscale (chroma " +
                      std::to_string(chroma(out, 16, 48)) + ")");
        if (run("saturation", 0.5))
            Check(chroma(out, 16, 48) > chroma(base, 16, 48) + 2,
                  "positive saturation increases chroma");
        if (run("vibrance", 0.8))
            Check(chroma(out, 16, 48) > chroma(base, 16, 48),
                  "positive vibrance increases chroma on a muted colour");

        // Tint moves green against magenta without changing overall brightness
        // -- the gains are luminance-normalised precisely so a colour shift is
        // not also an exposure change.
        //
        // This fixture is RGBA8 with no daylight reference, so `kelvin` does
        // nothing here by design and there is nothing to assert about it. The
        // Kelvin control has its own section, on a fixture that carries the
        // metadata it needs.
        if (run("tint", -0.5)) {
            ImageView v = out.MapCpuRead();
            ImageView bv = base.MapCpuRead();
            const uint8_t* p = v.At<uint8_t>(32, 32);
            const uint8_t* q = bv.At<uint8_t>(32, 32);
            Check(p[1] > q[1], "negative tint pushes towards green");
            Check(std::abs(bandMean(out, 16, 48) - bandMean(base, 16, 48)) < 12,
                  "...without a large brightness shift");
        }
        if (run("tint", 0.5)) {
            ImageView v = out.MapCpuRead();
            ImageView bv = base.MapCpuRead();
            const uint8_t* p = v.At<uint8_t>(32, 32);
            const uint8_t* q = bv.At<uint8_t>(32, 32);
            Check(p[1] < q[1], "positive tint pushes towards magenta");
        }

        if (run("blacks", -0.5))
            Check(bandMean(out, 48, 64) < bandMean(base, 48, 64),
                  "negative blacks crushes the shadows");
        if (run("whites", 0.5))
            Check(bandMean(out, 0, 16) >= bandMean(base, 0, 16),
                  "positive whites raises the white point");
    }

    // --- cost, on request ---------------------------------------------------
    //
    // TGLAB_FILTER_BENCH=1 times each filter at a realistic scan size. These
    // span four orders of magnitude, so knowing which are interactive and which
    // are "start it and wait" matters more here than in most libraries.
    if (std::getenv("TGLAB_FILTER_BENCH")) {
        constexpr int bw = 1024, bh = 1024;
        std::printf("\n--- cost at %dx%d (%.1f MP) ---\n",
                    bw, bh, double(bw) * bh / 1e6);

        const std::vector<std::pair<std::string, std::vector<std::pair<std::string, double>>>>
            benched = {
                {"box_blur",              {{"radius", 5}}},
                {"gaussian_blur",         {{"sigma", 2.0}}},
                {"median_blur",           {{"radius", 3}}},
                {"bilateral",             {{"sigma_space", 3.0}}},
                {"anisotropic_diffusion", {{"iterations", 10}}},
                {"kuwahara",              {{"radius", 4}}},
                {"kuwahara_generalized",  {{"radius", 4}, {"sectors", 8}}},
                {"guided_filter",         {{"radius", 5}}},
                {"nonlocal_means",        {{"patch", 1}, {"search", 5}}},
                {"symmetric_nearest",     {{"radius", 4}}},
            };

        for (const auto& [name, params] : benched) {
            Image big;
            big.Alloc({bw, bh, Format::RGBA8});
            ImageView bv = big.MapCpuWrite();
            for (int y = 0; y < bh; ++y)
                for (int x = 0; x < bw; ++x) {
                    uint8_t* p = bv.At<uint8_t>(x, y);
                    p[0] = p[1] = p[2] = uint8_t((x * 7 + y * 13) % 256);
                    p[3] = 255;
                }

            Image out;
            const auto t0 = std::chrono::steady_clock::now();
            const bool ok = RunFilter(name, std::move(big), params, &out, &err);
            const double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
            std::printf("  %-24s %9.1f ms%s\n", name.c_str(), ms, ok ? "" : "  (FAILED)");
        }
    }

    std::printf("\n%s\n", g_fail == 0 ? "all filter checks passed" : "FAILURES PRESENT");
    return g_fail == 0 ? 0 : 1;
}
