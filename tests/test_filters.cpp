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
#include <string>
#include <vector>

#include "../src/algo_util/pixel_buffer.h"
#include "../src/core/algorithm.h"
#include "../src/core/pipeline.h"
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

int main() {
    std::string err;
    TestKelvinWhiteBalance();


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
