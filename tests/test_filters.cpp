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

int main() {
    std::string err;

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
