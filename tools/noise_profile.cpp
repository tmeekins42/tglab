// noise_profile — how noise varies with brightness, on the DEVELOPED image.
//
// wavelet_denoise uses one threshold for the whole frame. Whether that is
// right depends on how flat the noise is across the brightness range, and the
// answer is not the textbook one.
//
// On the raw mosaic noise is shot-limited, so sigma grows as sqrt(level) and a
// variance-stabilising transform (Anscombe) is the standard fix. But
// wavelet_denoise does not see the mosaic -- it runs after develop, and the
// tone curve has already done most of the stabilising. Measuring where the
// algorithm actually runs is the whole point of this tool.
//
// METHOD
// ------
// Develop the raw exactly as the app would, bin pixels by their local level,
// and measure sigma within each bin.
//
// Sigma is measured as the MEDIAN ABSOLUTE DEVIATION of the high-pass residual,
// not the standard deviation of the pixels:
//
//   - the residual (pixel minus a small blur) removes real structure, so what
//     is left is noise plus the edges the blur could not follow;
//   - the MAD then discards those edges, since they are a small minority of
//     samples and a median ignores outliers. A standard deviation would be
//     dominated by them and would report "noise" that is really detail.
//
// Scaled by 1.4826 so the figure is comparable to a Gaussian sigma.
//
// Bins are by the BLURRED value rather than the pixel's own, because a noisy
// pixel would otherwise sort itself into a brighter bin than the region it
// belongs to -- which biases exactly the measurement being made.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "../src/algo_util/pixel_buffer.h"
#include "../src/core/algorithm.h"
#include "../src/core/image.h"
#include "../src/core/pipeline.h"
#include "../src/core/raw_io.h"
#include "../src/script/interp.h"
#include "../src/script/parser.h"

using namespace tglab;

namespace {

// Develops a raw through the ordinary path: hot-pixel repair, demosaic, and
// basic_adjust with auto-exposure -- which is what the app does, and therefore
// what wavelet_denoise sees.
bool Develop(const std::string& path, Image* out, std::string* err,
             double levelDep = -1.0) {
    Image mosaic;
    if (!LoadRawMosaic(path, &mosaic, err)) return false;

    std::vector<Data> sources;
    sources.push_back(Data{std::move(mosaic)});

    SourceImage si;
    si.name     = "src";
    si.index    = 0;
    si.isMosaic = true;
    std::vector<SourceImage> names{si};

    // levelDep < 0 means "no denoise", which is the profile of the input.
    // Otherwise the same develop followed by wavelet_denoise at that setting,
    // so the two profiles differ only by the thing being measured.
    char script[512];
    if (levelDep < 0.0) {
        std::snprintf(script, sizeof script,
                      "img = image(\"src\")\n"
                      "out = params(basic_adjust, auto_exposure = 1)(img)\n"
                      "display(out)\n");
    } else {
        std::snprintf(script, sizeof script,
                      "img = image(\"src\")\n"
                      "d = params(basic_adjust, auto_exposure = 1)(img)\n"
                      "out = wavelet_denoise(d, level_dep = %.3f)\n"
                      "display(out)\n", levelDep);
    }

    Program prog;
    if (!Parse(script, &prog, err)) return false;

    UiState ui;
    Pipeline pipe;
    const InterpResult r = Interpret(prog, names, &ui, &pipe);
    if (!r.ok) { *err = r.error; return false; }

    if (!pipe.Execute(&sources, nullptr, err)) return false;

    const Data* d = pipe.Resolve(pipe.Viewers().back().source, &sources);
    const auto* im = d ? std::get_if<Image>(d) : nullptr;
    if (!im) { *err = "no image produced"; return false; }
    *out = const_cast<Image&>(*im).Clone();
    return true;
}

struct Band {
    double level = 0.0;    // mean local level in this band
    double sigma = 0.0;    // MAD-based noise estimate
    size_t n     = 0;
};

// Sigma per brightness band, from the high-pass residual.
std::vector<Band> Profile(Image& img, int bands) {
    ImageView v = img.MapCpuRead();
    PixelBuffer pb;
    pb.Unpack(v);
    if (!pb.Valid()) return {};

    const int w = pb.Width(), h = pb.Height(), ch = pb.Channels();
    const std::vector<float>& px = pb.Data();

    // Luma, and a 3x3 box blur of it. The blur is the local level; the
    // difference is the residual the MAD is taken over.
    std::vector<float> lum(size_t(w) * size_t(h));
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const float* p = &px[(size_t(y) * size_t(w) + size_t(x)) * size_t(ch)];
            lum[size_t(y) * size_t(w) + size_t(x)] =
                (ch >= 3) ? 0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2] : p[0];
        }

    // Collect (level, |residual|) pairs on a stride: a few hundred thousand
    // samples pin these statistics down and a full pass costs far more.
    struct Sample { float level, absdev; };
    std::vector<Sample> s;
    s.reserve(500000);
    const int stride = std::max(1, std::min(w, h) / 900);

    for (int y = 1 + stride; y < h - 1; y += stride)
        for (int x = 1 + stride; x < w - 1; x += stride) {
            float sum = 0.0f;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx)
                    sum += lum[size_t(y + dy) * size_t(w) + size_t(x + dx)];
            const float local = sum / 9.0f;
            const float me    = lum[size_t(y) * size_t(w) + size_t(x)];
            s.push_back({local, std::fabs(me - local)});
        }

    if (s.size() < 5000) return {};

    // Bin by level, over the 1st..99th percentile so a handful of black or
    // blown pixels do not stretch the range and empty most of the bins.
    std::vector<float> levels;
    levels.reserve(s.size());
    for (const Sample& x : s) levels.push_back(x.level);
    std::sort(levels.begin(), levels.end());
    const float lo = levels[size_t(0.01 * double(levels.size()))];
    const float hi = levels[size_t(0.99 * double(levels.size()))];
    if (!(hi > lo)) return {};

    // Sized with a named count, not `buckets(size_t(bands))` -- that form is a
    // function declaration to the compiler (the most vexing parse), and the
    // errors it produces point at the USES rather than at the declaration.
    const size_t nb = size_t(bands);
    std::vector<std::vector<float>> buckets;
    buckets.resize(nb);
    std::vector<double> levelSum(nb, 0.0);
    for (const Sample& x : s) {
        if (x.level < lo || x.level > hi) continue;
        int b = int(double(bands) * double(x.level - lo) / double(hi - lo));
        b = std::clamp(b, 0, bands - 1);
        buckets[size_t(b)].push_back(x.absdev);
        levelSum[size_t(b)] += double(x.level);
    }

    std::vector<Band> out;
    for (int b = 0; b < bands; ++b) {
        std::vector<float>& v2 = buckets[size_t(b)];
        if (v2.size() < 200) continue;    // too few to trust
        std::nth_element(v2.begin(), v2.begin() + v2.size() / 2, v2.end());
        Band bd;
        bd.n     = v2.size();
        bd.level = levelSum[size_t(b)] / double(v2.size());
        // 1.4826 * MAD estimates a Gaussian sigma. The residual of a 3x3 box
        // blur has 8/9 of the pixel's own variance, so scale back up to report
        // the noise in the PIXEL rather than in the residual.
        bd.sigma = 1.4826 * double(v2[v2.size() / 2]) * std::sqrt(9.0 / 8.0);
        out.push_back(bd);
    }
    return out;
}

// How far sigma varies across the range: the number that says whether one
// fixed threshold can be right.
double Flatness(const std::vector<Band>& b) {
    if (b.size() < 2) return 1.0;
    double lo = 1e30, hi = 0.0;
    for (const Band& x : b) { lo = std::min(lo, x.sigma); hi = std::max(hi, x.sigma); }
    return (lo > 0.0) ? hi / lo : 0.0;
}

} // namespace

// Does the estimator report the sigma it is given?
//
// Run before trusting any figure below it. A MAD over a high-pass residual is
// an indirect measurement with two scale factors in it -- the 1.4826 and the
// residual's 8/9 of the pixel variance -- and getting either wrong would shift
// every number without changing their shape, so a bad estimator still produces
// a plausible-looking table.
//
// Also the honest place to see the failure mode: on a ramp, sigma should be
// FLAT because the noise is uniform. Whatever it reports there is the floor of
// what this tool can distinguish from real structure.
bool SelfTest() {
    constexpr int kW = 700, kH = 700;
    constexpr double kSigma = 0.010;

    Image img;
    ImageDesc d{kW, kH, Format::RGBA32F};
    d.linear = true;
    img.Alloc(d);
    ImageView v = img.MapCpuWrite();

    // Deterministic, so the check means the same thing every run.
    uint32_t st = 12345u;
    auto rnd = [&] {
        st ^= st << 13; st ^= st >> 17; st ^= st << 5;
        return double(st) / 4294967296.0;
    };
    auto gauss = [&] {
        // Box-Muller.
        const double u1 = std::max(rnd(), 1e-9), u2 = rnd();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307 * u2);
    };

    for (int y = 0; y < kH; ++y)
        for (int x = 0; x < kW; ++x) {
            // A gentle ramp, so the bins span a real brightness range, plus
            // UNIFORM noise of known sigma. A correct estimator reports kSigma
            // in every band regardless of level.
            const double base = 0.05 + 0.55 * double(x) / double(kW);
            float* p = v.At<float>(x, y);
            for (int c = 0; c < 3; ++c) p[c] = float(base + kSigma * gauss());
            p[3] = 1.0f;
        }

    const std::vector<Band> b = Profile(img, 6);
    if (b.size() < 4) { std::printf("self-test: too few bands\n"); return false; }

    double lo = 1e30, hi = 0.0;
    for (const Band& x : b) { lo = std::min(lo, x.sigma); hi = std::max(hi, x.sigma); }
    const double mid = 0.5 * (lo + hi);

    // Luma averages the noise DOWN, and by a specific amount.
    //
    // The three channels carry the same signal here but INDEPENDENT noise, so
    // the luma of them has sigma * sqrt(0.2126^2 + 0.7152^2 + 0.0722^2) =
    // 0.75 sigma, not sigma. The first version of this check ignored that,
    // reported "25% off", and sent me looking for a scale-factor bug in the
    // estimator that was not there. The estimator was right and the
    // EXPECTATION was wrong -- which is exactly the mistake a self-test exists
    // to make visible, so it is worth keeping the reason written down.
    //
    // This applies to the real frames too: every sigma this tool reports is a
    // luma sigma. That is the right quantity, since wavelet_denoise thresholds
    // a luma/chroma rotation rather than raw channels.
    constexpr double kLumaNoise = 0.7496;   // sqrt(0.2126^2 + 0.7152^2 + 0.0722^2)
    const double expect = kSigma * kLumaNoise;
    const double err = std::fabs(mid - expect) / expect;
    const double flat = hi / lo;

    std::printf("self-test: injected %.4f per channel, expect %.4f in luma, "
                "measured %.4f..%.4f (%.0f%% off, %.2fx spread)\n",
                kSigma, expect, lo, hi, 100.0 * err, flat);
    const bool ok = err < 0.15 && flat < 1.25;
    std::printf("self-test: %s\n\n", ok ? "PASS" : "FAIL -- figures below are not trustworthy");
    return ok;
}

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--self-test") { return SelfTest() ? 0 : 1; }
    SelfTest();

    if (argc < 2) {
        std::printf("usage: noise_profile <raw>...\n\n"
                    "Reports sigma across the brightness range of the DEVELOPED\n"
                    "image, and how flat it is. A flatness near 1 means one fixed\n"
                    "threshold is right; a large number means it cannot be.\n");
        return 1;
    }

    for (int i = 1; i < argc; ++i) {
        Image dev;
        std::string err;
        if (!Develop(argv[i], &dev, &err)) {
            std::printf("%s: %s\n", argv[i], err.c_str());
            continue;
        }

        const std::vector<Band> bands = Profile(dev, 8);
        if (bands.empty()) { std::printf("%s: too few samples\n", argv[i]); continue; }

        const char* name = argv[i];
        if (const char* slash = std::strrchr(name, '/'))  name = slash + 1;
        if (const char* slash = std::strrchr(name, '\\')) name = slash + 1;

        std::printf("\n%s\n", name);
        std::printf("  %-10s %-10s %s\n", "level", "sigma", "samples");
        for (const Band& b : bands)
            std::printf("  %-10.4f %-10.4f %zu\n", b.level, b.sigma, b.n);

        double lo = 1e30, hi = 0.0;
        for (const Band& b : bands) { lo = std::min(lo, b.sigma); hi = std::max(hi, b.sigma); }
        std::printf("  sigma %.4f .. %.4f  = %.1fx\n", lo, hi, Flatness(bands));

        // Which exponent would flatten that spread. sigma of L^e is
        // sigma * e * L^(e-1), so this is what a variance-stabilising transform
        // would leave behind. Measured rather than reasoned: the todo note
        // argued for L^0.75 on the grounds that develop had already stabilised
        // the noise, and the numbers say sqrt.
        for (double e : {0.5, 0.75, 1.0}) {
            double l2 = 1e30, h2 = 0.0;
            for (const Band& b : bands) {
                const double s2 = b.sigma * e * std::pow(std::max(b.level, 1e-6), e - 1.0);
                l2 = std::min(l2, s2);
                h2 = std::max(h2, s2);
            }
            std::printf("  after L^%.2f: %.1fx\n", e, (l2 > 0.0) ? h2 / l2 : 0.0);
        }

        // What wavelet_denoise actually leaves, with the level-dependent
        // threshold off and on. This is the question the parameter exists to
        // answer, and it is a different one from "which exponent flattens
        // sigma" -- a transform that flattens the profile has not necessarily
        // denoised anything.
        for (double ld : {0.0, 1.0}) {
            Image den;
            std::string e2;
            if (!Develop(argv[i], &den, &e2, ld)) continue;
            const std::vector<Band> db = Profile(den, 8);
            if (db.empty()) continue;

            double l3 = 1e30, h3 = 0.0, sum = 0.0;
            for (const Band& b : db) {
                l3 = std::min(l3, b.sigma);
                h3 = std::max(h3, b.sigma);
                sum += b.sigma;
            }
            std::printf("  denoised level_dep=%.1f: sigma %.4f..%.4f = %.1fx, mean %.4f\n",
                        ld, l3, h3, (l3 > 0.0) ? h3 / l3 : 0.0, sum / double(db.size()));
        }
    }
    return 0;
}
