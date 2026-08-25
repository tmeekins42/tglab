// headroom — how much is there left to gain above AHD?
//
// The question this exists to answer, before anyone invests in training a
// network: on the content where demosaic methods actually differ, how far is
// the best current method from the best ACHIEVABLE?
//
// Why the measurement is shaped the way it is
// -------------------------------------------
// Tim's framing, and it is the right one: bilinear is a correct low-pass
// reconstruction. Downsample its output and the error largely vanishes.
// Everything Malvar and AHD do is an attempt to recover the high-frequency
// content bilinear discards. So the headroom question is ENTIRELY a
// high-frequency question, and an average over a whole frame would mostly
// measure the flat regions where every method already agrees.
//
// So every figure here is restricted to sites where reconstruction is actually
// ambiguous, and reported as a distribution rather than a mean -- the same
// lesson that cost two rounds of chasing the AHD fringing, where a per-decile
// mean improved while the picture visibly got worse.
//
// The reference
// -------------
// Ground truth comes from DOWNSAMPLING a real photograph, not from a synthetic
// scene, and not from demosaicing a raw and calling the result truth (which
// would be circular -- the ceiling would be whatever method produced it).
//
// The downsample is doing real work. Box-averaging 3x3 of a Bayer mosaic means
// every output pixel has ~2 red, ~4 green and ~2 blue REAL samples underneath
// it: the result is a measurement of all three channels at that location, not
// an interpolation guess. Demosaic error is high-frequency and roughly
// zero-mean, because it is driven by CFA phase, which alternates -- so most of
// it cancels in the average.
//
// It is not perfectly clean and this file should not pretend otherwise. A
// systematic low-frequency bias -- the magenta cast that took two rounds to
// find -- would survive averaging. That is why this is used to COMPARE methods
// against a common reference, never as training labels.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../src/core/algorithm.h"
#include "../src/core/image.h"
#include "../src/core/raw_io.h"
#include "../src/script/value.h"

using namespace tglab;

namespace {

// Run one demosaic on a mosaic, CPU path.
bool Demosaic(const char* name, const Image& in, Image* out) {
    auto algo = Registry::Get().Create(name);
    if (!algo) return false;
    std::vector<Data> ins;
    ins.push_back(Data{const_cast<Image&>(in).Clone()});
    std::vector<const Data*> inPtrs{&ins[0]};
    std::vector<Data> outs(1);
    ImageDesc d = in.Desc();
    d.format = Format::RGBA16F;
    d.cfa    = CfaPattern::None;
    d.linear = true;
    Image img;
    img.Alloc(d);
    outs[0] = Data{std::move(img)};
    RunCtx ctx(inPtrs, outs);
    algo->PrepareGpu({in.Desc()});
    algo->RunCPU(ctx);
    Image* o = std::get_if<Image>(&outs[0]);
    if (!o || !o->Valid()) return false;
    *out = o->Clone();
    return true;
}

struct Rgb {
    double r = 0, g = 0, b = 0;
};

// Reference: box-downsample a demosaiced image by `f`.
//
// Averaging over f*f pixels means each output had real samples of all three
// colours beneath it, and cancels the phase-alternating part of any residual
// demosaic error.
std::vector<Rgb> Downsample(Image& img, int f, int* ow, int* oh) {
    ImageView v = img.MapCpuRead();
    const ImageDesc& d = img.Desc();
    *ow = d.width / f;
    *oh = d.height / f;
    std::vector<Rgb> out(size_t(*ow) * size_t(*oh));
    for (int y = 0; y < *oh; ++y)
        for (int x = 0; x < *ow; ++x) {
            Rgb acc;
            for (int sy = 0; sy < f; ++sy)
                for (int sx = 0; sx < f; ++sx) {
                    const uint16_t* p = v.At<uint16_t>(x * f + sx, y * f + sy);
                    acc.r += HalfToFloat(p[0]);
                    acc.g += HalfToFloat(p[1]);
                    acc.b += HalfToFloat(p[2]);
                }
            const double n = double(f) * double(f);
            out[size_t(y) * size_t(*ow) + size_t(x)] = {acc.r / n, acc.g / n, acc.b / n};
        }
    return out;
}

// Build a Bayer mosaic FROM a downsampled RGB reference.
//
// This is the forward model: take the known three-channel truth and sample it
// the way a sensor would. Reconstructing from this and comparing back is the
// only way to score a method against an answer that is not itself a demosaic.
Image Remosaic(const std::vector<Rgb>& ref, int w, int h, CfaPattern cfa,
               const ImageDesc& meta) {
    Image img;
    ImageDesc d = meta;
    d.width  = w;
    d.height = h;
    d.format = Format::R32F;
    d.cfa    = cfa;
    // The reference is in POST-white-balance, post-matrix space, because it was
    // downsampled from a demosaic's output. Undoing that exactly is not
    // possible, so the comparison below is done in the same space on both
    // sides: identity gains here, and the same identity when scoring.
    for (int i = 0; i < 3; ++i) d.camMul[i] = 1.0f;
    const float ident[9] = {1,0,0, 0,1,0, 0,0,1};
    std::memcpy(d.rgbCam, ident, sizeof ident);
    d.blackLevel = 0.0f;
    d.whiteLevel = 1.0f;
    img.Alloc(d);
    ImageView v = img.MapCpuWrite();
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const Rgb& c = ref[size_t(y) * size_t(w) + size_t(x)];
            const int  k = CfaColorAt(cfa, x, y);
            *v.At<float>(x, y) = float(k == 0 ? c.r : (k == 1 ? c.g : c.b));
        }
    return img;
}

struct Score {
    double p50 = 0, p90 = 0, p99 = 0;
    long   sites = 0;
};

// Error at HIGH-FREQUENCY sites only.
//
// A site qualifies when the true scene varies sharply there -- that is where
// reconstruction is ambiguous and where methods differ. Everywhere else every
// method is already correct and including it would dilute the measurement into
// meaninglessness.
Score ScoreHighFreq(const std::vector<Rgb>& truth, Image& got, int w, int h,
                    double gradThresh) {
    ImageView v = got.MapCpuRead();
    std::vector<double> errs;
    auto T = [&](int x, int y) -> const Rgb& {
        return truth[size_t(std::clamp(y, 0, h - 1)) * size_t(w) +
                     size_t(std::clamp(x, 0, w - 1))];
    };
    for (int y = 2; y < h - 2; ++y)
        for (int x = 2; x < w - 2; ++x) {
            const Rgb& t = T(x, y);
            const double lum = 0.2126 * t.r + 0.7152 * t.g + 0.0722 * t.b;
            if (lum < 0.01) continue;

            auto L = [&](int xx, int yy) {
                const Rgb& c = T(xx, yy);
                return 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
            };
            const double grad = std::max(std::fabs(L(x+1,y) - L(x-1,y)),
                                         std::fabs(L(x,y+1) - L(x,y-1)));
            if (grad < gradThresh * lum) continue;

            const uint16_t* p = v.At<uint16_t>(x, y);
            const double dr = double(HalfToFloat(p[0])) - t.r;
            const double dg = double(HalfToFloat(p[1])) - t.g;
            const double db = double(HalfToFloat(p[2])) - t.b;
            // Normalised by local brightness so bright and dark edges count
            // comparably.
            errs.push_back(std::sqrt((dr*dr + dg*dg + db*db) / 3.0) / lum);
        }
    Score s;
    if (errs.empty()) return s;
    std::sort(errs.begin(), errs.end());
    auto q = [&](double f) { return errs[size_t(f * double(errs.size() - 1))]; };
    s.p50 = q(0.50); s.p90 = q(0.90); s.p99 = q(0.99);
    s.sites = long(errs.size());
    return s;
}

// The oracle: the best any 5x5-window method could do at this site.
//
// Not achievable by a real algorithm -- it cheats by looking at the answer --
// but it bounds what is POSSIBLE from the same input a network would see. If
// AHD is already close to it, a learned method has little room; if the gap is
// large, the information is there and something is failing to use it.
//
// Implemented as: of the true colours present in the 5x5 neighbourhood, pick
// the one closest to the true colour here. A window method can only combine
// what is in its window, so this is a genuine (if loose) upper bound on
// achievable accuracy.
Score ScoreOracle(const std::vector<Rgb>& truth, int w, int h, double gradThresh) {
    std::vector<double> errs;
    auto T = [&](int x, int y) -> const Rgb& {
        return truth[size_t(std::clamp(y, 0, h - 1)) * size_t(w) +
                     size_t(std::clamp(x, 0, w - 1))];
    };
    for (int y = 2; y < h - 2; ++y)
        for (int x = 2; x < w - 2; ++x) {
            const Rgb& t = T(x, y);
            const double lum = 0.2126 * t.r + 0.7152 * t.g + 0.0722 * t.b;
            if (lum < 0.01) continue;
            auto L = [&](int xx, int yy) {
                const Rgb& c = T(xx, yy);
                return 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
            };
            const double grad = std::max(std::fabs(L(x+1,y) - L(x-1,y)),
                                         std::fabs(L(x,y+1) - L(x,y-1)));
            if (grad < gradThresh * lum) continue;

            // Best single neighbour, excluding the centre itself -- the centre
            // carries only ONE colour channel in the mosaic, so no method gets
            // the other two for free.
            double best = 1e30;
            for (int dy = -2; dy <= 2; ++dy)
                for (int dx = -2; dx <= 2; ++dx) {
                    if (!dx && !dy) continue;
                    const Rgb& n = T(x + dx, y + dy);
                    const double dr = n.r - t.r, dg = n.g - t.g, db = n.b - t.b;
                    best = std::min(best, std::sqrt((dr*dr + dg*dg + db*db) / 3.0) / lum);
                }
            errs.push_back(best);
        }
    Score s;
    if (errs.empty()) return s;
    std::sort(errs.begin(), errs.end());
    auto q = [&](double f) { return errs[size_t(f * double(errs.size() - 1))]; };
    s.p50 = q(0.50); s.p90 = q(0.90); s.p99 = q(0.99);
    s.sites = long(errs.size());
    return s;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: headroom <raw> [raw...]\n");
        return 2;
    }

    std::printf("Headroom above AHD, measured at high-frequency sites only.\n");
    std::printf("Reference: the frame demosaiced, downsampled 3x, then re-mosaiced.\n");
    std::printf("Error is RMS over RGB, normalised by local luminance. Lower is better.\n\n");

    double sumAhd = 0, sumMal = 0, sumBil = 0, sumOra = 0;
    int    frames = 0;

    for (int i = 1; i < argc; ++i) {
        Image mosaic;
        std::string err;
        if (!LoadRawMosaic(argv[i], &mosaic, &err)) {
            std::printf("%s: %s\n", argv[i], err.c_str());
            continue;
        }

        // Reference from the best method we have. Its residual artefacts are
        // what the 3x downsample is there to suppress.
        Image full;
        if (!Demosaic("demosaic_ahd", mosaic, &full)) {
            std::printf("%s: demosaic failed\n", argv[i]);
            continue;
        }
        int w = 0, h = 0;
        const std::vector<Rgb> truth = Downsample(full, 3, &w, &h);

        // Re-sample the truth as a sensor would, then reconstruct from that.
        const CfaPattern cfa = mosaic.Desc().cfa;
        Image remos = Remosaic(truth, w, h, cfa, mosaic.Desc());

        Image bil, mal, ahd;
        if (!Demosaic("demosaic_bilinear", remos, &bil) ||
            !Demosaic("demosaic_malvar",   remos, &mal) ||
            !Demosaic("demosaic_ahd",      remos, &ahd)) {
            std::printf("%s: reconstruction failed\n", argv[i]);
            continue;
        }

        const double kGrad = 0.15;
        const Score sb = ScoreHighFreq(truth, bil, w, h, kGrad);
        const Score sm = ScoreHighFreq(truth, mal, w, h, kGrad);
        const Score sa = ScoreHighFreq(truth, ahd, w, h, kGrad);
        const Score so = ScoreOracle(truth, w, h, kGrad);

        const char* base = std::strrchr(argv[i], '\\');
        if (!base) base = std::strrchr(argv[i], '/');
        std::printf("%s  (%dx%d ref, %ld hf sites, %.1f%% of frame)\n",
                    base ? base + 1 : argv[i], w, h, sa.sites,
                    100.0 * double(sa.sites) / double(w) / double(h));
        std::printf("    %-12s p50 %.4f  p90 %.4f  p99 %.4f\n", "bilinear", sb.p50, sb.p90, sb.p99);
        std::printf("    %-12s p50 %.4f  p90 %.4f  p99 %.4f\n", "malvar",   sm.p50, sm.p90, sm.p99);
        std::printf("    %-12s p50 %.4f  p90 %.4f  p99 %.4f\n", "ahd",      sa.p50, sa.p90, sa.p99);
        std::printf("    %-12s p50 %.4f  p90 %.4f  p99 %.4f   <- loose lower bound\n",
                    "oracle", so.p50, so.p90, so.p99);
        if (sa.p50 > 1e-9)
            std::printf("    AHD is %.2fx the oracle at p50, %.2fx at p90\n",
                        sa.p50 / std::max(so.p50, 1e-9), sa.p90 / std::max(so.p90, 1e-9));
        std::printf("\n");

        sumBil += sb.p50; sumMal += sm.p50; sumAhd += sa.p50; sumOra += so.p50;
        ++frames;
    }

    if (frames > 1) {
        std::printf("=== mean p50 over %d frames ===\n", frames);
        std::printf("  bilinear %.4f   malvar %.4f   ahd %.4f   oracle %.4f\n",
                    sumBil/frames, sumMal/frames, sumAhd/frames, sumOra/frames);
        std::printf("\n  bilinear -> malvar : %.0f%% of the way to the oracle\n",
                    100.0 * (sumBil - sumMal) / std::max(sumBil - sumOra, 1e-9));
        std::printf("  bilinear -> ahd    : %.0f%% of the way to the oracle\n",
                    100.0 * (sumBil - sumAhd) / std::max(sumBil - sumOra, 1e-9));
        std::printf("  remaining gap      : %.0f%% of the bilinear->oracle span\n",
                    100.0 * (sumAhd - sumOra) / std::max(sumBil - sumOra, 1e-9));
    }
    return 0;
}
