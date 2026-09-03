// Runs the real panorama pipeline on real frames, headlessly.
//
// The stitcher's failures are the kind that only appear at scale: a gain fit
// that is stable on a three-frame synthetic fixture can extrapolate wildly on
// fourteen frames whose overlaps are narrow strips. The unit fixture cannot
// show that, because three frames overlapping generously is the easy case.
//
// So this drives the same chain the script does -- detect, match, align, bundle,
// stitch -- over a directory of raws, and prints each stage's report. It is a
// measurement tool, not a test: it says what happened, and the operator decides
// whether that is right.
//
//   bench_stitch <raw> [raw...] [--gain N] [--wta]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../src/core/algorithm.h"
#include "../src/core/image.h"
#include "../src/core/pipeline.h"
#include "../src/algo_util/pixel_buffer.h"
#include "../src/core/raw_io.h"
#include "../src/core/shape.h"
#include "../src/script/value.h"

using namespace tglab;

namespace {

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

void SetParam(AlgorithmBase* a, const char* name, double v) {
    if (!a) return;
    if (ParamBase* pb = a->FindParam(name)) {
        std::string e;
        pb->SetFromScript(Value(v), &e);
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> files;
    int  gain = 0;
    bool wta  = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--gain") == 0 && i + 1 < argc) {
            gain = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--wta") == 0) {
            wta = true;
        } else {
            files.push_back(argv[i]);
        }
    }
    if (files.size() < 2) {
        std::printf("usage: bench_stitch <raw> [raw...] [--gain N] [--wta]\n");
        return 2;
    }

    std::printf("%d frames, gain_compensation=%d, winner_takes_all=%d\n\n",
                int(files.size()), gain, int(wta));

    ImageSet set;
    for (const std::string& f : files) {
        Image mosaic;
        std::string err;
        if (!LoadRawMosaic(f, &mosaic, &err)) {
            std::printf("  %s: %s\n", f.c_str(), err.c_str());
            return 1;
        }
        Image rgb;
        if (!Demosaic("demosaic_ahd", mosaic, &rgb)) {
            std::printf("  %s: demosaic failed\n", f.c_str());
            return 1;
        }
        set.images.push_back(std::move(rgb));
        std::printf("  loaded %s\n", f.c_str());
    }
    set.shape = Shape{{{"frame", int(set.images.size())}}};
    std::printf("\n");

    std::vector<Data> s;
    s.push_back(Data{std::move(set)});

    Pipeline p;
    p.AddStage(Registry::Get().Create("detect_orb"), "detect_orb", {{-1, 0}}, 1, 1);

    auto m = Registry::Get().Create("match_ann");
    SetParam(m.get(), "chain", 1.0);
    SetParam(m.get(), "window", 2.0);
    p.AddStage(std::move(m), "match_ann", {{0, 0}}, 1, 2);

    auto a = Registry::Get().Create("align_features");
    SetParam(a.get(), "model", 2.0);
    p.AddStage(std::move(a), "align_features", {{1, 0}}, 1, 3);

    p.AddStage(Registry::Get().Create("align"), "align", {{2, 0}}, 1, 4);
    p.AddStage(Registry::Get().Create("bundle_adjust"), "bundle_adjust",
               {{3, 0}}, 1, 5);

    auto st = Registry::Get().Create("stitch_panorama");
    SetParam(st.get(), "projection", 1.0);
    SetParam(st.get(), "gain_compensation", double(gain));
    SetParam(st.get(), "winner_takes_all", wta ? 1.0 : 0.0);
    p.AddStage(std::move(st), "stitch_panorama", {{4, 0}}, 1, 0);

    const auto t0 = std::chrono::steady_clock::now();
    std::string err;
    const bool ok = p.Execute(&s, nullptr, &err);
    const double ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();

    if (!ok) {
        std::printf("FAILED: %s\n", err.c_str());
        return 1;
    }

    for (const auto& stg : p.Stages()) {
        if (!stg.algo) continue;
        const std::string rep = stg.algo->RunReport();
        if (!rep.empty()) std::printf("  %s\n", rep.c_str());
    }

    // Discontinuity across the canvas: how much neighbouring COLUMNS differ.
    //
    // This is the number that matters for seams, and the band profile below is
    // not it -- a scene gradient moves the bands smoothly while a seam is a
    // STEP. Comparing each column to the one beside it measures the step and
    // ignores the gradient, so gain compensation should reduce this even when
    // the band profile barely moves.
    {
        const Data* d2 = p.Resolve({5, 0}, &s);
        if (const auto* im = d2 ? std::get_if<Image>(d2) : nullptr) {
            Image& mi = const_cast<Image&>(*im);
            ImageView v = mi.MapCpuRead();
            PixelBuffer pb;
            pb.Unpack(v);
            if (pb.Valid()) {
                std::vector<double> col(size_t(pb.Width()), 0.0);
                std::vector<long long> cnt(size_t(pb.Width()), 0);
                for (int y = 0; y < pb.Height(); y += 4)
                    for (int x = 0; x < pb.Width(); ++x) {
                        const float* q = pb.At(x, y);
                        const double y0 = (pb.Channels() >= 3)
                            ? 0.2126 * q[0] + 0.7152 * q[1] + 0.0722 * q[2]
                            : q[0];
                        if (y0 > 1e-6) { col[size_t(x)] += y0; ++cnt[size_t(x)]; }
                    }
                // The outermost columns are skipped: a canvas edge has partial
                // coverage, so its column mean is taken over a different set of
                // rows than its neighbour's and the difference is bookkeeping
                // rather than a seam. Measured, the largest "step" on this
                // panorama sat at x=11658 of 11660 for exactly that reason and
                // did not move when gain compensation was applied -- which made
                // the metric useless until it was excluded.
                const int margin = std::max(8, pb.Width() / 200);

                // Column steps, and how many rows each column actually covered.
                // A column whose coverage differs a lot from its neighbour's is
                // comparing different parts of the scene, so it is skipped too.
                double worst = 0.0;
                int wx = -1;
                double sum = 0.0;
                int n = 0;
                for (int x = margin; x < pb.Width() - margin; ++x) {
                    const long long ca = cnt[size_t(x)], cb = cnt[size_t(x - 1)];
                    if (!ca || !cb) continue;
                    if (std::labs(long(ca - cb)) > std::max<long>(4, long(ca) / 50))
                        continue;
                    const double a2 = col[size_t(x)]     / double(ca);
                    const double b2 = col[size_t(x - 1)] / double(cb);
                    if (a2 <= 0 || b2 <= 0) continue;
                    const double step = std::fabs(std::log(a2 / b2));
                    if (step > worst) { worst = step; wx = x; }
                    sum += step;
                    ++n;
                }
                std::printf("\n  column-to-column step: worst %.4f stops at x=%d,"
                            " mean %.5f (%d columns)\n",
                            worst / std::log(2.0), wx,
                            n ? (sum / double(n)) / std::log(2.0) : 0.0, n);
            }
        }
    }

    // Mean luminance in vertical bands across the finished canvas.
    //
    // The reports say what the stitcher INTENDED; this says what came out. A
    // gain that solves correctly and never reaches the pixels looks identical
    // in the report, so the two are worth checking separately -- and a band
    // profile also shows WHERE a correction landed, which a single number
    // cannot.
    const Data* outD = p.Resolve({5, 0}, &s);
    if (const auto* im = outD ? std::get_if<Image>(outD) : nullptr) {
        Image& mi = const_cast<Image&>(*im);
        ImageView v = mi.MapCpuRead();
        PixelBuffer pb;
        pb.Unpack(v);
        if (pb.Valid()) {
            const int nb = 12;
            std::printf("\n  luminance by band (left to right):\n   ");
            for (int b = 0; b < nb; ++b) {
                const int x0 = pb.Width() * b / nb;
                const int x1 = pb.Width() * (b + 1) / nb;
                double sum = 0.0;
                long long n = 0;
                for (int y = 0; y < pb.Height(); y += 8)
                    for (int x = x0; x < x1; x += 8) {
                        const float* q = pb.At(x, y);
                        const double y0 = (pb.Channels() >= 3)
                            ? 0.2126 * q[0] + 0.7152 * q[1] + 0.0722 * q[2]
                            : q[0];
                        if (y0 > 1e-6) { sum += y0; ++n; }
                    }
                std::printf(" %6.4f", n ? sum / double(n) : 0.0);
            }
            std::printf("\n");
        }
    }

    std::printf("\n  %.1f s\n", ms / 1000.0);
    return 0;
}
