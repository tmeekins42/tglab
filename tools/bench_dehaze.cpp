// Times dehaze on a real image, stage by stage.
//
// The algorithm has one stage that was quadratic in the radius and several that
// are linear in the pixel count, and at 45 MP the radius scales to ~98 -- so
// which one dominates is not obvious from reading the code, and was not what a
// first guess would have said.
//
//   bench_dehaze <image> [strength]

#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "../src/core/algorithm.h"
#include "../src/core/image.h"
#include "../src/algo_util/pixel_buffer.h"
#include "../src/core/image_io.h"
#include "../src/core/raw_io.h"
#include "../src/script/value.h"

using namespace tglab;

namespace {

double Ms(std::chrono::steady_clock::time_point a,
          std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

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

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: bench_dehaze <image> [strength]\n");
        return 2;
    }
    const double strength = (argc > 2) ? atof(argv[2]) : 0.8;

    // Raw FIRST, because LoadImageFile succeeds on a CR2 and hands back the
    // single-channel mosaic -- on which dehaze correctly takes its greyscale
    // early-out and reports 183 ms of doing nothing. A timing tool that
    // measures the wrong path is worse than none.
    Image src;
    std::string err;
    Image mosaic;
    if (LoadRawMosaic(argv[1], &mosaic, &err) && mosaic.Desc().IsMosaic()) {
        if (!Demosaic("demosaic_ahd", mosaic, &src)) {
            std::printf("demosaic failed\n");
            return 1;
        }
    } else if (!LoadImageFile(argv[1], &src, &err)) {
        std::printf("cannot load %s: %s\n", argv[1], err.c_str());
        return 1;
    }

    const ImageDesc d = src.Desc();
    std::printf("%dx%d (%.1f MP), strength %.2f\n\n",
                d.width, d.height, double(d.width) * double(d.height) / 1e6,
                strength);

    const char* algoName = std::getenv("BENCH_ALGO") ? std::getenv("BENCH_ALGO") : "dehaze";
    auto algo = Registry::Get().Create(algoName);
    if (!algo) { std::printf("dehaze not registered\n"); return 1; }
    if (ParamBase* p = algo->FindParam("strength")) {
        std::string e;
        p->SetFromScript(Value(strength), &e);
    }

    // Any parameter, by name, from the environment: BENCH_P_patch=1.
    //
    // Which stage dominates is decided by the two radii, so isolating a stage
    // means shrinking the others rather than instrumenting the algorithm.
    // Setting both radii to 1 leaves only the O(full pixels) work -- the
    // downsample and the recovery -- so the difference from a normal run is
    // what the map estimation actually costs. That is the number that decides
    // what is worth moving to the GPU, and guessing it from the code was
    // exactly the mistake the header of this file warns about.
    for (ParamBase* p : algo->Params()) {
        if (!p || !p->Name()) continue;
        const std::string key = std::string("BENCH_P_") + p->Name();
        if (const char* v = std::getenv(key.c_str())) {
            std::string e;
            if (p->SetFromScript(Value(std::atof(v)), &e))
                std::printf("  %s = %s\n", p->Name(), v);
            else
                std::printf("  %s: %s\n", p->Name(), e.c_str());
        }
    }

    std::vector<Data> ins;
    ins.push_back(Data{src.Clone()});
    std::vector<const Data*> inPtrs{&ins[0]};
    std::vector<Data> outs(1);
    Image out;
    out.Alloc(d);
    outs[0] = Data{std::move(out)};
    RunCtx ctx(inPtrs, outs);

    // Twice: the first run pays for page faults on freshly allocated buffers,
    // which is real cost the first time and noise every time after.
    const auto t0 = std::chrono::steady_clock::now();
    algo->RunCPU(ctx);
    const auto t1 = std::chrono::steady_clock::now();
    algo->RunCPU(ctx);
    const auto t2 = std::chrono::steady_clock::now();

    std::printf("  first run  %8.0f ms\n", Ms(t0, t1));
    std::printf("  warm run   %8.0f ms\n", Ms(t1, t2));
    std::printf("\n  %s\n", algo->RunReport().c_str());

    return 0;
}
