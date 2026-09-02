// Where each feature detector actually spends its time.
//
// The SIFT conversion paid because the blur pyramid was 65% of its runtime.
// That is a fact about SIFT, not about detectors: ORB's pyramid is a cheap box
// average and its cost is the FAST+Harris scan; AKAZE's diffusion is the whole
// algorithm; BRISK sits somewhere between. Converting the wrong stage is work
// that buys nothing, so measure before writing a kernel.
//
// This runs each detector twice -- once whole, once with its front end only --
// and reports the split. The front end is whatever the detector builds before
// it starts testing pixels: a pyramid, a blur stack, a diffusion scale space.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <d3d12.h>

#include "../src/algo_util/features.h"
#include "../src/core/algorithm.h"
#include "../src/core/pipeline.h"
#include "../src/gpu/compute.h"

using namespace tglab;

namespace {

// A scene with real structure at many scales. Flat noise would make every
// detector find nothing and time the empty case; a checkerboard would put every
// feature in one octave.
Image MakeScene(int w, int h) {
    Image img;
    img.Alloc({w, h, Format::RGBA8});
    ImageView v = img.MapCpuWrite();

    uint32_t s = 0x1234567u;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            uint8_t* p = v.At<uint8_t>(x, y);
            const uint8_t n = uint8_t(20 + (s & 31));
            p[0] = n; p[1] = n; p[2] = n; p[3] = 255;
        }

    const int count = (w * h) / 20000;
    for (int i = 0; i < count; ++i) {
        const int r  = 2 + (i % 12) * 2;
        const int cx = 24 + (i * 53) % std::max(1, w - 48);
        const int cy = 24 + (i * 97) % std::max(1, h - 48);
        const uint8_t tone = (i % 2) ? 245 : 10;
        for (int y = cy - r; y <= cy + r; ++y)
            for (int x = cx - r; x <= cx + r; ++x) {
                if (x < 0 || y < 0 || x >= w || y >= h) continue;
                if ((x - cx) * (x - cx) + (y - cy) * (y - cy) > r * r) continue;
                uint8_t* p = v.At<uint8_t>(x, y);
                p[0] = tone; p[1] = tone; p[2] = tone;
            }
    }
    return img;
}

double Ms(std::chrono::steady_clock::time_point a,
          std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

}  // namespace

int main(int argc, char** argv) {
    const int w = (argc > 1) ? std::atoi(argv[1]) : 2400;
    const int h = (argc > 2) ? std::atoi(argv[2]) : 1600;

    std::printf("feature detectors at %dx%d (%.1f MP)\n\n",
                w, h, double(w) * double(h) / 1e6);

    ID3D12Device* dev = nullptr;
    ComputeContext gpu;
    bool haveGpu = false;
    if (SUCCEEDED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
                                    IID_PPV_ARGS(&dev)))) {
        haveGpu = gpu.Init(dev);
    }
    std::printf("%s\n\n", haveGpu ? "device present" : "no device: CPU only");

    const char* kDetectors[] = {
        "detect_sift", "detect_orb", "detect_brisk", "detect_akaze",
    };

    std::printf("  %-16s %10s %10s %8s %10s\n",
                "detector", "CPU ms", "GPU ms", "speedup", "features");

    for (const char* name : kDetectors) {
        auto run = [&](ExecMode mode, double* ms, int* found) -> bool {
            std::vector<Data> sources;
            sources.push_back(Data{MakeScene(w, h)});

            Pipeline pipe;
            pipe.AddStage(Registry::Get().Create(name), name, {{-1, 0}}, 1, 1);

            std::string err;
            const auto t0 = std::chrono::steady_clock::now();
            const bool ok = pipe.Execute(&sources, nullptr, &err,
                                         haveGpu ? &gpu : nullptr, mode);
            *ms = Ms(t0, std::chrono::steady_clock::now());
            if (!ok) { std::printf("  %-16s FAILED: %s\n", name, err.c_str()); return false; }

            const Data* d = pipe.Resolve({0, 0}, &sources);
            *found = 0;
            if (d && std::holds_alternative<Image>(*d)) {
                const Image& im = std::get<Image>(*d);
                if (const FeatureSidecar* sc =
                        im.Sidecars().Get<FeatureSidecar>(kFeatureSidecar))
                    *found = int(sc->keypoints.size());
            }
            return true;
        };

        double cpuMs = 0.0, gpuMs = 0.0;
        int cpuN = 0, gpuN = 0;
        if (!run(ExecMode::ForceCPU, &cpuMs, &cpuN)) continue;

        if (haveGpu) {
            if (!run(ExecMode::Auto, &gpuMs, &gpuN)) continue;
            std::printf("  %-16s %10.0f %10.0f %7.1fx %10d%s\n",
                        name, cpuMs, gpuMs,
                        gpuMs > 0.0 ? cpuMs / gpuMs : 0.0, cpuN,
                        cpuN == gpuN ? "" : "   MISMATCH");
        } else {
            std::printf("  %-16s %10.0f %10s %8s %10d\n",
                        name, cpuMs, "-", "-", cpuN);
        }
    }

    if (dev) dev->Release();
    return 0;
}
