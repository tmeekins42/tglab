// Times the SIFT scale-space pyramid, CPU against GPU.
//
// The pyramid is 65% of SIFT's runtime at 22 MP (5,626 ms of 8,646 ms), which
// is why it was the first thing moved to the device. This measures whether that
// actually paid, at a size worth caring about -- a 512x512 benchmark would be
// dominated by dispatch overhead and would say nothing about the case that
// motivated the work.
//
// It times the pyramid ALONE, not the whole detector. Extrema detection,
// orientation, and descriptors all stay on the CPU, so quoting a detector-level
// speedup here would mix the part that changed with the part that did not.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <d3d12.h>

#include "../src/algorithms/features/gpu_pyramid.h"
#include "../src/gpu/compute.h"

using namespace tglab;

namespace {

struct Plane {
    std::vector<float> v;
    int w = 0, h = 0;

    float At(int x, int y) const {
        return v[size_t(std::clamp(y, 0, h - 1)) * size_t(w) +
                 size_t(std::clamp(x, 0, w - 1))];
    }
};

void CpuBlur(const Plane& src, Plane& dst, float sigma, std::vector<float>& tmp) {
    const int w = src.w, h = src.h;
    const int r = std::max(1, int(std::ceil(sigma * 3.0f)));

    std::vector<float> k(size_t(r) * 2 + 1);
    float sum = 0.0f;
    for (int i = -r; i <= r; ++i) {
        const float e = std::exp(-float(i * i) / (2.0f * sigma * sigma));
        k[size_t(i + r)] = e;
        sum += e;
    }
    for (float& x : k) x /= sum;

    tmp.assign(size_t(w) * size_t(h), 0.0f);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float a = 0.0f;
            for (int i = -r; i <= r; ++i) a += k[size_t(i + r)] * src.At(x + i, y);
            tmp[size_t(y) * size_t(w) + size_t(x)] = a;
        }

    dst.w = w; dst.h = h;
    dst.v.assign(size_t(w) * size_t(h), 0.0f);
    const Plane mid{tmp, w, h};
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float a = 0.0f;
            for (int i = -r; i <= r; ++i) a += k[size_t(i + r)] * mid.At(x, y + i);
            dst.v[size_t(y) * size_t(w) + size_t(x)] = a;
        }
}

double Ms(std::chrono::steady_clock::time_point a,
          std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

}  // namespace

int main(int argc, char** argv) {
    // Defaults are a 22 MP frame with SIFT's own defaults, so the number is
    // comparable to the measurement that motivated this work.
    int w = (argc > 1) ? std::atoi(argv[1]) : 5760;
    int h = (argc > 2) ? std::atoi(argv[2]) : 3840;
    const int octaves = 4;
    const int perOct  = 3;
    const int planes  = perOct + 3;
    const float sigma0 = 1.6f;
    const float k = std::pow(2.0f, 1.0f / float(perOct));

    std::printf("SIFT pyramid: %dx%d (%.1f MP), %d octaves x %d planes\n\n",
                w, h, double(w) * double(h) / 1e6, octaves, planes);

    Plane base;
    base.w = w; base.h = h;
    base.v.assign(size_t(w) * size_t(h), 0.0f);
    uint32_t s = 0x9E3779B9u;
    for (size_t i = 0; i < base.v.size(); ++i) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        base.v[i] = float(s & 0xFFFF) / 65535.0f;
    }

    std::vector<float> adds;
    for (int i = 1; i < planes; ++i) {
        const float prev = sigma0 * std::pow(k, float(i - 1));
        const float cur  = sigma0 * std::pow(k, float(i));
        adds.push_back(std::sqrt(std::max(cur * cur - prev * prev, 0.01f)));
    }

    // --- CPU ---------------------------------------------------------------
    double cpuMs = 0.0;
    {
        const auto t0 = std::chrono::steady_clock::now();
        Plane oct = base;
        std::vector<float> tmp;
        for (int o = 0; o < octaves && oct.w >= 16 && oct.h >= 16; ++o) {
            std::vector<Plane> g;
            g.resize(size_t(planes));
            g[0] = oct;
            for (int i = 1; i < planes; ++i)
                CpuBlur(g[size_t(i - 1)], g[size_t(i)], adds[size_t(i - 1)], tmp);

            const Plane& src = g[size_t(perOct)];
            Plane next;
            next.w = std::max(1, src.w / 2);
            next.h = std::max(1, src.h / 2);
            next.v.assign(size_t(next.w) * size_t(next.h), 0.0f);
            for (int y = 0; y < next.h; ++y)
                for (int x = 0; x < next.w; ++x)
                    next.v[size_t(y) * size_t(next.w) + size_t(x)] = src.At(x * 2, y * 2);
            oct = std::move(next);
        }
        cpuMs = Ms(t0, std::chrono::steady_clock::now());
    }
    std::printf("  CPU  %8.0f ms\n", cpuMs);

    // --- GPU ---------------------------------------------------------------
    ID3D12Device* dev = nullptr;
    if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
                                 IID_PPV_ARGS(&dev)))) {
        std::printf("  GPU       n/a (no device)\n");
        return 0;
    }
    ComputeContext gpu;
    if (!gpu.Init(dev)) {
        std::printf("  GPU       n/a (compute init failed)\n");
        dev->Release();
        return 1;
    }

    // Warm-up, excluded from the timing: the first call compiles both kernels
    // through DXC, which is milliseconds of CPU work that has nothing to do
    // with how fast the blur runs. The detector pays it once per process.
    {
        GpuPlane tiny;
        tiny.w = 64; tiny.h = 64;
        tiny.v.assign(64 * 64, 0.5f);
        std::vector<GpuPlane> out;
        std::string err;
        GpuBlurStack(&gpu, tiny, {1.0f}, &out, &err);
    }

    double gpuMs = 0.0;
    {
        const auto t0 = std::chrono::steady_clock::now();
        GpuPlane oct;
        oct.v = base.v; oct.w = base.w; oct.h = base.h;

        for (int o = 0; o < octaves && oct.w >= 16 && oct.h >= 16; ++o) {
            std::vector<GpuPlane> g;
            std::string err;
            if (!GpuBlurStack(&gpu, oct, adds, &g, &err)) {
                std::printf("  GPU  failed: %s\n", err.c_str());
                dev->Release();
                return 1;
            }
            const GpuPlane& src = g[size_t(perOct)];
            GpuPlane next;
            next.w = std::max(1, src.w / 2);
            next.h = std::max(1, src.h / 2);
            next.v.assign(size_t(next.w) * size_t(next.h), 0.0f);
            for (int y = 0; y < next.h; ++y)
                for (int x = 0; x < next.w; ++x)
                    next.v[size_t(y) * size_t(next.w) + size_t(x)] =
                        src.v[size_t(y * 2) * size_t(src.w) + size_t(x * 2)];
            oct = std::move(next);
        }
        gpuMs = Ms(t0, std::chrono::steady_clock::now());
    }
    std::printf("  GPU  %8.0f ms   %.1fx\n", gpuMs,
                gpuMs > 0.0 ? cpuMs / gpuMs : 0.0);

    dev->Release();
    return 0;
}
