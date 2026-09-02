// GPU/CPU conformance for the feature-detector scale-space pyramid.
//
// The GPU pyramid exists to make SIFT faster, not different. If the two paths
// disagree even slightly the DoG extrema move, so a detector would find
// different keypoints depending on whether a device was present -- which would
// invalidate every comparison this lab is for. So the bar here is not "close
// enough to look the same"; it is that the blur is the same arithmetic.
//
// WHY THE TEST IMAGE IS NOISE AND EDGES, NOT A GRADIENT
//
// gpu_audit reported three shipped demosaicers as clean while their GPU colour
// was badly wrong, because its synthetic gradient never drove a channel
// negative -- a smooth input cannot exercise a clamp. The same trap applies to
// a blur: a linear ramp is its own blur, so a kernel with wrong weights, a
// wrong radius, or no normalisation still reproduces it exactly in the
// interior. Only the edges would differ, and only slightly.
//
// This image therefore has hard steps, isolated impulses, and high-frequency
// noise -- content where a wrong kernel cannot hide. The impulses matter most:
// blurring a delta recovers the kernel itself, so any weight error shows up
// directly rather than being averaged away.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <d3d12.h>

#include "../src/algo_util/features.h"
#include "../src/algorithms/features/gpu_pyramid.h"
#include "../src/core/algorithm.h"
#include "../src/core/pipeline.h"
#include "../src/gpu/compute.h"

using namespace tglab;

namespace {

int g_failures = 0;

void Fail(const std::string& what) {
    std::printf("FAIL: %s\n", what.c_str());
    ++g_failures;
}

// The CPU blur, copied from detect_sift.cpp verbatim.
//
// Copied rather than shared on purpose. The point of the test is that the GPU
// agrees with what the detector ACTUALLY runs; if both sides called one helper,
// the test would still pass after someone changed that helper in a way the
// shader does not match. This is the reference, so it must be a transcription.
struct Plane {
    std::vector<float> v;
    int w = 0, h = 0;

    float At(int x, int y) const {
        return v[size_t(std::clamp(y, 0, h - 1)) * size_t(w) +
                 size_t(std::clamp(x, 0, w - 1))];
    }
};

void CpuBlur(const Plane& src, Plane& dst, float sigma) {
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

    std::vector<float> tmp(size_t(w) * size_t(h), 0.0f);
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

// Deterministic content a wrong kernel cannot reproduce.
Plane MakeTestPlane(int w, int h) {
    Plane p;
    p.w = w; p.h = h;
    p.v.assign(size_t(w) * size_t(h), 0.0f);

    uint32_t s = 0x9E3779B9u;   // fixed seed: a flaky conformance test is worse
                                // than none, because it teaches you to ignore it
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            float v = float(s & 0xFFFF) / 65535.0f * 0.25f;   // noise floor

            if (x > w / 3 && x < 2 * w / 3) v += 0.5f;        // vertical step
            if (y > h / 2) v += 0.25f;                        // horizontal step
            if ((x % 17) == 0 && (y % 13) == 0) v += 1.0f;    // impulses

            p.v[size_t(y) * size_t(w) + size_t(x)] = v;
        }
    }
    return p;
}

// Blobs at several scales plus noise, for the end-to-end detector check.
//
// NOT a plain checkerboard. A checkerboard's corners are all the same size, so
// they all land in one octave and yielded 15 keypoints -- too few to be
// convincing, and blind to whether the upper octaves agree at all. Varying the
// blob radius spreads keypoints across every octave, which is precisely where a
// pyramid bug would show: a wrong decimation only affects octaves 1 and up, and
// a test living entirely in octave 0 would pass through it.
Image MakeDetectorSource(int dim) {
    Image img;
    img.Alloc({dim, dim, Format::RGBA8});
    ImageView v = img.MapCpuWrite();

    uint32_t s = 0x1234567u;
    for (int y = 0; y < dim; ++y)
        for (int x = 0; x < dim; ++x) {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            uint8_t* p = v.At<uint8_t>(x, y);
            const uint8_t n = uint8_t(20 + (s & 31));
            p[0] = n; p[1] = n; p[2] = n; p[3] = 255;
        }

    // Radii from 2 to 20 px, so features exist at every scale the pyramid
    // covers rather than only the finest.
    for (int i = 0; i < 40; ++i) {
        const int r  = 2 + (i % 10) * 2;
        const int cx = 24 + (i * 53) % (dim - 48);
        const int cy = 24 + (i * 97) % (dim - 48);
        const uint8_t tone = (i % 2) ? 245 : 10;
        for (int y = cy - r; y <= cy + r; ++y)
            for (int x = cx - r; x <= cx + r; ++x) {
                if (x < 0 || y < 0 || x >= dim || y >= dim) continue;
                if ((x - cx) * (x - cx) + (y - cy) * (y - cy) > r * r) continue;
                uint8_t* p = v.At<uint8_t>(x, y);
                p[0] = tone; p[1] = tone; p[2] = tone;
            }
    }
    return img;
}

// Blurs are unit-gain, so an absolute tolerance is the right measure: the
// output range matches the input range and does not grow with sigma.
//
// 2e-5 is float rounding over a few hundred multiply-adds, not a fudge factor.
// The two paths sum the same weights in the same order, so they should agree to
// within an ulp or two; anything larger means a real difference in the
// arithmetic, which is exactly what this test is for.
constexpr float kTol = 2e-5f;

void Compare(const char* what, const Plane& cpu, const GpuPlane& gpu) {
    if (cpu.w != gpu.w || cpu.h != gpu.h) {
        Fail(std::string(what) + ": extent differs");
        return;
    }

    double worst = 0.0;
    int wx = -1, wy = -1;
    for (int y = 0; y < cpu.h; ++y)
        for (int x = 0; x < cpu.w; ++x) {
            const size_t i = size_t(y) * size_t(cpu.w) + size_t(x);
            const double d = std::fabs(double(cpu.v[i]) - double(gpu.v[i]));
            if (d > worst) { worst = d; wx = x; wy = y; }
        }

    if (worst > kTol) {
        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "%s: max |cpu-gpu| = %.3e at (%d,%d), tolerance %.1e",
                      what, worst, wx, wy, double(kTol));
        Fail(buf);
        if (const char* dbg = std::getenv("PYRAMID_DUMP")) {
            (void)dbg;
            std::printf("      row y=%d, x=%d..%d\n", wy,
                        std::max(0, wx - 3), std::min(cpu.w - 1, wx + 3));
            std::printf("      cpu:");
            for (int x = std::max(0, wx - 3); x <= std::min(cpu.w - 1, wx + 3); ++x)
                std::printf(" %8.4f", cpu.v[size_t(wy) * size_t(cpu.w) + size_t(x)]);
            std::printf("\n      gpu:");
            for (int x = std::max(0, wx - 3); x <= std::min(cpu.w - 1, wx + 3); ++x)
                std::printf(" %8.4f", gpu.v[size_t(wy) * size_t(cpu.w) + size_t(x)]);
            std::printf("\n      col x=%d, y=%d..%d\n", wx,
                        std::max(0, wy - 3), std::min(cpu.h - 1, wy + 3));
            std::printf("      cpu:");
            for (int y = std::max(0, wy - 3); y <= std::min(cpu.h - 1, wy + 3); ++y)
                std::printf(" %8.4f", cpu.v[size_t(y) * size_t(cpu.w) + size_t(wx)]);
            std::printf("\n      gpu:");
            for (int y = std::max(0, wy - 3); y <= std::min(cpu.h - 1, wy + 3); ++y)
                std::printf(" %8.4f", gpu.v[size_t(y) * size_t(cpu.w) + size_t(wx)]);
            std::printf("\n");
        }
    } else {
        std::printf("  ok  %-28s max diff %.3e\n", what, worst);
    }
}

}  // namespace

int main() {
    ID3D12Device* dev = nullptr;
    if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
                                 IID_PPV_ARGS(&dev)))) {
        // Not a failure. A machine without a device still builds and runs the
        // CPU path, and reporting this as a break would make the suite red for
        // a reason that has nothing to do with the code.
        std::printf("no D3D12 device; skipping GPU pyramid conformance\n");
        return 0;
    }

    ComputeContext gpu;
    if (!gpu.Init(dev)) {
        std::printf("compute context failed to initialise\n");
        dev->Release();
        return 1;
    }

    std::printf("GPU pyramid conformance\n");

    // Odd extents on purpose: a dispatch is 8x8, so an image that is not a
    // multiple of the tile has threads past the edge. Getting the bounds check
    // wrong there is the classic GPU bug, and an image sized 512x512 would
    // never reveal it.
    const Plane base = MakeTestPlane(197, 131);

    // Sigmas spanning what the detector actually asks for. The small ones are
    // the incremental blurs between levels (radius 2-3); the large one is a
    // radius-24 kernel, which is where a root-constant-sized weight buffer
    // would have overflowed.
    const float kSigmas[] = {0.5f, 0.8f, 1.2263f, 1.6f, 3.0f, 8.0f};

    for (float sigma : kSigmas) {
        Plane cpu;
        CpuBlur(base, cpu, sigma);

        GpuPlane gbase;
        gbase.v = base.v; gbase.w = base.w; gbase.h = base.h;

        GpuPlane got;
        std::string err;
        if (!GpuBlur(&gpu, gbase, &got, sigma, &err)) {
            Fail("blur sigma " + std::to_string(sigma) + ": " + err);
            continue;
        }

        char label[64];
        std::snprintf(label, sizeof label, "blur sigma %.4f", double(sigma));
        Compare(label, cpu, got);
    }

    // The stack, which is what the detector calls. Worth testing separately
    // from a single blur: it chains levels on the device without reading them
    // back, so a ping-pong mistake would leave a level blurred twice or not at
    // all -- and every INDIVIDUAL blur would still be correct.
    {
        const std::vector<float> adds = {0.8f, 1.0f, 1.2f, 1.5f};

        std::vector<Plane> cpu;
        cpu.push_back(base);
        for (float a : adds) {
            Plane next;
            CpuBlur(cpu.back(), next, a);
            cpu.push_back(std::move(next));
        }

        GpuPlane gbase;
        gbase.v = base.v; gbase.w = base.w; gbase.h = base.h;

        std::vector<GpuPlane> got;
        std::string err;
        if (!GpuBlurStack(&gpu, gbase, adds, &got, &err)) {
            Fail("blur stack: " + err);
        } else if (got.size() != cpu.size()) {
            Fail("blur stack: wrong depth");
        } else {
            for (size_t i = 0; i < cpu.size(); ++i) {
                char label[64];
                std::snprintf(label, sizeof label, "stack level %d", int(i));
                Compare(label, cpu[i], got[i]);
            }
        }
    }

    // --- end to end --------------------------------------------------------
    //
    // The blurs agreeing is necessary but not sufficient. What the lab actually
    // depends on is that detect_sift finds the SAME KEYPOINTS either way --
    // that is the property that makes a detector comparison mean anything, and
    // it is a stronger claim than the planes matching, because extrema
    // detection amplifies a small difference into a keypoint that exists on one
    // path and not the other.
    //
    // ForceCPU withholds the device (see pipeline.cpp), so the two runs really
    // do take different paths through the same detector.
    for (const char* name : {"detect_sift", "detect_orb", "detect_brisk",
                             "detect_akaze"}) {
        auto run = [&](ExecMode mode, std::vector<Keypoint>* out,
                       std::string* err) -> bool {
            std::vector<Data> sources;
            sources.push_back(Data{MakeDetectorSource(384)});

            Pipeline pipe;
            pipe.AddStage(Registry::Get().Create(name), name, {{-1, 0}}, 1, 1);

            if (!pipe.Execute(&sources, nullptr, err, &gpu, mode)) return false;

            const Data* d = pipe.Resolve({0, 0}, &sources);
            if (!d || !std::holds_alternative<Image>(*d)) {
                *err = "no detector output";
                return false;
            }
            const Image& im = std::get<Image>(*d);
            const FeatureSidecar* sc = im.Sidecars().Get<FeatureSidecar>(kFeatureSidecar);
            if (!sc) { *err = "no feature sidecar"; return false; }
            *out = sc->keypoints;
            return true;
        };

        std::vector<Keypoint> onCpu, onGpu;
        std::string err;

        if (!run(ExecMode::ForceCPU, &onCpu, &err)) {
            Fail(std::string(name) + " on CPU: " + err);
        } else if (!run(ExecMode::Auto, &onGpu, &err)) {
            Fail(std::string(name) + " with device: " + err);
        } else if (onCpu.size() != onGpu.size()) {
            char buf[160];
            std::snprintf(buf, sizeof buf,
                          "%s: keypoint COUNT differs: cpu %d, gpu %d",
                          name, int(onCpu.size()), int(onGpu.size()));
            Fail(buf);
        } else if (onCpu.empty()) {
            // An empty list would make every comparison below trivially pass,
            // so a detector that silently found nothing must not read as ok.
            Fail(std::string(name) + ": found no keypoints at all");
        } else {
            // Position, scale and ANGLE. Position alone would miss a descriptor
            // built from a different orientation -- the keypoint would sit in
            // the right place and describe the wrong thing.
            double worst = 0.0;
            for (size_t i = 0; i < onCpu.size(); ++i) {
                worst = std::max(worst, double(std::fabs(onCpu[i].x - onGpu[i].x)));
                worst = std::max(worst, double(std::fabs(onCpu[i].y - onGpu[i].y)));
                worst = std::max(worst,
                                 double(std::fabs(onCpu[i].scale - onGpu[i].scale)));
                worst = std::max(worst,
                                 double(std::fabs(onCpu[i].angle - onGpu[i].angle)));
            }
            if (worst > 1e-3) {
                char buf[160];
                std::snprintf(buf, sizeof buf,
                              "%s: keypoints differ by %.3e", name, worst);
                Fail(buf);
            } else {
                char label[64];
                std::snprintf(label, sizeof label, "%s cpu == gpu", name);
                std::printf("  ok  %-28s %d keypoints, max shift %.1e\n",
                            label, int(onCpu.size()), worst);
            }
        }
    }

    dev->Release();

    if (g_failures) {
        std::printf("\n%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("\nall pyramid checks passed\n");
    return 0;
}
