// bench_nn -- tglab's networks against PyTorch's, layer by layer, and timed.
//
// The networks in src/algo_util are transcriptions, and a transcription is
// only as good as its check. tools/nn_reference.py runs the published model
// in PyTorch (float32) and saves its input, its intermediate feature maps and
// its output; this runs ours on that SAME input -- so image loading and
// resizing cannot muddy the comparison -- on the CPU and on the GPU, and
// reports how far each tensor is from PyTorch's.
//
//   python tools/nn_convert.py dad dad.tgw
//   python tools/nn_reference.py dad photo.jpg dad_ref.tgw
//   bench_nn dad dad.tgw dad_ref.tgw [--cpu-only] [--gpu-only] [--runs N]
//
// A layer that disagrees where the one before it agreed is where to look.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <d3d12.h>

#include "../src/algo_util/dad.h"
#include "../src/algo_util/dedode.h"
#include "../src/algo_util/loma.h"
#include "../src/algo_util/nn.h"
#include "../src/gpu/compute.h"

using namespace tglab;

namespace {

double Ms(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

bool g_profile = false;   // --profile

// Each operation kind's share of one profiled run, largest first.
void PrintProfile(const std::map<std::string, double>& ms) {
    std::vector<std::pair<double, std::string>> v;
    double total = 0.0;
    for (const auto& [k, t] : ms) { v.push_back({t, k}); total += t; }
    std::sort(v.rbegin(), v.rend());
    std::printf("  profile (each op waited for; %.1f ms in all):\n", total);
    for (const auto& [t, k] : v)
        std::printf("    %-12s %7.1f ms  %4.1f%%\n", k.c_str(), t, 100.0 * t / total);
}

// Largest difference, relative to the reference's largest magnitude: a scale-
// free figure, since a feature map's values run anywhere from 1e-3 to 1e3.
void Compare(const char* name, const std::vector<float>& got, const nn::Weight* ref) {
    if (!ref) { std::printf("  %-10s (no reference)\n", name); return; }
    if (got.size() != ref->data.size()) {
        std::printf("  %-10s SIZE %zu vs %zu\n", name, got.size(), ref->data.size());
        return;
    }
    double maxDiff = 0.0, maxRef = 0.0, sumSq = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double d = std::fabs(double(got[i]) - double(ref->data[i]));
        maxDiff = std::max(maxDiff, d);
        maxRef = std::max(maxRef, std::fabs(double(ref->data[i])));
        sumSq += d * d;
    }
    std::printf("  %-10s max |diff| %.3e  (%.2e of max |ref| %.3g), rms %.3e\n", name,
                maxDiff, maxRef > 0 ? maxDiff / maxRef : 0.0, maxRef,
                std::sqrt(sumSq / double(got.size())));
}

// How many of PyTorch's keypoints we reproduce, and how closely.
void CompareKeypoints(const std::vector<DadPoint>& got, const nn::Weight* ref) {
    if (!ref || ref->shape.size() != 2) return;
    const size_t n = ref->shape[0];
    // Nearest of ours to each of theirs, by brute force over a coarse grid.
    std::map<long long, std::vector<size_t>> grid;
    auto key = [](float x, float y) { return (long long)(std::floor(y / 4)) * 100000 + (long long)(std::floor(x / 4)); };
    for (size_t i = 0; i < got.size(); ++i) grid[key(got[i].x, got[i].y)].push_back(i);
    size_t within01 = 0, within1 = 0;
    double worst = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const float rx = ref->data[i * 2], ry = ref->data[i * 2 + 1];
        double best = 1e30;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                const auto it = grid.find(key(rx + 4.0f * float(dx), ry + 4.0f * float(dy)));
                if (it == grid.end()) continue;
                for (size_t j : it->second)
                    best = std::min(best, std::hypot(double(got[j].x - rx), double(got[j].y - ry)));
            }
        if (best <= 0.01) ++within01;
        if (best <= 1.0) ++within1;
        else worst = std::max(worst, best);
    }
    std::printf("  keypoints  %zu of %zu within 0.01 px, %zu within 1 px\n", within01, n, within1);
}

bool RunDad(const char* label, nn::Engine& eng, const DadNet& net, const nn::WeightFile& ref,
            int runs) {
    const nn::Weight* in = ref.Find("input");
    if (!in || in->shape.size() != 3) { std::printf("reference has no input\n"); return false; }
    const int h = in->shape[1], w = in->shape[2];
    std::printf("%s, %dx%d\n", label, w, h);

    std::map<std::string, std::vector<float>> taps;
    std::vector<float> logits;
    std::string err;
    if (!net.Logits(eng, in->data, h, w, &logits, &err, &taps)) {
        std::printf("  FAILED: %s\n", err.c_str());
        return false;
    }
    for (const char* t : {"feat1", "feat2", "feat4", "feat8", "delta8", "delta4", "delta2", "delta1"})
        Compare(t, taps[t], ref.Find(t));
    Compare("logits", logits, ref.Find("logits"));
    const nn::Weight* kp = ref.Find("keypoints");
    const int n = kp ? kp->shape[0] : 4096;
    CompareKeypoints(DadSample(logits, h, w, n), kp);

    // Timed without taps, which download every intermediate.
    for (int r = 0; r < runs; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        if (!net.Logits(eng, in->data, h, w, &logits, &err)) return false;
        const double tNet = Ms(t0);
        const auto t1 = std::chrono::steady_clock::now();
        const auto pts = DadSample(logits, h, w, n);
        std::printf("  run %d: network %.1f ms, sampling %.1f ms (%zu points)\n", r + 1, tNet,
                    Ms(t1), pts.size());
    }
    if (g_profile) {
        std::map<std::string, double> ms;
        eng.Profile(&ms);
        net.Logits(eng, in->data, h, w, &logits, &err);
        eng.Profile(nullptr);
        PrintProfile(ms);
    }
    return true;
}

// The descriptor on PyTorch's own keypoints (kp_norm), so detection cannot
// muddy the comparison. Also how close the DIRECTIONS are -- what matching
// actually compares -- as the worst cosine between ours and theirs.
bool RunDedode(const char* label, nn::Engine& eng, const DedodeDescriptor& net,
               const nn::WeightFile& ref, int runs) {
    const nn::Weight* in = ref.Find("input");
    const nn::Weight* kp = ref.Find("kp_norm");
    if (!in || in->shape.size() != 3 || !kp) { std::printf("reference incomplete\n"); return false; }
    const int h = in->shape[1], w = in->shape[2];
    std::printf("%s, %dx%d, %d points\n", label, w, h, kp->shape[0]);

    NnTaps taps;
    std::vector<float> desc;
    std::string err;
    const bool layered = ref.Find("feat1") != nullptr;
    if (!net.Describe(eng, in->data, h, w, kp->data, &desc, &err, layered ? &taps : nullptr)) {
        std::printf("  FAILED: %s\n", err.c_str());
        return false;
    }
    if (layered)
        for (const char* t : {"feat1", "feat2", "feat4", "feat8", "delta8", "delta4", "delta2",
                              "delta1", "descmap"})
            Compare(t, taps[t], ref.Find(t));
    const nn::Weight* rd = ref.Find("descriptors");
    Compare("descript.", desc, rd);
    if (rd && rd->data.size() == desc.size()) {
        const int d = net.Dim();
        double worst = 1.0;
        for (size_t i = 0; i + size_t(d) <= desc.size(); i += size_t(d)) {
            double ab = 0, aa = 0, bb = 0;
            for (int k = 0; k < d; ++k) {
                ab += double(desc[i + size_t(k)]) * rd->data[i + size_t(k)];
                aa += double(desc[i + size_t(k)]) * desc[i + size_t(k)];
                bb += double(rd->data[i + size_t(k)]) * rd->data[i + size_t(k)];
            }
            worst = std::min(worst, ab / std::sqrt(std::max(aa * bb, 1e-30)));
        }
        std::printf("  worst cosine to PyTorch's descriptor: %.7f\n", worst);
    }
    for (int r = 0; r < runs; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        if (!net.Describe(eng, in->data, h, w, kp->data, &desc, &err)) return false;
        std::printf("  run %d: %.1f ms\n", r + 1, Ms(t0));
    }
    if (g_profile) {
        std::map<std::string, double> ms;
        eng.Profile(&ms);
        net.Describe(eng, in->data, h, w, kp->data, &desc, &err);
        eng.Profile(nullptr);
        PrintProfile(ms);
    }
    return true;
}

// The matcher on PyTorch's own keypoints and descriptors: each layer's
// descriptors, the similarity, and the matches filter_matches kept.
bool RunLoma(const char* label, nn::Engine& eng, const LomaMatcher& net,
             const nn::WeightFile& ref, int runs) {
    const nn::Weight *k0 = ref.Find("kpts0"), *k1 = ref.Find("kpts1");
    const nn::Weight *d0 = ref.Find("desc0"), *d1 = ref.Find("desc1"), *m0 = ref.Find("m0");
    if (!k0 || !k1 || !d0 || !d1 || !m0) { std::printf("reference incomplete\n"); return false; }
    std::printf("%s, %d x %d points, %d layers\n", label, k0->shape[0], k1->shape[0], net.Layers());

    NnTaps taps;
    std::vector<LomaMatcher::Pair> pairs;
    std::string err;
    if (!net.Match(eng, k0->data, d0->data, k1->data, d1->data, net.Layers(), 0.1f, &pairs, &err,
                   &taps)) {
        std::printf("  FAILED: %s\n", err.c_str());
        return false;
    }
    Compare("proj0", taps["proj0"], ref.Find("proj0"));
    for (int i = 0; i < net.Layers(); ++i)
        for (int s = 0; s < 2; ++s) {
            const std::string n = "layer" + std::to_string(i) + "_" + std::to_string(s);
            Compare(n.c_str(), taps[n], ref.Find(n));
        }
    Compare("sim", taps["sim"], ref.Find("sim"));

    // The matches, against PyTorch's m0 (-1 for none).
    int theirs = 0, same = 0;
    std::vector<int> ours(m0->data.size(), -1);
    for (const auto& p : pairs) ours[size_t(p.i)] = p.j;
    for (size_t i = 0; i < ours.size(); ++i) {
        const int t = int(m0->data[i]);
        if (t >= 0) ++theirs;
        if (t >= 0 && ours[i] == t) ++same;
    }
    std::printf("  matches    ours %zu, PyTorch %d, identical %d\n", pairs.size(), theirs, same);

    for (int r = 0; r < runs; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        if (!net.Match(eng, k0->data, d0->data, k1->data, d1->data, net.Layers(), 0.1f, &pairs, &err))
            return false;
        std::printf("  run %d: %.1f ms\n", r + 1, Ms(t0));
    }
    if (g_profile) {
        std::map<std::string, double> ms;
        eng.Profile(&ms);
        net.Match(eng, k0->data, d0->data, k1->data, d1->data, net.Layers(), 0.1f, &pairs, &err);
        eng.Profile(nullptr);
        PrintProfile(ms);
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    const bool isDad = argc >= 2 && !std::strcmp(argv[1], "dad");
    const bool isDedode = argc >= 2 && !std::strcmp(argv[1], "dedode");
    const bool isLoma = argc >= 2 && !std::strcmp(argv[1], "loma");
    if (argc < 4 || (!isDad && !isDedode && !isLoma)) {
        std::printf("usage: bench_nn dad|dedode|loma <weights.tgw> <reference.tgw> "
                    "[--cpu-only|--gpu-only] [--runs N]\n");
        return 2;
    }
    bool cpu = true, gpu = true;
    int runs = 3;
    for (int i = 4; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--cpu-only")) gpu = false;
        else if (!std::strcmp(argv[i], "--gpu-only")) cpu = false;
        else if (!std::strcmp(argv[i], "--runs") && i + 1 < argc) runs = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--profile")) g_profile = true;
    }

    std::string err;
    nn::WeightFile weights, ref;
    if (!weights.Load(argv[2], &err) || !ref.Load(argv[3], &err)) {
        std::printf("%s\n", err.c_str());
        return 1;
    }
    DadNet dad;
    DedodeDescriptor dedode;
    LomaMatcher loma;
    if (isDad ? !dad.Load(weights, &err)
              : isLoma ? !loma.Load(weights, &err) : !dedode.Load(weights, &err)) {
        std::printf("%s\n", err.c_str());
        return 1;
    }
    auto run = [&](const char* label, nn::Engine& eng, int n) {
        return isDad    ? RunDad(label, eng, dad, ref, n)
               : isLoma ? RunLoma(label, eng, loma, ref, n)
                        : RunDedode(label, eng, dedode, ref, n);
    };

    bool ok = true;
    if (gpu) {
        ID3D12Device* dev = nullptr;
        ComputeContext ctx;
        if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&dev))) ||
            !ctx.Init(dev)) {
            std::printf("no GPU\n");
        } else {
            nn::Engine eng(&ctx);
            ok = run("GPU", eng, runs) && ok;
        }
        ctx.Shutdown();
        if (dev) dev->Release();
    }
    if (cpu) {
        nn::Engine eng;
        ok = run("CPU", eng, std::min(runs, 1)) && ok;
    }
    return ok ? 0 : 1;
}
