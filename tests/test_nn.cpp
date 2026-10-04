// Neural-network operations: the CPU reference against PyTorch, and the GPU
// kernels against the CPU reference.
//
// Two layers of check, because they catch different mistakes.
//
//   1. The CPU path is a TRANSCRIPTION of PyTorch's definitions, so its
//      awkward parts -- the interpolation index arithmetic above all -- are
//      pinned against numbers PyTorch itself produced (F.interpolate, run
//      once and pasted in). A transcription that is merely self-consistent
//      would pass every other check here.
//   2. Each GPU kernel against the CPU on sizes chosen to be awkward: odd
//      extents (an 8x8 group overhangs the edge), channel counts that are not
//      multiples of the kernel's 16-channel blocks, and writes into a channel
//      offset, which is how concatenation is done.
//
// The whole-network check -- DaD against the published model, layer by
// layer -- needs the weights and a PyTorch run, so it lives in
// tools/bench_nn.cpp rather than here.
//
// Skips the GPU half, rather than failing, without a device.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include <d3d12.h>

#include "../src/algo_util/dad.h"
#include "../src/algo_util/gpu_knn.h"
#include "../src/algo_util/nn.h"
#include "../src/gpu/compute.h"

using namespace tglab;
using namespace tglab::nn;

namespace {

int g_fail = 0;

void Check(bool ok, const std::string& what) {
    std::printf("[%s]  %s\n", ok ? " ok " : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

std::vector<float> Random(size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (float& x : v) x = u(rng);
    return v;
}

float MaxDiff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return 1e30f;
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

Conv MakeConv(int cout, int cin, int k, bool depthwise, unsigned seed) {
    Conv c;
    c.cout = cout; c.cin = cin; c.k = k; c.groups = depthwise ? cin : 1;
    c.w = Random(size_t(cout) * size_t(depthwise ? 1 : cin) * size_t(k * k), seed);
    c.b = Random(size_t(cout), seed + 1);
    return c;
}

// The convolution by its definition, the slowest possible way: an
// independent check on CpuConv's shifted-row loops.
std::vector<float> NaiveConv(const std::vector<float>& in, int cin, int h, int w,
                             const Conv& cv, bool relu) {
    std::vector<float> out(size_t(cv.cout) * size_t(h) * size_t(w));
    const int r = cv.k / 2, cinG = cv.cin / cv.groups, coutG = cv.cout / cv.groups;
    for (int oc = 0; oc < cv.cout; ++oc)
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                double s = cv.b[size_t(oc)];
                const int g = oc / coutG;
                for (int ic = 0; ic < cinG; ++ic)
                    for (int ky = 0; ky < cv.k; ++ky)
                        for (int kx = 0; kx < cv.k; ++kx) {
                            const int sy = y + ky - r, sx = x + kx - r;
                            if (sy < 0 || sy >= h || sx < 0 || sx >= w) continue;
                            s += double(cv.w[((size_t(oc) * size_t(cinG) + size_t(ic)) * size_t(cv.k) +
                                              size_t(ky)) * size_t(cv.k) + size_t(kx)]) *
                                 double(in[(size_t(g * cinG + ic) * size_t(h) + size_t(sy)) * size_t(w) +
                                           size_t(sx)]);
                        }
                out[(size_t(oc) * size_t(h) + size_t(y)) * size_t(w) + size_t(x)] =
                    relu ? std::max(float(s), 0.0f) : float(s);
            }
    (void)cin;
    return out;
}

// --- 1. the CPU path against PyTorch -------------------------------------------

void TestResizeAgainstPyTorch() {
    // x = i*i*0.1 - i over a 3x4 ramp: curved, so interpolation is visible.
    std::vector<float> x(12);
    for (int i = 0; i < 12; ++i) x[size_t(i)] = float(i) * float(i) * 0.1f - float(i);

    struct Case { Interp mode; int h, w; std::vector<float> want; };
    const Case cases[] = {
        {Interp::Bilinear, 5, 7, {0.0f, -0.3214286f, -0.8357143f, -1.25f, -1.635714f, -1.921429f, -2.1f,
                                   -0.9600001f, -1.167143f, -1.498571f, -1.73f, -1.932857f, -2.035714f, -2.1f,
                                   -2.4f, -2.435714f, -2.492857f, -2.45f, -2.378571f, -2.207143f, -2.1f,
                                   -1.92f, -1.784285f, -1.567142f, -1.25f, -0.9042851f, -0.4585708f, -0.1799993f,
                                   -1.6f, -1.35f, -0.9499996f, -0.4499998f, 0.07857168f, 0.7071433f, 1.1f}},
        {Interp::Bilinear, 2, 3, {-0.7166667f, -1.55f, -2.05f, -1.716667f, -0.9499998f, 0.1500004f}},
        {Interp::Bicubic, 5, 7, {0.3273143f, -0.05466673f, -0.6546299f, -1.172301f, -1.592086f, -1.951231f, -2.157229f,
                                  -0.9245872f, -1.181827f, -1.578313f, -1.897099f, -2.118002f, -2.273669f, -2.354926f,
                                  -2.390078f, -2.440225f, -2.498897f, -2.4875f, -2.378217f, -2.196073f, -2.070235f,
                                  -2.127569f, -1.970621f, -1.691482f, -1.3499f, -0.9104315f, -0.3904766f, -0.05754396f,
                                  -1.600272f, -1.318582f, -0.8359655f, -0.2955001f, 0.3428527f, 1.066284f, 1.523959f}},
        {Interp::Bicubic, 2, 3, {-0.6695711f, -1.629687f, -2.172212f, -1.800105f, -0.9828123f, 0.2520723f}},
    };
    Engine cpu;
    for (const Case& c : cases) {
        Tensor in, out;
        cpu.Upload(x, 1, 3, 4, &in);
        cpu.Alloc(&out, 1, c.h, c.w);
        cpu.Resize(in, 0, 1, c.mode, &out, 0);
        const float d = MaxDiff(out.cpu, c.want);
        char buf[128];
        std::snprintf(buf, sizeof buf, "%s 3x4 -> %dx%d matches F.interpolate (max diff %.2e)",
                      c.mode == Interp::Bilinear ? "bilinear" : "bicubic", c.h, c.w, d);
        Check(d < 2e-5f, buf);
    }
}

void TestSampleAgainstPyTorch() {
    // F.grid_sample on a 2x3x4 map, including a corner (-1,-1), the exact
    // centre, a point past the right edge (zero padding) and one just inside
    // the far corner -- each place the index arithmetic can go wrong.
    std::vector<float> x(24);
    for (int i = 0; i < 24; ++i) x[size_t(i)] = float(i) * float(i) * 0.1f - float(i);
    const std::vector<float> xy = {-1.0f, -1.0f, 0.0f, 0.0f, 0.37f, -0.81f,
                                   0.99f, 0.99f, 1.2f, 0.1f, -0.5f, 0.6f};
    const std::vector<float> want = {0.0f, 0.6000001f, -2.45f, 13.15f, -1.3502f, 4.75396f,
                                     0.29458f, 8.007217f, -0.1619998f, 1.901999f, -1.37f, 20.47f};
    Engine cpu;
    Tensor map, out;
    cpu.Upload(x, 2, 3, 4, &map);
    cpu.SampleBilinear(map, xy, &out);
    const float d = MaxDiff(out.cpu, want);
    Check(out.h == 6 && out.w == 2 && d < 2e-5f,
          "point sampling matches F.grid_sample (max diff " + std::to_string(d) + ")");
}

void TestLayerNormAgainstPyTorch() {
    // F.gelu(F.layer_norm(x, (5,), g, b)) on 3 points of 5 channels, laid
    // out channel-major as the engine holds points.
    const std::vector<float> x = {0.0f, 2.883826f, -1.589509f, 1.084846f, 2.389696f, -2.402001f,
                                  2.022864f, 1.572133f, -2.889393f, 2.687096f, 0.5417887f,
                                  -2.98572f, 2.987643f, -0.5618837f, -2.677943f};
    const std::vector<float> want = {-0.09958632f, 1.192967f, 1.881921f, -0.1551607f, 0.1227417f,
                                     -0.04322728f, -0.1124791f, -0.08381276f, 0.7453787f,
                                     1.955138f, -0.1570129f, -0.08748794f, -0.04604988f,
                                     -0.02840198f, -0.129638f};
    Vec g, b;
    g.v = {1.0f, 0.5f, -1.2f, 2.0f, 0.8f};
    b.v = {0.1f, -0.2f, 0.0f, 0.3f, -1.0f};
    Engine cpu;
    Tensor in, out;
    cpu.Upload(x, 5, 1, 3, &in);
    cpu.LayerNormGelu(in, g, b, &out);
    const float d = MaxDiff(out.cpu, want);
    Check(d < 2e-6f, "LayerNorm + GELU matches PyTorch (max diff " + std::to_string(d) + ")");
}

void TestConvAgainstDefinition() {
    Engine cpu;
    const int h = 11, w = 13, cin = 5;
    const std::vector<float> x = Random(size_t(cin) * h * w, 7);
    for (int k : {1, 3, 5}) {
        const Conv cv = MakeConv(19, cin, k, false, 100 + unsigned(k));
        Tensor in, out;
        cpu.Upload(x, cin, h, w, &in);
        cpu.Alloc(&out, 19, h, w);
        cpu.Conv2d(in, cv, true, &out);
        const float d = MaxDiff(out.cpu, NaiveConv(x, cin, h, w, cv, true));
        Check(d < 1e-4f, "CPU " + std::to_string(k) + "x" + std::to_string(k) +
                             " convolution matches its definition");
    }
    const Conv dw = MakeConv(cin, cin, 5, true, 300);
    Tensor in, out;
    cpu.Upload(x, cin, h, w, &in);
    cpu.Alloc(&out, cin, h, w);
    cpu.Conv2d(in, dw, false, &out);
    Check(MaxDiff(out.cpu, NaiveConv(x, cin, h, w, dw, false)) < 1e-4f,
          "CPU depthwise 5x5 matches its definition");
}

void TestWeightFile() {
    WeightFile f;
    Weight a;
    a.shape = {2, 3};
    a.data = {1, -2, 3.5f, 0, 1e-3f, -7};
    f.Put("layer.weight", a);
    const std::string path = "test_nn_weights.tgw";
    std::string err;
    WeightFile g;
    const bool ok = f.Save(path, &err) && g.Load(path, &err);
    const Weight* b = g.Find("layer.weight");
    Check(ok && b && b->shape == a.shape && b->data == a.data,
          "a .tgw round-trips float32 exactly" + (ok ? std::string() : ": " + err));

    // Float16, as nn_convert.py writes by default: 1, -2, 0.5, 65504 (the
    // largest half), 2^-24 (the smallest subnormal) and 0.
    FILE* fp = nullptr;
    fopen_s(&fp, path.c_str(), "wb");
    const uint16_t halves[6] = {0x3C00, 0xC000, 0x3800, 0x7BFF, 0x0001, 0x0000};
    const uint32_t count = 1;
    const uint16_t len = 1;
    const uint8_t dtype = 1, rank = 1;
    const int32_t dim = 6;
    std::fwrite("TGW1", 1, 4, fp);
    std::fwrite(&count, 4, 1, fp);
    std::fwrite(&len, 2, 1, fp);
    std::fwrite("h", 1, 1, fp);
    std::fwrite(&dtype, 1, 1, fp);
    std::fwrite(&rank, 1, 1, fp);
    std::fwrite(&dim, 4, 1, fp);
    std::fwrite(halves, 2, 6, fp);
    std::fclose(fp);
    WeightFile hf;
    const bool hok = hf.Load(path, &err);
    const Weight* h = hf.Find("h");
    const std::vector<float> want = {1.0f, -2.0f, 0.5f, 65504.0f, 5.9604645e-8f, 0.0f};
    Check(hok && h && h->data == want, "float16 weights decode, subnormals included");
    std::remove(path.c_str());
}

void TestSample() {
    // Two clean peaks and a plateau: NMS keeps both peaks, the stronger
    // first, and the sub-pixel offset leans toward the brighter neighbour.
    const int h = 16, w = 20;
    std::vector<float> lg(size_t(h) * w, 0.0f);
    lg[size_t(5) * w + 6] = 6.0f;
    lg[size_t(5) * w + 7] = 4.0f;    // pulls the first peak right
    lg[size_t(10) * w + 14] = 5.0f;
    const auto pts = DadSample(lg, h, w, 2);
    Check(pts.size() == 2 && pts[0].x > 6.5f && pts[0].x < 7.5f && std::fabs(pts[0].y - 5.5f) < 1e-3f &&
              std::fabs(pts[1].x - 14.5f) < 1e-3f && std::fabs(pts[1].y - 10.5f) < 1e-3f,
          "DadSample keeps both peaks, strongest first, refined toward the brighter side");
    // Logits 6 and 4 at temperature 0.5 are 12 and 8: a weight of e^-4 on
    // the right-hand offset, so +0.018 of a pixel.
    const float want = 6.5f + std::exp(8.0f) / (std::exp(12.0f) + std::exp(8.0f) + 7.0f);
    Check(pts.size() == 2 && std::fabs(pts[0].x - want) < 1e-5f,
          "...by the softmax-weighted offset (x " +
              std::to_string(pts.empty() ? 0.0f : pts[0].x) + ", want " + std::to_string(want) + ")");
}

// --- 2. the GPU against the CPU ------------------------------------------------

void TestGpu(ComputeContext* gpu) {
    Engine c, g(gpu);
    const int h = 19, w = 21;   // neither a multiple of 8

    auto both = [&](const std::vector<float>& v, int ch, Tensor* tc, Tensor* tg) {
        c.Upload(v, ch, h, w, tc);
        g.Upload(v, ch, h, w, tg);
    };
    auto same = [&](const Tensor& tc, const Tensor& tg, float tol, const std::string& what) {
        std::vector<float> a, b;
        c.Download(tc, &a);
        const bool ok = g.Download(tg, &b);
        const float d = MaxDiff(a, b);
        char buf[64];
        std::snprintf(buf, sizeof buf, " (max diff %.2e)", d);
        Check(ok && d < tol, "GPU " + what + " matches the CPU" + buf +
                                 (ok ? std::string() : ": " + g.Error()));
    };

    const std::vector<float> x = Random(size_t(37) * h * w, 11);
    Tensor xc, xg;
    both(x, 37, &xc, &xg);

    // Dense: 37 -> 41 channels (neither a multiple of 16), written at offset
    // 3 of a 50-channel tensor whose other channels must survive.
    for (int k : {1, 3, 5}) {
        const Conv cv = MakeConv(41, 37, k, false, 500 + unsigned(k));
        const std::vector<float> pre = Random(size_t(50) * h * w, 900);
        Tensor oc, og;
        both(pre, 50, &oc, &og);
        c.Conv2d(xc, cv, k != 3, &oc, 3);
        g.Conv2d(xg, cv, k != 3, &og, 3);
        same(oc, og, 2e-4f, std::to_string(k) + "x" + std::to_string(k) +
                                " convolution into a channel offset");
    }
    {
        const Conv dw = MakeConv(37, 37, 5, true, 600);
        Tensor oc, og;
        c.Alloc(&oc, 37, h, w);
        g.Alloc(&og, 37, h, w);
        c.Conv2d(xc, dw, true, &oc);
        g.Conv2d(xg, dw, true, &og);
        same(oc, og, 1e-5f, "depthwise 5x5 convolution");
    }
    {
        Tensor oc, og;
        c.MaxPool2(xc, &oc);
        g.MaxPool2(xg, &og);
        same(oc, og, 0.0f + 1e-30f, "2x2 max pooling of an odd size");
    }
    for (Interp mode : {Interp::Bilinear, Interp::Bicubic})
        for (int scale : {2, 3}) {
            Tensor oc, og;
            const int oh = h * scale / 2 + 1, ow = w * scale / 2 + 3;   // not integer ratios
            c.Alloc(&oc, 9, oh, ow);
            g.Alloc(&og, 9, oh, ow);
            // Channels 5..11 of x into 2..8 of the output; 0..1 left alone.
            const std::vector<float> pre = Random(size_t(9) * oh * ow, 77);
            c.Upload(pre, 9, oh, ow, &oc);
            g.Upload(pre, 9, oh, ow, &og);
            c.Resize(xc, 5, 7, mode, &oc, 2);
            g.Resize(xg, 5, 7, mode, &og, 2);
            same(oc, og, 1e-5f, std::string(mode == Interp::Bilinear ? "bilinear" : "bicubic") +
                                    " resize to " + std::to_string(ow) + "x" + std::to_string(oh));
        }
    {
        const std::vector<float> y = Random(size_t(37) * h * w, 12);
        Tensor yc, yg, sc, sg;
        both(y, 37, &yc, &yg);
        c.AddScaled(xc, yc, 1.0f / 1.4f, &sc);
        g.AddScaled(xg, yg, 1.0f / 1.4f, &sg);
        same(sc, sg, 1e-6f, "(a + b) * s");
        c.Accumulate(xc, 4, 6, &yc, 20);
        g.Accumulate(xg, 4, 6, &yg, 20);
        c.Copy(xc, 30, 7, &yc, 0);
        g.Copy(xg, 30, 7, &yg, 0);
        same(yc, yg, 1e-6f, "channel copy and accumulate");
    }

    {
        // 300 points, some outside the map, over 37 channels -- more than a
        // 64-thread group, and not a multiple of it.
        std::vector<float> xy = Random(600, 31);
        for (float& v : xy) v *= 1.1f;
        Tensor oc, og;
        c.SampleBilinear(xc, xy, &oc);
        g.SampleBilinear(xg, xy, &og);
        same(oc, og, 1e-6f, "point sampling");
    }

    // --- transformer operations, on point sets (d x 1 x n) ------------------
    // Counts that are neither multiples of the 64-thread groups nor of the
    // attention's 16-key tiles; cross attention between unequal sets.
    {
        const int n0 = 197, n1 = 133, d = 256, heads = 4, hd = 64;
        auto pts = [&](int ch, int n, unsigned seed, Tensor* tc, Tensor* tg) {
            const std::vector<float> v = Random(size_t(ch) * size_t(n), seed);
            c.Upload(v, ch, 1, n, tc);
            g.Upload(v, ch, 1, n, tg);
        };
        auto sameT = [&](const Tensor& tc, const Tensor& tg, float tol, const std::string& what) {
            std::vector<float> a, b;
            c.Download(tc, &a);
            const bool ok = g.Download(tg, &b);
            const float dd = MaxDiff(a, b);
            char buf[64];
            std::snprintf(buf, sizeof buf, " (max diff %.2e)", dd);
            Check(ok && dd < tol, "GPU " + what + " matches the CPU" + buf +
                                     (ok ? std::string() : ": " + g.Error()));
        };

        Tensor ac, ag, bc, bg;
        pts(512, n0, 61, &ac, &ag);
        Vec gam, bet;
        gam.v = Random(512, 62);
        bet.v = Random(512, 63);
        Tensor lc, lg;
        c.LayerNormGelu(ac, gam, bet, &lc);
        g.LayerNormGelu(ag, gam, bet, &lg);
        sameT(lc, lg, 2e-5f, "LayerNorm + GELU");

        Tensor qc, qg, pc, pg;
        pts(3 * d, n0, 64, &qc, &qg);
        pts(2, n0, 65, &pc, &pg);
        Vec fr;
        fr.v = Random(size_t(hd), 66);
        for (float& f : fr.v) f *= 3.0f;
        c.Rotary(&qc, 0, d, hd, pc, fr);
        g.Rotary(&qg, 0, d, hd, pg, fr);
        c.Rotary(&qc, d, d, hd, pc, fr);
        g.Rotary(&qg, d, d, hd, pg, fr);
        sameT(qc, qg, 1e-5f, "rotary encoding");

        Tensor oc, og;
        c.Alloc(&oc, d, 1, n0);
        g.Alloc(&og, d, 1, n0);
        c.Attention(qc, 0, qc, d, qc, 2 * d, heads, hd, &oc, 0);
        g.Attention(qg, 0, qg, d, qg, 2 * d, heads, hd, &og, 0);
        sameT(oc, og, 2e-5f, "self attention");

        Tensor kc, kg, xc2, xg2;
        pts(2 * d, n1, 67, &kc, &kg);
        c.Alloc(&xc2, d, 1, n0);
        g.Alloc(&xg2, d, 1, n0);
        c.Attention(qc, 0, kc, 0, kc, d, heads, hd, &xc2, 0);
        g.Attention(qg, 0, kg, 0, kg, d, heads, hd, &xg2, 0);
        sameT(xc2, xg2, 2e-5f, "cross attention between unequal sets");

        pts(d, n1, 68, &bc, &bg);
        Tensor a2c, a2g;
        pts(d, n0, 69, &a2c, &a2g);
        Tensor sc, sg;
        c.MatMulTN(a2c, bc, 1.0f / 16.0f, &sc);
        g.MatMulTN(a2g, bg, 1.0f / 16.0f, &sg);
        sameT(sc, sg, 2e-5f, "point-set similarity (a^T b)");

        std::vector<int> c0, c1, g0, g1;
        std::vector<float> cp, gp;
        c.DualSoftmaxBest(sc, &c0, &cp, &c1);
        g.DualSoftmaxBest(sg, &g0, &gp, &g1);
        Check(c0 == g0 && c1 == g1 && MaxDiff(cp, gp) < 1e-5f,
              "GPU dual-softmax bests match the CPU");
    }

    // Exact nearest neighbours against a CPU brute force. Sizes straddle the
    // kernel's 64-query groups and its database tiles and slices; dim 64 and
    // 128 are SURF's and SIFT's / DeDoDe's.
    for (int dim : {64, 128}) {
        const int nq = 301, nd = 1777, k = kKnnMax;
        const std::vector<float> q = Random(size_t(nq) * dim, 41), db = Random(size_t(nd) * dim, 42);
        GpuKnn knn(gpu);
        GpuBuffer gq, gd;
        std::vector<int> idx;
        std::vector<float> d2;
        std::string err;
        const bool ran = knn.Upload(q.data(), nq, dim, &gq, &err) &&
                         knn.Upload(db.data(), nd, dim, &gd, &err) &&
                         knn.Search(gq, nq, gd, nd, dim, k, &idx, &d2, &err);
        int wrongIdx = 0;
        float worstD = 0.0f;
        for (int i = 0; ran && i < nq; ++i) {
            std::vector<std::pair<float, int>> all;
            all.resize(size_t(nd));
            for (int j = 0; j < nd; ++j) {
                float s = 0.0f;
                for (int c = 0; c < dim; ++c) {
                    const float e = q[size_t(i) * dim + c] - db[size_t(j) * dim + c];
                    s += e * e;
                }
                all[size_t(j)] = {s, j};
            }
            std::partial_sort(all.begin(), all.begin() + k, all.end());
            for (int j = 0; j < k; ++j) {
                const size_t o = size_t(i) * k + size_t(j);
                worstD = std::max(worstD, std::fabs(d2[o] - all[size_t(j)].first));
                // An index may differ only where two distances tie to rounding.
                if (idx[o] != all[size_t(j)].second &&
                    std::fabs(all[size_t(j)].first - all[size_t(std::min(j + 1, k - 1))].first) > 1e-4f &&
                    (j == 0 || std::fabs(all[size_t(j)].first - all[size_t(j - 1)].first) > 1e-4f))
                    ++wrongIdx;
            }
        }
        char buf[128];
        std::snprintf(buf, sizeof buf, "GPU %d-nearest of %d-float descriptors matches brute force "
                      "(%d wrong, distances within %.1e)", k, dim, wrongIdx, worstD);
        Check(ran && wrongIdx == 0 && worstD < 1e-4f, buf + (ran ? std::string() : ": " + err));
    }

    // Many dispatches recorded, each with its own temporary freed straight
    // after recording: the batch must hold them until it has run.
    {
        Tensor acc;
        g.Upload(std::vector<float>(size_t(h) * w, 0.0f), 1, h, w, &acc);
        for (int i = 0; i < 40; ++i) {
            Tensor t;
            g.Upload(std::vector<float>(size_t(h) * w, 1.0f), 1, h, w, &t);
            g.Accumulate(t, 0, 1, &acc, 0);
        }   // each t is released here, before the flush
        std::vector<float> v;
        g.Download(acc, &v);
        bool all40 = !v.empty();
        for (float f : v) all40 = all40 && f == 40.0f;
        Check(all40, "temporaries freed after recording survive until their batch runs");
    }
}

} // namespace

int main() {
    std::printf("nn: CPU reference\n");
    TestResizeAgainstPyTorch();
    TestSampleAgainstPyTorch();
    TestLayerNormAgainstPyTorch();
    TestConvAgainstDefinition();
    TestWeightFile();
    TestSample();

    ID3D12Device* dev = nullptr;
    if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&dev)))) {
        std::printf("no D3D12 device; skipping the GPU half\n");
    } else {
        ComputeContext gpu;
        if (!gpu.Init(dev)) {
            Check(false, "compute context initialises");
        } else {
            std::printf("nn: GPU against CPU\n");
            TestGpu(&gpu);
        }
        gpu.Shutdown();
        dev->Release();
    }
    std::printf(g_fail ? "FAILURES PRESENT\n" : "all nn checks passed\n");
    return g_fail ? 1 : 0;
}
