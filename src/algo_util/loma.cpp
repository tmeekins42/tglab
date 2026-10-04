#include "loma.h"

#include <algorithm>
#include <cmath>

namespace tglab {

namespace {

// Points-major (n x d) to channel-major (d x n), and back.
std::vector<float> Transpose(const std::vector<float>& v, size_t rows, size_t cols) {
    std::vector<float> t(v.size());
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c) t[c * rows + r] = v[r * cols + c];
    return t;
}

} // namespace

bool LomaMatcher::LoadFfn(const nn::WeightFile& f, const std::string& name, Ffn* ffn,
                          std::string* err) const {
    return ffn->l0.FromLinear(f, name + ".0", err) && ffn->gamma.From(f, name + ".1.weight", err) &&
           ffn->beta.From(f, name + ".1.bias", err) && ffn->l3.FromLinear(f, name + ".3", err);
}

bool LomaMatcher::Load(const nn::WeightFile& f, std::string* err) {
    m_layers.clear();
    m_assign.clear();
    if (!m_inProj.FromLinear(f, "input_proj", err) || !m_freq.From(f, "posenc.Wr.weight", err))
        return false;
    const nn::Weight* wr = f.Find("posenc.Wr.weight");
    m_dim = m_inProj.cout;
    m_headDim = wr->shape.size() == 2 ? 2 * wr->shape[0] : 0;
    if (m_headDim == 0 || wr->shape[1] != 2 || m_dim % m_headDim) {
        *err = "unexpected positional encoding shape";
        return false;
    }
    m_heads = m_dim / m_headDim;

    for (int i = 0;; ++i) {
        const std::string t = "transformers." + std::to_string(i);
        if (!f.Find(t + ".self_attn.Wqkv.weight")) break;
        Layer L;
        if (!L.qkv.FromLinear(f, t + ".self_attn.Wqkv", err) ||
            !L.outProj.FromLinear(f, t + ".self_attn.out_proj", err) ||
            !LoadFfn(f, t + ".self_attn.ffn", &L.selfFfn, err) ||
            !L.toQk.FromLinear(f, t + ".cross_attn.to_qk", err) ||
            !L.toV.FromLinear(f, t + ".cross_attn.to_v", err) ||
            !L.toOut.FromLinear(f, t + ".cross_attn.to_out", err) ||
            !LoadFfn(f, t + ".cross_attn.ffn", &L.crossFfn, err))
            return false;

        // Wqkv's output is laid out [head][channel][q,k,v] -- the PyTorch
        // code unflattens it to (heads, -1, 3) -- so q, k and v interleave.
        // Reordered once here to [q | k | v], each [head][channel], so that
        // they are three contiguous channel ranges of one tensor.
        const std::vector<float> srcW = L.qkv.w, srcB = L.qkv.b;
        for (int j = 0; j < 3; ++j)
            for (int h = 0; h < m_heads; ++h)
                for (int d = 0; d < m_headDim; ++d) {
                    const size_t from = size_t((h * m_headDim + d) * 3 + j);
                    const size_t to = size_t(j * m_dim + h * m_headDim + d);
                    std::copy_n(&srcW[from * size_t(m_dim)], size_t(m_dim), &L.qkv.w[to * size_t(m_dim)]);
                    L.qkv.b[to] = srcB[from];
                }
        m_layers.push_back(std::move(L));

        nn::Conv a;
        if (!a.FromLinear(f, "log_assignment." + std::to_string(i) + ".final_proj", err)) return false;
        m_assign.push_back(std::move(a));
    }
    if (m_layers.empty()) { *err = "no transformer layers in the weight file"; return false; }
    return true;
}

// x + Linear(GELU(LayerNorm(Linear([x, msgProj(message)])))).
bool LomaMatcher::RunFfn(nn::Engine& e, const Ffn& ffn, const nn::Tensor& x,
                         const nn::Tensor& message, const nn::Conv& msgProj,
                         nn::Tensor* out) const {
    nn::Tensor cat, h, hn, y;
    const int n = x.w;
    return e.Alloc(&cat, 2 * m_dim, 1, n) && e.Copy(x, 0, m_dim, &cat, 0) &&
           e.Conv2d(message, msgProj, false, &cat, m_dim) &&
           e.Alloc(&h, ffn.l0.cout, 1, n) && e.Conv2d(cat, ffn.l0, false, &h) &&
           e.LayerNormGelu(h, ffn.gamma, ffn.beta, &hn) &&
           e.Alloc(&y, m_dim, 1, n) && e.Conv2d(hn, ffn.l3, false, &y) &&
           e.AddScaled(x, y, 1.0f, out);
}

bool LomaMatcher::SelfBlock(nn::Engine& e, const Layer& L, const nn::Tensor& x,
                            const nn::Tensor& pos, nn::Tensor* out) const {
    const int n = x.w;
    nn::Tensor qkv, ctx;
    return e.Alloc(&qkv, 3 * m_dim, 1, n) && e.Conv2d(x, L.qkv, false, &qkv) &&
           e.Rotary(&qkv, 0, m_dim, m_headDim, pos, m_freq) &&
           e.Rotary(&qkv, m_dim, m_dim, m_headDim, pos, m_freq) &&
           e.Alloc(&ctx, m_dim, 1, n) &&
           e.Attention(qkv, 0, qkv, m_dim, qkv, 2 * m_dim, m_heads, m_headDim, &ctx, 0) &&
           RunFfn(e, L.selfFfn, x, ctx, L.outProj, out);
}

bool LomaMatcher::CrossBlock(nn::Engine& e, const Layer& L, const nn::Tensor& x0,
                             const nn::Tensor& x1, nn::Tensor* out0, nn::Tensor* out1) const {
    const int n0 = x0.w, n1 = x1.w;
    nn::Tensor qk0, qk1, v0, v1, m0, m1;
    return e.Alloc(&qk0, m_dim, 1, n0) && e.Conv2d(x0, L.toQk, false, &qk0) &&
           e.Alloc(&qk1, m_dim, 1, n1) && e.Conv2d(x1, L.toQk, false, &qk1) &&
           e.Alloc(&v0, m_dim, 1, n0) && e.Conv2d(x0, L.toV, false, &v0) &&
           e.Alloc(&v1, m_dim, 1, n1) && e.Conv2d(x1, L.toV, false, &v1) &&
           e.Alloc(&m0, m_dim, 1, n0) &&
           e.Attention(qk0, 0, qk1, 0, v1, 0, m_heads, m_headDim, &m0, 0) &&
           e.Alloc(&m1, m_dim, 1, n1) &&
           e.Attention(qk1, 0, qk0, 0, v0, 0, m_heads, m_headDim, &m1, 0) &&
           RunFfn(e, L.crossFfn, x0, m0, L.toOut, out0) &&
           RunFfn(e, L.crossFfn, x1, m1, L.toOut, out1);
}

bool LomaMatcher::Match(nn::Engine& e, const std::vector<float>& kp0,
                        const std::vector<float>& desc0, const std::vector<float>& kp1,
                        const std::vector<float>& desc1, int layers, float threshold,
                        std::vector<Pair>* out, std::string* err, NnTaps* taps) const {
    out->clear();
    auto fail = [&] { *err = "loma: " + e.Error(); return false; };
    if (!Loaded()) { *err = "loma: no weights loaded"; return false; }
    const int din = InputDim();
    const int n0 = int(kp0.size() / 2), n1 = int(kp1.size() / 2);
    if (desc0.size() != size_t(n0) * size_t(din) || desc1.size() != size_t(n1) * size_t(din)) {
        *err = "loma: expected " + std::to_string(din) + "-float descriptors, one per keypoint";
        return false;
    }
    if (n0 == 0 || n1 == 0) return true;
    layers = std::clamp(layers, 1, Layers());

    auto tap = [&](const std::string& name, const nn::Tensor& t) {
        if (!taps) return true;
        std::vector<float> v;
        if (!e.Download(t, &v)) return false;
        (*taps)[name] = Transpose(v, size_t(t.c), size_t(t.w));   // back to points-major
        return true;
    };

    nn::Tensor pos0, pos1, d0, d1, x0, x1;
    if (!e.Upload(Transpose(kp0, size_t(n0), 2), 2, 1, n0, &pos0) ||
        !e.Upload(Transpose(kp1, size_t(n1), 2), 2, 1, n1, &pos1) ||
        !e.Upload(Transpose(desc0, size_t(n0), size_t(din)), din, 1, n0, &d0) ||
        !e.Upload(Transpose(desc1, size_t(n1), size_t(din)), din, 1, n1, &d1) ||
        !e.Alloc(&x0, m_dim, 1, n0) || !e.Conv2d(d0, m_inProj, false, &x0) ||
        !e.Alloc(&x1, m_dim, 1, n1) || !e.Conv2d(d1, m_inProj, false, &x1))
        return fail();
    if (!tap("proj0", x0) || !tap("proj1", x1)) return fail();

    for (int i = 0; i < layers; ++i) {
        const Layer& L = m_layers[size_t(i)];
        nn::Tensor s0, s1, c0, c1;
        if (!SelfBlock(e, L, x0, pos0, &s0) || !SelfBlock(e, L, x1, pos1, &s1) ||
            !CrossBlock(e, L, s0, s1, &c0, &c1))
            return fail();
        x0 = std::move(c0);
        x1 = std::move(c1);
        if (!tap("layer" + std::to_string(i) + "_0", x0) ||
            !tap("layer" + std::to_string(i) + "_1", x1))
            return fail();
    }

    // The assignment of the last layer run: both sides projected, each
    // scaled by dim^-1/4 -- together 1/sqrt(dim) on the similarity.
    const nn::Conv& fp = m_assign[size_t(layers - 1)];
    nn::Tensor a0, a1, sim;
    std::vector<int> best0, best1;
    std::vector<float> p0;
    if (!e.Alloc(&a0, m_dim, 1, n0) || !e.Conv2d(x0, fp, false, &a0) ||
        !e.Alloc(&a1, m_dim, 1, n1) || !e.Conv2d(x1, fp, false, &a1) ||
        !e.MatMulTN(a0, a1, 1.0f / std::sqrt(float(m_dim)), &sim))
        return fail();
    if (taps && !e.Download(sim, &(*taps)["sim"])) return fail();
    if (!e.DualSoftmaxBest(sim, &best0, &p0, &best1)) return fail();

    // filter_matches: mutual best, and confident.
    for (int i = 0; i < n0; ++i) {
        const int j = best0[size_t(i)];
        if (j < 0 || j >= n1 || best1[size_t(j)] != i || !(p0[size_t(i)] > threshold)) continue;
        out->push_back({i, j, p0[size_t(i)]});
    }
    return true;
}

} // namespace tglab
