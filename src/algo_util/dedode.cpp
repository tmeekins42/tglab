#include "dedode.h"

#include <algorithm>
#include <cmath>

#include "color.h"
#include "features.h"
#include "pixel_buffer.h"

namespace tglab {

namespace {
constexpr const char* kScale[4] = {"8", "4", "2", "1"};
} // namespace

bool DedodeNet::LoadRefiner(const nn::WeightFile& f, const std::string& name, Refiner* r,
                            std::string* err) {
    if (!r->in1.From(f, name + ".block1.0", err) || !r->in2.From(f, name + ".block1.3", err) ||
        !r->out.From(f, name + ".out_conv", err))
        return false;
    r->dw.clear();
    r->pw.clear();
    for (int i = 0;; ++i) {
        const std::string hb = name + ".hidden_blocks." + std::to_string(i);
        if (!f.Find(hb + ".0.weight")) break;
        nn::Conv dw, pw;
        if (!dw.From(f, hb + ".0", err) || !pw.From(f, hb + ".3", err)) return false;
        r->dw.push_back(std::move(dw));
        r->pw.push_back(std::move(pw));
    }
    return true;
}

bool DedodeNet::Load(const nn::WeightFile& f, int outDim, nn::Interp accumulate,
                     std::string* err) {
    m_blocks.clear();
    // The encoder's conv layers, in order; a gap of four indices is a pool.
    std::vector<int> idx;
    for (int i = 0; i < 64; ++i) {
        const nn::Weight* w = f.Find("encoder.layers." + std::to_string(i) + ".weight");
        if (w && w->shape.size() == 4) idx.push_back(i);
    }
    if (idx.empty()) { *err = "no encoder convolutions in the weight file"; return false; }
    std::vector<std::vector<nn::Conv>> blocks(1);
    for (size_t k = 0; k < idx.size(); ++k) {
        if (k > 0 && idx[k] - idx[k - 1] == 4) blocks.emplace_back();
        nn::Conv c;
        if (!c.From(f, "encoder.layers." + std::to_string(idx[k]), err)) return false;
        blocks.back().push_back(std::move(c));
    }
    if (blocks.size() != 4) {
        *err = "expected a four-scale encoder, found " + std::to_string(blocks.size());
        return false;
    }
    for (int s = 0; s < 4; ++s)
        if (!LoadRefiner(f, std::string("decoder.layers.") + kScale[s], &m_dec[s], err))
            return false;
    if (m_dec[0].out.cout <= outDim) { *err = "decoder has no context channels"; return false; }
    m_blocks = std::move(blocks);
    m_outDim = outDim;
    m_accumulate = accumulate;
    return true;
}

// ConvRefiner.forward:
//   x0  = conv1x1(relu(bn(conv1x1(in))))          block1, BatchNorm folded
//   x   = N x [ conv1x1(relu(bn(dw5x5(x)))) ]
//   out = conv1x1((x + x0) / 1.4)
bool DedodeNet::RunRefiner(nn::Engine& e, const Refiner& r, const nn::Tensor& in,
                           nn::Tensor* out) const {
    const int hidden = r.in1.cout;
    nn::Tensor a, x0, b, sum;
    if (!e.Alloc(&a, hidden, in.h, in.w) || !e.Conv2d(in, r.in1, true, &a)) return false;
    if (!e.Alloc(&x0, hidden, in.h, in.w) || !e.Conv2d(a, r.in2, false, &x0)) return false;
    if (!e.Alloc(&b, hidden, in.h, in.w)) return false;
    const nn::Tensor* x = &x0;
    for (size_t i = 0; i < r.dw.size(); ++i) {
        if (!e.Conv2d(*x, r.dw[i], true, &b)) return false;
        if (!e.Conv2d(b, r.pw[i], false, &a)) return false;
        x = &a;
    }
    if (!e.AddScaled(*x, x0, 1.0f / 1.4f, &sum)) return false;
    return e.Alloc(out, r.out.cout, in.h, in.w) && e.Conv2d(sum, r.out, false, out);
}

bool DedodeNet::Run(nn::Engine& e, const nn::Tensor& input, nn::Tensor* out, std::string* err,
                    NnTaps* taps) const {
    auto fail = [&] { *err = e.Error(); return false; };
    if (!Loaded()) { *err = "no weights loaded"; return false; }
    if (input.h % 8 || input.w % 8 || input.h < 16 || input.w < 16) {
        *err = "the network input must be a multiple of 8 on each side";
        return false;
    }
    auto tap = [&](const std::string& name, const nn::Tensor& t) {
        return !taps || e.Download(t, &(*taps)[name]);
    };

    // --- encoder -------------------------------------------------------------
    nn::Tensor feat[4], x, t;
    const nn::Tensor* cur = &input;
    for (int s = 0; s < 4; ++s) {
        if (s > 0 && !e.MaxPool2(feat[s - 1], &x)) return fail();
        if (s > 0) cur = &x;
        const auto& blk = m_blocks[size_t(s)];
        for (size_t k = 0; k < blk.size(); ++k) {
            nn::Tensor* dst = (k + 1 == blk.size()) ? &feat[s] : &t;
            nn::Tensor next;
            if (!e.Alloc(&next, blk[k].cout, cur->h, cur->w) || !e.Conv2d(*cur, blk[k], true, &next))
                return fail();
            *dst = std::move(next);
            cur = dst;
        }
        if (!tap("feat" + std::to_string(1 << s), feat[s])) return fail();
    }

    // --- decoder: coarse to fine ---------------------------------------------
    nn::Tensor acc, cat, ref, up;
    for (int s = 0; s < 4; ++s) {
        const nn::Tensor& f = feat[3 - s];
        const nn::Tensor* in = &f;
        if (s > 0) {
            const int ctx = ref.c - m_outDim;
            if (!e.Alloc(&cat, f.c + ctx, f.h, f.w) || !e.Copy(f, 0, f.c, &cat, 0) ||
                !e.Resize(ref, m_outDim, ctx, nn::Interp::Bilinear, &cat, f.c))
                return fail();
            in = &cat;
        }
        if (!RunRefiner(e, m_dec[s], *in, &ref)) return fail();
        if (s == 0) {
            if (!e.Alloc(&acc, m_outDim, f.h, f.w) || !e.Copy(ref, 0, m_outDim, &acc, 0))
                return fail();
        } else if (!e.Accumulate(ref, 0, m_outDim, &acc, 0)) {
            return fail();
        }
        if (taps) {
            nn::Tensor d;
            if (!e.Alloc(&d, m_outDim, f.h, f.w) || !e.Copy(ref, 0, m_outDim, &d, 0) ||
                !tap(std::string("delta") + kScale[s], d))
                return fail();
        }
        if (s < 3) {
            const nn::Tensor& next = feat[2 - s];
            if (!e.Alloc(&up, m_outDim, next.h, next.w) ||
                !e.Resize(acc, 0, m_outDim, m_accumulate, &up, 0))
                return fail();
            std::swap(acc, up);
        }
    }
    *out = std::move(acc);
    return true;
}

bool DedodeDescriptor::Load(const nn::WeightFile& f, std::string* err) {
    // The output channels that are descriptor rather than context: the
    // full-resolution refiner's output less its one context channel.
    const nn::Weight* w = f.Find("decoder.layers.1.out_conv.weight");
    if (!w || w->shape.empty()) { *err = "not a DeDoDe descriptor weight file"; return false; }
    return m_net.Load(f, w->shape[0] - 1, nn::Interp::Bilinear, err);
}

bool DedodeDescriptor::Describe(nn::Engine& e, const std::vector<float>& rgb, int h, int w,
                                const std::vector<float>& xy, std::vector<float>* desc,
                                std::string* err, NnTaps* taps) const {
    nn::Tensor in, map, pts;
    if (!e.Upload(rgb, 3, h, w, &in)) { *err = e.Error(); return false; }
    if (!m_net.Run(e, in, &map, err, taps)) return false;
    if (taps && !e.Download(map, &(*taps)["descmap"])) { *err = e.Error(); return false; }
    if (xy.empty()) { desc->clear(); return true; }
    if (!e.SampleBilinear(map, xy, &pts) || !e.Download(pts, desc)) {
        *err = e.Error();
        return false;
    }
    return true;
}

// --- shared preprocessing ------------------------------------------------------

namespace {

// PIL's bicubic_filter, a = -0.5 (Libimaging/Resample.c).
double PilCubic(double x) {
    const double a = -0.5;
    x = std::fabs(x);
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    return 0.0;
}

// PIL's precompute_coeffs: per output index, the first input index and the
// normalised weights over the (scaled) support.
struct Taps {
    std::vector<int>    first, count;
    std::vector<double> w;   // count[i] weights from offset i * maxTaps
    int maxTaps = 0;
};

Taps PilTaps(int in, int out) {
    Taps t;
    const double scale = double(in) / double(out);
    const double fs = std::max(scale, 1.0);
    const double support = 2.0 * fs;
    t.maxTaps = int(std::ceil(support)) * 2 + 1;
    t.first.resize(size_t(out));
    t.count.resize(size_t(out));
    t.w.assign(size_t(out) * size_t(t.maxTaps), 0.0);
    for (int i = 0; i < out; ++i) {
        const double centre = (double(i) + 0.5) * scale;
        int lo = int(centre - support + 0.5), hi = int(centre + support + 0.5);
        lo = std::max(lo, 0);
        hi = std::min(hi, in);
        const int n = std::min(hi - lo, t.maxTaps);
        double sum = 0.0;
        double* w = &t.w[size_t(i) * size_t(t.maxTaps)];
        for (int j = 0; j < n; ++j) sum += (w[j] = PilCubic((double(j + lo) - centre + 0.5) / fs));
        if (sum != 0.0)
            for (int j = 0; j < n; ++j) w[j] /= sum;
        t.first[size_t(i)] = lo;
        t.count[size_t(i)] = n;
    }
    return t;
}

} // namespace

std::vector<float> ResizePil(const std::vector<float>& src, int channels, int w, int h,
                             int nw, int nh) {
    // Horizontal pass, then vertical, as PIL does. PIL rounds and clamps to
    // 8 bits between the passes for an 8-bit image; this keeps floats, a
    // difference far below what the network can see.
    const Taps tx = PilTaps(w, nw), ty = PilTaps(h, nh);
    std::vector<float> mid(size_t(channels) * size_t(h) * size_t(nw));
    std::vector<float> out(size_t(channels) * size_t(nh) * size_t(nw));
    for (int c = 0; c < channels; ++c) {
        const float* s = &src[size_t(c) * size_t(w) * size_t(h)];
        float* m = &mid[size_t(c) * size_t(nw) * size_t(h)];
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < nw; ++x) {
                const double* wt = &tx.w[size_t(x) * size_t(tx.maxTaps)];
                double v = 0.0;
                for (int j = 0; j < tx.count[size_t(x)]; ++j)
                    v += wt[j] * s[size_t(y) * size_t(w) + size_t(tx.first[size_t(x)] + j)];
                m[size_t(y) * size_t(nw) + size_t(x)] = float(v);
            }
        float* o = &out[size_t(c) * size_t(nw) * size_t(nh)];
        for (int y = 0; y < nh; ++y) {
            const double* wt = &ty.w[size_t(y) * size_t(ty.maxTaps)];
            for (int x = 0; x < nw; ++x) {
                double v = 0.0;
                for (int j = 0; j < ty.count[size_t(y)]; ++j)
                    v += wt[j] * m[size_t(ty.first[size_t(y)] + j) * size_t(nw) + size_t(x)];
                o[size_t(y) * size_t(nw) + size_t(x)] = std::clamp(float(v), 0.0f, 1.0f);
            }
        }
    }
    return out;
}

std::vector<float> NetworkRgb(const PixelBuffer& in, bool linear) {
    const int w = in.Width(), h = in.Height(), ch = in.Channels();
    const float scale = in.ValueScale();
    const size_t P = size_t(w) * size_t(h);
    std::vector<float> rgb(3 * P);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const float* p = in.At(x, y);
            const size_t i = size_t(y) * size_t(w) + size_t(x);
            for (int c = 0; c < 3; ++c) rgb[size_t(c) * P + i] = p[ch >= 3 ? c : 0] / scale;
        }
    if (linear) {
        std::vector<float> luma(P);
        for (size_t i = 0; i < P; ++i) luma[i] = Luma(rgb[i], rgb[P + i], rgb[2 * P + i]);
        const float level = std::max(Percentile99(luma), 1e-6f);
        for (float& v : rgb) v = LinearToSrgb(std::clamp(v / level, 0.0f, 1.0f));
    } else {
        for (float& v : rgb) v = std::clamp(v, 0.0f, 1.0f);
    }
    return rgb;
}

} // namespace tglab
