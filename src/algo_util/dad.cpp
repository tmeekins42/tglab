#include "dad.h"

#include <algorithm>
#include <cmath>

namespace tglab {

bool DadNet::Load(const nn::WeightFile& f, std::string* err) {
    // One output channel, the score, summed across scales by bicubic
    // upsampling -- see dedode.h.
    if (!m_net.Load(f, 1, nn::Interp::Bicubic, err)) {
        *err = "dad: " + *err;
        return false;
    }
    return true;
}

bool DadNet::Logits(nn::Engine& e, const std::vector<float>& input, int h, int w,
                    std::vector<float>* logits, std::string* err, NnTaps* taps) const {
    nn::Tensor in, out;
    if (!e.Upload(input, 3, h, w, &in) || !m_net.Run(e, in, &out, err, taps) ||
        !e.Download(out, logits)) {
        if (err->empty()) *err = e.Error();
        *err = "dad: " + *err;
        return false;
    }
    return true;
}

std::vector<DadPoint> DadSample(const std::vector<float>& logits, int h, int w, int n) {
    std::vector<DadPoint> pts;
    const size_t P = size_t(h) * size_t(w);
    if (logits.size() != P || n <= 0) return pts;

    // Softmax over every pixel. Only the ranking and the reported share use
    // it, but the share is what the published code returns, so match it.
    const float mx = *std::max_element(logits.begin(), logits.end());
    std::vector<float> p(P);
    double sum = 0.0;
    for (size_t i = 0; i < P; ++i) sum += (p[i] = std::exp(logits[i] - mx));
    const float inv = float(1.0 / sum);
    for (float& v : p) v *= inv;

    // 3x3 non-maximum suppression: a pixel survives when it equals the
    // maximum of its neighbourhood (ties survive together, as in PyTorch).
    std::vector<std::pair<float, uint32_t>> peaks;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const float v = p[size_t(y) * size_t(w) + size_t(x)];
            if (v <= 0.0f) continue;
            bool isMax = true;
            for (int dy = -1; dy <= 1 && isMax; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    const int yy = y + dy, xx = x + dx;
                    if (yy < 0 || yy >= h || xx < 0 || xx >= w) continue;
                    if (p[size_t(yy) * size_t(w) + size_t(xx)] > v) { isMax = false; break; }
                }
            if (isMax) peaks.push_back({v, uint32_t(y * w + x)});
        }
    const size_t k = std::min(peaks.size(), size_t(n));
    std::partial_sort(peaks.begin(), peaks.begin() + long(k), peaks.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

    // Sub-pixel: the softmax of the 3x3 logits around the peak, at
    // temperature 0.5, weights the offsets -1, 0, +1. Outside the image the
    // logits read as zero (the published code unfolds with zero padding).
    pts.reserve(k);
    for (size_t i = 0; i < k; ++i) {
        const int x = int(peaks[i].second % uint32_t(w)), y = int(peaks[i].second / uint32_t(w));
        float s[9], smax = -1e30f;
        for (int j = 0; j < 9; ++j) {
            const int yy = y + j / 3 - 1, xx = x + j % 3 - 1;
            s[j] = (yy < 0 || yy >= h || xx < 0 || xx >= w)
                       ? 0.0f : logits[size_t(yy) * size_t(w) + size_t(xx)] / 0.5f;
            smax = std::max(smax, s[j]);
        }
        float tot = 0.0f, ox = 0.0f, oy = 0.0f;
        for (int j = 0; j < 9; ++j) {
            const float e = std::exp(s[j] - smax);
            tot += e;
            ox += e * float(j % 3 - 1);
            oy += e * float(j / 3 - 1);
        }
        DadPoint d;
        d.x = float(x) + 0.5f + ox / tot;
        d.y = float(y) + 0.5f + oy / tot;
        d.prob = peaks[i].first;
        pts.push_back(d);
    }
    return pts;
}

void DadNetworkSize(int w, int h, int longSide, int* nw, int* nh) {
    const double s = double(longSide) / double(std::max(w, h));
    *nw = int(s * double(w)) / 8 * 8;
    *nh = int(s * double(h)) / 8 * 8;
}

void DadNormalise(std::vector<float>* rgb) {
    static const float kMean[3] = {0.485f, 0.456f, 0.406f};
    static const float kStd[3]  = {0.229f, 0.224f, 0.225f};
    const size_t P = rgb->size() / 3;
    for (int c = 0; c < 3; ++c)
        for (size_t i = 0; i < P; ++i)
            (*rgb)[size_t(c) * P + i] = ((*rgb)[size_t(c) * P + i] - kMean[c]) / kStd[c];
}

} // namespace tglab
