// gaussian_blur — separable Gaussian.
//
// Separable: two 1D passes instead of one 2D kernel, so cost is O(2r) per
// pixel rather than O(r^2). Also the natural first algorithm to port to a
// compute shader in M3.
#include <algorithm>
#include <cmath>
#include <vector>

#include "../core/algorithm.h"

namespace tglab {

class GaussianBlur : public AlgorithmBase {
public:
    const char* Name()     const override { return "gaussian_blur"; }
    const char* Category() const override { return "filter"; }

    PortList Inputs()  const override { return {{"src", DataType::Image, FormatSpec::Any}}; }
    PortList Outputs() const override { return {{"out", DataType::Image, FormatSpec::SameAsInput}}; }

    void RunCPU(RunCtx& ctx) override {
        const ImageView src = ctx.In(0);
        ImageView       dst = ctx.Out(0);
        if (!src.Valid() || !dst.Valid()) return;

        const float sigma = std::max(0.01f, float(m_sigma));
        // Radius from sigma: 3 sigma captures ~99.7% of the kernel's weight.
        const int radius = std::clamp(int(std::ceil(sigma * 3.0f)), 1, 64);

        std::vector<float> k(size_t(radius) * 2 + 1);
        float sum = 0.0f;
        for (int i = -radius; i <= radius; ++i) {
            const float v = std::exp(-float(i * i) / (2.0f * sigma * sigma));
            k[size_t(i + radius)] = v;
            sum += v;
        }
        for (float& v : k) v /= sum;   // normalise so brightness is preserved

        const int w = src.desc.width;
        const int h = src.desc.height;
        const int ch = (src.desc.format == Format::R32F) ? 1 : 4;

        // Horizontal pass into a scratch buffer, then vertical into dst.
        m_scratch.assign(size_t(w) * size_t(h) * size_t(ch), 0.0f);

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                float acc[4] = {0, 0, 0, 0};
                for (int i = -radius; i <= radius; ++i) {
                    const int sx = std::clamp(x + i, 0, w - 1);   // clamp-to-edge
                    const float wgt = k[size_t(i + radius)];
                    for (int c = 0; c < ch; ++c) acc[c] += Read(src, sx, y, c) * wgt;
                }
                for (int c = 0; c < ch; ++c)
                    m_scratch[(size_t(y) * size_t(w) + size_t(x)) * size_t(ch) + size_t(c)] = acc[c];
            }
        }

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                float acc[4] = {0, 0, 0, 0};
                for (int i = -radius; i <= radius; ++i) {
                    const int sy = std::clamp(y + i, 0, h - 1);
                    const float wgt = k[size_t(i + radius)];
                    for (int c = 0; c < ch; ++c)
                        acc[c] += m_scratch[(size_t(sy) * size_t(w) + size_t(x)) * size_t(ch) + size_t(c)] * wgt;
                }
                Write(dst, x, y, acc, ch);
            }
        }
    }

private:
    static float Read(const ImageView& v, int x, int y, int c) {
        if (v.desc.format == Format::R32F) return *v.At<float>(x, y);
        return float(v.At<uint8_t>(x, y)[c]);
    }

    static void Write(ImageView& v, int x, int y, const float* acc, int ch) {
        if (v.desc.format == Format::R32F) {
            *v.At<float>(x, y) = acc[0];
            return;
        }
        uint8_t* p = v.At<uint8_t>(x, y);
        for (int c = 0; c < ch; ++c)
            p[c] = uint8_t(std::clamp(acc[c], 0.0f, 255.0f));
    }

    Param<float> m_sigma{this, "sigma", 2.0f, 0.1f, 20.0f};

    // Reused across runs to avoid reallocating on every slider drag.
    std::vector<float> m_scratch;
};

REGISTER_ALGORITHM(GaussianBlur);

} // namespace tglab
