// gaussian_blur — separable Gaussian.
//
// Separable: two 1D passes instead of one 2D kernel, so cost is O(2r) per
// pixel rather than O(r^2). Also the natural first algorithm to port to a
// compute shader in M3.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "../../core/algorithm.h"

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

        // Unpack to a flat float buffer once. Read() carries a per-call format
        // branch and the separable passes touch it (2r+1) times per pixel per
        // channel -- at 8 MP with sigma 8 that is billions of calls, and it is
        // where the time actually went.
        m_input.assign(size_t(w) * size_t(h) * size_t(ch), 0.0f);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                for (int c = 0; c < ch; ++c)
                    m_input[(size_t(y) * size_t(w) + size_t(x)) * size_t(ch) + size_t(c)] =
                        Read(src, x, y, c);

        // Horizontal pass into a scratch buffer, then vertical into dst.
        m_scratch.assign(size_t(w) * size_t(h) * size_t(ch), 0.0f);

        for (int y = 0; y < h; ++y) {
            const size_t rowBase = size_t(y) * size_t(w);
            for (int x = 0; x < w; ++x) {
                float acc[4] = {0, 0, 0, 0};
                // The interior needs no clamping, which lets the compiler
                // vectorise the common case; edges take the slow path.
                if (x >= radius && x + radius < w) {
                    const float* base = &m_input[(rowBase + size_t(x - radius)) * size_t(ch)];
                    for (int i = 0; i <= radius * 2; ++i) {
                        const float wgt = k[size_t(i)];
                        const float* s = base + size_t(i) * size_t(ch);
                        for (int c = 0; c < ch; ++c) acc[c] += s[c] * wgt;
                    }
                } else {
                    for (int i = -radius; i <= radius; ++i) {
                        const int sx = std::clamp(x + i, 0, w - 1);   // clamp-to-edge
                        const float wgt = k[size_t(i + radius)];
                        const float* s = &m_input[(rowBase + size_t(sx)) * size_t(ch)];
                        for (int c = 0; c < ch; ++c) acc[c] += s[c] * wgt;
                    }
                }
                float* d = &m_scratch[(rowBase + size_t(x)) * size_t(ch)];
                for (int c = 0; c < ch; ++c) d[c] = acc[c];
            }
        }

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                float acc[4] = {0, 0, 0, 0};
                if (y >= radius && y + radius < h) {
                    for (int i = 0; i <= radius * 2; ++i) {
                        const float wgt = k[size_t(i)];
                        const float* s = &m_scratch[(size_t(y - radius + i) * size_t(w) + size_t(x)) * size_t(ch)];
                        for (int c = 0; c < ch; ++c) acc[c] += s[c] * wgt;
                    }
                } else {
                    for (int i = -radius; i <= radius; ++i) {
                        const int sy = std::clamp(y + i, 0, h - 1);
                        const float wgt = k[size_t(i + radius)];
                        const float* s = &m_scratch[(size_t(sy) * size_t(w) + size_t(x)) * size_t(ch)];
                        for (int c = 0; c < ch; ++c) acc[c] += s[c] * wgt;
                    }
                }
                Write(dst, x, y, acc, ch);
            }
        }
    }

    // --- GPU implementation -------------------------------------------------
    // Single-pass here rather than separable: the two-pass version needs an
    // intermediate target, which the current one-dispatch-per-algorithm model
    // does not express. Still far faster than the CPU path, and it makes the
    // CPU/GPU comparison in M4 an honest one (same maths, different hardware).
    // Only when the radius stays small enough for a single O(r^2) dispatch.
    // Beyond that the pipeline falls back to the separable CPU path, which is
    // both correct and (at those radii) not much slower -- and crucially does
    // not trip the GPU watchdog, which takes the whole device down.
    static constexpr float kMaxGpuSigma = 4.0f;
    bool HasGPU() const override { return float(m_sigma) <= kMaxGpuSigma; }

    const char* GpuSource() const override {
        return R"(
Texture2D<float4>   Src : register(t0);
RWTexture2D<float4> Dst : register(u0);

cbuffer Params : register(b0) {
    uint Width;
    uint Height;
    uint SigmaBits;
};

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;

    float sigma  = max(asfloat(SigmaBits), 0.01);
    // Matches HasGPU()'s ceiling. A single O(r^2) pass at radius 60 would be
    // ~14,600 fetches per pixel -- billions per dispatch on a 512x512 image --
    // which trips the GPU watchdog (TDR) and removes the device.
    int   radius = min(int(ceil(sigma * 3.0)), 12);
    float twoSS  = 2.0 * sigma * sigma;

    float4 sum    = 0.0;
    float  weight = 0.0;

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            int2 p = int2(tid.xy) + int2(dx, dy);
            p = clamp(p, int2(0, 0), int2(Width - 1, Height - 1));   // clamp-to-edge
            float w = exp(-float(dx * dx + dy * dy) / twoSS);
            sum    += Src[p] * w;
            weight += w;
        }
    }
    Dst[tid.xy] = sum / max(weight, 1e-6);
}
)";
    }

    std::vector<uint32_t> GpuConstants() const override {
        const float sigma = m_sigma;
        uint32_t bits;
        std::memcpy(&bits, &sigma, sizeof(bits));
        return {bits};
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
    std::vector<float> m_input;
};

REGISTER_ALGORITHM(GaussianBlur);

} // namespace tglab
