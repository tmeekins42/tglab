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

        const float sigma = std::max(0.01f, ctx.ScaledPx(float(m_sigma)));
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
    // Separable on the GPU too, in two passes.
    //
    // It used to be a single O(r^2) dispatch, which forced a sigma ceiling of 4:
    // at sigma 20 the radius is 60, so one pass is ~14,600 fetches per pixel and
    // trips the GPU watchdog. Above the ceiling the stage fell back to the CPU,
    // which is exactly where a blur is most expensive -- the fast path gave up
    // precisely when it was most needed.
    //
    // Two separable passes make it O(2r) like the CPU path: 120 fetches at
    // sigma 20 rather than 14,600, a 120x reduction that removes the reason for
    // any ceiling at all. The Gaussian is separable exactly, so this is the same
    // maths and not an approximation.
    // Zero sigma is genuinely "no blur", so the stage is skipped outright --
    // no allocation, no dispatch, no copy -- rather than convolving with a
    // kernel that sums to the centre tap.
    //
    // The minimum used to be 0.1, so there was no way to express "off" at all.
    bool IsNoOp() const override { return float(m_sigma) <= 0.0f; }

    // The same 3-sigma rule the kernel uses, and the same clamp: a tile needs
    // exactly what a pixel reads, so declaring more wastes work and declaring
    // less puts a seam at every tile edge.
    int ReachPixels() const override {
        return std::clamp(int(std::ceil(std::max(0.01f, float(m_sigma)) * 3.0f)),
                          1, 64);
    }

    bool HasGPU() const override { return true; }
    int  GpuIterations() const override { return 2; }

    const char* GpuSource() const override {
        return R"(
Texture2D<float4>   Src : register(t0);   // ping-pong: source, then h-blurred
RWTexture2D<float4> Dst : register(u0);

cbuffer Params : register(b0) {
    uint Width;
    uint Height;
    uint SigmaBits;
    uint Pass;        // 0 = horizontal, 1 = vertical
};

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;

    float sigma  = max(asfloat(SigmaBits), 0.01);
    // Matches the CPU path's radius, so the two agree rather than merely
    // resembling each other. 64 is the CPU's own clamp, and at O(2r) a 129-tap
    // pass is unremarkable.
    int   radius = clamp(int(ceil(sigma * 3.0)), 1, 64);
    float twoSS  = 2.0 * sigma * sigma;

    float4 sum    = 0.0;
    float  weight = 0.0;

    for (int i = -radius; i <= radius; ++i) {
        int2 p = (Pass == 0) ? int2(int(tid.x) + i, int(tid.y))
                             : int2(int(tid.x), int(tid.y) + i);
        p = clamp(p, int2(0, 0), int2(Width - 1, Height - 1));   // clamp-to-edge
        float w = exp(-float(i * i) / twoSS);
        sum    += Src[p] * w;
        weight += w;
    }
    Dst[tid.xy] = sum / max(weight, 1e-6);
}
)";
    }

    std::vector<uint32_t> GpuConstants(int iteration) const override {
        // Scaled for the proxy, exactly as the CPU path does -- the two must
        // blur by the same fraction of the picture or the preview and the
        // final image disagree.
        const float sigma = GpuScaledPx(float(m_sigma));
        uint32_t bits;
        std::memcpy(&bits, &sigma, sizeof(bits));
        return {bits, uint32_t(iteration)};
    }

private:
    // Every format, not an R32F branch with an RGBA8 fallthrough -- that read a
    // half- or 32-bit-float image as bytes, which is not zeros but
    // plausible-looking nonsense. Values stay in the source's own units, which
    // is what the separable passes and the clamp on write both assume.
    static float Read(const ImageView& v, int x, int y, int c) {
        switch (v.desc.format) {
            case Format::R32F:    return *v.At<float>(x, y);
            case Format::RGBA32F: return v.At<float>(x, y)[c];
            case Format::RGBA16F: return HalfToFloat(v.At<uint16_t>(x, y)[c]);
            case Format::RGBA8:   return float(v.At<uint8_t>(x, y)[c]);
            default:              return 0.0f;
        }
    }

    static void Write(ImageView& v, int x, int y, const float* acc, int ch) {
        switch (v.desc.format) {
            case Format::R32F:
                *v.At<float>(x, y) = acc[0];
                return;
            case Format::RGBA32F: {
                float* p = v.At<float>(x, y);
                for (int c = 0; c < ch; ++c) p[c] = acc[c];
                // A 1-channel accumulator carries no alpha; keep it opaque.
                if (ch < 4) p[3] = 1.0f;
                return;
            }
            case Format::RGBA16F: {
                uint16_t* p = v.At<uint16_t>(x, y);
                for (int c = 0; c < ch; ++c) p[c] = FloatToHalf(acc[c]);
                if (ch < 4) p[3] = FloatToHalf(1.0f);
                return;
            }
            case Format::RGBA8: {
                uint8_t* p = v.At<uint8_t>(x, y);
                for (int c = 0; c < ch; ++c)
                    p[c] = uint8_t(std::clamp(acc[c], 0.0f, 255.0f));
                if (ch < 4) p[3] = 255;
                return;
            }
            default:
                return;
        }
    }

    Param<float> m_sigma{
        this, "sigma", 2.0f, 0.0f, 20.0f,
        {.help = "Width of the Gaussian, in pixels. Higher blurs more. "
                 "The kernel reaches about 3x this far.",
         .step = 0.1, .softMin = 0.0, .softMax = 5.0}};

    // Reused across runs to avoid reallocating on every slider drag.
    std::vector<float> m_scratch;
    std::vector<float> m_input;
};

REGISTER_ALGORITHM(GaussianBlur);

} // namespace tglab
