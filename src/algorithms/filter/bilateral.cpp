// bilateral — Gaussian in space *and* in intensity.
//
// A Gaussian blur weights neighbours by distance alone, so it happily averages
// across an edge and destroys it. The bilateral filter multiplies that spatial
// weight by a second Gaussian on the *intensity difference*, so a neighbour on
// the far side of an edge contributes almost nothing. The result smooths within
// regions while leaving boundaries intact.
//
//   w(p, q) = exp(-|p-q|^2 / 2 sigma_s^2) * exp(-|I(p)-I(q)|^2 / 2 sigma_r^2)
//
// The two parameters do quite different jobs:
//   sigma_space -- how far to reach, in pixels.
//   sigma_range -- how different an intensity may be and still count as "the
//                  same region". This is the edge-preservation knob: small
//                  values preserve almost everything, large values degrade
//                  towards a plain Gaussian.
//
// Not separable, because the range weight depends on the centre pixel, so the
// cost is O(r^2) per pixel. Tononi's and Durand's fast approximations exist but
// trade exactness for speed, which is the wrong trade in a comparison lab.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"

namespace tglab {

class Bilateral : public AlgorithmBase {
public:
    const char* Name()     const override { return "bilateral"; }
    const char* Category() const override { return "filter"; }

    PortList Inputs()  const override { return {{"src", DataType::Image, FormatSpec::Any}}; }
    PortList Outputs() const override { return {{"out", DataType::Image, FormatSpec::SameAsInput}}; }

    void RunCPU(RunCtx& ctx) override {
        const ImageView src = ctx.In(0);
        ImageView       dst = ctx.Out(0);
        if (!src.Valid() || !dst.Valid()) return;

        m_in.Unpack(src);
        if (!m_in.Valid()) return;
        m_out.AllocLike(m_in);

        const float sigmaS = std::max(0.01f, float(m_sigmaSpace));
        // sigma_range is declared in 0..1 so it means the same thing whatever
        // the source format, then scaled to the buffer's own range.
        const float sigmaR = std::max(1e-4f, float(m_sigmaRange)) * m_in.ValueScale();
        const int   radius = std::clamp(int(std::ceil(sigmaS * 2.0f)), 1, 32);

        // Spatial weights depend only on the offset, so build them once.
        const int side = radius * 2 + 1;
        m_spatial.assign(size_t(side) * size_t(side), 0.0f);
        for (int dy = -radius; dy <= radius; ++dy)
            for (int dx = -radius; dx <= radius; ++dx)
                m_spatial[size_t(dy + radius) * size_t(side) + size_t(dx + radius)] =
                    std::exp(-float(dx * dx + dy * dy) / (2.0f * sigmaS * sigmaS));

        const int w = m_in.Width(), h = m_in.Height(), ch = m_in.Channels();
        const int filtered = (ch == 1) ? 1 : 3;
        const float twoSR = 2.0f * sigmaR * sigmaR;

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const float* centre = m_in.At(x, y);

                float acc[4]    = {0, 0, 0, 0};
                float weightSum = 0.0f;

                for (int dy = -radius; dy <= radius; ++dy) {
                    for (int dx = -radius; dx <= radius; ++dx) {
                        const float* n = m_in.AtClamped(x + dx, y + dy);

                        // Colour distance across the filtered channels, so an
                        // edge visible only in chroma is still respected.
                        float dist2 = 0.0f;
                        for (int c = 0; c < filtered; ++c) {
                            const float d = n[c] - centre[c];
                            dist2 += d * d;
                        }

                        const float ws =
                            m_spatial[size_t(dy + radius) * size_t(side) + size_t(dx + radius)];
                        const float weight = ws * std::exp(-dist2 / twoSR);

                        for (int c = 0; c < filtered; ++c) acc[c] += n[c] * weight;
                        weightSum += weight;
                    }
                }

                const float inv = 1.0f / std::max(weightSum, 1e-8f);
                for (int c = 0; c < filtered; ++c) m_out.Set(x, y, c, acc[c] * inv);
                if (ch == 4) m_out.Set(x, y, 3, m_in.Get(x, y, 3));
            }
        }

        m_out.PackInto(dst);
    }

    // --- GPU implementation -------------------------------------------------
    // Maps directly: the weight for each neighbour depends only on that
    // neighbour and the centre, so every pixel is independent. The O(r^2)
    // gather that makes this slow on the CPU is exactly what a GPU absorbs.
    // Radius is capped for the usual watchdog reason.
    static constexpr float kMaxGpuSigmaSpace = 8.0f;
    bool HasGPU() const override { return float(m_sigmaSpace) <= kMaxGpuSigmaSpace; }

    const char* GpuSource() const override {
        return R"(
Texture2D<float4>   Src : register(t0);
RWTexture2D<float4> Dst : register(u0);

cbuffer Params : register(b0) {
    uint Width;
    uint Height;
    uint SigmaSpaceBits;
    uint SigmaRangeBits;
};

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;

    float sigmaS = max(asfloat(SigmaSpaceBits), 0.01);
    float sigmaR = max(asfloat(SigmaRangeBits), 1e-4);
    int   radius = min(int(ceil(sigmaS * 2.0)), 16);

    float twoSS = 2.0 * sigmaS * sigmaS;
    float twoSR = 2.0 * sigmaR * sigmaR;

    int2   hi     = int2(Width - 1, Height - 1);
    float4 centre = Src[int2(tid.xy)];

    float3 acc    = 0.0;
    float  weight = 0.0;

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            int2 p = clamp(int2(tid.xy) + int2(dx, dy), int2(0, 0), hi);
            float4 n = Src[p];

            // Colour distance over RGB, so an edge visible only in chroma is
            // still respected -- matching the CPU path.
            float3 d = n.rgb - centre.rgb;
            float dist2 = dot(d, d);

            float w = exp(-float(dx * dx + dy * dy) / twoSS) * exp(-dist2 / twoSR);
            acc    += n.rgb * w;
            weight += w;
        }
    }
    Dst[tid.xy] = float4(acc / max(weight, 1e-8), centre.a);
}
)";
    }

    std::vector<uint32_t> GpuConstants(int) const override {
        const float ss = std::max(0.01f, float(m_sigmaSpace));
        // Textures are normalised 0..1 on the GPU, so sigma_range -- already a
        // fraction of the range -- passes through unscaled.
        const float sr = std::max(1e-4f, float(m_sigmaRange));
        uint32_t a, b;
        std::memcpy(&a, &ss, sizeof(a));
        std::memcpy(&b, &sr, sizeof(b));
        return {a, b};
    }

private:
    Param<float> m_sigmaSpace{
        this, "sigma_space", 3.0f, 0.1f, 32.0f,
        {.help = "How far the filter reaches, in pixels. "
                 "Higher smooths over a wider area. Cost grows with its square.",
         .step = 0.1, .softMin = 0.5, .softMax = 8.0}};
    // Fraction of the full intensity range: 0.1 keeps all but the softest
    // edges, 1.0 is effectively a Gaussian.
    Param<float> m_sigmaRange{
        this, "sigma_range", 0.1f, 0.001f, 1.0f,
        {.help = "How different two pixels may be and still be averaged, "
                 "as a fraction of the intensity range. This is the "
                 "edge-preservation knob: LOWER keeps more edges, and near 1.0 "
                 "it degrades into a plain Gaussian blur.",
         .step = 0.005, .softMin = 0.01, .softMax = 0.4}};

    std::vector<float> m_spatial;
    PixelBuffer m_in, m_out;
};

REGISTER_ALGORITHM(Bilateral);

} // namespace tglab
