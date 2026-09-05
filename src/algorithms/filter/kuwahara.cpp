// kuwahara — edge-preserving smoothing by picking the flattest neighbourhood.
//
// Instead of averaging everything around a pixel, split the window into four
// overlapping quadrants, compute each one's mean and variance, and output the
// mean of whichever quadrant is most uniform:
//
//     +---+---+
//     | A | B |     each quadrant is (radius+1) x (radius+1)
//     +---+---+     and includes the centre pixel
//     | C | D |
//     +---+---+
//
// A pixel next to an edge has at least one quadrant lying entirely on its own
// side of that edge; that quadrant has the lowest variance, so its mean is
// used and the edge is never averaged across. The effect is painterly -- flat
// regions with crisp boundaries -- which is why it is also used as an artistic
// filter.
//
// Two known weaknesses, both worth being able to see:
//   - Only four orientations, so diagonal and curved edges are approximated
//     by blocky quadrants.
//   - Ties between quadrants are broken arbitrarily, which shows up as
//     directional artefacts in noise.
// Both motivated the generalized variants; see kuwahara_generalized.
#include <algorithm>
#include <cmath>
#include <vector>

#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"

namespace tglab {

class Kuwahara : public AlgorithmBase {
public:
    const char* Name()     const override { return "kuwahara"; }
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

        const int radius = std::max(1, ctx.ScaledRadius(int(m_radius)));
        const int w = m_in.Width(), h = m_in.Height(), ch = m_in.Channels();
        const int filtered = (ch == 1) ? 1 : 3;

        // The four quadrant origins, relative to the centre pixel.
        const int offsets[4][2] = {{-radius, -radius}, {0, -radius},
                                   {-radius, 0},       {0, 0}};

        for (int y = 0; y < h; ++y) {
            // Checked per row: cancelling only between stages is useless when a
            // single stage is the slow one.
            if (ctx.Cancelled()) return;
            for (int x = 0; x < w; ++x) {
                float bestMean[4] = {0, 0, 0, 0};
                float bestVar     = 0.0f;
                bool  haveBest    = false;

                for (const auto& off : offsets) {
                    float sum[4]   = {0, 0, 0, 0};
                    float sumSq    = 0.0f;   // variance measured on luminance
                    int   count    = 0;

                    for (int dy = 0; dy <= radius; ++dy) {
                        for (int dx = 0; dx <= radius; ++dx) {
                            const float* p = m_in.AtClamped(x + off[0] + dx, y + off[1] + dy);
                            for (int c = 0; c < filtered; ++c) sum[c] += p[c];

                            // A single scalar decides "most uniform", so colour
                            // channels cannot disagree about which quadrant wins.
                            const float lum = Luma(p, filtered);
                            sumSq += lum * lum;
                            ++count;
                        }
                    }

                    // Variance of the same luma the sum of squares was taken
                    // over, so E[x^2] - E[x]^2 is consistent.
                    const float inv     = 1.0f / float(count);
                    const float lumMean = LumaFromSums(sum, filtered) * inv;
                    const float variance = std::max(0.0f, sumSq * inv - lumMean * lumMean);

                    if (!haveBest || variance < bestVar) {
                        bestVar  = variance;
                        haveBest = true;
                        for (int c = 0; c < filtered; ++c) bestMean[c] = sum[c] * inv;
                    }
                }

                for (int c = 0; c < filtered; ++c) m_out.Set(x, y, c, bestMean[c]);
                if (ch == 4) m_out.Set(x, y, 3, m_in.Get(x, y, 3));
            }
        }

        m_out.PackInto(dst);
    }

    // --- GPU implementation -------------------------------------------------
    // Four independent quadrant reductions per pixel, no cross-pixel
    // dependency: a natural fit. Capped so a large radius cannot stall the
    // device -- four quadrants of (r+1)^2 each grows quickly.
    static constexpr int kMaxGpuRadius = 12;
    // Reads a window of this radius, so a tile needs that much margin.
    int ReachPixels() const override { return std::max(1, int(m_radius)); }

    bool HasGPU() const override { return int(m_radius) <= kMaxGpuRadius; }

    const char* GpuSource() const override {
        return R"(
Texture2D<float4>   Src : register(t0);
RWTexture2D<float4> Dst : register(u0);

cbuffer Params : register(b0) {
    uint Width;
    uint Height;
    uint Radius;
};

static const float3 kLuma = float3(0.299, 0.587, 0.114);

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;

    int  r  = int(Radius);
    int2 hi = int2(Width - 1, Height - 1);

    // The four quadrant origins, each overlapping at the centre pixel.
    int2 offsets[4] = {
        int2(-r, -r), int2(0, -r),
        int2(-r,  0), int2(0,  0)
    };

    float3 bestMean = 0.0;
    float  bestVar  = 0.0;
    bool   haveBest = false;

    for (int q = 0; q < 4; ++q) {
        float3 sum   = 0.0;
        float  sumSq = 0.0;
        float  n     = 0.0;

        for (int dy = 0; dy <= r; ++dy) {
            for (int dx = 0; dx <= r; ++dx) {
                int2 p = clamp(int2(tid.xy) + offsets[q] + int2(dx, dy), int2(0, 0), hi);
                float3 c = Src[p].rgb;
                sum   += c;
                // Variance measured on luma, so the colour channels cannot
                // disagree about which quadrant is flattest.
                float l = dot(c, kLuma);
                sumSq += l * l;
                n     += 1.0;
            }
        }

        float  inv     = 1.0 / max(n, 1.0);
        float3 mean    = sum * inv;
        float  lumMean = dot(mean, kLuma);
        float  variance = max(0.0, sumSq * inv - lumMean * lumMean);

        if (!haveBest || variance < bestVar) {
            bestVar  = variance;
            bestMean = mean;
            haveBest = true;
        }
    }

    Dst[tid.xy] = float4(bestMean, Src[int2(tid.xy)].a);
}
)";
    }

    std::vector<uint32_t> GpuConstants(int) const override {
        return {uint32_t(std::max(1, GpuScaledRadius(int(m_radius))))};
    }

private:
    // Rec. 601 luma, matching the default weights in the grayscale algorithm.
    static float Luma(const float* p, int channels) {
        if (channels == 1) return p[0];
        return 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
    }
    static float LumaFromSums(const float* sum, int channels) {
        if (channels == 1) return sum[0];
        return 0.299f * sum[0] + 0.587f * sum[1] + 0.114f * sum[2];
    }

    Param<int> m_radius{
        this, "radius", 3, 1, 32,
        {.help = "Size of each of the four quadrants, in pixels. Higher gives "
                 "broader flat regions and a more painterly result.",
         .softMin = 1, .softMax = 12}};

    PixelBuffer m_in, m_out;
};

REGISTER_ALGORITHM(Kuwahara);

} // namespace tglab
