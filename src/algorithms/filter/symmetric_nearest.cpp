// symmetric_nearest — SNN, symmetric nearest neighbour smoothing.
//
// A cheap edge-preserving filter that deserves to be better known, and a good
// foil for Kuwahara in a comparison: same goal, quite different mechanism.
//
// For every symmetric pair of neighbours around the centre -- (dx, dy) and
// (-dx, -dy) -- keep whichever of the two is closer in value to the centre
// pixel, and average the kept ones. At an edge, one member of each pair lies
// across it and the other does not, so the one on the centre's own side always
// wins. The edge is therefore never averaged across, while the interior of a
// region still gets a full neighbourhood average.
//
// Unlike Kuwahara this has no quadrant geometry at all, so it has no preferred
// orientation and no blocky artefacts -- but it also cannot enhance corners the
// way the Papari variant does. Cost is O(r^2) per pixel with a very small
// constant.
#include <algorithm>
#include <cmath>

#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"

namespace tglab {

class SymmetricNearest : public AlgorithmBase {
public:
    const char* Name()     const override { return "symmetric_nearest"; }
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

        for (int y = 0; y < h; ++y) {
            // Checked per row: cancelling only between stages is useless when a
            // single stage is the slow one.
            if (ctx.Cancelled()) return;
            for (int x = 0; x < w; ++x) {
                const float* centre = m_in.At(x, y);

                // The centre pixel itself is always included, so a completely
                // flat window returns exactly its own value.
                float acc[4] = {0, 0, 0, 0};
                for (int c = 0; c < filtered; ++c) acc[c] = centre[c];
                int count = 1;

                // Walk half the window; each offset's mirror is its partner.
                for (int dy = -radius; dy <= radius; ++dy) {
                    for (int dx = -radius; dx <= radius; ++dx) {
                        // Take each pair once: skip the half that mirrors one
                        // already handled, and the centre itself.
                        if (dy > 0 || (dy == 0 && dx >= 0)) continue;

                        const float* a = m_in.AtClamped(x + dx, y + dy);
                        const float* b = m_in.AtClamped(x - dx, y - dy);

                        // Compare on luma so colour channels pick the same
                        // member of the pair and cannot disagree.
                        const float da = std::abs(Luma(a, filtered) - Luma(centre, filtered));
                        const float db = std::abs(Luma(b, filtered) - Luma(centre, filtered));
                        const float* keep = (da <= db) ? a : b;

                        for (int c = 0; c < filtered; ++c) acc[c] += keep[c];
                        ++count;
                    }
                }

                const float inv = 1.0f / float(count);
                for (int c = 0; c < filtered; ++c) m_out.Set(x, y, c, acc[c] * inv);
                if (ch == 4) m_out.Set(x, y, 3, m_in.Get(x, y, 3));
            }
        }

        m_out.PackInto(dst);
    }

    // --- GPU implementation -------------------------------------------------
    // Each pixel walks its own window and never writes anywhere else, so this
    // ports directly.
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
    int2 me = int2(tid.xy);

    float4 centre = Src[me];
    float  cl     = dot(centre.rgb, kLuma);

    // The centre pixel is always included, so a flat window returns itself.
    float3 acc = centre.rgb;
    float  n   = 1.0;

    for (int dy = -r; dy <= r; ++dy) {
        for (int dx = -r; dx <= r; ++dx) {
            // Take each symmetric pair once: skip the mirrored half and the
            // centre itself.
            if (dy > 0 || (dy == 0 && dx >= 0)) continue;

            float3 a = Src[clamp(me + int2( dx,  dy), int2(0, 0), hi)].rgb;
            float3 b = Src[clamp(me + int2(-dx, -dy), int2(0, 0), hi)].rgb;

            // Whichever of the pair is closer to the centre in luma wins, so
            // the one across an edge is always the one discarded.
            float da = abs(dot(a, kLuma) - cl);
            float db = abs(dot(b, kLuma) - cl);
            acc += (da <= db) ? a : b;
            n   += 1.0;
        }
    }

    Dst[tid.xy] = float4(acc / n, centre.a);
}
)";
    }

    std::vector<uint32_t> GpuConstants(int) const override {
        return {uint32_t(std::max(1, int(m_radius)))};
    }

private:
    static float Luma(const float* p, int channels) {
        if (channels == 1) return p[0];
        return 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
    }

    Param<int> m_radius{
        this, "radius", 3, 1, 32,
        {.help = "Half-width of the window of symmetric pairs. Higher smooths "
                 "more while still never averaging across an edge.",
         .softMin = 1, .softMax = 12}};

    PixelBuffer m_in, m_out;
};

REGISTER_ALGORITHM(SymmetricNearest);

} // namespace tglab
