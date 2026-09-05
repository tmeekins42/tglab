// box_blur — the unweighted mean over a square window.
//
// Separable and incremental: a running sum updated one column at a time makes
// the cost O(1) per pixel regardless of radius, so a radius-50 box is no
// dearer than a radius-2 one. Useful as the cheap baseline every other blur
// here is compared against, and as the building block of a fast approximate
// Gaussian (three box passes converge on one by the central limit theorem).
#include <algorithm>
#include <vector>

#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"

namespace tglab {

class BoxBlur : public AlgorithmBase {
public:
    const char* Name()     const override { return "box_blur"; }
    const char* Category() const override { return "filter"; }

    PortList Inputs()  const override { return {{"src", DataType::Image, FormatSpec::Any}}; }
    PortList Outputs() const override { return {{"out", DataType::Image, FormatSpec::SameAsInput}}; }

    void RunCPU(RunCtx& ctx) override {
        const ImageView src = ctx.In(0);
        ImageView       dst = ctx.Out(0);
        if (!src.Valid() || !dst.Valid()) return;

        m_in.Unpack(src);
        if (!m_in.Valid()) return;

        const int radius = std::max(1, ctx.ScaledRadius(int(m_radius)));
        const int w = m_in.Width(), h = m_in.Height(), ch = m_in.Channels();

        m_tmp.AllocLike(m_in);
        m_out.AllocLike(m_in);

        // Horizontal pass, then vertical. Each keeps a running sum and moves
        // the window by adding the entering column and dropping the leaving
        // one, which is what makes the radius free.
        BlurAxis(m_in, m_tmp, w, h, ch, radius, /*horizontal=*/true);
        BlurAxis(m_tmp, m_out, w, h, ch, radius, /*horizontal=*/false);

        m_out.PackInto(dst);
    }

    // --- GPU implementation -------------------------------------------------
    // Separable, in two passes, like the CPU path.
    //
    // It was a single O(r^2) gather, on the reasoning that a two-pass version
    // "needs an intermediate target" -- true when that was written, but the
    // iterative dispatch path now supplies a ping-pong buffer. The cap that
    // followed from it is gone: at radius 24 one pass is 2,401 fetches per pixel
    // against 98 for two separable ones, and above the cap the stage fell back
    // to the CPU exactly where a blur costs most.
    // Reads a window of this radius, so a tile needs that much margin.
    int ReachPixels() const override { return std::max(1, int(m_radius)); }

    bool HasGPU() const override { return true; }
    int  GpuIterations() const override { return 2; }

    const char* GpuSource() const override {
        return R"(
Texture2D<float4>   Src : register(t0);
RWTexture2D<float4> Dst : register(u0);

cbuffer Params : register(b0) {
    uint Width;
    uint Height;
    uint Radius;
    uint Pass;        // 0 = horizontal, 1 = vertical
};

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;

    int r = int(Radius);
    float4 sum = 0.0;
    float  n   = 0.0;

    for (int i = -r; i <= r; ++i) {
        int2 p = (Pass == 0) ? int2(int(tid.x) + i, int(tid.y))
                             : int2(int(tid.x), int(tid.y) + i);
        p = clamp(p, int2(0, 0), int2(Width - 1, Height - 1));   // clamp-to-edge
        sum += Src[p];
        n   += 1.0;
    }
    Dst[tid.xy] = sum / max(n, 1.0);
}
)";
    }

    std::vector<uint32_t> GpuConstants(int iteration) const override {
        return {uint32_t(std::max(1, GpuScaledRadius(int(m_radius)))), uint32_t(iteration)};
    }

private:
    // One separable pass. `horizontal` picks the axis; the running sum is
    // seeded with the clamped window at position 0 and then slid along.
    static void BlurAxis(const PixelBuffer& in, PixelBuffer& out,
                         int w, int h, int ch, int radius, bool horizontal) {
        const int outer = horizontal ? h : w;
        const int inner = horizontal ? w : h;
        const float norm = 1.0f / float(radius * 2 + 1);

        std::vector<float> sum(size_t(ch), 0.0f);

        // Strides rather than a per-sample branch on `horizontal`: the inner
        // loop runs once per pixel per channel, so a branch there costs more
        // than the whole running-sum trick saves.
        const size_t stride = horizontal ? size_t(ch) : size_t(w) * size_t(ch);
        const size_t step   = horizontal ? size_t(w) * size_t(ch) : size_t(ch);
        const int    limit  = inner - 1;

        const float* base = in.Data().data();
        float*       dest = out.Data().data();

        for (int o = 0; o < outer; ++o) {
            const float* line = base + step * size_t(o);
            float*       dst  = dest + step * size_t(o);

            std::fill(sum.begin(), sum.end(), 0.0f);
            for (int i = -radius; i <= radius; ++i) {
                const float* p = line + stride * size_t(std::clamp(i, 0, limit));
                for (int c = 0; c < ch; ++c) sum[size_t(c)] += p[c];
            }

            for (int i = 0; i < inner; ++i) {
                float* o_ = dst + stride * size_t(i);
                for (int c = 0; c < ch; ++c) o_[c] = sum[size_t(c)] * norm;

                // Slide: the sample entering at i+radius+1 replaces the one
                // leaving at i-radius.
                const float* add = line + stride * size_t(std::clamp(i + radius + 1, 0, limit));
                const float* sub = line + stride * size_t(std::clamp(i - radius, 0, limit));
                for (int c = 0; c < ch; ++c) sum[size_t(c)] += add[c] - sub[c];
            }
        }
    }

    Param<int> m_radius{
        this, "radius", 3, 1, 64,
        {.help = "Half-width of the averaging window, in pixels. "
                 "Higher blurs more. Costs the same at any size.",
         .softMin = 1, .softMax = 25}};

    PixelBuffer m_in, m_tmp, m_out;
};

REGISTER_ALGORITHM(BoxBlur);

} // namespace tglab
