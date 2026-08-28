// brightness — the M1 vertical-slice algorithm.
//
// Deliberately trivial: it exists to debug the framework (script -> registry
// -> ports -> params -> display), not the math.
//
// It nonetheless has to handle every pixel format, and originally did not: it
// branched on RGBA8 and R32F by hand and fell out of the bottom for anything
// else, so a demosaiced raw (RGBA16F) came out as zeros -- a black image, with
// no error anywhere. PixelBuffer exists for exactly this, and unpacking through
// it is both shorter and complete.
#include <algorithm>
#include <cstring>
#include <vector>

#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"

namespace tglab {

class Brightness : public AlgorithmBase {
public:
    const char* Name()     const override { return "brightness"; }
    const char* Category() const override { return "adjust"; }

    PortList Inputs()  const override { return {{"src", DataType::Image, FormatSpec::Any}}; }
    PortList Outputs() const override { return {{"out", DataType::Image, FormatSpec::SameAsInput}}; }

    void RunCPU(RunCtx& ctx) override {
        const ImageView src = ctx.In(0);
        ImageView       dst = ctx.Out(0);
        if (!src.Valid() || !dst.Valid()) return;

        m_in.Unpack(src);
        if (!m_in.Valid()) return;
        m_out.AllocLike(m_in);

        const int w = m_in.Width(), h = m_in.Height(), ch = m_in.Channels();

        // Brightness is an offset, so it has to be expressed in the image's own
        // units: +0.5 means "half the intensity range" whether that range is
        // 0..255 or 0..1. Gain is a multiplier and needs no scaling.
        const float scale  = m_in.ValueScale();
        const float offset = float(m_brightness) * scale;
        const float gain   = float(m_gain);

        for (int y = 0; y < h; ++y) {
            if (ctx.Cancelled()) return;
            for (int x = 0; x < w; ++x) {
                // Alpha is carried through untouched: brightening an image
                // should not change what is transparent.
                const int colours = (ch == 4) ? 3 : ch;
                for (int c = 0; c < colours; ++c)
                    m_out.Set(x, y, c, m_in.Get(x, y, c) * gain + offset);
                if (ch == 4) m_out.Set(x, y, 3, m_in.Get(x, y, 3));
            }
        }

        m_out.PackInto(dst);
    }

    // No offset and unit gain is exactly the identity, so the stage is skipped
    // outright -- no allocation, no dispatch, no copy. This is the shape most
    // adjustments take: a neutral setting a stacked script leaves alone most of
    // the time, which is precisely when it should cost nothing.
    bool IsNoOp() const override {
        return float(m_brightness) == 0.0f && float(m_gain) == 1.0f;
    }

    // --- GPU implementation -------------------------------------------------
    // One fetch, two operations, one store. Nothing here is interesting except
    // the units, which are the one place the two paths could disagree.
    bool HasGPU() const override { return true; }

    const char* GpuSource() const override {
        return R"(
Texture2D<float4>   Src : register(t0);
RWTexture2D<float4> Dst : register(u0);

cbuffer Params : register(b0) {
    uint Width;
    uint Height;
    uint OffsetBits;   // already in 0..1, see below
    uint GainBits;
};

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;

    float4 c = Src[int2(tid.xy)];
    float offset = asfloat(OffsetBits);
    float gain   = asfloat(GainBits);

    // Alpha carried through: brightening should not change transparency.
    Dst[tid.xy] = float4(c.rgb * gain + offset, c.a);
}
)";
    }

    std::vector<uint32_t> GpuConstants(int) const override {
        auto bits = [](float f) {
            uint32_t u;
            std::memcpy(&u, &f, sizeof(u));
            return u;
        };

        // NOT scaled by 255 the way the CPU path scales it.
        //
        // The CPU works in the source's own units, so an RGBA8 image needs the
        // offset multiplied up to 0..255. A UNORM SRV hands the shader 0..1
        // whatever the storage format, so scaling here would apply the offset
        // 255 times over. The two paths therefore look different and agree,
        // which is exactly what the CPU/GPU agreement test exists to confirm.
        return {bits(float(m_brightness)), bits(float(m_gain))};
    }

private:
    // Named for what they do. `amount` said nothing -- an amount of what? --
    // and sat next to `gain`, which does say.
    Param<float> m_brightness{
        this, "brightness", 0.0f, -1.0f, 1.0f,
        {.help = "Added to every pixel, as a fraction of the intensity range. "
                 "+0.5 lifts everything by half the range. An offset, so it "
                 "moves shadows as much as highlights.",
         .step = 0.01, .softMin = -0.5, .softMax = 0.5}};

    Param<float> m_gain{
        this, "gain", 1.0f, 0.0f, 4.0f,
        {.help = "Multiplied into every pixel. 2.0 doubles it. A multiply, so "
                 "it moves highlights much further than shadows -- which is "
                 "what makes it different from brightness.",
         .step = 0.05, .softMin = 0.0, .softMax = 2.0}};

    PixelBuffer m_in, m_out;
};

REGISTER_ALGORITHM(Brightness);

} // namespace tglab
