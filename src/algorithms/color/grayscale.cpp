// grayscale — second algorithm, added without touching any central file.
// Also the M1 check that self-registration survives the linker.
//
// It originally bailed out on anything but RGBA8 -- `if (format != RGBA8)
// return;` -- which left the output buffer as allocated, i.e. all zeros. A
// demosaiced raw came out pure black, and because grayscale is the first stage
// in hello.tgl, everything downstream of it did too.
#include <algorithm>
#include <cstring>
#include <vector>

#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"

namespace tglab {

class Grayscale : public AlgorithmBase {
public:
    const char* Name()     const override { return "grayscale"; }
    const char* Category() const override { return "color"; }

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

        // Rec.601 by default; the sliders let the weights be explored, which
        // is the kind of thing this lab is for.
        const float wr = m_wr, wg = m_wg, wb = m_wb;
        const float sum = std::max(0.0001f, wr + wg + wb);

        for (int y = 0; y < h; ++y) {
            if (ctx.Cancelled()) return;
            for (int x = 0; x < w; ++x) {
                // A single-channel image is already grey; weighting it against
                // absent channels would only scale it.
                const float g =
                    (ch == 1) ? m_in.Get(x, y, 0)
                              : (m_in.Get(x, y, 0) * wr + m_in.Get(x, y, 1) * wg +
                                 m_in.Get(x, y, 2) * wb) / sum;

                for (int c = 0; c < std::min(ch, 3); ++c) m_out.Set(x, y, c, g);
                if (ch == 4) m_out.Set(x, y, 3, m_in.Get(x, y, 3));   // alpha untouched
            }
        }

        m_out.PackInto(dst);
    }

    // --- GPU implementation -------------------------------------------------
    // One dot product per pixel.
    bool HasGPU() const override { return true; }

    const char* GpuSource() const override {
        return R"(
Texture2D<float4>   Src : register(t0);
RWTexture2D<float4> Dst : register(u0);

cbuffer Params : register(b0) {
    uint Width;
    uint Height;
    uint WrBits, WgBits, WbBits;   // already divided by their sum
};

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;

    float4 c = Src[int2(tid.xy)];
    float3 w = float3(asfloat(WrBits), asfloat(WgBits), asfloat(WbBits));
    float g = dot(c.rgb, w);

    // Alpha untouched, as on the CPU path.
    Dst[tid.xy] = float4(g, g, g, c.a);
}
)";
    }

    std::vector<uint32_t> GpuConstants(int) const override {
        auto bits = [](float f) {
            uint32_t u;
            std::memcpy(&u, &f, sizeof(u));
            return u;
        };

        // Normalised here rather than in the shader, so the division happens
        // once per dispatch instead of once per pixel -- and so the two paths
        // share one expression of what the weights mean.
        const float sum = std::max(0.0001f, float(m_wr) + float(m_wg) + float(m_wb));
        return {bits(float(m_wr) / sum), bits(float(m_wg) / sum), bits(float(m_wb) / sum)};
    }

private:
    Param<float> m_wr{this, "r_weight", 0.299f, 0.0f, 1.0f};
    Param<float> m_wg{this, "g_weight", 0.587f, 0.0f, 1.0f};
    Param<float> m_wb{this, "b_weight", 0.114f, 0.0f, 1.0f};

    PixelBuffer m_in, m_out;
};

REGISTER_ALGORITHM(Grayscale);

} // namespace tglab
