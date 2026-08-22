// demosaic_passthrough — shows the sensor mosaic as it actually is.
//
// The identity function, and deliberately so. It exists to make the raw data
// visible: zoom in and you can see the Bayer pattern itself, each pixel
// carrying one colour rather than three. That is the baseline every other
// demosaic is judged against, and the thing you want on screen when asking
// "what did the sensor actually record?".
//
// Two display modes, because the raw pattern is legible in different ways:
//
//   grey    every sample as luminance, which is what the sensor stores
//   colour  each sample tinted by the colour of its filter, which makes the
//           mosaic structure obvious at a glance
//
// Neither interpolates anything. Colour mode is not a demosaic -- it is the
// same single sample per pixel, just tinted, so it looks dark and dotted. That
// is the point.
#include <algorithm>
#include <cstring>

#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"

namespace tglab {

class DemosaicPassthrough : public AlgorithmBase {
public:
    const char* Name()     const override { return "demosaic_passthrough"; }
    const char* Category() const override { return "demosaic"; }

    PortList Inputs()  const override { return {{"src", DataType::Image, FormatSpec::Any}}; }
    PortList Outputs() const override { return {{"out", DataType::Image, FormatSpec::RGBA16F}}; }

    void RunCPU(RunCtx& ctx) override {
        const ImageView src = ctx.In(0);
        ImageView       dst = ctx.Out(0);
        if (!src.Valid() || !dst.Valid()) return;

        m_in.Unpack(src);
        if (!m_in.Valid()) return;
        m_out.AllocLike(m_in);

        const int w = m_in.Width(), h = m_in.Height();
        const CfaPattern cfa = src.desc.cfa;
        const bool tint = m_colour && cfa != CfaPattern::None;

        // Sensor values are scaled to 0..1 using the black and white levels the
        // raw file reported, so the result is linear scene light rather than an
        // arbitrary integer range.
        const float black = src.desc.blackLevel;
        const float range = std::max(src.desc.whiteLevel - black, 1e-6f);

        m_rgba.assign(size_t(w) * size_t(h) * 4, 0.0f);

        for (int y = 0; y < h; ++y) {
            if (ctx.Cancelled()) return;
            for (int x = 0; x < w; ++x) {
                const float s = std::clamp((m_in.Get(x, y, 0) - black) / range, 0.0f, 4.0f);
                const size_t i = (size_t(y) * size_t(w) + size_t(x)) * 4;

                if (tint) {
                    // One channel lit, the others dark: the mosaic made visible.
                    const int c = CfaColorAt(cfa, x, y);
                    m_rgba[i + size_t(c)] = s;
                } else {
                    m_rgba[i + 0] = s;
                    m_rgba[i + 1] = s;
                    m_rgba[i + 2] = s;
                }
                m_rgba[i + 3] = 1.0f;
            }
        }

        WriteHalf(dst, w, h);
    }

    // --- GPU implementation -------------------------------------------------
    // Trivially parallel: every output pixel depends on exactly one input
    // sample. Worth having anyway, because a 45 MP mosaic is 45 million pixels
    // and even a copy is slow on one CPU thread.
    bool HasGPU() const override { return true; }

    const char* GpuSource() const override {
        return R"(
// The mosaic is single-channel R32F, so only .x carries a sample.
Texture2D<float4>   Src : register(t0);
RWTexture2D<float4> Dst : register(u0);

cbuffer Params : register(b0) {
    uint Width;
    uint Height;
    uint Cfa;        // 0 none, 1 RGGB, 2 BGGR, 3 GRBG, 4 GBRG
    uint Tint;       // draw each sample in its filter's colour
    uint BlackBits;
    uint RangeBits;
};

// Which colour a sample carries, as an RGB index. Mirrors CfaColorAt().
int CfaColor(uint cfa, int x, int y) {
    int q = (y & 1) * 2 + (x & 1);   // 0=TL 1=TR 2=BL 3=BR
    if (cfa == 1) { int c[4] = {0, 1, 1, 2}; return c[q]; }   // RGGB
    if (cfa == 2) { int c[4] = {2, 1, 1, 0}; return c[q]; }   // BGGR
    if (cfa == 3) { int c[4] = {1, 0, 2, 1}; return c[q]; }   // GRBG
    if (cfa == 4) { int c[4] = {1, 2, 0, 1}; return c[q]; }   // GBRG
    return 1;
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;

    float black = asfloat(BlackBits);
    float range = asfloat(RangeBits);
    float s = clamp((Src[int2(tid.xy)].x - black) / range, 0.0, 4.0);

    float3 rgb = float3(s, s, s);
    if (Tint != 0 && Cfa != 0) {
        // One channel lit, the others dark: the mosaic made visible. Not a
        // demosaic -- still one sample per pixel, just tinted.
        int c = CfaColor(Cfa, int(tid.x), int(tid.y));
        rgb = float3(c == 0 ? s : 0.0, c == 1 ? s : 0.0, c == 2 ? s : 0.0);
    }
    Dst[tid.xy] = float4(rgb, 1.0);
}
)";
    }

    void PrepareGpu(const std::vector<ImageDesc>& inputs) override {
        if (inputs.empty()) return;
        m_cfa   = int(inputs[0].cfa);
        m_black = inputs[0].blackLevel;
        m_range = std::max(inputs[0].whiteLevel - inputs[0].blackLevel, 1e-6f);
    }

    std::vector<uint32_t> GpuConstants(int) const override {
        return {uint32_t(m_cfa), uint32_t(m_colour ? 1 : 0),
                Bits(m_black), Bits(m_range)};
    }

private:
    // Sensor metadata, captured by PrepareGpu(). It lives on the input image
    // rather than on a parameter, and the GPU path never runs RunCPU(), so
    // there is no other point at which the shader could learn it.
    int   m_cfa   = 0;
    float m_black = 0.0f;
    float m_range = 1.0f;

    static uint32_t Bits(float f) {
        uint32_t u;
        std::memcpy(&u, &f, sizeof(u));
        return u;
    }

    void WriteHalf(ImageView& dst, int w, int h) const {
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                uint16_t* p = dst.At<uint16_t>(x, y);
                const size_t i = (size_t(y) * size_t(w) + size_t(x)) * 4;
                for (int c = 0; c < 4; ++c) p[c] = FloatToHalf(m_rgba[i + size_t(c)]);
            }
    }

    Param<bool> m_colour{
        this, "colour", true,
        "Tint each sample by the colour of its filter, which makes the Bayer "
        "pattern visible. Off shows the raw luminance the sensor actually "
        "stored. Neither interpolates -- both are one sample per pixel."};

    PixelBuffer        m_in, m_out;
    std::vector<float> m_rgba;
};

REGISTER_ALGORITHM(DemosaicPassthrough);

} // namespace tglab
