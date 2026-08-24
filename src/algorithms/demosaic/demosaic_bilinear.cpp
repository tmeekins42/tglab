// demosaic_bilinear — the standard baseline.
//
// Every pixel records one colour; the other two are averaged from the nearest
// neighbours that recorded them. On a Bayer pattern that means:
//
//   at a red site    R is measured; G from 4 edge neighbours, B from 4 corners
//   at a blue site   B is measured; G from 4 edge neighbours, R from 4 corners
//   at a green site  G is measured; R and B from the 2 neighbours each way
//
// Cheap, simple, and visibly wrong at edges: averaging across a boundary
// invents colour that was never there, which shows as red/blue fringing on
// high-contrast detail. That failure is exactly why it makes a good baseline --
// every better method is defined by how it avoids this.
//
// The output is linear light scaled by the sensor's black and white levels, in
// RGBA16F. That is the whole point of the raw path: 14 bits of real dynamic
// range reaching the adjustment stages instead of 8 bits already flattened.
#include <algorithm>
#include <cstring>
#include <vector>

#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"

namespace tglab {

class DemosaicBilinear : public AlgorithmBase {
public:
    const char* Name()     const override { return "demosaic_bilinear"; }
    const char* Category() const override { return "demosaic"; }

    PortList Inputs()  const override { return {{"src", DataType::Image, FormatSpec::Any}}; }
    PortList Outputs() const override { return {{"out", DataType::Image, FormatSpec::RGBA16F}}; }

    void RunCPU(RunCtx& ctx) override {
        const ImageView src = ctx.In(0);
        ImageView       dst = ctx.Out(0);
        if (!src.Valid() || !dst.Valid()) return;

        m_in.Unpack(src);
        if (!m_in.Valid()) return;

        const int w = m_in.Width(), h = m_in.Height();
        const CfaPattern cfa = src.desc.cfa;

        // Without a CFA this is not a mosaic at all. Pass it through rather
        // than inventing a pattern: a script may well apply a demosaic
        // unconditionally, and mangling an ordinary image would be worse than
        // doing nothing.
        if (cfa == CfaPattern::None || cfa == CfaPattern::XTrans) {
            PassThrough(dst, w, h);
            return;
        }

        const float black = src.desc.blackLevel;
        const float range = std::max(src.desc.whiteLevel - black, 1e-6f);

        // Normalised samples first, so the interpolation below is plain
        // arithmetic on linear values.
        m_s.assign(size_t(w) * size_t(h), 0.0f);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                m_s[size_t(y) * size_t(w) + size_t(x)] =
                    std::clamp((m_in.Get(x, y, 0) - black) / range, 0.0f, 4.0f);

        auto at = [&](int x, int y) {
            // Edge-clamped: reflecting would be marginally better at the
            // border, but clamping keeps the CFA phase correct, and getting
            // the phase wrong swaps red and blue along the edge.
            x = std::clamp(x, 0, w - 1);
            y = std::clamp(y, 0, h - 1);
            return m_s[size_t(y) * size_t(w) + size_t(x)];
        };

        for (int y = 0; y < h; ++y) {
            if (ctx.Cancelled()) return;
            for (int x = 0; x < w; ++x) {
                const int c = CfaColorAt(cfa, x, y);
                float rgb[3] = {0, 0, 0};
                rgb[c] = at(x, y);   // the one colour actually measured here

                if (c == 1) {
                    // A green site. Its horizontal and vertical neighbours are
                    // the other two colours -- which one is which depends on
                    // the row, so ask the pattern rather than assuming.
                    const float horiz = 0.5f * (at(x - 1, y) + at(x + 1, y));
                    const float vert   = 0.5f * (at(x, y - 1) + at(x, y + 1));
                    const int horizColor = CfaColorAt(cfa, x - 1, y);
                    rgb[horizColor]         = horiz;
                    rgb[2 - horizColor]     = vert;   // 0 <-> 2, red <-> blue
                } else {
                    // A red or blue site. Green comes from the 4 edge
                    // neighbours, the opposite colour from the 4 corners.
                    rgb[1] = 0.25f * (at(x - 1, y) + at(x + 1, y) +
                                      at(x, y - 1) + at(x, y + 1));
                    rgb[2 - c] = 0.25f * (at(x - 1, y - 1) + at(x + 1, y - 1) +
                                          at(x - 1, y + 1) + at(x + 1, y + 1));
                }

                ApplyColour(src.desc, rgb);

                uint16_t* p = dst.At<uint16_t>(x, y);
                p[0] = FloatToHalf(rgb[0]);
                p[1] = FloatToHalf(rgb[1]);
                p[2] = FloatToHalf(rgb[2]);
                p[3] = FloatToHalf(1.0f);
            }
        }
    }

    // --- GPU implementation -------------------------------------------------
    // Nine texture fetches per pixel with no cross-pixel dependency: exactly
    // what a compute shader is for. This runs once per image load rather than
    // per slider, but a 45 MP mosaic is 45 million pixels and the CPU path
    // takes tens of seconds.
    // Deliberately unconditional, and *not* a test of the input.
    //
    // HasGPU() is consulted before PrepareGpu() runs, so anything derived from
    // the input image is still at its default here -- gating on the CFA meant
    // the answer was always "no" on the first run and the GPU path never
    // engaged at all. The shader handles the non-Bayer case instead.
    bool HasGPU() const override { return true; }

    void PrepareGpu(const std::vector<ImageDesc>& inputs) override {
        if (inputs.empty()) return;
        // 0 for None and for X-Trans, whose 6x6 layout needs its own algorithm.
        // The shader passes those through rather than mangling them.
        const int cfa = int(inputs[0].cfa);
        m_cfa   = (cfa >= 1 && cfa <= 4) ? cfa : 0;
        m_black = inputs[0].blackLevel;
        m_range = std::max(inputs[0].whiteLevel - inputs[0].blackLevel, 1e-6f);
        for (int i = 0; i < 3; ++i) m_camMul[i] = inputs[0].camMul[i];
        for (int i = 0; i < 9; ++i) m_rgbCam[i] = inputs[0].rgbCam[i];
    }

    const char* GpuSource() const override {
        return R"(
Texture2D<float4>   Src : register(t0);   // R32F mosaic: only .x is a sample
RWTexture2D<float4> Dst : register(u0);

cbuffer Params : register(b0) {
    uint Width;
    uint Height;
    uint Cfa;        // 1 RGGB, 2 BGGR, 3 GRBG, 4 GBRG
    uint BlackBits;
    uint RangeBits;
    uint CamMul0, CamMul1, CamMul2;          // as-shot white balance
    uint M0, M1, M2, M3, M4, M5, M6, M7, M8; // camera -> sRGB, row-major
};

int CfaColor(uint cfa, int x, int y) {
    int q = (y & 1) * 2 + (x & 1);
    if (cfa == 1) { int c[4] = {0, 1, 1, 2}; return c[q]; }
    if (cfa == 2) { int c[4] = {2, 1, 1, 0}; return c[q]; }
    if (cfa == 3) { int c[4] = {1, 0, 2, 1}; return c[q]; }
    if (cfa == 4) { int c[4] = {1, 2, 0, 1}; return c[q]; }
    return 1;
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;

    float black = asfloat(BlackBits);
    float range = asfloat(RangeBits);
    int2  hi    = int2(Width - 1, Height - 1);

    // Edge-clamped, matching the CPU path. Clamping rather than mirroring
    // keeps the CFA phase correct at the border; mirroring would swap red and
    // blue along the edge.
    #define S(dx, dy) clamp((Src[clamp(int2(tid.xy) + int2(dx, dy), int2(0,0), hi)].x - black) / range, 0.0, 4.0)

    int x = int(tid.x), y = int(tid.y);

    if (Cfa == 0) {
        // Not a Bayer mosaic (an ordinary image, or X-Trans). Pass it through
        // rather than inventing a pattern, matching the CPU path -- a script
        // may apply a demosaic unconditionally, and mangling a PNG would be
        // worse than doing nothing.
        float4 t = Src[int2(tid.xy)];
        Dst[tid.xy] = float4(t.rgb, 1.0);
        return;
    }

    int c = CfaColor(Cfa, x, y);

    float3 rgb = float3(0, 0, 0);
    rgb[c] = S(0, 0);   // the one colour actually measured here

    if (c == 1) {
        // A green site: its horizontal and vertical neighbours are the other
        // two colours. Which is which depends on the row, so ask the pattern.
        float horiz = 0.5 * (S(-1, 0) + S(1, 0));
        float vert  = 0.5 * (S(0, -1) + S(0, 1));
        int horizColor = CfaColor(Cfa, x - 1, y);
        rgb[horizColor]     = horiz;
        rgb[2 - horizColor] = vert;
    } else {
        // A red or blue site: green from the 4 edge neighbours, the opposite
        // colour from the 4 corners.
        rgb[1] = 0.25 * (S(-1, 0) + S(1, 0) + S(0, -1) + S(0, 1));
        rgb[2 - c] = 0.25 * (S(-1, -1) + S(1, -1) + S(-1, 1) + S(1, 1));
    }

    #undef S

    // Record which channels saturated before white balance; repair after it.
    // Matches the CPU path -- see RecoverClipped in demosaic_malvar.cpp.
    const float kClip = 0.99;
    bool3 clipped = bool3(rgb.r >= kClip, rgb.g >= kClip, rgb.b >= kClip);

    // White balance, then camera primaries -> sRGB. Same order as the CPU
    // path; without either the image is heavily green with wrong hues.
    rgb *= float3(asfloat(CamMul0), asfloat(CamMul1), asfloat(CamMul2));

    if (any(clipped)) {
        float hi = max(rgb.r, max(rgb.g, rgb.b));
        rgb = float3(clipped.x ? hi : rgb.r,
                     clipped.y ? hi : rgb.g,
                     clipped.z ? hi : rgb.b);
    }
    rgb = float3(
        asfloat(M0) * rgb.r + asfloat(M1) * rgb.g + asfloat(M2) * rgb.b,
        asfloat(M3) * rgb.r + asfloat(M4) * rgb.g + asfloat(M5) * rgb.b,
        asfloat(M6) * rgb.r + asfloat(M7) * rgb.g + asfloat(M8) * rgb.b);

    // The matrix maps a wider gamut inward and has negative coefficients, so
    // out-of-gamut colours can go negative.
    Dst[tid.xy] = float4(max(rgb, 0.0), 1.0);
}
)";
    }

    std::vector<uint32_t> GpuConstants(int) const override {
        auto bits = [](float f) {
            uint32_t u;
            std::memcpy(&u, &f, sizeof(u));
            return u;
        };
        std::vector<uint32_t> c{uint32_t(m_cfa), bits(m_black), bits(m_range)};
        for (int i = 0; i < 3; ++i) c.push_back(bits(m_camMul[i]));
        for (int i = 0; i < 9; ++i) c.push_back(bits(m_rgbCam[i]));
        return c;
    }

private:
    // White balance then the colour matrix, in that order.
    //
    // Both are properties of the capture rather than of the demosaic, but this
    // is the only place they can be applied: before it there is one channel per
    // pixel, and after it the data has already been treated as sRGB. Skipping
    // them leaves a heavily green image with wrong hues -- the sensor's green
    // photosites are about twice as sensitive as its red and blue ones.
    static void ApplyColour(const ImageDesc& d, float* rgb) {
        // Which channels saturated is decided BEFORE white balance, against the
        // sensor's white level; the repair happens AFTER, where neutral means
        // the channels are equal. See RecoverClipped in demosaic_malvar.cpp --
        // the two must agree, since a script can switch between them.
        constexpr float kClip = 0.99f;
        const bool clipped[3] = {rgb[0] >= kClip, rgb[1] >= kClip, rgb[2] >= kClip};

        rgb[0] *= d.camMul[0];
        rgb[1] *= d.camMul[1];
        rgb[2] *= d.camMul[2];

        if (clipped[0] || clipped[1] || clipped[2]) {
            const float hi = std::max(rgb[0], std::max(rgb[1], rgb[2]));
            if (clipped[0]) rgb[0] = hi;
            if (clipped[1]) rgb[1] = hi;
            if (clipped[2]) rgb[2] = hi;
        }

        const float r = rgb[0], g = rgb[1], b = rgb[2];
        rgb[0] = d.rgbCam[0] * r + d.rgbCam[1] * g + d.rgbCam[2] * b;
        rgb[1] = d.rgbCam[3] * r + d.rgbCam[4] * g + d.rgbCam[5] * b;
        rgb[2] = d.rgbCam[6] * r + d.rgbCam[7] * g + d.rgbCam[8] * b;

        // The matrix has negative coefficients by design (it maps a wider
        // gamut inward), so out-of-gamut colours can go negative. Clamping
        // here keeps later stages from having to reason about it.
        for (int i = 0; i < 3; ++i) rgb[i] = std::max(rgb[i], 0.0f);
    }

    // Sensor metadata from the input descriptor, captured by PrepareGpu().
    int   m_cfa   = 0;
    float m_black = 0.0f;
    float m_range = 1.0f;
    float m_camMul[3] = {1.0f, 1.0f, 1.0f};
    float m_rgbCam[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    // Not a mosaic: copy through unchanged so an unconditional demosaic in a
    // script is harmless on an ordinary image.
    void PassThrough(ImageView& dst, int w, int h) {
        const int ch = m_in.Channels();
        const float scale = m_in.ValueScale();
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                uint16_t* p = dst.At<uint16_t>(x, y);
                for (int c = 0; c < 3; ++c)
                    p[c] = FloatToHalf(m_in.Get(x, y, ch == 1 ? 0 : c) / scale);
                p[3] = FloatToHalf(ch == 4 ? m_in.Get(x, y, 3) / scale : 1.0f);
            }
    }

    PixelBuffer        m_in;
    std::vector<float> m_s;   // normalised sensor samples
};

REGISTER_ALGORITHM(DemosaicBilinear);

} // namespace tglab
