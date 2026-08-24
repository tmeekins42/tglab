// demosaic_malvar — Malvar, He & Cutler (2004), "High-Quality Linear
// Interpolation for Demosaicing of Bayer-Patterned Color Images", ICASSP.
//
// The best cost/quality ratio in demosaicing, and the reason it is worth
// having next to bilinear: it is barely more arithmetic and dramatically
// better.
//
// The insight is that bilinear throws away information it already has.
// Interpolating green at a red site, it averages the four green neighbours and
// ignores the red sample sitting right there -- even though red and green are
// highly correlated in natural images, so the *local red gradient* predicts how
// green is changing. Malvar adds a correction term from the centre channel's
// second derivative, which is exactly that gradient.
//
// Concretely, green at a red site becomes the bilinear average plus
// alpha * (5*R(x,y) - R(x±2,y) - R(x,y±2)) / 8, and similar 5x5 kernels handle
// the red and blue cases. Still a linear filter -- no branching on gradients,
// no per-pixel decisions -- so it vectorises and maps to a GPU exactly like
// bilinear does. It just uses better weights.
//
// Visible effect: far less colour fringing on high-contrast edges, which is
// bilinear's characteristic failure. The gains alpha/beta/gamma are the paper's
// least-squares-optimal values and are exposed as parameters so the derivation
// can be poked at rather than taken on trust.
#include <algorithm>
#include <cstring>
#include <vector>

#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"

namespace tglab {

class DemosaicMalvar : public AlgorithmBase {
public:
    const char* Name()     const override { return "demosaic_malvar"; }
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

        // Not a mosaic: pass through rather than inventing a pattern, matching
        // the other demosaicers so an unconditional call is harmless.
        if (cfa == CfaPattern::None || cfa == CfaPattern::XTrans) {
            PassThrough(dst, w, h);
            return;
        }

        const float black = src.desc.blackLevel;
        const float range = std::max(src.desc.whiteLevel - black, 1e-6f);

        m_s.assign(size_t(w) * size_t(h), 0.0f);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                m_s[size_t(y) * size_t(w) + size_t(x)] =
                    std::clamp((m_in.Get(x, y, 0) - black) / range, 0.0f, 4.0f);

        auto at = [&](int x, int y) {
            x = std::clamp(x, 0, w - 1);
            y = std::clamp(y, 0, h - 1);
            return m_s[size_t(y) * size_t(w) + size_t(x)];
        };

        const float alpha = float(m_alpha);
        const float beta  = float(m_beta);
        const float gamma = float(m_gamma);

        for (int y = 0; y < h; ++y) {
            if (ctx.Cancelled()) return;
            for (int x = 0; x < w; ++x) {
                const int c = CfaColorAt(cfa, x, y);

                // Bilinear terms, then the gradient correction.
                const float centre = at(x, y);
                const float hSum = at(x - 1, y) + at(x + 1, y);
                const float vSum = at(x, y - 1) + at(x, y + 1);
                const float xSum = at(x - 1, y - 1) + at(x + 1, y - 1) +
                                   at(x - 1, y + 1) + at(x + 1, y + 1);

                // The centre channel's second derivative in each direction:
                // 4*C - C(±2) is the discrete Laplacian the correction uses.
                const float lapH = 2.0f * centre - at(x - 2, y) - at(x + 2, y);
                const float lapV = 2.0f * centre - at(x, y - 2) - at(x, y + 2);
                const float lap  = lapH + lapV;

                float rgb[3] = {0, 0, 0};
                rgb[c] = centre;

                if (c == 1) {
                    // Green site. The horizontal neighbours are one colour and
                    // the vertical ones the other; which is which depends on the
                    // row, so ask the pattern rather than assuming.
                    const int horizColor = CfaColorAt(cfa, x - 1, y);

                    // The paper's kernel is asymmetric here: the colour being
                    // interpolated lies along one axis, so its own axis gets
                    // 5/8 of the centre's Laplacian while the perpendicular one
                    // gets 1/8 less. Expressed against lapH/lapV directly
                    // rather than as a correction to a symmetric term.
                    rgb[horizColor]     = 0.5f * hSum + gamma * (5.0f * lapH - lapV);
                    rgb[2 - horizColor] = 0.5f * vSum + gamma * (5.0f * lapV - lapH);
                } else {
                    // Red or blue site. Green from the 4 edge neighbours plus
                    // the centre's Laplacian -- this is the term bilinear
                    // discards, and the reason edges come out cleaner.
                    rgb[1] = 0.25f * (hSum + vSum) + alpha * lap;

                    // The opposite colour from the 4 corners, with a stronger
                    // correction: it is sampled at half the density of green,
                    // so it has more to gain from the centre channel.
                    rgb[2 - c] = 0.25f * xSum + beta * lap;
                }

                ApplyColour(src.desc, rgb);

                uint16_t* p = dst.At<uint16_t>(x, y);
                p[0] = FloatToHalf(std::max(rgb[0], 0.0f));
                p[1] = FloatToHalf(std::max(rgb[1], 0.0f));
                p[2] = FloatToHalf(std::max(rgb[2], 0.0f));
                p[3] = FloatToHalf(1.0f);
            }
        }
    }

    // --- GPU implementation -------------------------------------------------
    // A wider stencil than bilinear (5x5 rather than 3x3) but still a pure
    // local filter with no cross-pixel dependency.
    //
    // Unconditional, deliberately: HasGPU() is consulted before PrepareGpu(),
    // so anything derived from the input is still at its default here. The
    // shader handles the non-Bayer case instead.
    bool HasGPU() const override { return true; }

    void PrepareGpu(const std::vector<ImageDesc>& inputs) override {
        if (inputs.empty()) return;
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
    uint Cfa;
    uint BlackBits;
    uint RangeBits;
    uint AlphaBits, BetaBits, GammaBits;
    uint CamMul0, CamMul1, CamMul2;
    uint M0, M1, M2, M3, M4, M5, M6, M7, M8;
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

    int x = int(tid.x), y = int(tid.y);

    if (Cfa == 0) {
        float4 t = Src[int2(tid.xy)];
        Dst[tid.xy] = float4(t.rgb, 1.0);
        return;
    }

    float black = asfloat(BlackBits);
    float range = asfloat(RangeBits);
    int2  hi    = int2(Width - 1, Height - 1);

    #define S(dx, dy) clamp((Src[clamp(int2(x + (dx), y + (dy)), int2(0,0), hi)].x - black) / range, 0.0, 4.0)

    float centre = S(0, 0);
    float hSum = S(-1, 0) + S(1, 0);
    float vSum = S(0, -1) + S(0, 1);
    float xSum = S(-1, -1) + S(1, -1) + S(-1, 1) + S(1, 1);

    // The centre channel's second derivative: what bilinear discards.
    float lapH = 2.0 * centre - S(-2, 0) - S(2, 0);
    float lapV = 2.0 * centre - S(0, -2) - S(0, 2);
    float lap  = lapH + lapV;
    #undef S

    float alpha = asfloat(AlphaBits);
    float beta  = asfloat(BetaBits);
    float gamma = asfloat(GammaBits);

    int c = CfaColor(Cfa, x, y);
    float3 rgb = float3(0, 0, 0);
    rgb[c] = centre;

    if (c == 1) {
        int horizColor = CfaColor(Cfa, x - 1, y);
        rgb[horizColor]     = 0.5 * hSum + gamma * (5.0 * lapH - lapV);
        rgb[2 - horizColor] = 0.5 * vSum + gamma * (5.0 * lapV - lapH);
    } else {
        rgb[1]     = 0.25 * (hSum + vSum) + alpha * lap;
        rgb[2 - c] = 0.25 * xSum + beta * lap;
    }

    // Which channels saturated is decided here, against the sensor's white
    // level, but the repair happens AFTER white balance -- see RecoverClipped
    // in the CPU path. Equalising first and then applying the gains makes the
    // magenta worse, not better.
    const float kClip = 0.99;
    bool3 clipped = bool3(rgb.r >= kClip, rgb.g >= kClip, rgb.b >= kClip);

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
        std::vector<uint32_t> c{uint32_t(m_cfa), bits(m_black), bits(m_range),
                                bits(float(m_alpha)), bits(float(m_beta)),
                                bits(float(m_gamma))};
        for (int i = 0; i < 3; ++i) c.push_back(bits(m_camMul[i]));
        for (int i = 0; i < 9; ++i) c.push_back(bits(m_rgbCam[i]));
        return c;
    }

private:
    static void ApplyColour(const ImageDesc& d, float* rgb) {
        // Which channels saturated must be decided BEFORE white balance --
        // that is the only place the sensor's white level means anything --
        // but the repair must happen AFTER, in the space where "neutral"
        // means the channels are equal. Doing both before simply hands the
        // gains an equal triple to pull apart again, which is the bug.
        const bool clipped[3] = {rgb[0] >= kClip, rgb[1] >= kClip, rgb[2] >= kClip};

        rgb[0] *= d.camMul[0];
        rgb[1] *= d.camMul[1];
        rgb[2] *= d.camMul[2];

        RecoverClipped(rgb, clipped);

        const float r = rgb[0], g = rgb[1], b = rgb[2];
        rgb[0] = d.rgbCam[0] * r + d.rgbCam[1] * g + d.rgbCam[2] * b;
        rgb[1] = d.rgbCam[3] * r + d.rgbCam[4] * g + d.rgbCam[5] * b;
        rgb[2] = d.rgbCam[6] * r + d.rgbCam[7] * g + d.rgbCam[8] * b;

        for (int i = 0; i < 3; ++i) rgb[i] = std::max(rgb[i], 0.0f);
    }

    // A sensel at the white level is not a measurement, it is a lower bound:
    // the scene was at least this bright, and how much brighter is unknowable.
    // Treating that bound as data and applying per-channel white-balance gains
    // to it is what turns a blown highlight magenta. On Tim's ARW the gains are
    // R 1.578, G 1.000, B 2.961, so a clipped white sensel comes out with green
    // crushed and red and blue running away -- the pink he saw on the
    // over-exposed lights, before any develop slider was touched.
    //
    // Called AFTER white balance, with a mask captured before it. In the
    // balanced space a neutral highlight is one where the channels are equal,
    // so lifting the clipped ones to the brightest present restores that.
    // A pixel where all three clipped becomes neutral; one where only red
    // clipped keeps its hue from the channels that still hold information.
    //
    // The first version of this ran BEFORE white balance and made the cast
    // WORSE, measured 1.95/1.00/1.32 to 2.34/1.00/1.72: equalising and then
    // multiplying by the gains simply hands them a neutral triple to pull
    // apart again. The order is the whole point.
    //
    // It cannot invent the true brightness -- nothing can -- but
    // neutral-and-blown is what the eye expects from a blown highlight, and
    // magenta is simply wrong.
    static void RecoverClipped(float* rgb, const bool (&clipped)[3]) {
        if (!clipped[0] && !clipped[1] && !clipped[2]) return;   // the common case
        const float hi = std::max(rgb[0], std::max(rgb[1], rgb[2]));
        if (clipped[0]) rgb[0] = hi;
        if (clipped[1]) rgb[1] = hi;
        if (clipped[2]) rgb[2] = hi;
    }

    // Normalised so 1.0 is the sensor's white level (the demosaic divides by
    // the black-to-white range first). A little under 1 to catch sensels that
    // saturate slightly early, which real ones do -- the ARW's own maximum
    // reads 16596 against a declared white of 16383.
    static constexpr float kClip = 0.99f;

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

    // The paper's least-squares-optimal gains. Exposed rather than baked in so
    // the derivation can be tested: set all three to 0 and this becomes exactly
    // bilinear, which is a useful thing to be able to demonstrate.
    Param<float> m_alpha{
        this, "alpha", 0.125f, 0.0f, 1.0f,
        {.help = "Correction strength for green at a red or blue site. 0 makes "
                 "this exactly bilinear; the paper's optimal value is 0.125 "
                 "(the 1/8 kernel).",
         .step = 0.01, .softMin = 0.0, .softMax = 1.0}};
    Param<float> m_beta{
        this, "beta", 0.1875f, 0.0f, 1.0f,
        {.help = "Correction strength for red at a blue site and vice versa. "
                 "Higher than alpha because those colours are sampled at half "
                 "green's density, so they gain more from the centre channel.",
         .step = 0.01, .softMin = 0.0, .softMax = 1.0}};
    Param<float> m_gamma{
        this, "gamma", 0.0625f, 0.0f, 0.5f,
        {.help = "Correction strength at a green site, where both red and blue "
                 "are interpolated. Lowest of the three: green is the densest "
                 "channel, so its neighbours are already close.",
         .step = 0.005, .softMin = 0.0, .softMax = 0.5}};

    int   m_cfa   = 0;
    float m_black = 0.0f;
    float m_range = 1.0f;
    float m_camMul[3] = {1.0f, 1.0f, 1.0f};
    float m_rgbCam[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    PixelBuffer        m_in;
    std::vector<float> m_s;
};

REGISTER_ALGORITHM(DemosaicMalvar);

} // namespace tglab
