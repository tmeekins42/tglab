// demosaic_ppg — Patterned Pixel Grouping (Chuan-kai Lin).
//
// Three passes, and the structure is the point: green everywhere first, then
// red and blue at the sites that are missing them, then the diagonals. Each
// pass uses the result of the one before, so by the time red is interpolated
// the green plane is already complete and can be leaned on.
//
// THE ASSUMPTION IT RESTS ON is that COLOUR DIFFERENCES vary more slowly across
// an image than the colours themselves. A red leaf against green grass has a
// violent R and G gradient at the boundary, but R-G is roughly constant within
// the leaf and within the grass -- the difference tracks the MATERIAL, while
// the absolute values track the material AND the light falling on it.
//
// So PPG never interpolates red directly. It interpolates R-G, which is smooth,
// and adds back the green it already knows exactly. That is why it produces so
// little false colour for how simple it is.
//
// HOW IT DIFFERS FROM THE OTHERS HERE, which is what makes it worth comparing:
//
//   bilinear  averages each channel independently -- no gradient sense at all.
//   malvar    one fixed kernel with a gradient correction, same everywhere.
//   ahd       picks horizontal OR vertical per pixel, then filters homogeneity.
//   vng       averages over however many of eight directions look smooth.
//   ppg       picks a direction for GREEN from a hue-aware gradient, then
//             rides colour differences for everything else.
//
// PPG's green gradient is the interesting part. Rather than measuring roughness
// on green alone, it mixes in the CENTRE channel's second derivative:
//
//     grad_H = |G(x-1) - G(x+1)| * 2 + |C(x-2) - 2*C(x) + C(x+2)|
//
// The first term is the green gradient; the second says whether the centre
// colour is curving, which catches a feature the green samples happen to
// straddle evenly. A pure-green measure is blind to exactly that case, and it
// is common -- a thin red line lying between two green photosites of equal
// value looks perfectly flat to green alone.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "clip_repair.h"

#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"

namespace tglab {

class DemosaicPpg : public AlgorithmBase {
public:
    const char* Name()     const override { return "demosaic_ppg"; }
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

        if (cfa == CfaPattern::None || cfa == CfaPattern::XTrans) {
            PassThrough(dst, w, h);
            return;
        }

        const float black = src.desc.blackLevel;
        const float range = std::max(src.desc.whiteLevel - black, 1e-6f);
        const size_t n = size_t(w) * size_t(h);

        m_s.assign(n, 0.0f);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                m_s[size_t(y) * size_t(w) + size_t(x)] =
                    std::clamp((m_in.Get(x, y, 0) - black) / range, 0.0f, 4.0f);

        auto S = [&](int x, int y) -> float {
            x = std::clamp(x, 0, w - 1);
            y = std::clamp(y, 0, h - 1);
            return m_s[size_t(y) * size_t(w) + size_t(x)];
        };

        // --- pass 1: green everywhere ---------------------------------------
        //
        // Green first because it is sampled twice as densely as the others, so
        // it carries the luminance detail and is the most reliable thing to
        // build the rest on.
        m_g.assign(n, 0.0f);
        for (int y = 0; y < h; ++y) {
            if (ctx.Cancelled()) return;
            for (int x = 0; x < w; ++x) {
                const size_t i = size_t(y) * size_t(w) + size_t(x);
                if (CfaColorAt(cfa, x, y) == 1) { m_g[i] = S(x, y); continue; }

                // Gradients that mix green roughness with the centre channel's
                // curvature -- see the note at the top for why the second term
                // matters.
                const float gH = 2.0f * std::fabs(S(x - 1, y) - S(x + 1, y)) +
                                 std::fabs(S(x - 2, y) - 2.0f * S(x, y) + S(x + 2, y));
                const float gV = 2.0f * std::fabs(S(x, y - 1) - S(x, y + 1)) +
                                 std::fabs(S(x, y - 2) - 2.0f * S(x, y) + S(x, y + 2));

                // A BAND rather than exact equality, because the tie branch is
                // reachable and the exact test is not reproducible.
                //
                // 0.262% of red/blue sites on a real frame have gH == gV to the
                // bit, and CPU and GPU do not order floating-point operations
                // identically -- so a pixel could take the horizontal branch on
                // one and the tie branch on the other, differing by |gH-gV|/2,
                // which at an edge is large. Measured before this: the two paths
                // disagreed on 0.072% of samples with a worst case of 0.167.
                //
                // The band is relative, so it means the same thing in shadow as
                // in highlight, and it costs nothing where the two gradients
                // genuinely differ.
                const float tie = 1e-4f * std::max(gH, gV);

                float g;
                if (gH < gV - tie) {
                    g = 0.5f * (S(x - 1, y) + S(x + 1, y)) +
                        0.25f * (2.0f * S(x, y) - S(x - 2, y) - S(x + 2, y));
                } else if (gV < gH - tie) {
                    g = 0.5f * (S(x, y - 1) + S(x, y + 1)) +
                        0.25f * (2.0f * S(x, y) - S(x, y - 2) - S(x, y + 2));
                } else {
                    // A genuine tie -- flat, or equally rough both ways. Use
                    // both rather than picking arbitrarily, which would make
                    // the result depend on floating-point noise.
                    g = 0.25f * (S(x - 1, y) + S(x + 1, y) + S(x, y - 1) + S(x, y + 1)) +
                        0.125f * (4.0f * S(x, y) - S(x - 2, y) - S(x + 2, y) -
                                  S(x, y - 2) - S(x, y + 2));
                }

                // Bounded by the samples it came from. Without this the
                // curvature term overshoots at a hard edge, which is the
                // ringing that makes naive sharpening unpleasant.
                const float lo = std::min(std::min(S(x - 1, y), S(x + 1, y)),
                                          std::min(S(x, y - 1), S(x, y + 1)));
                const float hi = std::max(std::max(S(x - 1, y), S(x + 1, y)),
                                          std::max(S(x, y - 1), S(x, y + 1)));
                m_g[i] = std::clamp(g, lo, hi);
            }
        }

        auto G = [&](int x, int y) -> float {
            x = std::clamp(x, 0, w - 1);
            y = std::clamp(y, 0, h - 1);
            return m_g[size_t(y) * size_t(w) + size_t(x)];
        };

        // --- passes 2 and 3: red and blue, through colour differences -------
        for (int y = 0; y < h; ++y) {
            if (ctx.Cancelled()) return;
            for (int x = 0; x < w; ++x) {
                const int c = CfaColorAt(cfa, x, y);
                float rgb[3];
                rgb[1] = G(x, y);

                if (c == 1) {
                    // A green site. Its horizontal and vertical neighbours are
                    // the other two colours -- which is which depends on the
                    // row, so ask the pattern rather than assuming.
                    const int hc = CfaColorAt(cfa, x - 1, y);
                    rgb[hc] = rgb[1] + 0.5f * ((S(x - 1, y) - G(x - 1, y)) +
                                               (S(x + 1, y) - G(x + 1, y)));
                    rgb[2 - hc] = rgb[1] + 0.5f * ((S(x, y - 1) - G(x, y - 1)) +
                                                   (S(x, y + 1) - G(x, y + 1)));
                } else {
                    // A red or blue site: its own colour is measured, and the
                    // opposite one comes from the four diagonals.
                    rgb[c] = S(x, y);

                    // Hue-aware diagonal choice, the same idea as the green
                    // pass: interpolate the opposite colour ALONG whichever
                    // diagonal is smoother in colour-difference terms.
                    const int o = 2 - c;
                    const float dA = std::fabs((S(x - 1, y - 1) - G(x - 1, y - 1)) -
                                               (S(x + 1, y + 1) - G(x + 1, y + 1)));
                    const float dB = std::fabs((S(x + 1, y - 1) - G(x + 1, y - 1)) -
                                               (S(x - 1, y + 1) - G(x - 1, y + 1)));
                    if (dA < dB) {
                        rgb[o] = rgb[1] + 0.5f * ((S(x - 1, y - 1) - G(x - 1, y - 1)) +
                                                  (S(x + 1, y + 1) - G(x + 1, y + 1)));
                    } else if (dB < dA) {
                        rgb[o] = rgb[1] + 0.5f * ((S(x + 1, y - 1) - G(x + 1, y - 1)) +
                                                  (S(x - 1, y + 1) - G(x - 1, y + 1)));
                    } else {
                        rgb[o] = rgb[1] + 0.25f * ((S(x - 1, y - 1) - G(x - 1, y - 1)) +
                                                   (S(x + 1, y + 1) - G(x + 1, y + 1)) +
                                                   (S(x + 1, y - 1) - G(x + 1, y - 1)) +
                                                   (S(x - 1, y + 1) - G(x - 1, y + 1)));
                    }
                }

                // Bound each reconstruction by the real samples of that colour
                // nearby, for the same reason the green pass is bounded.
                for (int k = 0; k < 3; ++k) {
                    if (k == c || k == 1) continue;
                    float klo = 1e30f, khi = -1e30f;
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx) {
                            if (CfaColorAt(cfa, x + dx, y + dy) != k) continue;
                            const float v = S(x + dx, y + dy);
                            klo = std::min(klo, v);
                            khi = std::max(khi, v);
                        }
                    if (khi >= klo) rgb[k] = std::clamp(rgb[k], klo, khi);
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
    //
    // Two passes, mirroring the CPU structure exactly: green everywhere, then
    // red and blue from colour differences against that green. It has to be two
    // dispatches rather than one because the second pass reads its NEIGHBOURS'
    // green, not just its own -- and threads within a dispatch have no ordering,
    // so a single kernel would read whatever green happened to be written
    // already. That is the same reason gaussian_blur is separable in two passes
    // rather than one.
    bool HasGPU() const override { return true; }

    int        GpuScratchCount()  const override { return 1; }
    FormatSpec GpuScratchPlanes() const override { return FormatSpec::RGBA32F; }

    std::vector<GpuPass> GpuPasses() const override {
        // Assembled once: GpuPasses returns raw pointers and is called per run,
        // so rebuilding the strings each time would dangle them.
        static const std::string grn =
            std::string(kCommon) + kClipRepairHlsl + kGreenHlsl;
        static const std::string col =
            std::string(kCommon) + kClipRepairHlsl + kColourHlsl;

        std::vector<GpuPass> p;
        // t0 is the mosaic throughout, because kCommon's Sample() reads it.
        p.push_back({grn.c_str(), "green",  {-1},    {0}});
        p.push_back({col.c_str(), "colour", {-1, 0}, {-1}});
        return p;
    }

    std::vector<uint32_t> GpuPassConstants(int) const override {
        auto bits = [](float f) { uint32_t u; std::memcpy(&u, &f, sizeof u); return u; };
        std::vector<uint32_t> c{uint32_t(m_cfa), bits(m_black), bits(m_range)};
        for (int i = 0; i < 3; ++i) c.push_back(bits(m_camMul[i]));
        for (int i = 0; i < 9; ++i) c.push_back(bits(m_rgbCam[i]));
        return c;
    }

    // The shader reads the CFA pattern and sensor levels from the IMAGE, not
    // from any parameter, and the GPU path never calls RunCPU -- so without
    // this the kernel would run on stale values and produce a plausible image
    // with the wrong colours.
    void PrepareGpu(const std::vector<ImageDesc>& inputs) override {
        if (inputs.empty()) return;
        const ImageDesc& d = inputs[0];
        const int cfa = int(d.cfa);
        m_cfa   = (cfa >= 1 && cfa <= 4) ? cfa : 0;
        m_black = d.blackLevel;
        m_range = std::max(d.whiteLevel - d.blackLevel, 1e-6f);
        for (int i = 0; i < 3; ++i) m_camMul[i] = d.camMul[i];
        for (int i = 0; i < 9; ++i) m_rgbCam[i] = d.rgbCam[i];
    }

private:
    static constexpr const char* kCommon = R"(
Texture2D<float4>   T0 : register(t0);
Texture2D<float4>   T1 : register(t1);
RWTexture2D<float4> U0 : register(u0);

cbuffer Params : register(b0) {
    uint Width;
    uint Height;
    uint Cfa;
    uint BlackBits;
    uint RangeBits;
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

int2 ClampXY(int x, int y) {
    return clamp(int2(x, y), int2(0, 0), int2(int(Width) - 1, int(Height) - 1));
}

// The raw mosaic, normalised. Clamped at 4.0 like the CPU path so a hot pixel
// cannot dominate a gradient.
float S(int x, int y) {
    return clamp((T0[ClampXY(x, y)].x - asfloat(BlackBits)) / asfloat(RangeBits),
                 0.0, 4.0);
}

// The green plane from pass 1.
float G(int x, int y) { return T1[ClampXY(x, y)].g; }
)";

    // Pass 1: green everywhere.
    //
    // Green first because it is sampled twice as densely as the others, so it
    // carries the luminance detail and is the most reliable thing to build the
    // rest on. The gradients mix green roughness with the CENTRE channel's
    // curvature -- see the note at the top of the file for why a pure-green
    // measure is blind to a thin feature the green samples straddle evenly.
    static constexpr const char* kGreenHlsl = R"(
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    int x = int(tid.x), y = int(tid.y);

    if (Cfa == 0) { U0[tid.xy] = float4(S(x, y).xxx, 1.0); return; }

    if (CfaColor(Cfa, x, y) == 1) {
        U0[tid.xy] = float4(0.0, S(x, y), 0.0, 1.0);
        return;
    }

    float gH = 2.0 * abs(S(x - 1, y) - S(x + 1, y)) +
               abs(S(x - 2, y) - 2.0 * S(x, y) + S(x + 2, y));
    float gV = 2.0 * abs(S(x, y - 1) - S(x, y + 1)) +
               abs(S(x, y - 2) - 2.0 * S(x, y) + S(x, y + 2));

    // A band rather than exact equality -- see the CPU path.
    float tie = 1e-4 * max(gH, gV);

    float g;
    if (gH < gV - tie) {
        g = 0.5 * (S(x - 1, y) + S(x + 1, y)) +
            0.25 * (2.0 * S(x, y) - S(x - 2, y) - S(x + 2, y));
    } else if (gV < gH - tie) {
        g = 0.5 * (S(x, y - 1) + S(x, y + 1)) +
            0.25 * (2.0 * S(x, y) - S(x, y - 2) - S(x, y + 2));
    } else {
        // A genuine tie -- flat, or equally rough both ways. Use both rather
        // than picking arbitrarily, which would make the result depend on
        // floating-point noise.
        g = 0.25 * (S(x - 1, y) + S(x + 1, y) + S(x, y - 1) + S(x, y + 1)) +
            0.125 * (4.0 * S(x, y) - S(x - 2, y) - S(x + 2, y) -
                     S(x, y - 2) - S(x, y + 2));
    }

    // Bounded by the samples it came from: without this the curvature term
    // overshoots at a hard edge, which is the ringing that makes naive
    // sharpening unpleasant.
    float lo = min(min(S(x - 1, y), S(x + 1, y)), min(S(x, y - 1), S(x, y + 1)));
    float hi = max(max(S(x - 1, y), S(x + 1, y)), max(S(x, y - 1), S(x, y + 1)));
    U0[tid.xy] = float4(0.0, clamp(g, lo, hi), 0.0, 1.0);
}
)";

    // Pass 2: red and blue through colour differences, then the colour step.
    static constexpr const char* kColourHlsl = R"(
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    int x = int(tid.x), y = int(tid.y);

    if (Cfa == 0) { U0[tid.xy] = float4(S(x, y).xxx, 1.0); return; }

    int c = CfaColor(Cfa, x, y);
    float3 rgb;
    rgb.g = G(x, y);

    if (c == 1) {
        // A green site: the horizontal and vertical neighbours are the other
        // two colours, and which is which depends on the row.
        int hc = CfaColor(Cfa, x - 1, y);
        float h = rgb.g + 0.5 * ((S(x - 1, y) - G(x - 1, y)) +
                                 (S(x + 1, y) - G(x + 1, y)));
        float v = rgb.g + 0.5 * ((S(x, y - 1) - G(x, y - 1)) +
                                 (S(x, y + 1) - G(x, y + 1)));
        rgb.r = (hc == 0) ? h : v;
        rgb.b = (hc == 0) ? v : h;
    } else {
        // A red or blue site: its own colour is measured, the opposite one
        // comes from the diagonals -- along whichever is smoother in
        // colour-difference terms.
        float own = S(x, y);
        float dA = abs((S(x - 1, y - 1) - G(x - 1, y - 1)) -
                       (S(x + 1, y + 1) - G(x + 1, y + 1)));
        float dB = abs((S(x + 1, y - 1) - G(x + 1, y - 1)) -
                       (S(x - 1, y + 1) - G(x - 1, y + 1)));
        float opp;
        if (dA < dB) {
            opp = rgb.g + 0.5 * ((S(x - 1, y - 1) - G(x - 1, y - 1)) +
                                 (S(x + 1, y + 1) - G(x + 1, y + 1)));
        } else if (dB < dA) {
            opp = rgb.g + 0.5 * ((S(x + 1, y - 1) - G(x + 1, y - 1)) +
                                 (S(x - 1, y + 1) - G(x - 1, y + 1)));
        } else {
            opp = rgb.g + 0.25 * ((S(x - 1, y - 1) - G(x - 1, y - 1)) +
                                  (S(x + 1, y + 1) - G(x + 1, y + 1)) +
                                  (S(x + 1, y - 1) - G(x + 1, y - 1)) +
                                  (S(x - 1, y + 1) - G(x - 1, y + 1)));
        }
        rgb.r = (c == 0) ? own : opp;
        rgb.b = (c == 0) ? opp : own;
    }

    // Bound each reconstruction by the real samples of that colour nearby, for
    // the same reason the green pass is bounded.
    [unroll] for (int k = 0; k < 3; ++k) {
        if (k == c || k == 1) continue;
        float klo = 1e30, khi = -1e30;
        [unroll] for (int dy = -1; dy <= 1; ++dy)
            [unroll] for (int dx = -1; dx <= 1; ++dx) {
                if (CfaColor(Cfa, x + dx, y + dy) != k) continue;
                float v2 = S(x + dx, y + dy);
                klo = min(klo, v2);
                khi = max(khi, v2);
            }
        if (khi >= klo) rgb[k] = clamp(rgb[k], klo, khi);
    }

    ApplyColour(rgb,
                float3(asfloat(CamMul0), asfloat(CamMul1), asfloat(CamMul2)),
                float3x3(asfloat(M0), asfloat(M1), asfloat(M2),
                         asfloat(M3), asfloat(M4), asfloat(M5),
                         asfloat(M6), asfloat(M7), asfloat(M8)));

    // Negatives NOT clamped -- see ApplyColour in clip_repair.h.
    U0[tid.xy] = float4(rgb, 1.0);
}
)";

    // Sensor metadata, captured by PrepareGpu.
    int   m_cfa   = 0;
    float m_black = 0.0f;
    float m_range = 1.0f;
    float m_camMul[3] = {1.0f, 1.0f, 1.0f};
    float m_rgbCam[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    void PassThrough(ImageView& dst, int w, int h) {
        const int ch = m_in.Channels();
        const float scale = m_in.ValueScale();
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                uint16_t* p = dst.At<uint16_t>(x, y);
                for (int c = 0; c < 3; ++c)
                    p[c] = FloatToHalf(m_in.Get(x, y, ch == 1 ? 0 : c) / scale);
                p[3] = FloatToHalf(1.0f);
            }
    }

    PixelBuffer        m_in;
    std::vector<float> m_s, m_g;
};

REGISTER_ALGORITHM(DemosaicPpg);

}  // namespace tglab
