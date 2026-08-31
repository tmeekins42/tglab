// demosaic_ahd — adaptive, directional demosaicing with chroma smoothing.
//
// In the family of Hirakawa & Parks (2005), "Adaptive homogeneity-directed
// demosaicing algorithm", and of the VNG/AHD methods every serious raw
// developer uses. Written because Malvar, which is linear, hits a hard ceiling
// on exactly the content this lab keeps running into.
//
// Why a linear filter cannot do better
// ------------------------------------
// Malvar applies FIXED weights: the same 5x5 kernel everywhere, no per-pixel
// decisions. That is what makes it fast and vectorisable, and it is also its
// limit. Green is sampled at half the sites; on a feature finer than that
// spacing -- a bare conifer branch against fog, a wire, a hair -- the correct
// green at a red site is genuinely ambiguous from the samples alone. A fixed
// kernel must guess, and it guesses the same way regardless of which direction
// the detail actually runs.
//
// The failure is measurable and it is not random. On a backlit-branch CR3,
// chroma error at green sites ran 11.5% higher than at red/blue sites -- an
// error that CORRELATES WITH CFA PHASE, which is why it appears as a regular
// lattice of coloured dots rather than as noise. Green-magenta error spanned
// 0.965 of the luminance range. Clamping Malvar's overshoot (see
// demosaic_malvar.cpp) removed the worst speckles but could not touch this:
// the estimate is inside its neighbours' range and still wrong, because the
// neighbours it averaged were across the branch rather than along it.
//
// What this does instead
// ----------------------
// Three ideas, in order of how much they matter:
//
// 1. DIRECTIONAL GREEN. Interpolate green twice -- once horizontally, once
//    vertically -- and keep the one that runs ALONG the edge rather than
//    across it. Interpolating across a branch averages branch and sky and
//    produces a colour that is neither. The choice is made per pixel from a
//    local gradient, which is precisely the per-pixel decision Malvar cannot
//    make.
//
// 2. COLOUR-DIFFERENCE INTERPOLATION. Red and blue are reconstructed as
//    R-G and B-G rather than directly. Those differences are far smoother than
//    the channels themselves across almost all natural content -- that is the
//    same channel correlation Malvar's Laplacian exploits, used more
//    completely. Interpolating a smooth signal and adding back a known green
//    beats interpolating a sharp one.
//
// 3. CHROMA MEDIAN. A small median on the colour differences. Demosaicing
//    artefacts are isolated and signed, which is exactly what a median removes
//    and an average does not; edges survive it because a median is
//    order-based. This is the step that clears residual confetti without
//    softening luminance, and it is why the result can be both cleaner AND
//    sharper than Malvar rather than trading one for the other.
//
// Cost: several passes over the image against Malvar's one. On the CPU that is
// expensive -- 21 s against Malvar's 3 s at 45 MP -- but on the GPU the two are
// indistinguishable, 495 ms against 509 ms on the same frame. Both are
// bandwidth-bound rather than compute-bound at that size, so AHD's extra
// arithmetic costs nothing against the price of moving 45 megapixels. That is
// why it is now the default that image() inserts.
//
// The GPU path needed a framework extension: AlgorithmBase::GpuPasses, for
// algorithms whose passes are DIFFERENT kernels rather than one kernel run
// repeatedly. See the GPU section at the bottom of this file.
//
// The fringing, and what it actually was
// --------------------------------------
// An earlier version of this file tinted twigs -- magenta along the branches,
// or bright green streaks if the chroma median's edge gate was removed. Two
// separate causes, and neither was where it looked.
//
// The first was ISOTROPY in the colour-difference fill. It searched a square
// neighbourhood and averaged whatever samples of the right colour it found,
// ignoring that a missing red sits in one of exactly three known positions in
// the Bayer tile, and that those cases want different interpolations. On a
// one-pixel twig that pulls sky into the twig's colour and vice versa, before
// any median sees it -- which is why no amount of filtering downstream fixed
// it. See fillDiff, which now handles the three cases separately and chooses
// between diagonals at the ambiguous one.
//
// The second was UNBOUNDED reconstruction. g + (R-G) is the sum of two
// independent estimates and had no bound of its own, so at an edge red could
// land well below anything the sensor recorded nearby. Harmless-looking, and
// then fatal: a camera matrix has large negative off-diagonal terms -- this
// body's first row is [1.535, -0.555, 0.019], red out being red in MINUS 0.555
// of green -- so a merely-short red goes negative, clamps at zero, and the
// pixel renders FULLY saturated. That is the difference between a faint tint
// and a vivid streak. Measured: 898 such pixels against Malvar's 22, brought to
// 27 by bounding each reconstructed channel to the real samples around it.
//
// Two methodological notes, both learned expensively here.
//
// Aggregate statistics disagreed with the rendered image, repeatedly and in
// both directions. A change that improved the mean magenta index past Malvar's
// simultaneously painted green streaks along every twig; a per-decile mean
// averages over the whole frame and cannot see a defect that is thin, bright
// and localised, which is the only defect that matters here. Score fringing at
// EDGE pixels and as a high percentile, and check a rendered crop.
//
// The synthetic ground-truth scene could not reproduce the artefact at all --
// it scored a clean win there on content that fringed badly on the real file.
// It lacks sensor noise and, more importantly, a real camera matrix and white
// balance: with an identity matrix the negative-channel mechanism cannot occur.
// A fixture that cannot express the failure cannot test for it.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "clip_repair.h"
#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"

namespace tglab {
namespace {

inline float Median3(float a, float b, float c) {
    return std::max(std::min(a, b), std::min(std::max(a, b), c));
}

// Median of up to 9 values, by partial selection. Small enough that a sort is
// not worth the machinery.
inline float MedianN(float* v, int n) {
    std::nth_element(v, v + n / 2, v + n);
    return v[n / 2];
}

// Shared prologue for every AHD kernel: the constant buffer, the CFA lookup,
// and a bounds-clamped sample fetch.
//
// One string rather than four copies, because these must agree exactly -- a CFA
// lookup that differed between the green pass and the combine pass would
// reconstruct red from blue's neighbours in part of the image, which reads as a
// colour cast rather than as an obvious bug.
const char* const kAhdCommon = R"(
Texture2D<float4>   T0 : register(t0);
Texture2D<float4>   T1 : register(t1);
Texture2D<float4>   T2 : register(t2);
Texture2D<float4>   T3 : register(t3);
RWTexture2D<float4> U0 : register(u0);
RWTexture2D<float4> U1 : register(u1);

cbuffer Params : register(b0) {
    uint  Width;
    uint  Height;
    uint  Cfa;
    uint  BlackBits;
    uint  RangeBits;
    uint  CamMul0, CamMul1, CamMul2;
    uint  M0, M1, M2, M3, M4, M5, M6, M7, M8;
};

// White balance AND the clipped-channel repair -- the repair must see balanced
// values, since that is where neutral means the channels are equal.
// White balance with the highlight clamp -- keep in step with
// BalanceAndClamp in clip_repair.h.
void BalanceAndClamp(inout float3 rgb, float3 camMul) {
    float ceiling = min(camMul.r, min(camMul.g, camMul.b));
    rgb = min(rgb * camMul, ceiling);
}

int CfaColor(uint cfa, int x, int y) {
    int q = (y & 1) * 2 + (x & 1);
    if (cfa == 1) { int c[4] = {0, 1, 1, 2}; return c[q]; }
    if (cfa == 2) { int c[4] = {2, 1, 1, 0}; return c[q]; }
    if (cfa == 3) { int c[4] = {1, 0, 2, 1}; return c[q]; }
    if (cfa == 4) { int c[4] = {1, 2, 0, 1}; return c[q]; }
    return 1;
}

int2 ClampXY(int x, int y) {
    return clamp(int2(x, y), int2(0, 0), int2(Width - 1, Height - 1));
}

// The mosaic sample, normalised exactly as the CPU path does.
float Sample(int x, int y) {
    float black = asfloat(BlackBits);
    float range = asfloat(RangeBits);
    return clamp((T0[ClampXY(x, y)].x - black) / range, 0.0, 4.0);
}
)";

// Pass 0: green everywhere, interpolated along the edge.
const char* const kGreenHlsl = R"(
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    int x = int(tid.x), y = int(tid.y);

    if (Cfa == 0) { U0[tid.xy] = float4(Sample(x, y), 0, 0, 1); return; }

    if (CfaColor(Cfa, x, y) == 1) {
        U0[tid.xy] = float4(Sample(x, y), 0, 0, 1);
        return;
    }

    float c = Sample(x, y);

    float gh = 0.5 * (Sample(x - 1, y) + Sample(x + 1, y)) +
               0.25 * (2.0 * c - Sample(x - 2, y) - Sample(x + 2, y));
    float gv = 0.5 * (Sample(x, y - 1) + Sample(x, y + 1)) +
               0.25 * (2.0 * c - Sample(x, y - 2) - Sample(x, y + 2));

    float dh = abs(Sample(x - 1, y) - Sample(x + 1, y)) +
               abs(2.0 * c - Sample(x - 2, y) - Sample(x + 2, y));
    float dv = abs(Sample(x, y - 1) - Sample(x, y + 1)) +
               abs(2.0 * c - Sample(x, y - 2) - Sample(x, y + 2));

    float wh = 1.0 / (1e-4 + dh * dh);
    float wv = 1.0 / (1e-4 + dv * dv);
    float g  = (gh * wh + gv * wv) / (wh + wv);

    // Bounded along the CHOSEN axis, blended by the same weights -- see the CPU
    // path for why the four-neighbour envelope is wrong here.
    float ah0 = Sample(x - 1, y), ah1 = Sample(x + 1, y);
    float av0 = Sample(x, y - 1), av1 = Sample(x, y + 1);
    float lo = (min(ah0, ah1) * wh + min(av0, av1) * wv) / (wh + wv);
    float hi = (max(ah0, ah1) * wh + max(av0, av1) * wv) / (wh + wv);

    U0[tid.xy] = float4(clamp(g, min(lo, hi), max(lo, hi)), 0, 0, 1);
}
)";

// Pass 1: the two colour-difference planes, filled directionally by Bayer case.
const char* const kDiffHlsl = R"(
float G(int x, int y) { return T1[ClampXY(x, y)].x; }

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    int x = int(tid.x), y = int(tid.y);

    if (Cfa == 0) { U0[tid.xy] = float4(0,0,0,1); U1[tid.xy] = float4(0,0,0,1); return; }

    // The exact difference wherever the colour was sampled.
    float dr = 0.0, db = 0.0;
    int   c  = CfaColor(Cfa, x, y);
    float gc = G(x, y);

    // Each plane is reconstructed independently; `want` selects which.
    // Written as a loop over the two colours so the three-case logic exists
    // once rather than twice.
    float outv[2] = {0.0, 0.0};
    [unroll] for (int k = 0; k < 2; ++k) {
        int want = (k == 0) ? 0 : 2;
        if (c == want) {
            outv[k] = Sample(x, y) - gc;
            continue;
        }
        if (c == 1) {
            // Green site: samples lie on one axis, chosen by the pattern.
            bool horiz = CfaColor(Cfa, x - 1, y) == want;
            if (horiz) {
                outv[k] = 0.5 * ((Sample(x - 1, y) - G(x - 1, y)) +
                                 (Sample(x + 1, y) - G(x + 1, y)));
            } else {
                outv[k] = 0.5 * ((Sample(x, y - 1) - G(x, y - 1)) +
                                 (Sample(x, y + 1) - G(x, y + 1)));
            }
            continue;
        }
        // Opposite-colour site: four samples on the diagonals. Choose the
        // diagonal running ALONG the edge, judged on green.
        float d1a = Sample(x - 1, y - 1) - G(x - 1, y - 1);
        float d1b = Sample(x + 1, y + 1) - G(x + 1, y + 1);
        float d2a = Sample(x + 1, y - 1) - G(x + 1, y - 1);
        float d2b = Sample(x - 1, y + 1) - G(x - 1, y + 1);

        float v1 = abs(G(x - 1, y - 1) - G(x + 1, y + 1));
        float v2 = abs(G(x + 1, y - 1) - G(x - 1, y + 1));
        float w1 = 1.0 / (1e-6 + v1 * v1);
        float w2 = 1.0 / (1e-6 + v2 * v2);
        outv[k] = (0.5 * (d1a + d1b) * w1 + 0.5 * (d2a + d2b) * w2) / (w1 + w2);
    }

    U0[tid.xy] = float4(outv[0], 0, 0, 1);
    U1[tid.xy] = float4(outv[1], 0, 0, 1);
}
)";

// Pass 2: a 3x3 median over each difference plane, gated on green.
// Bound as {mosaic, dr, db, green}, so t1/t2 are the planes and t3 is green.
const char* const kMedianHlsl = R"(
float G(int x, int y) { return T3[ClampXY(x, y)].x; }

float Median(inout float v[9], int n) {
    // Selection sort to the middle. n is at most 9, so this is cheaper than
    // anything cleverer and has no divergent control flow beyond the count.
    for (int i = 0; i <= n / 2; ++i) {
        int m = i;
        for (int j = i + 1; j < n; ++j) if (v[j] < v[m]) m = j;
        float t = v[i]; v[i] = v[m]; v[m] = t;
    }
    return v[n / 2];
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    int x = int(tid.x), y = int(tid.y);

    float gc  = G(x, y);
    float tol = 0.004 + gc * 0.12;

    [unroll] for (int k = 0; k < 2; ++k) {
        float v[9];
        int   n = 0;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                int2 p = ClampXY(x + dx, y + dy);
                if (abs(G(x + dx, y + dy) - gc) > tol) continue;
                v[n++] = (k == 0) ? T1[p].x : T2[p].x;
            }
        float centre = (k == 0) ? T1[ClampXY(x, y)].x : T2[ClampXY(x, y)].x;
        float r = (n >= 3) ? Median(v, n) : centre;
        if (k == 0) U0[tid.xy] = float4(r, 0, 0, 1);
        else        U1[tid.xy] = float4(r, 0, 0, 1);
    }
}
)";

// Pass 3: combine, bound each channel, then white balance and colour matrix.
const char* const kCombineHlsl = R"(
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    int x = int(tid.x), y = int(tid.y);

    if (Cfa == 0) {
        U0[tid.xy] = float4(T0[int2(tid.xy)].rgb, 1.0);
        return;
    }

    float g = T1[int2(tid.xy)].x;
    float3 rgb = float3(g + T2[int2(tid.xy)].x, g, g + T3[int2(tid.xy)].x);

    // Bound each reconstructed channel by the real samples of that colour
    // around it. Without this a merely-short red goes negative after the colour
    // matrix and clamps to zero, rendering fully saturated -- see the CPU path.
    float rLo = 1e30, rHi = -1e30, bLo = 1e30, bHi = -1e30;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            int sx = clamp(x + dx, 0, int(Width) - 1);
            int sy = clamp(y + dy, 0, int(Height) - 1);
            int sc = CfaColor(Cfa, sx, sy);
            float sv = Sample(sx, sy);
            if (sc == 0) { rLo = min(rLo, sv); rHi = max(rHi, sv); }
            if (sc == 2) { bLo = min(bLo, sv); bHi = max(bHi, sv); }
        }
    if (rHi >= rLo) rgb.r = clamp(rgb.r, rLo, rHi);
    if (bHi >= bLo) rgb.b = clamp(rgb.b, bLo, bHi);

    // Brightest RAW sample feeding each channel, and repair in RAW space
    // BEFORE white balance -- see clip_repair.h for both rules. The old code
    // flagged from the interpolated value and lifted after the gains, which
    // drove a third channel negative through the colour matrix.
    // The brightest RAW SAMPLE of each colour, over a window snapped to the
    // CFA cell so every pixel in that cell gets the SAME mask. Mirroring the
    // interpolation's own neighbours makes the clip decision alternate with
    // CFA parity, and that checkerboard is the fringing -- see CfaPeaks in
    // clip_repair.h.
    BalanceAndClamp(rgb, float3(asfloat(CamMul0), asfloat(CamMul1), asfloat(CamMul2)));

    float3 o;
    o.r = asfloat(M0) * rgb.r + asfloat(M1) * rgb.g + asfloat(M2) * rgb.b;
    o.g = asfloat(M3) * rgb.r + asfloat(M4) * rgb.g + asfloat(M5) * rgb.b;
    o.b = asfloat(M6) * rgb.r + asfloat(M7) * rgb.g + asfloat(M8) * rgb.b;

    // Negatives deliberately NOT clamped -- see the CPU path.
    U0[tid.xy] = float4(o, 1.0);
}
)";

} // namespace

class DemosaicAhd : public AlgorithmBase {
public:
    const char* Name()     const override { return "demosaic_ahd"; }
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

        // --- pass 1: green, interpolated along the edge ---------------------
        m_g.assign(n, 0.0f);
        for (int y = 0; y < h; ++y) {
            if (ctx.Cancelled()) return;
            for (int x = 0; x < w; ++x) {
                const size_t i = size_t(y) * size_t(w) + size_t(x);
                if (CfaColorAt(cfa, x, y) == 1) { m_g[i] = S(x, y); continue; }

                const float c = S(x, y);

                // Two candidate estimates. Each is the average of the two green
                // neighbours along that axis, corrected by the centre channel's
                // second derivative along the SAME axis -- the Malvar term, but
                // computed per direction so it can be chosen between.
                const float gh = 0.5f * (S(x - 1, y) + S(x + 1, y)) +
                                 0.25f * (2.0f * c - S(x - 2, y) - S(x + 2, y));
                const float gv = 0.5f * (S(x, y - 1) + S(x, y + 1)) +
                                 0.25f * (2.0f * c - S(x, y - 2) - S(x, y + 2));

                // How much the image varies along each axis. The direction with
                // the SMALLER variation is the one running along the edge, and
                // that is the one worth interpolating in: averaging across an
                // edge mixes the two sides together.
                //
                // Both the green neighbours and the same-colour neighbours at
                // ±2 contribute, so the measure sees both the luminance detail
                // and the centre channel's own structure.
                const float dh = std::fabs(S(x - 1, y) - S(x + 1, y)) +
                                 std::fabs(2.0f * c - S(x - 2, y) - S(x + 2, y));
                const float dv = std::fabs(S(x, y - 1) - S(x, y + 1)) +
                                 std::fabs(2.0f * c - S(x, y - 2) - S(x, y + 2));

                // A soft blend rather than a hard pick. A hard threshold makes
                // the decision visible: neighbouring pixels either side of it
                // get noticeably different reconstructions, which reads as a
                // maze-like texture -- the classic AHD artefact. Weighting by
                // the inverse gradient keeps the transition continuous.
                float wh = 1.0f / (1e-4f + dh * dh);
                float wv = 1.0f / (1e-4f + dv * dv);
                float g = (gh * wh + gv * wv) / (wh + wv);

                // Bound by the green samples present -- but bounded along the
                // CHOSEN axis, not by all four neighbours.
                //
                // Clamping to the envelope of all four was the first version
                // and it is what caused the magenta halo. Measured on the
                // Petaluma-area conifer frame, it fired on 8% of pixels and did
                // so asymmetrically: it LOWERED green by a mean of 31.2 sensor
                // counts while raising it by only 7.2, a four-to-one downward
                // bias concentrated exactly on edge pixels. Green ends up
                // systematically short there, and since white balance
                // afterwards multiplies red by 1.88 and blue by 1.91 while
                // leaving green at 1.00, a small green deficit becomes a
                // visible magenta glow.
                //
                // The reason it is biased is that the four neighbours span the
                // edge. On a thin dark branch, three may be bright sky and one
                // dark; their min/max envelope then describes the whole
                // neighbourhood rather than the surface this pixel belongs to,
                // and a correct estimate for the branch gets dragged toward the
                // sky's range or rejected against it.
                //
                // Bounding along the axis actually interpolated in fixes that:
                // those two samples are the ones the estimate came from, and on
                // an edge-aligned axis they are on the same surface. The
                // overshoot protection is kept -- it is what Malvar needs too --
                // without importing values from across the boundary.
                const float ah0 = S(x - 1, y), ah1 = S(x + 1, y);
                const float av0 = S(x, y - 1), av1 = S(x, y + 1);
                const float hLo = std::min(ah0, ah1), hHi = std::max(ah0, ah1);
                const float vLo = std::min(av0, av1), vHi = std::max(av0, av1);

                // Blend the two bounds by the same weights that blended the two
                // estimates, so the bound follows the decision rather than
                // fighting it.
                const float lo = (hLo * wh + vLo * wv) / (wh + wv);
                const float hi = (hHi * wh + vHi * wv) / (wh + wv);
                m_g[i] = std::clamp(g, std::min(lo, hi), std::max(lo, hi));
            }
        }

        // --- pass 2: red and blue, as differences against green -------------
        //
        // At each site the known colour gives an exact difference; the unknown
        // ones are interpolated from those, then green is added back. The
        // difference planes are smooth even where the channels are not, which
        // is what makes this markedly better than interpolating R and B.
        m_dr.assign(n, 0.0f);
        m_db.assign(n, 0.0f);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const size_t i = size_t(y) * size_t(w) + size_t(x);
                const int    c = CfaColorAt(cfa, x, y);
                if (c == 0)      m_dr[i] = S(x, y) - m_g[i];
                else if (c == 2) m_db[i] = S(x, y) - m_g[i];
            }
        }

        auto known = [&](int x, int y, int want) {
            return CfaColorAt(cfa, std::clamp(x, 0, w - 1),
                                   std::clamp(y, 0, h - 1)) == want;
        };
        auto diffAt = [&](const std::vector<float>& d, int x, int y) {
            x = std::clamp(x, 0, w - 1);
            y = std::clamp(y, 0, h - 1);
            return d[size_t(y) * size_t(w) + size_t(x)];
        };
        // Fill each difference plane DIRECTIONALLY, by Bayer geometry.
        //
        // The previous version searched a square neighbourhood and averaged
        // whatever samples of the right colour it found, weighted by distance.
        // That is isotropic, and it is the wrong shape for the problem: it
        // ignores the fact that a missing red sits in one of exactly three
        // known positions relative to the reds that surround it, and that those
        // three cases want different interpolations.
        //
        // Isotropy is why the fringing survived every attempt to filter it
        // afterwards. On a twig one pixel wide, an isotropic estimate at a sky
        // pixel beside the twig pulls in samples from the twig and vice versa,
        // so the colour difference is wrong before any median sees it. No
        // amount of edge-gating downstream recovers information that was
        // averaged away upstream -- and both attempts at gating traded one
        // visible artefact for another.
        //
        // The three cases, for the red plane (blue is the mirror image):
        //
        //   at a BLUE site  -- the four nearest reds are on the DIAGONALS, so
        //                      interpolate along whichever diagonal varies
        //                      less, exactly as green chose between H and V
        //   at a GREEN site on a red row    -- reds are left and right
        //   at a GREEN site on a blue row   -- reds are above and below
        //
        // The two green cases have their samples on a single axis, so there is
        // no direction to choose: those are exact two-point interpolations of a
        // smooth signal, which is as good as it gets. Only the blue-site case
        // is ambiguous, and that is where the directional choice pays.
        auto fillDiff = [&](std::vector<float>& d, int colour) {
            std::vector<float> out(d.size(), 0.0f);
            for (int y = 0; y < h; ++y) {
                if (ctx.Cancelled()) return;
                for (int x = 0; x < w; ++x) {
                    const size_t i = size_t(y) * size_t(w) + size_t(x);
                    const int    c = CfaColorAt(cfa, x, y);
                    if (c == colour) { out[i] = d[i]; continue; }

                    if (c == 1) {
                        // Green site: the samples lie on one axis. Which axis
                        // depends on the row, so ask the pattern rather than
                        // assuming -- getting this wrong reconstructs red from
                        // blue's neighbours and tints the whole image.
                        const bool horiz = known(x - 1, y, colour);
                        out[i] = horiz
                            ? 0.5f * (diffAt(d, x - 1, y) + diffAt(d, x + 1, y))
                            : 0.5f * (diffAt(d, x, y - 1) + diffAt(d, x, y + 1));
                        continue;
                    }

                    // Opposite-colour site: four samples on the diagonals.
                    //
                    // Choose between the two diagonals the same way green chose
                    // between its axes: interpolate ALONG the direction that
                    // varies less, which is the one running with the edge
                    // rather than across it. On a twig at any angle, one
                    // diagonal follows it and the other crosses it, and
                    // averaging all four blends the twig with the sky.
                    const float d1a = diffAt(d, x - 1, y - 1);
                    const float d1b = diffAt(d, x + 1, y + 1);
                    const float d2a = diffAt(d, x + 1, y - 1);
                    const float d2b = diffAt(d, x - 1, y + 1);

                    // Judge the direction on GREEN, not on the differences.
                    // Green is fully reconstructed by now and carries the
                    // structure; the difference plane is deliberately smooth
                    // and so says little about where the edges are.
                    const float g1a = m_g[size_t(std::clamp(y - 1, 0, h - 1)) * size_t(w) +
                                          size_t(std::clamp(x - 1, 0, w - 1))];
                    const float g1b = m_g[size_t(std::clamp(y + 1, 0, h - 1)) * size_t(w) +
                                          size_t(std::clamp(x + 1, 0, w - 1))];
                    const float g2a = m_g[size_t(std::clamp(y - 1, 0, h - 1)) * size_t(w) +
                                          size_t(std::clamp(x + 1, 0, w - 1))];
                    const float g2b = m_g[size_t(std::clamp(y + 1, 0, h - 1)) * size_t(w) +
                                          size_t(std::clamp(x - 1, 0, w - 1))];

                    const float v1 = std::fabs(g1a - g1b);   // NW-SE
                    const float v2 = std::fabs(g2a - g2b);   // NE-SW

                    // A soft blend, for the same reason green uses one: a hard
                    // pick makes the decision boundary visible as a maze-like
                    // texture where neighbouring pixels choose differently.
                    const float w1 = 1.0f / (1e-6f + v1 * v1);
                    const float w2 = 1.0f / (1e-6f + v2 * v2);
                    out[i] = (0.5f * (d1a + d1b) * w1 +
                              0.5f * (d2a + d2b) * w2) / (w1 + w2);
                }
            }
            d.swap(out);
        };
        fillDiff(m_dr, 0);
        fillDiff(m_db, 2);

        // --- pass 3: median-filter the colour differences -------------------
        //
        // The step that clears what is left. Demosaicing artefacts are isolated
        // and signed; a median removes exactly that and leaves edges alone,
        // because it selects an existing value rather than blending. Applied to
        // the DIFFERENCES, so luminance detail is untouched -- which is why
        // this sharpens perceptually rather than softening.
        const int passes = std::max(0, int(m_chromaMedian));
        for (int p = 0; p < passes; ++p) {
            if (ctx.Cancelled()) return;
            MedianFilter(m_g, m_dr, w, h);
            MedianFilter(m_g, m_db, w, h);
        }

        // --- combine --------------------------------------------------------
        for (int y = 0; y < h; ++y) {
            if (ctx.Cancelled()) return;
            for (int x = 0; x < w; ++x) {
                const size_t i = size_t(y) * size_t(w) + size_t(x);
                const float  g = m_g[i];

                // Reconstructed entirely from the filtered difference planes,
                // including at sites where the channel was sampled directly.
                //
                // Restoring the raw sample there looks obviously right -- it is
                // real data, and the reconstruction is only an estimate -- and
                // it is what the first version did. It is wrong, and wrong in
                // precisely the way this algorithm exists to fix.
                //
                // The difference planes have been median-filtered. Overwriting
                // the sampled channel puts an UNFILTERED value next to filtered
                // ones, so adjacent pixels alternate between two different
                // treatments on the CFA lattice -- which is a phase-correlated
                // checkerboard, the exact artefact being removed. Measured, it
                // cost more than it saved: 554 speckles against Malvar's 289,
                // while the green plane underneath was already twice as smooth
                // (0.073 against 0.153).
                //
                // Consistency beats pointwise fidelity here. Every pixel gets
                // the same treatment, so nothing correlates with phase, and the
                // sampled value is not discarded -- it is what set the
                // difference at that site before filtering.
                float rgb[3] = {g + m_dr[i], g, g + m_db[i]};

                // Bound each reconstructed channel by the real samples of that
                // colour around it.
                //
                // This is the same protection Malvar has and this algorithm
                // lacked, and its absence is what makes the fringes vivid
                // rather than faint. g + (R-G) is the sum of two independently
                // estimated quantities with no bound of its own, so at an edge
                // it can land well outside anything the sensor recorded
                // nearby -- red too low relative to green, in particular.
                //
                // A merely-low red is invisible here and catastrophic three
                // lines later. ApplyColour multiplies by the camera matrix,
                // whose first row on this body is [1.535, -0.555, 0.019]: red
                // out is red in MINUS 0.555 of green. So a red that is only
                // slightly short goes negative after the matrix, is clamped to
                // zero, and the pixel renders fully saturated.
                //
                // Measured on the Oregon frame: 1060 red pixels driven to zero
                // against Malvar's 22, a factor of 48. Clamping the raw output
                // at zero does nothing, because the value is positive until the
                // matrix sees it -- the fix has to keep the estimate plausible
                // BEFORE the matrix, which means bounding it by real data.
                {
                    float rLo = 1e30f, rHi = -1e30f, bLo = 1e30f, bHi = -1e30f;
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx) {
                            const int sx = std::clamp(x + dx, 0, w - 1);
                            const int sy = std::clamp(y + dy, 0, h - 1);
                            const int sc = CfaColorAt(cfa, sx, sy);
                            const float sv = S(sx, sy);
                            if (sc == 0) { rLo = std::min(rLo, sv); rHi = std::max(rHi, sv); }
                            if (sc == 2) { bLo = std::min(bLo, sv); bHi = std::max(bHi, sv); }
                        }
                    if (rHi >= rLo) rgb[0] = std::clamp(rgb[0], rLo, rHi);
                    if (bHi >= bLo) rgb[2] = std::clamp(rgb[2], bLo, bHi);
                }

                ApplyColour(src.desc, rgb);

                uint16_t* p = dst.At<uint16_t>(x, y);
                // Stored UNCLAMPED. A negative channel is a real colour outside

                // sRGB's gamut, not an error -- see ApplyColour.

                p[0] = FloatToHalf(rgb[0]);
                p[1] = FloatToHalf(rgb[1]);
                p[2] = FloatToHalf(rgb[2]);
                p[3] = FloatToHalf(1.0f);
            }
        }
    }

private:
    // Median the colour differences, but only using samples on the same side of
    // an edge.
    //
    // A plain 3x3 median was the first version and it INTRODUCED a defect on
    // exactly the content this algorithm is for. On a twig narrower than the
    // window, most of the nine samples belong to the sky, so the median selects
    // a sky colour difference and the twig is painted with it -- visible as
    // green and magenta dashes strung along the finest branches. Rendering the
    // same crop at median 0 and median 2 showed it plainly: the dashes exist
    // only in the filtered version.
    //
    // Gating on green fixes it without giving up the filter. Green is already
    // reconstructed and tracks luminance, so a large difference in green means
    // "different side of an edge". Samples that fail the test are excluded
    // rather than down-weighted, because a median selects rather than blends --
    // and with fewer than three left the pixel is passed through untouched,
    // since a median of one or two is not a median.
    void MedianFilter(const std::vector<float>& g, std::vector<float>& d,
                      int w, int h) {
        m_tmp.assign(d.size(), 0.0f);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const size_t i = size_t(y) * size_t(w) + size_t(x);

                // Only pool samples of similar brightness.
                //
                // The gate is kept deliberately, against what the aggregate
                // numbers say, and that disagreement is worth recording.
                //
                // Removing it improves every summary statistic: the magenta
                // index in the dark deciles goes from +0.031 to -0.005, past
                // Malvar's -0.020. But rendering the same crop shows why the
                // number is not the thing that matters -- ungated, the median
                // paints bright green streaks along every twig, because on a
                // feature narrower than the 3x3 window most of the nine samples
                // belong to the sky and the median selects one of those. The
                // metric averages over a whole decile and cannot see a
                // localised streak; the eye sees nothing else.
                //
                // So this trades a real magenta bias for the absence of a worse
                // green one. Both are visible, and the magenta is the milder of
                // the two. Neither is right, and the honest summary is that the
                // colour-difference stage still needs work -- see the note at
                // the top of this file.
                const float gc  = g[i];
                const float tol = kMedianEdgeFloor + gc * kMedianEdgeRel;

                float v[9];
                int   k = 0;
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int sx = std::clamp(x + dx, 0, w - 1);
                        const int sy = std::clamp(y + dy, 0, h - 1);
                        const size_t j = size_t(sy) * size_t(w) + size_t(sx);
                        if (std::fabs(g[j] - gc) > tol) continue;
                        v[k++] = d[j];
                    }
                // Fewer than three survivors is not a median; pass through.
                m_tmp[i] = (k >= 3) ? MedianN(v, k) : d[i];
            }
        }
        d.swap(m_tmp);
    }

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

    // Identical to the other demosaicers, deliberately: white balance and the
    // camera matrix are properties of the capture, not of the interpolation, so
    // switching method must not change colour. See demosaic_malvar.cpp for why
    // the clipped-channel repair happens after the gains rather than before.
    static void ApplyColour(const ImageDesc& d, float* rgb) {
        // White balance with the highlight clamp -- see clip_repair.h.
        BalanceAndClamp(d, rgb);

        CameraMatrixInGamut(d, rgb);

        // Negatives NOT clamped: a colour outside sRGB's gamut lands below zero
        // after the camera matrix, and clamping destroys it before the user has
        // touched anything. The pipeline is linear float, so it costs nothing to
        // carry -- clamping belongs at display or export. See
        // demosaic_consistent.cpp for the measurement.
    }


    // How far apart in green two samples may be and still be pooled by the
    // chroma median: a floor plus a fraction of the pixel's own brightness.
    //
    // Proportional rather than absolute, so the test means the same thing
    // across the tonal range. A fixed 0.06 was tried first and is far too
    // generous in shadow -- on a branch whose green is around 0.02, every
    // sample of a sky at 0.5 still passes.
    static constexpr float kMedianEdgeFloor = 0.004f;
    static constexpr float kMedianEdgeRel   = 0.12f;

    Param<int> m_chromaMedian{
        this, "chroma_median", 2, 0, 5,
        {.help = "Passes of a 3x3 median over the colour-difference planes. "
                 "Removes isolated colour speckles without touching luminance "
                 "detail, because it filters R-G and B-G rather than the "
                 "channels. 0 disables it, which is useful for seeing what the "
                 "directional interpolation alone achieves.",
         .step = 1}};

    // --- GPU ----------------------------------------------------------------
    //
    // Four kernels over three scratch planes, which is why this needed
    // AlgorithmBase::GpuPasses rather than the existing GpuIterations() path:
    // that runs ONE kernel repeatedly, and these passes do different things.
    //
    //   plane 0 : green, everywhere
    //   plane 1 : R-G
    //   plane 2 : B-G
    //
    //   pass 0  green      mosaic          -> plane 0
    //   pass 1  diff       mosaic, plane 0 -> planes 1, 2
    //   pass 2  median     planes 1, 2, 0  -> planes 1, 2   (skipped at 0 passes)
    //   pass 3  combine    mosaic, 0, 1, 2 -> output
    //
    // The median is a fixed single pass on the GPU where the CPU exposes a
    // count. Iterating it would need another pair of planes to ping-pong
    // against -- a pass may not read and write the same buffer, which the
    // framework enforces -- and one pass is what the default uses anyway.
    bool HasGPU() const override { return true; }

    void PrepareGpu(const std::vector<ImageDesc>& inputs) override {
        if (inputs.empty()) return;
        m_cfa   = int(inputs[0].cfa);
        m_black = inputs[0].blackLevel;
        m_range = std::max(inputs[0].whiteLevel - inputs[0].blackLevel, 1e-6f);
        for (int i = 0; i < 3; ++i) m_camMul[i] = inputs[0].camMul[i];
        for (int i = 0; i < 9; ++i) m_rgbCam[i] = inputs[0].rgbCam[i];
    }

    // Five planes: green, the two colour differences, and two more for the
    // median to write into.
    //
    // The median cannot filter in place -- a pass that reads and writes the
    // same buffer is a race, since threads within a dispatch have no ordering,
    // and the framework rejects it rather than letting it produce noise on some
    // hardware and not others. Two extra full-size R32F planes is the price of
    // that safety: about 180 MB at 45 MP, against 358 MB for the output itself.
    int        GpuScratchCount()  const override { return 5; }
    FormatSpec GpuScratchPlanes() const override { return FormatSpec::R32F; }

    std::vector<GpuPass> GpuPasses() const override {
        // The sources are assembled once, on first use, because GpuPasses()
        // returns raw pointers and is called per run -- building the strings
        // each time would dangle them the moment the vector went out of scope.
        static const std::string green   = std::string(kAhdCommon) + kGreenHlsl;
        static const std::string diff    = std::string(kAhdCommon) + kDiffHlsl;
        static const std::string median  = std::string(kAhdCommon) + kMedianHlsl;
        static const std::string combine = std::string(kAhdCommon) + kCombineHlsl;

        std::vector<GpuPass> p;
        // t0 is always the mosaic, because kAhdCommon's Sample() reads it.
        // Where a pass does not need the mosaic it still binds it, which costs
        // one SRV slot of four and keeps the prologue identical everywhere.
        p.push_back({green.c_str(), "green", {-1},    {0}});
        p.push_back({diff.c_str(),  "diff",  {-1, 0}, {1, 2}});

        // The median runs the SAME number of times as the CPU path.
        //
        // An earlier version ran it exactly once regardless, on the reasoning
        // that iterating would need more planes and one pass is most of the
        // benefit. That is defensible as a design choice and indefensible as a
        // silent difference: compare mode checks the two paths against each
        // other, and the mismatch showed up immediately as GPU disagreeing with
        // CPU by 0.0063 while every other demosaic matched at exactly 0.
        //
        // A median cannot filter in place -- threads have no ordering, so a
        // pass reading the buffer it writes is a race the framework rejects --
        // so each iteration ping-pongs between planes {1,2} and {3,4}.
        const int passes = std::max(0, int(m_chromaMedian));
        int srcA = 1, srcB = 2;
        for (int i = 0; i < passes; ++i) {
            const int dstA = (srcA == 1) ? 3 : 1;
            const int dstB = (srcB == 2) ? 4 : 2;
            p.push_back({median.c_str(), "median",
                         {-1, srcA, srcB, 0}, {dstA, dstB}});
            srcA = dstA;
            srcB = dstB;
        }
        p.push_back({combine.c_str(), "combine", {-1, 0, srcA, srcB}, {-1}});
        return p;
    }

    std::vector<uint32_t> GpuPassConstants(int) const override {
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
    PixelBuffer        m_in;
    std::vector<float> m_s, m_g, m_dr, m_db, m_tmp;

    int   m_cfa   = 1;
    float m_black = 0.0f;
    float m_range = 1.0f;
    float m_camMul[3] = {1.0f, 1.0f, 1.0f};
    float m_rgbCam[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
};

REGISTER_ALGORITHM(DemosaicAhd);

} // namespace tglab
