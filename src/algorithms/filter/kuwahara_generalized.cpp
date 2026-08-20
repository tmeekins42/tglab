// kuwahara_generalized — Papari, Petkov & Campisi (2007), "Artistic Edge and
// Corner Enhancing Smoothing", IEEE TIP 16(10).
//
// Classic Kuwahara has two defects, both visible on real images: only four
// square quadrants, so it cannot follow a diagonal or curved edge; and a hard
// "pick the winner" rule, so a tie flips between quadrants and leaves blocky,
// directional artefacts.
//
// Papari et al. fix both:
//
//   1. N overlapping sectors of a disc instead of 4 squares. Eight sectors
//      follow an edge of any orientation far better than four quadrants, and
//      corners survive because a sector can fit inside the corner's wedge.
//
//   2. A smooth weighted average instead of a hard minimum. Each sector's
//      contribution is weighted by
//
//          w_i = 1 / (1 + (variance_i)^(q/2))
//
//      so low-variance sectors dominate without any single one winning
//      outright. Raising q sharpens the selection towards classic Kuwahara;
//      lowering it towards a plain disc average. This is the parameter that
//      removes the artefacts, and it is why the output looks painterly rather
//      than blocky.
//
// Sector weights are precomputed once per (radius, sectors) pair: each is a
// Gaussian-decayed wedge, normalised so the sectors sum to unity everywhere.
// That normalisation is what keeps flat regions from developing radial
// banding, and it is the fiddly part of a correct implementation.
#include <algorithm>
#include <cmath>
#include <vector>

#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"

namespace tglab {

class KuwaharaGeneralized : public AlgorithmBase {
public:
    const char* Name()     const override { return "kuwahara_generalized"; }
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

        const int radius  = std::max(1, int(m_radius));
        const int sectors = std::max(3, int(m_sectors));
        const float q     = std::max(0.1f, float(m_sharpness));

        BuildSectorWeights(radius, sectors);

        const int w = m_in.Width(), h = m_in.Height(), ch = m_in.Channels();
        const int filtered = (ch == 1) ? 1 : 3;
        const int side = radius * 2 + 1;

        std::vector<float> sum(size_t(sectors) * 4, 0.0f);
        std::vector<float> sumSq(size_t(sectors), 0.0f);
        std::vector<float> weightSum(size_t(sectors), 0.0f);

        for (int y = 0; y < h; ++y) {
            // Checked per row: cancelling only between stages is useless when a
            // single stage is the slow one.
            if (ctx.Cancelled()) return;
            for (int x = 0; x < w; ++x) {
                std::fill(sum.begin(), sum.end(), 0.0f);
                std::fill(sumSq.begin(), sumSq.end(), 0.0f);
                std::fill(weightSum.begin(), weightSum.end(), 0.0f);

                // One pass over the disc, accumulating into every sector the
                // sample has nonzero weight in. Sectors overlap, so a sample
                // near a boundary contributes to both -- that overlap is what
                // makes the result continuous as an edge rotates.
                for (int dy = -radius; dy <= radius; ++dy) {
                    for (int dx = -radius; dx <= radius; ++dx) {
                        const size_t cell =
                            size_t(dy + radius) * size_t(side) + size_t(dx + radius);
                        const float* p = m_in.AtClamped(x + dx, y + dy);
                        const float lum = Luma(p, filtered);

                        for (int s = 0; s < sectors; ++s) {
                            const float wgt = m_weights[size_t(s) * size_t(side) * size_t(side) + cell];
                            if (wgt <= 0.0f) continue;
                            for (int c = 0; c < filtered; ++c)
                                sum[size_t(s) * 4 + size_t(c)] += p[c] * wgt;
                            sumSq[size_t(s)]     += lum * lum * wgt;
                            weightSum[size_t(s)] += wgt;
                        }
                    }
                }

                // Weighted blend across sectors: 1 / (1 + variance^(q/2)).
                float acc[4]  = {0, 0, 0, 0};
                float accWeight = 0.0f;

                for (int s = 0; s < sectors; ++s) {
                    const float ws = weightSum[size_t(s)];
                    if (ws <= 1e-8f) continue;

                    const float inv     = 1.0f / ws;
                    float mean[4] = {0, 0, 0, 0};
                    for (int c = 0; c < filtered; ++c)
                        mean[c] = sum[size_t(s) * 4 + size_t(c)] * inv;

                    const float lumMean  = Luma(mean, filtered);
                    const float variance =
                        std::max(0.0f, sumSq[size_t(s)] * inv - lumMean * lumMean);

                    // Normalise variance by the value range so `sharpness`
                    // behaves the same for 8-bit and float sources.
                    const float scale = m_in.ValueScale();
                    const float vNorm = variance / (scale * scale);
                    const float weight = 1.0f / (1.0f + std::pow(vNorm, q * 0.5f));

                    for (int c = 0; c < filtered; ++c) acc[c] += mean[c] * weight;
                    accWeight += weight;
                }

                const float inv = 1.0f / std::max(accWeight, 1e-8f);
                for (int c = 0; c < filtered; ++c) m_out.Set(x, y, c, acc[c] * inv);
                if (ch == 4) m_out.Set(x, y, 3, m_in.Get(x, y, 3));
            }
        }

        m_out.PackInto(dst);
    }

private:
    static float Luma(const float* p, int channels) {
        if (channels == 1) return p[0];
        return 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
    }

    // A weight map per sector: a wedge of the disc, smoothed across its angular
    // boundaries and decaying with radius, then normalised so that summing all
    // sectors gives the same total at every offset. Without that normalisation
    // the filter brightens or darkens with distance from the centre.
    void BuildSectorWeights(int radius, int sectors) {
        if (radius == m_builtRadius && sectors == m_builtSectors) return;
        m_builtRadius  = radius;
        m_builtSectors = sectors;

        const int side = radius * 2 + 1;
        m_weights.assign(size_t(sectors) * size_t(side) * size_t(side), 0.0f);

        const float twoPi     = 6.28318530718f;
        const float sectorArc = twoPi / float(sectors);
        // Angular smoothing of about half a sector: wide enough to overlap the
        // neighbouring sector, narrow enough to stay directional.
        const float angSigma  = sectorArc * 0.5f;
        // Radial decay, so samples near the rim matter less than the centre.
        const float radSigma  = float(radius) * 0.5f;

        std::vector<float> total(size_t(side) * size_t(side), 0.0f);

        for (int s = 0; s < sectors; ++s) {
            const float centreAngle = sectorArc * float(s);
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    const float r = std::sqrt(float(dx * dx + dy * dy));
                    if (r > float(radius)) continue;      // stay inside the disc

                    const size_t cell =
                        size_t(dy + radius) * size_t(side) + size_t(dx + radius);

                    float weight;
                    if (r < 1e-4f) {
                        // The centre pixel belongs to every sector equally;
                        // giving it to one would bias the whole result.
                        weight = 1.0f / float(sectors);
                    } else {
                        float delta = std::atan2(float(dy), float(dx)) - centreAngle;
                        // Wrap into [-pi, pi] so the seam at 0/2pi is smooth.
                        while (delta >  3.14159265f) delta -= twoPi;
                        while (delta < -3.14159265f) delta += twoPi;

                        const float ang = std::exp(-(delta * delta) /
                                                   (2.0f * angSigma * angSigma));
                        const float rad = std::exp(-(r * r) /
                                                   (2.0f * radSigma * radSigma));
                        weight = ang * rad;
                    }

                    m_weights[size_t(s) * size_t(side) * size_t(side) + cell] = weight;
                    total[cell] += weight;
                }
            }
        }

        // Normalise across sectors so every offset contributes exactly once.
        for (int s = 0; s < sectors; ++s)
            for (size_t cell = 0; cell < total.size(); ++cell)
                if (total[cell] > 1e-8f)
                    m_weights[size_t(s) * total.size() + cell] /= total[cell];
    }

    Param<int> m_radius{
        this, "radius", 4, 1, 32,
        {.help = "Radius of the disc split into sectors, in pixels. Higher "
                 "gives broader flat regions.",
         .softMin = 2, .softMax = 12}};
    Param<int> m_sectors{
        this, "sectors", 8, 3, 16,
        {.help = "How many directions the disc is divided into. More follows "
                 "curved and diagonal edges better; 8 is the paper's value and "
                 "is usually enough.",
         .softMin = 4, .softMax = 8}};
    // q in the paper. Higher is closer to classic Kuwahara's hard selection.
    Param<float> m_sharpness{
        this, "sharpness", 8.0f, 0.1f, 32.0f,
        {.help = "How decisively the flattest sector wins (q in the paper). "
                 "Higher approaches classic Kuwahara's hard choice and crisper "
                 "edges; lower blends sectors towards a plain disc average.",
         .step = 0.5, .softMin = 1.0, .softMax = 16.0}};

    // Cached weight maps; rebuilt only when the geometry changes.
    std::vector<float> m_weights;
    int m_builtRadius  = -1;
    int m_builtSectors = -1;

    PixelBuffer m_in, m_out;
};

REGISTER_ALGORITHM(KuwaharaGeneralized);

} // namespace tglab
