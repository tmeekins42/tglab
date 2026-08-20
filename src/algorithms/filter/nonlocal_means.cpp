// nonlocal_means — Buades, Coll & Morel (2005).
//
// A different principle from every other filter here. The others decide how
// much to trust a neighbour from its *distance* and its *own* intensity. NLM
// compares the small patch around each neighbour with the patch around the
// centre, and weights by how similar those patches are:
//
//     w(p, q) = exp( -||patch(p) - patch(q)||^2 / (h^2) )
//
// So a pixel on the far side of the image, sitting in a similar texture,
// contributes more than an adjacent pixel in a different structure. That is
// what makes it excellent at preserving repeated detail (text, weave, grain)
// that a bilateral filter smears -- directly relevant to scanned material.
//
// The cost is real: for each pixel it compares patches against every candidate
// in a search window, giving O(search^2 * patch^2) per pixel. `search` is
// therefore the parameter to raise carefully; the defaults keep an 8 MP scan
// tolerable, and the soft ranges reflect that rather than the hard limits.
#include <algorithm>
#include <cmath>
#include <vector>

#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"

namespace tglab {

class NonLocalMeans : public AlgorithmBase {
public:
    const char* Name()     const override { return "nonlocal_means"; }
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

        const int patch  = std::max(1, int(m_patch));
        const int search = std::max(1, int(m_search));
        const float scale = m_in.ValueScale();
        // h is declared as a fraction of the range, then squared against the
        // per-pixel mean squared difference below.
        const float hParam = std::max(1e-4f, float(m_strength)) * scale;
        const float h2 = hParam * hParam;

        const int w = m_in.Width(), hgt = m_in.Height(), ch = m_in.Channels();
        const int filtered = (ch == 1) ? 1 : 3;
        const float patchNorm = 1.0f / float((patch * 2 + 1) * (patch * 2 + 1) * filtered);

        for (int y = 0; y < hgt; ++y) {
            for (int x = 0; x < w; ++x) {
                float acc[4]    = {0, 0, 0, 0};
                float weightSum = 0.0f;

                for (int sy = -search; sy <= search; ++sy) {
                    for (int sx = -search; sx <= search; ++sx) {
                        // Cheap rejection first. exp(-d^2/h^2) is already
                        // negligible once the centre pixels alone differ by
                        // several h, and the full patch distance can only be
                        // larger. Skipping those candidates is the difference
                        // between this being usable on a scan and not.
                        {
                            const float* a = m_in.At(x, y);
                            const float* b = m_in.AtClamped(x + sx, y + sy);
                            float centre2 = 0.0f;
                            for (int c = 0; c < filtered; ++c) {
                                const float d = a[c] - b[c];
                                centre2 += d * d;
                            }
                            // Per-channel mean, matching the scale dist2 is on
                            // after patchNorm. Comparing the raw sum here would
                            // reject almost everything and defeat the filter.
                            centre2 /= float(filtered);
                            if (centre2 > kRejectSigmas * h2) continue;
                        }

                        // Mean squared difference between the two patches.
                        float dist2 = 0.0f;
                        for (int py = -patch; py <= patch; ++py) {
                            for (int px = -patch; px <= patch; ++px) {
                                const float* a = m_in.AtClamped(x + px, y + py);
                                const float* b =
                                    m_in.AtClamped(x + sx + px, y + sy + py);
                                for (int c = 0; c < filtered; ++c) {
                                    const float d = a[c] - b[c];
                                    dist2 += d * d;
                                }
                            }
                        }
                        dist2 *= patchNorm;

                        const float weight = std::exp(-dist2 / h2);
                        const float* n = m_in.AtClamped(x + sx, y + sy);
                        for (int c = 0; c < filtered; ++c) acc[c] += n[c] * weight;
                        weightSum += weight;
                    }
                }

                const float inv = 1.0f / std::max(weightSum, 1e-8f);
                for (int c = 0; c < filtered; ++c) m_out.Set(x, y, c, acc[c] * inv);
                if (ch == 4) m_out.Set(x, y, 3, m_in.Get(x, y, 3));
            }
        }

        m_out.PackInto(dst);
    }

private:
    // exp(-9) is about 1e-4, small enough that dropping those candidates does
    // not visibly change the result but large enough not to bias it.
    static constexpr float kRejectSigmas = 9.0f;

    // Patch radius: 1 (3x3) or 2 (5x5) is standard; larger rarely helps.
    Param<int> m_patch{
        this, "patch", 1, 1, 7,
        {.help = "Radius of the patch compared between pixels. 1 (3x3) or "
                 "2 (5x5) is standard; larger is more selective about what "
                 "counts as a match but rarely helps.",
         .softMin = 1, .softMax = 3}};
    // Search radius dominates the cost -- it is quadratic in this value, and
    // NLM is by far the most expensive filter here (order 10 s per megapixel at
    // search 5). The default is deliberately modest so that selecting it from a
    // dropdown does not look like a hang on a full-size scan; raise it
    // knowingly, ideally on a cropped region first.
    Param<int> m_search{
        this, "search", 3, 1, 21,
        {.help = "How far to look for matching patches, in pixels. Higher "
                 "finds more matches and denoises better, but cost grows with "
                 "its SQUARE -- this is by far the most expensive knob here.",
         .softMin = 2, .softMax = 10}};
    Param<float> m_strength{
        this, "strength", 0.1f, 0.001f, 1.0f,
        {.help = "How different two patches may be and still be averaged, as "
                 "a fraction of the intensity range. Higher denoises more but "
                 "starts blending genuinely different texture.",
         .step = 0.005, .softMin = 0.02, .softMax = 0.4}};

    PixelBuffer m_in, m_out;
};

REGISTER_ALGORITHM(NonLocalMeans);

} // namespace tglab
