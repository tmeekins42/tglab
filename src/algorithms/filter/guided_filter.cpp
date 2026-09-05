// guided_filter — He, Sun & Tang (2010), "Guided Image Filtering".
//
// Edge-preserving like the bilateral, but with two properties the bilateral
// lacks, which is why it largely replaced it in practice:
//
//   - O(1) per pixel regardless of radius. It is built entirely from box means,
//     so a radius-40 guided filter costs the same as radius-2. The bilateral is
//     O(r^2) and becomes unusable at large radii.
//   - No gradient reversal. The bilateral produces halo artefacts near strong
//     edges because its range weight can starve a pixel of similar neighbours;
//     the guided filter's local linear model does not.
//
// The model: within each window, assume the output is a linear function of the
// guide, q = a*I + b. Least squares over the window gives
//
//     a = cov(I, p) / (var(I) + eps)
//     b = mean(p) - a * mean(I)
//
// then a and b are themselves box-averaged so neighbouring windows blend.
// Here the guide is the input itself (self-guided), which is the edge-preserving
// smoothing case.
//
// eps plays the role of sigma_range in the bilateral: it is the variance below
// which a region counts as flat. Because it compares against *variance*, it is
// declared as a fraction of the range and squared internally, so the slider
// stays perceptually even.
#include <algorithm>
#include <cmath>
#include <vector>

#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"

namespace tglab {

class GuidedFilter : public AlgorithmBase {
public:
    const char* Name()     const override { return "guided_filter"; }
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

        const int radius = std::max(1, ctx.ScaledRadius(int(m_radius)));
        const float epsFrac = std::max(1e-5f, float(m_eps));
        const float scale = m_in.ValueScale();
        // eps is compared against a variance, so square the fractional value.
        const float eps = epsFrac * epsFrac * scale * scale;

        const int w = m_in.Width(), h = m_in.Height(), ch = m_in.Channels();
        const int filtered = (ch == 1) ? 1 : 3;
        const size_t n = size_t(w) * size_t(h);

        m_mean.assign(n, 0.0f);
        m_meanSq.assign(n, 0.0f);
        m_a.assign(n, 0.0f);
        m_b.assign(n, 0.0f);
        m_scratch.assign(n, 0.0f);
        m_plane.assign(n, 0.0f);

        // Each channel is filtered independently, guided by itself.
        for (int c = 0; c < filtered; ++c) {
            if (ctx.Cancelled()) return;   // per channel: each is a full set of passes
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x)
                    m_plane[size_t(y) * size_t(w) + size_t(x)] = m_in.Get(x, y, c);

            BoxMean(m_plane, m_mean, w, h, radius);

            for (size_t i = 0; i < n; ++i) m_scratch[i] = m_plane[i] * m_plane[i];
            BoxMean(m_scratch, m_meanSq, w, h, radius);

            // Self-guided, so cov(I, p) == var(I) and the algebra collapses to
            // a = var / (var + eps), b = (1 - a) * mean.
            for (size_t i = 0; i < n; ++i) {
                const float var = std::max(0.0f, m_meanSq[i] - m_mean[i] * m_mean[i]);
                const float a   = var / (var + eps);
                m_a[i] = a;
                m_b[i] = (1.0f - a) * m_mean[i];
            }

            // Averaging a and b is what makes the result smooth where windows
            // overlap; without it the output shows window-sized blocking.
            BoxMean(m_a, m_scratch, w, h, radius);
            m_a.swap(m_scratch);
            BoxMean(m_b, m_scratch, w, h, radius);
            m_b.swap(m_scratch);

            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x) {
                    const size_t i = size_t(y) * size_t(w) + size_t(x);
                    m_out.Set(x, y, c, m_a[i] * m_plane[i] + m_b[i]);
                }
        }

        if (ch == 4)
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x) m_out.Set(x, y, 3, m_in.Get(x, y, 3));

        m_out.PackInto(dst);
    }

    // Reads a window of this radius, so a tile needs that much margin.
    int ReachPixels() const override { return std::max(1, int(m_radius)); }

private:
    // Separable running-sum box mean: O(1) per pixel, which is the whole point.
    void BoxMean(const std::vector<float>& in, std::vector<float>& out,
                 int w, int h, int radius) {
        m_rowTmp.assign(in.size(), 0.0f);
        const float normX = 1.0f / float(radius * 2 + 1);

        for (int y = 0; y < h; ++y) {
            const size_t row = size_t(y) * size_t(w);
            float sum = 0.0f;
            for (int i = -radius; i <= radius; ++i)
                sum += in[row + size_t(std::clamp(i, 0, w - 1))];

            for (int x = 0; x < w; ++x) {
                m_rowTmp[row + size_t(x)] = sum * normX;
                sum += in[row + size_t(std::clamp(x + radius + 1, 0, w - 1))] -
                       in[row + size_t(std::clamp(x - radius, 0, w - 1))];
            }
        }

        const float normY = 1.0f / float(radius * 2 + 1);
        for (int x = 0; x < w; ++x) {
            float sum = 0.0f;
            for (int i = -radius; i <= radius; ++i)
                sum += m_rowTmp[size_t(std::clamp(i, 0, h - 1)) * size_t(w) + size_t(x)];

            for (int y = 0; y < h; ++y) {
                out[size_t(y) * size_t(w) + size_t(x)] = sum * normY;
                sum += m_rowTmp[size_t(std::clamp(y + radius + 1, 0, h - 1)) * size_t(w) + size_t(x)] -
                       m_rowTmp[size_t(std::clamp(y - radius, 0, h - 1)) * size_t(w) + size_t(x)];
            }
        }
    }

    Param<int> m_radius{
        this, "radius", 4, 1, 64,
        {.help = "Half-width of the local window, in pixels. Higher smooths "
                 "over a wider area, and costs the same at any size.",
         .softMin = 1, .softMax = 20}};
    // Fraction of the intensity range below which a region counts as flat.
    Param<float> m_eps{
        this, "eps", 0.1f, 0.001f, 1.0f,
        {.help = "How much local variation counts as noise rather than an "
                 "edge, as a fraction of the intensity range. Plays the same "
                 "role as the bilateral's sigma_range: LOWER preserves more "
                 "edges, higher smooths through them.",
         .step = 0.005, .softMin = 0.01, .softMax = 0.4}};

    std::vector<float> m_mean, m_meanSq, m_a, m_b, m_scratch, m_plane, m_rowTmp;
    PixelBuffer m_in, m_out;
};

REGISTER_ALGORITHM(GuidedFilter);

} // namespace tglab
