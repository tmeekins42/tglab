// median_blur — the per-channel median over a square window.
//
// The classic edge-preserving smoother, and the right tool for salt-and-pepper
// noise: a single extreme pixel cannot drag the median the way it drags a mean.
// Edges survive because the median of a window straddling one returns a value
// from whichever side holds the majority, rather than blending the two.
//
// Two implementations, chosen by format:
//
//   - 8-bit data uses a 256-bin sliding histogram. Moving one pixel to the
//     right adds a column and removes a column, so the cost per pixel is O(r)
//     rather than the O(r^2 log r) of sorting each window from scratch. This
//     is Huang's algorithm, and it is what makes a large radius usable.
//   - float data has no bounded bin set, so it falls back to nth_element,
//     which is linear-time selection rather than a full sort.
#include <algorithm>
#include <vector>

#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"

namespace tglab {

class MedianBlur : public AlgorithmBase {
public:
    const char* Name()     const override { return "median_blur"; }
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

        const int radius = std::max(1, int(m_radius));
        if (src.desc.format == Format::RGBA8) RunHistogram(radius);
        else                                  RunSelection(radius);

        m_out.PackInto(dst);
    }

private:
    // Huang's sliding histogram. Values are 0..255, so 256 bins suffice and the
    // median is found by walking bins until half the window's weight is passed.
    void RunHistogram(int radius) {
        const int w = m_in.Width(), h = m_in.Height(), ch = m_in.Channels();
        const int windowCount = (radius * 2 + 1) * (radius * 2 + 1);
        const int half = windowCount / 2;

        // Alpha is copied rather than filtered: median-ing it would erode
        // hard-edged transparency for no benefit.
        const int filtered = std::min(ch, 3);

        std::vector<int> hist(256, 0);

        for (int y = 0; y < h; ++y) {
            for (int c = 0; c < filtered; ++c) {
                std::fill(hist.begin(), hist.end(), 0);

                // Seed the histogram with the window at x = 0.
                for (int dy = -radius; dy <= radius; ++dy)
                    for (int dx = -radius; dx <= radius; ++dx)
                        ++hist[Bin(dx, y + dy, c, w, h)];

                for (int x = 0; x < w; ++x) {
                    // Walk bins until the cumulative count passes the midpoint.
                    int count = 0, bin = 0;
                    for (; bin < 256; ++bin) {
                        count += hist[size_t(bin)];
                        if (count > half) break;
                    }
                    m_out.Set(x, y, c, float(std::min(bin, 255)));

                    if (x + 1 >= w) continue;
                    // Slide one column: drop x-radius, add x+radius+1.
                    for (int dy = -radius; dy <= radius; ++dy) {
                        --hist[Bin(x - radius, y + dy, c, w, h)];
                        ++hist[Bin(x + radius + 1, y + dy, c, w, h)];
                    }
                }
            }
            if (ch == 4)
                for (int x = 0; x < w; ++x) m_out.Set(x, y, 3, m_in.Get(x, y, 3));
        }
    }

    // nth_element selects the median without fully sorting the window.
    void RunSelection(int radius) {
        const int w = m_in.Width(), h = m_in.Height(), ch = m_in.Channels();
        const int filtered = (ch == 1) ? 1 : 3;

        std::vector<float> window;
        window.reserve(size_t(radius * 2 + 1) * size_t(radius * 2 + 1));

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                for (int c = 0; c < filtered; ++c) {
                    window.clear();
                    for (int dy = -radius; dy <= radius; ++dy)
                        for (int dx = -radius; dx <= radius; ++dx)
                            window.push_back(m_in.AtClamped(x + dx, y + dy)[c]);

                    const size_t mid = window.size() / 2;
                    std::nth_element(window.begin(), window.begin() + mid, window.end());
                    m_out.Set(x, y, c, window[mid]);
                }
                if (ch == 4) m_out.Set(x, y, 3, m_in.Get(x, y, 3));
            }
        }
    }

    int Bin(int x, int y, int c, int w, int h) const {
        const float v = m_in.AtClamped(x, y)[c];
        return std::clamp(int(v + 0.5f), 0, 255);
    }

    Param<int> m_radius{
        this, "radius", 2, 1, 32,
        {.help = "Half-width of the window the median is taken over. "
                 "Higher removes larger specks but rounds off fine corners.",
         .softMin = 1, .softMax = 10}};

    PixelBuffer m_in, m_out;
};

REGISTER_ALGORITHM(MedianBlur);

} // namespace tglab
