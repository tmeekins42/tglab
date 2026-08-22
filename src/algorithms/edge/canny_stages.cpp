// Canny's distinctive stages, exposed as separate algorithms.
//
// Keeping these callable on their own is the point of a research lab: you can
// put non_max_suppression's output in a viewer and actually see what it did,
// rather than only seeing the finished edge map. canny() (canny.cpp) chains
// them for when you just want edges.
#include <algorithm>
#include <cmath>
#include <vector>

#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"

namespace tglab {

// --- non-maximum suppression ------------------------------------------------
// Thins gradient ridges to single-pixel width by keeping a pixel only when its
// magnitude is >= both neighbours along the gradient direction.
class NonMaxSuppression : public AlgorithmBase {
public:
    const char* Name()     const override { return "non_max_suppression"; }

    // "edge stage", not "edge": a category is meant to hold interchangeable
    // algorithms, and this one takes three inputs (gx, gy, mag) rather than an
    // image. Leaving it in "edge" put it in choose("op", "edge") dropdowns
    // where selecting it could only ever fail.
    const char* Category() const override { return "edge stage"; }

    PortList Inputs() const override {
        return {{"gx",  DataType::Image, FormatSpec::R32F},
                {"gy",  DataType::Image, FormatSpec::R32F},
                {"mag", DataType::Image, FormatSpec::R32F}};
    }
    PortList Outputs() const override {
        return {{"thin", DataType::Image, FormatSpec::R32F}};
    }

    void RunCPU(RunCtx& ctx) override {
        const ImageView gx  = ctx.In(0);
        const ImageView gy  = ctx.In(1);
        const ImageView mag = ctx.In(2);
        ImageView out = ctx.Out(0);
        if (!gx.Valid() || !gy.Valid() || !mag.Valid() || !out.Valid()) return;

        const int w = mag.desc.width;
        const int h = mag.desc.height;

        for (int y = 0; y < h; ++y) {
            float* dst = out.At<float>(0, y);
            for (int x = 0; x < w; ++x) {
                const float m = *mag.At<float>(x, y);
                const float dx = *gx.At<float>(x, y);
                const float dy = *gy.At<float>(x, y);

                // Quantise the gradient direction to one of four neighbour
                // pairs (0/45/90/135 degrees).
                int ox = 0, oy = 0;
                const float ax = std::fabs(dx), ay = std::fabs(dy);
                if (ax > 2.414f * ay)        { ox = 1; oy = 0; }      // ~horizontal
                else if (ay > 2.414f * ax)   { ox = 0; oy = 1; }      // ~vertical
                else if ((dx > 0) == (dy > 0)) { ox = 1; oy = 1; }
                else                           { ox = 1; oy = -1; }

                const int x0 = std::clamp(x - ox, 0, w - 1), y0 = std::clamp(y - oy, 0, h - 1);
                const int x1 = std::clamp(x + ox, 0, w - 1), y1 = std::clamp(y + oy, 0, h - 1);
                const float a = *mag.At<float>(x0, y0);
                const float b = *mag.At<float>(x1, y1);

                dst[x] = (m >= a && m >= b) ? m : 0.0f;
            }
        }
    }
};

REGISTER_ALGORITHM(NonMaxSuppression);

// --- hysteresis thresholding ------------------------------------------------
// Keeps strong pixels, plus weak pixels connected to a strong one. This is what
// stops a single noisy edge from breaking into dashes.
class Hysteresis : public AlgorithmBase {
public:
    const char* Name()     const override { return "hysteresis"; }

    // Also a stage, not a standalone detector: it expects an R32F gradient
    // image, not a source image. See the note on NonMaxSuppression above.
    const char* Category() const override { return "edge stage"; }

    PortList Inputs()  const override { return {{"src", DataType::Image, FormatSpec::R32F}}; }
    PortList Outputs() const override { return {{"edges", DataType::Image, FormatSpec::R32F}}; }

    void RunCPU(RunCtx& ctx) override {
        const ImageView src = ctx.In(0);
        ImageView out = ctx.Out(0);
        if (!src.Valid() || !out.Valid()) return;

        // Through PixelBuffer rather than reading At<float> directly.
        //
        // The input port declares R32F, but that is documentation: the pipeline
        // uses FormatSpec to size OUTPUTS and never converts an input, so an
        // RGBA8 image reaching here had its bytes read as floats -- garbage,
        // and the thresholds below then rejected everything.
        m_in.Unpack(src);
        if (!m_in.Valid()) return;

        const int w = m_in.Width();
        const int h = m_in.Height();
        const float unit = m_in.ValueScale();   // 255 for RGBA8, 1 for float
        const float lo = std::min(float(m_low), float(m_high));
        const float hi = std::max(float(m_low), float(m_high));

        // 0 = suppressed, 1 = weak, 2 = strong.
        m_mark.assign(size_t(w) * size_t(h), 0);
        m_stack.clear();

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const float v = m_in.Get(x, y, 0) / unit;
                const size_t i = size_t(y) * size_t(w) + size_t(x);
                if (v >= hi)      { m_mark[i] = 2; m_stack.push_back(int(i)); }
                else if (v >= lo) { m_mark[i] = 1; }
            }
        }

        // Flood weak pixels that touch a strong one (8-connected).
        while (!m_stack.empty()) {
            const int i = m_stack.back();
            m_stack.pop_back();
            const int x = i % w, y = i / w;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    const int nx = x + dx, ny = y + dy;
                    if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                    const size_t n = size_t(ny) * size_t(w) + size_t(nx);
                    if (m_mark[n] == 1) {
                        m_mark[n] = 2;
                        m_stack.push_back(int(n));
                    }
                }
            }
        }

        for (int y = 0; y < h; ++y) {
            float* dst = out.At<float>(0, y);
            for (int x = 0; x < w; ++x)
                dst[x] = (m_mark[size_t(y) * size_t(w) + size_t(x)] == 2) ? 1.0f : 0.0f;
        }
    }

private:
    Param<float> m_low {this, "low",  0.10f, 0.0f, 4.0f};
    Param<float> m_high{this, "high", 0.30f, 0.0f, 4.0f};

    PixelBuffer          m_in;
    std::vector<uint8_t> m_mark;
    std::vector<int>     m_stack;
};

REGISTER_ALGORITHM(Hysteresis);

} // namespace tglab
