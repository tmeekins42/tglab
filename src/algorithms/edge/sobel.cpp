// sobel — 3x3 gradient operator.
//
// The first genuinely multi-output algorithm: gx and gy are *signed*, so they
// need R32F rather than RGBA8. Together these exercise N-ports and float
// intermediates, neither of which the M1 algorithms touched.
#include <algorithm>
#include <cmath>

#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"

namespace tglab {

namespace {

// Luma at a pixel, whatever the storage format, normalised to 0..1.
//
// This used to be an R32F branch with an RGBA8 fallthrough, which read a
// half-float or 32-bit-float image as bytes -- not zeros but plausible-looking
// nonsense, which is harder to notice than a black image. PixelBuffer knows
// every format; `scale` is 255 for RGBA8 and 1 for the float ones.
inline float Sample(const PixelBuffer& b, int x, int y, float unitScale) {
    const float* p = b.AtClamped(x, y);   // clamp-to-edge
    if (b.Channels() == 1) return p[0] / unitScale;
    return (0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2]) / unitScale;
}

} // namespace

class Sobel : public AlgorithmBase {
public:
    const char* Name()     const override { return "sobel"; }
    const char* Category() const override { return "edge"; }

    PortList Inputs() const override {
        return {{"src", DataType::Image, FormatSpec::Any}};
    }
    PortList Outputs() const override {
        return {{"gx",  DataType::Image, FormatSpec::R32F},
                {"gy",  DataType::Image, FormatSpec::R32F},
                {"mag", DataType::Image, FormatSpec::R32F}};
    }

    void RunCPU(RunCtx& ctx) override {
        const ImageView src = ctx.In(0);
        ImageView gx  = ctx.Out(0);
        ImageView gy  = ctx.Out(1);
        ImageView mag = ctx.Out(2);
        if (!src.Valid() || !gx.Valid() || !gy.Valid() || !mag.Valid()) return;

        m_in.Unpack(src);
        if (!m_in.Valid()) return;

        const float scale = m_scale;
        const float unit  = m_in.ValueScale();   // 255 for RGBA8, 1 for float
        const int w = m_in.Width();
        const int h = m_in.Height();

        for (int y = 0; y < h; ++y) {
            if (ctx.Cancelled()) return;
            float* rowX = gx.At<float>(0, y);
            float* rowY = gy.At<float>(0, y);
            float* rowM = mag.At<float>(0, y);

            for (int x = 0; x < w; ++x) {
                const float tl = Sample(m_in, x - 1, y - 1, unit);
                const float tc = Sample(m_in, x,     y - 1, unit);
                const float tr = Sample(m_in, x + 1, y - 1, unit);
                const float ml = Sample(m_in, x - 1, y,     unit);
                const float mr = Sample(m_in, x + 1, y,     unit);
                const float bl = Sample(m_in, x - 1, y + 1, unit);
                const float bc = Sample(m_in, x,     y + 1, unit);
                const float br = Sample(m_in, x + 1, y + 1, unit);

                //  gx: [-1 0 1; -2 0 2; -1 0 1]      gy: [-1 -2 -1; 0 0 0; 1 2 1]
                const float dx = (tr + 2.0f * mr + br) - (tl + 2.0f * ml + bl);
                const float dy = (bl + 2.0f * bc + br) - (tl + 2.0f * tc + tr);

                rowX[x] = dx * scale;
                rowY[x] = dy * scale;
                rowM[x] = std::sqrt(dx * dx + dy * dy) * scale;
            }
        }
    }

private:
    Param<float> m_scale{this, "scale", 1.0f, 0.0f, 8.0f};

    PixelBuffer m_in;
};

REGISTER_ALGORITHM(Sobel);

} // namespace tglab
