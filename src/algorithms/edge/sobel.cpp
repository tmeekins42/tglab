// sobel — 3x3 gradient operator.
//
// The first genuinely multi-output algorithm: gx and gy are *signed*, so they
// need R32F rather than RGBA8. Together these exercise N-ports and float
// intermediates, neither of which the M1 algorithms touched.
#include <algorithm>
#include <cmath>

#include "../../core/algorithm.h"

namespace tglab {

namespace {

// Luma of an RGBA8 pixel, or the value itself when already single-channel.
inline float Sample(const ImageView& v, int x, int y) {
    x = std::clamp(x, 0, v.desc.width  - 1);   // clamp-to-edge
    y = std::clamp(y, 0, v.desc.height - 1);
    if (v.desc.format == Format::R32F) return *v.At<float>(x, y);
    const uint8_t* p = v.At<uint8_t>(x, y);
    return (0.299f * float(p[0]) + 0.587f * float(p[1]) + 0.114f * float(p[2])) / 255.0f;
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

        const float scale = m_scale;
        const int w = src.desc.width;
        const int h = src.desc.height;

        for (int y = 0; y < h; ++y) {
            float* rowX = gx.At<float>(0, y);
            float* rowY = gy.At<float>(0, y);
            float* rowM = mag.At<float>(0, y);

            for (int x = 0; x < w; ++x) {
                const float tl = Sample(src, x - 1, y - 1), tc = Sample(src, x, y - 1), tr = Sample(src, x + 1, y - 1);
                const float ml = Sample(src, x - 1, y    ),                             mr = Sample(src, x + 1, y    );
                const float bl = Sample(src, x - 1, y + 1), bc = Sample(src, x, y + 1), br = Sample(src, x + 1, y + 1);

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
};

REGISTER_ALGORITHM(Sobel);

} // namespace tglab
