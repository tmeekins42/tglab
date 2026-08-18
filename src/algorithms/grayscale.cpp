// grayscale — second algorithm, added without touching any central file.
// Also the M1 check that self-registration survives the linker.
#include <algorithm>

#include "../core/algorithm.h"

namespace tglab {

class Grayscale : public AlgorithmBase {
public:
    const char* Name()     const override { return "grayscale"; }
    const char* Category() const override { return "color"; }

    PortList Inputs()  const override { return {{"src", DataType::Image, FormatSpec::Any}}; }
    PortList Outputs() const override { return {{"out", DataType::Image, FormatSpec::SameAsInput}}; }

    void RunCPU(RunCtx& ctx) override {
        const ImageView src = ctx.In(0);
        ImageView       dst = ctx.Out(0);
        if (!src.Valid() || !dst.Valid()) return;
        if (src.desc.format != Format::RGBA8) return;

        // Rec.601 by default; the sliders let the weights be explored, which
        // is the kind of thing this lab is for.
        const float wr = m_wr, wg = m_wg, wb = m_wb;
        const float sum = std::max(0.0001f, wr + wg + wb);

        const int w = src.desc.width;
        const int h = src.desc.height;
        for (int y = 0; y < h; ++y) {
            const uint8_t* s = src.At<uint8_t>(0, y);
            uint8_t*       d = dst.At<uint8_t>(0, y);
            for (int x = 0; x < w * 4; x += 4) {
                const float g = (float(s[x]) * wr + float(s[x + 1]) * wg + float(s[x + 2]) * wb) / sum;
                const uint8_t v = uint8_t(std::clamp(g, 0.0f, 255.0f));
                d[x] = d[x + 1] = d[x + 2] = v;
                d[x + 3] = s[x + 3];
            }
        }
    }

private:
    Param<float> m_wr{this, "r_weight", 0.299f, 0.0f, 1.0f};
    Param<float> m_wg{this, "g_weight", 0.587f, 0.0f, 1.0f};
    Param<float> m_wb{this, "b_weight", 0.114f, 0.0f, 1.0f};
};

REGISTER_ALGORITHM(Grayscale);

} // namespace tglab
