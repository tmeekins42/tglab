// brightness — the M1 vertical-slice algorithm.
//
// Deliberately trivial: it exists to debug the framework (script -> registry
// -> ports -> params -> display), not the math.
#include <algorithm>

#include "../../core/algorithm.h"

namespace tglab {

class Brightness : public AlgorithmBase {
public:
    const char* Name()     const override { return "brightness"; }
    const char* Category() const override { return "adjust"; }

    PortList Inputs()  const override { return {{"src", DataType::Image, FormatSpec::Any}}; }
    PortList Outputs() const override { return {{"out", DataType::Image, FormatSpec::SameAsInput}}; }

    void RunCPU(RunCtx& ctx) override {
        const ImageView src = ctx.In(0);
        ImageView       dst = ctx.Out(0);
        if (!src.Valid() || !dst.Valid()) return;

        const float amount = m_amount;   // reads as a plain float
        const float gain   = m_gain;

        const int w = src.desc.width;
        const int h = src.desc.height;

        if (src.desc.format == Format::RGBA8) {
            for (int y = 0; y < h; ++y) {
                const uint8_t* s = src.At<uint8_t>(0, y);
                uint8_t*       d = dst.At<uint8_t>(0, y);
                for (int x = 0; x < w * 4; x += 4) {
                    for (int c = 0; c < 3; ++c) {
                        const float v = float(s[x + c]) * gain + amount * 255.0f;
                        d[x + c] = uint8_t(std::clamp(v, 0.0f, 255.0f));
                    }
                    d[x + 3] = s[x + 3];   // alpha untouched
                }
            }
        } else if (src.desc.format == Format::R32F) {
            for (int y = 0; y < h; ++y) {
                const float* s = src.At<float>(0, y);
                float*       d = dst.At<float>(0, y);
                for (int x = 0; x < w; ++x) d[x] = s[x] * gain + amount;
            }
        }
    }

private:
    Param<float> m_amount{this, "amount", 0.0f, -1.0f, 1.0f};
    Param<float> m_gain  {this, "gain",   1.0f,  0.0f, 4.0f};
};

REGISTER_ALGORITHM(Brightness);

} // namespace tglab
