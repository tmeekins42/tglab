// canny — the composite edge detector.
//
// Chains the same registered algorithms the script could call individually
// (gaussian_blur -> sobel -> non_max_suppression -> hysteresis) rather than
// duplicating their code. Use canny() when you just want edges; call the
// stages separately when you want to see what each one did.
#include <memory>
#include <vector>

#include "../../core/algorithm.h"

namespace tglab {

class Canny : public AlgorithmBase {
public:
    const char* Name()     const override { return "canny"; }
    const char* Category() const override { return "edge"; }

    PortList Inputs()  const override { return {{"src", DataType::Image, FormatSpec::Any}}; }
    PortList Outputs() const override { return {{"edges", DataType::Image, FormatSpec::R32F}}; }

    void RunCPU(RunCtx& ctx) override {
        const ImageView srcView = ctx.In(0);
        ImageView outView = ctx.Out(0);
        if (!srcView.Valid() || !outView.Valid()) return;

        EnsureStages();
        if (!m_blur || !m_sobel || !m_nms || !m_hyst) return;

        const ImageDesc srcDesc = srcView.desc;
        const ImageDesc f32{srcDesc.width, srcDesc.height, Format::R32F};

        // Forward this algorithm's parameters to the stages it wraps.
        SetParam(*m_blur, "sigma", float(m_sigma));
        SetParam(*m_hyst, "low",   float(m_low));
        SetParam(*m_hyst, "high",  float(m_high));

        // 1. blur (same format as input)
        Data blurred = MakeImage(srcDesc);
        RunStage(*m_blur, {ctx.InData(0)}, {&blurred});

        // 2. gradients
        Data gx = MakeImage(f32), gy = MakeImage(f32), mag = MakeImage(f32);
        RunStage(*m_sobel, {&blurred}, {&gx, &gy, &mag});

        // 3. thin the ridges
        Data thin = MakeImage(f32);
        RunStage(*m_nms, {&gx, &gy, &mag}, {&thin});

        // 4. hysteresis, straight into our own output
        Data edges = MakeImage(f32);
        RunStage(*m_hyst, {&thin}, {&edges});

        // Copy the result into the port the pipeline allocated for us.
        ImageView e = std::get<Image>(edges).MapCpuRead();
        const int w = srcDesc.width, h = srcDesc.height;
        for (int y = 0; y < h; ++y) {
            const float* s = e.At<float>(0, y);
            float* d = outView.At<float>(0, y);
            for (int x = 0; x < w; ++x) d[x] = s[x];
        }
    }

private:
    static Data MakeImage(const ImageDesc& d) {
        Image img;
        img.Alloc(d);
        return Data{std::move(img)};
    }

    static void SetParam(AlgorithmBase& a, const char* name, float v) {
        if (ParamBase* p = a.FindParam(name))
            if (auto* f = dynamic_cast<Param<float>*>(p)) f->set(v);
    }

    static void RunStage(AlgorithmBase& a,
                         std::vector<const Data*> in,
                         std::vector<Data*> outPtrs) {
        // RunCtx wants a contiguous span of Data for outputs.
        std::vector<Data> outs(outPtrs.size());
        for (size_t i = 0; i < outPtrs.size(); ++i) outs[i] = std::move(*outPtrs[i]);

        RunCtx c(in, outs);
        a.RunCPU(c);

        for (size_t i = 0; i < outPtrs.size(); ++i) *outPtrs[i] = std::move(outs[i]);
    }

    void EnsureStages() {
        if (m_blur) return;
        m_blur  = Registry::Get().Create("gaussian_blur");
        m_sobel = Registry::Get().Create("sobel");
        m_nms   = Registry::Get().Create("non_max_suppression");
        m_hyst  = Registry::Get().Create("hysteresis");
    }

    Param<float> m_sigma{this, "sigma", 1.4f, 0.1f, 20.0f, {.step = 0.1, .softMin = 0.1, .softMax = 5.0}};
    Param<float> m_low  {this, "low",   0.10f, 0.0f, 4.0f};
    Param<float> m_high {this, "high",  0.30f, 0.0f, 4.0f};

    std::unique_ptr<AlgorithmBase> m_blur, m_sobel, m_nms, m_hyst;
};

REGISTER_ALGORITHM(Canny);

} // namespace tglab
