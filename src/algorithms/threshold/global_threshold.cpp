// Global thresholding: one threshold for the whole image.
//
// `threshold` takes the level directly. The automatic methods (Otsu, Triangle,
// IsoData) differ only in how they pick that level from the histogram, so they
// share a base class and implement one function each.
#include <algorithm>
#include <cstring>

#include "../../algo_util/histogram.h"
#include "../../core/algorithm.h"

namespace tglab {

namespace {

// Writes a binary result: 1 where the source is above the threshold.
// R32F output rather than RGBA8 so the result composes with the float stages
// (non_max_suppression, hysteresis) without a format conversion.
void WriteBinary(const ImageView& src, ImageView& dst, double threshold, bool invert) {
    const int w = src.desc.width;
    const int h = src.desc.height;
    for (int y = 0; y < h; ++y) {
        float* row = dst.At<float>(0, y);
        for (int x = 0; x < w; ++x) {
            const bool above = SampleValue(src, x, y, -1) > threshold;
            row[x] = (above != invert) ? 1.0f : 0.0f;
        }
    }
}

} // namespace

// Shared plumbing for every global method: the only difference is PickLevel().
class GlobalThresholdBase : public AlgorithmBase {
public:
    const char* Category() const override { return "threshold"; }

    PortList Inputs()  const override { return {{"src", DataType::Image, FormatSpec::Any}}; }
    PortList Outputs() const override { return {{"mask", DataType::Image, FormatSpec::R32F}}; }

    void RunCPU(RunCtx& ctx) override {
        const ImageView src = ctx.In(0);
        ImageView       dst = ctx.Out(0);
        if (!src.Valid() || !dst.Valid()) return;
        WriteBinary(src, dst, PickLevel(src), m_invert);
    }

protected:
    // Returns the threshold in the source's own units.
    virtual double PickLevel(const ImageView& src) = 0;

    Param<bool> m_invert{this, "invert", false};
};

// --- manual -----------------------------------------------------------------

class Threshold : public GlobalThresholdBase {
public:
    const char* Name() const override { return "threshold"; }

    bool HasGPU() const override { return true; }
    const char* GpuSource() const override {
        return R"(
Texture2D<float4>   Src : register(t0);
RWTexture2D<float4> Dst : register(u0);
cbuffer Params : register(b0) {
    uint Width;
    uint Height;
    uint LevelBits;
    uint Invert;
};
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    float level = asfloat(LevelBits);
    float4 c = Src[tid.xy];
    float luma = dot(c.rgb, float3(0.299, 0.587, 0.114)) * 255.0;
    bool above = luma > level;
    if (Invert != 0) above = !above;
    Dst[tid.xy] = above ? 1.0 : 0.0;
}
)";
    }
    std::vector<uint32_t> GpuConstants(int) const override {
        const float level = m_level;
        uint32_t bits;
        std::memcpy(&bits, &level, sizeof(bits));
        return {bits, m_invert ? 1u : 0u};
    }

protected:
    double PickLevel(const ImageView&) override { return double(float(m_level)); }

private:
    Param<float> m_level{this, "level", 128.0f, 0.0f, 255.0f};
};

REGISTER_ALGORITHM(Threshold);

// --- automatic, histogram-based ---------------------------------------------
//
// These are pure histogram analysis: build once, pick a level, then do exactly
// what `threshold` does. That is the whole point of putting the analysis in
// algo_util rather than duplicating it per algorithm.

class OtsuThreshold : public GlobalThresholdBase {
public:
    const char* Name() const override { return "threshold_otsu"; }

protected:
    double PickLevel(const ImageView& src) override {
        Histogram h;
        h.Build(src);
        return h.OtsuThreshold();
    }
};

REGISTER_ALGORITHM(OtsuThreshold);

class TriangleThreshold : public GlobalThresholdBase {
public:
    const char* Name() const override { return "threshold_triangle"; }

protected:
    double PickLevel(const ImageView& src) override {
        Histogram h;
        h.Build(src);
        return h.TriangleThreshold();
    }
};

REGISTER_ALGORITHM(TriangleThreshold);

class IsoDataThreshold : public GlobalThresholdBase {
public:
    const char* Name() const override { return "threshold_isodata"; }

protected:
    double PickLevel(const ImageView& src) override {
        Histogram h;
        h.Build(src);
        return h.IsoDataThreshold();
    }
};

REGISTER_ALGORITHM(IsoDataThreshold);

} // namespace tglab
