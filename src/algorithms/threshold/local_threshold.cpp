// Local (adaptive) thresholding: a different threshold per pixel, computed
// from a window around it. These handle uneven lighting, where any single
// global threshold necessarily fails somewhere in the image.
//
// Every method here is O(1) or O(2r) per pixel, never O(r^2). Niblack,
// Sauvola and the adaptive-mean pair need windowed mean/stddev, so they share
// an integral image; Bernsen needs window min/max, which an integral image
// cannot give, so it uses a separable min/max filter instead. At 8 MP with a
// 15x15 window the direct O(r^2) version took over two minutes -- long enough
// to look like the app had hung.
#include <algorithm>
#include <cmath>
#include <cstring>

#include "../../algo_util/histogram.h"
#include "../../core/algorithm.h"

namespace tglab {

namespace {

// Window radius from an odd window size, the form these methods are published
// in (Niblack's paper uses 15x15).
inline int RadiusFromWindow(int window) {
    return std::max(1, (std::max(3, window) - 1) / 2);
}

} // namespace

// Shared plumbing. Subclasses supply the per-pixel threshold given the local
// statistics, which is the only thing that actually differs between them.
// Largest window a GPU kernel takes. A pass fetches (2r+1) texels, so at
// window 51 that is 51 per pixel per pass -- comfortable. Beyond it the CPU
// path is separable and O(2r) and so is not the bottleneck anyway, while a
// single dispatch doing far more would risk the watchdog.
//
// At file scope because it is a property of the dispatch shape rather than of
// any one class: AdaptiveGaussianThreshold does not derive from the base below.
constexpr int kMaxGpuWindow = 51;

class LocalThresholdBase : public AlgorithmBase {
public:
    const char* Category() const override { return "threshold"; }

    PortList Inputs()  const override { return {{"src", DataType::Image, FormatSpec::Any}}; }
    PortList Outputs() const override { return {{"mask", DataType::Image, FormatSpec::R32F}}; }

    void RunCPU(RunCtx& ctx) override {
        const ImageView src = ctx.In(0);
        ImageView       dst = ctx.Out(0);
        if (!src.Valid() || !dst.Valid()) return;

        const int radius = RadiusFromWindow(int(m_window));
        const int w = src.desc.width;
        const int h = src.desc.height;
        const bool invert = m_invert;

        if (UsesIntegralImage()) {
            m_integral.Build(src);
            for (int y = 0; y < h; ++y) {
                float* row = dst.At<float>(0, y);
                for (int x = 0; x < w; ++x) {
                    const LocalStats s = m_integral.Window(x, y, radius);
                    const double t = Threshold(s);
                    const bool above = SampleValue(src, x, y, -1) > t;
                    row[x] = (above != invert) ? 1.0f : 0.0f;
                }
            }
        } else {
            // Separable min/max rather than a per-pixel window scan. The direct
            // version is O(r^2): at 8 MP with a 15x15 window that is ~1.8
            // billion samples, which took over two minutes and looked like the
            // app had hung.
            m_minmax.Build(src, radius);
            for (int y = 0; y < h; ++y) {
                float* row = dst.At<float>(0, y);
                for (int x = 0; x < w; ++x) {
                    LocalStats s;
                    s.minV = m_minmax.Min(x, y);
                    s.maxV = m_minmax.Max(x, y);
                    s.mean = 0.5 * (s.minV + s.maxV);   // Bernsen uses the midpoint
                    const double t = Threshold(s);
                    const bool above = SampleValue(src, x, y, -1) > t;
                    row[x] = (above != invert) ? 1.0f : 0.0f;
                }
            }
        }
    }


public:
    // --- GPU implementation -------------------------------------------------
    //
    // Two passes of a separable box sum: horizontal into scratch, then vertical
    // while thresholding.
    //
    // The CPU gets its windowed moments from an integral image, which is a 2D
    // prefix sum and therefore inherently sequential -- that is what made this
    // look like it needed a multi-dispatch GPU scan. But the window is a
    // RECTANGLE, and a rectangular sum separates: sum over x, then over y. Two
    // passes, no scan, no extra scratch buffer.
    //
    // The intermediate carries a sum AND a sum-of-squares (stddev needs the
    // second moment), plus the count. That is what GpuScratchFormat::RGBA32F
    // provides; sized from the R32F output only the first would survive.
    //
    // Border handling matches the CPU exactly and deliberately: the window is
    // CLIPPED to the image and divided by however many pixels actually fell
    // inside, rather than clamp-sampling the edge pixel repeatedly. The two
    // give different answers along the border, and the agreement test is what
    // holds them together.
    bool HasGPU() const override { return int(m_window) <= kMaxGpuWindow; }
    int  GpuIterations() const override { return 2; }
    FormatSpec GpuScratchFormat() const override { return FormatSpec::RGBA32F; }

    const char* GpuSource() const override {
        return R"(
Texture2D<float4>   Src  : register(t0);   // ping-pong: source, then h-moments
Texture2D<float4>   Orig : register(t1);   // the untouched input, every pass
RWTexture2D<float4> Dst  : register(u0);

cbuffer Params : register(b0) {
    uint Width;
    uint Height;
    uint Radius;
    uint Mode;      // 0 = niblack, 1 = sauvola, 2 = adaptive mean
    uint ABits;     // niblack k / sauvola k / mean c   (c in 0..1 units)
    uint BBits;     // sauvola R, in 0..1 units
    uint Invert;
    uint Pass;      // 0 = horizontal, 1 = vertical + threshold
};

float Luma(float4 c) { return dot(c.rgb, float3(0.299, 0.587, 0.114)); }

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;

    int r = int(Radius);

    if (Pass == 0) {
        // Horizontal: sum, sum of squares, and the number of pixels actually
        // inside the image -- clipped, not clamped, to match the CPU.
        int x0 = max(int(tid.x) - r, 0);
        int x1 = min(int(tid.x) + r, int(Width) - 1);
        float s = 0.0, s2 = 0.0;
        for (int x = x0; x <= x1; ++x) {
            float v = Luma(Src[int2(x, int(tid.y))]);
            s  += v;
            s2 += v * v;
        }
        Dst[tid.xy] = float4(s, s2, float(x1 - x0 + 1), 1.0);
        return;
    }

    // Vertical: accumulate the horizontal partials over the clipped row span.
    int y0 = max(int(tid.y) - r, 0);
    int y1 = min(int(tid.y) + r, int(Height) - 1);
    float s = 0.0, s2 = 0.0, n = 0.0;
    for (int y = y0; y <= y1; ++y) {
        float3 p = Src[int2(int(tid.x), y)].rgb;
        s  += p.r;
        s2 += p.g;
        n  += p.b;
    }
    if (n <= 0.0) { Dst[tid.xy] = 0.0; return; }

    float mean   = s / n;
    float var    = max(0.0, s2 / n - mean * mean);
    float stddev = sqrt(var);

    float a = asfloat(ABits);
    float t;
    if (Mode == 0) {
        t = mean + a * stddev;                       // niblack
    } else if (Mode == 1) {
        float R = max(1e-6, asfloat(BBits));         // sauvola
        t = mean * (1.0 + a * (stddev / R - 1.0));
    } else {
        t = mean - a;                                // adaptive mean
    }

    float v = Luma(Orig[int2(tid.xy)]);
    bool above = v > t;
    if (Invert != 0) above = !above;
    Dst[tid.xy] = above ? 1.0 : 0.0;
}
)";
    }

    std::vector<uint32_t> GpuConstants(int iteration) const override {
        auto bits = [](float f) {
            uint32_t u;
            std::memcpy(&u, &f, sizeof(u));
            return u;
        };
        return {uint32_t(RadiusFromWindow(int(m_window))),
                uint32_t(GpuMode()),
                bits(GpuParamA()),
                bits(GpuParamB()),
                uint32_t(bool(m_invert) ? 1 : 0),
                uint32_t(iteration)};
    }

protected:
    // Which branch of the shared kernel this thresholder wants, and its
    // coefficients. One kernel rather than three near-identical ones: the
    // moment passes are the whole cost and are byte-for-byte the same, so
    // splitting them would triple the shader compiles to no benefit.
    virtual int   GpuMode()   const = 0;
    virtual float GpuParamA() const = 0;
    virtual float GpuParamB() const { return 0.0f; }

protected:
    virtual double Threshold(const LocalStats& s) const = 0;
    virtual bool   UsesIntegralImage() const { return true; }

    Param<int>  m_window{this, "window", 15, 3, 201, {.help = "Side length of the neighbourhood each pixel's threshold is computed from, in pixels. Should be larger than the strokes you want to keep but smaller than the illumination changes you want to correct. Odd values only.", .step = 2, .softMin = 3, .softMax = 51}};
    Param<bool> m_invert{this, "invert", false,
                         "Swap foreground and background: on gives white text on black."};

private:
    IntegralImage m_integral;
    MinMaxFilter  m_minmax;
};

// --- Niblack ----------------------------------------------------------------
// t = mean + k * stddev, with k negative. Simple and effective on text, but
// noisy in flat regions because stddev there is near zero and the threshold
// collapses onto the mean.
class NiblackThreshold : public LocalThresholdBase {
public:
    const char* Name() const override { return "threshold_niblack"; }

    // Niblack: t = mean + k*stddev. k multiplies a stddev, so it is
    // scale-free -- no 0..255 conversion, unlike sauvola's R.
    int   GpuMode()   const override { return 0; }
    float GpuParamA() const override { return float(m_k); }


protected:
    double Threshold(const LocalStats& s) const override {
        return s.mean + double(float(m_k)) * s.stddev;
    }

private:
    Param<float> m_k{this, "k", -0.2f, -1.0f, 1.0f,
                     {.help = "Weight on the local standard deviation. Niblack sets the "
                              "threshold at mean + k*stddev, so more negative keeps more "
                              "foreground. Typically around -0.2.",
                      .step = 0.01}};
};

REGISTER_ALGORITHM(NiblackThreshold);

// --- Sauvola ----------------------------------------------------------------
// t = mean * (1 + k * (stddev / R - 1)). The dynamic range R normalises the
// stddev term, so flat regions push the threshold below the mean instead of
// onto it — which is exactly Niblack's failure mode on background noise.
class SauvolaThreshold : public LocalThresholdBase {
public:
    const char* Name() const override { return "threshold_sauvola"; }

    // Sauvola: t = mean * (1 + k*(stddev/R - 1)). k is scale-free; R is a
    // stddev in the source's units, so it scales to 0..1 for the shader.
    int   GpuMode()   const override { return 1; }
    float GpuParamA() const override { return float(m_k); }
    float GpuParamB() const override { return float(m_r) / 255.0f; }


protected:
    double Threshold(const LocalStats& s) const override {
        const double R = std::max(1e-6, double(float(m_r)));
        return s.mean * (1.0 + double(float(m_k)) * (s.stddev / R - 1.0));
    }

private:
    Param<float> m_k{this, "k", 0.2f, 0.0f, 1.0f,
                     {.help = "Sensitivity to local contrast. Sauvola sets the threshold at "
                              "mean * (1 + k*(stddev/r - 1)), so HIGHER k thins strokes and "
                              "removes more background stain. Typically 0.2 to 0.5.",
                      .step = 0.01}};
    // R is the expected dynamic range of the stddev; 128 is the standard
    // choice for 8-bit input.
    Param<float> m_r{this, "r", 128.0f, 1.0f, 255.0f,
                     {.help = "Expected dynamic range of the local standard deviation. "
                              "Rarely needs changing from 128 for 8-bit scans.",
                      .step = 1.0}};
};

REGISTER_ALGORITHM(SauvolaThreshold);

// --- Bernsen ----------------------------------------------------------------
// t = midpoint of the window's min and max. Where local contrast is below a
// cutoff the window is treated as uniform and assigned wholesale, which stops
// it from amplifying noise in flat areas.
class BernsenThreshold : public LocalThresholdBase {
public:
    const char* Name() const override { return "threshold_bernsen"; }

protected:
    // Needs window min/max, which an integral image cannot give.
    bool UsesIntegralImage() const override { return false; }

    double Threshold(const LocalStats& s) const override {
        const double contrast = s.maxV - s.minV;
        if (contrast < double(float(m_contrastMin))) {
            // Uniform window: force the whole thing to one side by returning a
            // threshold outside the data range.
            return (s.mean >= double(float(m_globalLevel))) ? -1e30 : 1e30;
        }
        return 0.5 * (s.minV + s.maxV);
    }

public:
    // --- GPU implementation -------------------------------------------------
    //
    // Two passes of a separable min/max, the same decomposition the CPU uses:
    // horizontal into scratch, then vertical while thresholding. Separable
    // because a direct window scan is O(r^2) -- at 8 MP with a 15x15 window
    // that is ~1.8 billion samples.
    //
    // The scratch is RGBA32F rather than the R32F output, because this pass has
    // TWO values to carry: the window minimum and maximum. An earlier version
    // wrote both into a single-channel scratch, silently kept only the min, and
    // got 50% of pixels wrong -- which is what GpuScratchFormat exists to fix.
    bool HasGPU() const override { return int(m_window) <= kMaxGpuWindow; }
    int  GpuIterations() const override { return 2; }
    FormatSpec GpuScratchFormat() const override { return FormatSpec::RGBA32F; }

    // Bernsen replaces the shared moments kernel entirely (it needs window
    // min/max, not mean/stddev), so these exist only to satisfy the base.
    // GpuMode is never read: GpuConstants below is overridden too.
    int   GpuMode()   const override { return -1; }
    float GpuParamA() const override { return 0.0f; }

    const char* GpuSource() const override {
        return R"(
Texture2D<float4>   Src  : register(t0);   // ping-pong: source, then h-min/max
Texture2D<float4>   Orig : register(t1);   // the untouched input, every pass
RWTexture2D<float4> Dst  : register(u0);

cbuffer Params : register(b0) {
    uint Width;
    uint Height;
    uint Radius;
    uint ContrastMinBits;   // in 0..1 units
    uint GlobalLevelBits;   // in 0..1 units
    uint Invert;
    uint Pass;              // 0 = horizontal min/max, 1 = vertical + threshold
};

float Luma(float4 c) { return dot(c.rgb, float3(0.299, 0.587, 0.114)); }

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;

    int  r  = int(Radius);
    int2 hi = int2(Width - 1, Height - 1);

    float lo = 1e30, hiV = -1e30;
    for (int i = -r; i <= r; ++i) {
        int2 p = (Pass == 0) ? int2(int(tid.x) + i, int(tid.y))
                             : int2(int(tid.x), int(tid.y) + i);
        p = clamp(p, int2(0, 0), hi);
        if (Pass == 0) {
            // Colour in, luma out.
            float v = Luma(Src[p]);
            lo  = min(lo,  v);
            hiV = max(hiV, v);
        } else {
            // Pass 0 packed min into .r and max into .g -- the two channels the
            // RGBA32F scratch exists to provide.
            lo  = min(lo,  Src[p].r);
            hiV = max(hiV, Src[p].g);
        }
    }

    if (Pass == 0) {
        Dst[tid.xy] = float4(lo, hiV, 0.0, 1.0);
        return;
    }

    float contrast = hiV - lo;
    float v = Luma(Orig[int2(tid.xy)]);

    bool above;
    if (contrast < asfloat(ContrastMinBits)) {
        // Uniform window: resolve the whole thing to one side, exactly as the
        // CPU's +/-1e30 sentinel threshold does.
        above = (0.5 * (lo + hiV)) >= asfloat(GlobalLevelBits);
    } else {
        above = v > (0.5 * (lo + hiV));
    }
    if (Invert != 0) above = !above;
    Dst[tid.xy] = above ? 1.0 : 0.0;
}
)";
    }

    std::vector<uint32_t> GpuConstants(int iteration) const override {
        auto bits = [](float f) {
            uint32_t u;
            std::memcpy(&u, &f, sizeof(u));
            return u;
        };
        // Both thresholds are declared in 0..255 but a UNORM SRV hands the
        // shader 0..1, so they are scaled here -- the same units trap as
        // brightness's offset.
        return {uint32_t(RadiusFromWindow(int(m_window))),
                bits(float(m_contrastMin) / 255.0f),
                bits(float(m_globalLevel) / 255.0f),
                uint32_t(bool(m_invert) ? 1 : 0),
                uint32_t(iteration)};
    }
private:
    Param<float> m_contrastMin{this, "contrast_min", 15.0f, 0.0f, 128.0f,
        {.help = "Below this local contrast (max minus min) the window is treated as "
                 "uniform and uniform_level is used instead. Stops blank paper from "
                 "being thresholded into noise."}};
    // Which way a low-contrast window resolves.
    Param<float> m_globalLevel{this, "uniform_level", 128.0f, 0.0f, 255.0f,
        {.help = "Threshold used where local contrast falls below contrast_min. "
                 "Set above the paper brightness so blank areas come out as background."}};
public:
};

REGISTER_ALGORITHM(BernsenThreshold);

// --- adaptive mean ----------------------------------------------------------
// t = local mean - c. The simplest local method, and the right baseline to
// judge whether Niblack/Sauvola are actually earning their extra parameters.
class AdaptiveMeanThreshold : public LocalThresholdBase {
public:
    const char* Name() const override { return "threshold_adaptive_mean"; }

    // Adaptive mean: t = mean - c. c is an intensity offset, so it scales.
    int   GpuMode()   const override { return 2; }
    float GpuParamA() const override { return float(m_c) / 255.0f; }


protected:
    double Threshold(const LocalStats& s) const override {
        return s.mean - double(float(m_c));
    }

private:
    Param<float> m_c{this, "c", 5.0f, -64.0f, 64.0f,
        {.help = "Constant subtracted from the local mean. Higher removes more faint "
                 "background; negative keeps more foreground."}};
};

REGISTER_ALGORITHM(AdaptiveMeanThreshold);

// --- adaptive gaussian ------------------------------------------------------
// As above, but the window is gaussian-weighted, so nearby pixels count for
// more. Less blocky than the box-window version at large window sizes.
class AdaptiveGaussianThreshold : public AlgorithmBase {
public:
    const char* Name()     const override { return "threshold_adaptive_gaussian"; }
    const char* Category() const override { return "threshold"; }

    PortList Inputs()  const override { return {{"src", DataType::Image, FormatSpec::Any}}; }
    PortList Outputs() const override { return {{"mask", DataType::Image, FormatSpec::R32F}}; }

    void RunCPU(RunCtx& ctx) override {
        const ImageView src = ctx.In(0);
        ImageView       dst = ctx.Out(0);
        if (!src.Valid() || !dst.Valid()) return;

        const int radius = RadiusFromWindow(int(m_window));
        const double sigma = std::max(0.1, double(float(m_sigma)));
        const double c = double(float(m_c));
        const bool invert = m_invert;

        // Separable gaussian weights, so the window costs O(2r) not O(r^2).
        std::vector<double> k(size_t(radius) * 2 + 1);
        double sum = 0;
        for (int i = -radius; i <= radius; ++i) {
            const double v = std::exp(-double(i * i) / (2.0 * sigma * sigma));
            k[size_t(i + radius)] = v;
            sum += v;
        }
        for (double& v : k) v /= sum;

        const int w = src.desc.width;
        const int h = src.desc.height;

        // Convert to a flat luma buffer once. SampleValue() carries a format
        // switch and clamping, and the separable passes call it ~2*(2r+1) times
        // per pixel -- at 8 MP that is hundreds of millions of calls, which is
        // where the time went rather than in the maths.
        m_luma.assign(size_t(w) * size_t(h), 0.0f);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                m_luma[size_t(y) * size_t(w) + size_t(x)] =
                    float(SampleValue(src, x, y, -1));

        auto luma = [&](int x, int y) {
            x = std::clamp(x, 0, w - 1);
            y = std::clamp(y, 0, h - 1);
            return double(m_luma[size_t(y) * size_t(w) + size_t(x)]);
        };

        // Horizontal pass into scratch, then vertical while thresholding.
        m_scratch.assign(size_t(w) * size_t(h), 0.0);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                double acc = 0;
                for (int i = -radius; i <= radius; ++i)
                    acc += luma(x + i, y) * k[size_t(i + radius)];
                m_scratch[size_t(y) * size_t(w) + size_t(x)] = acc;
            }

        for (int y = 0; y < h; ++y) {
            float* row = dst.At<float>(0, y);
            for (int x = 0; x < w; ++x) {
                double acc = 0;
                for (int i = -radius; i <= radius; ++i) {
                    const int sy = std::clamp(y + i, 0, h - 1);
                    acc += m_scratch[size_t(sy) * size_t(w) + size_t(x)] * k[size_t(i + radius)];
                }
                const bool above = luma(x, y) > (acc - c);
                row[x] = (above != invert) ? 1.0f : 0.0f;
            }
        }
    }


    // --- GPU implementation -------------------------------------------------
    //
    // Two passes, the same separable split the CPU uses: horizontal into the
    // ping-pong buffer, then vertical while thresholding. The second pass reads
    // the blurred neighbourhood from t0 AND the original pixel from t1, which is
    // why the iterative path binds the source on every pass.
    bool HasGPU() const override { return int(m_window) <= kMaxGpuWindow; }
    int  GpuIterations() const override { return 2; }

    const char* GpuSource() const override {
        return R"(
Texture2D<float4>   Src  : register(t0);   // ping-pong: source, then h-blurred
Texture2D<float4>   Orig : register(t1);   // the untouched input, every pass
RWTexture2D<float4> Dst  : register(u0);

cbuffer Params : register(b0) {
    uint Width;
    uint Height;
    uint Radius;
    uint SigmaBits;
    uint CBits;        // offset, in 0..1 units
    uint Invert;
    uint Pass;         // 0 = horizontal, 1 = vertical + threshold
};

float Luma(float4 c) { return dot(c.rgb, float3(0.299, 0.587, 0.114)); }

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;

    int   r     = int(Radius);
    float sigma = max(0.1, asfloat(SigmaBits));
    int2  hi    = int2(Width - 1, Height - 1);

    // Weights recomputed per pixel rather than shared: at r <= 25 that is at
    // most 51 exp() calls against 51 texture fetches, and it keeps the kernel
    // free of groupshared setup and its barriers.
    float acc = 0.0, wsum = 0.0;
    for (int i = -r; i <= r; ++i) {
        float wgt = exp(-float(i * i) / (2.0 * sigma * sigma));
        int2  p   = (Pass == 0) ? int2(int(tid.x) + i, int(tid.y))
                                : int2(int(tid.x), int(tid.y) + i);
        p = clamp(p, int2(0, 0), hi);
        // Pass 0 reads colour and takes luma; pass 1 reads the pass-0 result,
        // which is already a scalar in .r.
        acc  += ((Pass == 0) ? Luma(Src[p]) : Src[p].r) * wgt;
        wsum += wgt;
    }
    acc /= max(wsum, 1e-6);

    if (Pass == 0) {
        Dst[tid.xy] = float4(acc, acc, acc, 1.0);
        return;
    }

    // The original pixel, which the ping-pong buffer no longer holds.
    float v = Luma(Orig[int2(tid.xy)]);
    bool above = v > (acc - asfloat(CBits));
    if (Invert != 0) above = !above;
    Dst[tid.xy] = above ? 1.0 : 0.0;
}
)";
    }

    std::vector<uint32_t> GpuConstants(int iteration) const override {
        auto bits = [](float f) {
            uint32_t u;
            std::memcpy(&u, &f, sizeof(u));
            return u;
        };
        // `c` is declared in 0..255 but a UNORM SRV hands the shader 0..1, so it
        // is scaled here -- the same units trap as brightness's offset, and the
        // CPU/GPU agreement test is what catches getting it wrong.
        return {uint32_t(RadiusFromWindow(int(m_window))),
                bits(float(m_sigma)),
                bits(float(m_c) / 255.0f),
                uint32_t(bool(m_invert) ? 1 : 0),
                uint32_t(iteration)};
    }
private:
    Param<int>   m_window{this, "window", 15, 3, 201, {.help = "Side length of the neighbourhood each pixel's threshold is computed from, in pixels. Should be larger than the strokes you want to keep but smaller than the illumination changes you want to correct. Odd values only.", .step = 2, .softMin = 3, .softMax = 51}};
    Param<float> m_sigma {this, "sigma", 3.0f, 0.1f, 32.0f,
        {.help = "Width of the Gaussian weighting within the window, in pixels. "
                 "Lower weights nearby pixels more heavily than distant ones.",
         .step = 0.1, .softMin = 0.1, .softMax = 8.0}};
    Param<float> m_c     {this, "c", 5.0f, -64.0f, 64.0f};
    Param<bool>  m_invert{this, "invert", false};

    std::vector<double> m_scratch;
    std::vector<float>  m_luma;
};

REGISTER_ALGORITHM(AdaptiveGaussianThreshold);

} // namespace tglab
