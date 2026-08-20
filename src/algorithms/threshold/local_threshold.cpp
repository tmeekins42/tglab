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

private:
    Param<float> m_contrastMin{this, "contrast_min", 15.0f, 0.0f, 128.0f,
        {.help = "Below this local contrast (max minus min) the window is treated as "
                 "uniform and uniform_level is used instead. Stops blank paper from "
                 "being thresholded into noise."}};
    // Which way a low-contrast window resolves.
    Param<float> m_globalLevel{this, "uniform_level", 128.0f, 0.0f, 255.0f,
        {.help = "Threshold used where local contrast falls below contrast_min. "
                 "Set above the paper brightness so blank areas come out as background."}};
};

REGISTER_ALGORITHM(BernsenThreshold);

// --- adaptive mean ----------------------------------------------------------
// t = local mean - c. The simplest local method, and the right baseline to
// judge whether Niblack/Sauvola are actually earning their extra parameters.
class AdaptiveMeanThreshold : public LocalThresholdBase {
public:
    const char* Name() const override { return "threshold_adaptive_mean"; }

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
