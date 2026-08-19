// Local (adaptive) thresholding: a different threshold per pixel, computed
// from a window around it. These handle uneven lighting, where any single
// global threshold necessarily fails somewhere in the image.
//
// Niblack, Sauvola and the adaptive-mean pair all need windowed mean/stddev,
// so they share an integral image — O(1) per pixel instead of O(r^2), which is
// the difference between usable and unusable at large window sizes. Bernsen
// needs window min/max, which an integral image cannot provide, so it uses the
// direct path.
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
            for (int y = 0; y < h; ++y) {
                float* row = dst.At<float>(0, y);
                for (int x = 0; x < w; ++x) {
                    const LocalStats s = WindowStats(src, x, y, radius);
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

    Param<int>  m_window{this, "window", 15, 3, 201};
    Param<bool> m_invert{this, "invert", false};

private:
    IntegralImage m_integral;
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
    Param<float> m_k{this, "k", -0.2f, -1.0f, 1.0f};
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
    Param<float> m_k{this, "k", 0.2f, 0.0f, 1.0f};
    // R is the expected dynamic range of the stddev; 128 is the standard
    // choice for 8-bit input.
    Param<float> m_r{this, "r", 128.0f, 1.0f, 255.0f};
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
    Param<float> m_contrastMin{this, "contrast_min", 15.0f, 0.0f, 128.0f};
    // Which way a low-contrast window resolves.
    Param<float> m_globalLevel{this, "uniform_level", 128.0f, 0.0f, 255.0f};
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
    Param<float> m_c{this, "c", 5.0f, -64.0f, 64.0f};
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

        // Horizontal pass into scratch, then vertical while thresholding.
        m_scratch.assign(size_t(w) * size_t(h), 0.0);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                double acc = 0;
                for (int i = -radius; i <= radius; ++i)
                    acc += SampleValue(src, x + i, y, -1) * k[size_t(i + radius)];
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
                const bool above = SampleValue(src, x, y, -1) > (acc - c);
                row[x] = (above != invert) ? 1.0f : 0.0f;
            }
        }
    }

private:
    Param<int>   m_window{this, "window", 15, 3, 201};
    Param<float> m_sigma {this, "sigma", 3.0f, 0.1f, 32.0f};
    Param<float> m_c     {this, "c", 5.0f, -64.0f, 64.0f};
    Param<bool>  m_invert{this, "invert", false};

    std::vector<double> m_scratch;
};

REGISTER_ALGORITHM(AdaptiveGaussianThreshold);

} // namespace tglab
