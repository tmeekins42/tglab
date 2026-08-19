// Histogram — shared statistics for algorithms.
//
// Lives in algo_util/ rather than being an algorithm itself: it is a building
// block many algorithms need (Otsu, Triangle, IsoData all just analyse one),
// not something you would put in a pipeline. Anything here is fair game for
// any algorithm to use, and nothing here self-registers.
#pragma once

#include <cstdint>
#include <vector>

#include "../core/image.h"

namespace tglab {

// Intensity histogram over a single channel, or over luma for colour images.
// Fixed 256 bins for 8-bit input; float formats are binned over their own
// min/max range, which is what makes methods like Otsu work on gradients.
class Histogram {
public:
    static constexpr int kBins = 256;

    void Clear();

    // Builds from the whole image. `channel` < 0 means luma (Rec.601).
    void Build(const ImageView& v, int channel = -1);

    // Builds from a rectangular window, clamped to the image bounds. Used by
    // the local thresholding methods.
    void BuildWindow(const ImageView& v, int cx, int cy, int radius, int channel = -1);

    // --- raw access ---------------------------------------------------------
    const uint32_t* Bins() const { return m_bins; }
    uint32_t Bin(int i) const { return m_bins[i]; }
    uint64_t Count() const { return m_count; }

    // Value range the bins span. For RGBA8 this is [0, 255].
    double RangeMin() const { return m_min; }
    double RangeMax() const { return m_max; }

    // Converts a bin index back to a value in the source's units.
    double BinToValue(double bin) const;
    int    ValueToBin(double value) const;

    // --- statistics ---------------------------------------------------------
    double Mean() const;
    double Variance() const;
    double StdDev() const;
    double Median() const { return Percentile(0.5); }
    double Percentile(double fraction) const;   // 0..1
    double MinValue() const;                    // first non-empty bin
    double MaxValue() const;                    // last non-empty bin
    int    PeakBin() const;                     // most populated bin

    // --- threshold selection ------------------------------------------------
    // These return a threshold in the source's units, not a bin index.

    // Otsu: the split maximising between-class variance.
    double OtsuThreshold() const;

    // Triangle: furthest point from the line joining the peak to the far end
    // of the histogram. Better than Otsu on skewed, single-peaked histograms.
    double TriangleThreshold() const;

    // IsoData / intermeans: iterate until the threshold is the midpoint of the
    // two class means. The classic alternative to Otsu.
    double IsoDataThreshold() const;

private:
    void Accumulate(const ImageView& v, int x, int y, int channel);
    void SetRangeFor(const ImageView& v);

    uint32_t m_bins[kBins] = {};
    uint64_t m_count = 0;
    double   m_min = 0.0;
    double   m_max = 255.0;
};

// Reads one channel as a double; channel < 0 gives Rec.601 luma.
double SampleValue(const ImageView& v, int x, int y, int channel);

// Local mean and standard deviation over a square window. Computed directly
// rather than via a Histogram because the local thresholders need exact
// moments, not binned approximations.
struct LocalStats {
    double mean = 0;
    double stddev = 0;
    double minV = 0;
    double maxV = 0;
};
LocalStats WindowStats(const ImageView& v, int cx, int cy, int radius, int channel = -1);

// Integral images (summed-area tables) make windowed mean/stddev O(1) per
// pixel instead of O(r^2) — the difference between a usable and an unusable
// local thresholder at large window sizes.
class IntegralImage {
public:
    void Build(const ImageView& v, int channel = -1);

    // Mean and stddev over the window centred on (cx, cy), clamped to bounds.
    LocalStats Window(int cx, int cy, int radius) const;

    bool Valid() const { return m_w > 0 && m_h > 0; }

private:
    int m_w = 0, m_h = 0;
    std::vector<double> m_sum;    // (w+1) x (h+1)
    std::vector<double> m_sumSq;
};

} // namespace tglab
