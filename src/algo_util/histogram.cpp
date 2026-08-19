#include "histogram.h"

#include <algorithm>
#include <cmath>

namespace tglab {

double SampleValue(const ImageView& v, int x, int y, int channel) {
    x = std::clamp(x, 0, v.desc.width  - 1);   // clamp-to-edge
    y = std::clamp(y, 0, v.desc.height - 1);

    switch (v.desc.format) {
        case Format::R32F:
            return double(*v.At<float>(x, y));
        case Format::RGBA32F: {
            const float* p = v.At<float>(x, y);
            if (channel >= 0) return double(p[channel]);
            return 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
        }
        case Format::RGBA8: {
            const uint8_t* p = v.At<uint8_t>(x, y);
            if (channel >= 0) return double(p[channel]);
            return 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
        }
        default:
            return 0.0;
    }
}

// --- Histogram --------------------------------------------------------------

void Histogram::Clear() {
    std::fill(std::begin(m_bins), std::end(m_bins), 0u);
    m_count = 0;
    m_min = 0.0;
    m_max = 255.0;
}

void Histogram::SetRangeFor(const ImageView& v) {
    if (v.desc.format == Format::RGBA8) {
        m_min = 0.0;
        m_max = 255.0;
        return;
    }
    // Float formats have no natural range, so bin over what is actually there.
    // Without this, a gradient image in [-2, 2] would collapse into one bin.
    double lo = 1e300, hi = -1e300;
    for (int y = 0; y < v.desc.height; ++y)
        for (int x = 0; x < v.desc.width; ++x) {
            const double s = SampleValue(v, x, y, -1);
            lo = std::min(lo, s);
            hi = std::max(hi, s);
        }
    if (lo > hi) { lo = 0.0; hi = 1.0; }
    if (hi - lo < 1e-12) hi = lo + 1.0;
    m_min = lo;
    m_max = hi;
}

int Histogram::ValueToBin(double value) const {
    const double t = (value - m_min) / (m_max - m_min);
    return std::clamp(int(t * (kBins - 1) + 0.5), 0, kBins - 1);
}

double Histogram::BinToValue(double bin) const {
    return m_min + (bin / double(kBins - 1)) * (m_max - m_min);
}

void Histogram::Accumulate(const ImageView& v, int x, int y, int channel) {
    ++m_bins[ValueToBin(SampleValue(v, x, y, channel))];
    ++m_count;
}

void Histogram::Build(const ImageView& v, int channel) {
    Clear();
    if (!v.Valid()) return;
    SetRangeFor(v);
    for (int y = 0; y < v.desc.height; ++y)
        for (int x = 0; x < v.desc.width; ++x)
            Accumulate(v, x, y, channel);
}

void Histogram::BuildWindow(const ImageView& v, int cx, int cy, int radius, int channel) {
    Clear();
    if (!v.Valid()) return;
    SetRangeFor(v);
    for (int y = cy - radius; y <= cy + radius; ++y)
        for (int x = cx - radius; x <= cx + radius; ++x)
            Accumulate(v, x, y, channel);   // SampleValue clamps to edge
}

double Histogram::Mean() const {
    if (m_count == 0) return 0.0;
    double sum = 0;
    for (int i = 0; i < kBins; ++i) sum += double(m_bins[i]) * BinToValue(i);
    return sum / double(m_count);
}

double Histogram::Variance() const {
    if (m_count == 0) return 0.0;
    const double mu = Mean();
    double acc = 0;
    for (int i = 0; i < kBins; ++i) {
        const double d = BinToValue(i) - mu;
        acc += double(m_bins[i]) * d * d;
    }
    return acc / double(m_count);
}

double Histogram::StdDev() const { return std::sqrt(Variance()); }

double Histogram::Percentile(double fraction) const {
    if (m_count == 0) return 0.0;
    const double target = std::clamp(fraction, 0.0, 1.0) * double(m_count);
    uint64_t running = 0;
    for (int i = 0; i < kBins; ++i) {
        running += m_bins[i];
        if (double(running) >= target) return BinToValue(i);
    }
    return BinToValue(kBins - 1);
}

double Histogram::MinValue() const {
    for (int i = 0; i < kBins; ++i) if (m_bins[i]) return BinToValue(i);
    return 0.0;
}

double Histogram::MaxValue() const {
    for (int i = kBins - 1; i >= 0; --i) if (m_bins[i]) return BinToValue(i);
    return 0.0;
}

int Histogram::PeakBin() const {
    int best = 0;
    uint32_t bestN = 0;
    for (int i = 0; i < kBins; ++i)
        if (m_bins[i] > bestN) { bestN = m_bins[i]; best = i; }
    return best;
}

double Histogram::OtsuThreshold() const {
    if (m_count == 0) return BinToValue(kBins / 2);

    // Single pass: maximise between-class variance, which is equivalent to
    // minimising within-class variance but far cheaper to evaluate.
    double total = 0;
    for (int i = 0; i < kBins; ++i) total += double(i) * double(m_bins[i]);

    double sumB = 0, wB = 0, best = -1;
    // On a cleanly bimodal image every threshold between the two modes scores
    // identically, so track the whole plateau and return its midpoint. Taking
    // the first (what a naive > comparison gives) lands the threshold right on
    // the lower mode, which then depends on > vs >= to classify correctly.
    int firstBest = kBins / 2, lastBest = kBins / 2;
    const double n = double(m_count);

    for (int i = 0; i < kBins; ++i) {
        wB += double(m_bins[i]);
        if (wB == 0) continue;
        const double wF = n - wB;
        if (wF == 0) break;

        sumB += double(i) * double(m_bins[i]);
        const double mB = sumB / wB;
        const double mF = (total - sumB) / wF;
        const double between = wB * wF * (mB - mF) * (mB - mF);

        // Relative epsilon: between-class variance runs to ~1e11 here, where an
        // absolute 1e-9 is far below float resolution and every plateau bin
        // would look "greater", collapsing the midpoint back onto the first.
        const double eps = std::max(1e-9, std::fabs(best) * 1e-12);
        if (between > best + eps) {
            best = between;
            firstBest = lastBest = i;
        } else if (between > best - eps) {
            lastBest = i;   // still on the plateau
        }
    }
    return BinToValue(0.5 * (double(firstBest) + double(lastBest)));
}

double Histogram::TriangleThreshold() const {
    if (m_count == 0) return BinToValue(kBins / 2);

    const int peak = PeakBin();

    int first = 0, last = kBins - 1;
    while (first < kBins && m_bins[first] == 0) ++first;
    while (last > 0 && m_bins[last] == 0) --last;

    // Work on the longer tail: the method assumes one dominant peak with a
    // long slope running away from it.
    const bool rightTailLonger = (peak - first) < (last - peak);
    int lo, hi;
    if (rightTailLonger) { lo = peak; hi = last; }
    else                 { lo = first; hi = peak; }
    if (hi <= lo) return BinToValue(peak);

    // Distance from each bin top to the chord joining the two endpoints. The
    // maximum is the corner of the shoulder, which is the threshold.
    const double x0 = double(lo), y0 = double(m_bins[lo]);
    const double x1 = double(hi), y1 = double(m_bins[hi]);
    const double dx = x1 - x0, dy = y1 - y0;
    const double len = std::sqrt(dx * dx + dy * dy);
    if (len < 1e-12) return BinToValue(peak);

    double bestD = -1;
    int bestBin = lo;
    for (int i = lo; i <= hi; ++i) {
        const double d = std::fabs(dy * (double(i) - x0) - dx * (double(m_bins[i]) - y0)) / len;
        if (d > bestD) { bestD = d; bestBin = i; }
    }
    return BinToValue(bestBin);
}

double Histogram::IsoDataThreshold() const {
    if (m_count == 0) return BinToValue(kBins / 2);

    // Start at the midpoint and iterate until the threshold sits halfway
    // between the two class means.
    int t = kBins / 2;
    for (int iter = 0; iter < 128; ++iter) {
        double sumLo = 0, nLo = 0, sumHi = 0, nHi = 0;
        for (int i = 0; i < kBins; ++i) {
            const double c = double(m_bins[i]);
            if (i <= t) { sumLo += double(i) * c; nLo += c; }
            else        { sumHi += double(i) * c; nHi += c; }
        }
        if (nLo == 0 || nHi == 0) break;
        const int next = int((sumLo / nLo + sumHi / nHi) * 0.5 + 0.5);
        if (next == t) break;
        t = std::clamp(next, 0, kBins - 1);
    }
    return BinToValue(t);
}

// --- window statistics ------------------------------------------------------

LocalStats WindowStats(const ImageView& v, int cx, int cy, int radius, int channel) {
    LocalStats s;
    if (!v.Valid()) return s;

    double sum = 0, sumSq = 0;
    double lo = 1e300, hi = -1e300;
    int n = 0;

    for (int y = cy - radius; y <= cy + radius; ++y) {
        for (int x = cx - radius; x <= cx + radius; ++x) {
            const double t = SampleValue(v, x, y, channel);   // clamps to edge
            sum   += t;
            sumSq += t * t;
            lo = std::min(lo, t);
            hi = std::max(hi, t);
            ++n;
        }
    }
    if (n == 0) return s;

    s.mean = sum / double(n);
    s.stddev = std::sqrt(std::max(0.0, sumSq / double(n) - s.mean * s.mean));
    s.minV = lo;
    s.maxV = hi;
    return s;
}

// --- integral image ---------------------------------------------------------

void IntegralImage::Build(const ImageView& v, int channel) {
    if (!v.Valid()) { m_w = m_h = 0; return; }
    m_w = v.desc.width;
    m_h = v.desc.height;

    // (w+1) x (h+1) with a zero row and column, so the four-corner lookup
    // needs no bounds special-casing.
    m_sum.assign(size_t(m_w + 1) * size_t(m_h + 1), 0.0);
    m_sumSq.assign(size_t(m_w + 1) * size_t(m_h + 1), 0.0);

    for (int y = 0; y < m_h; ++y) {
        double rowSum = 0, rowSumSq = 0;
        for (int x = 0; x < m_w; ++x) {
            const double t = SampleValue(v, x, y, channel);
            rowSum   += t;
            rowSumSq += t * t;
            const size_t i  = size_t(y + 1) * size_t(m_w + 1) + size_t(x + 1);
            const size_t up = size_t(y)     * size_t(m_w + 1) + size_t(x + 1);
            m_sum[i]   = m_sum[up]   + rowSum;
            m_sumSq[i] = m_sumSq[up] + rowSumSq;
        }
    }
}

LocalStats IntegralImage::Window(int cx, int cy, int radius) const {
    LocalStats s;
    if (!Valid()) return s;

    const int x0 = std::clamp(cx - radius, 0, m_w - 1);
    const int y0 = std::clamp(cy - radius, 0, m_h - 1);
    const int x1 = std::clamp(cx + radius, 0, m_w - 1);
    const int y1 = std::clamp(cy + radius, 0, m_h - 1);

    const int stride = m_w + 1;
    auto at = [&](const std::vector<double>& t, int x, int y) {
        return t[size_t(y) * size_t(stride) + size_t(x)];
    };

    const double sum = at(m_sum, x1 + 1, y1 + 1) - at(m_sum, x0, y1 + 1)
                     - at(m_sum, x1 + 1, y0)     + at(m_sum, x0, y0);
    const double sumSq = at(m_sumSq, x1 + 1, y1 + 1) - at(m_sumSq, x0, y1 + 1)
                       - at(m_sumSq, x1 + 1, y0)     + at(m_sumSq, x0, y0);

    const double n = double(x1 - x0 + 1) * double(y1 - y0 + 1);
    if (n <= 0) return s;

    s.mean   = sum / n;
    s.stddev = std::sqrt(std::max(0.0, sumSq / n - s.mean * s.mean));
    // An integral image cannot give min/max; the one thresholder that needs
    // them (Bernsen) uses WindowStats instead.
    s.minV = s.maxV = s.mean;
    return s;
}

} // namespace tglab
