#include "image_stats.h"

#include <algorithm>
#include <cstring>

#include "../algo_util/histogram.h"

namespace tglab {
namespace {

// Enough pixels that the histogram shape is settled, few enough that copying
// them is imperceptible. 512x512 is ~262k samples.
constexpr int64_t kMaxStatsPixels = 512 * 512;

// Nearest-neighbour decimation. Point sampling rather than averaging is the
// right choice here: averaging would invent intermediate values and smooth away
// exactly the clipping spikes the panel exists to show.
Image Subsample(const Image& src, int64_t maxPixels) {
    ImageView v = const_cast<Image&>(src).MapCpuRead();
    if (!v.Valid()) return {};

    const int64_t total = int64_t(v.desc.width) * int64_t(v.desc.height);
    if (total <= maxPixels) return const_cast<Image&>(src).Clone();

    // One step for both axes keeps the sampling grid square, so a wide image is
    // not sampled more densely in one direction than the other.
    int step = 1;
    while ((int64_t(v.desc.width / (step + 1)) * int64_t(v.desc.height / (step + 1))) > maxPixels)
        ++step;

    const int dw = std::max(1, v.desc.width / step);
    const int dh = std::max(1, v.desc.height / step);

    Image out;
    out.Alloc({dw, dh, v.desc.format});
    ImageView o = out.MapCpuWrite();

    const size_t px = size_t(BytesPerPixel(v.desc.format));
    for (int y = 0; y < dh; ++y)
        for (int x = 0; x < dw; ++x)
            std::memcpy(o.At<uint8_t>(x, y), v.At<uint8_t>(x * step, y * step), px);

    return out;
}

} // namespace

void ImageStats::Start() {
    if (m_thread.joinable()) return;
    m_quit.store(false);
    m_thread = std::thread([this] { Run(); });
}

void ImageStats::Stop() {
    if (!m_thread.joinable()) return;
    {
        std::lock_guard<std::mutex> lock(m_mtx);
        m_quit.store(true);
    }
    m_cv.notify_all();
    m_thread.join();
}

void ImageStats::Request(const Image& img, std::string source, uint64_t version) {
    auto req = std::make_unique<Job>();
    // A subsampled copy, not the full image. A 256-bin histogram of several
    // hundred thousand evenly-spaced pixels is indistinguishable from one of
    // eight million -- the shape is a distribution -- so this keeps the copy
    // small and lets the result land in a few milliseconds instead of tens.
    // The copy itself is cheap either way; it is the four histogram passes on
    // the worker that this actually shortens.
    req->image   = Subsample(img, kMaxStatsPixels);
    req->source  = std::move(source);
    req->version = version;

    {
        std::lock_guard<std::mutex> lock(m_mtx);
        m_pending = std::move(req);   // drops any request not yet started
        m_busy.store(true, std::memory_order_relaxed);
    }
    m_cv.notify_one();
}

bool ImageStats::TryFetch(StatsResult* out) {
    std::lock_guard<std::mutex> lock(m_mtx);
    if (!m_result) return false;
    *out = std::move(*m_result);
    m_result.reset();
    return true;
}

void ImageStats::Run() {
    for (;;) {
        std::unique_ptr<Job> req;
        {
            std::unique_lock<std::mutex> lock(m_mtx);
            m_cv.wait(lock, [this] { return m_quit.load() || m_pending != nullptr; });
            if (m_quit.load()) return;
            req = std::move(m_pending);
        }

        auto out = std::make_unique<StatsResult>();
        out->source  = req->source;
        out->version = req->version;

        ImageView v = req->image.MapCpuRead();
        if (v.Valid()) {
            Histogram h;

            auto fill = [&](int channel, std::vector<float>& dst) {
                h.Build(v, channel);
                dst.assign(Histogram::kBins, 0.0f);
                for (int i = 0; i < Histogram::kBins; ++i)
                    dst[size_t(i)] = float(h.Bin(i));
            };

            if (v.desc.format != Format::R32F) {
                fill(0, out->r);
                fill(1, out->g);
                fill(2, out->b);
            }

            // Luma last, so `h` still holds it for the statistics below and
            // the image is not walked a fifth time.
            fill(-1, out->luma);
            out->mean   = h.Mean();
            out->median = h.Median();
            out->stddev = h.StdDev();

            const double n = double(h.Count() ? h.Count() : 1);
            out->clipLow  = double(h.Bin(0)) / n;
            out->clipHigh = double(h.Bin(Histogram::kBins - 1)) / n;

            // Scaled to the tallest *interior* bin: a large flat background
            // spikes one end, and scaling to that would flatten everything
            // else to invisibility.
            out->peak = 1.0f;
            auto peak = [&](const std::vector<float>& c) {
                if (c.empty()) return;
                for (int i = 1; i < Histogram::kBins - 1; ++i)
                    out->peak = std::max(out->peak, c[size_t(i)]);
            };
            peak(out->r); peak(out->g); peak(out->b);
            if (out->r.empty()) peak(out->luma);

            out->valid = true;
        }

        {
            std::lock_guard<std::mutex> lock(m_mtx);
            m_result = std::move(out);
            if (!m_pending) m_busy.store(false, std::memory_order_relaxed);
        }
    }
}

} // namespace tglab
