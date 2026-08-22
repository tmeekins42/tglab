// ImageStats — histogram and summary statistics, computed off the UI thread.
//
// The info panel is purely informational, so it must never cost frame time.
// Four 256-bin passes over an 8 MP image is tens of milliseconds; doing that
// during a frame -- worse, every frame -- turns a panel nobody is reading into
// the slowest thing in the app.
//
// Same shape as ImageLoader: request work, poll for a result, apply it on the
// UI thread. Requests coalesce newest-wins, since a stale histogram of an image
// the user has already moved past is worthless.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "image.h"

namespace tglab {

struct StatsResult {
    // Identifies what these bins describe, so a result that arrives after the
    // user has switched images can be recognised as stale.
    std::string source;
    uint64_t    version = 0;

    // 256 bins per channel. The RGB vectors are empty for single-channel
    // images, where three identical curves would be misleading rather than
    // informative.
    std::vector<float> r, g, b, luma;

    float  peak   = 1.0f;    // tallest interior bin, for scaling the plot
    double mean   = 0.0;
    double median = 0.0;
    double stddev = 0.0;

    // The units mean/median/stddev are in: 255 for an 8-bit image, 1 for a
    // float one. Without this the panel formatted a linear float image's 0..1
    // values with one decimal place and showed "mean 0.0 median 0.0 stddev
    // 0.0" for a perfectly good photograph.
    double scale = 255.0;

    // True when the image carries values above 1.0 -- real highlight headroom
    // that an 8-bit conversion would have clipped. Worth saying explicitly,
    // since it is the whole reason for the raw path.
    bool   hasHeadroom = false;
    double maxValue    = 0.0;
    double clipLow  = 0.0;   // fraction of pixels at pure black
    double clipHigh = 0.0;   // ... and at pure white

    bool valid = false;
};

class ImageStats {
public:
    void Start();
    void Stop();

    // Queues a computation, replacing any request not yet started. Takes its
    // own copy: the UI thread may swap the image out from under us the moment
    // this returns.
    void Request(const Image& img, std::string source, uint64_t version);

    // Non-blocking. True when a newer result is ready.
    bool TryFetch(StatsResult* out);

    bool Busy() const { return m_busy.load(std::memory_order_relaxed); }

private:
    void Run();

    std::thread             m_thread;
    std::mutex              m_mtx;
    std::condition_variable m_cv;

    struct Job {
        Image       image;
        std::string source;
        uint64_t    version = 0;
    };
    std::unique_ptr<Job>         m_pending;   // newest wins
    std::unique_ptr<StatsResult> m_result;

    std::atomic<bool> m_quit{false};
    std::atomic<bool> m_busy{false};
};

} // namespace tglab
