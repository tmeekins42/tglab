// Progress: what a long run is doing, published for the UI to read.
//
// Written by the worker thread, read by the UI thread, with no lock -- the same
// shape as CancelToken and for the same reason. The UI reads it once a frame
// and only to draw a label, so a value one frame stale is invisible; a mutex
// here would put the UI thread behind whatever the worker is doing, which is
// the opposite of the point.
//
// The label is a fixed buffer rather than a std::string: a string assigned on
// one thread and read on another is a data race even when the characters
// happen to survive it. Copying at most 63 bytes is cheap enough to do per
// stage and impossible to get wrong.
#pragma once

#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>

namespace tglab {

class Progress {
public:
    // `done` of `total` units finished. A total of 0 means "no useful count" --
    // a single-image run, where a bar would be a lie.
    void Set(int done, int total, const char* what) {
        m_done.store(done, std::memory_order_relaxed);
        m_total.store(total, std::memory_order_relaxed);
        if (what) {
            char buf[kLabel];
            std::snprintf(buf, kLabel, "%s", what);
            std::memcpy(m_label, buf, kLabel);
            m_labelVersion.fetch_add(1, std::memory_order_release);
        }
    }

    // Running tallies, published as each stage finishes rather than at the end
    // of the run.
    //
    // The point is the slow case. A pipeline that takes ten seconds used to
    // show nothing about where the time went until it was over, which is
    // exactly when the information stops being useful -- if one stage is
    // dominating, you want to know WHILE it is happening, and if a stage
    // silently fell back to the CPU you want to see the GPU count stop moving.
    //
    // Same no-lock contract as the label: relaxed atomics the UI reads once a
    // frame to draw text. A tally one frame stale is invisible.
    virtual ~Progress() = default;

    // Virtual only so a test can record the sequence of publishes and prove
    // they happen per stage rather than once at the end. The atomic reads below
    // are unaffected -- they are still plain loads, and this is called a handful
    // of times per run, not per pixel.
    virtual void SetStats(int cpuStages, int gpuStages, double elapsedMs,
                          double gpuMs = 0.0) {
        m_cpuStages.store(cpuStages, std::memory_order_relaxed);
        m_gpuStages.store(gpuStages, std::memory_order_relaxed);
        m_elapsedMs.store(elapsedMs, std::memory_order_relaxed);
        m_gpuMs.store(gpuMs, std::memory_order_relaxed);
    }

    void Clear() {
        m_done.store(0, std::memory_order_relaxed);
        m_total.store(0, std::memory_order_relaxed);
        m_cpuStages.store(0, std::memory_order_relaxed);
        m_gpuStages.store(0, std::memory_order_relaxed);
        m_elapsedMs.store(0.0, std::memory_order_relaxed);
        m_label[0] = '\0';
    }

    int  Done()  const { return m_done.load(std::memory_order_relaxed); }
    int  Total() const { return m_total.load(std::memory_order_relaxed); }

    int    CpuStages() const { return m_cpuStages.load(std::memory_order_relaxed); }
    int    GpuStages() const { return m_gpuStages.load(std::memory_order_relaxed); }
    double ElapsedMs() const { return m_elapsedMs.load(std::memory_order_relaxed); }

    // Of the elapsed time, how much was spent submitting GPU work and waiting
    // for it. The remainder is CPU-side: algorithm loops, allocation, and the
    // recording of the GPU commands themselves.
    double GpuMs() const { return m_gpuMs.load(std::memory_order_relaxed); }

    // A copy, because the buffer can change under the caller. Cheap: 64 bytes.
    std::string Label() const {
        char buf[kLabel];
        std::memcpy(buf, m_label, kLabel);
        buf[kLabel - 1] = '\0';
        return std::string(buf);
    }

private:
    static constexpr int kLabel = 64;

    std::atomic<int>      m_done{0};
    std::atomic<int>      m_total{0};
    std::atomic<int>      m_cpuStages{0};
    std::atomic<int>      m_gpuStages{0};
    std::atomic<double>   m_elapsedMs{0.0};
    std::atomic<double>   m_gpuMs{0.0};
    std::atomic<unsigned> m_labelVersion{0};
    char                  m_label[kLabel] = {};
};

} // namespace tglab
