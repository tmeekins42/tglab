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

    void Clear() {
        m_done.store(0, std::memory_order_relaxed);
        m_total.store(0, std::memory_order_relaxed);
        m_label[0] = '\0';
    }

    int  Done()  const { return m_done.load(std::memory_order_relaxed); }
    int  Total() const { return m_total.load(std::memory_order_relaxed); }

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
    std::atomic<unsigned> m_labelVersion{0};
    char                  m_label[kLabel] = {};
};

} // namespace tglab
