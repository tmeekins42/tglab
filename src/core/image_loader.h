// ImageLoader — decodes image files off the UI thread.
//
// stbi_load() reads the file directly, so a large JPEG (or any file on a slow
// or network drive) takes hundreds of milliseconds to many seconds. Doing that
// inside WM_DROPFILES blocks the message loop, which makes the entire window
// unresponsive -- no panels, no menus, nothing -- for the whole read. That
// looks like a hang even though the pipeline worker is idle.
#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "image.h"

namespace tglab {

struct LoadRequest {
    std::string path;
    std::string targetSlot;   // "" means add a new entry
};

struct LoadResult {
    std::string path;
    std::string targetSlot;
    Image       image;
    bool        ok = false;
    std::string error;
};

class ImageLoader {
public:
    ~ImageLoader() { Stop(); }

    void Start();
    void Stop();

    // Queues a file to decode. Unlike the pipeline worker these are NOT
    // coalesced: dropping several files means loading all of them.
    void Request(std::string path, std::string targetSlot);

    // Non-blocking; returns one finished load, oldest first.
    bool TryFetch(LoadResult* out);

    // Files queued or being read, for a "loading..." indicator.
    int Pending() const { return m_pending.load(std::memory_order_relaxed); }

private:
    void Run();

    std::thread             m_thread;
    std::mutex              m_mtx;
    std::condition_variable m_cv;
    std::vector<LoadRequest> m_queue;
    std::vector<LoadResult>  m_done;
    std::atomic<bool>       m_quit{false};
    std::atomic<int>        m_pending{0};
};

} // namespace tglab
