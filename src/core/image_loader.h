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
    uint64_t    seq = 0;      // request order, preserved across the pool
};

struct LoadResult {
    std::string path;
    std::string targetSlot;
    Image       image;
    bool        ok = false;
    std::string error;
    uint64_t    seq = 0;
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

    // Several workers, not one. Dropping a card full of raws decoded them one
    // at a time -- a 45 MP CR3 is seconds, so 20 files was a minute of nothing.
    // Decoding is CPU-bound and independent per file, so it parallelises
    // cleanly; the queue and its mutex already existed.
    std::vector<std::thread> m_threads;
    std::mutex              m_mtx;
    std::condition_variable m_cv;
    std::vector<LoadRequest> m_queue;
    std::vector<LoadResult>  m_done;
    std::atomic<bool>       m_quit{false};
    std::atomic<int>        m_pending{0};

    // Results are delivered in REQUEST order, not completion order.
    //
    // With a pool a small JPEG finishes before a 45 MP CR3 queued ahead of it,
    // and a group would then hold its files in whatever order they happened to
    // decode. File order matters -- a bracket is an ordered thing -- so a
    // finished load waits until every earlier one has been handed over.
    uint64_t                m_nextSeq = 0;      // next request number
    uint64_t                m_deliverSeq = 0;   // next to hand to the UI
};

} // namespace tglab
