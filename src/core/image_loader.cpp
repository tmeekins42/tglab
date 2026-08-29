#include "image_loader.h"

#include <algorithm>

#include "image_io.h"

namespace tglab {

void ImageLoader::Start() {
    if (!m_threads.empty()) return;
    m_quit.store(false);

    // One worker per core, less a couple left for the UI and the pipeline
    // worker -- both of which matter more than finishing a drop a moment
    // sooner. Clamped to at least two, so a small machine still overlaps
    // decoding with the read that follows it.
    unsigned n = std::thread::hardware_concurrency();
    n = (n > 3) ? n - 2 : 2;
    n = std::min(n, 8u);   // beyond this the disk, not the CPU, is the limit

    m_threads.reserve(n);
    for (unsigned i = 0; i < n; ++i) m_threads.emplace_back([this] { Run(); });
}

void ImageLoader::Stop() {
    if (m_threads.empty()) return;
    {
        std::lock_guard<std::mutex> lock(m_mtx);
        m_quit.store(true);
    }
    m_cv.notify_all();
    for (std::thread& t : m_threads) if (t.joinable()) t.join();
    m_threads.clear();
}

void ImageLoader::Request(std::string path, std::string targetSlot) {
    {
        std::lock_guard<std::mutex> lock(m_mtx);
        m_queue.push_back({std::move(path), std::move(targetSlot), m_nextSeq++});
        m_pending.fetch_add(1, std::memory_order_relaxed);
    }
    m_cv.notify_one();
}

bool ImageLoader::TryFetch(LoadResult* out) {
    std::lock_guard<std::mutex> lock(m_mtx);

    // Hand over the next result IN REQUEST ORDER, even if a later one finished
    // first. Holding a finished load back briefly costs nothing; delivering a
    // group's files out of order would silently reorder a bracket.
    for (size_t i = 0; i < m_done.size(); ++i) {
        if (m_done[i].seq != m_deliverSeq) continue;
        *out = std::move(m_done[i]);
        m_done.erase(m_done.begin() + i);
        ++m_deliverSeq;
        return true;
    }
    return false;
}

void ImageLoader::Run() {
    for (;;) {
        LoadRequest req;
        {
            std::unique_lock<std::mutex> lock(m_mtx);
            m_cv.wait(lock, [this] { return m_quit.load() || !m_queue.empty(); });
            if (m_quit.load()) return;
            req = std::move(m_queue.front());
            m_queue.erase(m_queue.begin());
        }

        LoadResult res;
        res.path       = req.path;
        res.targetSlot = req.targetSlot;
        res.seq        = req.seq;
        res.ok         = LoadImageFile(req.path, &res.image, &res.error);

        // EXIF on the worker too, for the same reason the pixels are here. Read
        // even when the decode failed: a file can be an unsupported format and
        // still carry readable metadata, and reporting the camera settings of
        // something that would not load is more useful than reporting nothing.
        res.exif = ReadExif(req.path);

        {
            std::lock_guard<std::mutex> lock(m_mtx);
            m_done.push_back(std::move(res));
            m_pending.fetch_sub(1, std::memory_order_relaxed);
        }
    }
}

} // namespace tglab
