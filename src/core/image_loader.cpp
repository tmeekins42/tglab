#include "image_loader.h"

#include "image_io.h"

namespace tglab {

void ImageLoader::Start() {
    if (m_thread.joinable()) return;
    m_quit.store(false);
    m_thread = std::thread([this] { Run(); });
}

void ImageLoader::Stop() {
    if (!m_thread.joinable()) return;
    {
        std::lock_guard<std::mutex> lock(m_mtx);
        m_quit.store(true);
    }
    m_cv.notify_all();
    m_thread.join();
}

void ImageLoader::Request(std::string path, std::string targetSlot) {
    {
        std::lock_guard<std::mutex> lock(m_mtx);
        m_queue.push_back({std::move(path), std::move(targetSlot)});
        m_pending.fetch_add(1, std::memory_order_relaxed);
    }
    m_cv.notify_one();
}

bool ImageLoader::TryFetch(LoadResult* out) {
    std::lock_guard<std::mutex> lock(m_mtx);
    if (m_done.empty()) return false;
    *out = std::move(m_done.front());
    m_done.erase(m_done.begin());
    return true;
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
        res.ok         = LoadImageFile(req.path, &res.image, &res.error);

        {
            std::lock_guard<std::mutex> lock(m_mtx);
            m_done.push_back(std::move(res));
            m_pending.fetch_sub(1, std::memory_order_relaxed);
        }
    }
}

} // namespace tglab
