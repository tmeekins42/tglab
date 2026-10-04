// ParallelFor — run a loop body across a few threads and wait for it.
//
// WHY NOT ImageLoader'S POOL, which already exists. That one is a request
// queue: work is pushed in, results are fetched later, and the caller never
// waits. This is the opposite shape -- the caller has N independent items and
// cannot continue until all of them are done -- and bolting a completion wait
// onto a queue designed not to have one would make both harder to read.
//
// It DOES follow that pool's sizing policy exactly rather than inventing a
// second one: see Threads() below.
//
// THE BODY MUST BE INDEPENDENT ACROSS INDICES. Nothing here provides mutual
// exclusion, because the case this was written for -- mapping a stage across
// the frames of a group -- genuinely has none: each frame reads its own input
// and writes its own output slot. An algorithm that shares state between
// frames is not made safe by running it through this.
#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace tglab {

// How many threads to use for `n` items of work.
//
// One per core less two, which are left for the UI and for whatever else is
// running -- a preview that stutters while a background stage saturates every
// core is a worse trade than finishing that stage a moment later. Clamped to
// at least one, and never more than the work available.
//
// NO CAP OF EIGHT, which this used to copy from ImageLoader::Start. There
// it is right -- past eight decoders the disk, not the CPU, is the limit --
// and here it was not: on a 16-core machine every solver ran on half the
// cores. Measured with the threads kept (see WorkerPool), solve_cameras on
// a 179-frame video took 26.7 s at 8 threads and 21.9 s at 30.
//
// What eight still bounds is MEMORY: how many frames a stage holds at once.
// See kMaxFramesInFlight.
inline unsigned ParallelThreads(size_t n) {
    if (n <= 1) return 1;
    unsigned c = std::thread::hardware_concurrency();
    c = (c > 3) ? c - 2 : 2;
    return std::min(unsigned(n), std::max(1u, c));
}

// The most threads a ParallelFor creates for itself when the kept ones are
// busy -- nested inside another's body, or called while another runs. That
// work is already sharing the machine with the outer loop, so it keeps the
// old cap rather than adding a full machine's worth of threads per caller.
constexpr unsigned kMaxSpawned = 8;

// The most frames of a group a stage works on at once. Each holds its own
// buffers -- a raw frame's working set is hundreds of megabytes -- so this is
// the limit that keeps a broadcast over sixty raws inside memory, whatever
// the core count.
constexpr unsigned kMaxFramesInFlight = 8;

namespace detail {

// THE THREADS ARE KEPT, not made per call. ParallelFor used to create and
// join its threads every time, which is nothing for a frame-sized body and
// a great deal for a small one called often: the sparse Cholesky calls it
// once per supernode, hundreds of times per factorisation and a hundred
// factorisations per solve. Measured: global positioning on a 179-frame
// video got SLOWER with more threads (13.7 s at 8, 20.2 s at 30) -- the
// cost was in starting them.
//
// One job at a time. A call that finds the pool busy -- from another thread,
// or nested inside a body already running on it -- creates its own threads
// as before, so concurrent and nested use behave exactly as they always did.
class WorkerPool {
public:
    static WorkerPool& Get() {
        static WorkerPool pool;
        return pool;
    }

    // fn(ctx, i) for every i in [0, n), on `threads` threads counting the
    // caller's. False, having run nothing, when the pool is in use.
    bool TryRun(size_t n, unsigned threads, void (*fn)(void*, size_t), void* ctx) {
        if (t_inBody || !m_run.try_lock()) return false;
        std::lock_guard<std::mutex> runLock(m_run, std::adopt_lock);
        const unsigned helpers = std::min<unsigned>(threads - 1, unsigned(m_threads.size()));
        {
            std::lock_guard<std::mutex> lk(m_mtx);
            m_fn = fn;
            m_ctx = ctx;
            m_n = n;
            m_next.store(0, std::memory_order_relaxed);
            m_helpers = helpers;
            m_busy = helpers;
            ++m_gen;
        }
        m_cv.notify_all();
        Work();
        std::unique_lock<std::mutex> lk(m_mtx);
        m_doneCv.wait(lk, [&] { return m_busy == 0; });
        return true;
    }

private:
    WorkerPool() {
        const unsigned n = ParallelThreads(size_t(1) << 20);
        for (unsigned i = 0; i + 1 < n; ++i) m_threads.emplace_back([this, i] { Loop(i); });
    }
    ~WorkerPool() {
        {
            std::lock_guard<std::mutex> lk(m_mtx);
            m_quit = true;
        }
        m_cv.notify_all();
        for (std::thread& t : m_threads) t.join();
    }

    void Work() {
        t_inBody = true;
        for (;;) {
            const size_t i = m_next.fetch_add(1, std::memory_order_relaxed);
            if (i >= m_n) break;
            m_fn(m_ctx, i);
        }
        t_inBody = false;
    }

    void Loop(unsigned id) {
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(m_mtx);
                m_cv.wait(lk, [&] { return m_quit || m_gen != seen; });
                if (m_quit) return;
                seen = m_gen;
                if (id >= m_helpers) continue;   // not wanted for this one
            }
            Work();
            std::lock_guard<std::mutex> lk(m_mtx);
            if (--m_busy == 0) m_doneCv.notify_all();
        }
    }

    std::mutex m_run;                        // held for a whole job
    std::mutex m_mtx;
    std::condition_variable m_cv, m_doneCv;
    std::vector<std::thread> m_threads;
    void (*m_fn)(void*, size_t) = nullptr;
    void* m_ctx = nullptr;
    size_t m_n = 0;
    std::atomic<size_t> m_next{0};
    unsigned m_helpers = 0, m_busy = 0;
    uint64_t m_gen = 0;
    bool m_quit = false;
    static inline thread_local bool t_inBody = false;
};

}  // namespace detail

// Calls body(i) for every i in [0, n), possibly on several threads, and
// returns once every call has finished.
//
// Runs INLINE on the calling thread when there is one item or one thread, so
// the single-image case -- which is most of them -- pays no thread creation at
// all, and a debugger stepping through it sees an ordinary loop.
//
// Indices are handed out from a shared counter rather than sliced into equal
// ranges up front: frames of a group can differ enormously in cost (a detector
// on a flat sky against one on foliage), and static slicing would leave threads
// idle waiting for whichever got the expensive half.
template <typename Body>
void ParallelFor(size_t n, Body body, unsigned maxThreads = 0) {
    if (n == 0) return;

    unsigned threads = ParallelThreads(n);
    if (maxThreads > 0) threads = std::min(threads, maxThreads);
    if (threads <= 1) {
        for (size_t i = 0; i < n; ++i) body(i);
        return;
    }

    // The kept threads when they are free; see WorkerPool.
    auto trampoline = [](void* ctx, size_t i) { (*static_cast<Body*>(ctx))(i); };
    if (detail::WorkerPool::Get().TryRun(n, threads, trampoline, &body)) return;
    threads = std::min(threads, kMaxSpawned);

    std::atomic<size_t> next{0};
    std::vector<std::thread> pool;
    pool.reserve(threads - 1);

    auto run = [&] {
        for (;;) {
            const size_t i = next.fetch_add(1, std::memory_order_relaxed);
            if (i >= n) break;
            body(i);
        }
    };

    // One fewer thread than requested, because the caller's own thread takes a
    // share too rather than blocking on a join while work is outstanding.
    for (unsigned t = 0; t + 1 < threads; ++t) pool.emplace_back(run);
    run();
    for (std::thread& th : pool) th.join();
}

}  // namespace tglab
