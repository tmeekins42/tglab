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
#include <cstddef>
#include <functional>
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
// Matches ImageLoader::Start's policy deliberately. Two different answers to
// "how many threads" in one program is how a machine ends up oversubscribed
// with neither place obviously at fault.
inline unsigned ParallelThreads(size_t n) {
    if (n <= 1) return 1;
    unsigned c = std::thread::hardware_concurrency();
    c = (c > 3) ? c - 2 : 2;
    c = std::min(c, 8u);
    return std::min(unsigned(n), std::max(1u, c));
}

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
void ParallelFor(size_t n, Body body) {
    if (n == 0) return;

    const unsigned threads = ParallelThreads(n);
    if (threads <= 1) {
        for (size_t i = 0; i < n; ++i) body(i);
        return;
    }

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
