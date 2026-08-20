// CancelToken — lets a long-running pipeline be abandoned mid-flight.
//
// The worker already coalesces *queued* jobs (newest wins), but a job that has
// already started runs to completion. For a filter costing a minute on a large
// scan that is the whole problem: nudging a slider waits out the old value
// before the new one even starts, so the UI is responsive while being useless.
//
// The token is set by whoever supersedes the work and polled by whoever is
// doing it. Polling is a relaxed atomic load -- cheap enough for a per-row
// check in an inner loop, which is the granularity that matters: per-stage
// alone is useless when a single stage is the slow part.
//
// Cancellation is cooperative and best-effort. An algorithm that never polls
// simply finishes, and the result is discarded because a newer job supersedes
// it, so nothing is left inconsistent by ignoring it.
#pragma once

#include <atomic>
#include <memory>

namespace tglab {

class CancelToken {
public:
    void Cancel()          { m_cancelled.store(true, std::memory_order_relaxed); }
    bool Cancelled() const { return m_cancelled.load(std::memory_order_relaxed); }
    void Reset()           { m_cancelled.store(false, std::memory_order_relaxed); }

private:
    std::atomic<bool> m_cancelled{false};
};

using CancelTokenPtr = std::shared_ptr<CancelToken>;

} // namespace tglab
