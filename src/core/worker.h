// PipelineWorker — runs pipeline execution off the UI thread.
//
// Split of responsibilities, chosen so the shared state stays tiny:
//
//   UI thread   parse + interpret (phase 1). This is microseconds, and it is
//               the part that touches UiState (declaring sliders/dropdowns),
//               so keeping it here avoids sharing UiState at all.
//   Worker      Pipeline::Execute (phase 2) — the expensive part.
//
// The worker never mutates anything the UI thread reads. It builds a complete
// result and publishes it under a mutex; the UI swaps a pointer at frame
// start, so a frame can never see a half-updated pipeline.
#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "pipeline.h"

// Declared at global scope: writing `struct ID3D12Device*` inside namespace
// tglab would silently declare a *different* type and fail to link.
struct ID3D12Device;

namespace tglab {

// One unit of work: an already-interpreted pipeline plus its source images.
struct PipelineJob {
    uint64_t          seq = 0;
    Pipeline          pipe;
    std::vector<Data> sources;
};

// What the UI actually needs to draw: one image per declared viewer. The
// worker keeps the pipeline itself, because Execute() moves cached outputs out
// of the previous run and that cache is what makes slider drags fast.
struct ViewerImage {
    std::string name;
    Image       image;
};

struct PipelineOutcome {
    uint64_t                 seq = 0;
    bool                     ok  = false;
    std::string              error;
    std::vector<ViewerImage> viewers;
};

class ComputeContext;

class PipelineWorker {
public:
    ~PipelineWorker() { Stop(); }

    // `device` may be null to run CPU-only. When given, the worker creates its
    // own ComputeContext (and therefore its own compute queue) on the worker
    // thread, so no D3D12 object is shared with the UI thread's submission.
    void Start(ID3D12Device* device = nullptr);
    void Stop();

    void SetExecMode(ExecMode m) { m_mode.store(m, std::memory_order_relaxed); }
    ExecMode GetExecMode() const { return m_mode.load(std::memory_order_relaxed); }

    // Stages that ran on the GPU in the last completed job.
    int LastGpuStages() const { return m_lastGpuStages.load(std::memory_order_relaxed); }
    double LastRunMs() const { return m_lastMs.load(std::memory_order_relaxed); }

    // Replaces any pending job. A slider drag produces ~60 requests/second
    // against a run that may take much longer, so queueing them would build an
    // unbounded backlog of results nobody will ever see — only the newest
    // matters. Returns the sequence number assigned to this job.
    uint64_t Submit(Pipeline pipe, std::vector<Data> sources);

    // Non-blocking. Returns true and fills `out` when a newer result is ready.
    bool TryFetch(PipelineOutcome* out);

    // True while a job is queued or running (drives the "working" indicator).
    bool Busy() const { return m_busy.load(std::memory_order_relaxed); }

    uint64_t LastFinishedSeq() const { return m_lastFinished.load(std::memory_order_relaxed); }

private:
    void Run();

    std::thread             m_thread;
    std::mutex              m_mtx;
    std::condition_variable m_cv;

    std::unique_ptr<PipelineJob>     m_pending;    // at most one, newest wins
    std::unique_ptr<PipelineOutcome> m_result;     // awaiting UI pickup

    std::atomic<bool>     m_quit{false};
    std::atomic<bool>     m_busy{false};
    std::atomic<uint64_t> m_lastFinished{0};
    std::atomic<ExecMode> m_mode{ExecMode::Auto};
    std::atomic<int>      m_lastGpuStages{0};
    std::atomic<double>   m_lastMs{0.0};
    uint64_t              m_nextSeq = 1;
    ID3D12Device*         m_device = nullptr;
};

} // namespace tglab
