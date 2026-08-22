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
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "cancel.h"
#include "compare.h"
#include "pipeline.h"

// Declared at global scope: writing `struct ID3D12Device*` inside namespace
// tglab would silently declare a *different* type and fail to link.
struct ID3D12Device;

namespace tglab {

// One unit of work: an already-interpreted pipeline plus its source images.
// `compare` runs the pipeline twice (CPU then GPU) and diffs, instead of the
// normal single run.
struct PipelineJob {
    uint64_t              seq = 0;
    Pipeline              pipe;
    std::vector<Data>     sources;
    // Parallel to `sources`: changes when that palette image is replaced, so
    // the stage cache can tell a swapped image from the same one.
    std::vector<uint64_t> sourceVersions;
    bool              compare    = false;
    int               compareStage = -1;   // -1 = last stage
};

// What the UI actually needs to draw: one image per declared viewer. The
// worker keeps the pipeline itself, because Execute() moves cached outputs out
// of the previous run and that cache is what makes slider drags fast.
struct ViewerImage {
    std::string name;
    Image       image;

    // Bumped only when THIS viewer's pixels actually changed. The UI uploads
    // to its display texture on a version change, so a viewer reading from a
    // cached stage -- display(src, ...) while a slider drives a later stage --
    // costs nothing per frame instead of a full readback and conversion.
    uint64_t    version = 0;
};

struct PipelineOutcome {
    uint64_t                 seq = 0;
    bool                     ok  = false;
    std::string              error;
    std::vector<ViewerImage> viewers;

    // Set when the job was a comparison rather than a normal run.
    bool                           isCompare = false;
    std::shared_ptr<CompareResult> compare;
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

    // How the last completed job split across backends.
    int LastGpuStages()    const { return m_lastGpuStages.load(std::memory_order_relaxed); }
    int LastCpuStages()    const { return m_lastCpuStages.load(std::memory_order_relaxed); }
    int LastCachedStages() const { return m_lastCachedStages.load(std::memory_order_relaxed); }
    double LastRunMs() const { return m_lastMs.load(std::memory_order_relaxed); }

    // Replaces any pending job. A slider drag produces ~60 requests/second
    // against a run that may take much longer, so queueing them would build an
    // unbounded backlog of results nobody will ever see — only the newest
    // matters. Returns the sequence number assigned to this job.
    uint64_t Submit(Pipeline pipe, std::vector<Data> sources,
                    std::vector<uint64_t> sourceVersions = {});

    // Runs the pipeline twice (CPU then GPU) and diffs the chosen stage.
    // Deliberately not coalesced with normal runs in the caller's mind: it is
    // an explicit, one-off request, not something a slider triggers.
    uint64_t SubmitCompare(Pipeline pipe, std::vector<Data> sources, int stageIndex = -1);

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

    // Token for the job currently executing, so a newly submitted job can
    // abandon it. Guarded by m_mtx like m_pending; the worker clears it when
    // the run finishes. shared_ptr because the worker thread keeps using it
    // after releasing the lock.
    CancelTokenPtr                   m_running;
    std::unique_ptr<PipelineOutcome> m_result;     // awaiting UI pickup

    std::atomic<bool>     m_quit{false};
    std::atomic<bool>     m_busy{false};
    std::atomic<uint64_t> m_lastFinished{0};
    std::atomic<ExecMode> m_mode{ExecMode::Auto};
    std::atomic<int>      m_lastGpuStages{0};
    std::atomic<int>      m_lastCpuStages{0};
    std::atomic<int>      m_lastCachedStages{0};
    std::atomic<double>   m_lastMs{0.0};

    // Per-viewer bookkeeping for skipping unchanged viewers (see Run()).
    // Worker-thread only, so unguarded.
    std::map<std::string, uint64_t> m_viewerVersions;
    std::vector<uint64_t>           m_lastSourceVersions;
    uint64_t              m_nextSeq = 1;
    ID3D12Device*         m_device = nullptr;
};

} // namespace tglab
