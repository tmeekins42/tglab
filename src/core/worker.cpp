#include "worker.h"

#include <chrono>

#include "../gpu/compute.h"
#include "../gpu/gpu_image.h"

namespace tglab {

void PipelineWorker::Start(ID3D12Device* device) {
    if (m_thread.joinable()) return;
    m_device = device;
    m_quit.store(false);
    m_thread = std::thread([this] { Run(); });
}

void PipelineWorker::Stop() {
    if (!m_thread.joinable()) return;
    {
        std::lock_guard<std::mutex> lock(m_mtx);
        m_quit.store(true);
    }
    m_cv.notify_all();
    m_thread.join();
}

uint64_t PipelineWorker::Submit(Pipeline pipe, std::vector<Data> sources,
                               std::vector<uint64_t> sourceVersions) {
    uint64_t seq;
    {
        std::lock_guard<std::mutex> lock(m_mtx);
        seq = m_nextSeq++;
        auto job = std::make_unique<PipelineJob>();
        job->seq     = seq;
        job->pipe    = std::move(pipe);
        job->sources = std::move(sources);
        job->sourceVersions = std::move(sourceVersions);
        m_pending    = std::move(job);   // drops any older pending job
        // Abandon whatever is running: this newer job supersedes it, and for a
        // slow filter the old value would otherwise have to finish first.
        if (m_running) m_running->Cancel();
        m_busy.store(true, std::memory_order_relaxed);
    }
    m_cv.notify_one();
    return seq;
}

uint64_t PipelineWorker::SubmitCompare(Pipeline pipe, std::vector<Data> sources,
                                       int stageIndex) {
    uint64_t seq;
    {
        std::lock_guard<std::mutex> lock(m_mtx);
        seq = m_nextSeq++;
        auto job = std::make_unique<PipelineJob>();
        job->seq          = seq;
        job->pipe         = std::move(pipe);
        job->sources      = std::move(sources);
        job->compare      = true;
        job->compareStage = stageIndex;
        m_pending         = std::move(job);
        if (m_running) m_running->Cancel();
        m_busy.store(true, std::memory_order_relaxed);
    }
    m_cv.notify_one();
    return seq;
}

bool PipelineWorker::TryFetch(PipelineOutcome* out) {
    std::lock_guard<std::mutex> lock(m_mtx);
    if (!m_result) return false;
    *out = std::move(*m_result);
    m_result.reset();
    return true;
}

void PipelineWorker::Run() {
    // The worker owns the previous pipeline: Execute() *moves* cached outputs
    // out of it, so it must not be reachable from the UI thread while a job is
    // running. Keeping it here means the cache survives across jobs without
    // any sharing.
    Pipeline prev;
    bool havePrev = false;

    // Created here, on the worker thread, so its command queue and allocators
    // are never touched by the UI thread.
    ComputeContext gpu;
    const bool haveGpu = m_device && gpu.Init(m_device);
    // Lets Image::MapCpuRead() pull pixels back when they live only on the GPU.
    if (haveGpu) InstallGpuResidencyHooks();

    for (;;) {
        std::unique_ptr<PipelineJob> job;
        CancelTokenPtr token;
        {
            std::unique_lock<std::mutex> lock(m_mtx);
            m_cv.wait(lock, [this] { return m_quit.load() || m_pending != nullptr; });
            if (m_quit.load()) break;
            job = std::move(m_pending);
            // A fresh token per job. Submit() cancels through this while the
            // run is in flight.
            m_running = std::make_shared<CancelToken>();
            token = m_running;
        }

        auto outcome = std::make_unique<PipelineOutcome>();
        outcome->seq = job->seq;

        if (job->compare) {
            // Comparison runs the pipeline twice with explicit modes, so it
            // must not reuse the cache (which would make the second run free
            // and the timing meaningless).
            outcome->isCompare = true;
            auto cr = std::make_shared<CompareResult>(
                CompareCpuGpu(job->pipe, &job->sources,
                              haveGpu ? &gpu : nullptr, job->compareStage));
            outcome->ok      = cr->ok;
            outcome->error   = cr->error;
            outcome->compare = std::move(cr);

            // A comparison leaves the pipeline's cache in an unknown mode, so
            // drop it rather than let the next normal run inherit it.
            havePrev = false;
            prev.Clear();

            const uint64_t seq = outcome->seq;
            {
                std::lock_guard<std::mutex> lock(m_mtx);
                m_result = std::move(outcome);
                m_running.reset();
                if (!m_pending) m_busy.store(false, std::memory_order_relaxed);
            }
            m_lastFinished.store(seq, std::memory_order_relaxed);
            continue;
        }

        const auto t0 = std::chrono::steady_clock::now();
        std::string err;
        const bool ok = job->pipe.Execute(&job->sources, havePrev ? &prev : nullptr, &err,
                                          haveGpu ? &gpu : nullptr,
                                          m_mode.load(std::memory_order_relaxed),
                                          &job->sourceVersions, token.get());
        m_lastMs.store(
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(),
            std::memory_order_relaxed);
        m_lastGpuStages.store(job->pipe.GpuStageCount(), std::memory_order_relaxed);
        m_lastCpuStages.store(job->pipe.CpuStageCount(), std::memory_order_relaxed);
        m_lastCachedStages.store(job->pipe.CachedStageCount(), std::memory_order_relaxed);

        // A cancelled run is not a failure and has nothing to show: a newer job
        // is already queued. Publishing it would flash a stale or partial image
        // and, if reported, an error the user never caused.
        const bool cancelled = !ok && err == Pipeline::kCancelled;
        if (cancelled) {
            // The abandoned pipeline's stage cache is half-finished, so it
            // cannot seed the next run -- a cancelled stage is marked invalid,
            // but dropping the whole thing is simpler than reasoning about
            // which prefix survived.
            havePrev = false;
            prev.Clear();
            {
                std::lock_guard<std::mutex> lock(m_mtx);
                m_running.reset();
                if (!m_pending) m_busy.store(false, std::memory_order_relaxed);
            }
            continue;
        }

        outcome->ok    = ok;
        outcome->error = err;

        if (ok) {
            // Copy out only what the UI draws. The pipeline (and its stage
            // cache) stays here, so the next job can still skip unchanged
            // stages — that cache is exactly what makes dragging a slider on a
            // late stage cheap.
            for (const ViewerDecl& vd : job->pipe.Viewers()) {
                const Data* d = job->pipe.Resolve(vd.source, &job->sources);
                if (!d || !std::holds_alternative<Image>(*d)) continue;
                outcome->viewers.push_back(
                    {vd.name, const_cast<Image&>(std::get<Image>(*d)).Clone()});
            }

            prev     = std::move(job->pipe);
            havePrev = true;
        }

        const uint64_t finishedSeq = outcome->seq;
        {
            std::lock_guard<std::mutex> lock(m_mtx);
            m_result = std::move(outcome);
            m_running.reset();
            // Only idle when nothing newer arrived while we were working.
            if (!m_pending) m_busy.store(false, std::memory_order_relaxed);
        }
        m_lastFinished.store(finishedSeq, std::memory_order_relaxed);
    }

    // Stage kernels hold PSOs created by this context, so they must go first.
    prev.Clear();
    if (haveGpu) gpu.Shutdown();
}

} // namespace tglab
