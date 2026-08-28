#include "worker.h"

#include <cmath>
#include <chrono>

#include "../gpu/compute.h"
#include "../gpu/gpu_image.h"

namespace tglab {
namespace {

// Turns GPU bins into the StatsResult the info panel draws.
//
// The statistics come from the bins rather than the pixels, exactly as the CPU
// path computes them from algo_util/Histogram -- so mean, median and stddev are
// bin-quantised in both, and the two agree.
StatsResult StatsFromBins(const ComputeContext::HistogramResult& h,
                          const ImageDesc& desc,
                          std::string source, uint64_t version) {
    constexpr int kBins = 256;

    StatsResult s;
    s.source  = std::move(source);
    s.version = version;
    s.valid   = true;

    auto toFloat = [](const std::vector<uint32_t>& in, std::vector<float>* out) {
        out->assign(in.size(), 0.0f);
        for (size_t i = 0; i < in.size(); ++i) (*out)[i] = float(in[i]);
    };
    if (!h.r.empty()) { toFloat(h.r, &s.r); toFloat(h.g, &s.g); toFloat(h.b, &s.b); }
    toFloat(h.luma, &s.luma);

    const double span = h.rangeMax - h.rangeMin;
    auto binValue = [&](int i) {
        return h.rangeMin + (double(i) / double(kBins - 1)) * span;
    };

    const double n = double(h.count ? h.count : 1);

    double sum = 0.0;
    for (int i = 0; i < kBins && i < int(h.luma.size()); ++i)
        sum += binValue(i) * double(h.luma[size_t(i)]);
    s.mean = sum / n;

    double var = 0.0;
    for (int i = 0; i < kBins && i < int(h.luma.size()); ++i) {
        const double d = binValue(i) - s.mean;
        var += d * d * double(h.luma[size_t(i)]);
    }
    s.stddev = std::sqrt(var / n);

    // Median by walking to the half-count bin, as Percentile() does.
    {
        const double target = n * 0.5;
        double acc = 0.0;
        s.median = binValue(0);
        for (int i = 0; i < kBins && i < int(h.luma.size()); ++i) {
            acc += double(h.luma[size_t(i)]);
            if (acc >= target) { s.median = binValue(i); break; }
        }
    }

    // The tallest interior bin scales the plot: the end bins collect
    // everything clipped, so including them would flatten the curve.
    float peak = 1.0f;
    for (int i = 1; i + 1 < kBins && i < int(s.luma.size()); ++i)
        peak = std::max(peak, s.luma[size_t(i)]);
    s.peak = peak;

    s.scale = (desc.format == Format::RGBA8) ? 255.0 : 1.0;

    // Headroom above 1.0 exists only for a float image, and only when the
    // capture actually had it. The bins span the observed range, so the top of
    // that range is where to look.
    s.maxValue    = h.rangeMax;
    s.hasHeadroom = (s.scale == 1.0) && (h.rangeMax > 1.0001);

    if (!h.luma.empty()) {
        s.clipLow  = double(h.luma.front()) / n;
        s.clipHigh = double(h.luma.back()) / n;
    }
    return s;
}

} // namespace

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

uint64_t PipelineWorker::Submit(Pipeline pipe, std::shared_ptr<std::vector<Data>> sources,
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

uint64_t PipelineWorker::SubmitCompare(Pipeline pipe, std::shared_ptr<std::vector<Data>> sources,
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


void PipelineWorker::SetVisibleViewers(std::vector<std::string> names) {
    std::lock_guard<std::mutex> lock(m_mtx);
    m_visibleViewers = std::move(names);
}

void PipelineWorker::SetHistogramViewer(std::string name) {
    std::lock_guard<std::mutex> lock(m_mtx);
    m_histViewer = std::move(name);
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
                CompareCpuGpu(job->pipe, job->sources.get(),
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

        // Did any palette image change since the last run? A viewer that
        // displays a source directly has no stage whose cache could tell us, so
        // the versions are compared here.
        bool sourcesChanged = (job->sourceVersions != m_lastSourceVersions);
        m_lastSourceVersions = job->sourceVersions;

        const auto t0 = std::chrono::steady_clock::now();
        std::string err;
        m_progress.Set(0, 0, "starting");
        const bool ok = job->pipe.Execute(job->sources.get(), havePrev ? &prev : nullptr, &err,
                                          haveGpu ? &gpu : nullptr,
                                          m_mode.load(std::memory_order_relaxed),
                                          &job->sourceVersions, token.get(), &m_progress);
        const double elapsedMs =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

        // A cancelled run is not a failure and has nothing to show: a newer job
        // is already queued. Publishing it would flash a stale or partial image
        // and, if reported, an error the user never caused.
        const bool cancelled = !ok && err == Pipeline::kCancelled;

        // Timing and stage counts describe the run the user can SEE. Recording
        // them for a cancelled run made the status bar flicker between the real
        // figure and a near-zero one: mid-drag, each frame's job is abandoned
        // almost immediately by the next, so "53 ms" alternated with "0 ms" and
        // read as though the pipeline were running twice. It was not -- only
        // one run's output was ever displayed.
        if (!cancelled) {
            m_lastMs.store(elapsedMs, std::memory_order_relaxed);
            // Taken from Progress rather than from the context directly: the
            // pipeline has already published the run's final figure there, and
            // reading the context here would race with a job that has started.
            m_lastGpuMs.store(m_progress.GpuMs(), std::memory_order_relaxed);
            m_lastGpuStages.store(job->pipe.GpuStageCount(), std::memory_order_relaxed);
            m_lastCpuStages.store(job->pipe.CpuStageCount(), std::memory_order_relaxed);
            m_lastCachedStages.store(job->pipe.CachedStageCount(), std::memory_order_relaxed);
            m_lastBypassed.store(job->pipe.BypassedStageCount(), std::memory_order_relaxed);

            // Whatever the stages have to say about what they just did.
            //
            // Collected here, on the worker, rather than letting the UI reach
            // into the pipeline: the algorithm objects live on this thread and
            // are replaced under the lock, so a UI-thread call into them would
            // be a race. Copying a handful of short strings is cheap and makes
            // the ownership obvious.
            //
            // Same `!cancelled` guard as the timings, and for the same reason:
            // a cancelled run's counts describe work that was abandoned
            // part-way, and showing them would flicker mid-drag.
            std::vector<std::string> reports;
            for (const Stage& s : job->pipe.Stages()) {
                if (!s.algo || !s.valid) continue;
                std::string r = s.algo->RunReport();
                if (!r.empty()) reports.push_back(std::move(r));
            }
            // A GPU fallback belongs with the reports: it is the single most
            // useful thing to know about a run that was unexpectedly slow.
            for (const std::string& f : job->pipe.GpuFallbacks())
                reports.push_back("GPU fallback -- " + f);

            {
                std::lock_guard<std::mutex> lock(m_mtx);
                m_lastReports = std::move(reports);
            }
        }
        if (cancelled) {
            // KEEP the abandoned pipeline as the cache for the next run.
            //
            // This used to drop it, on the reasoning that a half-finished cache
            // "cannot seed the next run" and that reasoning about which prefix
            // survived was not worth it. Both halves of that were wrong, and
            // the cost was severe: dragging a develop slider far enough to
            // cancel a run threw away the demosaic, the hot-pixel repair and
            // the whole HDR merge, so a one-parameter change re-ran seven raws.
            //
            // Nothing needs to be reasoned about, because the invariant already
            // holds. Execute() sets s.valid = false BEFORE running each stage
            // and true only on success, so a cancelled stage is already marked
            // invalid; and the cache scan stops at the first invalid stage
            // (`if (!ps.valid) break;`). A half-finished pipeline is therefore
            // exactly a pipeline whose completed prefix is valid -- which is
            // the same shape as any other prev, and is handled by the same
            // code path.
            //
            // The stages after the cancellation point are not merely unused:
            // they are marked invalid, so the scan cannot reach past them even
            // if a later run's hashes happen to match.
            prev = std::move(job->pipe);
            havePrev = true;
            {
                std::lock_guard<std::mutex> lock(m_mtx);
                m_running.reset();
                if (!m_pending) {
                    m_busy.store(false, std::memory_order_relaxed);
                    m_progress.Clear();
                }
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
            //
            // Each viewer carries a version that changes only when its own
            // pixels do. The UI uploads on a version change, so displaying a
            // source next to a slider-driven stage no longer re-converts the
            // source every frame.
            //
            // The worker cannot skip *sending* an unchanged viewer: m_result
            // holds one outcome and a newer run overwrites an unfetched one, so
            // "already sent" is not something this side can know. Cloning an
            // unchanged image is cheap anyway once it is CPU-resident -- the
            // expensive parts were the readback and the RGBA8 conversion, and
            // the version check skips both.
            const size_t firstDirty = job->pipe.FirstDirtyStage();
            for (const ViewerDecl& vd : job->pipe.Viewers()) {
                const Data* d = job->pipe.Resolve(vd.source, job->sources.get());
                if (!d || !std::holds_alternative<Image>(*d)) continue;

                // A palette source (stage < 0) changes only when the file
                // behind it is replaced, which arrives as a new source version.
                const bool changed = (vd.source.stage < 0)
                                         ? sourcesChanged
                                         : size_t(vd.source.stage) >= firstDirty;
                uint64_t& ver = m_viewerVersions[vd.name];
                if (changed || ver == 0) ++ver;

                const Image& result = std::get<Image>(*d);

                // Still on the GPU: hand over a reference and do NOT read it
                // back. Clone() would map the pixels, which is the 86 ms this
                // whole path exists to avoid.
                //
                // A CPU-only result (a stage with no GPU kernel) has no shared
                // texture, so it takes the clone as before -- both cases have
                // to work, since a script can mix the two.
                std::shared_ptr<SharedGpuTexture> shared = ShareGpuTexture(result);

                ViewerImage vi;
                vi.name    = vd.name;
                vi.version = ver;
                vi.gpu     = shared;
                if (shared) {
                    // Descriptor only -- deliberately NOT Image(desc), which
                    // allocates and zero-fills a full CPU buffer and reports
                    // itself CPU-resident. That would spend the 84 MB this path
                    // exists to save and hand every reader an image of zeros.
                    vi.image.AdoptDesc(result.Desc());
                } else {
                    vi.image = const_cast<Image&>(result).Clone();
                }
                outcome->viewers.push_back(std::move(vi));
            }

            // The info panel's histogram, measured here while the pixels are
            // still resident. Only the viewer the panel is showing, and only
            // when it is open -- the panel draws one at a time.
            std::string wantStats;
            {
                std::lock_guard<std::mutex> lock(m_mtx);
                wantStats = m_histViewer;
            }
            if (haveGpu && !wantStats.empty()) {
                for (const ViewerDecl& vd : job->pipe.Viewers()) {
                    if (vd.name != wantStats) continue;
                    const Data* d = job->pipe.Resolve(vd.source, job->sources.get());
                    if (!d || !std::holds_alternative<Image>(*d)) break;
                    const Image& result = std::get<Image>(*d);
                    const GpuResidency* g = result.RawGpu();
                    if (!g || !g->image.Valid()) break;   // CPU-only stage

                    ComputeContext::HistogramResult hr;
                    std::string herr;
                    if (gpu.BuildHistogram(g->image, &hr, &herr)) {
                        outcome->stats = StatsFromBins(hr, result.Desc(), vd.name,
                                                       m_viewerVersions[vd.name]);
                        outcome->haveStats = true;
                    }
                    break;
                }
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
