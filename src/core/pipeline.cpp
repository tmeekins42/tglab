#include "pipeline.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <functional>
#include <cassert>
#include <map>
#include <set>

#include "../gpu/compute.h"
#include "../gpu/gpu_image.h"

#include <windows.h>
#include <cstdio>

namespace tglab {

// Diagnostic tracing for GPU fallbacks, off unless TGLAB_GPUWHY is set.
// A silent fallback looks like the GPU merely being slow, which is what made
// the multi-pass device hang invisible for so long.
static bool GpuWhy() {
    static const bool on = GetEnvironmentVariableA("TGLAB_GPUWHY", nullptr, 0) > 0;
    return on;
}

// Submit and wait after every multi-pass dispatch, so a device hang can be
// attributed to one kernel. Diagnostic only: it defeats batching entirely.
static bool GpuPassTiming() {
    static const bool on = GetEnvironmentVariableA("TGLAB_PASSTIME", nullptr, 0) > 0;
    return on;
}

// Optional per-frame callback for diagnosing fused-reduction memory growth.
// Null in normal builds; set by tools/group_merge.
std::function<void(int)> g_frameTrace;

// Running tallies for the status line. See the declaration for why the GPU
// number cannot come from timing individual stages.
void Pipeline::PublishStats(Progress* progress, const ComputeContext* gpu) const {
    if (!progress) return;
    progress->SetStats(m_cpuStages, m_gpuStages,
                       std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - m_runStart).count(),
                       gpu ? gpu->GpuMs() : 0.0);
}

void Pipeline::Clear() {
    m_stages.clear();
    m_viewers.clear();
}

int Pipeline::AddStage(std::unique_ptr<AlgorithmBase> algo, std::string name,
                       std::vector<PortRef> inputs, size_t numOutputs, int line,
                       std::string reduceAxis, Shape reshapeTo) {
    Stage s;
    s.paramHash = algo->ParamHash();
    s.algo      = std::move(algo);
    s.algoName  = std::move(name);
    s.inputs    = std::move(inputs);
    s.outputs.resize(numOutputs);
    s.line      = line;
    s.reduceAxis = std::move(reduceAxis);
    s.reshapeTo  = std::move(reshapeTo);
    m_stages.push_back(std::move(s));
    return int(m_stages.size()) - 1;
}

void Pipeline::AddViewer(std::string name, PortRef src) {
    // Re-declaring the same viewer name updates it rather than duplicating.
    for (ViewerDecl& v : m_viewers) {
        if (v.name == name) {
            v.source = src;
            return;
        }
    }
    m_viewers.push_back({std::move(name), src});
}

const Data* Pipeline::Resolve(PortRef r, const std::vector<Data>* sources) const {
    if (r.stage < 0) {
        if (!sources || r.port < 0 || size_t(r.port) >= sources->size()) return nullptr;
        return &(*sources)[size_t(r.port)];
    }
    if (size_t(r.stage) >= m_stages.size()) return nullptr;
    const Stage& s = m_stages[size_t(r.stage)];
    if (r.port < 0 || size_t(r.port) >= s.outputs.size()) return nullptr;
    return &s.outputs[size_t(r.port)];
}

bool Pipeline::SameStage(const Stage& a, const Stage& b) {
    return a.algoName == b.algoName && a.inputs == b.inputs &&
           a.paramHash == b.paramHash && a.sourceHash == b.sourceHash &&
           a.reduceAxis == b.reduceAxis && a.reshapeTo == b.reshapeTo &&
           a.outputs.size() == b.outputs.size();
}

// Streams one axis of a set through an algorithm's accumulator.
//
// Rank 1 only for now: the input is [axis=n] and the result is a single image.
// Reducing one axis of a higher-rank set means mapping the reduction across the
// axes that are not being reduced -- the broadcasting rule the design describes
// -- which needs a set-valued OUTPUT to hold the results. That is the next step
// and is deliberately not faked here: returning a wrong-shaped answer would be
// worse than refusing.
// Finds the run of broadcast stages feeding a reduction, so they can be fused.
//
// This is the memory fix. Run stage by stage, a reduction over five 45 MP raws
// demosaics all five and holds all five results before merging -- 1.7 GB of
// intermediates. Run frame by frame, one frame is live at a time.
//
// The chain qualifies only when it is a simple line: each stage takes its ONLY
// set input from the previous one, produces one output, and nothing else reads
// that output. Anything else -- a branch, a second consumer, a display() on an
// intermediate -- means the intermediate is genuinely wanted and fusing it away
// would change what the script produces.
//
// Returns the stages in execution order, or empty when the chain does not
// qualify. `srcPort` is left pointing at whatever feeds the head of it.
std::vector<int> Pipeline::FusableChain(int reduceStage, PortRef* srcPort) const {
    std::vector<int> chain;
    const Stage& red = m_stages[size_t(reduceStage)];
    if (red.inputs.size() != 1) return {};

    PortRef cur = red.inputs[0];
    while (cur.stage >= 0) {
        const Stage& s = m_stages[size_t(cur.stage)];

        // One output only, and this must be it: a multi-output stage's other
        // ports would be discarded by fusing.
        if (s.outputs.size() != 1 || cur.port != 0) break;

        // A reduction inside the chain is a different shape of problem -- two
        // accumulators live at once -- and is not fused here.
        if (s.algo->IsReduction()) break;

        // A reshape is not a per-frame operation: it reinterprets the WHOLE
        // set at once, so it cannot be run one frame at a time. Fusing it away
        // silently dropped the reshape and the reduction then saw the original
        // flat shape -- the chain has to stop here.
        if (s.algo->IsReshape()) break;

        // An aligner is not a per-frame operation either: it solves every frame
        // against a common reference, so it cannot be carried one frame at a
        // time. Fusing it away meant it silently never ran -- the same failure
        // the reshape check above exists to prevent.
        if (s.algo->IsAligner()) break;

        // Exactly one input, so the frame identity is unambiguous. A stage
        // combining a set with a second image is legitimate but needs the
        // scalar input held across frames, which is a further step.
        if (s.inputs.size() != 1) break;

        // Nothing else may read this stage's output, or fusing would delete an
        // intermediate something still wants.
        int readers = 0;
        for (const Stage& other : m_stages)
            for (const PortRef& r : other.inputs)
                if (r.stage == cur.stage) ++readers;
        for (const ViewerDecl& v : m_viewers)
            if (v.source.stage == cur.stage) ++readers;
        if (readers != 1) break;

        chain.push_back(cur.stage);
        cur = s.inputs[0];
    }

    std::reverse(chain.begin(), chain.end());
    *srcPort = cur;
    return chain;
}

// Runs a broadcast chain and its reduction one frame at a time.
//
// The difference from running them separately is only WHEN the work happens,
// not what it computes: instead of demosaicing every frame and then merging,
// each frame is carried through the whole chain and handed to the accumulator
// before the next is started. One frame is live at a time rather than N.
//
// Measured on five 45 MP CR3s: 1.7 GB of demosaiced intermediates becomes one
// frame's 341 MB. The design calls this depth-first evaluation and predicted
// the level-by-level order would be the expensive one, which is what the
// unfused path was doing.
bool Pipeline::RunFusedReduction(int reduceStage, const std::vector<int>& chain,
                                 PortRef srcPort, const std::vector<Data>* sources,
                                 ComputeContext* gpu, ExecMode mode,
                                 const CancelToken* cancel, Progress* progress,
                                 std::string* err) {
    Stage& red = m_stages[size_t(reduceStage)];
    auto Where = [&] { return "line " + std::to_string(red.line) + ": '" + red.algoName + "' "; };

    const Data* srcData = Resolve(srcPort, sources);
    if (!srcData) { *err = Where() + "has an input that produced no data"; return false; }
    const auto* set = std::get_if<ImageSet>(srcData);
    if (!set) { *err = Where() + "reduces a set, but its input is a single image"; return false; }
    if (set->shape.Rank() != 1) {
        // Should not happen: the planner checks this. Refusing rather than
        // reducing the wrong axis, since a wrong merge does not announce itself.
        *err = Where() + "cannot fuse a reduction of " + set->shape.ToString();
        return false;
    }
    if (set->images.empty()) { *err = Where() + "was given an empty group"; return false; }

    Reducer* r = red.algo->AsReducer();
    if (!r) { *err = Where() + "declares itself a reduction but provides no accumulator"; return false; }

    const std::string axis = red.reduceAxis.empty() ? set->shape.Axes()[0].name : red.reduceAxis;
    if (!r->Begin(int(set->images.size()), axis, err)) return false;
    if (GpuWhy()) std::fprintf(stderr, "[fuse] chain of %zu stages, %zu frames\n",
                               chain.size(), set->images.size());

    for (size_t f = 0; f < set->images.size(); ++f) {
        if (cancel && cancel->Cancelled()) { *err = kCancelled; return false; }

        // Per-FRAME progress, which is the granularity that matters here: a
        // five-frame merge is one stage but several seconds, so stage-level
        // progress alone would sit at one number the whole time.
        if (progress) {
            char what[64];
            std::snprintf(what, sizeof what, "%s  frame %d/%d", red.algoName.c_str(),
                          int(f) + 1, int(set->images.size()));
            progress->Set(int(f), int(set->images.size()), what);
            // Per frame as well as per stage: this loop IS the slow part of a
            // multi-frame merge, so an elapsed time that only advanced between
            // stages would sit frozen for the whole of it.
            PublishStats(progress, gpu);
        }

        // Carry this one frame through every stage of the chain. `carried` owns
        // the frame between stages and is overwritten each time, which is what
        // makes the peak one frame rather than N.
        Data carried{set->images[f].Clone()};
        for (int si : chain) {
            Stage& cs = m_stages[size_t(si)];
            const Data* one = &carried;
            std::vector<const Data*> csIn{one};
            if (!RunStageOnce(cs, csIn, gpu, mode, cancel, err)) return false;
            if (cs.outputs.size() != 1) { *err = Where() + "chain stage produced no result"; return false; }
            carried = std::move(cs.outputs[0]);
            cs.outputs[0] = Data{};
        }

        const auto* img = std::get_if<Image>(&carried);
        if (!img) { *err = Where() + "chain produced no image for frame " + std::to_string(f); return false; }

        // Accept() reads CPU pixels, which pulls this frame back from the GPU
        // (and flushes the batch on the way, since Readback submits).
        if (!r->Accept(int(f), *img, err)) return false;


        // Per-frame trace, for diagnosing GPU memory growth. Installed by a
        // harness rather than queried here, since core/ has no DXGI adapter.
        if (g_frameTrace) g_frameTrace(int(f));
        // Release the frame NOW rather than at the top of the next iteration.
        //
        // This is the whole point of fusing, and it is specifically the GPU
        // side that matters: every stage's output holds a GPU resource, and
        // letting five 45 MP frames' worth accumulate exhausted VRAM -- at
        // which point AcquireGpuRead fails, every stage silently falls back to
        // the CPU, and the "faster" path measured SLOWER than the unfused one.
        // That is what the timing said before this line existed.
        carried = Data{};
    }

    Image out;
    if (!r->Finish(&out, err)) return false;
    if (!out.Valid()) { *err = Where() + "produced no result"; return false; }

    red.outputs.clear();
    red.outputs.resize(1);
    red.outputs[0] = Data{std::move(out)};

    // The chain stages ran but hold nothing: their outputs were consumed frame
    // by frame and never existed as sets. `fusedInto` records which reduction
    // owns them, so the cache scan can skip past an empty stage to a valid
    // result rather than stopping at it -- otherwise ONE fused chain forces
    // every later stage to recompute, and a develop slider re-ran the whole
    // merge.
    for (int si : chain) {
        m_stages[size_t(si)].valid      = false;
        m_stages[size_t(si)].fusedInto  = reduceStage;
    }
    red.valid = true;
    return true;
}

bool Pipeline::RunReduction(Stage& s, const std::vector<const Data*>& in,
                            const CancelToken* cancel, std::string* err) {
    auto Where = [&] { return "line " + std::to_string(s.line) + ": '" + s.algoName + "' "; };

    if (in.size() != 1 || !in[0]) { *err = Where() + "takes one input"; return false; }

    const auto* set = std::get_if<ImageSet>(in[0]);
    if (!set) {
        *err = Where() + "reduces a set, but its input is a single image";
        return false;
    }
    if (set->images.empty()) { *err = Where() + "was given an empty group"; return false; }

    // The extent and the image count must agree. They always do when the
    // palette built the set, since every mutation restates the shape -- so a
    // mismatch here means something constructed one by hand and got it wrong,
    // which is worth saying rather than trusting.
    if (set->shape.Count() != int64_t(set->images.size())) {
        *err = Where() + "has a shape of " + set->shape.ToString() + " but " +
               std::to_string(set->images.size()) + " images";
        return false;
    }

    Reducer* r = s.algo->AsReducer();
    if (!r) { *err = Where() + "declares itself a reduction but provides no accumulator"; return false; }

    const std::string axis = s.reduceAxis.empty() ? set->shape.Axes()[0].name : s.reduceAxis;
    const int ax = set->shape.Find(axis);
    if (ax < 0) {
        *err = Where() + "cannot reduce over '" + axis + "': its input is " +
               set->shape.ToString();
        return false;
    }

    // Reducing one axis of a higher-rank set runs the accumulator ONCE PER
    // remaining coordinate -- [12,3] over "exposure" is twelve merges of three
    // frames each, not one merge of thirty-six. The result keeps the axes that
    // were not reduced, so a chain of reductions peels them off one at a time.
    //
    // The accumulator is reused across those runs rather than one being made
    // per output: Begin() is documented to reset it, and an algorithm that
    // holds state across Begin would be broken for the rank-1 case too.
    const Shape   outShape = set->shape.Without(ax);
    const int     extent   = set->shape.Axes()[size_t(ax)].extent;
    const int64_t stride   = set->shape.Stride(ax);
    const int64_t outCount = outShape.Count();

    ImageSet result;
    result.shape = outShape;
    result.images.reserve(size_t(outCount));

    for (int64_t o = 0; o < outCount; ++o) {
        if (cancel && cancel->Cancelled()) { *err = kCancelled; return false; }

        // Flat index of this output's first input image.
        //
        // The output's own coordinates are the input's with `ax` removed, so
        // the base is rebuilt by walking the input's axes and taking each
        // coordinate from the output except at `ax`, which is zero.
        int64_t rem = o, base = 0;
        for (int i = int(outShape.Rank()) - 1; i >= 0; --i) {
            const int e = outShape.Axes()[size_t(i)].extent;
            const int c = e > 0 ? int(rem % e) : 0;
            if (e > 0) rem /= e;
            const int inAxis = (i < ax) ? i : i + 1;
            base += int64_t(c) * set->shape.Stride(inAxis);
        }

        if (!r->Begin(extent, axis, err)) return false;
        for (int k = 0; k < extent; ++k) {
            if (cancel && cancel->Cancelled()) { *err = kCancelled; return false; }
            const int64_t idx = base + int64_t(k) * stride;
            if (idx < 0 || idx >= int64_t(set->images.size())) {
                *err = Where() + "indexed past the end of its input";
                return false;
            }
            if (!r->Accept(k, set->images[size_t(idx)], err)) return false;
        }

        Image out;
        if (!r->Finish(&out, err)) return false;
        if (!out.Valid()) { *err = Where() + "produced no result"; return false; }
        result.images.push_back(std::move(out));
    }

    s.outputs.clear();
    s.outputs.resize(1);
    // A reduction to rank 0 is a single image, not a set of one: everything
    // downstream takes an Image, and wrapping it would make every consumer
    // unwrap a set that can only ever hold one thing.
    if (outShape.IsScalar()) s.outputs[0] = Data{std::move(result.images[0])};
    else                     s.outputs[0] = Data{std::move(result)};
    return true;
}

// Runs one stage against one set of scalar inputs, writing scalar outputs.
//
// Extracted from Execute so broadcasting can call it once per frame. It writes
// into s.outputs, which RunStageGpu also reads and writes throughout -- so the
// broadcast loop swaps each frame's outputs in and out around the call rather
// than threading a buffer through the GPU path. Everything
// here assumes single images -- allocation from input 0, the GPU/CPU choice,
// cancellation -- which is exactly what makes it the right unit to map across a
// set rather than duplicating any of it.
bool Pipeline::RunStageOnce(Stage& s, const std::vector<const Data*>& in,
                            ComputeContext* gpu, ExecMode mode,
                            const CancelToken* cancel, std::string* err) {
    // Allocate outputs. Every port is an image sized from input 0, with the
    // format resolved from the port's FormatSpec.
    const PortList outPorts = s.algo->Outputs();
    ImageDesc base{};
    if (!in.empty() && std::holds_alternative<Image>(*in[0]))
        base = std::get<Image>(*in[0]).Desc();

    // A set reaching here means the broadcast dispatch was bypassed. Refuse
    // rather than allocating from a zeroed descriptor, which would silently
    // produce a zero-sized output instead of an error.
    for (const Data* d : in) {
        if (TypeOf(*d) == DataType::ImageSet) {
            *err = "line " + std::to_string(s.line) + ": '" + s.algoName +
                   "' was given " + ShapeOf(*d).ToString() + " but is not broadcasting";
            return false;
        }
    }

    for (size_t p = 0; p < s.outputs.size(); ++p) {
        ImageDesc d = base;
        switch (outPorts[p].format) {
            case FormatSpec::RGBA8:       d.format = Format::RGBA8;   break;
            case FormatSpec::R32F:        d.format = Format::R32F;    break;
            case FormatSpec::RGBA32F:     d.format = Format::RGBA32F; break;
            case FormatSpec::RGBA16F:     d.format = Format::RGBA16F; break;
            case FormatSpec::SameAsInput: /* keep base.format */      break;
            case FormatSpec::Any:         /* keep base.format */      break;
        }

        // The output of a multi-channel stage is no longer a mosaic, even
        // though it inherited the input's descriptor. Anything that widens a
        // single sample per pixel into RGB has, by definition, done the
        // demosaicing -- and leaving the CFA metadata set would make a finished
        // image claim to still need it, so a second demosaic in the chain would
        // happily mangle it.
        if (base.IsMosaic() && d.format != Format::R32F) {
            d.cfa = CfaPattern::None;

            // ...and what comes out is scene-linear. A demosaic converts sensor
            // counts to linear RGB; it never applies a transfer function, so the
            // result carries headroom above 1.0 wherever the white-balance gains
            // pushed a channel past saturation. Tonal algorithms downstream need
            // to know not to gamma-decode it and not to clamp it.
            d.linear = true;
        }
        if (!d.Valid()) {
            *err = "line " + std::to_string(s.line) + ": '" + s.algoName +
                   "' could not determine output size";
            return false;
        }
        Image img;
        img.Alloc(d);
        s.outputs[p] = Data{std::move(img)};
    }

    // GPU when asked for and available; otherwise CPU. A GPU failure falls back
    // rather than failing the run -- a broken kernel should degrade to a slow
    // correct result, not an empty viewer.
    //
    // A multi-pass stage carries its HLSL in GpuPasses() rather than in
    // GpuSource(), so requiring the latter would silently keep every multi-pass
    // algorithm on the CPU -- looking merely slow, with no error anywhere.
    const bool wantGpu = gpu && mode != ExecMode::ForceCPU && s.algo->HasGPU() &&
                         (s.algo->GpuSource() || !s.algo->GpuPasses().empty());
    bool ranOnGpu = false;
    if (wantGpu) {
        std::string gpuErr;
        if (RunStageGpu(s, in, gpu, &gpuErr)) {
            ranOnGpu = true;
            ++m_gpuStages;
        } else {
            // ALWAYS recorded and logged, not gated behind a diagnostic flag.
            // A silent fallback reads as "the GPU is slow today" -- which is how
            // a device hang on every 45 MP raw went unnoticed.
            const std::string note = s.algoName + ": " + gpuErr;
            m_gpuFallbacks.push_back(note);
            std::fprintf(stderr, "[gpu-fallback] %s\n", note.c_str());
        }
        if (!ranOnGpu && mode == ExecMode::ForceGPU) {
            *err = "line " + std::to_string(s.line) + ": '" + s.algoName +
                   "' GPU path failed: " + gpuErr;
            return false;
        }
    }

    if (!ranOnGpu) {
        RunCtx ctx(in, s.outputs, cancel);
        s.algo->RunCPU(ctx);
        ++m_cpuStages;

        // An algorithm that honoured the token has written only part of its
        // output. Leaving the stage valid would cache that partial result and,
        // worse, let a later run reuse it as though it were finished.
        if (cancel && cancel->Cancelled()) {
            *err = kCancelled;
            return false;
        }
    }
    return true;
}

// Maps a single-image stage across every frame of a set.
//
// This is what lets an ordinary algorithm -- one that knows nothing about
// groups -- be applied to all of them: develop every frame of a bracket, or
// demosaic every raw in a group. The algorithm still declares "one image in,
// one out"; the framework does the mapping, which is the same rule as NumPy
// broadcasting or SQL GROUP BY.
//
// Only ONE input may be a set, and only rank 1. Broadcasting several sets
// together raises the question of what to do when their shapes differ, and
// answering it before anything needs it would be guessing.
bool Pipeline::BroadcastStage(Stage& s, const std::vector<const Data*>& in,
                              ComputeContext* gpu, ExecMode mode,
                              const CancelToken* cancel, Progress* progress,
                              std::string* err) {
    auto Where = [&] { return "line " + std::to_string(s.line) + ": '" + s.algoName + "' "; };

    int setIdx = -1;
    for (size_t i = 0; i < in.size(); ++i) {
        if (TypeOf(*in[i]) != DataType::ImageSet) continue;
        if (setIdx >= 0) {
            *err = Where() + "cannot broadcast over two groups at once";
            return false;
        }
        setIdx = int(i);
    }
    if (setIdx < 0) { *err = Where() + "has no group to broadcast over"; return false; }

    const ImageSet& src = std::get<ImageSet>(*in[size_t(setIdx)]);

    // Any rank. Broadcasting maps the stage over every image and keeps the
    // shape, so the axes never have to be interpreted -- a [12,3] set produces
    // a [12,3] set of results. Only a reduction needs to know which axis is
    // which.
    if (src.shape.Count() != int64_t(src.images.size())) {
        *err = Where() + "has a shape of " + src.shape.ToString() + " but " +
               std::to_string(src.images.size()) + " images";
        return false;
    }

    // One output set per port, all sharing the input's shape.
    const size_t nPorts = s.outputs.size();
    std::vector<ImageSet> results(nPorts);
    for (ImageSet& r : results) {
        r.shape = src.shape;
        r.images.reserve(src.images.size());
    }

    // s.outputs is the scalar working buffer for one frame. RunStageGpu reads
    // and writes it, so each frame's result is moved out afterwards rather than
    // the buffer being threaded through the GPU path.
    std::vector<Data> saved = std::move(s.outputs);
    s.outputs.clear();
    s.outputs.resize(nPorts);

    bool ok = true;
    for (size_t f = 0; f < src.images.size() && ok; ++f) {
        if (cancel && cancel->Cancelled()) { *err = kCancelled; ok = false; break; }

        if (progress) {
            char what[64];
            std::snprintf(what, sizeof what, "%s  frame %d/%d", s.algoName.c_str(),
                          int(f) + 1, int(src.images.size()));
            progress->Set(int(f), int(src.images.size()), what);
            PublishStats(progress, gpu);   // same reason as the fused loop
        }

        // Swap this frame in for the set input; every other input is passed
        // through unchanged, which is what lets a broadcast stage still take
        // ordinary scalar parameters from earlier stages.
        Data frame{src.images[f].Clone()};
        std::vector<const Data*> frameIn = in;
        frameIn[size_t(setIdx)] = &frame;

        if (!RunStageOnce(s, frameIn, gpu, mode, cancel, err)) { ok = false; break; }

        for (size_t p = 0; p < nPorts; ++p) {
            auto* img = std::get_if<Image>(&s.outputs[p]);
            if (!img) { *err = Where() + "produced no image for frame " + std::to_string(f);
                        ok = false; break; }
            results[p].images.push_back(std::move(*img));
            s.outputs[p] = Data{};
        }
    }

    if (!ok) { s.outputs = std::move(saved); return false; }

    s.outputs.clear();
    s.outputs.resize(nPorts);
    for (size_t p = 0; p < nPorts; ++p) s.outputs[p] = Data{std::move(results[p])};
    return true;
}

bool Pipeline::Execute(std::vector<Data>* sources, Pipeline* prev, std::string* err,
                       ComputeContext* gpu, ExecMode mode,
                       const std::vector<uint64_t>* sourceVersions,
                       const CancelToken* cancel, Progress* progress) {
    // Stamp each stage with the versions of the palette images it reads, before
    // any cache comparison. PortRef{-1, i} is identical whichever file backs
    // slot i, so without this a swapped image reuses the cached output -- which
    // looks like the app ignoring the new file until some other parameter
    // changes and incidentally busts the cache.
    if (sourceVersions) {
        for (Stage& s : m_stages) {
            uint64_t h = 1469598103934665603ull;   // FNV-1a
            for (const PortRef& r : s.inputs) {
                if (r.stage >= 0) continue;        // not a palette image
                const uint64_t v = size_t(r.port) < sourceVersions->size()
                                       ? (*sourceVersions)[size_t(r.port)] : 0;
                for (int b = 0; b < 8; ++b) {
                    h ^= (v >> (b * 8)) & 0xff;
                    h *= 1099511628211ull;
                }
            }
            s.sourceHash = h;
        }
    }

    m_gpuStages = 0;
    m_gpuFallbacks.clear();
    m_cpuStages = 0;
    m_cachedStages = 0;

    // For the live elapsed time reported between stages. The worker times the
    // whole call for its own final figure; this one exists so the status line
    // can count up during a slow run instead of sitting at the previous run's
    // number until this one finishes.
    m_runStart = std::chrono::steady_clock::now();
    // The context outlives a single run, so its accumulator has to be zeroed
    // here or the GPU figure would be the total since the app started.
    if (gpu) gpu->ResetGpuMs();
    // Find the first stage that differs from the previous run. Everything
    // before it can reuse its cached output. Comparing by hash means there is
    // no dirty flag to forget to set.
    size_t firstDirty = 0;
    if (prev) {
        const size_t n = std::min(m_stages.size(), prev->m_stages.size());
        while (firstDirty < n && SameStage(m_stages[firstDirty], prev->m_stages[firstDirty])) {
            const Stage& ps = prev->m_stages[firstDirty];

            // A stage consumed by a fused reduction holds nothing and is not
            // valid, but that is not a reason to stop: its RESULT lives in the
            // reduction downstream, which is cached like anything else. Treat
            // it as unchanged and carry on.
            //
            // Without this the scan stopped at the first chain stage, so a
            // develop slider re-ran the entire HDR merge -- seven raws
            // demosaiced and merged for a one-line parameter change.
            if (!ps.valid && ps.fusedInto >= 0) {
                m_stages[firstDirty].fusedInto = ps.fusedInto;
                ++firstDirty;
                continue;
            }
            if (!ps.valid) break;

            // Steal the cached outputs — Data is move-only, and the previous
            // pipeline is discarded right after this call. The compiled kernel
            // comes along too, so an unchanged stage never recompiles.
            m_stages[firstDirty].outputs = std::move(prev->m_stages[firstDirty].outputs);
            m_stages[firstDirty].kernel  = prev->m_stages[firstDirty].kernel;
            m_stages[firstDirty].valid   = true;
            ++m_cachedStages;
            ++firstDirty;
        }
    }

    m_firstDirty = firstDirty;

    // Dirty stages still reuse their compiled kernel when they are the same
    // algorithm as last run — a parameter change does not alter the HLSL, and
    // recompiling on every slider tick would defeat the point of the GPU path.
    //
    // The iterative scratch carries over for the same reason. It is a
    // full-size texture -- 16 MB for an RGBA32F scratch at 1024x1024 -- and
    // reallocating one per slider tick is churn the driver has to absorb even
    // now that it is correctly freed. ExecuteGpuStage re-checks the descriptor
    // and reallocates if the size or format actually changed, so carrying a
    // stale one is safe.
    if (prev) {
        for (size_t i = firstDirty; i < m_stages.size(); ++i) {
            if (i >= prev->m_stages.size()) break;
            if (m_stages[i].algoName == prev->m_stages[i].algoName) {
                m_stages[i].kernel      = prev->m_stages[i].kernel;
                m_stages[i].gpuScratch  = prev->m_stages[i].gpuScratch;
                m_stages[i].scratchDesc = prev->m_stages[i].scratchDesc;
            }
        }
    }

    // Any exit from the stage loop -- an error, a cancellation, or falling off
    // the end -- must submit what the GPU stages recorded. Dispatch() no longer
    // flushes, so leaving early would strand a half-built command list, and the
    // *next* run's BeginRecording() would append to it rather than starting
    // clean. There are several early returns below, so this is a guard rather
    // than a line repeated at each of them.
    struct GpuBatchGuard {
        ComputeContext* gpu;
        ~GpuBatchGuard() {
            if (!gpu || !gpu->Ready()) return;
            std::string ignored;
            gpu->Flush(&ignored);   // the caller already has the real error
        }
    } batchGuard{gpu};

    // Reductions that will run their input chain frame by frame, and the chain
    // stages they will consume.
    //
    // Computed BEFORE the loop because a chain runs ahead of the reduction that
    // owns it: by the time the reduction is reached, its stages would already
    // have run and built the whole intermediate set -- which is the memory this
    // exists to avoid.
    std::map<int, std::pair<std::vector<int>, PortRef>> fusedPlan;
    std::set<int> fused;
    for (size_t i = firstDirty; i < m_stages.size(); ++i) {
        // TGLAB_NOFUSE runs the chain stage by stage instead, for measuring what
        // fusing actually saves.
        static const bool noFuse = GetEnvironmentVariableA("TGLAB_NOFUSE", nullptr, 0) > 0;
        if (noFuse) break;
        if (!m_stages[i].algo->IsReduction()) continue;
        PortRef srcPort{};
        std::vector<int> chain = FusableChain(int(i), &srcPort);
        if (chain.empty()) continue;

        // Only a rank-1 reduction can be fused. Fusing carries ONE frame
        // through the chain at a time, which assumes every frame feeds the same
        // accumulator -- a rank-2 reduction runs one accumulator per remaining
        // coordinate, so the frames are not interchangeable. The unfused path
        // handles any rank; it costs memory, not correctness.
        if (const Data* sd = Resolve(srcPort, sources)) {
            const auto* ss = std::get_if<ImageSet>(sd);
            if (!ss || ss->shape.Rank() != 1) continue;
        }

        // A stage already claimed by an earlier reduction cannot also feed this
        // one -- FusableChain's single-reader rule makes that impossible, but
        // the check is cheap and the failure would be silent.
        bool clash = false;
        for (int c : chain) if (fused.count(c)) clash = true;
        if (clash) continue;

        for (int c : chain) fused.insert(c);
        fusedPlan.emplace(int(i), std::make_pair(std::move(chain), srcPort));
    }

    for (size_t i = firstDirty; i < m_stages.size(); ++i) {
        // Between stages as well as within them: a pipeline of several
        // moderately slow stages should abandon at the next boundary even if no
        // single algorithm polls the token itself.
        if (cancel && cancel->Cancelled()) { *err = kCancelled; return false; }

        Stage& s = m_stages[i];

        // Already consumed frame by frame by a fused reduction below.
        if (fused.count(int(i))) continue;

        // Stage-level progress. Counted from firstDirty rather than from zero,
        // so a cache hit that skips most of the pipeline does not show a bar
        // crawling through work that is not happening.
        if (progress) {
            progress->Set(int(i - firstDirty), int(m_stages.size() - firstDirty),
                          s.algoName.c_str());
            // Tallies as of the stages ALREADY finished, so the numbers on
            // screen always describe completed work rather than the stage
            // currently running. Published here, before the stage runs, for
            // exactly that reason: a count incremented mid-stage would claim
            // work that has not happened yet.
            PublishStats(progress, gpu);
        }

        s.valid = false;

        // A reduction whose input is a straight line of broadcast stages runs
        // them per FRAME rather than per stage, so one frame is live at a time
        // instead of all N. Checked before the inputs are gathered, because
        // gathering is what would materialise the whole intermediate set.
        if (auto fp = fusedPlan.find(int(i)); fp != fusedPlan.end()) {
            if (!RunFusedReduction(int(i), fp->second.first, fp->second.second,
                                   sources, gpu, mode, cancel, progress, err))
                return false;
            continue;
        }

        // Gather inputs.
        std::vector<const Data*> in;
        in.reserve(s.inputs.size());
        for (const PortRef& r : s.inputs) {
            const Data* d = Resolve(r, sources);
            if (!d || TypeOf(*d) == DataType::None) {
                *err = "line " + std::to_string(s.line) + ": '" + s.algoName +
                       "' has an input that produced no data";
                return false;
            }
            in.push_back(d);
        }
        // An aligner annotates the whole set at once: every frame is solved
        // against a common reference, so it cannot run one frame at a time.
        // Handled here for the same reason as a reshape.
        //
        // It copies the group and attaches transforms to the copy rather than
        // warping anything. Nothing downstream is obliged to use them -- a merge
        // that samples through TransformOf() gets aligned pixels, and one that
        // does not behaves exactly as before.
        if (s.algo->IsAligner()) {
            const auto* src = std::get_if<ImageSet>(in[0]);
            if (!src) {
                *err = "line " + std::to_string(s.line) +
                       ": align() needs a group, not a single image";
                return false;
            }
            ImageSet out;
            out.shape = src->shape;
            out.images.reserve(src->images.size());
            for (const Image& im : src->images) out.images.push_back(im.Clone());

            std::string aerr;
            if (!s.algo->RunAlign(&out.images, &aerr)) {
                *err = "line " + std::to_string(s.line) + ": " + aerr;
                return false;
            }
            s.outputs.clear();
            s.outputs.resize(1);
            s.outputs[0] = Data{std::move(out)};
            s.valid = true;
            continue;
        }


        // A reshape moves no pixels: it hands its input through with a
        // different shape. Done here rather than in RunCPU because that gets a
        // per-image view, and this is a fact about the whole set.
        if (s.algo->IsReshape()) {
            const auto* src = std::get_if<ImageSet>(in[0]);
            if (!src) {
                *err = "line " + std::to_string(s.line) +
                       ": shape() needs a group, not a single image";
                return false;
            }
            if (s.reshapeTo.Count() != int64_t(src->images.size())) {
                *err = "line " + std::to_string(s.line) + ": shape() describes " +
                       std::to_string(s.reshapeTo.Count()) + " images (" +
                       s.reshapeTo.ToString() + ") but the group holds " +
                       std::to_string(src->images.size());
                return false;
            }
            ImageSet out;
            out.shape = s.reshapeTo;
            out.images.reserve(src->images.size());
            for (const Image& im : src->images) out.images.push_back(im.Clone());
            s.outputs.clear();
            s.outputs.resize(1);
            s.outputs[0] = Data{std::move(out)};
            s.valid = true;
            continue;
        }


        // A reduction runs its own way: stream the frames through an
        // accumulator rather than allocating an output from input 0 and calling
        // RunCPU. Handled here, before the ordinary path, because almost none
        // of that path applies -- the output's size comes from the accumulator,
        // not from an input, and there is no single input image to size it by.
        if (s.algo->IsReduction()) {
            if (!RunReduction(s, in, cancel, err)) return false;
            s.valid = true;
            continue;
        }

        // A set on an input that accepts one image means the framework maps the
        // stage across the frames -- the broadcasting rule. Checked here rather
        // than in the interpreter because it depends on what actually arrived.
        bool anySet = false;
        for (const Data* d : in) if (TypeOf(*d) == DataType::ImageSet) anySet = true;

        if (anySet) {
            if (!BroadcastStage(s, in, gpu, mode, cancel, progress, err)) return false;
        } else if (!RunStageOnce(s, in, gpu, mode, cancel, err)) {
            return false;
        }
        s.valid = true;
    }

    // Submit explicitly on the success path so a failure here is *reported*.
    // The guard above would flush anyway, but it deliberately swallows the
    // error -- on an early return the caller already has a better one, whereas
    // here this is the only thing that went wrong. Flushing twice is harmless:
    // the second call finds nothing pending.
    if (gpu && gpu->Ready()) {
        std::string flushErr;
        if (!gpu->Flush(&flushErr)) {
            *err = flushErr;
            return false;
        }
    }

    // The last stage's contribution. Every publish above describes the stages
    // finished BEFORE it, so without this the final stage never appears in the
    // live tally -- on a one-stage pipeline the counts would stay at zero for
    // the entire run.
    if (progress)
        PublishStats(progress, gpu);

    return true;
}

bool Pipeline::RunStageGpu(Stage& s, const std::vector<const Data*>& in,
                           ComputeContext* gpu, std::string* err) {
    // M3 supports the common shape: image inputs and image outputs, one
    // dispatch. Anything else stays on the CPU.
    for (const Data* d : in)
        if (!d || !std::holds_alternative<Image>(*d)) { *err = "non-image input"; return false; }
    for (const Data& d : s.outputs)
        if (!std::holds_alternative<Image>(d)) { *err = "non-image output"; return false; }

    // Hand the algorithm its input descriptors before anything is bound. A
    // demosaic needs the CFA pattern and sensor levels, which live on the
    // image rather than on a parameter, and RunCPU -- the usual place to read
    // them -- never runs on this path.
    {
        std::vector<ImageDesc> inDescs;
        inDescs.reserve(in.size());
        for (const Data* d : in)
            inDescs.push_back(std::get<Image>(*d).Desc());
        s.algo->PrepareGpu(inDescs);
    }

    // Kernels are compiled once per stage and cached on the stage, so dragging
    // a slider does not recompile HLSL every frame.
    const std::vector<AlgorithmBase::GpuPass> passes = s.algo->GpuPasses();

    if (!passes.empty()) {
        if (s.passKernels.size() != passes.size()) {
            s.passKernels.clear();
            for (const AlgorithmBase::GpuPass& p : passes) {
                auto k = std::make_shared<ComputeKernel>();
                std::string compileErr;
                const std::string label = s.algoName + "." + p.name;
                if (!gpu->CreateKernel(p.source, "main", label, k.get(), &compileErr)) {
                    s.passKernels.clear();
                    *err = compileErr;
                    return false;
                }
                s.passKernels.push_back(std::move(k));
            }
        }
    } else if (!s.kernel) {
        s.kernel = std::make_shared<ComputeKernel>();
        std::string compileErr;
        if (!gpu->CreateKernel(s.algo->GpuSource(), "main", s.algoName,
                               s.kernel.get(), &compileErr)) {
            s.kernel.reset();
            *err = compileErr;
            return false;
        }
    }

    // Acquire through Image, which owns its GPU resource and only transfers
    // when a side is stale. Chained GPU stages therefore upload once at the
    // head and never read back in the middle: the intermediate images stay
    // GPU-resident, and nothing asks for their CPU pixels.
    std::vector<const GpuImage*> gin;
    std::vector<GpuImage*>       gout;
    gin.reserve(in.size());
    gout.reserve(s.outputs.size());

    const auto tAcq = std::chrono::steady_clock::now();
    for (const Data* d : in) {
        Image& img = const_cast<Image&>(std::get<Image>(*d));
        GpuResidency* g = img.AcquireGpuRead(*gpu);
        if (!g) { *err = "could not make an input GPU-resident"; return false; }
        gin.push_back(&g->image);
    }
    if (GpuPassTiming())
        std::fprintf(stderr, "[pass] %s.<input upload> %.0f ms\n", s.algoName.c_str(),
                     std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - tAcq).count());
    for (Data& d : s.outputs) {
        Image& img = std::get<Image>(d);
        GpuResidency* g = img.AcquireGpuWrite(*gpu);
        if (!g) { *err = "could not allocate a GPU output"; return false; }
        gout.push_back(&g->image);
    }

    // TGLAB_GPUDBG=1 logs every dispatch. Worth keeping: when the device
    // hung after a script switch, the absence of these lines is what proved
    // the fault was not in compute at all, but in freeing view textures.
    if (GetEnvironmentVariableA("TGLAB_GPUDBG", nullptr, 0) > 0) {
        const ImageDesc& od = std::get<Image>(s.outputs[0]).Desc();
        std::fprintf(stderr, "[gpu] dispatch %s %dx%d\n",
                     s.algoName.c_str(), od.width, od.height);
        std::fflush(stderr);
    }
    // Multi-pass: several DIFFERENT kernels over a shared pool of scratch
    // planes. See AlgorithmBase::GpuPasses.
    if (!passes.empty()) {
        const ImageDesc outDesc = std::get<Image>(s.outputs[0]).Desc();
        ImageDesc planeDesc = outDesc;
        switch (s.algo->GpuScratchPlanes()) {
            case FormatSpec::RGBA8:   planeDesc.format = Format::RGBA8;   break;
            case FormatSpec::R32F:    planeDesc.format = Format::R32F;    break;
            case FormatSpec::RGBA32F: planeDesc.format = Format::RGBA32F; break;
            case FormatSpec::RGBA16F: planeDesc.format = Format::RGBA16F; break;
            case FormatSpec::Any:
            case FormatSpec::SameAsInput: break;
        }
        // The planes are intermediates, never displayed, so they carry no CFA
        // and no transfer function. Leaving the input's CFA on them would make
        // a later stage think it still had a mosaic to demosaic.
        planeDesc.cfa    = CfaPattern::None;
        planeDesc.linear = true;

        const int planeCount = std::max(0, s.algo->GpuScratchCount());
        const bool resized = s.planeDesc.width  != planeDesc.width  ||
                             s.planeDesc.height != planeDesc.height ||
                             s.planeDesc.format != planeDesc.format;
        if (int(s.gpuPlanes.size()) != planeCount || resized) {
            s.gpuPlanes.clear();
            for (int i = 0; i < planeCount; ++i) {
                auto p = std::make_shared<GpuImage>();
                if (!gpu->CreateImage(planeDesc, p.get())) {
                    s.gpuPlanes.clear();
                    *err = "could not allocate scratch planes for a multi-pass stage";
                    return false;
                }
                s.gpuPlanes.push_back(std::move(p));
            }
            s.planeDesc = planeDesc;
        }

        // Resolve a declared index to a real buffer. Negative addresses the
        // stage's own ports, non-negative the scratch pool. Checked rather than
        // trusted: an out-of-range index would otherwise be a wild pointer, and
        // the failure mode of a bad binding is a silent race.
        auto resolveRead = [&](int idx, const GpuImage** out) {
            if (idx < 0) {
                const size_t port = size_t(-idx - 1);
                if (port >= gin.size()) return false;
                *out = gin[port];
                return true;
            }
            if (size_t(idx) >= s.gpuPlanes.size()) return false;
            *out = s.gpuPlanes[size_t(idx)].get();
            return true;
        };
        auto resolveWrite = [&](int idx, GpuImage** out) {
            if (idx < 0) {
                const size_t port = size_t(-idx - 1);
                if (port >= gout.size()) return false;
                *out = gout[port];
                return true;
            }
            if (size_t(idx) >= s.gpuPlanes.size()) return false;
            *out = s.gpuPlanes[size_t(idx)].get();
            return true;
        };

        // Drain anything already recorded (the input upload, an earlier stage)
        // so the per-pass timings below measure only their own dispatch.
        if (GpuPassTiming()) {
            const auto tu = std::chrono::steady_clock::now();
            std::string ue;
            gpu->Flush(&ue);
            std::fprintf(stderr, "[pass] %s.<upload+prior> %.0f ms\n", s.algoName.c_str(),
                         std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - tu).count());
        }

        for (size_t p = 0; p < passes.size(); ++p) {
            std::vector<const GpuImage*> passIn;
            std::vector<GpuImage*>       passOut;
            for (int idx : passes[p].reads) {
                const GpuImage* g = nullptr;
                if (!resolveRead(idx, &g)) {
                    *err = "pass '" + std::string(passes[p].name) +
                           "' reads a buffer that does not exist";
                    return false;
                }
                passIn.push_back(g);
            }
            for (int idx : passes[p].writes) {
                GpuImage* g = nullptr;
                if (!resolveWrite(idx, &g)) {
                    *err = "pass '" + std::string(passes[p].name) +
                           "' writes a buffer that does not exist";
                    return false;
                }
                // A pass that reads and writes the same buffer is a race: the
                // threads have no ordering, so some read the old value and some
                // the new. It shows as noise on one GPU and not another, which
                // is the worst kind of bug to chase.
                for (const GpuImage* r : passIn)
                    if (r == g) {
                        *err = "pass '" + std::string(passes[p].name) +
                               "' reads and writes the same buffer";
                        return false;
                    }
                passOut.push_back(g);
            }
            if (!gpu->Dispatch(*s.passKernels[p], passIn, passOut,
                               s.algo->GpuPassConstants(int(p)), err))
                return false;

            // Diagnostic: submit and wait per pass, so a hang can be attributed
            // to ONE kernel rather than to the batch. Far slower than batching --
            // this is for finding a watchdog overrun, not for normal use.
            if (GpuPassTiming()) {
                const auto t0 = std::chrono::steady_clock::now();
                std::string fe;
                const bool fok = gpu->Flush(&fe);
                const double ms = std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - t0).count();
                std::fprintf(stderr, "[pass] %s.%s %.0f ms%s\n",
                             s.algoName.c_str(), passes[p].name, ms,
                             fok ? "" : "  <-- FAILED HERE");
                if (!fok) { *err = fe; return false; }
            }
        }
        return true;
    }

    const int iterations = std::max(1, s.algo->GpuIterations());

    if (iterations == 1) {
        const auto tRec = std::chrono::steady_clock::now();
        if (!gpu->Dispatch(*s.kernel, gin, gout, s.algo->GpuConstants(0), err)) return false;
        if (GpuPassTiming()) {
            const double recMs = std::chrono::duration<double, std::milli>(
                                     std::chrono::steady_clock::now() - tRec).count();
            const auto tFl = std::chrono::steady_clock::now();
            std::string fe;
            const bool fok = gpu->Flush(&fe);
            std::fprintf(stderr, "[pass] %s record %.0f ms, gpu %.0f ms%s\n",
                         s.algoName.c_str(), recMs,
                         std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - tFl).count(),
                         fok ? "" : "  <-- FAILED");
            if (!fok) { *err = fe; return false; }
        }
        // No readback here. Outputs stay GPU-resident; whoever needs CPU pixels
        // (a later CPU stage, or a viewer) triggers the transfer via
        // MapCpuRead().
        return true;
    }

    // Iterative: ping-pong between the real output and one scratch image.
    //
    // Each pass must read the *completed* previous pass, which a single
    // dispatch cannot provide -- threads within one dispatch have no ordering
    // relative to each other. Writing into the image being read would be a
    // race, so two buffers alternate.
    //
    // Only single-input, single-output stages iterate; anything else has no
    // obvious pairing to alternate between.
    if (gin.size() != 1 || gout.size() != 1) {
        *err = "iterative GPU stages must have exactly one input and one output";
        return false;
    }

    // The scratch may deliberately differ from the output: see
    // AlgorithmBase::GpuScratchFormat. Only the format changes -- dimensions
    // always match, since every pass covers the same pixel grid.
    ImageDesc desc = std::get<Image>(s.outputs[0]).Desc();
    switch (s.algo->GpuScratchFormat()) {
        case FormatSpec::RGBA8:   desc.format = Format::RGBA8;   break;
        case FormatSpec::R32F:    desc.format = Format::R32F;    break;
        case FormatSpec::RGBA32F: desc.format = Format::RGBA32F; break;
        case FormatSpec::RGBA16F: desc.format = Format::RGBA16F; break;
        case FormatSpec::Any:
        case FormatSpec::SameAsInput:
            break;   // keep the output's format
    }
    // A scratch that differs from the output is only coherent when the passes
    // alternate exactly once: intermediate in scratch, result in output. With
    // three or more passes the ping-pong would write the rich intermediate into
    // the single-channel output and lose it -- the same silent truncation this
    // hook exists to prevent, just moved. Refuse rather than corrupt.
    if (desc.format != std::get<Image>(s.outputs[0]).Desc().format && iterations != 2) {
        *err = "a GPU stage with its own scratch format must use exactly 2 passes";
        return false;
    }

    if (!s.gpuScratch) {
        s.gpuScratch = std::make_shared<GpuImage>();
        if (!gpu->CreateImage(desc, s.gpuScratch.get())) {
            s.gpuScratch.reset();
            *err = "could not allocate the scratch image for an iterative stage";
            return false;
        }
        s.scratchDesc = desc;
    } else if (s.scratchDesc.width != desc.width || s.scratchDesc.height != desc.height ||
               s.scratchDesc.format != desc.format) {
        // The image changed size, so the cached scratch no longer matches.
        s.gpuScratch = std::make_shared<GpuImage>();
        if (!gpu->CreateImage(desc, s.gpuScratch.get())) {
            s.gpuScratch.reset();
            *err = "could not resize the scratch image for an iterative stage";
            return false;
        }
        s.scratchDesc = desc;
    }

    // Choose the starting parity so the LAST pass lands in the real output;
    // otherwise the result would sit in scratch and be silently discarded.
    // With an odd count, start by writing into the output; with an even count,
    // start in scratch so the final flip ends up there.
    const GpuImage* readFrom = gin[0];
    GpuImage*       writeTo  = (iterations % 2 == 1) ? gout[0] : s.gpuScratch.get();

    for (int i = 0; i < iterations; ++i) {
        // t0 is the ping-pong buffer; t1 is the ORIGINAL input, bound on every
        // pass.
        //
        // A separable filter's last pass usually needs both: the accumulated
        // result AND the untouched source. A local threshold compares each pixel
        // against a blurred neighbourhood, so with only t0 the final pass has
        // the neighbourhood but no longer the pixel. Binding the source costs
        // one SRV slot of the four and nothing at all to a kernel that ignores
        // it -- anisotropic_diffusion, the only other iterative algorithm, reads
        // t0 alone and is unaffected.
        const std::vector<const GpuImage*> passIn{readFrom, gin[0]};
        const std::vector<GpuImage*>       passOut{writeTo};
        if (!gpu->Dispatch(*s.kernel, passIn, passOut, s.algo->GpuConstants(i), err))
            return false;

        // This pass's output becomes the next pass's input, and the buffer just
        // read becomes the next target -- except on the first pass, where the
        // input is the upstream image and must not be overwritten.
        // Submit before the next pass, which READS what this one just wrote.
        //
        // Dispatch()'s UAV barrier orders write-against-write on the same
        // binding; it does not order a write followed by a read through an SRV.
        // A state-transition barrier does not help either -- transitioning out
        // and back gives the driver nothing to synchronise against -- so the
        // ordering has to come from a submit. That was free while every
        // dispatch flushed, and became a garbage image the moment they started
        // batching.
        //
        // The cost is bounded: only iterative stages pay it, and only between
        // their own passes. Ordinary chains still batch, which is where the
        // measured win came from.
        if (i + 1 < iterations) {
            std::string fe;
            if (!gpu->Flush(&fe)) { *err = fe; return false; }
        }

        readFrom = writeTo;
        writeTo  = (writeTo == gout[0]) ? s.gpuScratch.get() : gout[0];
    }

    return true;
}

} // namespace tglab
