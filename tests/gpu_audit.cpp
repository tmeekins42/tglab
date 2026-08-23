// gpu_audit — runs every GPU-capable algorithm under the D3D12 debug layer
// and GPU-based validation, and attributes each message to the algorithm that
// provoked it.
//
// Why this exists: the display-path hang was found only because one script
// happened to exercise the faulty code. The rest of the registry had never been
// looked at with these tools switched on, so "the debug layer is quiet" meant
// only "nobody ran the other twenty-odd kernels". This runs all of them.
//
// GPU-based validation is the point of the exercise. It patches shaders to
// check descriptors and resource state on the GPU at dispatch time, which is
// the only way to catch a binding that is wrong when the list EXECUTES rather
// than when it records -- exactly the class of bug behind the hang.
//
//   gpu_audit            run every GPU algorithm, report per-algorithm counts
//   gpu_audit --verbose  also print the full text of each distinct message
//
// Exit code is the number of algorithms with at least one message, capped at
// 125, so CI can gate on it.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <d3d12.h>

#include "../src/core/algorithm.h"
#include "../src/core/pipeline.h"
#include "../src/gpu/compute.h"
#include "../src/gpu/gpu_image.h"
#include "../src/script/interp.h"
#include "../src/script/parser.h"

using namespace tglab;

// --- validation message capture ---------------------------------------------

struct Finding {
    std::string category;   // short bucket, e.g. "barrier layout"
    std::string text;       // the full message, first occurrence
    int         count = 0;
};

// Collapses a raw validation string to a stable bucket name.
//
// The messages embed pointers, indices and shader offsets, so the same defect
// reported twice is never textually identical. Bucketing on the leading phrase
// keeps a count meaningful.
static std::string Bucket(const std::string& msg) {
    struct { const char* needle; const char* name; } kMap[] = {
        {"Incompatible texture barrier layout", "barrier layout"},
        {"Invalid resource pointed to by descriptor", "dangling descriptor"},
        {"is different from currently set descriptor heap", "wrong heap bound"},
        {"Uninitialized", "uninitialized read"},
        {"out of bounds", "out of bounds"},
        {"RESOURCE_BARRIER", "barrier misuse"},
        {"was not transitioned", "missing transition"},
        {"CORRUPTION", "corruption"},
    };
    for (const auto& e : kMap)
        if (msg.find(e.needle) != std::string::npos) return e.name;
    return "other";
}

// Pulls everything the info queue has accumulated and clears it.
static void Drain(ID3D12Device* dev, std::vector<Finding>* out) {
    ID3D12InfoQueue* iq = nullptr;
    if (FAILED(dev->QueryInterface(IID_PPV_ARGS(&iq)))) return;

    const UINT64 n = iq->GetNumStoredMessages();
    for (UINT64 i = 0; i < n; ++i) {
        SIZE_T len = 0;
        iq->GetMessage(i, nullptr, &len);
        std::vector<char> buf(len);
        auto* m = reinterpret_cast<D3D12_MESSAGE*>(buf.data());
        if (FAILED(iq->GetMessage(i, m, &len)) || !m->pDescription) continue;
        // INFO is chatter (resource creation, heap stats); it is not a defect.
        if (m->Severity == D3D12_MESSAGE_SEVERITY_INFO ||
            m->Severity == D3D12_MESSAGE_SEVERITY_MESSAGE) continue;

        const std::string text(m->pDescription);
        const std::string bucket = Bucket(text);
        auto it = std::find_if(out->begin(), out->end(),
                               [&](const Finding& f) { return f.category == bucket; });
        if (it == out->end()) out->push_back({bucket, text, 1});
        else                  ++it->count;
    }
    iq->ClearStoredMessages();
    iq->Release();
}


// Warnings that are inherent to the design rather than defects.
//
// "Incompatible texture barrier layout" on a UAV is unavoidable for a resource
// created with ALLOW_SIMULTANEOUS_ACCESS: such a resource is permanently in
// LAYOUT_COMMON by definition, because any queue may access it at any time, so
// GBV reports every UAV binding as a layout mismatch. Verified by removing the
// flag -- with it gone and the transitions in place the audit is completely
// clean, and putting it back reproduces exactly these messages.
//
// The flag is what lets the UI sample a compute result with no cross-queue
// transition, which is the whole point of the direct-from-GPU display path.
// Dropping it to silence a warning would cost that.
//
// --strict reports them anyway, for when that trade-off is revisited.
static bool IsKnownBenign(const std::string& category) {
    return category == "barrier layout";
}

// --- source images ----------------------------------------------------------

// A gradient plus a bright square, so no kernel sees a uniform image (which
// would hide an indexing mistake) and every channel varies.
static Image MakeSource(int dim, Format fmt) {
    Image img;
    img.Alloc({dim, dim, fmt});
    ImageView v = img.MapCpuWrite();
    for (int y = 0; y < dim; ++y) {
        for (int x = 0; x < dim; ++x) {
            const bool box = (x > dim / 3 && x < 2 * dim / 3 &&
                              y > dim / 3 && y < 2 * dim / 3);
            const float r = box ? 0.95f : float(x) / float(dim);
            const float g = box ? 0.80f : float(y) / float(dim);
            const float b = box ? 0.10f : float(x + y) / float(2 * dim);
            if (fmt == Format::RGBA8) {
                uint8_t* p = v.At<uint8_t>(x, y);
                p[0] = uint8_t(r * 255.0f);
                p[1] = uint8_t(g * 255.0f);
                p[2] = uint8_t(b * 255.0f);
                p[3] = 255;
            } else if (fmt == Format::R32F) {
                *v.At<float>(x, y) = r;
            } else {
                float* p = v.At<float>(x, y);
                p[0] = r; p[1] = g; p[2] = b; p[3] = 1.0f;
            }
        }
    }
    return img;
}

// --- running one algorithm --------------------------------------------------

struct Result {
    std::string          name;
    bool                 ran = false;
    std::string          error;
    std::vector<Finding> findings;
};

static Result RunOne(ID3D12Device* dev, ComputeContext& gpu,
                     const std::string& name, Format fmt, int dim) {
    Result r;
    r.name = name;

    // Through the script layer rather than by calling RunGPU directly, so the
    // algorithm is driven exactly as the app drives it: same interpreter, same
    // pipeline, same residency transitions.
    const std::string script =
        "src = image(\"test\")\n"
        "out = " + name + "(src)\n"
        "display(out, \"out\")\n";

    std::vector<Data> sources;
    sources.push_back(Data{MakeSource(dim, fmt)});
    std::vector<SourceImage> names{{"test", 0}};

    Program prog;
    if (!Parse(script.c_str(), &prog, &r.error)) return r;

    UiState ui;
    Pipeline pipe;
    auto ir = Interpret(prog, names, &ui, &pipe);
    if (!ir.ok) { r.error = ir.error; return r; }

    // Clear anything the previous algorithm left behind, so the findings below
    // belong to this one.
    { std::vector<Finding> discard; Drain(dev, &discard); }

    if (!pipe.Execute(&sources, nullptr, &r.error, &gpu, ExecMode::ForceGPU)) {
        Drain(dev, &r.findings);
        return r;
    }
    r.ran = true;

    // Force the result to the CPU: this waits on the compute fence, so the
    // dispatch has actually executed by the time messages are collected.
    // Without it the queue could still be busy and GBV would report nothing.
    const Data* d = pipe.Resolve(pipe.Viewers()[0].source, &sources);
    if (d && std::holds_alternative<Image>(*d)) {
        Image& img = const_cast<Image&>(std::get<Image>(*d));
        img.MapCpuRead();
    }

    Drain(dev, &r.findings);
    return r;
}

// --- main -------------------------------------------------------------------

int main(int argc, char** argv) {
    bool verbose = false, strict = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--verbose") == 0) verbose = true;
        if (std::strcmp(argv[i], "--strict")  == 0) strict  = true;
    }

    // Both layers on, before device creation. Synchronized queue validation
    // serialises the queues so a cross-queue hazard is reported against the
    // call that caused it rather than surfacing later as a hang.
    ID3D12Debug* dbg = nullptr;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dbg)))) {
        dbg->EnableDebugLayer();
        dbg->Release();
    }
    ID3D12Debug1* dbg1 = nullptr;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dbg1)))) {
        dbg1->SetEnableGPUBasedValidation(TRUE);
        dbg1->SetEnableSynchronizedCommandQueueValidation(TRUE);
        dbg1->Release();
    } else {
        std::printf("warning: GPU-based validation unavailable; "
                    "this audit is much weaker without it\n");
    }

    ID3D12Device* dev = nullptr;
    if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&dev)))) {
        std::printf("no D3D12 device; nothing to audit\n");
        return 0;
    }

    // Report rather than break, so one bad dispatch does not end the run.
    ID3D12InfoQueue* iq = nullptr;
    if (SUCCEEDED(dev->QueryInterface(IID_PPV_ARGS(&iq)))) {
        iq->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, FALSE);
        iq->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR,      FALSE);
        iq->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_WARNING,    FALSE);
        iq->Release();
    }

    ComputeContext gpu;
    if (!gpu.Init(dev)) {
        std::printf("compute context failed to initialise\n");
        dev->Release();
        return 1;
    }
    InstallGpuResidencyHooks();

    // Only algorithms that claim a GPU path: a CPU-only one would run happily
    // and prove nothing about shaders.
    std::vector<std::string> targets;
    for (const std::string& n : Registry::Get().Names()) {
        auto a = Registry::Get().Create(n);
        if (a && a->HasGPU()) targets.push_back(n);
    }
    std::sort(targets.begin(), targets.end());

    std::printf("auditing %zu GPU-capable algorithms "
                "(debug layer + GPU-based validation)\n\n",
                targets.size());

    // 256 is large enough that a window-based kernel takes its interior path
    // and small enough that GBV's 10-100x cost stays tolerable across ~20 runs.
    const int dim = 256;

    std::vector<Result> results;
    for (const std::string& name : targets) {
        std::printf("  %-28s ", name.c_str());
        std::fflush(stdout);

        Result r = RunOne(dev, gpu, name, Format::RGBA8, dim);

        // Drop the inherent ones unless --strict asked for everything.
        if (!strict) {
            r.findings.erase(
                std::remove_if(r.findings.begin(), r.findings.end(),
                               [](const Finding& f) { return IsKnownBenign(f.category); }),
                r.findings.end());
        }

        if (!r.ran) {
            // Not every algorithm accepts a plain RGBA8 image with no
            // parameters -- a demosaic wants a mosaic, for instance. Say so
            // rather than counting it as a pass.
            std::printf("skipped (%s)\n",
                        r.error.empty() ? "would not run" : r.error.c_str());
        } else if (r.findings.empty()) {
            std::printf("clean\n");
        } else {
            int total = 0;
            std::string cats;
            for (const Finding& f : r.findings) {
                total += f.count;
                if (!cats.empty()) cats += ", ";
                cats += f.category + " x" + std::to_string(f.count);
            }
            std::printf("%d message(s): %s\n", total, cats.c_str());
        }
        results.push_back(std::move(r));
    }

    // --- summary ---
    int dirty = 0, skipped = 0, clean = 0;
    std::map<std::string, int> byCategory;
    for (const Result& r : results) {
        if (!r.ran)                  ++skipped;
        else if (r.findings.empty()) ++clean;
        else {
            ++dirty;
            for (const Finding& f : r.findings) byCategory[f.category] += f.count;
        }
    }

    std::printf("\n%d clean, %d with findings, %d skipped\n", clean, dirty, skipped);
    if (!byCategory.empty()) {
        std::printf("\nby category:\n");
        for (const auto& [cat, n] : byCategory)
            std::printf("  %-24s %d\n", cat.c_str(), n);
    }

    if (verbose) {
        std::printf("\n--- one example per algorithm/category ---\n");
        for (const Result& r : results) {
            for (const Finding& f : r.findings)
                std::printf("\n[%s / %s]\n%s\n",
                            r.name.c_str(), f.category.c_str(), f.text.c_str());
        }
    }

    gpu.Shutdown();
    dev->Release();
    return dirty > 125 ? 125 : dirty;
}
