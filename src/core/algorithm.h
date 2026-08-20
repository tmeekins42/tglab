// AlgorithmBase: the interface every algorithm derives from.
//
// Ports and parameters are both *declared* rather than implied by a signature,
// so the script, the UI, and the stage cache all discover them the same way.
#pragma once

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "cancel.h"
#include "data.h"
#include "image.h"
#include "param.h"

namespace tglab {

struct Port {
    const char* name;
    DataType    type   = DataType::Image;
    FormatSpec  format = FormatSpec::Any;   // only meaningful for Image ports
};

using PortList = std::vector<Port>;

// Handed to an algorithm at run time. Indices match the declared port order.
class RunCtx {
public:
    RunCtx(std::span<const Data* const> in, std::span<Data> out,
           const CancelToken* cancel = nullptr)
        : m_in(in), m_out(out), m_cancel(cancel) {}

    // True when this run has been superseded and should stop.
    //
    // A long algorithm should check this every row or two and return early.
    // Returning with a partly-written output is safe: the pipeline marks a
    // cancelled stage invalid, so the partial result is never displayed or
    // cached, and the newer run recomputes it from scratch.
    bool Cancelled() const { return m_cancel && m_cancel->Cancelled(); }

    // Image convenience accessors (M1 has only image ports).
    ImageView In(size_t i) const;
    ImageView Out(size_t i) const;

    const ImageDesc& InDesc(size_t i) const;

    // Raw port data, for composite algorithms that forward an input straight
    // into a sub-algorithm (see canny.cpp).
    const Data* InData(size_t i) const { return i < m_in.size() ? m_in[i] : nullptr; }

    size_t NumIn()  const { return m_in.size(); }
    size_t NumOut() const { return m_out.size(); }

private:
    std::span<const Data* const> m_in;
    std::span<Data>              m_out;
    const CancelToken*           m_cancel = nullptr;
};

class AlgorithmBase {
public:
    AlgorithmBase()          = default;
    virtual ~AlgorithmBase() = default;

    // MUST stay deleted: this object holds raw pointers to its own Param
    // members, so a default copy/move would leave them dangling into the
    // old object. Algorithms are always heap-allocated via the registry.
    AlgorithmBase(const AlgorithmBase&)            = delete;
    AlgorithmBase& operator=(const AlgorithmBase&) = delete;
    AlgorithmBase(AlgorithmBase&&)                 = delete;
    AlgorithmBase& operator=(AlgorithmBase&&)      = delete;

    virtual const char* Name() const = 0;

    // Groups interchangeable implementations (e.g. "quantize"). Used by the
    // UI and, from M2, by choose() to offer every algorithm in a category.
    virtual const char* Category() const { return ""; }

    virtual PortList Inputs()  const = 0;
    virtual PortList Outputs() const = 0;

    virtual void RunCPU(RunCtx& ctx) = 0;

    // --- GPU path (M3) ------------------------------------------------------
    // An algorithm opts in by returning true from HasGPU() and providing the
    // HLSL for a compute kernel. The framework handles residency, descriptor
    // binding and dispatch, so an algorithm only writes the kernel itself.
    virtual bool HasGPU() const { return false; }

    // Compute shader source with a `main` entry point. Bindings by convention:
    //   t0..t3  inputs        u0..u3  outputs
    //   b0      uint Width, uint Height, then GpuConstants() below
    virtual const char* GpuSource() const { return nullptr; }

    // Extra root constants, bit-cast to uint. Floats go through asfloat() in
    // the shader. Order must match the cbuffer declaration.
    //
    // `iteration` is the 0-based pass index, for iterative algorithms; it is 0
    // for everything else. A kernel that behaves the same every pass can ignore
    // it entirely.
    virtual std::vector<uint32_t> GpuConstants(int iteration = 0) const {
        (void)iteration;
        return {};
    }

    // How many times to dispatch the kernel, feeding each pass's output back in
    // as the next pass's input. 1 (the default) is the ordinary single-dispatch
    // case.
    //
    // Iterative schemes -- Perona-Malik being the motivating one -- cannot be
    // expressed as a single dispatch: each step must see the *completed*
    // previous step, and threads within one dispatch have no such ordering. The
    // framework ping-pongs between two GPU images and hands back whichever
    // holds the final result, so the algorithm only writes one pass.
    virtual int GpuIterations() const { return 1; }

    std::span<ParamBase* const> Params() const { return m_params; }
    ParamBase* FindParam(std::string_view name) const;

    // Folds every parameter value into one hash for dirty detection.
    uint64_t ParamHash() const;

private:
    friend class ParamBase;
    std::vector<ParamBase*> m_params;
};

// ---------------------------------------------------------------------------
// Registry — self-registration, so adding an algorithm touches no central file.

class Registry {
public:
    using Factory = std::unique_ptr<AlgorithmBase> (*)();

    static Registry& Get();

    void Add(const char* name, Factory f);
    std::unique_ptr<AlgorithmBase> Create(std::string_view name) const;
    bool Contains(std::string_view name) const;

    std::vector<std::string> Names() const;
    std::vector<std::string> NamesInCategory(std::string_view category) const;

private:
    struct Entry {
        std::string name;
        Factory     factory;
    };
    std::vector<Entry> m_entries;
};

#define REGISTER_ALGORITHM(T)                                                  \
    namespace {                                                                \
    struct T##_Registrar {                                                     \
        T##_Registrar() {                                                      \
            ::tglab::Registry::Get().Add(                                      \
                T{}.Name(),                                                    \
                []() -> std::unique_ptr<::tglab::AlgorithmBase> {              \
                    return std::make_unique<T>();                              \
                });                                                            \
        }                                                                      \
    };                                                                         \
    T##_Registrar g_##T##_registrar;                                           \
    }

} // namespace tglab
