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
    RunCtx(std::span<const Data* const> in, std::span<Data> out)
        : m_in(in), m_out(out) {}

    // Image convenience accessors (M1 has only image ports).
    ImageView In(size_t i) const;
    ImageView Out(size_t i) const;

    const ImageDesc& InDesc(size_t i) const;

    size_t NumIn()  const { return m_in.size(); }
    size_t NumOut() const { return m_out.size(); }

private:
    std::span<const Data* const> m_in;
    std::span<Data>              m_out;
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

    virtual bool HasGPU() const { return false; }
    // M3: virtual void RunGPU(GpuRunCtx&) {}

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
