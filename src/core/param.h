// Param<T>: the single storage location for an algorithm parameter.
//
// The script writes it, the inspector widget writes it, the algorithm reads it.
// Nothing is ever copied between homes, so there is no sync problem to solve.
//
// Params self-collect: ParamBase's constructor pushes itself onto the owner's
// list, so declaring a member *is* declaring the descriptor.
#pragma once

#include <algorithm>
#include <cstdint>
#include <string>

namespace tglab {

class AlgorithmBase;
class Value;

enum class ParamType : uint8_t { Float, Int, Bool };

const char* ParamTypeName(ParamType t);

class ParamBase {
public:
    ParamBase(AlgorithmBase* owner, const char* name);
    virtual ~ParamBase() = default;

    // Non-copyable: the owner holds raw pointers to these members.
    ParamBase(const ParamBase&)            = delete;
    ParamBase& operator=(const ParamBase&) = delete;

    virtual ParamType Type() const = 0;

    // Returns true if the value actually changed (drives the dirty hash).
    virtual bool SetFromScript(const Value& v, std::string* err) = 0;
    virtual bool DrawWidget() = 0;

    // Folded into Stage::paramHash for dirty detection.
    virtual uint64_t HashValue() const = 0;

    const char* Name() const { return m_name; }

protected:
    const char* m_name;
};

template <class T>
class Param : public ParamBase {
public:
    Param(AlgorithmBase* owner, const char* name, T def, T lo, T hi)
        : ParamBase(owner, name), m_v(def), m_lo(lo), m_hi(hi), m_def(def) {}

    // Read as a plain T — no lookup, no string, usable in inner loops.
    operator T() const { return m_v; }
    T get() const { return m_v; }

    // The only mutator. Clamps to range; returns true if the value changed.
    bool set(T nv) {
        nv = std::clamp(nv, m_lo, m_hi);
        if (nv == m_v) return false;
        m_v = nv;
        return true;
    }

    T Min() const { return m_lo; }
    T Max() const { return m_hi; }
    T Default() const { return m_def; }

    ParamType Type() const override;
    bool      SetFromScript(const Value& v, std::string* err) override;
    bool      DrawWidget() override;
    uint64_t  HashValue() const override;

private:
    T m_v, m_lo, m_hi, m_def;
};

// Bool has no meaningful range; provide a narrower constructor.
template <>
class Param<bool> : public ParamBase {
public:
    Param(AlgorithmBase* owner, const char* name, bool def)
        : ParamBase(owner, name), m_v(def), m_def(def) {}

    operator bool() const { return m_v; }
    bool get() const { return m_v; }
    bool set(bool nv) {
        if (nv == m_v) return false;
        m_v = nv;
        return true;
    }

    ParamType Type() const override { return ParamType::Bool; }
    bool      SetFromScript(const Value& v, std::string* err) override;
    bool      DrawWidget() override;
    uint64_t  HashValue() const override { return m_v ? 1ull : 0ull; }

private:
    bool m_v, m_def;
};

extern template class Param<float>;
extern template class Param<int>;

} // namespace tglab
