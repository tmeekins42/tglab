#include "param.h"

#include <cstring>

#include "algorithm.h"
#include "../script/value.h"

#include "imgui.h"

namespace tglab {

const char* ParamTypeName(ParamType t) {
    switch (t) {
        case ParamType::Float: return "float";
        case ParamType::Int:   return "int";
        case ParamType::Bool:  return "bool";
    }
    return "?";
}

// Self-collection: declaring a Param member registers its descriptor.
ParamBase::ParamBase(AlgorithmBase* owner, const char* name) : m_name(name) {
    owner->m_params.push_back(this);
}

// --- float ------------------------------------------------------------------

template <> ParamType Param<float>::Type() const { return ParamType::Float; }

template <>
bool Param<float>::SetFromScript(const Value& v, std::string* err) {
    if (!v.IsNumber()) {
        if (err) *err = std::string("parameter '") + m_name + "' expects a number, got " + v.TypeName();
        return false;
    }
    set(static_cast<float>(v.AsNumber()));
    return true;
}

template <>
bool Param<float>::DrawWidget() {
    float tmp = m_v;
    if (ImGui::SliderFloat(m_name, &tmp, m_lo, m_hi)) return set(tmp);
    return false;
}

template <>
uint64_t Param<float>::HashValue() const {
    uint32_t bits;
    static_assert(sizeof(bits) == sizeof(m_v));
    std::memcpy(&bits, &m_v, sizeof(bits));
    return uint64_t(bits);
}

// --- int --------------------------------------------------------------------

template <> ParamType Param<int>::Type() const { return ParamType::Int; }

template <>
bool Param<int>::SetFromScript(const Value& v, std::string* err) {
    if (!v.IsNumber()) {
        if (err) *err = std::string("parameter '") + m_name + "' expects a number, got " + v.TypeName();
        return false;
    }
    set(static_cast<int>(v.AsNumber()));
    return true;
}

template <>
bool Param<int>::DrawWidget() {
    int tmp = m_v;
    if (ImGui::SliderInt(m_name, &tmp, m_lo, m_hi)) return set(tmp);
    return false;
}

template <>
uint64_t Param<int>::HashValue() const {
    return uint64_t(static_cast<uint32_t>(m_v));
}

// --- bool -------------------------------------------------------------------

bool Param<bool>::SetFromScript(const Value& v, std::string* err) {
    if (!v.IsNumber()) {
        if (err) *err = std::string("parameter '") + m_name + "' expects a number (0/1), got " + v.TypeName();
        return false;
    }
    set(v.AsNumber() != 0.0);
    return true;
}

bool Param<bool>::DrawWidget() {
    bool tmp = m_v;
    if (ImGui::Checkbox(m_name, &tmp)) return set(tmp);
    return false;
}

template class Param<float>;
template class Param<int>;

} // namespace tglab
