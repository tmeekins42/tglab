#include "param.h"

#include <cstring>

#include "algorithm.h"
#include "../script/interp.h"
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

namespace {

// Fine-tuning affordances shared by the numeric widgets, all of them hidden
// until used so the panel stays one row per parameter:
//   - left/right arrows step by exactly one increment while the slider is
//     active, which is the "fixed increments" a drag cannot give;
//   - ctrl+click types an exact value (ImGui built-in, but undiscoverable);
//   - right-click resets to the declared default.
// Returns the requested delta in steps, and sets `reset` when the value should
// go back to its default.
int StepInput(bool* reset, const char* help) {
    *reset = false;
    int steps = 0;

    if (ImGui::IsItemActive() || ImGui::IsItemFocused()) {
        if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow,  true)) --steps;
        if (ImGui::IsKeyPressed(ImGuiKey_RightArrow, true)) ++steps;
    }
    if (ImGui::IsItemHovered()) {
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) *reset = true;
        ImGui::BeginTooltip();
        if (help && *help) {
            ImGui::TextUnformatted(help);
            ImGui::Separator();
        }
        ImGui::TextDisabled("ctrl+click to type   arrows to step   right-click to reset");
        ImGui::EndTooltip();
    }
    return steps;
}

} // namespace

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
    // AlwaysClamp keeps a typed value inside the real range even when the
    // slider only spans the soft one. Ctrl+click to type an exact value;
    // the tooltip says so, since the affordance is invisible otherwise.
    const ImGuiSliderFlags flags = ImGuiSliderFlags_AlwaysClamp;
    const char* fmt = m_opts.HasStep() && m_opts.step >= 0.01 ? "%.2f" : "%.3f";

    bool changed = false;
    if (ImGui::SliderFloat(m_name, &tmp, SliderMin(), SliderMax(), fmt, flags))
        changed = set(tmp);

    bool reset = false;
    const int steps = StepInput(&reset, Help());
    if (reset) changed |= set(m_def);
    else if (steps) {
        // Without a declared step, fall back to 1% of the slider extent so the
        // arrows still do something sensible.
        const float s = m_opts.HasStep() ? float(m_opts.step)
                                         : (SliderMax() - SliderMin()) * 0.01f;
        changed |= set(m_v + float(steps) * s);
    }
    return changed;
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
    bool changed = false;
    if (ImGui::SliderInt(m_name, &tmp, SliderMin(), SliderMax(), "%d",
                         ImGuiSliderFlags_AlwaysClamp))
        changed = set(tmp);

    bool reset = false;
    const int steps = StepInput(&reset, Help());
    if (reset) changed |= set(m_def);
    else if (steps) {
        const int s = m_opts.HasStep() ? int(m_opts.step) : 1;
        changed |= set(m_v + steps * s);
    }
    return changed;
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
    const bool changed = ImGui::Checkbox(m_name, &tmp) && set(tmp);
    if (m_help && *m_help && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", m_help);
    return changed;
}
// --- control description (for the script's params() builtin) ----------------
//
// Each parameter knows its own widget kind, range and default, so params() can
// build the UI without the script restating any of it.

template <>
bool Param<float>::DescribeControl(UiControl* out) const {
    out->kind  = UiControl::Kind::Slider;
    out->lo    = double(m_lo);
    out->hi    = double(m_hi);
    out->def   = double(m_def);
    out->value = double(m_def);
    out->step   = m_opts.step;
    out->softLo = m_opts.softMin;
    out->softHi = m_opts.softMax;
    return true;
}

template <>
bool Param<int>::DescribeControl(UiControl* out) const {
    // Ints ride the same slider; the value truncates at the Param boundary.
    out->kind  = UiControl::Kind::Slider;
    out->lo    = double(m_lo);
    out->hi    = double(m_hi);
    out->def   = double(m_def);
    out->value = double(m_def);
    out->step   = m_opts.step;
    out->softLo = m_opts.softMin;
    out->softHi = m_opts.softMax;
    return true;
}

bool Param<bool>::DescribeControl(UiControl* out) const {
    out->kind  = UiControl::Kind::Check;
    out->lo    = 0.0;
    out->hi    = 1.0;
    out->def   = m_def ? 1.0 : 0.0;
    out->value = out->def;
    return true;
}

template class Param<float>;
template class Param<int>;

} // namespace tglab
