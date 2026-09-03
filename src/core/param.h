// Param<T>: the single storage location for an algorithm parameter.
//
// The script writes it, the inspector widget writes it, the algorithm reads it.
// Nothing is ever copied between homes, so there is no sync problem to solve.
//
// Params self-collect: ParamBase's constructor pushes itself onto the owner's
// list, so declaring a member *is* declaring the descriptor.
#pragma once

#include <cmath>
#include <algorithm>
#include <cstdint>
#include <span>
#include <string>

namespace tglab {

class AlgorithmBase;
struct UiControl;
class Value;

enum class ParamType : uint8_t { Float, Int, Bool, Text };

// Optional tuning for a numeric parameter's widget.
//
// The full [lo, hi] range is often far wider than the range actually worth
// dragging through: sigma is declared 0.1..20 but tuned around 0.5..3, so a
// plain slider spends most of its travel on values nobody wants and makes the
// useful part impossible to hit. These let a parameter say how it wants to be
// driven without the script or the UI restating it.
struct ParamOpts {
    // One line saying what this parameter actually controls, shown in the
    // tooltip. A name and a range do not tell you what `k` or `eps` does, and
    // for an unfamiliar algorithm that is the whole question. Say what changes
    // and in which direction -- "higher keeps more edges" beats "edge
    // threshold".
    const char* help = nullptr;

    // Quantum for arrow keys and dragging. 0 means continuous.
    double step = 0.0;

    // Preferred slider extent. When set, the slider covers [softMin, softMax]
    // while values outside it stay reachable by typing (ctrl+click). 0/0 means
    // "use the full range".
    double softMin = 0.0;
    double softMax = 0.0;

    // NAMES FOR AN INTEGER THAT SELECTS A MODE, rather than measures a
    // quantity.
    //
    // A slider is the wrong control for a choice. "projection: 0 plane, 1
    // cylinder, 2 sphere" has to spell the mapping out in its own label because
    // the widget shows a number, the tooltip carries documentation that ought to
    // be the control itself, and picking a value means dragging a slider onto an
    // exact integer.
    //
    // Listing the names here fixes all three at once: the inspector draws a
    // dropdown, the script's pick() offers the same names, and the label goes
    // back to being a label. The VALUE is still the integer, so nothing
    // downstream changes and an algorithm reads its parameter exactly as before.
    //
    // Names map to lo, lo+1, ... in order, so the range must cover them. That is
    // deliberate rather than a general int-to-string map: modes that are not
    // consecutive are almost always a sign the enum wants renumbering, and the
    // simpler rule keeps the declaration a single line.
    const char* const* choices = nullptr;
    int choiceCount = 0;

    bool HasStep() const { return step > 0.0; }
    bool HasSoftRange() const { return softMin != softMax; }
    bool HasChoices() const { return choices && choiceCount > 0; }
};

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

    // Fills in kind/range/default so the script can auto-expose this parameter
    // as a UI control (the params() builtin). Returns false for kinds that have
    // no widget yet. The label is set by the caller.
    virtual bool DescribeControl(UiControl* out) const = 0;

    // Folded into Stage::paramHash for dirty detection.
    virtual uint64_t HashValue() const = 0;

    const char* Name() const { return m_name; }

    // One-line description, or nullptr. Declared through ParamOpts.
    virtual const char* Help() const { return nullptr; }

    // The names for a mode-selecting integer, or empty. See ParamOpts::choices.
    //
    // On the base so the inspector and the script can both ask without knowing
    // the concrete type -- the UI needs it to draw a dropdown instead of a
    // slider, and pick() needs it to turn a name back into the integer.
    virtual std::span<const char* const> Choices() const { return {}; }

    // The lowest value the choices count up from; only meaningful with them.
    virtual int ChoiceBase() const { return 0; }

protected:
    const char* m_name;
};

template <class T>
class Param : public ParamBase {
public:
    Param(AlgorithmBase* owner, const char* name, T def, T lo, T hi, ParamOpts opts = {})
        : ParamBase(owner, name), m_v(def), m_lo(lo), m_hi(hi), m_def(def), m_opts(opts) {}

    // Read as a plain T — no lookup, no string, usable in inner loops.
    operator T() const { return m_v; }
    T get() const { return m_v; }

    // The only mutator. Snaps to the step, clamps to range; returns true if the
    // value changed. Snapping lives here rather than in the widget so a typed
    // or script-assigned value lands on the same grid as a dragged one.
    bool set(T nv) {
        nv = Snap(nv);
        nv = std::clamp(nv, m_lo, m_hi);
        if (nv == m_v) return false;
        m_v = nv;
        return true;
    }

    T Min() const { return m_lo; }
    T Max() const { return m_hi; }
    T Default() const { return m_def; }

    // Replaces the declared default with one derived from the image, and moves
    // the current value with it when the value is still the old default -- so an
    // untouched control follows the source, while one the user or the script has
    // set stays put.
    //
    // Used by basic_adjust so the kelvin slider opens at the temperature the
    // camera actually chose. See AlgorithmBase::PrepareDefaults.
    void SetDefault(T d) {
        d = std::clamp(d, m_lo, m_hi);
        if (m_v == m_def) m_v = d;
        m_def = d;
    }
    const ParamOpts& Opts() const { return m_opts; }
    const char* Help() const override { return m_opts.help; }

    std::span<const char* const> Choices() const override {
        return {m_opts.choices, size_t(m_opts.HasChoices() ? m_opts.choiceCount : 0)};
    }
    int ChoiceBase() const override { return int(m_lo); }

    // Slider extent: the soft range when one is declared, else the full range.
    T SliderMin() const { return m_opts.HasSoftRange() ? T(m_opts.softMin) : m_lo; }
    T SliderMax() const { return m_opts.HasSoftRange() ? T(m_opts.softMax) : m_hi; }

    ParamType Type() const override;
    bool      SetFromScript(const Value& v, std::string* err) override;
    bool      DrawWidget() override;
    bool      DescribeControl(UiControl* out) const override;
    uint64_t  HashValue() const override;

private:
    // Rounds to the nearest multiple of the step measured from m_lo, so a
    // window declared 3..201 step 2 snaps to odd sizes rather than even ones.
    T Snap(T v) const {
        if (!m_opts.HasStep()) return v;
        const double s = m_opts.step;
        const double n = std::round((double(v) - double(m_lo)) / s);
        return T(double(m_lo) + n * s);
    }

    T m_v, m_lo, m_hi, m_def;
    ParamOpts m_opts;
};

// Bool has no meaningful range; provide a narrower constructor.
template <>
class Param<bool> : public ParamBase {
public:
    // Bools have no range to tune, but they still benefit from help: a flag
    // named `exponential` says nothing about what it selects.
    Param(AlgorithmBase* owner, const char* name, bool def, const char* help = nullptr)
        : ParamBase(owner, name), m_v(def), m_def(def), m_help(help) {}

    const char* Help() const override { return m_help; }

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
    bool      DescribeControl(UiControl* out) const override;
    uint64_t  HashValue() const override { return m_v ? 1ull : 0ull; }

private:
    bool m_v, m_def;
    const char* m_help = nullptr;
};

// A string parameter, for the one thing a number cannot express: a file path.
//
// It deliberately has NO widget -- DescribeControl returns false, so params()
// skips it and the inspector does not draw a row. A path is not something to
// drag, and a text field in the controls panel would be a worse way to choose a
// file than the script line that is already there. The script sets it and that
// is the whole interface.
template <>
class Param<std::string> : public ParamBase {
public:
    Param(AlgorithmBase* owner, const char* name, std::string def = {},
          const char* help = nullptr)
        : ParamBase(owner, name), m_v(std::move(def)), m_help(help) {}

    const char* Help() const override { return m_help; }

    const std::string& get() const { return m_v; }
    bool set(std::string nv) {
        if (nv == m_v) return false;
        m_v = std::move(nv);
        return true;
    }

    ParamType Type() const override { return ParamType::Text; }
    bool      SetFromScript(const Value& v, std::string* err) override;
    bool      DrawWidget() override;
    bool      DescribeControl(UiControl*) const override { return false; }
    uint64_t  HashValue() const override {
        // FNV-1a over the path, so changing the file re-runs the stage.
        uint64_t h = 1469598103934665603ull;
        for (unsigned char c : m_v) {
            h ^= c;
            h *= 1099511628211ull;
        }
        return h;
    }

private:
    std::string m_v;
    const char* m_help = nullptr;
};

extern template class Param<float>;
extern template class Param<int>;

} // namespace tglab
