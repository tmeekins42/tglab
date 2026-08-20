// Interpreter — phase 1 of execution.
//
// Walks the AST and RECORDS pipeline stages, declares UI controls, and
// declares viewers. It never runs an algorithm, so re-running on every slider
// drag costs microseconds.
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "../core/pipeline.h"
#include "ast.h"
#include "value.h"

namespace tglab {

// A UI control declared by the script. Values persist across re-runs and hot
// reloads (matched by label), so editing a script does not reset tuning.
struct UiControl {
    enum class Kind { Slider, Check, Choose };

    Kind        kind = Kind::Slider;

    // Identity, kept distinct from what the user reads. `label` must be unique
    // across the whole panel -- it is how a control's value is matched across
    // re-runs -- so for params() it carries the algorithm and instance name.
    // Showing that full string is what clipped the panel and made the controls
    // unreadable, hence `display`: the short text drawn next to the widget,
    // with `group` naming the box it sits in.
    std::string label;
    std::string display;   // empty = use `label`
    std::string group;     // empty = ungrouped, drawn at the top level
    std::string help;      // one-line description, shown in the tooltip
    double      value = 0;       // Slider / Check
    double      lo = 0, hi = 1, def = 0;

    // Widget tuning carried over from the parameter's ParamOpts, so a control
    // built by params() drags exactly like the algorithm's own inspector row.
    double      step = 0.0;                  // 0 = continuous
    double      softLo = 0.0, softHi = 0.0;  // equal = use [lo, hi]

    // Choose (M2): candidate algorithm names and the selected index.
    std::vector<std::string> options;
    int                      selected = 0;

    bool seenThisRun = false;
};

// Holds control state across runs. Owned by the app, not the interpreter.
class UiState {
public:
    UiControl* Find(const std::string& label);
    UiControl& FindOrAdd(const UiControl& proto);

    void BeginRun();                     // clears seenThisRun
    void DropUnseen();                   // removes controls the script no longer declares

    std::vector<UiControl>&       Controls()       { return m_controls; }
    const std::vector<UiControl>& Controls() const { return m_controls; }

private:
    std::vector<UiControl> m_controls;
};

// Named source images available to the script via image("name").
struct SourceImage {
    std::string name;
    int         index = 0;   // index into the palette Data vector
};

struct InterpResult {
    bool        ok = false;
    std::string error;
};

InterpResult Interpret(const Program& prog,
                       const std::vector<SourceImage>& sources,
                       UiState* ui,
                       Pipeline* out);

} // namespace tglab
