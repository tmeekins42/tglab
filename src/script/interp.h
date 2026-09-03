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
    // Pick is a dropdown like Choose, but its value is an INTEGER rather than
    // an algorithm name -- the mode selector on an ordinary parameter. Kept
    // distinct because the two mean different things downstream: Choose feeds a
    // registry lookup, Pick feeds a Param<int>. Sharing one kind would make
    // every consumer ask which flavour it was holding.
    enum class Kind { Slider, Check, Choose, Pick };

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
    // Which option a reset returns to. Defaults to the first, which without a
    // script-declared default is alphabetical and therefore arbitrary.
    int                      defaultIndex = 0;

    // Position in this run's declaration sequence. The panel draws in this
    // order, so it must follow the script rather than when a control was first
    // created -- re-declaring one would otherwise move it to the end.
    int  declOrder   = 0;
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
    int                    m_declOrder = 0;   // counts declarations this run
};

// Named source images available to the script via image("name").
struct SourceImage {
    std::string name;
    int         index = 0;   // index into the palette Data vector

    // True when this palette entry holds an undemosaiced sensor mosaic.
    //
    // The palette stores the mosaic rather than a demosaiced copy, because that
    // is where the dynamic range lives -- a 14-bit sensor's ~15,000 levels
    // against the 256 an 8-bit conversion leaves. image() then inserts the
    // demosaic automatically, so a script never has to mention it and works
    // unchanged whether a PNG or a CR3 is dropped on the slot.
    bool isMosaic = false;

    // What this palette entry holds. Scalar -- one image -- for everything
    // dropped as a single file, which is every entry today.
    //
    // image() returns a port carrying this shape, so a group flows into the
    // script as a set and the interpreter's check catches it being handed to a
    // single-image algorithm at the line that did it.
    Shape shape;

    // The colour temperature and green/magenta offset the camera chose for this
    // shot, recovered from its white-balance metadata. Zero when the file
    // carries no daylight reference to measure against.
    //
    // Here so that a control declared by params() can DEFAULT to what this
    // photograph actually is. The alternative was a sentinel -- kelvin 0 meaning
    // "leave it alone" -- which is a poor answer to "what temperature is this?"
    // when the file knows, and which Tim reported as confusing on sight.
    float asShotKelvin = 0.0f;
    float asShotTint   = 0.0f;

    // What a measurement of the sensor data suggests, for basic_adjust's
    // auto-exposure. Computed once when the file loads rather than per run: it
    // costs a pass over the mosaic, and the answer cannot change until the
    // pixels do.
    bool  hasAutoExposure = false;
    float autoExposure    = 0.0f;   // stops
    float autoHighlights  = 0.0f;   // negative recovers
    float autoShadows     = 0.0f;
    float autoBlacks      = 0.0f;
};

struct InterpResult {
    bool        ok = false;
    std::string error;
};

// `defaultDemosaic` names the algorithm image() inserts for a mosaic source.
// It is an app setting rather than something the script states, so that a
// script never has to mention demosaicing on the chance a raw file is dropped
// on it. A script that wants a specific method calls mosaic() and demosaics
// explicitly, which is what makes side-by-side comparison possible.
//
// demosaic_consistent, which measures better than AHD on both axes that
// matter and costs 31 ms more at 45 MP:
//
//                 detail   luma noise   chroma noise
//   ahd             150%       102%          74%
//   consistent      174%       106%          72%      (ISO 100)
//   ahd             144%       100%          90%
//   consistent      148%       102%          74%      (ISO 12800)
//
// It also reads visibly better on fine texture -- fur, feathers, a fluffy
// jacket against a background -- which is the content where the headroom
// measurement showed the largest gap and where an error metric is least able to
// judge, since squared error rewards not being wrong rather than being sharp.
//
// The history here is worth keeping: Malvar was the default until AHD gained a
// GPU path, and AHD until this did. Each time the argument was the same -- at
// 45 MP these are bandwidth-bound rather than compute-bound, so extra
// arithmetic is nearly free against the cost of moving the pixels, and given
// near-equal cost the better reconstruction should be what a raw opens with.
InterpResult Interpret(const Program& prog,
                       const std::vector<SourceImage>& sources,
                       UiState* ui,
                       Pipeline* out,
                       const std::string& defaultDemosaic = "demosaic_ahd");

} // namespace tglab
