// Verifies auto-exposure end to end through the script layer: the defaults
// must reach the declared controls, and only when asked for.
#include <cstdio>
#include <string>
#include <vector>
#include "../src/core/raw_io.h"
#include "../src/core/image.h"
#include "../src/core/pipeline.h"
#include "../src/algo_util/auto_develop.h"
#include "../src/script/interp.h"
#include "../src/script/parser.h"
using namespace tglab;

static int g_fail = 0;
static void Check(bool ok, const std::string& what) {
    std::printf("%s  %s\n", ok ? "[ ok ]" : "[FAIL]", what.c_str());
    if (!ok) ++g_fail;
}

static bool ControlValue(UiState& ui, const std::string& suffix, double* out) {
    for (UiControl& c : ui.Controls())
        if (c.label == suffix) {
            *out = c.value; return true;
        }
    return false;
}

int main(int argc, char** argv) {
    Image m; std::string err;
    if (!LoadRawMosaic(argv[1], &m, &err)) { std::printf("load: %s\n", err.c_str()); return 1; }
    const AutoDevelopSuggestion s = SuggestExposure(m);
    std::printf("measurement: exposure %+.2f highlights %+.2f shadows %+.2f "
                "blacks %+.2f\n\n",
                s.exposure, s.highlights, s.shadows, s.blacks);

    auto run = [&](const char* script, UiState* ui) {
        std::vector<Data> sources; sources.push_back(Data{m.Clone()});
        SourceImage si;
        si.name = "test"; si.index = 0; si.isMosaic = true;
        si.hasAutoExposure = s.valid;
        si.autoHighlights = s.highlights;
        si.autoExposure = s.exposure;
        si.autoShadows  = s.shadows;
        si.autoBlacks   = s.blacks;
        std::vector<SourceImage> names{si};
        Program prog; std::string e;
        if (!Parse(script, &prog, &e)) { std::printf("parse: %s\n", e.c_str()); return false; }
        Pipeline p;
        auto r = Interpret(prog, names, ui, &p);
        if (!r.ok) { std::printf("interp: %s\n", r.error.c_str()); return false; }
        return true;
    };

    // Off by default.
    {
        UiState ui;
        if (!run("src = image(\"test\")\no = params(basic_adjust)(src)\ndisplay(o)\n", &ui)) return 1;
        double v = -999;
        Check(ControlValue(ui, "basic_adjust.exposure", &v), "exposure control is declared");
        Check(v == 0.0, "exposure opens at 0 with auto off (got " + std::to_string(v) + ")");
    }
    // On.
    {
        UiState ui;
        if (!run("src = image(\"test\")\n"
                 "o = params(basic_adjust, auto_exposure = 1)(src)\ndisplay(o)\n", &ui)) return 1;
        double v = -999;
        ControlValue(ui, "basic_adjust.exposure", &v);
        std::printf("       with auto on, exposure control = %+.2f\n", v);
        Check(std::abs(v - s.exposure) < 0.01,
              "exposure opens at the measured suggestion");
    }
    // The question Tim's note actually raised: "a copy of the sliders so the
    // user can adjust them after they've been auto-fixed... this might be
    // tricky if the whole pipeline is re-run".
    //
    // There is no copy. The suggestion is a DEFAULT, and a value the user moved
    // is not a default -- so a re-run leaves it alone. This checks exactly that,
    // because it is the property the whole design rests on: if a re-run
    // stamped the measurement back over a dragged slider, the feature would be
    // unusable rather than merely wrong.
    {
        UiState ui;
        const char* script =
            "src = image(\"test\")\n"
            "o = params(basic_adjust, auto_exposure = 1)(src)\ndisplay(o)\n";
        if (!run(script, &ui)) return 1;

        double v = 0;
        ControlValue(ui, "basic_adjust.exposure", &v);
        Check(std::abs(v - s.exposure) < 0.01, "first run opens at the suggestion");

        // The user drags it somewhere else.
        for (UiControl& c : ui.Controls())
            if (c.label == "basic_adjust.exposure") c.value = 0.25;

        // Re-run, as every slider tick does.
        if (!run(script, &ui)) return 1;
        ControlValue(ui, "basic_adjust.exposure", &v);
        std::printf("       after dragging to 0.25 and re-running: %+.2f\n", v);
        Check(std::abs(v - 0.25) < 1e-6, "a dragged slider survives the re-run");
    }

    // Highlights and shadows follow the same path, and are only suggested when
    // the measurement says there is something to recover -- which is what makes
    // them safe to apply unasked. A frame with no clipping must get zero, or
    // the "auto" would be flattening pictures that were fine.
    {
        UiState ui;
        if (!run("src = image(\"test\")\n"
                 "o = params(basic_adjust, auto_exposure = 1)(src)\ndisplay(o)\n", &ui)) return 1;

        double hi = 0, sh = 0;
        ControlValue(ui, "basic_adjust.highlights", &hi);
        ControlValue(ui, "basic_adjust.shadows", &sh);
        std::printf("       highlights %+.2f (clipped %.2f%% -> %.2f%% after push)\n",
                    hi, 100.0 * s.clippedFrac, 100.0 * s.clippedAfter);
        std::printf("       shadows    %+.2f (crushed %.2f%%)\n", sh, 100.0 * s.crushedFrac);

        Check(std::abs(hi - s.highlights) < 0.01, "highlights opens at the suggestion");
        Check(std::abs(sh - s.shadows) < 0.01, "shadows opens at the suggestion");

        // Direction, not just magnitude: a positive highlights value would
        // push blown areas further rather than recovering them.
        Check(hi <= 0.0, "highlight suggestion recovers rather than pushes");
        Check(sh >= 0.0, "shadow suggestion opens rather than deepens");

        // And the rule that keeps this from touching a clean frame.
        if (s.clippedAfter <= 0.01)
            Check(hi == 0.0, "no highlight recovery when nothing is clipped");
        if (s.crushedFrac <= 0.02)
            Check(sh == 0.0, "no shadow lift when nothing is crushed");
    }

    // Toggling the checkbox on the panel must take effect.
    //
    // params() exposes auto_exposure as a control, and ticking it did nothing:
    // PrepareDefaults ran on the probe using only what the SCRIPT said, and the
    // control's value was read further down, after the defaults had already
    // been decided. So the checkbox was written every frame and never once
    // looked at before the decision it was meant to make.
    //
    // Driven the way the app does it -- set the control, re-interpret -- since
    // that is the path that was broken.
    {
        UiState ui;
        const char* script =
            "src = image(\"test\")\n"
            "o = params(basic_adjust)(src)\ndisplay(o)\n";   // no script argument
        if (!run(script, &ui)) return 1;

        double v = -999;
        ControlValue(ui, "basic_adjust.exposure", &v);
        Check(v == 0.0, "exposure starts at 0 with the box unticked");

        // Tick it, as clicking the checkbox does.
        bool found = false;
        for (UiControl& c : ui.Controls())
            if (c.label == "basic_adjust.auto_exposure") { c.value = 1; found = true; }
        Check(found, "the auto_exposure control exists on the panel");

        if (!run(script, &ui)) return 1;
        ControlValue(ui, "basic_adjust.exposure", &v);
        std::printf("       after ticking auto_exposure: %+.2f\n", v);
        Check(std::abs(v - s.exposure) < 0.01,
              "ticking the box moves exposure to the suggestion");

        // And untick it again: the control must go back, not stay stuck on a
        // value the measurement chose.
        for (UiControl& c : ui.Controls())
            if (c.label == "basic_adjust.auto_exposure") c.value = 0;
        if (!run(script, &ui)) return 1;
        ControlValue(ui, "basic_adjust.exposure", &v);
        std::printf("       after unticking:             %+.2f\n", v);
        Check(v == 0.0, "unticking the box returns exposure to 0");
    }

    std::printf("\n%s\n", g_fail ? "FAILURES" : "auto-exposure wiring ok");
    return g_fail ? 1 : 0;
}
