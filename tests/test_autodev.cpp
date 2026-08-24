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
    std::printf("measurement: exposure %+.2f shadows %+.2f blacks %+.2f\n\n",
                s.exposure, s.shadows, s.blacks);

    auto run = [&](const char* script, UiState* ui) {
        std::vector<Data> sources; sources.push_back(Data{m.Clone()});
        SourceImage si;
        si.name = "test"; si.index = 0; si.isMosaic = true;
        si.hasAutoExposure = s.valid;
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

    std::printf("\n%s\n", g_fail ? "FAILURES" : "auto-exposure wiring ok");
    return g_fail ? 1 : 0;
}
