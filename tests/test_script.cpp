// Headless checks for the script spine: parse -> interpret -> execute.
// Runs without a window, so failures here are quick to diagnose.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "../src/core/algorithm.h"
#include "../src/algo_util/histogram.h"
#include "../src/core/pipeline.h"
#include "../src/script/interp.h"
#include "../src/script/parser.h"

using namespace tglab;

static int g_fail = 0;

static void Check(bool cond, const std::string& what) {
    std::printf("%s  %s\n", cond ? "[ ok ]" : "[FAIL]", what.c_str());
    if (!cond) ++g_fail;
}

// Runs a script against one 4x4 source image named "test".
static bool RunScript(const std::string& src, UiState* ui, Pipeline* pipe,
                      std::string* err, std::vector<Data>* sources) {
    ImageDesc d{4, 4, Format::RGBA8};
    Image img;
    img.Alloc(d);
    ImageView v = img.MapCpuWrite();
    for (int i = 0; i < 4 * 4 * 4; ++i) v.data[i] = uint8_t(i * 3);
    sources->clear();
    sources->push_back(Data{std::move(img)});

    std::vector<SourceImage> names{{"test", 0}};

    Program prog;
    if (!Parse(src, &prog, err)) return false;

    InterpResult r = Interpret(prog, names, ui, pipe);
    if (!r.ok) { *err = r.error; return false; }

    return pipe->Execute(sources, nullptr, err);
}

int main() {
    std::printf("registry: ");
    for (const std::string& n : Registry::Get().Names()) std::printf("%s ", n.c_str());
    std::printf("\n\n");

    // Risk 2: self-registration must survive the linker.
    Check(Registry::Get().Contains("brightness"), "brightness is registered");
    Check(Registry::Get().Contains("grayscale"),  "grayscale is registered (2nd algo, no central file edited)");

    // --- happy path ---------------------------------------------------------
    {
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        const bool ok = RunScript(
            "src = image(\"test\")\n"
            "a = slider(\"amount\", -1, 1, 0.5)\n"
            "out = brightness(src, amount = a)\n"
            "display(out, \"result\")\n",
            &ui, &p, &err, &src);
        Check(ok, "basic script runs" + (ok ? "" : ": " + err));
        Check(p.Stages().size() == 1, "one stage recorded");
        Check(p.Viewers().size() == 1, "one viewer declared");
        Check(ui.Controls().size() == 1, "one slider declared");
        if (!ui.Controls().empty())
            Check(ui.Controls()[0].value == 0.5, "slider default applied");
    }

    // --- slider value persists across re-runs (hot reload keeps tuning) ------
    {
        UiState ui; std::string err; std::vector<Data> src;
        Pipeline p1, p2;
        RunScript("a = slider(\"amount\", 0, 10, 1)\nsrc = image(\"test\")\n"
                  "o = brightness(src, amount = a)\ndisplay(o)\n", &ui, &p1, &err, &src);
        ui.Controls()[0].value = 7.0;    // user drags the slider
        RunScript("a = slider(\"amount\", 0, 10, 1)\nsrc = image(\"test\")\n"
                  "o = brightness(src, amount = a)\ndisplay(o)\n", &ui, &p2, &err, &src);
        Check(ui.Controls().size() == 1 && ui.Controls()[0].value == 7.0,
              "slider value survives a re-run");
    }

    // --- errors report a line number and do not crash -----------------------
    {
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        RunScript("src = image(\"test\")\nout = brightness(src, radiuss = 3)\n", &ui, &p, &err, &src);
        Check(err.find("line 2") != std::string::npos && err.find("radiuss") != std::string::npos,
              "unknown parameter reports line 2: \"" + err + "\"");
    }
    {
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        RunScript("src = image(\"test\")\nout = brightness(\n", &ui, &p, &err, &src);
        Check(!err.empty() && err.find("line") != std::string::npos,
              "syntax error reports a line: \"" + err + "\"");
    }
    {
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        RunScript("src = image(\"test\")\na, b = brightness(src)\n", &ui, &p, &err, &src);
        Check(err.find("returns 1") != std::string::npos,
              "asking for more values than an algorithm returns is caught: \"" + err + "\"");
    }
    {
        // Too FEW targets is not an error: one target takes port 0, matching
        // what a nested call already does. Requiring exact arity made
        // `t = sobel(x)` fail while `f(sobel(x))` succeeded, and made
        // multi-output algorithms unusable from a choose() dropdown.
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        const bool ok = RunScript("src = image(\"test\")\nout = sobel(src)\ndisplay(out)\n",
                                  &ui, &p, &err, &src);
        Check(ok, "single target from a multi-output algorithm takes port 0" +
                      (ok ? "" : ": " + err));
    }
    {
        // The two spellings must agree.
        UiState ui; Pipeline p1, p2; std::string err; std::vector<Data> src;
        const bool a = RunScript("src = image(\"test\")\nout = sobel(src)\ndisplay(out)\n",
                                 &ui, &p1, &err, &src);
        const bool b = RunScript("src = image(\"test\")\nout = hysteresis(sobel(src))\ndisplay(out)\n",
                                 &ui, &p2, &err, &src);
        Check(a && b, "assigned and nested multi-output calls both work");
    }
    // --- `_` discards an output (Rust/Go style) -----------------------------
    {
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        const bool ok = RunScript(
            "src = image(\"test\")\n_, _, mag = sobel(src)\ndisplay(mag)\n", &ui, &p, &err, &src);
        Check(ok, "`_` discards unwanted outputs, repeatable" + (ok ? "" : ": " + err));
    }
    {
        // Write-only: reading it back would silently return whichever output
        // was assigned last, which is meaningless.
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        RunScript("src = image(\"test\")\n_, _, mag = sobel(src)\ndisplay(_)\n", &ui, &p, &err, &src);
        Check(err.find("cannot be read back") != std::string::npos,
              "reading `_` is an error: \"" + err + "\"");
    }
    {
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        RunScript("src = image(\"test\")\n_ = sobel(src)\n_.scale = 2\n", &ui, &p, &err, &src);
        Check(!err.empty(), "setting a parameter on `_` is an error: \"" + err + "\"");
    }
    {
        // Only a bare underscore is special.
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        const bool ok = RunScript(
            "src = image(\"test\")\n_tmp, gy, mag = sobel(src)\ndisplay(_tmp)\n",
            &ui, &p, &err, &src);
        Check(ok, "`_tmp` is an ordinary variable, not a discard" + (ok ? "" : ": " + err));
    }

    {
        // A category is only useful if every member is interchangeable, i.e.
        // selectable from a choose() dropdown with the same call shape.
        bool allOneInput = true;
        for (const std::string& n : Registry::Get().NamesInCategory("edge")) {
            auto algo = Registry::Get().Create(n);
            if (!algo || algo->Inputs().size() != 1) allOneInput = false;
        }
        Check(allOneInput, "every algorithm in the 'edge' category takes one image");
    }
    {
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        RunScript("src = image(\"nope\")\n", &ui, &p, &err, &src);
        Check(err.find("nope") != std::string::npos, "missing palette image is reported");
    }

    // --- grammar built ahead of need ---------------------------------------
    {
        Program prog; std::string err;
        Check(Parse("k = [[-1,0,1],[-2,0,2],[-1,0,1]]\n", &prog, &err),
              "matrix literal parses" + (err.empty() ? "" : ": " + err));
    }
    {
        Program prog; std::string err;
        Check(Parse("f = choose(\"q\", [brightness, grayscale])\nout = f(src)\n", &prog, &err),
              "call-as-postfix parses (variable in call position)" + (err.empty() ? "" : ": " + err));
    }
    {
        Program prog; std::string err;
        Check(Parse("a, b, c = sobel(img)\n", &prog, &err),
              "multi-assign parses" + (err.empty() ? "" : ": " + err));
    }

    // --- the algorithm actually changed the pixels --------------------------
    {
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        const bool ok = RunScript(
            "src = image(\"test\")\n"
            "out = brightness(src, gain = 2.0)\n"
            "display(out)\n", &ui, &p, &err, &src);
        bool changed = false;
        if (ok && !p.Stages().empty()) {
            Data& o = p.Stages()[0].outputs[0];
            Image& oi = std::get<Image>(o);
            Image& si = std::get<Image>(src[0]);
            ImageView ov = oi.MapCpuRead();
            ImageView sv = si.MapCpuRead();
            // gain=2 must brighten a mid-grey pixel.
            changed = ov.data[4] != sv.data[4];
        }
        Check(changed, "brightness(gain=2) actually modifies pixels");
    }

    // --- dirty-hash caching reuses unchanged stages -------------------------
    {
        UiState ui; std::string err; std::vector<Data> src;
        Pipeline p1;
        RunScript("src = image(\"test\")\no = brightness(src, gain = 1.5)\ndisplay(o)\n",
                  &ui, &p1, &err, &src);

        // Same script again: the stage is unchanged, so Execute should adopt
        // the cached output rather than recompute.
        Program prog;
        Parse("src = image(\"test\")\no = brightness(src, gain = 1.5)\ndisplay(o)\n", &prog, &err);
        std::vector<SourceImage> names{{"test", 0}};
        Pipeline p2;
        Interpret(prog, names, &ui, &p2);
        const bool ok = p2.Execute(&src, &p1, &err);
        Check(ok && p2.Stages()[0].valid, "unchanged stage reuses its cached output");
    }

    // --- the M1 acceptance criterion, headless ------------------------------
    // Moving a slider must change the produced pixels. This is what the GUI
    // demonstrates interactively; testing it here makes it a real assertion
    // rather than something only verifiable by eye.
    {
        UiState ui; std::string err; std::vector<Data> src;
        const char* kScript =
            "src = image(\"test\")\n"
            "g = slider(\"gain\", 0, 4, 1)\n"
            "out = brightness(src, gain = g)\n"
            "display(out)\n";

        Pipeline p1;
        RunScript(kScript, &ui, &p1, &err, &src);
        const uint8_t before = std::get<Image>(p1.Stages()[0].outputs[0]).MapCpuRead().data[4];

        // Simulate the user dragging the slider.
        ui.Controls()[0].value = 3.0;

        Pipeline p2;
        RunScript(kScript, &ui, &p2, &err, &src);
        const uint8_t after = std::get<Image>(p2.Stages()[0].outputs[0]).MapCpuRead().data[4];

        Check(before != after,
              "moving a slider changes the output pixels (" +
                  std::to_string(before) + " -> " + std::to_string(after) + ")");

        // And the changed parameter must invalidate the cache.
        Program prog;
        Parse(kScript, &prog, &err);
        std::vector<SourceImage> names{{"test", 0}};
        ui.Controls()[0].value = 0.25;
        Pipeline p3;
        Interpret(prog, names, &ui, &p3);
        p3.Execute(&src, &p2, &err);
        const uint8_t third = std::get<Image>(p3.Stages()[0].outputs[0]).MapCpuRead().data[4];
        Check(third != after, "changed parameter busts the stage cache");
    }

    // --- UTF-8 BOM ----------------------------------------------------------
    // Windows editors write a BOM by default; without skipping it every such
    // script failed on line 1 with "unexpected character".
    {
        Program prog; std::string err;
        const std::string bom = "\xEF\xBB\xBF" "a = 1\n";
        Check(Parse(bom, &prog, &err), "script with a UTF-8 BOM parses" + (err.empty() ? "" : ": " + err));
    }

    // --- multiple viewers ---------------------------------------------------
    {
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        RunScript("src = image(\"test\")\n"
                  "a = brightness(src, gain = 2)\n"
                  "b = grayscale(src)\n"
                  "display(src, \"original\")\n"
                  "display(a, \"bright\")\n"
                  "display(b, \"grey\")\n", &ui, &p, &err, &src);
        Check(p.Viewers().size() == 3, "three named viewers declared side by side");
    }

    // --- multi-output ports (sobel) -----------------------------------------
    {
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        const bool ok = RunScript(
            "src = image(\"test\")\n"
            "gx, gy, mag = sobel(src)\n"
            "display(gx, \"gx\")\ndisplay(gy, \"gy\")\ndisplay(mag, \"mag\")\n",
            &ui, &p, &err, &src);
        Check(ok, "sobel multi-output runs" + (ok ? "" : ": " + err));
        Check(ok && p.Stages()[0].outputs.size() == 3, "sobel produces three outputs");

        if (ok) {
            // gx/gy are signed; R32F is what makes that representable.
            Image& gxi = std::get<Image>(p.Stages()[0].outputs[0]);
            Check(gxi.Desc().format == Format::R32F, "sobel gx is R32F");
        }
    }

    // Signs need a pattern whose gradient actually reverses. The shared 4x4
    // fixture is a monotonic ramp, so its gx is single-signed by construction —
    // use a bright bar, which must give + on one edge and - on the other.
    {
        ImageDesc d{5, 5, Format::RGBA8};
        Image img;
        img.Alloc(d);
        ImageView v = img.MapCpuWrite();
        for (int y = 0; y < 5; ++y) {
            for (int x = 0; x < 5; ++x) {
                const uint8_t c = (x == 2 && y >= 1 && y <= 3) ? 255 : 0;
                uint8_t* p = v.At<uint8_t>(x, y);
                p[0] = p[1] = p[2] = c;
                p[3] = 255;
            }
        }

        auto algo = Registry::Get().Create("sobel");
        Data in{std::move(img)};
        const Data* ins[1] = {&in};
        std::vector<Data> outs(3);
        for (auto& o : outs) {
            Image t;
            t.Alloc({5, 5, Format::R32F});
            o = Data{std::move(t)};
        }
        RunCtx c(ins, outs);
        algo->RunCPU(c);

        ImageView gx = std::get<Image>(outs[0]).MapCpuRead();
        bool neg = false, pos = false;
        for (int i = 0; i < 25; ++i) {
            const float f = reinterpret_cast<const float*>(gx.data)[i];
            if (f < -1e-6f) neg = true;
            if (f >  1e-6f) pos = true;
        }
        Check(neg && pos, "sobel gx is signed: + on one edge of a bar, - on the other");
    }

    // --- choose(): explicit list --------------------------------------------
    {
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        const char* kScript =
            "src = image(\"test\")\n"
            "f = choose(\"op\", [brightness, grayscale])\n"
            "out = f(src)\n"
            "display(out)\n";

        const bool ok = RunScript(kScript, &ui, &p, &err, &src);
        Check(ok, "choose() with a list runs" + (ok ? "" : ": " + err));
        Check(ok && p.Stages()[0].algoName == "brightness",
              "choose() defaults to the first option");

        // Switching the dropdown must select the other algorithm.
        UiControl* c = ui.Find("op");
        Check(c && c->options.size() == 2, "choose() recorded both options");
        if (c) c->selected = 1;

        Pipeline p2;
        RunScript(kScript, &ui, &p2, &err, &src);
        Check(p2.Stages().size() == 1 && p2.Stages()[0].algoName == "grayscale",
              "switching the dropdown swaps the algorithm");
    }

    // --- choose(): by category ----------------------------------------------
    // A newly written algorithm in the category must appear with no script edit.
    {
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        const bool ok = RunScript(
            "src = image(\"test\")\n"
            "f = choose(\"edge op\", \"edge\")\n"
            "out = f(src)\n"
            "display(out)\n", &ui, &p, &err, &src);
        Check(ok, "choose() by category runs" + (ok ? "" : ": " + err));
        UiControl* c = ui.Find("edge op");
        Check(c && c->options.size() >= 2,
              "category lookup found the edge algorithms (" +
                  std::to_string(c ? c->options.size() : 0) + " found)");
    }

    // --- choose() error handling --------------------------------------------
    {
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        RunScript("f = choose(\"x\", \"no_such_category\")\n", &ui, &p, &err, &src);
        Check(err.find("no_such_category") != std::string::npos,
              "unknown category is reported: \"" + err + "\"");
    }
    {
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        RunScript("src = image(\"test\")\nf = choose(\"x\", [1, 2])\n", &ui, &p, &err, &src);
        Check(!err.empty(), "choose() rejects a list of non-algorithms: \"" + err + "\"");
    }

    // --- composite canny chains its stages ----------------------------------
    {
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        const bool ok = RunScript(
            "src = image(\"test\")\n"
            "e = canny(src, sigma = 1.4)\n"
            "display(e)\n", &ui, &p, &err, &src);
        Check(ok, "canny() runs as one stage" + (ok ? "" : ": " + err));

        if (ok) {
            ImageView v = std::get<Image>(p.Stages()[0].outputs[0]).MapCpuRead();
            int edgePixels = 0;
            for (int i = 0; i < v.desc.width * v.desc.height; ++i)
                if (reinterpret_cast<const float*>(v.data)[i] > 0.5f) ++edgePixels;
            // The test image has a checker patch, so it must find *some* edges
            // but must not mark everything.
            Check(edgePixels > 0 && edgePixels < v.desc.width * v.desc.height,
                  "canny finds edges without flooding (" + std::to_string(edgePixels) + " px)");
        }
    }

    // --- the same pipeline, built by hand from stages -----------------------
    {
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        const bool ok = RunScript(
            "src = image(\"test\")\n"
            "b = gaussian_blur(src, sigma = 1.4)\n"
            "gx, gy, mag = sobel(b)\n"
            "thin = non_max_suppression(gx, gy, mag)\n"
            "edges = hysteresis(thin, low = 0.1, high = 0.3)\n"
            "display(thin,  \"thinned\")\n"
            "display(edges, \"edges\")\n", &ui, &p, &err, &src);
        Check(ok, "canny stages compose in script" + (ok ? "" : ": " + err));
        Check(ok && p.Stages().size() == 4, "four separate stages recorded");
        Check(ok && p.Viewers().size() == 2, "intermediates are individually viewable");
    }

    // --- control reset ------------------------------------------------------
    // Mirrors App::IsModified/ResetControl. The UI wiring needs the app, but
    // the rules themselves are worth pinning down here.
    {
        auto modified = [](const UiControl& c) {
            return c.kind == UiControl::Kind::Choose ? c.selected != 0 : c.value != c.def;
        };
        auto reset = [](UiControl& c) {
            if (c.kind == UiControl::Kind::Choose) c.selected = 0;
            else                                   c.value = c.def;
        };

        UiControl s;
        s.kind = UiControl::Kind::Slider;
        s.lo = 0; s.hi = 10; s.def = 1.4; s.value = 1.4;
        Check(!modified(s), "a fresh slider is not 'modified'");
        s.value = 7.0;
        reset(s);
        Check(s.value == 1.4, "slider resets to its scripted default");

        UiControl d;
        d.kind = UiControl::Kind::Choose;
        d.options = {"canny", "sobel"};
        d.selected = 1;
        Check(modified(d), "a changed dropdown is 'modified'");
        reset(d);
        Check(d.selected == 0, "dropdown resets to the first listed option");
    }
    {
        // The reset target must be the value the script declared, and that has
        // to survive a re-run (which is what makes reset meaningful at all).
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        const char* kScript =
            "src = image(\"test\")\ng = slider(\"gain\", 0, 4, 2.5)\n"
            "o = brightness(src, gain = g)\ndisplay(o)\n";
        RunScript(kScript, &ui, &p, &err, &src);
        ui.Controls()[0].value = 3.9;          // user drags it
        Pipeline p2;
        RunScript(kScript, &ui, &p2, &err, &src);
        Check(ui.Controls()[0].def == 2.5,
              "the scripted default survives a re-run, so reset stays correct");
    }

    // --- params() auto-exposes an algorithm's own controls -------------------
    {
        // The point: one script serves a whole category, and each method's own
        // parameters appear without the script naming any of them.
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        const bool ok = RunScript(
            "src = image(\"test\")\n"
            "op = choose(\"method\", \"threshold\")\n"
            "m = params(op)(src)\n"
            "display(m)\n", &ui, &p, &err, &src);
        Check(ok, "params() runs" + (ok ? "" : ": " + err));

        // One dropdown plus one control per parameter of the selected method.
        bool sawDropdown = false, sawParamControl = false;
        for (const UiControl& c : ui.Controls()) {
            if (c.kind == UiControl::Kind::Choose) sawDropdown = true;
            if (c.label.rfind("threshold.", 0) == 0) sawParamControl = true;
        }
        Check(sawDropdown, "choose() declared its dropdown");
        Check(sawParamControl, "params() declared the algorithm's own controls");
    }
    {
        // Switching the dropdown must swap the control set, not accumulate it:
        // controls a run does not re-declare are dropped.
        UiState ui; std::string err; std::vector<Data> src;
        const char* kScript =
            "src = image(\"test\")\n"
            "op = choose(\"method\", [threshold_niblack, threshold_bernsen])\n"
            "m = params(op)(src)\n"
            "display(m)\n";
        Pipeline p1;
        RunScript(kScript, &ui, &p1, &err, &src);
        bool niblack = false;
        for (const UiControl& c : ui.Controls())
            if (c.label.find("niblack") != std::string::npos) niblack = true;
        Check(niblack, "the first method's parameters are exposed");

        if (UiControl* c = ui.Find("method")) c->selected = 1;   // switch method
        Pipeline p2;
        RunScript(kScript, &ui, &p2, &err, &src);

        bool stillNiblack = false, nowBernsen = false;
        for (const UiControl& c : ui.Controls()) {
            if (c.label.find("niblack") != std::string::npos) stillNiblack = true;
            if (c.label.find("bernsen") != std::string::npos) nowBernsen = true;
        }
        Check(nowBernsen, "switching the dropdown exposes the new method's parameters");
        Check(!stillNiblack, "the previous method's controls are dropped, not accumulated");
        Check(p2.Stages()[0].algoName == "threshold_bernsen",
              "the selected algorithm is the one that runs");
    }
    {
        // An explicit named argument must still beat the params() slider.
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        const bool ok = RunScript(
            "src = image(\"test\")\n"
            "m = params(threshold_niblack)(src, window = 31)\n"
            "display(m)\n", &ui, &p, &err, &src);
        Check(ok, "params() accepts an explicit override" + (ok ? "" : ": " + err));
        if (ok) {
            ParamBase* w = p.Stages()[0].algo->FindParam("window");
            Check(w && w->HashValue() == uint64_t(uint32_t(31)),
                  "the explicit argument wins over the declared control");
        }
    }
    {
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        RunScript("m = params(42)\n", &ui, &p, &err, &src);
        Check(!err.empty(), "params() rejects a non-algorithm: \"" + err + "\"");
    }

    // --- histogram and thresholding -----------------------------------------
    {
        // Bimodal: half at 60, half at 200. Every statistic has a known answer.
        ImageDesc d{64, 64, Format::RGBA8};
        Image img;
        img.Alloc(d);
        ImageView v = img.MapCpuWrite();
        for (int y = 0; y < 64; ++y)
            for (int x = 0; x < 64; ++x) {
                uint8_t* p = v.At<uint8_t>(x, y);
                const uint8_t g = (x < 32) ? 60 : 200;
                p[0] = p[1] = p[2] = g;
                p[3] = 255;
            }

        Histogram h;
        h.Build(v);
        Check(h.Count() == 64 * 64, "histogram counts every pixel");
        Check(std::fabs(h.Mean() - 130.0) < 2.0, "histogram mean is midway between the modes");
        Check(h.MinValue() <= 61 && h.MaxValue() >= 199, "histogram min/max span both modes");

        // Every threshold between the modes scores identically, so Otsu returns
        // the plateau midpoint. Comparing with an ABSOLUTE epsilon here is a
        // real bug: between-class variance runs to ~1e11, where 1e-9 is below
        // float resolution and the plateau collapses onto its first bin.
        const double otsu = h.OtsuThreshold();
        Check(otsu > 61 && otsu < 199,
              "Otsu splits between the modes, not on one (" + std::to_string(otsu) + ")");
        Check(std::fabs(h.IsoDataThreshold() - 130.0) < 12.0, "IsoData lands near the midpoint");

        LocalStats s = WindowStats(v, 10, 10, 3);
        Check(std::fabs(s.mean - 60.0) < 0.01 && s.stddev < 0.01,
              "uniform window has the region mean and zero stddev");

        // The integral image is only worth having if it agrees with the direct
        // computation it replaces.
        IntegralImage ii;
        ii.Build(v);
        LocalStats a = ii.Window(20, 20, 5);
        LocalStats b = WindowStats(v, 20, 20, 5);
        Check(std::fabs(a.mean - b.mean) < 0.01, "integral image mean matches direct");
        Check(std::fabs(a.stddev - b.stddev) < 0.01, "integral image stddev matches direct");
    }
    {
        // Every threshold algorithm must produce a usable mask, not all-on or
        // all-off, on an image that plainly has two regions.
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        const char* kMethods[] = {
            "threshold", "threshold_otsu", "threshold_triangle", "threshold_isodata",
            "threshold_niblack", "threshold_sauvola", "threshold_bernsen",
            "threshold_adaptive_mean", "threshold_adaptive_gaussian"};
        int degenerate = 0;
        for (const char* m : kMethods) {
            Pipeline pipe;
            UiState u;
            std::vector<Data> s;
            const std::string script =
                std::string("src = image(\"test\")\nm = ") + m + "(src)\ndisplay(m)\n";
            if (!RunScript(script.c_str(), &u, &pipe, &err, &s)) {
                Check(false, std::string(m) + " runs: " + err);
                continue;
            }
            ImageView o = std::get<Image>(pipe.Stages()[0].outputs[0]).MapCpuRead();
            int on = 0;
            const int n = o.desc.width * o.desc.height;
            for (int i = 0; i < n; ++i)
                if (reinterpret_cast<const float*>(o.data)[i] > 0.5f) ++on;
            if (on == 0 || on == n) ++degenerate;
        }
        Check(degenerate == 0, "no threshold method produced an all-on or all-off mask");
    }
    {
        // Categories must stay interchangeable: choose("x", "threshold") offers
        // all of these, so every one must take a single image.
        bool allOneInput = true;
        const auto names = Registry::Get().NamesInCategory("threshold");
        for (const std::string& n : names) {
            auto a = Registry::Get().Create(n);
            if (!a || a->Inputs().size() != 1) allOneInput = false;
        }
        Check(names.size() >= 9, "the threshold category has all the methods (" +
                                     std::to_string(names.size()) + ")");
        Check(allOneInput, "every threshold algorithm takes one image");
    }

    {
        // A params()-declared slider must reach the algorithm and re-run the
        // stage. This is what "the slider does nothing and everything shows as
        // cached" would look like if the value path were broken.
        //
        // Needs a bigger image than RunScript's 4x4: at sigma 2 the blur radius
        // already exceeds 4 px, so every pixel is the whole-image average and
        // raising sigma changes literally nothing.
        auto runOn = [](const char* script, UiState* ui, Pipeline* pipe,
                        std::vector<Data>* src, std::string* err) {
            ImageDesc d{64, 64, Format::RGBA8};
            Image img;
            img.Alloc(d);
            ImageView v = img.MapCpuWrite();
            for (int y = 0; y < 64; ++y)
                for (int x = 0; x < 64; ++x) {
                    uint8_t* p = v.At<uint8_t>(x, y);
                    const uint8_t g = ((x / 8) + (y / 8)) % 2 ? 230 : 20;
                    p[0] = p[1] = p[2] = g;
                    p[3] = 255;
                }
            src->clear();
            src->push_back(Data{std::move(img)});
            std::vector<SourceImage> names{{"test", 0}};
            Program prog;
            if (!Parse(script, &prog, err)) return false;
            auto r = Interpret(prog, names, ui, pipe);
            if (!r.ok) { *err = r.error; return false; }
            return pipe->Execute(src, nullptr, err);
        };

        auto checksum = [](Pipeline& p) {
            ImageView v = std::get<Image>(p.Stages()[0].outputs[0]).MapCpuRead();
            unsigned long long sum = 0;
            for (int i = 0; i < v.desc.width * v.desc.height * 4; ++i) sum += v.data[i];
            return sum;
        };

        UiState ui; std::string err; std::vector<Data> src;
        const char* kScript =
            "src = image(\"test\")\n"
            "prep = choose(\"preprocess\", [gaussian_blur, grayscale])\n"
            "out = params(prep)(src)\n"
            "display(out)\n";

        Pipeline p1;
        const bool ok1 = runOn(kScript, &ui, &p1, &src, &err);
        Check(ok1, "params() pipeline runs" + (ok1 ? "" : ": " + err));
        const unsigned long long before = checksum(p1);

        bool dragged = false;
        for (UiControl& c : ui.Controls())
            if (c.label.find("sigma") != std::string::npos) { c.value = 6.0; dragged = true; }
        Check(dragged, "params() declared a sigma control to drag");

        Pipeline p2;
        runOn(kScript, &ui, &p2, &src, &err);
        Check(checksum(p2) != before, "a changed params() value changes the output");
    }

    {
        // Swapping the image behind a palette slot must invalidate the stage
        // cache. PortRef{-1, 0} is identical whichever file backs slot 0, so
        // without a source version the pipeline reuses the cached output and
        // the app looks like it is ignoring the new image -- until some
        // unrelated parameter change incidentally busts the cache.
        auto flat = [](int dim, uint8_t v) {
            Image img;
            img.Alloc({dim, dim, Format::RGBA8});
            ImageView iv = img.MapCpuWrite();
            for (int i = 0; i < dim * dim; ++i) {
                iv.data[size_t(i) * 4 + 0] = v;
                iv.data[size_t(i) * 4 + 1] = v;
                iv.data[size_t(i) * 4 + 2] = v;
                iv.data[size_t(i) * 4 + 3] = 255;
            }
            return img;
        };
        auto checksum = [](Pipeline& p) {
            ImageView v = std::get<Image>(p.Stages()[0].outputs[0]).MapCpuRead();
            unsigned long long sum = 0;
            for (int i = 0; i < v.desc.width * v.desc.height * 4; ++i) sum += v.data[i];
            return sum;
        };

        const char* kScript =
            "src = image(\"test\")\n"
            "out = brightness(src, gain = 1.0)\n"
            "display(out)\n";
        std::vector<SourceImage> names{{"test", 0}};
        UiState ui;
        std::string err;
        Program prog;
        Parse(kScript, &prog, &err);

        std::vector<Data> s1;
        s1.push_back(Data{flat(32, 100)});
        std::vector<uint64_t> v1{1};
        Pipeline p1;
        Interpret(prog, names, &ui, &p1);
        p1.Execute(&s1, nullptr, &err, nullptr, ExecMode::Auto, &v1);
        const unsigned long long before = checksum(p1);

        // Same image, same version: must still cache.
        std::vector<Data> s2;
        s2.push_back(Data{flat(32, 100)});
        Pipeline p2;
        Interpret(prog, names, &ui, &p2);
        p2.Execute(&s2, &p1, &err, nullptr, ExecMode::Auto, &v1);
        Check(p2.CachedStageCount() == 1, "an unchanged source still hits the cache");

        // Different image, bumped version: what dropping a file on a slot does.
        std::vector<Data> s3;
        s3.push_back(Data{flat(32, 200)});
        std::vector<uint64_t> v3{2};
        Pipeline p3;
        Interpret(prog, names, &ui, &p3);
        p3.Execute(&s3, &p2, &err, nullptr, ExecMode::Auto, &v3);
        Check(p3.CachedStageCount() == 0, "a swapped source does not hit the cache");
        Check(checksum(p3) != before,
              "the output reflects the new image with no other change");
    }

    // --- params() with an instance name -------------------------------------
    //
    // The comparison case: the same algorithm twice, at different settings.
    // Keyed by algorithm name alone, the two calls shared one control set and
    // one pending-value entry, so the second silently had no controls at all.
    {
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        const bool ok = RunScript(
            "src = image(\"test\")\n"
            "a = params(threshold_niblack, \"A\")(src)\n"
            "b = params(threshold_niblack, \"B\")(src)\n"
            "display(a)\n"
            "display(b)\n", &ui, &p, &err, &src);
        Check(ok, "the same algorithm can be declared twice" + (ok ? "" : ": " + err));

        int aControls = 0, bControls = 0;
        for (const UiControl& c : ui.Controls()) {
            if (c.label.rfind("threshold_niblack@A.", 0) == 0) ++aControls;
            if (c.label.rfind("threshold_niblack@B.", 0) == 0) ++bControls;
        }
        Check(aControls > 0 && aControls == bControls,
              "each instance gets its own full set of controls");

        // Grouping and short names are what make the panel readable.
        bool grouped = false, shortNamed = false;
        for (const UiControl& c : ui.Controls()) {
            if (c.group == "A (threshold_niblack)") grouped = true;
            // The row shows "k", not "threshold_niblack@A.k".
            if (c.display == "k") shortNamed = true;
        }
        Check(grouped, "params() names a group per instance");
        Check(shortNamed, "controls carry a short display name distinct from the key");
    }
    {
        // Independence has to hold for the values, not just the labels: setting
        // one instance's parameter must not move the other's.
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        const bool ok = RunScript(
            "src = image(\"test\")\n"
            "a = params(threshold_niblack, \"A\")(src)\n"
            "b = params(threshold_niblack, \"B\")(src)\n"
            "display(a)\n"
            "display(b)\n", &ui, &p, &err, &src);
        Check(ok, "two instances run" + (ok ? "" : ": " + err));

        UiControl* ak = ui.Find("threshold_niblack@A.k");
        UiControl* bk = ui.Find("threshold_niblack@B.k");
        if (ak && bk) {
            ak->value = 0.5;
            Check(bk->value != 0.5, "changing instance A's k leaves B's alone");
        } else {
            Check(false, "both instances declared a 'k' control");
        }
    }
    {
        // Without a name, the old spelling must keep working unchanged.
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        const bool ok = RunScript(
            "src = image(\"test\")\n"
            "m = params(threshold_niblack)(src)\n"
            "display(m)\n", &ui, &p, &err, &src);
        Check(ok, "params() without a name still works" + (ok ? "" : ": " + err));
        Check(ui.Find("threshold_niblack.k") != nullptr,
              "an unnamed params() keys controls by algorithm name as before");
    }

    // --- parameter help text ------------------------------------------------
    //
    // Help is declared on the Param and has to survive the trip through
    // DescribeControl into the UiControl the panel draws from, or the tooltip
    // silently shows nothing.
    {
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        const bool ok = RunScript(
            "src = image(\"test\")\n"
            "m = params(threshold_sauvola, \"S\")(src)\n"
            "display(m)\n", &ui, &p, &err, &src);
        Check(ok, "a script using a documented algorithm runs" + (ok ? "" : ": " + err));

        const UiControl* k = ui.Find("threshold_sauvola@S.k");
        Check(k && !k->help.empty(), "help text reaches the control the panel draws");

        // Every filter parameter should be documented: an undocumented one is
        // exactly the case the user could not interpret.
        int undocumented = 0;
        std::string missing;
        for (const std::string& name : Registry::Get().Names()) {
            auto a = Registry::Get().Create(name);
            if (!a || std::string(a->Category()) != "filter") continue;
            for (ParamBase* pb : a->Params())
                if (!pb->Help() || !*pb->Help()) {
                    ++undocumented;
                    missing += " " + name + "." + pb->Name();
                }
        }
        Check(undocumented == 0, "every filter parameter has help text" +
                                     (undocumented ? ":" + missing : ""));
    }

    // --- parameter step / soft range ---------------------------------------
    //
    // Snapping lives in Param::set(), so it must apply to script and typed
    // values too, not only to dragging.
    {
        struct Dummy : AlgorithmBase {
            const char* Name() const override { return "dummy"; }
            PortList Inputs()  const override { return {}; }
            PortList Outputs() const override { return {}; }
            void RunCPU(RunCtx&) override {}
            // Odd window sizes, as the local-threshold algorithms declare them.
            Param<int>   window{this, "window", 15, 3, 201, {.step = 2, .softMin = 3, .softMax = 51}};
            Param<float> k     {this, "k", -0.2f, -1.0f, 1.0f, {.step = 0.01}};
            Param<float> plain {this, "plain", 0.5f, 0.0f, 1.0f};
        } d;

        d.window.set(14);
        Check(d.window.get() == 15, "an even window snaps to the odd grid");
        d.window.set(20);
        Check(d.window.get() == 21, "snapping measures from the minimum, not zero");

        d.k.set(-0.184f);
        Check(std::abs(d.k.get() - (-0.18f)) < 1e-5f, "k snaps to hundredths");

        d.plain.set(0.3456f);
        Check(std::abs(d.plain.get() - 0.3456f) < 1e-6f,
              "a parameter with no step stays continuous");

        // The slider spans the soft range; the full range remains settable.
        Check(d.window.SliderMin() == 3 && d.window.SliderMax() == 51,
              "the slider covers the soft range when one is declared");
        d.window.set(199);
        Check(d.window.get() == 199, "values beyond the soft range are still reachable");
        d.window.set(500);
        Check(d.window.get() == 201, "the hard maximum still clamps");

        Check(d.k.SliderMin() == -1.0f && d.k.SliderMax() == 1.0f,
              "without a soft range the slider covers the full range");
    }

    std::printf("\n%s\n", g_fail == 0 ? "all checks passed" : "FAILURES PRESENT");
    return g_fail == 0 ? 0 : 1;
}
