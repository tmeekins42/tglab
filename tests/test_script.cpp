// Headless checks for the script spine: parse -> interpret -> execute.
// Runs without a window, so failures here are quick to diagnose.
#include <cstdio>
#include <string>
#include <vector>

#include "../src/core/algorithm.h"
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

    std::printf("\n%s\n", g_fail == 0 ? "all checks passed" : "FAILURES PRESENT");
    return g_fail == 0 ? 0 : 1;
}
