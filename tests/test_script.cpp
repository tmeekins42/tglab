// Headless checks for the script spine: parse -> interpret -> execute.
// Runs without a window, so failures here are quick to diagnose.
#include <cmath>
#include <cstdio>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <atomic>
#include <thread>
#include <string>
#include <vector>

#include "../src/core/algorithm.h"
#include "../src/algo_util/features.h"
#include "../src/algo_util/histogram.h"
#include "../src/core/exif.h"
#include "../src/algo_util/pixel_buffer.h"
#include "../src/algo_util/tone_curve.h"
#include "../src/algo_util/transform.h"
#include "../src/core/image_group.h"
#include "../src/core/image_loader.h"
#include "../src/core/image_io.h"
#include "../src/core/image_stats.h"
#include "../src/core/raw_io.h"
#include "../src/core/cancel.h"
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

// Writes a minimal JPEG carrying EXIF with known values, so ReadExif can be
// checked against ground truth. Hand-built rather than committed as a binary:
// the expected values are then visible right here beside the assertions.
static bool WriteExifFixture(const std::string& path) {
    using B = std::vector<uint8_t>;
    auto u16 = [](B& b, uint16_t v) { b.push_back(uint8_t(v)); b.push_back(uint8_t(v >> 8)); };
    auto u32 = [](B& b, uint32_t v) { for (int i = 0; i < 4; ++i) b.push_back(uint8_t(v >> (8 * i))); };

    const char* make  = "TestCam";
    const char* model = "Model X100";
    const char* lens  = "TestLens 50mm";

    B tiff;
    tiff.push_back('I'); tiff.push_back('I');   // little-endian
    u16(tiff, 42);
    u32(tiff, 8);                                // IFD0 immediately follows

    // Values longer than 4 bytes sit in a heap after both IFDs, so their
    // offsets have to be computed before the entries are written.
    const uint32_t ifd0Size  = 2 + 3 * 12 + 4;
    const uint32_t exifIfdAt = 8 + ifd0Size;
    const uint32_t exifCount = 5;
    const uint32_t exifSize  = 2 + exifCount * 12 + 4;
    uint32_t heap = exifIfdAt + exifSize;

    const uint32_t makeAt  = heap; heap += uint32_t(std::strlen(make) + 1);
    const uint32_t modelAt = heap; heap += uint32_t(std::strlen(model) + 1);
    const uint32_t lensAt  = heap; heap += uint32_t(std::strlen(lens) + 1);
    const uint32_t expAt   = heap; heap += 8;
    const uint32_t fnumAt  = heap; heap += 8;
    const uint32_t focAt   = heap; heap += 8;

    u16(tiff, 3);
    u16(tiff, 0x010F); u16(tiff, 2); u32(tiff, uint32_t(std::strlen(make) + 1));  u32(tiff, makeAt);
    u16(tiff, 0x0110); u16(tiff, 2); u32(tiff, uint32_t(std::strlen(model) + 1)); u32(tiff, modelAt);
    u16(tiff, 0x8769); u16(tiff, 4); u32(tiff, 1);                                u32(tiff, exifIfdAt);
    u32(tiff, 0);                                // no IFD1

    u16(tiff, uint16_t(exifCount));
    u16(tiff, 0x829A); u16(tiff, 5); u32(tiff, 1); u32(tiff, expAt);   // ExposureTime
    u16(tiff, 0x829D); u16(tiff, 5); u32(tiff, 1); u32(tiff, fnumAt);  // FNumber
    u16(tiff, 0x8827); u16(tiff, 3); u32(tiff, 1); u32(tiff, 800);     // ISO, stored inline
    u16(tiff, 0x920A); u16(tiff, 5); u32(tiff, 1); u32(tiff, focAt);   // FocalLength
    u16(tiff, 0xA434); u16(tiff, 2); u32(tiff, uint32_t(std::strlen(lens) + 1)); u32(tiff, lensAt);
    u32(tiff, 0);

    auto str = [&](const char* s) {
        for (const char* p = s; *p; ++p) tiff.push_back(uint8_t(*p));
        tiff.push_back(0);
    };
    str(make); str(model); str(lens);
    u32(tiff, 1);  u32(tiff, 250);   // 1/250 s
    u32(tiff, 28); u32(tiff, 10);    // f/2.8
    u32(tiff, 50); u32(tiff, 1);     // 50 mm

    B jpg{0xFF, 0xD8, 0xFF, 0xE1};
    const uint32_t segLen = uint32_t(tiff.size()) + 8;
    jpg.push_back(uint8_t(segLen >> 8));
    jpg.push_back(uint8_t(segLen));
    const char sig[6] = {'E', 'x', 'i', 'f', 0, 0};
    jpg.insert(jpg.end(), sig, sig + 6);
    jpg.insert(jpg.end(), tiff.begin(), tiff.end());
    jpg.push_back(0xFF); jpg.push_back(0xD9);

    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    std::fwrite(jpg.data(), 1, jpg.size(), f);
    std::fclose(f);
    return true;
}

int main() {
    // Unbuffered, so a crash does not swallow the output that would say where.
    setvbuf(stdout, nullptr, _IONBF, 0);
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
            "out = brightness(src, brightness = a)\n"
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
                  "o = brightness(src, brightness = a)\ndisplay(o)\n", &ui, &p1, &err, &src);
        ui.Controls()[0].value = 7.0;    // user drags the slider
        RunScript("a = slider(\"amount\", 0, 10, 1)\nsrc = image(\"test\")\n"
                  "o = brightness(src, brightness = a)\ndisplay(o)\n", &ui, &p2, &err, &src);
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

        // Read through Resolve rather than reaching into Stages()[0].outputs.
        //
        // A stage can now be BYPASSED -- skipped because its settings would
        // change nothing -- and a bypassed stage holds no image at all: its
        // result is whatever its input resolves to. This script starts with
        // gain = 1, which is exactly that case, so indexing the outputs
        // directly reads an empty variant. Resolve() is the accessor that
        // knows about the alias, and is what every non-test caller uses.
        auto firstPixel = [&](Pipeline& p, std::vector<Data>& s) -> int {
            const Data* d = p.Resolve({0, 0}, &s);
            const auto* im = d ? std::get_if<Image>(d) : nullptr;
            if (!im) return -1;
            return int(const_cast<Image*>(im)->MapCpuRead().data[4]);
        };

        Pipeline p1;
        RunScript(kScript, &ui, &p1, &err, &src);
        const int before = firstPixel(p1, src);

        // Simulate the user dragging the slider.
        ui.Controls()[0].value = 3.0;

        Pipeline p2;
        RunScript(kScript, &ui, &p2, &err, &src);
        const int after = firstPixel(p2, src);

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
        // Through Resolve, not Stages()[0].outputs: gain = 1.0 with no offset
        // is exactly the identity, so this stage is BYPASSED and holds no
        // image of its own -- its result is the palette source it aliases.
        //
        // Which is exactly what this test needs to keep working: swapping the
        // image behind the slot must still be seen, bypass or not. A bypassed
        // stage aliasing a source that then changes is the one case where the
        // alias could go stale, so it is worth exercising rather than avoiding.
        auto checksum = [](Pipeline& p, std::vector<Data>& s) -> unsigned long long {
            const Data* d = p.Resolve({0, 0}, &s);
            const auto* im = d ? std::get_if<Image>(d) : nullptr;
            if (!im) return 0ull;
            ImageView v = const_cast<Image*>(im)->MapCpuRead();
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
        const unsigned long long before = checksum(p1, s1);

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
        Check(checksum(p3, s3) != before,
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

    // --- EXIF ---------------------------------------------------------------
    //
    // Built here rather than checked against a sample file, so the expected
    // values are known exactly. stb_image discards metadata, so this parser is
    // the only source of capture settings, and a silent misparse would show
    // plausible-but-wrong numbers in the info panel.
    {
        const std::string path = "exif_test_fixture.jpg";
        Check(WriteExifFixture(path), "wrote a JPEG carrying known EXIF");

        const ExifData e = ReadExif(path);
        Check(e.present, "EXIF is detected");
        Check(e.cameraMake == "TestCam", "make: " + e.cameraMake);
        Check(e.cameraModel == "Model X100", "model: " + e.cameraModel);
        Check(e.lens == "TestLens 50mm", "lens: " + e.lens);
        // Rationals, the part most easily got wrong: 1/250, 28/10, 50/1.
        Check(e.exposureTime == "1/250 s", "exposure: " + e.exposureTime);
        Check(e.aperture == "f/2.8", "aperture: " + e.aperture);
        Check(e.focalLength == "50 mm", "focal length: " + e.focalLength);
        // A SHORT stored inline in the offset field rather than at an offset.
        Check(e.iso == "ISO 800", "iso: " + e.iso);

        // The loader reads EXIF alongside the pixels, so the info panel does not
        // have to open the file again on the UI thread.
        //
        // Checked through the loader rather than by calling ReadExif twice: the
        // point is that the DELIVERED result carries it, and a LoadResult that
        // quietly left the field default would look identical from outside.
        {
            ImageLoader loader;
            loader.Start();
            loader.Request(path, "");

            LoadResult got;
            bool have = false;
            for (int i = 0; i < 2000 && !have; ++i) {
                have = loader.TryFetch(&got);
                if (!have) std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
            loader.Stop();

            Check(have, "the loader delivered the fixture");
            // The fixture is a JPEG carrying only an EXIF block, so the decode
            // itself is expected to fail -- which is the interesting case:
            // metadata must survive a file whose pixels do not.
            Check(have && got.exif.present, "the load carries EXIF with it");
            Check(have && got.exif.iso == "ISO 800",
                  "the delivered EXIF is the right file's: " + got.exif.iso);
        }

        std::remove(path.c_str());
    }
    {
        // Absence and malformation are normal inputs, not errors: a PNG has no
        // EXIF, and a truncated header must not read past the buffer.
        Check(!ReadExif("assets/test.png").present, "a PNG reports no EXIF");
        Check(!ReadExif("no_such_file_at_all.jpg").present,
              "a missing file reports no EXIF rather than failing");

        const std::string junk = "exif_truncated_fixture.jpg";
        if (FILE* f = std::fopen(junk.c_str(), "wb")) {
            // A JPEG that claims EXIF and then simply stops.
            const unsigned char bytes[] = {0xFF, 0xD8, 0xFF, 0xE1, 0x00, 0x40,
                                           'E', 'x', 'i', 'f', 0, 0, 'I', 'I', 42, 0};
            std::fwrite(bytes, 1, sizeof bytes, f);
            std::fclose(f);
            Check(!ReadExif(junk).present, "a truncated EXIF header is handled safely");
            std::remove(junk.c_str());
        }
    }

    // --- controls stay in declaration order ---------------------------------
    //
    // The panel draws groups in the order UiState holds them, so that order
    // must follow the script. FindOrAdd() used to append new controls and leave
    // existing ones alone, which meant re-declaring a control moved it to the
    // end: switching filter A's algorithm dropped A's controls and re-added
    // them *after* B's, silently swapping the two groups in the panel.
    {
        UiState ui; std::string err; std::vector<Data> src;
        const char* kScript =
            "src = image(\"test\")\n"
            "a = params(threshold_niblack, \"A\")(src)\n"
            "b = params(threshold_sauvola, \"B\")(src)\n"
            "display(a)\n"
            "display(b)\n";

        Pipeline p1;
        RunScript(kScript, &ui, &p1, &err, &src);

        auto firstGroup = [&](UiState& s) {
            for (const UiControl& c : s.Controls())
                if (!c.group.empty()) return c.group;
            return std::string();
        };
        Check(firstGroup(ui).rfind("A ", 0) == 0,
              "A's group comes first: " + firstGroup(ui));

        // Now the case that broke it: A changes algorithm, B does not.
        const char* kSwitched =
            "src = image(\"test\")\n"
            "a = params(threshold_bernsen, \"A\")(src)\n"
            "b = params(threshold_sauvola, \"B\")(src)\n"
            "display(a)\n"
            "display(b)\n";
        Pipeline p2;
        RunScript(kSwitched, &ui, &p2, &err, &src);

        Check(firstGroup(ui).rfind("A ", 0) == 0,
              "A's group still comes first after A changes algorithm: " +
                  firstGroup(ui));
    }

    // --- a script error survives a re-run ------------------------------------
    //
    // The app clears m_error when a new image load starts, so a failed drop's
    // message does not linger over the next (possibly slow) load. That must not
    // lose a *script* error, which is still true regardless of what is loading:
    // the drop triggers a re-run, and the re-run has to report it again.
    {
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        RunScript("src = image(\"test\")\nout = brightness(src, nosuchparam = 1)\n",
                  &ui, &p, &err, &src);
        Check(!err.empty(), "a bad script reports an error");

        // Re-running the same bad script must report it again rather than
        // succeeding silently.
        UiState ui2; Pipeline p2; std::string err2; std::vector<Data> src2;
        RunScript("src = image(\"test\")\nout = brightness(src, nosuchparam = 1)\n",
                  &ui2, &p2, &err2, &src2);
        Check(err2 == err, "the same error is reported on every run, not just the first");
    }

    // --- choose() takes a default -------------------------------------------
    //
    // Without one the initial selection is whichever name sorts first in the
    // registry, which is alphabetical and therefore arbitrary. A script should
    // be able to say which method it means, with the dropdown still overriding.
    {
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        const bool ok = RunScript(
            "src = image(\"test\")\n"
            "op = choose(\"method\", \"threshold\", threshold_otsu)\n"
            "m = op(src)\n"
            "display(m)\n", &ui, &p, &err, &src);
        Check(ok, "choose() accepts a default" + (ok ? "" : ": " + err));

        const UiControl* c = ui.Find("method");
        Check(c && !c->options.empty(), "the dropdown was declared");
        if (c) {
            Check(c->options[size_t(c->selected)] == "threshold_otsu",
                  "the named default is selected, not the alphabetical first");
            Check(c->options[size_t(c->defaultIndex)] == "threshold_otsu",
                  "and a reset returns to it");
        }
    }
    {
        // A default naming something not in the list is a script error, not a
        // silent fallback -- a typo should say so.
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        RunScript("src = image(\"test\")\n"
                  "op = choose(\"method\", \"threshold\", gaussian_blur)\n"
                  "display(op(src))\n", &ui, &p, &err, &src);
        Check(err.find("not one of the options") != std::string::npos,
              "a default outside the list is reported: \"" + err + "\"");
    }
    {
        // The old two-argument form must keep working unchanged.
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        const bool ok = RunScript(
            "src = image(\"test\")\n"
            "op = choose(\"method\", \"threshold\")\n"
            "display(op(src))\n", &ui, &p, &err, &src);
        Check(ok, "choose() without a default still works" + (ok ? "" : ": " + err));
    }

    // --- image() demosaics a mosaic automatically ---------------------------
    //
    // The requirement: a script must not have to mention demosaicing on the
    // chance a raw file is dropped on it. image() inserts the default method
    // for a mosaic source and does nothing for an ordinary one, so the same
    // script works either way.
    {
        ImageDesc d{4, 4, Format::R32F};
        d.cfa = CfaPattern::RGGB;
        Image mos;
        mos.Alloc(d);

        std::vector<Data> src;
        src.push_back(Data{std::move(mos)});
        std::vector<SourceImage> names;
        { SourceImage s; s.name = "test"; s.index = 0; s.isMosaic = true; names.push_back(s); }

        Program prog;
        std::string err;
        Check(Parse("src = image(\"test\")\ndisplay(src)\n", &prog, &err),
              "a plain script parses");
        UiState ui; Pipeline p;
        const auto r = Interpret(prog, names, &ui, &p);
        Check(r.ok, "it interprets against a mosaic source" + (r.ok ? "" : ": " + r.error));
        // Asserts a demosaic was inserted, not *which* one: the default is an
        // app setting, and pinning the name here would make changing it a test
        // failure rather than a decision.
        // A mosaic gets TWO automatic stages: hot_pixel_repair, then the
        // demosaic. Repair runs first because a demosaic smears a stuck sensel
        // across a neighbourhood, after which it is no longer a single-sample
        // outlier. Both are ordinary stages, so either can be inspected,
        // retuned, or switched off -- which matters for astrophotography, where
        // a star is a genuine one-pixel highlight.
        Check(p.Stages().size() == 2 &&
                  p.Stages()[0].algoName == "hot_pixel_repair" &&
                  p.Stages()[1].algoName.rfind("demosaic_", 0) == 0,
              "image() inserted hot_pixel_repair then the default demosaic"
              + (p.Stages().size() < 2
                     ? std::string()
                     : " (" + p.Stages()[0].algoName + ", " +
                           p.Stages()[1].algoName + ")"));

        // Two image() calls on one source must share the stages rather than
        // repairing and demosaicing the same sensor data twice.
        Program prog2;
        Parse("a = image(\"test\")\nb = image(\"test\")\ndisplay(a)\ndisplay(b)\n",
              &prog2, &err);
        UiState ui2; Pipeline p2;
        Interpret(prog2, names, &ui2, &p2);
        Check(p2.Stages().size() == 2,
              "two image() calls share one repair+demosaic pair, not two");
    }
    {
        // An ordinary image gets no demosaic stage at all.
        std::vector<Data> src;
        Image plain;
        plain.Alloc({4, 4, Format::RGBA8});
        src.push_back(Data{std::move(plain)});
        std::vector<SourceImage> names{{"test", 0, false}};

        Program prog;
        std::string err;
        Parse("src = image(\"test\")\ndisplay(src)\n", &prog, &err);
        UiState ui; Pipeline p;
        Interpret(prog, names, &ui, &p);
        Check(p.Stages().empty(), "an ordinary image gets no demosaic stage");
    }
    {
        // mosaic() is the explicit counterpart, and must refuse a non-raw
        // source rather than hand back something that merely looks like one.
        std::vector<Data> src;
        Image plain;
        plain.Alloc({4, 4, Format::RGBA8});
        src.push_back(Data{std::move(plain)});
        std::vector<SourceImage> names{{"test", 0, false}};

        Program prog;
        std::string err;
        Parse("m = mosaic(\"test\")\ndisplay(m)\n", &prog, &err);
        UiState ui; Pipeline p;
        const auto r = Interpret(prog, names, &ui, &p);
        Check(!r.ok && r.error.find("not a raw image") != std::string::npos,
              "mosaic() on a non-raw image explains itself: \"" + r.error + "\"");
    }

    // --- half float ---------------------------------------------------------
    //
    // RGBA16F is the working format for raw, so a rounding error here would
    // quietly cost precision everywhere rather than failing visibly.
    {
        Check(HalfToFloat(FloatToHalf(0.0f)) == 0.0f, "half round-trips zero");
        Check(HalfToFloat(FloatToHalf(1.0f)) == 1.0f, "half round-trips one");
        Check(std::abs(HalfToFloat(FloatToHalf(-1.0f)) + 1.0f) < 1e-6f,
              "half round-trips a negative");

        // Within half's precision across the range image data actually uses.
        double worst = 0.0;
        for (int i = 0; i <= 1000; ++i) {
            const float f = float(i) / 1000.0f;
            worst = std::max(worst, std::abs(double(HalfToFloat(FloatToHalf(f))) - f));
        }
        // Half has ~11 bits of mantissa, so 1e-3 is generous; a real bug here
        // (a shifted exponent, say) would be off by orders of magnitude.
        Check(worst < 1e-3, "half round-trips 0..1 within precision (worst " +
                                std::to_string(worst) + ")");

        // A 14-bit sensor's smallest step must survive, since that is the whole
        // reason for choosing half over 8-bit.
        const float step = 1.0f / 16384.0f;
        Check(HalfToFloat(FloatToHalf(step)) > 0.0f,
              "a 14-bit sensor's smallest step does not flush to zero");

        // Overflow saturates rather than producing inf, which would poison
        // every later average.
        Check(HalfToFloat(FloatToHalf(1e30f)) > 60000.0f &&
              std::isfinite(HalfToFloat(FloatToHalf(1e30f))),
              "overflow saturates rather than becoming infinity");
    }

    // --- CFA pattern --------------------------------------------------------
    {
        // RGGB: R G / G B, as an RGB channel index.
        Check(CfaColorAt(CfaPattern::RGGB, 0, 0) == 0, "RGGB top-left is red");
        Check(CfaColorAt(CfaPattern::RGGB, 1, 0) == 1, "RGGB top-right is green");
        Check(CfaColorAt(CfaPattern::RGGB, 0, 1) == 1, "RGGB bottom-left is green");
        Check(CfaColorAt(CfaPattern::RGGB, 1, 1) == 2, "RGGB bottom-right is blue");
        // BGGR is RGGB with red and blue exchanged.
        Check(CfaColorAt(CfaPattern::BGGR, 0, 0) == 2, "BGGR top-left is blue");
        Check(CfaColorAt(CfaPattern::BGGR, 1, 1) == 0, "BGGR bottom-right is red");
        // The pattern tiles, so (2,2) must match (0,0).
        Check(CfaColorAt(CfaPattern::GRBG, 2, 2) == CfaColorAt(CfaPattern::GRBG, 0, 0),
              "the pattern tiles every 2 pixels");
        // Half the samples are green in every Bayer layout.
        for (CfaPattern p : {CfaPattern::RGGB, CfaPattern::BGGR,
                             CfaPattern::GRBG, CfaPattern::GBRG}) {
            int greens = 0;
            for (int y = 0; y < 2; ++y)
                for (int x = 0; x < 2; ++x)
                    if (CfaColorAt(p, x, y) == 1) ++greens;
            Check(greens == 2, std::string(CfaPatternName(p)) + " has two greens per tile");
        }
    }

    // --- mosaic metadata rides on ImageDesc ---------------------------------
    {
        ImageDesc plain{4, 4, Format::RGBA8};
        Check(!plain.IsMosaic(), "an ordinary image is not a mosaic");

        ImageDesc mosaic{4, 4, Format::R32F};
        mosaic.cfa = CfaPattern::RGGB;
        Check(mosaic.IsMosaic(), "a CFA pattern makes it a mosaic");

        // The stage cache compares descriptors, so a mosaic and an ordinary
        // image of the same size must not look interchangeable.
        ImageDesc sameSize{4, 4, Format::R32F};
        Check(!(mosaic == sameSize),
              "a mosaic and a plain image of the same size compare unequal");
    }

    // --- camera raw ---------------------------------------------------------
    //
    // Only the dispatch is tested here: decoding needs a real camera file,
    // which is far too large to commit. What must hold without one is that raw
    // files are routed to LibRaw, everything else is not, and a file that
    // cannot be read reports why instead of failing silently.
    {
        Check(IsRawExtension("shot.CR3"), "CR3 is recognised as raw");
        Check(IsRawExtension("shot.cr2"), "extension matching is case-insensitive");
        Check(IsRawExtension("a/b/c.NEF"), "a path prefix does not confuse it");
        Check(!IsRawExtension("scan.png"), "PNG is not raw");
        Check(!IsRawExtension("scan.jpg"), "JPEG is not raw");
        Check(!IsRawExtension("noextension"), "a file with no extension is not raw");
        // A dot in a directory name is not an extension, which would otherwise
        // send "my.photos/scan" (no extension) down the raw path.
        Check(!IsRawExtension("my.raw.folder/image"),
              "a dot in a directory name is not an extension");

        Image img;
        std::string err;
        Check(!LoadImageFile("no_such_file.cr3", &img, &err),
              "a missing raw file fails rather than returning an empty image");
        Check(err.find("no_such_file.cr3") != std::string::npos,
              "the error names the file: \"" + err + "\"");
    }

    // --- image statistics ---------------------------------------------------
    //
    // The info panel's histogram is computed off a subsample, so the statistics
    // it reports must still match the full image. If they drift, the panel is
    // quietly lying about the data a filter is being tuned against.
    {
        // A gradient plus a black and a white block, so mean, spread and
        // clipping all have known-ish values to compare against.
        Image img;
        img.Alloc({800, 600, Format::RGBA8});
        ImageView v = img.MapCpuWrite();
        for (int y = 0; y < 600; ++y)
            for (int x = 0; x < 800; ++x) {
                uint8_t* p = v.At<uint8_t>(x, y);
                uint8_t s = uint8_t(x * 255 / 799);
                if (y < 60)  s = 0;      // 10% pure black
                if (y >= 540) s = 255;   // 10% pure white
                p[0] = p[1] = p[2] = s;
                p[3] = 255;
            }

        Histogram full;
        full.Build(img.MapCpuRead(), -1);

        ImageStats stats;
        stats.Start();
        stats.Request(img, "test", 1);

        StatsResult r;
        for (int i = 0; i < 500 && !stats.TryFetch(&r); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        stats.Stop();

        Check(r.valid, "the stats worker returns a result");
        if (r.valid) {
            // Within a bin's width: subsampling shifts which exact pixels are
            // counted, not the distribution they came from.
            Check(std::abs(r.mean - full.Mean()) < 2.0,
                  "subsampled mean matches the full image (" +
                      std::to_string(r.mean) + " vs " + std::to_string(full.Mean()) + ")");
            Check(std::abs(r.stddev - full.StdDev()) < 2.0,
                  "subsampled stddev matches the full image");
            // Clipping is the one thing subsampling could plausibly hide, and
            // it is the reason to point-sample rather than average.
            Check(r.clipLow > 0.08 && r.clipLow < 0.12,
                  "black clipping is measured (~10%): " + std::to_string(r.clipLow));
            Check(r.clipHigh > 0.08 && r.clipHigh < 0.12,
                  "white clipping is measured (~10%): " + std::to_string(r.clipHigh));
            Check(r.r.size() == 256 && r.luma.size() == 256,
                  "per-channel and luma bins are populated");
        }
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


    // --- a control follows a changed default, unless the user moved it -------
    //
    // Almost every parameter's default is fixed, so FindOrAdd could simply
    // preserve the current value across re-runs. kelvin's is not: it comes from
    // the image, and on startup the control is created before the raw finishes
    // loading. It was therefore built at 0 and preserved forever -- the slider
    // read 0 K on a file whose metadata says 5381.
    //
    // Reported twice: once as "the kelvin slider defaulted to 0, which was
    // confusing", and again after I fixed the recovery but not this.
    {
        UiState ui;

        UiControl proto;
        proto.kind  = UiControl::Kind::Slider;
        proto.label = "basic_adjust.kelvin";
        proto.lo    = 0.0;
        proto.hi    = 25000.0;
        proto.def   = 0.0;      // as declared, before any image is known
        proto.value = 0.0;

        ui.BeginRun();
        UiControl& first = ui.FindOrAdd(proto);
        Check(first.value == 0.0, "a fresh control takes the declared default");

        // The image arrives and the default becomes the camera's temperature.
        UiControl updated = proto;
        updated.def   = 5381.0;
        updated.value = 5381.0;

        ui.BeginRun();
        UiControl& second = ui.FindOrAdd(updated);
        Check(second.value == 5381.0,
              "an untouched control follows the new default (got " +
                  std::to_string(second.value) + ", want 5381)");
        Check(second.def == 5381.0, "...and reports it as the default");

        // But a value the user set is theirs. Move it, then change the default
        // again: the control must not jump.
        second.value = 3200.0;
        UiControl moved = updated;
        moved.def   = 6504.0;
        moved.value = 6504.0;

        ui.BeginRun();
        UiControl& third = ui.FindOrAdd(moved);
        Check(third.value == 3200.0,
              "a control the user moved keeps its value when the default changes "
              "(got " + std::to_string(third.value) + ", want 3200)");
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
    // --- shape ---------------------------------------------------------------
    //
    // Shape is introduced with nothing in the registry producing a non-scalar
    // value, so a test written only against real algorithms would pass no
    // matter what the check did. These two exist purely so the mismatch can
    // actually occur -- the same trap that let three earlier fixtures pass
    // against a broken implementation.
    {
        // Produces a set: 5 exposures on one named axis.
        struct FakeBracket : AlgorithmBase {
            const char* Name() const override { return "test_bracket"; }
            PortList Inputs()  const override { return {{"src"}}; }
            PortList Outputs() const override {
                return {{"out", DataType::ImageSet, FormatSpec::Any, ShapeSpec::SameAsInput}};
            }
            void RunCPU(RunCtx&) override {}
        };
        // Reduces a set back to one image.
        struct FakeMerge : AlgorithmBase {
            const char* Name() const override { return "test_merge"; }
            PortList Inputs()  const override {
                return {{"src", DataType::ImageSet, FormatSpec::Any, ShapeSpec::Any}};
            }
            PortList Outputs() const override {
                return {{"out", DataType::Image, FormatSpec::Any, ShapeSpec::Reduced}};
            }
            void RunCPU(RunCtx&) override {}
        };
        Registry::Get().Add("test_bracket",
            []() -> std::unique_ptr<AlgorithmBase> { return std::make_unique<FakeBracket>(); });
        Registry::Get().Add("test_merge",
            []() -> std::unique_ptr<AlgorithmBase> { return std::make_unique<FakeMerge>(); });

        auto Build = [](const char* src, std::string* err) {
            Program prog;
            if (!Parse(src, &prog, err)) return false;
            std::vector<SourceImage> names{{"test", 0}};
            UiState ui; Pipeline pipe;
            InterpResult r = Interpret(prog, names, &ui, &pipe);
            if (!r.ok) { *err = r.error; return false; }
            return true;
        };

        std::string err;

        // The whole point: an existing single-image algorithm must still be
        // reachable through a plain palette image.
        Check(Build("src = image(\"test\")\no = brightness(src)\ndisplay(o)\n", &err),
              "a scalar image still binds to a scalar port");

        // FakeBracket declares SameAsInput against a scalar input, so its
        // output is still scalar -- shape only appears once a real producer
        // exists. This documents the current state rather than asserting a
        // set was made.
        Check(Build("src = image(\"test\")\nb = test_bracket(src)\no = brightness(b)\n", &err),
              "SameAsInput over a scalar input stays scalar");

        // And the check itself. Forced directly, because nothing in the
        // registry yet produces a non-scalar port -- without this the test
        // could not distinguish a working check from no check at all.
        {
            Shape s = Shape::Of("exposure", 5);
            Check(!s.IsScalar() && s.Count() == 5 && s.Rank() == 1,
                  "a named axis describes 5 images");
            Check(s.ToString() == "[exposure=5]", "shape prints its axis names");
            Check(s.Find("exposure") == 0 && s.Find("focus") == -1,
                  "an axis is found by name");
            Check(s.Without(0).IsScalar(), "reducing the only axis gives a scalar");

            Shape two{{Axis{"exposure", 5}, Axis{"focus", 12}}};
            Check(two.Count() == 60, "two axes multiply");
            Check(two.Without(two.Find("exposure")) == Shape::Of("focus", 12),
                  "reducing by name removes that axis and keeps the other");
            Check(Shape::Scalar().Count() == 1,
                  "a scalar is one image, not none");
        }

        // End to end, with a REAL set in the palette rather than a forced
        // shape. This is what makes the check live: image("bracket") returns a
        // port carrying [exposure=5], and handing it to a single-image
        // algorithm must fail at the line that did it.
        {
            auto MakeSet = [](int n) {
                ImageSet s;
                s.shape = Shape::Of("exposure", n);
                for (int i = 0; i < n; ++i) {
                    Image img;
                    ImageDesc d; d.width = 4; d.height = 4; d.format = Format::RGBA8;
                    img.Alloc(d);
                    s.images.push_back(std::move(img));
                }
                return s;
            };

            std::vector<SourceImage> names{{"test", 0}};
            SourceImage bset;
            bset.name  = "bracket";
            bset.index = 1;
            bset.shape = Shape::Of("exposure", 5);
            names.push_back(bset);

            auto BuildWith = [&](const char* src, std::string* err) {
                Program prog;
                if (!Parse(src, &prog, err)) return false;
                UiState ui; Pipeline pipe;
                InterpResult r = Interpret(prog, names, &ui, &pipe);
                if (!r.ok) { *err = r.error; return false; }
                return true;
            };

            std::string e2;

            // A set on a single-image port now BROADCASTS rather than being
            // rejected: the framework maps the algorithm across the frames.
            // This assertion is inverted from its first version deliberately --
            // rejecting was correct only while broadcasting did not exist.
            Check(BuildWith("b = image(\"bracket\")\no = brightness(b)\n", &e2),
                  "a set on a single-image port broadcasts");

            // The same set into a port declaring Any is fine.
            Check(BuildWith("b = image(\"bracket\")\no = test_merge(b)\n", &e2),
                  "a set binds to a port that accepts any shape");

            // And a scalar palette image still works, which is the property
            // every existing script depends on.
            Check(BuildWith("s = image(\"test\")\no = brightness(s)\n", &e2),
                  "a scalar palette image still binds to a scalar port");

            // The reduction's output is scalar again, so it chains into
            // ordinary single-image algorithms.
            Check(BuildWith("b = image(\"bracket\")\nm = test_merge(b)\no = brightness(m)\n", &e2),
                  "a reduced set chains into a single-image algorithm");


        // The group state machine, which the palette UI drives. Extracted from
        // the app so it can be tested: the invariant is that the shape's extent
        // always equals the image count, and a stale extent would surface much
        // later as a reduction over the wrong number of frames.
        {
            auto Px = [](int w) {
                Image img;
                ImageDesc d; d.width = w; d.height = 4; d.format = Format::RGBA8;
                img.Alloc(d);
                return img;
            };

            // A single image becomes a group of one, not of zero.
            Data d{Px(4)};
            tglab::MakeGroup(&d, "frame");
            Check(TypeOf(d) == DataType::ImageSet, "MakeGroup produces a set");
            Check(ShapeOf(d) == Shape::Of("frame", 1),
                  "a grouped single image is a group of one");

            tglab::AppendToGroup(&d, Px(4), "frame");
            tglab::AppendToGroup(&d, Px(4), "frame");
            Check(ShapeOf(d) == Shape::Of("frame", 3), "appending restates the extent");
            Check(std::get<ImageSet>(d).images.size() == 3, "and the images are there");

            // Renaming the axis keeps the extent.
            tglab::SetGroupAxis(&d, "exposure");
            Check(ShapeOf(d) == Shape::Of("exposure", 3),
                  "renaming the axis keeps the count");

            tglab::RemoveLastFromGroup(&d, "exposure");
            Check(ShapeOf(d) == Shape::Of("exposure", 2), "removing restates the extent");

            // Ungroup keeps the FIRST image, so a script naming the slot still
            // resolves rather than the entry going empty.
            std::get<ImageSet>(d).images.front() = Px(99);
            tglab::Ungroup(&d);
            Check(TypeOf(d) == DataType::Image, "Ungroup returns a single image");
            Check(std::get<Image>(d).Desc().width == 99, "and it keeps the first");

            // An empty slot grouped and appended to is still consistent -- the
            // path a fresh entry takes when group mode is set before any drop.
            Data e2{};
            tglab::MakeGroup(&e2, "focus");
            Check(ShapeOf(e2) == Shape::Of("focus", 0),
                  "grouping an empty slot gives an empty group");
            tglab::AppendToGroup(&e2, Px(4), "focus");
            Check(ShapeOf(e2) == Shape::Of("focus", 1), "and appending starts the count");

            // Removing past the end must not underflow the extent.
            tglab::RemoveLastFromGroup(&e2, "focus");
            tglab::RemoveLastFromGroup(&e2, "focus");
            Check(ShapeOf(e2) == Shape::Of("focus", 0), "removing past empty stays at zero");

            // Ungrouping an empty group yields an invalid image rather than
            // crashing -- the slot stays addressable and fails at run time with
            // "produced no data".
            tglab::Ungroup(&e2);
            Check(TypeOf(e2) == DataType::Image, "ungrouping an empty group is safe");

            // MakeGroup on an existing group is idempotent, not nesting.
            Data e3{Px(4)};
            tglab::MakeGroup(&e3, "frame");
            tglab::AppendToGroup(&e3, Px(4), "frame");
            tglab::MakeGroup(&e3, "frame");
            Check(ShapeOf(e3) == Shape::Of("frame", 2), "MakeGroup on a group changes nothing");
        }

        // A real reduction, end to end: a group of known pixel values through
        // merge_mean, checking the ARITHMETIC and not merely that it ran.
        {
            auto Flat = [](int w, int h, uint8_t v) {
                Image img;
                ImageDesc d; d.width = w; d.height = h; d.format = Format::RGBA8;
                img.Alloc(d);
                ImageView vw = img.MapCpuWrite();
                for (int i = 0; i < w * h * 4; ++i) vw.data[i] = v;
                return img;
            };

            // 10, 20, 60 -> mean 30.
            ImageSet set;
            set.images.push_back(Flat(4, 4, 10));
            set.images.push_back(Flat(4, 4, 20));
            set.images.push_back(Flat(4, 4, 60));
            set.shape = Shape::Of("exposure", 3);

            std::vector<SourceImage> names{{"test", 0}};
            SourceImage gs; gs.name = "grp"; gs.index = 1;
            gs.shape = Shape::Of("exposure", 3);
            names.push_back(gs);

            Program prog; std::string err;
            Check(Parse("g = image(\"grp\")\nm = merge_mean(g, over=\"exposure\")\ndisplay(m)\n",
                        &prog, &err), "a reduction parses");

            UiState ui; Pipeline pipe;
            InterpResult ir = Interpret(prog, names, &ui, &pipe);
            Check(ir.ok, ir.ok ? "a reduction interprets" : ("interpret: " + ir.error).c_str());

            std::vector<Data> srcs;
            srcs.push_back(Data{});                    // slot 0, unused
            srcs.push_back(Data{std::move(set)});      // slot 1, the group
            std::string xerr;
            const bool ran = pipe.Execute(&srcs, nullptr, &xerr);
            Check(ran, ran ? "a reduction runs" : ("execute: " + xerr).c_str());

            if (ran) {
                const Data& o = pipe.Stages().back().outputs[0];
                Check(TypeOf(o) == DataType::Image, "a reduction produces a single image");
                Image& oi = const_cast<Image&>(std::get<Image>(o));
                ImageView ov = oi.MapCpuRead();
                Check(ov.data && ov.data[0] == 30,
                      "merge_mean averages: (10+20+60)/3 == 30");
                Check(oi.Desc().width == 4 && oi.Desc().height == 4,
                      "and keeps the frame size");
            }

            // over= naming an axis that is not there must fail at the line.
            Program p2; std::string e2;
            Parse("g = image(\"grp\")\nm = merge_mean(g, over=\"focus\")\n", &p2, &e2);
            UiState u2; Pipeline pl2;
            InterpResult r2 = Interpret(p2, names, &u2, &pl2);
            Check(!r2.ok, "reducing over an axis that is not there fails");
            Check(r2.error.find("focus") != std::string::npos &&
                  r2.error.find("[exposure=3]") != std::string::npos,
                  "and the error names both the axis asked for and the shape");

            // A single axis makes over= optional.
            Program p3; std::string e3;
            Parse("g = image(\"grp\")\nm = merge_mean(g)\n", &p3, &e3);
            UiState u3; Pipeline pl3;
            Check(Interpret(p3, names, &u3, &pl3).ok,
                  "over= is optional when there is only one axis");

            // over= on something that does not reduce is a mistake worth
            // reporting rather than ignoring.
            Program p4; std::string e4;
            Parse("s = image(\"test\")\no = brightness(s, over=\"frame\")\n", &p4, &e4);
            UiState u4; Pipeline pl4;
            InterpResult r4 = Interpret(p4, names, &u4, &pl4);
            Check(!r4.ok && r4.error.find("not a reduction") != std::string::npos,
                  "over= on a non-reduction is rejected");

            // And a reduction handed a single image says so.
            Program p5; std::string e5;
            Parse("s = image(\"test\")\nm = merge_mean(s)\n", &p5, &e5);
            UiState u5; Pipeline pl5;
            InterpResult r5 = Interpret(p5, names, &u5, &pl5);
            Check(!r5.ok, "a reduction given a single image is rejected");
        }

        // Broadcasting at run time: a scalar algorithm mapped across a group
        // must produce a group of the same shape, with the algorithm actually
        // applied to every frame -- not just to the first.
        {
            auto Flat = [](int v) {
                Image img;
                ImageDesc d; d.width = 2; d.height = 2; d.format = Format::RGBA8;
                img.Alloc(d);
                ImageView vw = img.MapCpuWrite();
                for (int i = 0; i < 2 * 2 * 4; ++i) vw.data[i] = uint8_t(v);
                return img;
            };

            ImageSet set;
            set.images.push_back(Flat(10));
            set.images.push_back(Flat(20));
            set.images.push_back(Flat(30));
            set.shape = Shape::Of("frame", 3);

            std::vector<SourceImage> names2{{"test", 0}};
            SourceImage gs; gs.name = "g"; gs.index = 1; gs.shape = Shape::Of("frame", 3);
            names2.push_back(gs);

            Program prog; std::string err;
            Parse("g = image(\"g\")\nb = brightness(g, gain = 2.0)\ndisplay(b)\n", &prog, &err);
            UiState ui; Pipeline pipe;
            InterpResult ir = Interpret(prog, names2, &ui, &pipe);
            Check(ir.ok, ir.ok ? "a broadcast interprets" : ("interpret: " + ir.error).c_str());

            std::vector<Data> srcs;
            srcs.push_back(Data{});
            srcs.push_back(Data{std::move(set)});
            std::string xerr;
            const bool ran = pipe.Execute(&srcs, nullptr, &xerr);
            Check(ran, ran ? "a broadcast runs" : ("execute: " + xerr).c_str());

            if (ran) {
                const Data& o = pipe.Stages().back().outputs[0];
                Check(TypeOf(o) == DataType::ImageSet,
                      "broadcasting a scalar algorithm produces a set");
                if (const auto* os = std::get_if<ImageSet>(&o)) {
                    Check(os->shape == Shape::Of("frame", 3),
                          "and the result keeps the input's shape");
                    Check(os->images.size() == 3, "with one image per frame");
                    // gain 2.0 doubles: 10,20,30 -> 20,40,60. Checking EVERY
                    // frame, since mapping only the first and copying it would
                    // pass any test that looked at one.
                    bool allRight = os->images.size() == 3;
                    const int want[3] = {20, 40, 60};
                    for (size_t f = 0; f < os->images.size() && allRight; ++f) {
                        ImageView v = const_cast<Image&>(os->images[f]).MapCpuRead();
                        if (!v.data || v.data[0] != want[f]) allRight = false;
                    }
                    Check(allRight, "and the algorithm ran on every frame, not just the first");
                }
            }

            // Broadcast then reduce, which is the whole point: develop each
            // frame, then merge them. 20,40,60 -> mean 40.
            Program p2; std::string e2;
            Parse("g = image(\"g\")\nb = brightness(g, gain = 2.0)\n"
                  "m = merge_mean(b)\ndisplay(m)\n", &p2, &e2);
            UiState u2; Pipeline pl2;
            Check(Interpret(p2, names2, &u2, &pl2).ok, "broadcast then reduce interprets");

            ImageSet set2;
            set2.images.push_back(Flat(10));
            set2.images.push_back(Flat(20));
            set2.images.push_back(Flat(30));
            set2.shape = Shape::Of("frame", 3);
            std::vector<Data> s2;
            s2.push_back(Data{});
            s2.push_back(Data{std::move(set2)});
            std::string x2;
            const bool ran2 = pl2.Execute(&s2, nullptr, &x2);
            Check(ran2, ran2 ? "broadcast then reduce runs" : ("execute: " + x2).c_str());
            if (ran2) {
                const Data& o = pl2.Stages().back().outputs[0];
                Check(TypeOf(o) == DataType::Image, "and collapses back to one image");
                if (const auto* oi = std::get_if<Image>(&o)) {
                    ImageView v = const_cast<Image&>(*oi).MapCpuRead();
                    Check(v.data && v.data[0] == 40,
                          "with the right value: mean(20,40,60) == 40");
                }
            }
        }
        }
    }


    // --- image loader ordering ----------------------------------------------
    //
    // The loader runs a pool now, so results finish out of order: a small PNG
    // queued after a 45 MP raw completes first. File order is meaningful for a
    // group -- a bracket is an ordered thing -- so TryFetch must deliver in
    // REQUEST order regardless.
    //
    // Exercised with real files of deliberately different sizes, because the
    // whole point is that decode time varies. A test with identical inputs
    // would pass even if ordering were dropped entirely.
    {
        ImageLoader loader;
        loader.Start();

        // page.png is the smaller of the two committed assets, so queuing it
        // AFTER the larger one is what creates the overtaking opportunity.
        const char* files[] = {
            "assets/test.png", "assets/page.png", "assets/test.png",
            "assets/page.png", "assets/test.png",
        };
        const int n = int(sizeof(files) / sizeof(files[0]));
        for (int i = 0; i < n; ++i) loader.Request(files[i], "");

        std::vector<std::string> got;
        for (int spins = 0; spins < 20000 && int(got.size()) < n; ++spins) {
            LoadResult r;
            while (loader.TryFetch(&r)) got.push_back(r.path);
            if (int(got.size()) < n) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        loader.Stop();

        Check(int(got.size()) == n, "every queued file comes back");
        bool ordered = (int(got.size()) == n);
        for (int i = 0; i < int(got.size()) && ordered; ++i)
            if (got[size_t(i)] != files[i]) ordered = false;
        Check(ordered, "results arrive in request order, not completion order");
    }

    // --- filename ordering ---------------------------------------------------
    //
    // Windows hands a multi-file drop over in selection order, and the file
    // clicked last routinely arrives out of place, so a bracket dropped in
    // visual order can still reach the palette shuffled. A reduction then
    // consumes it in the wrong order and produces a silently wrong result --
    // which is why this sorts numerically rather than lexicographically.
    {
        Check(FilenameLess("IMG_9.CR3", "IMG_10.CR3"),
              "9 sorts before 10, not after");
        Check(!FilenameLess("IMG_10.CR3", "IMG_9.CR3"),
              "and the reverse does not hold");
        Check(FilenameLess("_U0A0999.CR3", "_U0A1000.CR3"),
              "a digit-run carry sorts correctly");
        Check(FilenameLess("a.CR3", "b.CR3"), "plain text still compares");
        Check(FilenameLess("IMG_2.CR3", "IMG_02b.CR3"),
              "equal numbers fall through to the characters after them");
        Check(!FilenameLess("IMG_1.CR3", "IMG_1.CR3"),
              "a name is not less than itself");
        Check(FilenameLess("IMG_1", "IMG_1x"), "a prefix sorts first");

        // Case-insensitive, since a camera and a card reader disagree about it.
        Check(!FilenameLess("IMG_2.CR3", "img_1.CR3"),
              "case does not override the number");

        // The ordering must be a strict weak ordering or std::sort is UB.
        const char* names[] = {"IMG_10", "IMG_9", "img_9", "IMG_100", "a", "IMG_9x"};
        for (const char* x : names)
            for (const char* y : names)
                if (FilenameLess(x, y))
                    Check(!FilenameLess(y, x), "the ordering is antisymmetric");
    }

    // --- date ordering --------------------------------------------------------
    //
    // Sorting a group by when the shutter fired, which disagrees with the
    // filename sort exactly where it matters: two bodies at one event interleave
    // in time and separate completely by name, and a card that has wrapped past
    // 9999 sorts IMG_0001 before IMG_9999 by name and after it by time.
    {
        // EXIF's format is fixed width and zero padded, so comparing the
        // strings compares the instants -- no parsing, no timezone.
        Check(DateSortKey("2026:08:19 14:32:07", 0) <
              DateSortKey("2026:08:19 14:32:08", 0),
              "one second later sorts later");
        Check(DateSortKey("2026:08:19 09:00:00", 0) <
              DateSortKey("2026:08:19 14:00:00", 0),
              "the zero padding is what makes 09 sort before 14");
        Check(DateSortKey("2026:08:19 23:59:59", 0) <
              DateSortKey("2026:08:20 00:00:00", 0),
              "midnight rolls over correctly");
        Check(DateSortKey("2025:12:31 23:59:59", 0) <
              DateSortKey("2026:01:01 00:00:00", 0),
              "and so does new year");

        // A frame with no EXIF date falls back to its file time, which is a
        // different clock. Every fallback must sort before every real date
        // rather than interleaving on digits -- otherwise the order depends on
        // whether an epoch count happens to start with a bigger digit than a
        // year does.
        const std::string noExif = DateSortKey("", 1755600000LL);
        const std::string dated  = DateSortKey("2026:08:19 14:32:07", 0);
        Check(noExif < dated, "a file with no EXIF date sorts before dated ones");

        // Including a file time whose digits would otherwise sort AFTER a year.
        Check(DateSortKey("", 9999999999LL) < dated,
              "even when its epoch count starts with a larger digit");

        // The unknowns still order among themselves by file time, so a folder
        // of PNGs is not shuffled arbitrarily.
        Check(DateSortKey("", 100) < DateSortKey("", 200),
              "undated files keep their own chronological order");

        // Across a digit-count boundary, which is what the zero padding is for:
        // compared as raw text, "99" sorts AFTER "100". Windows file times are
        // uniformly 18 digits so this would not bite today, which is what would
        // have made it a nasty surprise on some other platform later.
        Check(DateSortKey("", 99) < DateSortKey("", 100),
              "99 sorts before 100 despite being compared as text");

        // Same instant on two frames of a burst: neither is less, so
        // stable_sort keeps the order they already had.
        const std::string same = "2026:08:19 14:32:07";
        Check(!(DateSortKey(same, 0) < DateSortKey(same, 0)),
              "frames sharing a second are ties, not reordered");
    }

    // --- rank-2 reduction ----------------------------------------------------
    //
    // Reducing ONE axis of a multi-axis set: [position=2, exposure=3] over
    // "exposure" is two merges of three frames each, not one merge of six. The
    // result keeps the axis that was not reduced, which is what lets a chain of
    // reductions peel them off one at a time.
    //
    // Values are chosen so a wrong grouping cannot coincidentally give the
    // right answer: position 0 averages to 20 and position 1 to 50, while
    // merging all six would give 35. Reducing the WRONG axis would give
    // {30, 35, 40}, which is a different length as well as different values.
    {
        auto Flat = [](uint8_t v) {
            Image img;
            ImageDesc d; d.width = 2; d.height = 2; d.format = Format::RGBA8;
            img.Alloc(d);
            ImageView vw = img.MapCpuWrite();
            for (int i = 0; i < 2 * 2 * 4; ++i) vw.data[i] = v;
            return img;
        };

        // Row-major, last axis fastest: position 0's three exposures first.
        ImageSet set;
        set.images.push_back(Flat(10));   // p0 e0
        set.images.push_back(Flat(20));   // p0 e1
        set.images.push_back(Flat(30));   // p0 e2   -> mean 20
        set.images.push_back(Flat(40));   // p1 e0
        set.images.push_back(Flat(50));   // p1 e1
        set.images.push_back(Flat(60));   // p1 e2   -> mean 50
        set.shape = Shape{{Axis{"position", 2}, Axis{"exposure", 3}}};

        std::vector<SourceImage> names{{"test", 0}};
        SourceImage gs; gs.name = "shoot"; gs.index = 1;
        gs.shape = Shape{{Axis{"position", 2}, Axis{"exposure", 3}}};
        names.push_back(gs);

        Program prog; std::string err;
        Check(Parse("g = image(\"shoot\")\nm = merge_mean(g, over=\"exposure\")\ndisplay(m)\n",
                    &prog, &err), "a rank-2 reduction parses");

        UiState ui; Pipeline pipe;
        InterpResult ir = Interpret(prog, names, &ui, &pipe);
        Check(ir.ok, ir.ok ? "a rank-2 reduction interprets" : ("interpret: " + ir.error).c_str());

        std::vector<Data> srcs;
        srcs.push_back(Data{});
        srcs.push_back(Data{std::move(set)});
        std::string xerr;
        const bool ran = pipe.Execute(&srcs, nullptr, &xerr);
        Check(ran, ran ? "a rank-2 reduction runs" : ("execute: " + xerr).c_str());

        if (ran) {
            const Data& o = pipe.Stages().back().outputs[0];
            Check(TypeOf(o) == DataType::ImageSet,
                  "reducing one axis of two leaves a set, not an image");
            if (const auto* os = std::get_if<ImageSet>(&o)) {
                Check(os->shape == Shape::Of("position", 2),
                      "and the reduced axis is gone, the other kept");
                Check(os->images.size() == 2, "with one result per remaining coordinate");

                bool right = os->images.size() == 2;
                const int want[2] = {20, 50};
                for (size_t k = 0; k < os->images.size() && right; ++k) {
                    ImageView v = const_cast<Image&>(os->images[k]).MapCpuRead();
                    if (!v.data || v.data[0] != want[k]) right = false;
                }
                Check(right, "each position averages its OWN exposures: 20 and 50");
            }
        }

        // Reducing the other axis gives a different shape and different values,
        // which is what proves the axis name is actually being used.
        {
            ImageSet s2;
            for (uint8_t v : {10, 20, 30, 40, 50, 60}) s2.images.push_back(Flat(v));
            s2.shape = Shape{{Axis{"position", 2}, Axis{"exposure", 3}}};

            Program p2; std::string e2;
            Parse("g = image(\"shoot\")\nm = merge_mean(g, over=\"position\")\ndisplay(m)\n",
                  &p2, &e2);
            UiState u2; Pipeline pl2;
            Check(Interpret(p2, names, &u2, &pl2).ok, "reducing the other axis interprets");

            std::vector<Data> sv;
            sv.push_back(Data{});
            sv.push_back(Data{std::move(s2)});
            std::string x2;
            if (pl2.Execute(&sv, nullptr, &x2)) {
                const Data& o = pl2.Stages().back().outputs[0];
                if (const auto* os = std::get_if<ImageSet>(&o)) {
                    Check(os->shape == Shape::Of("exposure", 3),
                          "reducing position leaves the exposure axis");
                    bool right = os->images.size() == 3;
                    const int want[3] = {25, 35, 45};   // (10+40)/2, (20+50)/2, (30+60)/2
                    for (size_t k = 0; k < os->images.size() && right; ++k) {
                        ImageView v = const_cast<Image&>(os->images[k]).MapCpuRead();
                        if (!v.data || v.data[0] != want[k]) right = false;
                    }
                    Check(right, "and strides across position: 25, 35, 45");

    // --- shape() and a chain of reductions -----------------------------------
    //
    // The design's headline example, at a size that can be checked by hand: a
    // flat drop of 6 declared as [position=2, exposure=3], then reduced twice.
    // The point is that the script never iterates and the algorithm never knows
    // it is inside a chain -- merge_mean is the same rank-1 reducer both times.
    {
        auto Flat = [](uint8_t v) {
            Image img;
            ImageDesc d; d.width = 2; d.height = 2; d.format = Format::RGBA8;
            img.Alloc(d);
            ImageView vw = img.MapCpuWrite();
            for (int i = 0; i < 2 * 2 * 4; ++i) vw.data[i] = v;
            return img;
        };

        std::vector<SourceImage> names{{"test", 0}};
        SourceImage gs; gs.name = "shoot"; gs.index = 1;
        gs.shape = Shape::Of("frame", 6);          // a flat drop, as it arrives
        names.push_back(gs);

        auto Run = [&](const char* src, std::string* err) -> Data {
            Program prog;
            if (!Parse(src, &prog, err)) return Data{};
            UiState ui; Pipeline pipe;
            InterpResult r = Interpret(prog, names, &ui, &pipe);
            if (!r.ok) { *err = r.error; return Data{}; }

            ImageSet s;
            for (uint8_t v : {10, 20, 30, 40, 50, 60}) s.images.push_back(Flat(v));
            s.shape = Shape::Of("frame", 6);

            std::vector<Data> srcs;
            srcs.push_back(Data{});
            srcs.push_back(Data{std::move(s)});
            if (!pipe.Execute(&srcs, nullptr, err)) return Data{};

            // Resolve what display() actually shows, rather than the last stage
            // recorded -- shape() adds a stage, so "last" is not the reduction.
            if (pipe.Viewers().empty()) { *err = "no viewer declared"; return Data{}; }
            const Data* op = pipe.Resolve(pipe.Viewers().back().source, &srcs);
            if (!op) { *err = "viewer resolved to nothing"; return Data{}; }
            const Data& o = *op;
            if (const auto* im = std::get_if<Image>(&o)) return Data{im->Clone()};
            if (const auto* st = std::get_if<ImageSet>(&o)) {
                ImageSet c; c.shape = st->shape;
                for (const Image& i : st->images) c.images.push_back(i.Clone());
                return Data{std::move(c)};
            }
            return Data{};
        };

        std::string e1;
        Data d1 = Run("g = image(\"shoot\")\n"
                      "g = shape(g, position=2, exposure=3)\n"
                      "m = merge_mean(g, over=\"exposure\")\n"
                      "display(m)\n", &e1);
        Check(TypeOf(d1) == DataType::ImageSet,
              TypeOf(d1) == DataType::ImageSet ? "shape() then reduce leaves a set"
                                               : ("failed: " + e1).c_str());
        if (const auto* os = std::get_if<ImageSet>(&d1)) {
            Check(os->shape == Shape::Of("position", 2), "shape() named the axes");
            bool right = os->images.size() == 2;
            const int want[2] = {20, 50};
            for (size_t k = 0; k < os->images.size() && right; ++k) {
                ImageView v = const_cast<Image&>(os->images[k]).MapCpuRead();
                if (!v.data || v.data[0] != want[k]) right = false;
            }
            Check(right, "and the layout is row-major: last axis varies fastest");
        }

        // Two reductions in a row, the second consuming the first's output.
        // mean(20, 50) == 35.
        std::string e2;
        Data d2 = Run("g = image(\"shoot\")\n"
                      "g = shape(g, position=2, exposure=3)\n"
                      "a = merge_mean(g, over=\"exposure\")\n"
                      "b = merge_mean(a, over=\"position\")\n"
                      "display(b)\n", &e2);
        Check(TypeOf(d2) == DataType::Image,
              TypeOf(d2) == DataType::Image ? "a chain of reductions ends at one image"
                                            : ("failed: " + e2).c_str());
        if (const auto* im = std::get_if<Image>(&d2)) {
            ImageView v = const_cast<Image&>(*im).MapCpuRead();
            Check(v.data && v.data[0] == 35, "mean(mean(10,20,30), mean(40,50,60)) == 35");
        }

        // shape() must describe exactly the images present.
        std::string e3;
        Data d3 = Run("g = image(\"shoot\")\n"
                      "g = shape(g, position=2, exposure=4)\n"
                      "display(g)\n", &e3);
        Check(TypeOf(d3) == DataType::None, "shape() rejects a count that does not match");
        Check(e3.find("8") != std::string::npos && e3.find("6") != std::string::npos,
              "and the error names both counts");
    }

    // --- merge_hdr -----------------------------------------------------------
    //
    // The radiometric merge, checked against numbers worked out by hand.
    //
    // Three frames of a uniform scene, one stop apart. If the merge is correct,
    // each frame divided by its own exposure gives the SAME radiance, so the
    // weighted average is that radiance regardless of the weights -- which is
    // exactly the property that makes the merge physically meaningful, and a
    // plain average would not have.
    {
        // shutter is the only term that varies, so relative exposure is
        // shutter * ISO / aperture^2 with ISO and aperture held constant.
        auto Frame = [](float value, float shutter) {
            Image img;
            ImageDesc d;
            d.width = 2; d.height = 2; d.format = Format::RGBA32F;
            d.linear   = true;
            d.shutter  = shutter;
            d.aperture = 2.0f;
            d.iso      = 100.0f;
            img.Alloc(d);
            ImageView v = img.MapCpuWrite();
            float* p = reinterpret_cast<float*>(v.data);
            for (int i = 0; i < 2 * 2 * 4; ++i) p[i] = ((i % 4) == 3) ? 1.0f : value;
            return img;
        };

        // Radiance 4.0 seen at three shutter speeds: 0.1 -> 0.4, 0.2 -> 0.8,
        // 0.05 -> 0.2. Every frame is mid-toned, so all three carry weight.
        ImageSet set;
        set.images.push_back(Frame(0.4f,  0.1f));
        set.images.push_back(Frame(0.8f,  0.2f));
        set.images.push_back(Frame(0.2f,  0.05f));
        set.shape = Shape::Of("exposure", 3);

        std::vector<SourceImage> names{{"test", 0}};
        SourceImage gs; gs.name = "bracket"; gs.index = 1;
        gs.shape = Shape::Of("exposure", 3);
        names.push_back(gs);

        Program prog; std::string err;
        Check(Parse("g = image(\"bracket\")\nh = merge_hdr(g, over=\"exposure\")\ndisplay(h)\n",
                    &prog, &err), "merge_hdr parses");

        UiState ui; Pipeline pipe;
        InterpResult ir = Interpret(prog, names, &ui, &pipe);
        Check(ir.ok, ir.ok ? "merge_hdr interprets" : ("interpret: " + ir.error).c_str());

        std::vector<Data> srcs;
        srcs.push_back(Data{});
        srcs.push_back(Data{std::move(set)});
        std::string xerr;
        const bool ran = pipe.Execute(&srcs, nullptr, &xerr);
        Check(ran, ran ? "merge_hdr runs" : ("execute: " + xerr).c_str());

        if (ran) {
            const Data* op = pipe.Resolve(pipe.Viewers().back().source, &srcs);
            const auto* im = op ? std::get_if<Image>(op) : nullptr;
            Check(im != nullptr, "merge_hdr produces a single image");
            if (im) {
                Check(im->Desc().format == Format::RGBA32F,
                      "and it is 32-bit float, to hold the extra range");
                Check(im->Desc().linear, "and is marked scene-linear");

                ImageView v = const_cast<Image&>(*im).MapCpuRead();
                const float* p = reinterpret_cast<const float*>(v.data);
                // All three frames agree on radiance 4.0 * (ISO/f^2 scaling),
                // which is the same constant for each -- so the result is the
                // ratio value/shutter, i.e. 0.4/0.1 = 4, scaled by 1/(iso/f^2).
                const float expect = 0.4f / (0.1f * 100.0f / 4.0f);
                Check(p && std::abs(p[0] - expect) < 1e-3f,
                      "every frame recovers the SAME radiance, so the merge is exact");

                // The headroom is the point: a value above 1.0 is what a
                // bracket buys and what an averaging merge would have lost.
                Check(p && p[0] > 0.15f, "the result is scene-linear radiance, not a 0..1 average");
            }
        }

        // A frame with no exposure metadata must be reported, not silently
        // treated as equal -- that is a confidently wrong merge.
        {
            auto Plain = [](float value) {
                Image img;
                ImageDesc d; d.width = 2; d.height = 2; d.format = Format::RGBA32F;
                img.Alloc(d);
                ImageView v = img.MapCpuWrite();
                float* p = reinterpret_cast<float*>(v.data);
                for (int i = 0; i < 2 * 2 * 4; ++i) p[i] = ((i % 4) == 3) ? 1.0f : value;
                return img;
            };
            ImageSet s2;
            s2.images.push_back(Plain(0.3f));
            s2.images.push_back(Plain(0.6f));
            s2.shape = Shape::Of("exposure", 2);

            SourceImage g2; g2.name = "noexif"; g2.index = 2;
            g2.shape = Shape::Of("exposure", 2);
            std::vector<SourceImage> n2 = names;
            n2.push_back(g2);

            Program p2; std::string e2;
            Parse("g = image(\"noexif\")\nh = merge_hdr(g)\ndisplay(h)\n", &p2, &e2);
            UiState u2; Pipeline pl2;
            Check(Interpret(p2, n2, &u2, &pl2).ok, "a bracket with no EXIF still interprets");

            std::vector<Data> sv;
            sv.push_back(Data{});
            sv.push_back(Data{});
            sv.push_back(Data{std::move(s2)});
            std::string x2;
            if (pl2.Execute(&sv, nullptr, &x2)) {
                std::string report;
                for (const Stage& st : pl2.Stages())
                    if (st.algo && st.algoName == "merge_hdr") report = st.algo->RunReport();
                Check(report.find("NO EXIF") != std::string::npos,
                      "and says so rather than merging as though exposures matched");
            }
        }
    }

    // --- tonemap -------------------------------------------------------------
    //
    // Fits scene-linear data into the range the display curve expects. The
    // measurement is the whole point: a merge's absolute scale depends on the
    // shutter speeds in the bracket, so no fixed anchor can be right and the
    // operator has to read its own input.
    {
        // A gradient with a huge range, like a merged bracket: most of the
        // frame dark, a few pixels enormously bright.
        auto Ramp = []() {
            Image img;
            ImageDesc d;
            d.width = 64; d.height = 64; d.format = Format::RGBA32F;
            d.linear = true;
            img.Alloc(d);
            ImageView v = img.MapCpuWrite();
            float* p = reinterpret_cast<float*>(v.data);
            for (int i = 0; i < 64 * 64; ++i) {
                // 0.5 .. 500, so the median sits near 5 -- well above grey.
                const float t = float(i) / float(64 * 64 - 1);
                const float val = 0.5f * std::pow(1000.0f, t);
                for (int c = 0; c < 3; ++c) p[i * 4 + c] = val;
                p[i * 4 + 3] = 1.0f;
            }
            return img;
        };

        auto RunTone = [&](const char* src, std::string* err) -> Data {
            Program prog;
            if (!Parse(src, &prog, err)) return Data{};
            std::vector<SourceImage> nm{{"test", 0}};
            UiState ui; Pipeline pipe;
            InterpResult r = Interpret(prog, nm, &ui, &pipe);
            if (!r.ok) { *err = r.error; return Data{}; }
            std::vector<Data> srcs;
            srcs.push_back(Data{Ramp()});
            if (!pipe.Execute(&srcs, nullptr, err)) return Data{};
            const Data* o = pipe.Resolve(pipe.Viewers().back().source, &srcs);
            if (const auto* im = o ? std::get_if<Image>(o) : nullptr) return Data{im->Clone()};
            return Data{};
        };

        auto Stats = [](const Data& d, float* median, float* hi) {
            const auto* im = std::get_if<Image>(&d);
            if (!im) return false;
            ImageView v = const_cast<Image&>(*im).MapCpuRead();
            if (!v.data) return false;
            PixelBuffer pb;
            pb.Unpack(v);
            std::vector<float> lum;
            for (int y = 0; y < pb.Height(); ++y)
                for (int x = 0; x < pb.Width(); ++x) lum.push_back(pb.At(x, y)[0]);
            std::sort(lum.begin(), lum.end());
            *median = lum[lum.size() / 2];
            *hi     = lum.back();
            return true;
        };

        std::string e1;
        Data d1 = RunTone("s = image(\"test\")\nt = tonemap(s)\ndisplay(t)\n", &e1);
        float med = 0.0f, hi = 0.0f;
        Check(Stats(d1, &med, &hi), e1.empty() ? "tonemap runs" : ("failed: " + e1).c_str());

        // The median lands NEAR middle grey rather than exactly on it.
        //
        // The operator no longer pins the median: it fits both ENDS of the
        // scene and lets the midpoint fall where the scene puts it, because
        // where the ends sit relative to the median is a property of the scene
        // and varies enormously. Measured on two of Tim's brackets, one is 3.8
        // stops below the median and 1.5 above, the other 2.1 below and 3.7
        // above -- fixing the midpoint threw the long end off the display in
        // each case, in opposite directions.
        //
        // A clamp keeps the median within two stops of grey, so the bound here
        // is that rather than an equality.
        Check(med > 0.18f * 0.25f && med < 0.18f * 4.0f,
              "the midtones land within a couple of stops of middle grey (" +
                  std::to_string(med) + ")");

        // The top must be bounded, and bounded IN DISPLAY UNITS.
        //
        // The first version of this checked hi < 0.95 linear, which passed
        // while the operator was badly wrong: it capped at 0.9 linear, which
        // the display curve renders as 0.733, so every tone from the midtones
        // up was squeezed into 0.46-0.73 and the picture came out flat with no
        // whites. A linear bound cannot see that -- only following the value
        // through the display curve can.
        Check(ToneCurve(hi) < 1.0f, "the highlights do not clip on the display");
        Check(ToneCurve(hi) > 0.85f,
              "and reach most of the way to white, rather than stopping short");
        Check(ToneCurve(hi) - ToneCurve(med) > 0.35f,
              "leaving real separation between the midtones and the highlights");

        // The exposure parameter moves the median, proportionally.
        std::string e2;
        Data d2 = RunTone("s = image(\"test\")\nt = tonemap(s, exposure = 2.0)\ndisplay(t)\n", &e2);
        float med2 = 0.0f, hi2 = 0.0f;
        if (Stats(d2, &med2, &hi2))
            Check(med2 > med * 1.5f, "exposure = 2 places the midtones brighter");

        // Below middle grey the mapping is exactly linear, so shadows are not
        // touched at all. Checked directly rather than inferred.
        {
            Image dark;
            ImageDesc dd; dd.width = 4; dd.height = 4; dd.format = Format::RGBA32F;
            dd.linear = true;
            dark.Alloc(dd);
            ImageView dv = dark.MapCpuWrite();
            float* dp = reinterpret_cast<float*>(dv.data);
            for (int i = 0; i < 4 * 4; ++i) {
                for (int c = 0; c < 3; ++c) dp[i * 4 + c] = 0.18f;   // exactly grey
                dp[i * 4 + 3] = 1.0f;
            }
            Program p3; std::string e3;
            Parse("s = image(\"test\")\nt = tonemap(s)\ndisplay(t)\n", &p3, &e3);
            std::vector<SourceImage> nm{{"test", 0}};
            UiState u3; Pipeline pl3;
            Interpret(p3, nm, &u3, &pl3);
            std::vector<Data> sv;
            sv.push_back(Data{std::move(dark)});
            if (pl3.Execute(&sv, nullptr, &e3)) {
                const Data* o = pl3.Resolve(pl3.Viewers().back().source, &sv);
                if (const auto* im = o ? std::get_if<Image>(o) : nullptr) {
                    ImageView v = const_cast<Image&>(*im).MapCpuRead();
                    const float* p = reinterpret_cast<const float*>(v.data);
                    Check(p && std::abs(p[0] - 0.18f) < 1e-4f,
                          "an image already at middle grey passes through unchanged");
                }
            }
        }

        // Two scenes with OPPOSITE shapes must both land usably.
        //
        // This is the check the median anchor could not pass, and the reason it
        // was replaced. Tim's brackets measured 3.8 stops below the median and
        // 1.5 above (a sunlit building) versus 2.1 below and 3.7 above (a
        // valley and sky). Pinning the midpoint sent the long end off the
        // display in each case -- the building's shadows to 0.013 linear, black
        // with nothing to recover; the sky to 2.3, flat and needing heavy
        // manual contrast.
        //
        // Checked in DISPLAY units at both ends, because that is where the
        // failure showed. A linear bound would pass on values that render as
        // black or as flat white.
        {
            // `dark` is the FRACTION of the frame that is shadow, and it is what
            // makes this fixture able to fail.
            //
            // My first attempt used three equal bands, which put the median in
            // the middle band by construction -- so the old median anchor
            // happened to land it correctly and the test passed against the very
            // bug it was written for. A real scene is not balanced: a shadow-
            // heavy frame is MOSTLY shadow, which drags the median down toward
            // one end, and that skew is exactly what the median anchor cannot
            // see. Weighting the areas reproduces it.
            auto Skewed = [](float lowStops, float highStops, float dark) {
                Image img;
                ImageDesc d;
                d.width = 48; d.height = 48; d.format = Format::RGBA32F;
                d.linear = true;
                img.Alloc(d);
                ImageView v = img.MapCpuWrite();
                float* p = reinterpret_cast<float*>(v.data);
                const float mid = 5.0f;   // arbitrary merge scale
                const float lo  = mid / std::pow(2.0f, lowStops);
                const float hi  = mid * std::pow(2.0f, highStops);
                const int darkRows = int(48.0f * dark);
                const int midRows  = (48 - darkRows) / 2;
                for (int y = 0; y < 48; ++y)
                    for (int x = 0; x < 48; ++x) {
                        const float val = (y < darkRows) ? lo
                                        : (y < darkRows + midRows ? mid : hi);
                        const int i = y * 48 + x;
                        for (int c = 0; c < 3; ++c) p[i * 4 + c] = val;
                        p[i * 4 + 3] = 1.0f;
                    }
                return img;
            };

            auto CheckScene = [&](float lowStops, float highStops, float dark,
                                  const char* what) {
                Program prog; std::string err;
                Parse("s = image(\"test\")\nt = tonemap(s)\ndisplay(t)\n", &prog, &err);
                std::vector<SourceImage> nm{{"test", 0}};
                UiState ui; Pipeline pipe;
                if (!Interpret(prog, nm, &ui, &pipe).ok) { Check(false, "interpret"); return; }
                std::vector<Data> srcs;
                srcs.push_back(Data{Skewed(lowStops, highStops, dark)});
                if (!pipe.Execute(&srcs, nullptr, &err)) { Check(false, err); return; }

                const Data* o = pipe.Resolve(pipe.Viewers().back().source, &srcs);
                const auto* im = o ? std::get_if<Image>(o) : nullptr;
                if (!im) { Check(false, "no result"); return; }
                ImageView v = const_cast<Image&>(*im).MapCpuRead();
                const float* p = reinterpret_cast<const float*>(v.data);

                // Row 0 is the shadow band, row 40 the highlight band.
                const float shadow = ToneCurve(p[0]);
                const float high   = ToneCurve(p[(40 * 48) * 4]);

                Check(shadow > 0.04f,
                      std::string(what) + ": the shadows stay off the floor (" +
                          std::to_string(shadow) + ")");
                Check(high < 0.999f,
                      std::string(what) + ": the highlights do not clip (" +
                          std::to_string(high) + ")");
                Check(high - shadow > 0.45f,
                      std::string(what) + ": and the picture keeps real range (" +
                          std::to_string(high - shadow) + ")");
            };

            // 70% shadow and 70% highlight respectively, which is what drags
            // the median toward one end the way a real scene does.
            CheckScene(3.8f, 1.5f, 0.70f, "shadow-heavy scene");
            CheckScene(2.1f, 3.7f, 0.15f, "highlight-heavy scene");
        }
    }

    // --- align ---------------------------------------------------------------
    //
    // A KNOWN shift must come back. The whole value of a sub-pixel solver is
    // that it recovers a displacement more precisely than a pixel, so the test
    // shifts a synthetic frame by a fractional amount and checks the solved
    // transform against it -- not merely that the solve ran.
    {
        // Textured noise, because a solver needs gradients. A smooth gradient
        // would leave the fit under-constrained along its own direction, which
        // is the aperture problem and would make this pass for the wrong reason.
        auto Textured = [](float dx, float dy) {
            Image img;
            ImageDesc d;
            d.width = 256; d.height = 256; d.format = Format::RGBA32F;
            d.linear = true;
            d.shutter = 0.01f; d.aperture = 4.0f; d.iso = 100.0f;
            img.Alloc(d);
            ImageView v = img.MapCpuWrite();
            float* p = reinterpret_cast<float*>(v.data);
            for (int y = 0; y < 256; ++y)
                for (int x = 0; x < 256; ++x) {
                    // Sampled at (x - dx), so the CONTENT moves by +dx: a
                    // deterministic pattern with detail at several scales.
                    const float sx = float(x) - dx, sy = float(y) - dy;
                    const float val =
                        0.30f + 0.20f * std::sin(sx * 0.21f) * std::cos(sy * 0.17f)
                              + 0.10f * std::sin(sx * 0.73f + sy * 0.41f)
                              + 0.05f * std::cos(sx * 1.9f) * std::sin(sy * 1.7f);
                    const int i = y * 256 + x;
                    for (int c = 0; c < 3; ++c) p[i * 4 + c] = val;
                    p[i * 4 + 3] = 1.0f;
                }
            return img;
        };

        auto SolveShift = [&](float dx, float dy, float* gotX, float* gotY) {
            ImageSet set;
            set.images.push_back(Textured(0.0f, 0.0f));
            set.images.push_back(Textured(dx, dy));
            set.shape = Shape::Of("frame", 2);

            std::vector<SourceImage> nm{{"test", 0}};
            SourceImage gs; gs.name = "g"; gs.index = 1;
            gs.shape = Shape::Of("frame", 2);
            nm.push_back(gs);

            Program prog; std::string err;
            if (!Parse("g = image(\"g\")\na = align(g, min_shift = 0)\ndisplay(a)\n", &prog, &err))
                return false;
            UiState ui; Pipeline pipe;
            if (!Interpret(prog, nm, &ui, &pipe).ok) return false;

            std::vector<Data> srcs;
            srcs.push_back(Data{});
            srcs.push_back(Data{std::move(set)});
            if (!pipe.Execute(&srcs, nullptr, &err)) return false;

            const Data* o = pipe.Resolve(pipe.Viewers().back().source, &srcs);
            const auto* os = o ? std::get_if<ImageSet>(o) : nullptr;
            if (!os || os->images.size() != 2) return false;

            const Affine t = TransformOf(os->images[1]);
            *gotX = t.Dx();
            *gotY = t.Dy();
            return true;
        };

        // A whole-pixel shift first, which is the easy case.
        float gx = 0.0f, gy = 0.0f;
        if (SolveShift(2.0f, -1.0f, &gx, &gy)) {
            Check(std::abs(gx - 2.0f) < 0.02f && std::abs(gy + 1.0f) < 0.02f,
                  "align recovers a whole-pixel shift (" + std::to_string(gx) +
                      ", " + std::to_string(gy) + " for 2, -1)");
        } else {
            Check(false, "align runs on a two-frame group");
        }

        // And a SUB-pixel one, which is the point of the algorithm. A solver
        // that only found whole pixels would pass the check above and fail
        // here, which is why both are tested.
        if (SolveShift(0.4f, 0.25f, &gx, &gy)) {
            Check(std::abs(gx - 0.4f) < 0.05f && std::abs(gy - 0.25f) < 0.05f,
                  "and a SUB-pixel shift (" + std::to_string(gx) + ", " +
                      std::to_string(gy) + " for 0.4, 0.25)");
        }

        // The reference frame keeps identity, so a merge that samples through
        // the transform reads it verbatim.
        {
            ImageSet set;
            set.images.push_back(Textured(0.0f, 0.0f));
            set.images.push_back(Textured(1.0f, 0.0f));
            set.shape = Shape::Of("frame", 2);
            std::vector<SourceImage> nm{{"test", 0}};
            SourceImage gs; gs.name = "g"; gs.index = 1;
            gs.shape = Shape::Of("frame", 2);
            nm.push_back(gs);

            Program prog; std::string err;
            Parse("g = image(\"g\")\na = align(g)\ndisplay(a)\n", &prog, &err);
            UiState ui; Pipeline pipe;
            Interpret(prog, nm, &ui, &pipe);
            std::vector<Data> srcs;
            srcs.push_back(Data{});
            srcs.push_back(Data{std::move(set)});
            if (pipe.Execute(&srcs, nullptr, &err)) {
                const Data* o = pipe.Resolve(pipe.Viewers().back().source, &srcs);
                if (const auto* os = o ? std::get_if<ImageSet>(o) : nullptr) {
                    Check(TransformOf(os->images[0]).IsIdentity(),
                          "the reference frame keeps the identity transform");
                }
            }
        }

        // An image with no transform attached reads as identity, so a consumer
        // needs no branch and an unaligned merge resamples nothing.
        {
            Image plain;
            ImageDesc d{4, 4, Format::RGBA32F};
            plain.Alloc(d);
            Check(TransformOf(plain).IsIdentity(),
                  "an image with no transform reads as identity");
        }

        // A correction too small to be worth its resample is DROPPED.
        //
        // Tim's rule, from measuring that aligning his two brackets made the
        // merged result 8% and 16% LESS sharp: bilinear at a fraction of a
        // pixel is a mild low-pass, and below some displacement that softening
        // costs more than the misalignment it removes.
        //
        // Checked at both ends, because a threshold that always fires is as
        // wrong as one that never does.
        {
            auto SolvedShift = [&](float dx, float dy, float minShift) {
                ImageSet set;
                set.images.push_back(Textured(0.0f, 0.0f));
                set.images.push_back(Textured(dx, dy));
                set.shape = Shape::Of("frame", 2);

                std::vector<SourceImage> nm{{"test", 0}};
                SourceImage gs; gs.name = "g"; gs.index = 1;
                gs.shape = Shape::Of("frame", 2);
                nm.push_back(gs);

                const std::string src =
                    "g = image(\"g\")\na = align(g, min_shift = " +
                    std::to_string(minShift) + ")\ndisplay(a)\n";

                Program prog; std::string err;
                Parse(src, &prog, &err);
                UiState ui; Pipeline pipe;
                Interpret(prog, nm, &ui, &pipe);
                std::vector<Data> srcs;
                srcs.push_back(Data{});
                srcs.push_back(Data{std::move(set)});
                if (!pipe.Execute(&srcs, nullptr, &err)) return Affine{};
                const Data* o = pipe.Resolve(pipe.Viewers().back().source, &srcs);
                const auto* os = o ? std::get_if<ImageSet>(o) : nullptr;
                return (os && os->images.size() == 2) ? TransformOf(os->images[1])
                                                      : Affine{};
            };

            // A third of a pixel, under the half-pixel default: dropped.
            Check(SolvedShift(0.3f, 0.0f, 0.5f).IsIdentity(),
                  "a sub-threshold shift is left unwarped rather than resampled");

            // Three pixels, well over it: applied.
            const Affine big = SolvedShift(3.0f, 0.0f, 0.5f);
            Check(!big.IsIdentity() && std::abs(big.Dx() - 3.0f) < 0.1f,
                  "and a shift worth correcting still is (" +
                      std::to_string(big.Dx()) + ")");

            // The threshold is a control, not a hard rule: min_shift = 0 warps
            // everything, which is what the solver tests above rely on.
            Check(!SolvedShift(0.3f, 0.0f, 0.0f).IsIdentity(),
                  "min_shift = 0 applies every correction");
        }

        // ROUND TRIP: align then merge must be SHARPER than merging unaligned.
        //
        // This is the check that was missing, and its absence let a real bug
        // ship. The tests above verify the solver reports the right shift, and
        // it did -- but merge_hdr then applied the transform's INVERSE, which
        // negated the correction: instead of removing a displacement of d it
        // applied -d, leaving 2d of error. Tim saw it immediately as an aligned
        // merge blurrier than an unaligned one.
        //
        // No test of the solver alone can catch that, because the solver was
        // right. Only measuring the END of the chain can -- and sharpness is the
        // measure, because a wrong-magnitude warp blurs while a wrong-DIRECTION
        // warp blurs WORSE THAN DOING NOTHING. That asymmetry is what makes this
        // check able to tell them apart.
        {
            // Gradient energy: high for a sharp image, low for a blurred one.
            // Summed over the interior so the edge clamp does not contribute.
            auto Sharpness = [](const Image& im) {
                ImageView v = const_cast<Image&>(im).MapCpuRead();
                if (!v.data) return 0.0;
                PixelBuffer pb;
                pb.Unpack(v);
                double e = 0.0;
                for (int y = 4; y < pb.Height() - 4; ++y)
                    for (int x = 4; x < pb.Width() - 4; ++x) {
                        const float gx = pb.At(x + 1, y)[0] - pb.At(x - 1, y)[0];
                        const float gy = pb.At(x, y + 1)[0] - pb.At(x, y - 1)[0];
                        e += double(gx) * double(gx) + double(gy) * double(gy);
                    }
                return e;
            };

            auto MergeShifted = [&](bool withAlign, double* sharp) {
                ImageSet set;
                set.images.push_back(Textured(0.0f, 0.0f));
                set.images.push_back(Textured(1.7f, -1.3f));
                set.images.push_back(Textured(-0.9f, 2.1f));
                set.shape = Shape::Of("frame", 3);

                std::vector<SourceImage> nm{{"test", 0}};
                SourceImage gs; gs.name = "g"; gs.index = 1;
                gs.shape = Shape::Of("frame", 3);
                nm.push_back(gs);

                const char* src = withAlign
                    ? "g = image(\"g\")\na = align(g)\nm = merge_mean(a)\ndisplay(m)\n"
                    : "g = image(\"g\")\nm = merge_mean(g)\ndisplay(m)\n";

                Program prog; std::string err;
                if (!Parse(src, &prog, &err)) return false;
                UiState ui; Pipeline pipe;
                if (!Interpret(prog, nm, &ui, &pipe).ok) return false;

                std::vector<Data> srcs;
                srcs.push_back(Data{});
                srcs.push_back(Data{std::move(set)});
                if (!pipe.Execute(&srcs, nullptr, &err)) return false;

                const Data* o = pipe.Resolve(pipe.Viewers().back().source, &srcs);
                const auto* im = o ? std::get_if<Image>(o) : nullptr;
                if (!im) return false;
                *sharp = Sharpness(*im);
                return true;
            };

            double plain = 0.0, aligned = 0.0;
            if (MergeShifted(false, &plain) && MergeShifted(true, &aligned)) {
                // The threshold is set from measurement, not taste. On this
                // fixture the correct direction gives 1.03x and the inverted
                // one -- the bug that shipped -- gives 0.61x. Anything above 1
                // separates them cleanly; 1.01 leaves room for the bilinear
                // softening that any resample costs while still failing hard on
                // a wrong direction.
                //
                // The gain looks modest because a mean of three frames is
                // forgiving: two of them shift in different directions, so the
                // unaligned average is blurred but not catastrophically. The
                // asymmetry is the signal -- a wrong-direction warp lands at
                // 0.61, well below merging nothing at all.
                Check(aligned > plain * 1.01,
                      "align then merge is SHARPER than merging unaligned (" +
                          std::to_string(aligned / std::max(plain, 1e-9)) + "x)");
            } else {
                Check(false, "the align-then-merge round trip runs");
            }
        }
    }
                }
            }
        }
    }

    // --- the pipe operator ---------------------------------------------------
    //
    // `=>` is sugar, so the test that matters is EQUIVALENCE: a piped script and
    // its nested twin must build the same pipeline. If that holds, everything
    // downstream -- caching, the stage list, error messages -- needs no separate
    // coverage, because it cannot tell the two apart.
    {
        UiState ui; Pipeline pPipe, pNest; std::string err; std::vector<Data> src;

        const bool okPipe = RunScript(
            "src = image(\"test\")\n"
            "out = src => gaussian_blur(sigma = 2) => brightness(brightness = 0.1)\n"
            "display(out)\n", &ui, &pPipe, &err, &src);
        Check(okPipe, "a pipe chain runs" + (okPipe ? "" : ": " + err));

        UiState ui2; std::vector<Data> src2;
        const bool okNest = RunScript(
            "src = image(\"test\")\n"
            "out = brightness(gaussian_blur(src, sigma = 2), brightness = 0.1)\n"
            "display(out)\n", &ui2, &pNest, &err, &src2);
        Check(okNest, "the nested twin runs" + (okNest ? "" : ": " + err));

        bool same = okPipe && okNest && pPipe.Stages().size() == pNest.Stages().size();
        if (same)
            for (size_t i = 0; i < pPipe.Stages().size(); ++i)
                if (pPipe.Stages()[i].algoName != pNest.Stages()[i].algoName) same = false;
        Check(same, "a pipe chain builds the same pipeline as the nested form");

        // Order matters: the pipe must read left to right. Reversed, this would
        // still be two stages and still "pass" a count-only check.
        Check(pPipe.Stages().size() == 2 &&
              pPipe.Stages()[0].algoName == "gaussian_blur" &&
              pPipe.Stages()[1].algoName == "brightness",
              "the pipe applies left to right, not inside out");
    }

    // The piped value lands FIRST, ahead of arguments already written.
    {
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        const bool ok = RunScript(
            "src = image(\"test\")\n"
            "out = src => brightness(brightness = 0.25)\n"
            "display(out)\n", &ui, &p, &err, &src);
        Check(ok && p.Stages().size() == 1,
              "a pipe splices ahead of named arguments" + (ok ? "" : ": " + err));
    }

    // Bare callee: `x => f` with no argument list means f(x).
    {
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        const bool ok = RunScript(
            "src = image(\"test\")\n"
            "out = src => grayscale\n"
            "display(out)\n", &ui, &p, &err, &src);
        Check(ok && p.Stages().size() == 1 && p.Stages()[0].algoName == "grayscale",
              "a pipe into a bare name needs no empty parens" + (ok ? "" : ": " + err));
    }

    // The question that prompted the feature: does a pipe work when the
    // algorithm is a VARIABLE returned by choose()? It does, and for a reason
    // worth stating -- the parser desugars into the same Call node `f(src)`
    // builds, and call position was already general enough to hold a variable.
    {
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        const char* kScript =
            "src = image(\"test\")\n"
            "f = choose(\"op\", [brightness, grayscale])\n"
            "out = src => f()\n"
            "display(out)\n";

        const bool ok = RunScript(kScript, &ui, &p, &err, &src);
        Check(ok && p.Stages().size() == 1 && p.Stages()[0].algoName == "brightness",
              "a pipe into a choose() variable runs" + (ok ? "" : ": " + err));

        if (UiControl* c = ui.Find("op")) c->selected = 1;
        Pipeline p2;
        RunScript(kScript, &ui, &p2, &err, &src);
        Check(p2.Stages().size() == 1 && p2.Stages()[0].algoName == "grayscale",
              "switching the dropdown swaps the piped algorithm too");
    }

    // A pipe with no parens into a choose() variable: `src => f`.
    {
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        const bool ok = RunScript(
            "src = image(\"test\")\n"
            "f = choose(\"op\", [grayscale, brightness])\n"
            "out = src => f\n"
            "display(out)\n", &ui, &p, &err, &src);
        Check(ok && p.Stages().size() == 1 && p.Stages()[0].algoName == "grayscale",
              "a bare pipe into a choose() variable runs" + (ok ? "" : ": " + err));
    }

    // A chain broken across lines after the arrow, which is how a long one
    // wants to be written.
    {
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        const bool ok = RunScript(
            "src = image(\"test\")\n"
            "out = src =>\n"
            "      gaussian_blur(sigma = 1) =>\n"
            "      grayscale()\n"
            "display(out)\n", &ui, &p, &err, &src);
        Check(ok && p.Stages().size() == 2,
              "a pipe chain may break across lines" + (ok ? "" : ": " + err));
    }

    // The other line-break style: the arrow LEADING the continuation line,
    // which is how these chains are usually written when each stage gets its
    // own line. Worth supporting because it is the more common convention, and
    // the one a reader is most likely to try.
    {
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        const bool ok = RunScript(
            "src = image(\"test\")\n"
            "out = src\n"
            "    => gaussian_blur(sigma = 1)\n"
            "    => grayscale()\n"
            "display(out)\n", &ui, &p, &err, &src);
        Check(ok && p.Stages().size() == 2,
              "a chain may lead its continuation lines with '=>'" + (ok ? "" : ": " + err));
    }

    // Precedence against arithmetic, in both directions. Pipes bind loosest, so
    // the left side may be a full expression while the right side is only the
    // callee -- `2 + 3 => f()` pipes 5, and trailing arithmetic after a piped
    // call is NOT folded into the pipe.
    {
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        const bool ok = RunScript(
            "src = image(\"test\")\n"
            "n = 1 + 1\n"
            "out = src => gaussian_blur(sigma = n)\n"
            "display(out)\n", &ui, &p, &err, &src);
        Check(ok && p.Stages().size() == 1,
              "arithmetic evaluates before it is piped" + (ok ? "" : ": " + err));
    }

    // Piping into display(). It is a builtin rather than a registered
    // algorithm, but the pipe does not care: it splices into the argument list
    // before anything looks at the name, and display() already takes its image
    // first. So a whole chain can end in a viewer.
    {
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        const bool ok = RunScript(
            "src = image(\"test\")\n"
            "src => gaussian_blur(sigma = 2) => display(\"piped view\")\n",
            &ui, &p, &err, &src);
        Check(ok && p.Viewers().size() == 1 && p.Stages().size() == 1,
              "a chain can end in display()" + (ok ? "" : ": " + err));
        Check(ok && !p.Viewers().empty() && p.Viewers()[0].name == "piped view",
              "the piped display() keeps its name argument");
    }

    // The same, into a bare display with no arguments at all.
    {
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        const bool ok = RunScript(
            "src = image(\"test\")\n"
            "src => grayscale() => display\n", &ui, &p, &err, &src);
        Check(ok && p.Viewers().size() == 1,
              "a chain can end in a bare display" + (ok ? "" : ": " + err));
    }

    // Piping into params(). This is the shape autodevelop.tgl uses:
    //
    //     developed = params(basic_adjust, auto_exposure = 1)(src)
    //
    // params() returns an algorithm handle and the trailing (src) calls it, so
    // the piped form just moves src to the front of that call. Nothing about
    // params() had to change -- it was already an expression in call position,
    // which is the same generality that made choose() work.
    {
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        const bool ok = RunScript(
            "src = image(\"test\")\n"
            "out = src => params(gaussian_blur, sigma = 3)()\n"
            "display(out)\n", &ui, &p, &err, &src);
        Check(ok && p.Stages().size() == 1 && p.Stages()[0].algoName == "gaussian_blur",
              "a pipe into params() runs" + (ok ? "" : ": " + err));

        // The value params() set must survive the splice -- the whole point of
        // params() is that it seeds the control, so a pipe that dropped it
        // would still run and still be wrong.
        // Compared against the nested form rather than against a hardcoded 3.0:
        // what matters is that the pipe changes nothing, and if params()
        // seeding works differently than I assume, this still tests the pipe.
        UiState ui2; Pipeline p2; std::vector<Data> src2;
        RunScript("src = image(\"test\")\n"
                  "out = params(gaussian_blur, sigma = 3)(src)\n"
                  "display(out)\n", &ui2, &p2, &err, &src2);

        bool same = ui.Controls().size() == ui2.Controls().size();
        if (same)
            for (size_t i = 0; i < ui.Controls().size(); ++i)
                if (ui.Controls()[i].label != ui2.Controls()[i].label ||
                    std::abs(ui.Controls()[i].value - ui2.Controls()[i].value) > 1e-9)
                    same = false;
        Check(same, "params() through a pipe declares the same controls as nested");

        // And the value params() set is actually in there. Checked on `def`
        // rather than `value`: params() seeds the DEFAULT -- that is its whole
        // job, "where the controls start" -- and a fresh control opens on it.
        // The value params() set must actually be there. This was NOT true
        // before -- DescribeControl reported the declared default rather than
        // the probe's current value, so `params(op, sigma = 3)` opened the
        // slider at 2 and ran the stage at 2, silently. Found while testing
        // pipes; the nested form was equally affected.
        //
        // `def` stays the declared default, because that is where double-click
        // resets to and what marks a control untouched.
        for (const UiControl& c : ui.Controls())
            if (c.label == "gaussian_blur.sigma") {
                Check(std::abs(c.value - 3.0) < 1e-6,
                      "params() sets the control's value, not just its default");
                Check(std::abs(c.def - 2.0) < 1e-6,
                      "params() leaves the declared default alone, so reset still works");
            }
    }

    // The same, with two instances kept apart by name. If the pipe interfered
    // with the instance key, these would collapse onto one set of controls.
    {
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        const bool ok = RunScript(
            "src = image(\"test\")\n"
            "a = src => params(gaussian_blur, \"soft\", sigma = 1)()\n"
            "b = src => params(gaussian_blur, \"hard\", sigma = 8)()\n"
            "display(a)\n"
            "display(b)\n", &ui, &p, &err, &src);
        Check(ok && p.Stages().size() == 2,
              "two piped params() instances both run" + (ok ? "" : ": " + err));

        int sigmas = 0;
        for (const UiControl& c : ui.Controls())
            if (c.label.find("sigma") != std::string::npos) ++sigmas;
        Check(sigmas == 2, "piped params() instances keep separate controls");
    }

    // display() mid-chain.
    {
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        const bool ok = RunScript(
            "src = image(\"test\")\n"
            "out = src => gaussian_blur(sigma = 2) => display(\"mid\") => grayscale()\n"
            "display(out, \"final\")\n", &ui, &p, &err, &src);
        Check(ok, "display() can sit mid-chain" + (ok ? "" : ": " + err));
        Check(ok && p.Stages().size() == 2 && p.Viewers().size() == 2,
              "a mid-chain display taps without adding a stage");

        // The tap must show the INTERMEDIATE, not the final result -- that is
        // the entire point. The "mid" viewer should point at the blur, and
        // "final" at the grayscale after it.
        const ViewerDecl* mid   = nullptr;
        const ViewerDecl* final = nullptr;
        for (const ViewerDecl& v : p.Viewers()) {
            if (v.name == "mid")   mid   = &v;
            if (v.name == "final") final = &v;
        }
        Check(mid && final && mid->source.stage != final->source.stage,
              "the tap and the final view show different stages");
        Check(mid && p.Stages()[mid->source.stage].algoName == "gaussian_blur",
              "the tap shows the stage it was placed after");
    }

    // Two taps in one chain, which is the case that makes it worth having.
    {
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        const bool ok = RunScript(
            "src = image(\"test\")\n"
            "out = src => display(\"1 source\")\n"
            "          => gaussian_blur(sigma = 2) => display(\"2 blurred\")\n"
            "          => grayscale() => display(\"3 gray\")\n", &ui, &p, &err, &src);
        Check(ok && p.Viewers().size() == 3 && p.Stages().size() == 2,
              "a chain can be tapped at every step" + (ok ? "" : ": " + err));
    }

    // Returning a value must not break a bare display() statement, which has no
    // targets. Every existing script is this shape.
    {
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        const bool ok = RunScript(
            "src = image(\"test\")\n"
            "display(src, \"plain\")\n", &ui, &p, &err, &src);
        Check(ok && p.Viewers().size() == 1,
              "a bare display() statement still works" + (ok ? "" : ": " + err));
    }

    // And assigning from it gives back the same image, so `x = display(y)` is
    // a pass-through rather than a surprise.
    {
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        const bool ok = RunScript(
            "src = image(\"test\")\n"
            "a = src => gaussian_blur(sigma = 2)\n"
            "b = display(a, \"tap\")\n"
            "display(b, \"same\")\n", &ui, &p, &err, &src);
        Check(ok && p.Stages().size() == 1 && p.Viewers().size() == 2,
              "display() returns its input unchanged" + (ok ? "" : ": " + err));
        bool samePort = ok && p.Viewers().size() == 2 &&
                        p.Viewers()[0].source.stage == p.Viewers()[1].source.stage &&
                        p.Viewers()[0].source.port  == p.Viewers()[1].source.port;
        Check(samePort, "the returned port is the same one, not a copy");
    }

    // Control order follows the order the stages run, which is denoise.tgl's
    // shape: two params() calls in one pipe chain.
    {
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        const bool ok = RunScript(
            "image(\"test\") => params(basic_adjust, auto_exposure = 1)()\n"
            "               => display(\"original\")\n"
            "               => params(wavelet_denoise)()\n"
            "               => display(\"denoised\")\n", &ui, &p, &err, &src);
        Check(ok, "denoise.tgl's two-params chain runs" + (ok ? "" : ": " + err));

        // basic_adjust runs first, so its sliders come first. Before, the
        // callee was resolved before its arguments, so the LAST stage in a
        // chain declared its controls first and the panel read backwards.
        int firstAdjust = -1, firstDenoise = -1;
        for (const UiControl& c : ui.Controls()) {
            if (firstAdjust < 0 && c.label.rfind("basic_adjust.", 0) == 0)
                firstAdjust = c.declOrder;
            if (firstDenoise < 0 && c.label.rfind("wavelet_denoise.", 0) == 0)
                firstDenoise = c.declOrder;
        }
        Check(firstAdjust >= 0 && firstDenoise >= 0 && firstAdjust < firstDenoise,
              "controls appear in the order their stages run, not reversed");
    }

    // The same ordering for the nested spelling, which had the identical bug.
    {
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        const bool ok = RunScript(
            "src = image(\"test\")\n"
            "out = params(wavelet_denoise)(params(basic_adjust)(src))\n"
            "display(out)\n", &ui, &p, &err, &src);

        int firstAdjust = -1, firstDenoise = -1;
        for (const UiControl& c : ui.Controls()) {
            if (firstAdjust < 0 && c.label.rfind("basic_adjust.", 0) == 0)
                firstAdjust = c.declOrder;
            if (firstDenoise < 0 && c.label.rfind("wavelet_denoise.", 0) == 0)
                firstDenoise = c.declOrder;
        }
        Check(ok && firstAdjust >= 0 && firstDenoise >= 0 && firstAdjust < firstDenoise,
              "nested calls order their controls the same way" + (ok ? "" : ": " + err));
    }

    // Evaluating arguments before the callee must not record a stage twice --
    // the arguments are cached rather than evaluated again.
    {
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        const bool ok = RunScript(
            "src = image(\"test\")\n"
            "out = src => params(gaussian_blur, sigma = 2)()\n"
            "          => params(grayscale)()\n"
            "display(out)\n", &ui, &p, &err, &src);
        Check(ok && p.Stages().size() == 2,
              "a piped params() chain records each stage once" + (ok ? "" : ": " + err));
    }

    // '=>' must not confuse the assignment lookahead. This is the concrete
    // reason the lexer emits one token rather than '=' followed by '>': the
    // statement parser calls a line an assignment when it sees a bare '=' at
    // depth zero, so a two-token pipe would make this an assignment to `src`
    // and then fail on the '>'.
    {
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        const bool ok = RunScript(
            "src = image(\"test\")\n"
            "display(src => grayscale())\n", &ui, &p, &err, &src);
        Check(ok && p.Stages().size() == 1,
              "a pipe in expression position is not read as an assignment" +
                  (ok ? "" : ": " + err));
    }

    // Piping into something that is not callable must say so clearly, naming
    // what it actually got.
    {
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        RunScript("src = image(\"test\")\n"
                  "n = 5\n"
                  "out = src => n()\n", &ui, &p, &err, &src);
        Check(!err.empty(), "piping into a number is an error: \"" + err + "\"");
    }

    // --- every shipped script parses ----------------------------------------
    //
    // The scripts in scripts/ are what the app opens and what a new user reads
    // first, so a syntax error in one is a broken front door. Nothing else
    // covered them: the tests all use inline source, so a stale script could
    // ship indefinitely without a single test noticing.
    //
    // Parse only, not run: running needs images and a GPU, but a typo is a
    // parse error, and that is the failure this is guarding against.
#ifdef TGLAB_SOURCE_DIR
    {
        namespace fs = std::filesystem;
        const fs::path dir = fs::path(TGLAB_SOURCE_DIR) / "scripts";
        int seen = 0;
        std::error_code ec;
        for (const fs::directory_entry& f : fs::directory_iterator(dir, ec)) {
            if (f.path().extension() != ".tgl") continue;
            ++seen;
            std::ifstream in(f.path());
            std::stringstream ss;
            ss << in.rdbuf();

            Program prog;
            std::string err;
            const bool ok = Parse(ss.str(), &prog, &err);
            Check(ok, "scripts/" + f.path().filename().string() + " parses" +
                          (ok ? "" : ": " + err));
        }
        Check(seen > 0, "found scripts to check (" + std::to_string(seen) + ")");

        // The two scripts written in pipe form get run, not just parsed: they
        // are the ones exercising params() inside a chain, which is where the
        // control-order and dropped-value bugs lived. Parsing would not have
        // caught either.
        //
        // Only these two, and only against the stock test image: most scripts
        // want a raw file that is not in the repo.
        for (const char* name : {"autodevelop.tgl", "denoise.tgl"}) {
            std::ifstream in(dir / name);
            std::stringstream ss;
            ss << in.rdbuf();

            UiState ui; Pipeline p; std::string err; std::vector<Data> psrc;
            const bool ok = RunScript(ss.str(), &ui, &p, &err, &psrc);
            Check(ok, std::string("scripts/") + name + " runs" + (ok ? "" : ": " + err));
            Check(ok && !p.Stages().empty() && !p.Viewers().empty(),
                  std::string("scripts/") + name + " records stages and viewers");
        }
    }
#endif

    // --- live per-stage stats ------------------------------------------------
    //
    // The tallies must advance DURING a run, not only at the end. Checked by
    // watching Progress from another thread while a multi-stage pipeline runs
    // and recording whether the counts were ever seen part-way -- which is the
    // only way to tell "updated per stage" from "updated once at the end", and
    // the thing the todo item actually asked for.
    {
        UiState ui; Pipeline p; std::string err; std::vector<Data> src;
        const bool built = RunScript(
            "src = image(\"test\")\n"
            "out = src => gaussian_blur(sigma = 3)\n"
            "          => gaussian_blur(sigma = 3)\n"
            "          => gaussian_blur(sigma = 3)\n"
            "          => gaussian_blur(sigma = 3)\n"
            "display(out)\n", &ui, &p, &err, &src);
        Check(built && p.Stages().size() == 4,
              "built a four-stage pipeline to watch" + (built ? "" : ": " + err));

        if (built) {
            // Recorded from INSIDE the run, via the SetStats hook, rather than
            // by polling from another thread.
            //
            // A watcher thread was the obvious approach and measured nothing:
            // the test image is 4x4, so four blurs finish long before the
            // watcher is scheduled. It failed every run while telling us only
            // about thread startup. Overriding the publish point instead makes
            // the sequence exact and the test deterministic.
            struct Recording : Progress {
                std::vector<std::pair<int, double>> seen;   // {stages, elapsedMs}
                void SetStats(int cpu, int gpu, double ms, double gpuMs = 0.0) override {
                    Progress::SetStats(cpu, gpu, ms, gpuMs);
                    seen.emplace_back(cpu + gpu, ms);
                }
            } prog;

            std::string rerr;
            Pipeline fresh = std::move(p);
            const bool ran = fresh.Execute(&src, nullptr, &rerr, nullptr,
                                           ExecMode::ForceCPU, nullptr, nullptr, &prog);
            Check(ran, "the watched pipeline ran" + (ran ? "" : ": " + rerr));

            // Published more than once: that is the whole point of the change.
            // One publish would mean the tallies still only appear at the end.
            Check(prog.seen.size() >= 4,
                  "stats are published per stage, not once at the end (" +
                      std::to_string(prog.seen.size()) + " publishes)");

            // And they climb: each publish reports the stages finished so far,
            // so the sequence must be non-decreasing and must reach 4.
            bool climbs = !prog.seen.empty();
            for (size_t i = 1; i < prog.seen.size(); ++i)
                if (prog.seen[i].first < prog.seen[i - 1].first) climbs = false;
            Check(climbs, "the published stage count never goes backwards");

            // The final tally accounts for every stage INCLUDING the last --
            // publishes happen before each stage runs, so without the one after
            // the loop the last stage would never be counted.
            Check(!prog.seen.empty() && prog.seen.back().first >= 4,
                  "the last stage is included in the final tally");
            Check(prog.CpuStages() + prog.GpuStages() >= 4,
                  "the published total matches the stage count");

            // No GPU context, so the GPU share must be exactly zero rather
            // than some small leaked number. The status line divides by the
            // total to show a percentage, and a stray value would render a
            // CPU-only run as partly GPU.
            Check(prog.GpuMs() == 0.0, "a CPU-only run reports no GPU time");
        }
    }

    // --- tonemap_local -------------------------------------------------------
    //
    // Built on a synthetic scene with the structure the operator exists for: a
    // large ILLUMINATION step (bright region beside dark) with fine TEXTURE of
    // the same relative amplitude in both. A global curve cannot keep the
    // texture in both halves while also closing the gap between them; that
    // separation is exactly what is being tested.
    {
        constexpr int kW = 128, kH = 96;
        auto scene = [&] {
            Image img;
            ImageDesc d{kW, kH, Format::RGBA32F};
            d.linear = true;
            img.Alloc(d);
            ImageView v = img.MapCpuWrite();
            for (int y = 0; y < kH; ++y)
                for (int x = 0; x < kW; ++x) {
                    // Four stops of illumination between the halves.
                    const float illum = (x < kW / 2) ? 0.25f : 4.0f;
                    // Multiplicative texture: the SAME local contrast ratio in
                    // both halves, which is what the log split should preserve
                    // identically on each side.
                    const float tex = 1.0f + 0.25f * float(((x / 2 + y / 2) & 1) ? 1 : -1);
                    float* p = v.At<float>(x, y);
                    p[0] = p[1] = p[2] = illum * tex;
                    p[3] = 1.0f;
                }
            return img;
        };

        // Mean |gradient| / value: local contrast, invariant to a rescale.
        // Raw gradient energy would credit any operator that merely brightened
        // the image, which is the one thing a tone mapper is guaranteed to do.
        auto localContrast = [&](Image& im, int x0, int x1) {
            ImageView v = im.MapCpuRead();
            double acc = 0.0; size_t n = 0;
            for (int y = 2; y < kH - 2; ++y)
                for (int x = x0 + 2; x < x1 - 2; ++x) {
                    const float c = v.At<float>(x, y)[1];
                    if (c < 1e-5f) continue;
                    const float gx = v.At<float>(x + 1, y)[1] - v.At<float>(x - 1, y)[1];
                    const float gy = v.At<float>(x, y + 1)[1] - v.At<float>(x, y - 1)[1];
                    acc += std::sqrt(double(gx) * gx + double(gy) * gy) / double(c);
                    ++n;
                }
            return n ? acc / double(n) : 0.0;
        };

        auto runOp = [&](const std::string& call, Image* out) {
            auto algo = Registry::Get().Create("tonemap_local");
            if (!algo) return false;
            Pipeline p;
            std::vector<Data> srcs;
            srcs.push_back(Data{scene()});
            p.AddStage(std::move(algo), "tonemap_local", {{-1, 0}}, 1, 1);
            std::string e;
            if (!p.Execute(&srcs, nullptr, &e)) return false;
            const Data* d = p.Resolve({0, 0}, &srcs);
            const auto* im = d ? std::get_if<Image>(d) : nullptr;
            if (!im) return false;
            *out = const_cast<Image&>(*im).Clone();
            (void)call;
            return true;
        };

        Image src = scene();
        const double refDark   = localContrast(src, 0, kW / 2);
        const double refBright = localContrast(src, kW / 2, kW);

        // The fixture must actually have equal relative contrast on both sides,
        // or the comparison below proves nothing about the operator.
        Check(refDark > 0.05 && std::fabs(refDark - refBright) / refDark < 0.05,
              "fixture: both halves carry the same local contrast");

        Image mapped;
        if (runOp("tonemap_local(src)", &mapped)) {
            const double gotDark   = localContrast(mapped, 0, kW / 2);
            const double gotBright = localContrast(mapped, kW / 2, kW);

            // The illumination step must shrink. This is the compression.
            ImageView a = src.MapCpuRead();
            ImageView b = mapped.MapCpuRead();
            const double stepBefore =
                std::log2(double(a.At<float>(kW - 8, kH / 2)[1]) /
                          std::max(double(a.At<float>(8, kH / 2)[1]), 1e-9));
            const double stepAfter =
                std::log2(double(b.At<float>(kW - 8, kH / 2)[1]) /
                          std::max(double(b.At<float>(8, kH / 2)[1]), 1e-9));
            Check(std::fabs(stepAfter) < std::fabs(stepBefore) * 0.95,
                  "the illumination step is compressed (" +
                      std::to_string(stepBefore) + " -> " +
                      std::to_string(stepAfter) + " stops)");

            // And the texture must survive in BOTH halves, not just the one the
            // curve happened to favour. A global operator squeezing this scene
            // loses the bright half's texture; that is the failure this
            // algorithm exists to avoid.
            //
            // The threshold is 0.9, not something looser, and that was
            // calibrated rather than guessed: compressing in LINEAR space
            // instead of log -- scaling the detail along with the base, the
            // single most likely way to get this wrong -- leaves 0.629 against
            // the reference 0.754, which is 83%. A 0.6 threshold passed that
            // broken version happily. Verified by making the change and
            // watching this check stay green before tightening it.
            Check(gotDark > refDark * 0.9,
                  "texture survives in the dark half (" + std::to_string(gotDark) +
                      " vs " + std::to_string(refDark) + ")");
            Check(gotBright > refBright * 0.9,
                  "texture survives in the bright half (" + std::to_string(gotBright) +
                      " vs " + std::to_string(refBright) + ")");

            // Both halves must keep it to a SIMILAR degree. Preserving one and
            // crushing the other is precisely what a global curve does, so this
            // is the check that distinguishes local from global.
            const double ratio = (gotBright > 0.0) ? gotDark / gotBright : 0.0;
            Check(ratio > 0.7 && ratio < 1.4,
                  "both halves keep their texture equally (ratio " +
                      std::to_string(ratio) + ")");
        } else {
            Check(false, "tonemap_local ran");
        }
    }

    // --- a cancelled run keeps its finished prefix ---------------------------
    //
    // Tim's report: dragging a develop slider far enough to cancel a run made
    // the ENTIRE pipeline re-run -- demosaic, hot-pixel repair, the HDR merge,
    // everything -- rather than just the stage whose parameter moved.
    //
    // The cause was the worker dropping the whole stage cache on cancellation.
    // The invariant that makes keeping it safe was already in place: Execute()
    // marks a stage invalid BEFORE running it and valid only on success, and
    // the cache scan stops at the first invalid stage. So the completed prefix
    // of an abandoned run is exactly as reusable as any other prev.
    //
    // Tested at the Pipeline level rather than through the worker, because the
    // worker's cancellation is driven by a second job arriving and that race
    // cannot be made deterministic. Cancelling from the progress hook at a
    // chosen stage boundary reproduces the same state exactly.
    {
        // Cancels the run once a named stage is reached, which is the same
        // thing a newer job does mid-drag.
        struct CancelAt : Progress {
            std::string   target;
            CancelToken*  token = nullptr;
            int           reached = 0;
            void Set(int done, int total, const char* what) override {
                Progress::Set(done, total, what);
                if (what && target == what && token) { ++reached; token->Cancel(); }
            }
        };

        auto build = [&](Pipeline* p, std::vector<Data>* srcs, UiState* ui,
                         const char* script, std::string* err) {
            return RunScript(script, ui, p, err, srcs);
        };

        // Four stages, so there is a meaningful prefix to preserve.
        const char* kScript =
            "src = image(\"test\")\n"
            "out = src => gaussian_blur(sigma = 2)\n"
            "          => grayscale()\n"
            "          => box_blur(radius = 2)\n"
            "          => brightness(brightness = 0.1)\n"
            "display(out)\n";

        UiState ui1; Pipeline first; std::vector<Data> src1; std::string err;
        const bool built = build(&first, &src1, &ui1, kScript, &err);
        Check(built && first.Stages().size() == 4,
              "built a four-stage pipeline to cancel" + (built ? "" : ": " + err));

        if (built) {
            CancelToken tok;
            CancelAt prog;
            prog.target = "box_blur";     // cancel at stage 2 of 4
            prog.token  = &tok;

            std::string cerr;
            const bool ran = first.Execute(&src1, nullptr, &cerr, nullptr,
                                           ExecMode::ForceCPU, nullptr, &tok, &prog);
            Check(!ran && cerr == Pipeline::kCancelled,
                  "the run was cancelled part-way");
            Check(prog.reached == 1, "cancellation happened at the chosen stage");

            // The stages BEFORE the cancellation point completed and must still
            // be valid; the one it stopped at must not be.
            Check(first.Stages()[0].valid && first.Stages()[1].valid,
                  "stages finished before the cancel stay valid");
            Check(!first.Stages()[2].valid,
                  "the cancelled stage is marked invalid");

            // Now re-run with the abandoned pipeline as prev, exactly as the
            // worker now does. The finished prefix must be REUSED rather than
            // recomputed -- that is the whole fix.
            UiState ui2; Pipeline second; std::vector<Data> src2;
            if (build(&second, &src2, &ui2, kScript, &err)) {
                std::string rerr;
                const bool ok2 = second.Execute(&src2, &first, &rerr, nullptr,
                                                ExecMode::ForceCPU);
                Check(ok2, "the follow-up run completed" + (ok2 ? "" : ": " + rerr));
                Check(ok2 && second.CachedStageCount() == 2,
                      "the finished prefix was reused, not recomputed (" +
                          std::to_string(second.CachedStageCount()) + " cached)");
                Check(ok2 && second.FirstDirtyStage() == 2,
                      "the re-run starts at the stage that was cancelled");
            } else {
                Check(false, "rebuilt the pipeline for the follow-up run");
            }
        }
    }

    // --- bypassing a stage that would do nothing -----------------------------
    //
    // For stacking. A script emulating a full develop pipeline wants twenty
    // effects available and three of them used, and an unused effect must cost
    // NOTHING -- not "one cheap pass", since twenty cheap passes over a 45 MP
    // image is not cheap and each one also allocates a full-size intermediate.
    {
        UiState ui; Pipeline p; std::vector<Data> src; std::string err;
        const bool ok = RunScript(
            "src = image(\"test\")\n"
            "out = src => gaussian_blur(sigma = 0)\n"
            "          => gaussian_blur(sigma = 0)\n"
            "          => brightness(brightness = 0.25)\n"
            "display(out)\n", &ui, &p, &err, &src);

        Check(ok && p.Stages().size() == 3,
              "a stack with two disabled effects builds" + (ok ? "" : ": " + err));
        Check(ok && p.BypassedStageCount() == 2,
              "both zero-sigma blurs were bypassed (" +
                  std::to_string(p.BypassedStageCount()) + ")");

        // Free means FREE: a bypassed stage holds no image at all. Checking the
        // output is empty is what distinguishes a real skip from a stage that
        // ran and happened to copy its input.
        Check(ok && !p.Stages()[0].outputs.empty() &&
              TypeOf(p.Stages()[0].outputs[0]) == DataType::None,
              "a bypassed stage allocates nothing");

        // And the pixels must still be right. Two bypassed stages back to back
        // means Resolve has to walk the chain, not just one link.
        const Data* d = p.Resolve({2, 0}, &src);
        const auto* got = d ? std::get_if<Image>(d) : nullptr;
        Check(got && got->Valid(), "the stack still produces an image");

        // Compare against the same script with the disabled stages removed
        // entirely -- the bypass must be indistinguishable from not writing
        // them.
        UiState ui2; Pipeline p2; std::vector<Data> src2;
        const bool ok2 = RunScript(
            "src = image(\"test\")\n"
            "out = src => brightness(brightness = 0.25)\n"
            "display(out)\n", &ui2, &p2, &err, &src2);
        const Data* d2 = ok2 ? p2.Resolve({0, 0}, &src2) : nullptr;
        const auto* ref = d2 ? std::get_if<Image>(d2) : nullptr;

        if (got && ref) {
            ImageView a = const_cast<Image*>(got)->MapCpuRead();
            ImageView b = const_cast<Image*>(ref)->MapCpuRead();
            int worst = 0;
            for (int y = 0; y < a.desc.height; ++y)
                for (int x = 0; x < a.desc.width; ++x)
                    for (int c = 0; c < 3; ++c)
                        worst = std::max(worst,
                            std::abs(int(a.At<uint8_t>(x, y)[c]) -
                                     int(b.At<uint8_t>(x, y)[c])));
            Check(worst == 0,
                  "bypassing is identical to omitting the stage (worst " +
                      std::to_string(worst) + ")");
        } else {
            Check(false, "both forms produced an image");
        }
    }

    // Turning an effect back on re-runs it, because IsNoOp is derived from
    // parameters and those are already in the stage hash.
    {
        UiState ui; Pipeline p; std::vector<Data> src; std::string err;
        const char* kScript =
            "src = image(\"test\")\n"
            "s = slider(\"blur\", 0.0, 8.0, 0.0)\n"
            "out = src => gaussian_blur(sigma = s)\n"
            "display(out)\n";

        RunScript(kScript, &ui, &p, &err, &src);
        Check(p.BypassedStageCount() == 1, "the slider starts at zero and bypasses");

        if (UiControl* c = ui.Find("blur")) c->value = 3.0;
        Pipeline p2; std::vector<Data> src2;
        RunScript(kScript, &ui, &p2, &err, &src2);
        Check(p2.BypassedStageCount() == 0, "raising the slider runs the stage again");
    }

    // A stage whose output type differs from its input's must NOT be bypassed,
    // whatever it claims. tonemap declares RGBA32F, so bypassing it on an RGBA8
    // input would hand downstream a different format than the port promises.
    // The pipeline refuses regardless of what the algorithm says.
    {
        UiState ui; Pipeline p; std::vector<Data> src; std::string err;
        const bool ok = RunScript(
            "src = image(\"test\")\n"
            "out = src => params(tonemap_local, range = 12)()\n"
            "display(out)\n", &ui, &p, &err, &src);
        // range=12 makes it an identity in effect, but it does not claim
        // IsNoOp, so this is really checking that nothing bypasses by accident.
        Check(ok && p.BypassedStageCount() == 0,
              "a format-changing stage is not bypassed" + (ok ? "" : ": " + err));
    }

    // --- the enabled switch --------------------------------------------------
    //
    // Tim's ask, and it is a different thing from IsNoOp: he wants to get the
    // sliders right and THEN toggle the stage, without neutralising the
    // settings to disable it. So `enabled` is a real parameter that every
    // algorithm has, and turning it off bypasses the stage while leaving every
    // other control exactly where it was.
    {
        UiState ui; Pipeline p; std::vector<Data> src; std::string err;
        const char* kScript =
            "src = image(\"test\")\n"
            "out = src => params(gaussian_blur, sigma = 6)()\n"
            "display(out)\n";

        const bool ok = RunScript(kScript, &ui, &p, &err, &src);
        Check(ok && p.BypassedStageCount() == 0,
              "an enabled stage runs" + (ok ? "" : ": " + err));

        // Every algorithm has the control, without declaring it.
        UiControl* en = nullptr;
        for (UiControl& c : ui.Controls())
            if (c.label == "gaussian_blur.enabled") en = &c;
        Check(en != nullptr, "every algorithm has an `enabled` control");
        Check(en && en->kind == UiControl::Kind::Check,
              "`enabled` is a checkbox, not a slider");

        // First in the panel, above the controls it governs. That falls out of
        // base members being constructed before derived ones, so it is worth an
        // assertion -- a reordering elsewhere would move it silently.
        Check(!ui.Controls().empty() &&
              ui.Controls()[0].label == "gaussian_blur.enabled",
              "`enabled` sits above the controls it governs");

        // Switch it off: the stage is bypassed...
        if (en) en->value = 0.0;
        Pipeline p2; std::vector<Data> src2;
        RunScript(kScript, &ui, &p2, &err, &src2);
        Check(p2.BypassedStageCount() == 1, "switching it off bypasses the stage");

        // ...and sigma is STILL 6. This is the whole point: disabling by
        // neutralising would have lost it.
        bool kept = false;
        for (const UiControl& c : ui.Controls())
            if (c.label == "gaussian_blur.sigma") {
                // `value`, not `def`. params() seeds the DEFAULT, and once the
                // control exists a re-run preserves the user's value while def
                // stays the algorithm's own 2.0. The setting that must survive
                // being switched off is the one in effect.
                kept = std::abs(c.value - 6.0) < 1e-6;
            }
        Check(kept, "the settings survive being switched off");

        // Switch it back on and the stage runs again, at the settings it had.
        for (UiControl& c : ui.Controls())
            if (c.label == "gaussian_blur.enabled") c.value = 1.0;
        Pipeline p3; std::vector<Data> src3;
        RunScript(kScript, &ui, &p3, &err, &src3);
        Check(p3.BypassedStageCount() == 0, "switching it back on runs the stage");
    }

    // A disabled stage passes its input through unchanged -- the same guarantee
    // IsNoOp gives, reached the other way.
    {
        UiState ui; Pipeline p; std::vector<Data> src; std::string err;
        RunScript("src = image(\"test\")\n"
                  "out = src => params(gaussian_blur, sigma = 6)()\n"
                  "display(out)\n", &ui, &p, &err, &src);
        for (UiControl& c : ui.Controls())
            if (c.label == "gaussian_blur.enabled") c.value = 0.0;

        Pipeline off; std::vector<Data> srcOff;
        RunScript("src = image(\"test\")\n"
                  "out = src => params(gaussian_blur, sigma = 6)()\n"
                  "display(out)\n", &ui, &off, &err, &srcOff);

        // Against the source itself, which is what a disabled stage must equal.
        const Data* got = off.Resolve({0, 0}, &srcOff);
        const auto* im = got ? std::get_if<Image>(got) : nullptr;
        Check(im && im->Valid(), "a disabled stage still resolves to an image");
        if (im) {
            ImageView a = const_cast<Image*>(im)->MapCpuRead();
            ImageView b = std::get<Image>(srcOff[0]).MapCpuRead();
            int worst = 0;
            for (int y = 0; y < a.desc.height; ++y)
                for (int x = 0; x < a.desc.width; ++x)
                    for (int c = 0; c < 3; ++c)
                        worst = std::max(worst,
                            std::abs(int(a.At<uint8_t>(x, y)[c]) -
                                     int(b.At<uint8_t>(x, y)[c])));
            Check(worst == 0, "a disabled stage passes its input through untouched");
        }
    }

    // --- saving ---------------------------------------------------------------
    {
        Check(SaveFormatFromPath("a.png") == SaveFormat::Png, "png by extension");
        Check(SaveFormatFromPath("a.JPG") == SaveFormat::Jpg, "extensions are case-insensitive");
        Check(SaveFormatFromPath("a.jpeg") == SaveFormat::Jpg, "jpeg is jpg");
        Check(SaveFormatFromPath("a.hdr") == SaveFormat::Hdr, "hdr by extension");
        Check(SaveFormatFromPath("noext") == SaveFormat::Png, "png when there is nothing to go on");

        // A LINEAR image written to 8 bits must go through the display curve,
        // or the file does not match what the app showed. Middle grey is the
        // case worth pinning: 0.18 linear is 46/255 written raw and ~117/255
        // through the curve, which is the difference between a correct export
        // and a very dark one.
        {
            Image img;
            ImageDesc d{4, 4, Format::RGBA32F};
            d.linear = true;
            img.Alloc(d);
            ImageView v = img.MapCpuWrite();
            for (int y = 0; y < 4; ++y)
                for (int x = 0; x < 4; ++x) {
                    float* p = v.At<float>(x, y);
                    p[0] = p[1] = p[2] = 0.18f;
                    p[3] = 1.0f;
                }

            const std::string path = "save_test_linear.png";
            std::string err;
            Check(SaveImage(path, img, &err), "a linear float image saves" +
                  (err.empty() ? "" : ": " + err));

            Image back;
            if (LoadImageFile(path, &back, &err)) {
                ImageView bv = back.MapCpuRead();
                const int g = bv.At<uint8_t>(2, 2)[1];
                Check(g > 100 && g < 135,
                      "middle grey survives the round trip as " + std::to_string(g) +
                          "/255, not 46");
            } else {
                Check(false, "the saved PNG reads back: " + err);
            }
            std::remove(path.c_str());
        }

        // A GAMMA-encoded image is already display referred and must only be
        // clamped -- running the curve again would lighten it a second time.
        {
            Image img;
            img.Alloc({4, 4, Format::RGBA8});
            ImageView v = img.MapCpuWrite();
            for (int i = 0; i < 4 * 4 * 4; ++i) v.data[i] = 117;

            const std::string path = "save_test_gamma.png";
            std::string err;
            Check(SaveImage(path, img, &err), "an 8-bit image saves");

            Image back;
            if (LoadImageFile(path, &back, &err)) {
                ImageView bv = back.MapCpuRead();
                const int g = bv.At<uint8_t>(2, 2)[1];
                Check(g == 117, "8-bit values pass through untouched (got " +
                                    std::to_string(g) + ")");
            } else {
                Check(false, "the saved PNG reads back");
            }
            std::remove(path.c_str());
        }

        // .hdr keeps the linear values, which is the whole reason to offer it:
        // a merged bracket reaching far above 1.0 cannot survive 8 bits, and
        // clipping it silently would discard the headroom the merge captured.
        {
            Image img;
            ImageDesc d{8, 8, Format::RGBA32F};
            d.linear = true;
            img.Alloc(d);
            ImageView v = img.MapCpuWrite();
            for (int y = 0; y < 8; ++y)
                for (int x = 0; x < 8; ++x) {
                    float* p = v.At<float>(x, y);
                    p[0] = p[1] = p[2] = 40.0f;    // far above any 8-bit range
                    p[3] = 1.0f;
                }

            const std::string path = "save_test.hdr";
            std::string err;
            Check(SaveImage(path, img, SaveFormat::Hdr, 92, &err),
                  "a high-range image saves as .hdr" + (err.empty() ? "" : ": " + err));

            // Read back through stb, which returns .hdr as float.
            Image back;
            if (LoadImageFile(path, &back, &err)) {
                // LoadImageFile gives RGBA8, so it cannot show the headroom --
                // what matters here is that the file exists and is a valid HDR.
                Check(back.Valid(), "the .hdr reads back as an image");
            } else {
                Check(false, "the .hdr reads back: " + err);
            }
            std::remove(path.c_str());
        }

        // Auto-increment, for a script saving a group without clobbering.
        {
            const std::string base = "save_test_seq.png";
            Check(NextFreePath(base) == base, "a free name is returned unchanged");

            Image img;
            img.Alloc({2, 2, Format::RGBA8});
            std::string err;
            SaveImage(base, img, &err);

            const std::string next = NextFreePath(base);
            Check(next == "save_test_seq_1.png",
                  "an occupied name increments before the extension (got " + next + ")");

            SaveImage(next, img, &err);
            Check(NextFreePath(base) == "save_test_seq_2.png",
                  "and keeps counting past the first");

            std::remove(base.c_str());
            std::remove(next.c_str());
            std::remove("save_test_seq_2.png");
        }

        // A path whose only dot is in a DIRECTORY name has no extension, so the
        // counter must not split there and produce "a_1.b/out".
        Check(NextFreePath("no_such_dir.d/out") == "no_such_dir.d/out",
              "a dot in a directory name is not an extension");
    }

    // --- save() from a script -------------------------------------------------
    {
        UiState ui; Pipeline p; std::vector<Data> src; std::string err;
        const bool ok = RunScript(
            "src = image(\"test\")\n"
            "src => brightness(brightness = 0.1) => save(\"save_script.png\")\n",
            &ui, &p, &err, &src);
        Check(ok, "a script save() runs" + (ok ? "" : ": " + err));
        Check(ok && p.Saves().size() == 1, "the save was recorded");

        // Nothing is written until RunSaves: a save must not happen on every
        // slider tick, so Execute deliberately does not do it.
        Check(!std::filesystem::exists("save_script.png"),
              "Execute alone writes nothing");

        std::vector<std::string> wrote;
        const bool sok = p.RunSaves(&src, &err, &wrote);
        Check(sok && wrote.size() == 1, "RunSaves writes the file" +
              (sok ? "" : ": " + err));
        Check(!wrote.empty() && wrote[0] == "save_script.png",
              "at the path the script asked for");

        // Saving again must not clobber, since re-running a script is the
        // normal thing to do in this app.
        std::vector<std::string> again;
        p.RunSaves(&src, &err, &again);
        Check(!again.empty() && again[0] == "save_script_1.png",
              "a second run increments rather than overwriting (got " +
                  (again.empty() ? "nothing" : again[0]) + ")");

        for (const std::string& f : wrote) std::remove(f.c_str());
        for (const std::string& f : again) std::remove(f.c_str());
    }

    // existing = "overwrite" is the opposite choice, for exporting the current
    // state rather than accumulating copies.
    {
        UiState ui; Pipeline p; std::vector<Data> src; std::string err;
        const char* kScript =
            "src = image(\"test\")\n"
            "save(src, \"save_over.png\", existing = \"overwrite\")\n";
        RunScript(kScript, &ui, &p, &err, &src);

        std::vector<std::string> a, b;
        p.RunSaves(&src, &err, &a);
        p.RunSaves(&src, &err, &b);
        Check(!b.empty() && b[0] == "save_over.png",
              "overwrite keeps writing the same path");
        std::remove("save_over.png");
    }

    // A GROUP writes one file per frame, numbered -- a single path cannot name
    // N images, and a reduction consumes them in order so the numbering has to
    // match that order.
    {
        UiState ui; Pipeline p; std::string err;
        std::vector<Data> src;

        ImageSet set;
        for (int i = 0; i < 3; ++i) {
            Image im;
            im.Alloc({2, 2, Format::RGBA8});
            ImageView v = im.MapCpuWrite();
            for (int k = 0; k < 2 * 2 * 4; ++k) v.data[k] = uint8_t(40 * (i + 1));
            set.images.push_back(std::move(im));
        }
        set.shape = Shape{{{"frame", 3}}};
        src.push_back(Data{std::move(set)});

        std::vector<SourceImage> names;
        { SourceImage s; s.name = "g"; s.index = 0; s.shape = ShapeOf(src[0]); names.push_back(s); }

        Program prog;
        Check(Parse("g = image(\"g\")\nsave(g, \"save_grp.png\")\n", &prog, &err),
              "a group save parses");
        const auto r = Interpret(prog, names, &ui, &p);
        Check(r.ok, "and interprets" + (r.ok ? "" : ": " + r.error));

        if (r.ok && p.Execute(&src, nullptr, &err)) {
            std::vector<std::string> wrote;
            const bool sok = p.RunSaves(&src, &err, &wrote);
            Check(sok && wrote.size() == 3,
                  "a group writes one file per frame (" +
                      std::to_string(wrote.size()) + ")");
            // Zero padded so a directory listing sorts them in frame order.
            Check(wrote.size() == 3 && wrote[0] == "save_grp_001.png" &&
                  wrote[2] == "save_grp_003.png",
                  "numbered in frame order, zero padded");
            for (const std::string& f : wrote) std::remove(f.c_str());
        } else {
            Check(false, "the group pipeline ran: " + err);
        }
    }

    // Bad options are reported rather than silently ignored.
    {
        UiState ui; Pipeline p; std::vector<Data> src; std::string err;
        RunScript("src = image(\"test\")\nsave(src, \"x.png\", existing = \"maybe\")\n",
                  &ui, &p, &err, &src);
        Check(!err.empty(), "an unknown existing mode is an error: \"" + err + "\"");

        std::string err2;
        RunScript("src = image(\"test\")\nsave(src, \"x.png\", nonsense = 1)\n",
                  &ui, &p, &err2, &src);
        Check(!err2.empty(), "an unknown option is an error: \"" + err2 + "\"");

        std::string err3;
        RunScript("src = image(\"test\")\nsave(src)\n", &ui, &p, &err3, &src);
        Check(!err3.empty(), "a missing path is an error: \"" + err3 + "\"");
    }

    // --- SIFT -----------------------------------------------------------------
    //
    // The claim a detector has to earn is REPEATABILITY: the same scene point
    // must be found again after the image changes. Counting features proves
    // nothing -- a detector finding noise produces plenty.
    //
    // So the fixture is a synthetic scene of blobs at known positions, and the
    // tests shift it, scale it, and brighten it, then ask whether the same
    // points come back.
    {
        constexpr int kW = 256, kH = 256;

        // Blobs of varied size, at positions chosen not to be on a grid --
        // a regular lattice would let a detector score well by finding a
        // pattern rather than the blobs.
        struct Blob { float x, y, r; };
        const Blob kBlobs[] = {
            { 60.0f,  50.0f,  6.0f}, {170.0f,  40.0f, 10.0f},
            { 45.0f, 150.0f,  8.0f}, {200.0f, 120.0f,  5.0f},
            {120.0f, 190.0f, 12.0f}, { 90.0f, 105.0f,  7.0f},
        };

        auto scene = [&](float dx, float dy, float gain) {
            Image img;
            ImageDesc d{kW, kH, Format::RGBA32F};
            img.Alloc(d);
            ImageView v = img.MapCpuWrite();
            for (int y = 0; y < kH; ++y)
                for (int x = 0; x < kW; ++x) {
                    // Background texture, so the neighbourhoods around
                    // different blobs are actually DIFFERENT. Isotropic blobs
                    // on a flat field give near-identical rotation-normalised
                    // descriptors -- correctly, since the neighbourhoods really
                    // are the same shape -- which makes them useless for
                    // testing discrimination.
                    //
                    // NOT a sine grid, which was the first attempt. A periodic
                    // texture is self-similar, so a feature in it genuinely has
                    // several equally good positions and repeatability
                    // collapsed to 10% -- the detector was right and the
                    // fixture was asking an impossible question. Measured on a
                    // real photograph the same code repeats at ~70%.
                    //
                    // A hash of the coordinates instead: aperiodic, so each
                    // neighbourhood is unique, and deterministic so the test
                    // means the same thing every run.
                    const float fx = float(x) + dx, fy = float(y) + dy;
                    auto noise = [](int ix, int iy) {
                        uint32_t s = uint32_t(ix) * 374761393u + uint32_t(iy) * 668265263u;
                        s = (s ^ (s >> 13)) * 1274126177u;
                        return float(s >> 8) / float(1 << 24);
                    };
                    // Smoothed by bilinear interpolation over a coarse lattice,
                    // so the texture has structure at a scale SIFT can find
                    // rather than being per-pixel noise.
                    const float gx = fx / 9.0f, gy = fy / 9.0f;
                    const int   x0 = int(std::floor(gx)), y0 = int(std::floor(gy));
                    const float tx = gx - float(x0), ty = gy - float(y0);
                    const float n00 = noise(x0, y0),     n10 = noise(x0 + 1, y0);
                    const float n01 = noise(x0, y0 + 1), n11 = noise(x0 + 1, y0 + 1);
                    const float nx0 = n00 + (n10 - n00) * tx;
                    const float nx1 = n01 + (n11 - n01) * tx;
                    float a = 0.15f + 0.22f * (nx0 + (nx1 - nx0) * ty);
                    for (const Blob& b : kBlobs) {
                        const float ex = float(x) - (b.x + dx);
                        const float ey = float(y) - (b.y + dy);
                        a += 0.7f * std::exp(-(ex * ex + ey * ey) / (2.0f * b.r * b.r));
                    }
                    float* p = v.At<float>(x, y);
                    // `gain` is an OFFSET, not a multiply. The descriptor is
                    // normalised, so it is invariant to a scale on the
                    // gradients; what a multiply ALSO does is change contrast,
                    // which moves which features clear the detector threshold
                    // -- so the pairing compares different features and the
                    // test measures the threshold rather than the descriptor.
                    p[0] = p[1] = p[2] = a + gain;
                    p[3] = 1.0f;
                }
            return img;
        };

        auto detect = [&](Image&& in, std::vector<Keypoint>* kps,
                          DescriptorSet* desc) {
            auto algo = Registry::Get().Create("detect_sift");
            if (!algo) return false;
            Pipeline p;
            std::vector<Data> srcs;
            srcs.push_back(Data{std::move(in)});
            p.AddStage(std::move(algo), "detect_sift", {{-1, 0}}, 1, 1);
            std::string e;
            if (!p.Execute(&srcs, nullptr, &e)) return false;
            const Data* d = p.Resolve({0, 0}, &srcs);
            const auto* im = d ? std::get_if<Image>(d) : nullptr;
            if (!im) return false;
            const FeatureSidecar* fs = FeaturesOf(*im);
            if (!fs) return false;
            *kps  = fs->keypoints;
            *desc = fs->descriptors;
            return true;
        };

        std::vector<Keypoint> a;
        DescriptorSet da;
        const bool ok = detect(scene(0, 0, 0.0f), &a, &da);
        Check(ok, "detect_sift runs and attaches a sidecar");
        Check(ok && !a.empty(), "it finds features (" + std::to_string(a.size()) + ")");

        // Descriptors: one per keypoint, 128 floats, L2-normalised. A wrong
        // count here means the two arrays have drifted apart, which a matcher
        // would read as garbage rather than as an error.
        Check(da.kind == DescriptorKind::Float && da.dim == 128,
              "the descriptor is 128 floats");
        Check(da.Count() == a.size(),
              "one descriptor per keypoint (" + std::to_string(da.Count()) +
                  " vs " + std::to_string(a.size()) + ")");
        if (da.Count() > 0) {
            float n = 0.0f;
            for (int i = 0; i < 128; ++i) {
                const float v = da.FloatAt(0)[i];
                n += v * v;
            }
            Check(std::fabs(std::sqrt(n) - 1.0f) < 0.01f,
                  "descriptors are unit length (got " + std::to_string(std::sqrt(n)) + ")");
        }

        // Features must land ON the blobs. Without this the count could come
        // from anywhere in the frame and still look healthy.
        if (ok && !a.empty()) {
            int onBlob = 0;
            for (const Keypoint& k : a) {
                for (const Blob& b : kBlobs) {
                    const float dx = k.x - b.x, dy = k.y - b.y;
                    if (std::sqrt(dx * dx + dy * dy) < b.r * 1.5f) { ++onBlob; break; }
                }
            }
            Check(onBlob * 2 >= int(a.size()),
                  "most features are on the blobs (" + std::to_string(onBlob) +
                      " of " + std::to_string(a.size()) + ")");
        }

        // TRANSLATION: shift the scene and the same points must come back,
        // shifted by the same amount. This is the property alignment depends on.
        {
            std::vector<Keypoint> b;
            DescriptorSet db;
            if (detect(scene(12.0f, -7.0f, 0.0f), &b, &db) && !a.empty()) {
                // Counted over the features that COULD survive. A shift of
                // (12, -7) moves a band of the frame off the edge, and a
                // feature that left the image is not a repeatability failure --
                // scoring against every feature would penalise the detector for
                // the fixture's geometry.
                int matched = 0, eligible = 0;
                for (const Keypoint& ka : a) {
                    const float sx = ka.x + 12.0f, sy = ka.y - 7.0f;
                    if (sx < 8.0f || sy < 8.0f || sx > 248.0f || sy > 248.0f) continue;
                    ++eligible;
                    for (const Keypoint& kb : b) {
                        const float dx = (kb.x - 12.0f) - ka.x;
                        const float dy = (kb.y + 7.0f) - ka.y;
                        if (std::sqrt(dx * dx + dy * dy) < 2.0f) { ++matched; break; }
                    }
                }
                Check(eligible > 0 && matched * 2 >= eligible,
                      "most features survive a translation (" +
                          std::to_string(matched) + " of " +
                          std::to_string(eligible) + " that stayed in frame)");
            } else {
                Check(false, "the shifted scene detected");
            }
        }

        // ILLUMINATION: the descriptor is normalised and clipped precisely so a
        // brightness change does not alter it. Same scene at 1.4x gain must
        // give near-identical descriptors, not merely a similar count.
        {
            std::vector<Keypoint> b;
            DescriptorSet db;
            if (detect(scene(0, 0, 0.15f), &b, &db) && !a.empty() && !b.empty()) {
                // Pair by position AND ORIENTATION.
                //
                // Position alone is not enough, and getting that wrong is what
                // made this fail at 1.21 while the descriptor was fine: SIFT
                // emits several keypoints at ONE position when the gradient
                // histogram has several strong peaks (Lowe's rule, and it is
                // implemented here). Pairing on position alone therefore
                // compared a feature at angle 0 against the same point at angle
                // pi -- two deliberately different descriptors of the same
                // pixel.
                int compared = 0;
                double worst = 0.0;
                for (size_t i = 0; i < a.size(); ++i) {
                    for (size_t j = 0; j < b.size(); ++j) {
                        const float dx = b[j].x - a[i].x, dy = b[j].y - a[i].y;
                        if (std::sqrt(dx * dx + dy * dy) > 1.0f) continue;
                        float da2 = std::fabs(b[j].angle - a[i].angle);
                        if (da2 > 3.14159265f) da2 = 6.2831853f - da2;
                        if (da2 > 0.2f) continue;
                        const float d = DistanceL2Sq(da.FloatAt(i), db.FloatAt(j), 128);
                        worst = std::max(worst, double(std::sqrt(d)));
                        ++compared;
                        break;
                    }
                }
                Check(compared > 0, "found the same features after a brightness change");
                // 0.3 in L2 over unit vectors is a loose bound and still far
                // below the ~1.0 that unrelated descriptors sit at.
                Check(compared > 0 && worst < 0.3,
                      "descriptors are illumination invariant (worst " +
                          std::to_string(worst) + ")");
            } else {
                Check(false, "the brightened scene detected");
            }
        }

        // Unrelated descriptors must be FAR apart, or the bound above means
        // nothing -- a detector emitting constant descriptors would pass every
        // invariance test ever written.
        if (da.Count() >= 2) {
            double best = 1e9;
            for (size_t i = 0; i < da.Count() && i < 20; ++i)
                for (size_t j = i + 1; j < da.Count() && j < 20; ++j) {
                    const float dx = a[i].x - a[j].x, dy = a[i].y - a[j].y;
                    if (std::sqrt(dx * dx + dy * dy) < 10.0f) continue;  // same blob
                    best = std::min(best,
                                    double(std::sqrt(DistanceL2Sq(da.FloatAt(i),
                                                                  da.FloatAt(j), 128))));
                }
            // 0.25 rather than a rounder number, and it is a FLOOR on
            // separation rather than a target. What this check exists for is to
            // make the invariance bound above mean something: that measured
            // 0.00001 for the same feature under a brightness change, and the
            // closest DIFFERENT pair here sits around 0.3 -- four orders of
            // magnitude apart. A detector emitting constant descriptors would
            // pass every invariance test ever written and fail this one.
            Check(best > 0.25,
                  "descriptors of different features differ (closest " +
                      std::to_string(best) + ")");
        }

        // A flat image has nothing to find, and must report that rather than
        // inventing features from noise.
        {
            Image flat;
            ImageDesc d{64, 64, Format::RGBA32F};
            flat.Alloc(d);
            ImageView v = flat.MapCpuWrite();
            for (int y = 0; y < 64; ++y)
                for (int x = 0; x < 64; ++x) {
                    float* p = v.At<float>(x, y);
                    p[0] = p[1] = p[2] = 0.5f;
                    p[3] = 1.0f;
                }
            std::vector<Keypoint> k;
            DescriptorSet dd;
            detect(std::move(flat), &k, &dd);
            Check(k.empty(), "a flat image yields no features (" +
                                 std::to_string(k.size()) + ")");
        }
    }

    // --- every detector, through the same checks -------------------------------
    //
    // The interface is the thing being tested here, not any one algorithm: each
    // detector must attach a sidecar, agree with itself about how many
    // descriptors it produced, declare a kind the matcher can act on, and find
    // features that repeat under a translation.
    //
    // Repeatability is the property that matters and the one a count cannot
    // show. Measured on a real photograph rather than the synthetic fixture,
    // because the fixture's texture is the thing that misled me once already.
    {
        // Through TGLAB_SOURCE_DIR rather than a relative path: ctest runs from
        // build/, where "assets/test.png" does not exist. The suite passed
        // standalone and failed under ctest until this used the same absolute
        // path the script-parsing checks already do.
        Image photo;
        std::string perr;
#ifdef TGLAB_SOURCE_DIR
        const std::string photoPath =
            (std::filesystem::path(TGLAB_SOURCE_DIR) / "assets" / "test.png").string();
#else
        const std::string photoPath = "assets/test.png";
#endif
        const bool haveFile = LoadImageFile(photoPath, &photo, &perr);
        Check(haveFile, "loaded a real photograph to detect on" +
              (haveFile ? "" : ": " + perr));

        auto shift = [](Image& src, int dx, int dy) {
            ImageView v = src.MapCpuRead();
            PixelBuffer in;
            in.Unpack(v);
            Image out;
            out.Alloc(v.desc);
            ImageView ov = out.MapCpuWrite();
            PixelBuffer ob;
            ob.Unpack(ov);
            const int w = in.Width(), h = in.Height(), ch = in.Channels();
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x) {
                    const float* p = in.At(std::clamp(x - dx, 0, w - 1),
                                           std::clamp(y - dy, 0, h - 1));
                    float* o = ob.At(x, y);
                    for (int c = 0; c < ch; ++c) o[c] = p[c];
                }
            ob.PackInto(ov);
            return out;
        };

        auto run = [](const char* name, Image&& in, const FeatureSidecar** fs,
                      Pipeline* keep, std::vector<Data>* keepSrc) {
            auto algo = Registry::Get().Create(name);
            if (!algo) return false;
            keepSrc->clear();
            keepSrc->push_back(Data{std::move(in)});
            keep->AddStage(std::move(algo), name, {{-1, 0}}, 1, 1);
            std::string e;
            if (!keep->Execute(keepSrc, nullptr, &e)) return false;
            const Data* d = keep->Resolve({0, 0}, keepSrc);
            const auto* im = d ? std::get_if<Image>(d) : nullptr;
            if (!im) return false;
            *fs = FeaturesOf(*im);
            return *fs != nullptr;
        };

        if (haveFile) {
            // THE CAP MUST NOT TRUNCATE SPATIALLY.
            //
            // Every detector here scans top-to-bottom, so a cap applied by
            // STOPPING the scan keeps everything above some scanline and
            // nothing below it. That is not "fewer features" -- it is a
            // contiguous band of the frame with none at all, and if that band
            // is where the next frame overlaps, the pair cannot be matched.
            //
            // Tim found it on a panorama: the right-hand end failed to align
            // while the feature count sat pinned at max_features. Measured on
            // the pair that failed, before and after: 2% of candidates kept and
            // 0% inliers, against 11% and 64% -- at the SAME cap.
            //
            // Three of the five detectors had it. ORB was the exception, and
            // ranking by response then capping is what it was already doing.
            //
            // The test: cap hard, and require features in the BOTTOM of the
            // frame. A truncating detector puts none there at all.
            for (const char* name : {"detect_sift", "detect_surf", "detect_akaze",
                                     "detect_orb", "detect_brisk"}) {
                auto algo = Registry::Get().Create(name);
                if (!algo) continue;
                if (ParamBase* pb = algo->FindParam("max_features")) {
                    std::string e; pb->SetFromScript(Value(40.0), &e);
                }

                std::vector<Data> s;
                s.push_back(Data{photo.Clone()});
                Pipeline p;
                p.AddStage(std::move(algo), name, {{-1, 0}}, 1, 1);
                std::string e;
                if (!p.Execute(&s, nullptr, &e)) { Check(false, std::string(name) + " ran"); continue; }
                const Data* d = p.Resolve({0, 0}, &s);
                const auto* im = d ? std::get_if<Image>(d) : nullptr;
                const FeatureSidecar* fs = im ? FeaturesOf(*im) : nullptr;
                if (!fs || fs->keypoints.empty()) continue;

                // THE PROPERTY IS "STRONGEST KEPT", not "spread over the frame".
                //
                // Counting features in the lower half was the obvious test and
                // it does NOT work: reintroducing the truncation bug leaves it
                // passing, because assets/test.png yields fewer candidates than
                // the cap in its upper half, so nothing is ever truncated away.
                // A test that passes with the bug present is worse than none.
                //
                // What truncation actually breaks is the RANKING: a scan that
                // stops early keeps whatever it met first, regardless of
                // strength. So compare against the uncapped run -- every
                // capped feature must be among the strongest the detector
                // found, which raster-order truncation cannot satisfy unless
                // the image happens to be sorted.
                auto uncapped = Registry::Get().Create(name);
                if (ParamBase* pb = uncapped->FindParam("max_features")) {
                    std::string e2; pb->SetFromScript(Value(50000.0), &e2);
                }
                std::vector<Data> s2;
                s2.push_back(Data{photo.Clone()});
                Pipeline p2;
                p2.AddStage(std::move(uncapped), name, {{-1, 0}}, 1, 1);
                std::string e2;
                if (!p2.Execute(&s2, nullptr, &e2)) continue;
                const Data* d2 = p2.Resolve({0, 0}, &s2);
                const auto* im2 = d2 ? std::get_if<Image>(d2) : nullptr;
                const FeatureSidecar* all = im2 ? FeaturesOf(*im2) : nullptr;
                if (!all || all->keypoints.size() <= fs->keypoints.size()) continue;

                // The response of the 40th strongest, uncapped.
                std::vector<float> resp;
                resp.reserve(all->keypoints.size());
                for (const Keypoint& k : all->keypoints)
                    resp.push_back(std::abs(k.response));
                const size_t nth = fs->keypoints.size() - 1;
                std::nth_element(resp.begin(), resp.begin() + nth, resp.end(),
                                 std::greater<float>());
                const float cutoff = resp[nth];

                int weak = 0;
                for (const Keypoint& k : fs->keypoints)
                    if (std::abs(k.response) < cutoff * 0.999f) ++weak;

                Check(weak == 0,
                      std::string(name) + ": a hard cap keeps the STRONGEST "
                      "features, not the first ones scanned (" +
                          std::to_string(weak) + " of " +
                          std::to_string(fs->keypoints.size()) +
                          " were weaker than the cutoff)");
            }

            for (const char* name : {"detect_sift", "detect_surf", "detect_akaze",
                                     "detect_orb", "detect_brisk"}) {
                Image a = photo.Clone();
                Image b = shift(photo, 11, -6);

                Pipeline pa, pb;
                std::vector<Data> sa, sb;
                const FeatureSidecar* fa = nullptr;
                const FeatureSidecar* fb = nullptr;

                const bool ok = run(name, std::move(a), &fa, &pa, &sa) &&
                                run(name, std::move(b), &fb, &pb, &sb);
                Check(ok, std::string(name) + " attaches a sidecar");
                if (!ok) continue;

                Check(!fa->keypoints.empty(),
                      std::string(name) + " finds features (" +
                          std::to_string(fa->keypoints.size()) + ")");

                // The two arrays must agree, or a matcher reads one feature's
                // descriptor as another's -- which produces matches rather than
                // an error, and they are wrong.
                Check(fa->descriptors.Count() == fa->keypoints.size(),
                      std::string(name) + ": one descriptor per keypoint (" +
                          std::to_string(fa->descriptors.Count()) + " vs " +
                          std::to_string(fa->keypoints.size()) + ")");

                Check(fa->descriptors.kind != DescriptorKind::None &&
                      fa->descriptors.dim > 0,
                      std::string(name) + " declares a usable descriptor kind");

                Check(!fa->detector.empty(),
                      std::string(name) + " names itself in the sidecar");

                // REPEATABILITY. Counted over features that stayed in frame:
                // a feature shifted off the edge is not a failure.
                int matched = 0, eligible = 0;
                ImageView pv = photo.MapCpuRead();
                const float W = float(pv.desc.width), H = float(pv.desc.height);
                for (const Keypoint& ka : fa->keypoints) {
                    const float sx = ka.x + 11.0f, sy = ka.y - 6.0f;
                    if (sx < 16.0f || sy < 16.0f || sx > W - 16.0f || sy > H - 16.0f)
                        continue;
                    ++eligible;
                    for (const Keypoint& kb : fb->keypoints) {
                        const float dx = (kb.x - 11.0f) - ka.x;
                        const float dy = (kb.y + 6.0f) - ka.y;
                        if (std::sqrt(dx * dx + dy * dy) < 2.0f) { ++matched; break; }
                    }
                }
                // Half is a floor, not a target. Measured on this image: SIFT
                // 70%, SURF 62%, AKAZE 96% -- AKAZE highest because its
                // edge-preserving diffusion is precisely what stops features
                // drifting with scale, which is the reason to have it.
                Check(eligible > 0 && matched * 2 >= eligible,
                      std::string(name) + " features repeat under a shift (" +
                          std::to_string(matched) + "/" + std::to_string(eligible) + ")");
            }
        }
    }

    // --- matching -------------------------------------------------------------
    //
    // The claim to earn is CORRECTNESS, not count. A matcher that returns
    // hundreds of pairs is useless if they point at the wrong places, and a
    // count alone cannot tell the difference.
    //
    // So the fixture is an image and a KNOWN SHIFT of itself: every correct
    // match must have the two keypoints separated by exactly that shift, which
    // makes each individual pair checkable rather than only the total.
    {
        Image photo;
        std::string perr;
#ifdef TGLAB_SOURCE_DIR
        const std::string pp =
            (std::filesystem::path(TGLAB_SOURCE_DIR) / "assets" / "test.png").string();
#else
        const std::string pp = "assets/test.png";
#endif
        if (LoadImageFile(pp, &photo, &perr)) {
            constexpr int kDx = 11, kDy = -6;

            auto shifted = [&](Image& src) {
                ImageView v = src.MapCpuRead();
                PixelBuffer in;
                in.Unpack(v);
                Image out;
                out.Alloc(v.desc);
                ImageView ov = out.MapCpuWrite();
                PixelBuffer ob;
                ob.Unpack(ov);
                const int w = in.Width(), h = in.Height(), ch = in.Channels();
                for (int y = 0; y < h; ++y)
                    for (int x = 0; x < w; ++x) {
                        const float* p = in.At(std::clamp(x - kDx, 0, w - 1),
                                               std::clamp(y - kDy, 0, h - 1));
                        float* o = ob.At(x, y);
                        for (int c = 0; c < ch; ++c) o[c] = p[c];
                    }
                ob.PackInto(ov);
                return out;
            };

            // detect on both frames, then match the group.
            auto detectAndMatch = [&](const char* det, const char* matcher,
                                      double ratio, bool cross,
                                      const FeatureSidecar** refOut,
                                      const FeatureSidecar** othOut,
                                      const MatchSidecar** msOut,
                                      Pipeline* keep, std::vector<Data>* keepSrc,
                                      std::string* err) {
                ImageSet set;
                set.images.push_back(photo.Clone());
                set.images.push_back(shifted(photo));
                set.shape = Shape{{{"frame", 2}}};

                keepSrc->clear();
                keepSrc->push_back(Data{std::move(set)});

                auto d = Registry::Get().Create(det);
                auto m = Registry::Get().Create(matcher);
                if (!d || !m) return false;
                if (ParamBase* p = m->FindParam("ratio")) {
                    std::string e; p->SetFromScript(Value(ratio), &e);
                }
                if (ParamBase* p = m->FindParam("cross_check")) {
                    std::string e; p->SetFromScript(Value(cross ? 1.0 : 0.0), &e);
                }

                keep->AddStage(std::move(d), det, {{-1, 0}}, 1, 1);
                keep->AddStage(std::move(m), matcher, {{0, 0}}, 1, 2);
                if (!keep->Execute(keepSrc, nullptr, err)) return false;

                const Data* out = keep->Resolve({1, 0}, keepSrc);
                const auto* os = out ? std::get_if<ImageSet>(out) : nullptr;
                if (!os || os->images.size() != 2) return false;

                *refOut = FeaturesOf(os->images[0]);
                *othOut = FeaturesOf(os->images[1]);
                *msOut  = MatchesOf(os->images[1]);
                return *refOut && *othOut && *msOut;
            };

            const FeatureSidecar *fa = nullptr, *fb = nullptr;
            const MatchSidecar* ms = nullptr;
            Pipeline p;
            std::vector<Data> s;
            std::string merr;

            const bool ok = detectAndMatch("detect_sift", "match_brute", 0.8, true,
                                           &fa, &fb, &ms, &p, &s, &merr);
            Check(ok, "detect then match_brute runs" + (ok ? "" : ": " + merr));

            if (ok) {
                Check(!ms->Matches().empty(),
                      "it finds matches (" + std::to_string(ms->Matches().size()) + ")");
                Check(ms->Reference() == 0, "the match set names its reference");
                Check(ms->considered >= int(ms->Matches().size()),
                      "candidates considered is at least matches kept");

                // EVERY match must be geometrically right. The two frames
                // differ by a known shift, so a correct pair has its keypoints
                // separated by exactly that -- which makes this a test of the
                // matches rather than of how many there are.
                int correct = 0;
                for (const Match& m : ms->Matches()) {
                    if (m.a < 0 || m.a >= int(fa->keypoints.size())) continue;
                    if (m.b < 0 || m.b >= int(fb->keypoints.size())) continue;
                    const Keypoint& ka = fa->keypoints[size_t(m.a)];
                    const Keypoint& kb = fb->keypoints[size_t(m.b)];
                    const float ex = (kb.x - float(kDx)) - ka.x;
                    const float ey = (kb.y - float(kDy)) - ka.y;
                    if (std::sqrt(ex * ex + ey * ey) < 3.0f) ++correct;
                }
                // 90% is a high bar and the right one: with the ratio test and
                // cross check both on, a matcher that is working should have
                // very few wrong pairs. Measured here at 99%.
                Check(correct * 10 >= int(ms->Matches().size()) * 9,
                      "matches are geometrically correct (" +
                          std::to_string(correct) + " of " +
                          std::to_string(ms->Matches().size()) + ")");

                // Every match's ratio must satisfy the test that produced it.
                bool ratiosOk = true;
                for (const Match& m : ms->Matches())
                    if (m.ratio > 0.8f + 1e-4f) ratiosOk = false;
                Check(ratiosOk, "every kept match passes the ratio threshold");
            }

            // THE RATIO TEST HAS TO BE DOING SOMETHING. Disabling it must let
            // more candidates through AND make them worse -- otherwise the
            // parameter is decoration and the default is untested.
            {
                const FeatureSidecar *ga = nullptr, *gb = nullptr;
                const MatchSidecar* gms = nullptr;
                Pipeline p2;
                std::vector<Data> s2;
                std::string e2;
                if (detectAndMatch("detect_sift", "match_brute", 1.0, false,
                                   &ga, &gb, &gms, &p2, &s2, &e2) && ok) {
                    Check(gms->Matches().size() > ms->Matches().size(),
                          "disabling the ratio test keeps more candidates (" +
                              std::to_string(gms->Matches().size()) + " vs " +
                              std::to_string(ms->Matches().size()) + ")");

                    int wrong = 0;
                    for (const Match& m : gms->Matches()) {
                        if (m.a < 0 || m.a >= int(ga->keypoints.size())) continue;
                        if (m.b < 0 || m.b >= int(gb->keypoints.size())) continue;
                        const Keypoint& ka = ga->keypoints[size_t(m.a)];
                        const Keypoint& kb = gb->keypoints[size_t(m.b)];
                        const float ex = (kb.x - float(kDx)) - ka.x;
                        const float ey = (kb.y - float(kDy)) - ka.y;
                        if (std::sqrt(ex * ex + ey * ey) >= 3.0f) ++wrong;
                    }
                    // The point of the test: without it, a real fraction of the
                    // extra matches are wrong. If this were zero the ratio test
                    // would be discarding good matches for nothing.
                    Check(wrong > 0,
                          "and those extra matches include wrong ones (" +
                              std::to_string(wrong) + ")");
                }
            }

            // A BINARY descriptor goes through the same matcher, dispatched on
            // the kind rather than on the detector's name. Hamming rather than
            // L2, and getting that wrong returns matches that mean nothing --
            // which is the whole reason DescriptorKind travels with the data.
            {
                const FeatureSidecar *aa = nullptr, *ab = nullptr;
                const MatchSidecar* ams = nullptr;
                Pipeline p3;
                std::vector<Data> s3;
                std::string e3;
                if (detectAndMatch("detect_akaze", "match_brute", 0.8, true,
                                   &aa, &ab, &ams, &p3, &s3, &e3)) {
                    Check(aa->descriptors.kind == DescriptorKind::Binary,
                          "the AKAZE run really is binary");
                    Check(!ams->Matches().empty(),
                          "binary descriptors match too (" +
                              std::to_string(ams->Matches().size()) + ")");

                    int correct = 0;
                    for (const Match& m : ams->Matches()) {
                        if (m.a < 0 || m.a >= int(aa->keypoints.size())) continue;
                        if (m.b < 0 || m.b >= int(ab->keypoints.size())) continue;
                        const Keypoint& ka = aa->keypoints[size_t(m.a)];
                        const Keypoint& kb = ab->keypoints[size_t(m.b)];
                        const float ex = (kb.x - float(kDx)) - ka.x;
                        const float ey = (kb.y - float(kDy)) - ka.y;
                        if (std::sqrt(ex * ex + ey * ey) < 3.0f) ++correct;
                    }
                    Check(correct * 10 >= int(ams->Matches().size()) * 9,
                          "binary matches are geometrically correct (" +
                              std::to_string(correct) + " of " +
                              std::to_string(ams->Matches().size()) + ")");
                }
            }
        }
    }

    // --- approximate matching -------------------------------------------------
    //
    // The claim an approximate matcher has to earn is RECALL: what fraction of
    // the exact answer it reproduces. That is only measurable against an exact
    // answer, which is why match_brute was built first.
    //
    // Precision is NOT at risk here and it is worth being clear why: the
    // approximation is in which candidates are considered, not in how they are
    // judged, so a match match_ann returns passed the same ratio test. What
    // approximation costs is matches missed entirely.
    {
        Image photo;
        std::string perr;
#ifdef TGLAB_SOURCE_DIR
        const std::string ap =
            (std::filesystem::path(TGLAB_SOURCE_DIR) / "assets" / "test.png").string();
#else
        const std::string ap = "assets/test.png";
#endif
        if (LoadImageFile(ap, &photo, &perr)) {
            constexpr int kDx = 11, kDy = -6;

            auto shifted = [&](Image& src) {
                ImageView v = src.MapCpuRead();
                PixelBuffer in;
                in.Unpack(v);
                Image out;
                out.Alloc(v.desc);
                ImageView ov = out.MapCpuWrite();
                PixelBuffer ob;
                ob.Unpack(ov);
                const int w = in.Width(), h = in.Height(), ch = in.Channels();
                for (int y = 0; y < h; ++y)
                    for (int x = 0; x < w; ++x) {
                        const float* p = in.At(std::clamp(x - kDx, 0, w - 1),
                                               std::clamp(y - kDy, 0, h - 1));
                        float* o = ob.At(x, y);
                        for (int c = 0; c < ch; ++c) o[c] = p[c];
                    }
                ob.PackInto(ov);
                return out;
            };

            // Runs a detector then a named matcher over a two-frame group.
            auto pipeline = [&](const char* det, const char* matcher, int checks,
                                const FeatureSidecar** fa, const FeatureSidecar** fb,
                                const MatchSidecar** ms, Pipeline* keep,
                                std::vector<Data>* keepSrc, std::string* err) {
                ImageSet set;
                set.images.push_back(photo.Clone());
                set.images.push_back(shifted(photo));
                set.shape = Shape{{{"frame", 2}}};
                keepSrc->clear();
                keepSrc->push_back(Data{std::move(set)});

                auto d = Registry::Get().Create(det);
                auto m = Registry::Get().Create(matcher);
                if (!d || !m) return false;
                if (checks > 0)
                    if (ParamBase* p = m->FindParam("checks")) {
                        std::string e; p->SetFromScript(Value(double(checks)), &e);
                    }

                keep->AddStage(std::move(d), det, {{-1, 0}}, 1, 1);
                keep->AddStage(std::move(m), matcher, {{0, 0}}, 1, 2);
                if (!keep->Execute(keepSrc, nullptr, err)) return false;

                const Data* out = keep->Resolve({1, 0}, keepSrc);
                const auto* os = out ? std::get_if<ImageSet>(out) : nullptr;
                if (!os || os->images.size() != 2) return false;
                *fa = FeaturesOf(os->images[0]);
                *fb = FeaturesOf(os->images[1]);
                *ms = MatchesOf(os->images[1]);
                return *fa && *fb && *ms;
            };

            // Fraction of `exact`'s pairs that `approx` also found. Compared as
            // PAIRS rather than counts: two matchers can return the same number
            // of matches and disagree about every one.
            auto recall = [](const MatchSidecar& exact, const MatchSidecar& approx) {
                if (exact.Matches().empty()) return 1.0;
                int found = 0;
                for (const Match& e : exact.Matches())
                    for (const Match& a : approx.Matches())
                        if (a.a == e.a && a.b == e.b) { ++found; break; }
                return double(found) / double(exact.Matches().size());
            };

            // FLOAT: the k-d forest.
            {
                const FeatureSidecar *ea = nullptr, *eb = nullptr;
                const MatchSidecar* ems = nullptr;
                Pipeline pe; std::vector<Data> se; std::string e1;
                const bool okE = pipeline("detect_sift", "match_brute", 0,
                                          &ea, &eb, &ems, &pe, &se, &e1);
                Check(okE, "exact matching ran for the comparison" +
                      (okE ? "" : ": " + e1));

                const FeatureSidecar *aa = nullptr, *ab = nullptr;
                const MatchSidecar* ams = nullptr;
                Pipeline pa; std::vector<Data> sa; std::string e2;
                const bool okA = pipeline("detect_sift", "match_ann", 512,
                                          &aa, &ab, &ams, &pa, &sa, &e2);
                Check(okA, "match_ann runs on float descriptors" +
                      (okA ? "" : ": " + e2));

                if (okE && okA) {
                    const double r = recall(*ems, *ams);
                    Check(r >= 0.7,
                          "the k-d forest recovers most exact matches (" +
                              std::to_string(int(r * 100)) + "%)");

                    // Its matches must be as CORRECT as brute force's -- the
                    // approximation costs recall, not precision, and if this
                    // failed the ratio test would be being applied wrongly.
                    int correct = 0;
                    for (const Match& m : ams->Matches()) {
                        if (m.a < 0 || m.a >= int(aa->keypoints.size())) continue;
                        if (m.b < 0 || m.b >= int(ab->keypoints.size())) continue;
                        const Keypoint& ka = aa->keypoints[size_t(m.a)];
                        const Keypoint& kb = ab->keypoints[size_t(m.b)];
                        const float ex = (kb.x - float(kDx)) - ka.x;
                        const float ey = (kb.y - float(kDy)) - ka.y;
                        if (std::sqrt(ex * ex + ey * ey) < 3.0f) ++correct;
                    }
                    Check(ams->Matches().empty() ||
                          correct * 10 >= int(ams->Matches().size()) * 8,
                          "and they are geometrically correct (" +
                              std::to_string(correct) + " of " +
                              std::to_string(ams->Matches().size()) + ")");
                }

                // MORE CHECKS MUST FIND MORE. If recall did not climb with the
                // budget, `checks` would be decoration and the whole
                // speed/accuracy trade would be unavailable.
                const FeatureSidecar *la = nullptr, *lb = nullptr;
                const MatchSidecar* lms = nullptr;
                Pipeline pl; std::vector<Data> sl; std::string e3;
                if (okE && pipeline("detect_sift", "match_ann", 8,
                                    &la, &lb, &lms, &pl, &sl, &e3)) {
                    const double low  = recall(*ems, *lms);
                    const double high = okA ? recall(*ems, *ams) : 0.0;
                    Check(high >= low,
                          "raising checks does not lower recall (" +
                              std::to_string(int(low * 100)) + "% -> " +
                              std::to_string(int(high * 100)) + "%)");
                }
            }

            // BINARY: LSH. A k-d tree cannot be built over Hamming space at
            // all, so this is a genuinely different path rather than the same
            // code with a different distance.
            {
                const FeatureSidecar *ea = nullptr, *eb = nullptr;
                const MatchSidecar* ems = nullptr;
                Pipeline pe; std::vector<Data> se; std::string e1;
                const bool okE = pipeline("detect_akaze", "match_brute", 0,
                                          &ea, &eb, &ems, &pe, &se, &e1);

                const FeatureSidecar *aa = nullptr, *ab = nullptr;
                const MatchSidecar* ams = nullptr;
                Pipeline pa; std::vector<Data> sa; std::string e2;
                const bool okA = pipeline("detect_akaze", "match_ann", 512,
                                          &aa, &ab, &ams, &pa, &sa, &e2);
                Check(okA, "match_ann runs on binary descriptors" +
                      (okA ? "" : ": " + e2));

                if (okE && okA) {
                    Check(!ams->Matches().empty(),
                          "LSH finds matches (" +
                              std::to_string(ams->Matches().size()) + ")");

                    int correct = 0;
                    for (const Match& m : ams->Matches()) {
                        if (m.a < 0 || m.a >= int(aa->keypoints.size())) continue;
                        if (m.b < 0 || m.b >= int(ab->keypoints.size())) continue;
                        const Keypoint& ka = aa->keypoints[size_t(m.a)];
                        const Keypoint& kb = ab->keypoints[size_t(m.b)];
                        const float ex = (kb.x - float(kDx)) - ka.x;
                        const float ey = (kb.y - float(kDy)) - ka.y;
                        if (std::sqrt(ex * ex + ey * ey) < 3.0f) ++correct;
                    }
                    Check(correct * 10 >= int(ams->Matches().size()) * 8,
                          "LSH matches are geometrically correct (" +
                              std::to_string(correct) + " of " +
                              std::to_string(ams->Matches().size()) + ")");
                }
            }
        }
    }

    // --- feature alignment ----------------------------------------------------
    //
    // The claim: recover a transform from matches, accurately enough to warp
    // by. So the fixture applies a KNOWN transform and the test asks how close
    // the recovered one is -- against ground truth rather than against the
    // eye, and measured where it matters, at the corners.
    {
        Image photo;
        std::string perr;
#ifdef TGLAB_SOURCE_DIR
        const std::string lp =
            (std::filesystem::path(TGLAB_SOURCE_DIR) / "assets" / "test.png").string();
#else
        const std::string lp = "assets/test.png";
#endif
        if (LoadImageFile(lp, &photo, &perr)) {
            // Warps `src` by the INVERSE of `t`, so that the aligner solving
            // for t is solving for the transform the fixture applied.
            auto warpBy = [&](Image& src, const Affine& t) {
                ImageView v = src.MapCpuRead();
                PixelBuffer in;
                in.Unpack(v);
                Image out;
                out.Alloc(v.desc);
                ImageView ov = out.MapCpuWrite();
                PixelBuffer ob;
                ob.Unpack(ov);
                const int w = in.Width(), h = in.Height(), ch = in.Channels();

                bool ok = false;
                const Affine inv = t.Inverse(&ok);
                for (int y = 0; y < h; ++y)
                    for (int x = 0; x < w; ++x) {
                        float sx, sy;
                        inv.MapPoint(float(x), float(y), &sx, &sy);
                        float sm[4] = {0, 0, 0, 0};
                        SampleBilinear(in, sx, sy, sm);
                        float* o = ob.At(x, y);
                        for (int c = 0; c < ch; ++c) o[c] = sm[c];
                    }
                ob.PackInto(ov);
                return out;
            };

            // The detector is a parameter of the fixture because the two differ
            // sharply under rotation, which is worth recording rather than
            // working around: measured on this fixture, SIFT keeps 89% of its
            // matches as inliers under a 3-degree rotation while AKAZE keeps
            // 14-20%. AKAZE's edge-preserving scale space makes it the more
            // repeatable detector under translation and its orientation
            // estimate is the weaker of the two.
            auto solve = [&](const Affine& truth, int model, const char* det,
                             Affine* got, int* inliers, std::string* err) {
                ImageSet set;
                set.images.push_back(photo.Clone());
                set.images.push_back(warpBy(photo, truth));
                set.shape = Shape{{{"frame", 2}}};

                std::vector<Data> s;
                s.push_back(Data{std::move(set)});

                Pipeline p;
                // A LOW threshold on purpose: a homography has eight
                // parameters, and fitting eight from twenty noisy matches is
                // the textbook overfitting case rather than a test of the
                // solver. With the stock threshold this fixture yielded 19
                // matches and the corner error ran 1.0 px for the 4-parameter
                // similarity, 2.8 for affine and 60 for the homography --
                // exactly the curve too few points per parameter produces.
                p.AddStage(Registry::Get().Create(det), det, {{-1, 0}}, 1, 1);
                p.AddStage(Registry::Get().Create("match_brute"), "match_brute",
                           {{0, 0}}, 1, 2);
                auto a = Registry::Get().Create("align_features");
                if (ParamBase* pb = a->FindParam("model")) {
                    std::string e; pb->SetFromScript(Value(double(model)), &e);
                }
                p.AddStage(std::move(a), "align_features", {{1, 0}}, 1, 3);

                if (!p.Execute(&s, nullptr, err)) return false;
                const Data* out = p.Resolve({2, 0}, &s);
                const auto* os = out ? std::get_if<ImageSet>(out) : nullptr;
                if (!os || os->images.size() != 2) return false;
                *got = TransformOf(os->images[1]);
                const MatchSidecar* ms = MatchesOf(os->images[1]);
                *inliers = ms ? int(ms->Matches().size()) : 0;
                if (p.Stages().size() > 2 && p.Stages()[2].algo)
                    std::printf("        [dbg] %s\n",
                                p.Stages()[2].algo->RunReport().c_str());
                return true;
            };

            // Corner error against ground truth: the number that says whether
            // the transform is usable. An error in the linear part shows at the
            // corners and barely at the centre, so a centre-only check would
            // pass a visibly wrong warp.
            auto cornerError = [&](const Affine& a, const Affine& b) {
                ImageView v = photo.MapCpuRead();
                const float w = float(v.desc.width), h = float(v.desc.height);
                const float cx[4] = {0, w, 0, w};
                const float cy[4] = {0, 0, h, h};
                float worst = 0.0f;
                for (int i = 0; i < 4; ++i) {
                    float ax, ay, bx, by;
                    a.MapPoint(cx[i], cy[i], &ax, &ay);
                    b.MapPoint(cx[i], cy[i], &bx, &by);
                    const float dx = ax - bx, dy = ay - by;
                    worst = std::max(worst, std::sqrt(dx * dx + dy * dy));
                }
                return worst;
            };

            // TRANSLATION, recovered by every model. The easiest case, and the
            // one that would catch a sign error or a transposed matrix.
            {
                Affine truth;
                truth.m[2] = 9.0f;
                truth.m[5] = -5.0f;

                for (int model = 0; model <= 2; ++model) {
                    Affine got;
                    int inliers = 0;
                    std::string e;
                    const char* names[] = {"similarity", "affine", "homography"};
                    if (solve(truth, model, "detect_akaze", &got, &inliers, &e)) {
                        const float err = cornerError(truth, got);
                        Check(err < 1.0f,
                              std::string(names[model]) +
                                  " recovers a translation (corner error " +
                                  std::to_string(err) + " px)");
                    } else {
                        Check(false, std::string(names[model]) + " solved: " + e);
                    }
                }
            }

            // ROTATION AND SCALE. A similarity, so all three models can express
            // it -- and each should, since a model with spare parameters must
            // not use them to fit noise.
            {
                Affine truth;
                const float a = 0.05f;    // ~3 degrees
                const float s = 1.03f;
                truth.m[0] =  s * std::cos(a);
                truth.m[1] = -s * std::sin(a);
                truth.m[3] =  s * std::sin(a);
                truth.m[4] =  s * std::cos(a);
                // About the centre rather than the origin, or the corners move
                // far enough that most of the frame leaves the image.
                ImageView v = photo.MapCpuRead();
                const float cx = float(v.desc.width) * 0.5f;
                const float cy = float(v.desc.height) * 0.5f;
                truth.m[2] = cx - (truth.m[0] * cx + truth.m[1] * cy);
                truth.m[5] = cy - (truth.m[3] * cx + truth.m[4] * cy);

                for (int model = 0; model <= 2; ++model) {
                    Affine got;
                    int inliers = 0;
                    std::string e;
                    const char* names[] = {"similarity", "affine", "homography"};
                    if (solve(truth, model, "detect_sift", &got, &inliers, &e)) {
                        const float err = cornerError(truth, got);
                        // The tolerance widens with the model, and that is the
                        // behaviour rather than a concession: this warp IS a
                        // similarity, so the affine's two extra parameters and
                        // the homography's four have nothing real to fit and
                        // spend themselves on noise. Measured here: 0.66 px for
                        // 4 parameters, 1.24 for 6, 2.67 for 8, from the same
                        // 37 matches. That is why `model` is a parameter -- a
                        // homography is not a free upgrade.
                        const float tol = (model == 0) ? 1.5f : (model == 1) ? 2.0f : 3.5f;
                        Check(err < tol,
                              std::string(names[model]) +
                                  " recovers a rotation and scale (corner error " +
                                  std::to_string(err) + " px, " +
                                  std::to_string(inliers) + " matches)");
                    } else {
                        Check(false, std::string(names[model]) + " solved: " + e);
                    }
                }
            }

            // PERSPECTIVE. Only the homography can express this, and that is
            // the point of having it: an affine fit leaves a systematic error
            // that grows toward the edges -- exactly where a panorama shows it.
            {
                Affine truth;
                ImageView v = photo.MapCpuRead();
                const float w = float(v.desc.width);
                truth.m[6] = 0.00008f;   // a gentle keystone: ~4% compression at the far edge
                truth.m[2] = 0.0f;
                truth.m[5] = 0.0f;
                // Keep the centre roughly put, so the warp stays in frame.
                const float scale = 1.0f + truth.m[6] * w * 0.5f;
                truth.m[0] = scale;
                truth.m[4] = scale;

                Affine gotH, gotA;
                int inH = 0, inA = 0;
                std::string e1, e2;
                const bool okH = solve(truth, 2, "detect_sift", &gotH, &inH, &e1);
                const bool okA = solve(truth, 1, "detect_sift", &gotA, &inA, &e2);

                if (okH) {
                    const float err = cornerError(truth, gotH);
                    Check(err < 3.0f,
                          "homography recovers a perspective warp (corner error " +
                              std::to_string(err) + " px)");
                    Check(!gotH.IsAffine(),
                          "and the result really carries perspective terms");
                } else {
                    Check(false, "homography solved a perspective warp: " + e1);
                }

                // The affine fit must be measurably WORSE, or the homography's
                // two extra parameters are buying nothing and the test above
                // proves nothing either.
                if (okH && okA) {
                    const float eH = cornerError(truth, gotH);
                    const float eA = cornerError(truth, gotA);
                    Check(eA > eH,
                          "and beats the affine fit on it (" +
                              std::to_string(eA) + " vs " + std::to_string(eH) + " px)");
                }
            }

            // CHAINED MATCHING, and the composition that has to follow it.
            //
            // This is the property a panorama depends on: with `chain` on, each
            // frame matches its PREDECESSOR, and the aligner must then compose
            // the links so every transform is relative to frame 0. Getting the
            // composition wrong -- or omitting it -- produces transforms that
            // are each individually correct against a neighbour, which looks
            // like a stitch whose every seam is fine and whose whole is wrong.
            //
            // Three frames, each shifted a further 7 px right and 3 down. Frame
            // 2's transform must come back as the SUM, 14 and 6, even though it
            // was only ever matched against frame 1.
            {
                Affine step;
                step.m[2] = 7.0f;
                step.m[5] = 3.0f;

                Affine twice;
                twice.m[2] = 14.0f;
                twice.m[5] = 6.0f;

                ImageSet set;
                set.images.push_back(photo.Clone());
                set.images.push_back(warpBy(photo, step));
                set.images.push_back(warpBy(photo, twice));
                set.shape = Shape{{{"frame", 3}}};

                std::vector<Data> s;
                s.push_back(Data{std::move(set)});

                Pipeline p;
                p.AddStage(Registry::Get().Create("detect_sift"), "detect_sift",
                           {{-1, 0}}, 1, 1);
                auto m = Registry::Get().Create("match_brute");
                if (ParamBase* pb = m->FindParam("chain")) {
                    std::string e; pb->SetFromScript(Value(1.0), &e);
                }
                // A LOOSE ratio, for a reason specific to this fixture rather
                // than to chaining. Every frame here is a resampled copy of one
                // photo, so frame 2 has been through bilinear interpolation
                // twice and its descriptors are measurably softer than frame
                // 1's. At the stock 0.8 the 1->2 pair kept 16 of 360 candidates
                // and the solve failed for want of points -- which the chain
                // then correctly refused to compose past, reporting "CHAIN
                // BROKE at frame 2".
                //
                // That is the fixture being harder than real frames, not the
                // matcher being wrong: consecutive frames of an actual pan are
                // each sampled once, from the sensor, and hold 40-94% inliers
                // on hundreds of matches.
                if (ParamBase* pb = m->FindParam("ratio")) {
                    std::string e; pb->SetFromScript(Value(0.95), &e);
                }
                p.AddStage(std::move(m), "match_brute", {{0, 0}}, 1, 2);
                auto a = Registry::Get().Create("align_features");
                if (ParamBase* pb = a->FindParam("model")) {
                    std::string e; pb->SetFromScript(Value(0.0), &e);  // similarity
                }
                p.AddStage(std::move(a), "align_features", {{1, 0}}, 1, 3);

                std::string e;
                const bool ran = p.Execute(&s, nullptr, &e);
                Check(ran, "a chained detect -> match -> align runs" +
                           std::string(ran ? "" : ": " + e));
                if (ran) {
                    for (size_t st = 1; st < p.Stages().size(); ++st)
                        if (p.Stages()[st].algo)
                            std::printf("        [dbg] %s\n",
                                        p.Stages()[st].algo->RunReport().c_str());
                    const Data* out = p.Resolve({2, 0}, &s);
                    const auto* os = out ? std::get_if<ImageSet>(out) : nullptr;
                    if (os && os->images.size() == 3) {
                        // Frame 1 is one step; frame 2 must be TWO, which it
                        // can only be if the links were composed.
                        const float e1 = cornerError(TransformOf(os->images[1]), step);
                        const float e2 = cornerError(TransformOf(os->images[2]), twice);
                        Check(e1 < 1.0f,
                              "chain: frame 1 matches its own step (" +
                                  std::to_string(e1) + " px)");
                        Check(e2 < 1.5f,
                              "chain: frame 2 composed to twice the step (" +
                                  std::to_string(e2) + " px)");

                        // ...and the matcher really did pair 2 with 1, not with
                        // 0. Without this the check above would also pass for a
                        // matcher that ignored `chain` entirely, since frame 2
                        // IS twice the step from frame 0.
                        const MatchSidecar* ms = MatchesOf(os->images[2]);
                        Check(ms && ms->Reference() == 1,
                              "chain: frame 2 was matched against frame 1");
                    } else {
                        Check(false, "the chained group came back intact");
                    }
                }
            }

            // STITCHING, end to end on the same three chained frames.
            //
            // The properties worth asserting are geometric rather than
            // pictorial: the canvas has to GROW to hold frames that no longer
            // overlap, it has to stay near the source height for a horizontal
            // pan, and the frames have to agree where they overlap. The last is
            // the one that catches a stitcher that runs and produces nonsense.
            {
                // A ROTATION, not a translation, and the distinction is the
                // whole reason the curved projections exist.
                //
                // The obvious fixture -- shift each frame 7 px sideways -- is
                // the wrong question to ask them. A pure translation is not a
                // rotation about the camera centre, so Orthonormalise
                // correctly DISCARDS it, and all three frames land on top of
                // each other: measured, the canvas came back 498 px from a
                // 512 px source (smaller than one frame) with the overlap
                // disagreeing by 9.6%, where the plane projection managed
                // 526 px and 0.6%. That is not the curved path failing, it is
                // the curved path declining to invent a rotation that did not
                // happen.
                //
                // So the fixture pans the camera instead: H = K R K^-1 for a
                // rotation of `deg` about the vertical axis, which is exactly
                // what a tripod head produces and what the projections are
                // built to undo.
                const float kFocal = 600.0f;
                auto panBy = [&](float deg) {
                    ImageView v = photo.MapCpuRead();
                    const float cx = 0.5f * float(v.desc.width);
                    const float cy = 0.5f * float(v.desc.height);
                    const float a  = deg * 3.14159265f / 180.0f;
                    const float c = std::cos(a), s = std::sin(a);

                    // R about the vertical axis.
                    const float R[9] = {   c, 0.0f,    s,
                                        0.0f, 1.0f, 0.0f,
                                          -s, 0.0f,    c };

                    // K R, then (K R) K^-1, each written out. Plainly rather
                    // than through a general 3x3 multiply, because the point of
                    // this fixture is that the transform really IS of the form
                    // the stitcher assumes -- building it via a helper the
                    // stitcher also uses would test the two against each other
                    // rather than against the mathematics.
                    float M[9];
                    for (int c2 = 0; c2 < 3; ++c2) {
                        M[0 * 3 + c2] = kFocal * R[0 * 3 + c2] + cx * R[2 * 3 + c2];
                        M[1 * 3 + c2] = kFocal * R[1 * 3 + c2] + cy * R[2 * 3 + c2];
                        M[2 * 3 + c2] =                               R[2 * 3 + c2];
                    }
                    // K^-1 = [[1/f, 0, -cx/f], [0, 1/f, -cy/f], [0, 0, 1]],
                    // applied on the right.
                    float N[9];
                    for (int r = 0; r < 3; ++r) {
                        N[r * 3 + 0] = M[r * 3 + 0] / kFocal;
                        N[r * 3 + 1] = M[r * 3 + 1] / kFocal;
                        N[r * 3 + 2] = M[r * 3 + 2]
                                     - (cx / kFocal) * M[r * 3 + 0]
                                     - (cy / kFocal) * M[r * 3 + 1];
                    }
                    return Affine::From3x3(N);
                };

                auto stitch = [&](int projection, int* w, int* h,
                                  float* disagree, std::string* err) {
                    ImageSet set;
                    set.images.push_back(photo.Clone());
                    set.images.push_back(warpBy(photo, panBy(4.0f)));
                    set.images.push_back(warpBy(photo, panBy(8.0f)));
                    set.shape = Shape{{{"frame", 3}}};

                    std::vector<Data> s;
                    s.push_back(Data{std::move(set)});

                    Pipeline p;
                    p.AddStage(Registry::Get().Create("detect_sift"), "detect_sift",
                               {{-1, 0}}, 1, 1);
                    auto m = Registry::Get().Create("match_brute");
                    if (ParamBase* pb = m->FindParam("chain")) {
                        std::string e; pb->SetFromScript(Value(1.0), &e);
                    }
                    if (ParamBase* pb = m->FindParam("ratio")) {
                        std::string e; pb->SetFromScript(Value(0.95), &e);
                    }
                    p.AddStage(std::move(m), "match_brute", {{0, 0}}, 1, 2);
                    auto a = Registry::Get().Create("align_features");
                    if (ParamBase* pb = a->FindParam("model")) {
                        std::string e; pb->SetFromScript(Value(0.0), &e);
                    }
                    p.AddStage(std::move(a), "align_features", {{1, 0}}, 1, 3);
                    auto st = Registry::Get().Create("stitch_panorama");
                    if (ParamBase* pb = st->FindParam("projection")) {
                        std::string e;
                        pb->SetFromScript(Value(double(projection)), &e);
                    }
                    // Pinned to the focal length the fixture actually used, so
                    // this measures the PROJECTIONS rather than the estimator.
                    // The estimator gets its own check below.
                    if (ParamBase* pb = st->FindParam("focal")) {
                        std::string e;
                        pb->SetFromScript(Value(double(kFocal)), &e);
                    }
                    p.AddStage(std::move(st), "stitch_panorama", {{2, 0}}, 1, 0);

                    if (!p.Execute(&s, nullptr, err)) return false;
                    if (p.Stages().size() > 3 && p.Stages()[3].algo)
                        std::printf("        [dbg] %s\n",
                                    p.Stages()[3].algo->RunReport().c_str());
                    const Data* out = p.Resolve({3, 0}, &s);
                    const auto* im = out ? std::get_if<Image>(out) : nullptr;
                    if (!im) return false;
                    *w = im->Desc().width;
                    *h = im->Desc().height;

                    // Disagreement is read back off the report rather than
                    // recomputed: the algorithm already measures it where the
                    // frames overlap, and a second implementation here could
                    // disagree with the one that matters.
                    const std::string rep = p.Stages()[3].algo->RunReport();
                    const size_t k = rep.find("disagreeing ");
                    *disagree = (k == std::string::npos)
                        ? -1.0f : float(atof(rep.c_str() + k + 12));
                    return true;
                };

                ImageView pv = photo.MapCpuRead();
                const int srcW = pv.desc.width, srcH = pv.desc.height;

                for (int proj = 0; proj <= 2; ++proj) {
                    const char* names[3] = {"plane", "cylindrical", "spherical"};
                    int w = 0, h = 0;
                    float dis = -1.0f;
                    std::string e;
                    const bool ok = stitch(proj, &w, &h, &dis, &e);
                    Check(ok, std::string("stitch_panorama runs (") + names[proj] +
                              ")" + (ok ? "" : ": " + e));
                    if (!ok) continue;

                    // Wider than one frame, because three frames 7 px apart
                    // cover more than one frame's worth. Not much wider -- a
                    // canvas that exploded would be the divergence this is
                    // built to avoid.
                    Check(w > srcW && w < srcW * 2,
                          std::string(names[proj]) + ": the canvas grew to hold the pan (" +
                              std::to_string(w) + " from " + std::to_string(srcW) + ")");

                    // A horizontal pan must not grow the height appreciably.
                    // This is the check that fails when a projection is wrong:
                    // an over-bent surface pushes the corners up and down.
                    Check(h < srcH * 3 / 2,
                          std::string(names[proj]) + ": the height stayed near the source (" +
                              std::to_string(h) + " from " + std::to_string(srcH) + ")");

                    // And the frames agree where they overlap. Loose, because
                    // this fixture resamples the same photo twice and the
                    // interpolation alone costs a few percent -- what it
                    // catches is a stitch that placed a frame in the wrong
                    // place entirely, which runs to tens of percent.
                    Check(dis >= 0.0f && dis < 12.0f,
                          std::string(names[proj]) + ": the frames agree where they overlap (" +
                              std::to_string(dis) + "%)");
                }

                // THE FOCAL ESTIMATOR, on a fixture whose focal length is
                // known. This is the one number the stitcher infers rather than
                // being told, and a wrong one bends every ray: too small and
                // the seams bow outward, too large and they bow in.
                //
                // Checked by unpinning `focal` and reading back what the report
                // says, rather than by exposing the estimator -- what matters
                // is the number the algorithm actually uses.
                {
                    ImageSet set;
                    set.images.push_back(photo.Clone());
                    // WIDER rotations than the projection checks above, and
                    // that is the point of doing this separately. h20 is
                    // sin(a)/f: at 4 degrees it is 1.2e-4, which the DLT has
                    // almost no leverage on -- measured, the solve
                    // under-recovered it by 70% and the estimate came back
                    // 1.83x high. At 12 degrees there is three times the signal
                    // in the same term.
                    //
                    // Real frames are the easy case here, which is worth
                    // recording: on the 15-frame panorama the estimate landed
                    // at 4503 px and was optimal, with every larger value
                    // measurably worse. This fixture is harder than the data it
                    // stands in for.
                    set.images.push_back(warpBy(photo, panBy(12.0f)));
                    set.images.push_back(warpBy(photo, panBy(24.0f)));
                    set.shape = Shape{{{"frame", 3}}};

                    std::vector<Data> s;
                    s.push_back(Data{std::move(set)});

                    Pipeline p;
                    p.AddStage(Registry::Get().Create("detect_sift"), "detect_sift",
                               {{-1, 0}}, 1, 1);
                    auto m = Registry::Get().Create("match_brute");
                    if (ParamBase* pb = m->FindParam("chain")) {
                        std::string e; pb->SetFromScript(Value(1.0), &e);
                    }
                    if (ParamBase* pb = m->FindParam("ratio")) {
                        std::string e; pb->SetFromScript(Value(0.95), &e);
                    }
                    p.AddStage(std::move(m), "match_brute", {{0, 0}}, 1, 2);
                    auto a = Registry::Get().Create("align_features");
                    if (ParamBase* pb = a->FindParam("model")) {
                        std::string e; pb->SetFromScript(Value(2.0), &e);  // homography
                    }
                    p.AddStage(std::move(a), "align_features", {{1, 0}}, 1, 3);
                    // focal left at 0, so it is estimated.
                    p.AddStage(Registry::Get().Create("stitch_panorama"),
                               "stitch_panorama", {{2, 0}}, 1, 0);

                    std::string e;
                    if (p.Execute(&s, nullptr, &e) && p.Stages().size() > 3) {
                        const std::string rep = p.Stages()[3].algo->RunReport();
                        std::printf("        [dbg] %s\n", rep.c_str());
                        const size_t k = rep.find("focal ");
                        const float got = (k == std::string::npos)
                            ? 0.0f : float(atof(rep.c_str() + k + 6));
                        // Within 25%: this recovers f from two small rotations of a
                        // resampled photo, which is the hard end of the problem.
                        // What it has to catch is an estimate off by a FACTOR --
                        // a transposed index or a missing principal-point shift
                        // gives that, and it is what makes a panorama fold.
                        Check(got > kFocal * 0.75f && got < kFocal * 1.25f,
                              "the focal estimate recovers the fixture's (" +
                                  std::to_string(int(got)) + " vs " +
                                  std::to_string(int(kFocal)) + " px)");
                    } else {
                        Check(false, "the focal-estimate pipeline ran: " + e);
                    }
                }

                // A LONG CHAIN MUST NOT FLIP, which is the regression this
                // exists for and the bug Tim found in the real panorama.
                //
                // Composing homographies multiplies their perspective error, so
                // the accumulated matrix drifts away from being a rotation
                // exponentially. Measured on the 15-frame sweep, the
                // determinant of K^-1 H K -- which must be 1 for a rotation --
                // ran 2.4 at frame 3, 118.9 at frame 10, 26041 at frame 11, and
                // then went NEGATIVE. A negative determinant is a reflection,
                // and orthonormalising a reflection yields a valid rotation
                // that is 180 degrees off: the last three frames came out
                // upside down.
                //
                // The fix was to orthonormalise each LINK before composing, so
                // the product is a product of rotations and stays one exactly.
                // This asserts the property that fix provides: over a chain
                // long enough for the old code to have flipped, every frame
                // still maps its own centre to somewhere sane, and none of them
                // is inverted.
                {
                    // Six degrees a step rather than three, and warped from the
                    // ORIGINAL each time rather than incrementally.
                    //
                    // At three degrees this fixture dropped frame 5 for a link
                    // that would not read as a rotation -- and that is the
                    // fixture being harsher than reality rather than the code
                    // being wrong. Each frame here is one resampling of the
                    // same photo at a different angle, so two consecutive
                    // frames differ by an interpolation difference ON TOP of
                    // their rotation, and the smaller the rotation the more the
                    // interpolation dominates it.
                    //
                    // The real 15-frame panorama drops nothing at any chain
                    // length, which is the case that matters: its frames come
                    // off the sensor once each.
                    ImageSet set;
                    set.images.push_back(photo.Clone());
                    const int kChain = 9;
                    for (int i = 1; i < kChain; ++i)
                        set.images.push_back(warpBy(photo, panBy(6.0f * float(i))));
                    set.shape = Shape{{{"frame", kChain}}};

                    std::vector<Data> s;
                    s.push_back(Data{std::move(set)});

                    Pipeline p;
                    p.AddStage(Registry::Get().Create("detect_sift"), "detect_sift",
                               {{-1, 0}}, 1, 1);
                    auto m = Registry::Get().Create("match_brute");
                    if (ParamBase* pb = m->FindParam("chain")) {
                        std::string e; pb->SetFromScript(Value(1.0), &e);
                    }
                    if (ParamBase* pb = m->FindParam("ratio")) {
                        std::string e; pb->SetFromScript(Value(0.95), &e);
                    }
                    p.AddStage(std::move(m), "match_brute", {{0, 0}}, 1, 2);
                    auto a = Registry::Get().Create("align_features");
                    if (ParamBase* pb = a->FindParam("model")) {
                        std::string e; pb->SetFromScript(Value(2.0), &e);
                    }
                    p.AddStage(std::move(a), "align_features", {{1, 0}}, 1, 3);
                    auto st = Registry::Get().Create("stitch_panorama");
                    if (ParamBase* pb = st->FindParam("focal")) {
                        std::string e;
                        pb->SetFromScript(Value(double(kFocal)), &e);
                    }
                    p.AddStage(std::move(st), "stitch_panorama", {{2, 0}}, 1, 0);

                    std::string e;
                    if (p.Execute(&s, nullptr, &e) && p.Stages().size() > 3) {
                        const std::string rep = p.Stages()[3].algo->RunReport();
                        std::printf("        [dbg] %s\n", rep.c_str());

                        // No frame was dropped for being unreadable as a
                        // rotation. The old code did not drop them -- it placed
                        // them upside down -- so this also asserts the new
                        // rejection path stays quiet on good data.
                        Check(rep.find("not a rotation") == std::string::npos,
                              "a 9-frame chain keeps every frame");

                        // ...and the result is still one panorama rather than a
                        // canvas stretched to hold a flipped frame. A frame
                        // rotated 180 degrees lands on the far side of the
                        // reference, which roughly doubles the width.
                        const Data* out = p.Resolve({3, 0}, &s);
                        const auto* im = out ? std::get_if<Image>(out) : nullptr;
                        ImageView pv2 = photo.MapCpuRead();
                        const int srcW = pv2.desc.width;
                        Check(im && im->Desc().width < srcW * 3,
                              "a 9-frame chain stays one panorama (" +
                                  std::to_string(im ? im->Desc().width : 0) +
                                  " px from " + std::to_string(srcW) + ")");
                    } else {
                        Check(false, "the long-chain pipeline ran: " + e);
                    }
                }

                // THE REFLECTION ITSELF, fed in directly.
                //
                // The two checks above assert that a healthy chain stays
                // healthy, and it is worth being honest that they do NOT catch
                // the bug they were written for: reintroducing the old
                // compose-then-orthonormalise order leaves both passing,
                // because this fixture's links are clean enough that nine of
                // them do not compound into a reflection. The real panorama's
                // did -- its determinant reached 26041 before flipping -- and a
                // 512 px synthetic photo warped nine times does not reproduce
                // fourteen links of real solver residual.
                //
                // So the invariant is tested where it actually lives: a
                // transform whose linear part is MIRRORED must not come back as
                // a confident 180-degree rotation. That is the property whose
                // absence put three of Tim's frames upside down, and it holds
                // or fails independently of how the chain got there.
                {
                    // A frame flipped in y, which is what a reflected R
                    // produces: determinant negative, and every other entry
                    // perfectly reasonable.
                    Affine mirrored;
                    mirrored.m[4] = -1.0f;          // y -> -y
                    mirrored.m[5] = float(photo.Desc().height);

                    ImageSet set;
                    set.images.push_back(photo.Clone());
                    set.images.push_back(photo.Clone());
                    set.shape = Shape{{{"frame", 2}}};
                    AttachTransform(&set.images[1], mirrored);

                    std::vector<Data> s;
                    s.push_back(Data{std::move(set)});

                    Pipeline p;
                    auto st = Registry::Get().Create("stitch_panorama");
                    if (ParamBase* pb = st->FindParam("focal")) {
                        std::string e;
                        pb->SetFromScript(Value(double(kFocal)), &e);
                    }
                    p.AddStage(std::move(st), "stitch_panorama", {{-1, 0}}, 1, 0);

                    std::string e;
                    if (p.Execute(&s, nullptr, &e)) {
                        const std::string rep = p.Stages()[0].algo->RunReport();
                        std::printf("        [dbg] %s\n", rep.c_str());
                        Check(rep.find("not a rotation") != std::string::npos,
                              "a mirrored transform is rejected, not silently "
                              "turned into a 180-degree rotation");
                    } else {
                        Check(false, "the mirrored-frame pipeline ran: " + e);
                    }
                }
            }

            // FRAMES OUT OF ORDER: a solve that succeeds and is still wrong.
            //
            // Tim hit this dropping files on the palette -- Windows handed them
            // over re-ordered, so the chain paired frames that were never
            // neighbours. The link did not fail: it reported 59 of 65 matches
            // as inliers, 91%, against the 90% a genuinely adjacent pair gives,
            // and the run announced "solved 7 of 7 frames".
            //
            // The inlier RATE cannot separate those. RANSAC finds the largest
            // consistent subset, and a few dozen mutually consistent false
            // matches are still consistent. What separates them is the SHIFT:
            // 1081 px for the adjacent pair, 10908 px for the spurious one.
            //
            // Here: two frames displaced most of a frame width apart, which no
            // sane pairing produces.
            {
                // A SMALL frame, so the guard's threshold -- 1.5 frame widths --
                // is reachable by a shift the matcher can still see.
                //
                // Two fixtures were tried and rejected before this. Warping the
                // 512 px photo by 400 px leaves so little overlap that the
                // matcher finds nothing and the pipeline errors before the
                // guard is reached; TILING it wraps the content around, so the
                // solver correctly recovers the 170 px period rather than the
                // 900 px displacement asked for. Both were the fixture being
                // wrong, not the code -- worth recording, because each looked
                // like a guard failure at first.
                //
                // 64 px frames with 96 px of shift: 1.5x the width exactly, and
                // the wrapped content still matches strongly.
                constexpr int kSW = 64;
                auto makeFrame = [&](int shift) {
                    Image im;
                    ImageDesc d{kSW, kSW, Format::RGBA32F};
                    im.Alloc(d);
                    ImageView v = im.MapCpuWrite();
                    for (int y = 0; y < kSW; ++y)
                        for (int x = 0; x < kSW; ++x) {
                            // An aperiodic hash texture, so every neighbourhood
                            // is distinct and the matches are unambiguous.
                            auto noise = [](int ix, int iy) {
                                uint32_t s = uint32_t(ix) * 374761393u +
                                             uint32_t(iy) * 668265263u;
                                s = (s ^ (s >> 13)) * 1274126177u;
                                return float(s >> 8) / float(1 << 24);
                            };
                            const int sx = x + shift;
                            const float gx = float(sx) / 5.0f, gy = float(y) / 5.0f;
                            const int x0 = int(std::floor(gx)), y0 = int(std::floor(gy));
                            const float tx = gx - float(x0), ty = gy - float(y0);
                            const float n00 = noise(x0, y0),     n10 = noise(x0 + 1, y0);
                            const float n01 = noise(x0, y0 + 1), n11 = noise(x0 + 1, y0 + 1);
                            const float nx0 = n00 + (n10 - n00) * tx;
                            const float nx1 = n01 + (n11 - n01) * tx;
                            float* p = v.At<float>(x, y);
                            p[0] = p[1] = p[2] = 0.2f + 0.6f * (nx0 + (nx1 - nx0) * ty);
                            p[3] = 1.0f;
                        }
                    return im;
                };

                ImageSet set;
                set.images.push_back(makeFrame(0));
                set.images.push_back(makeFrame(96));   // 1.5x the frame width
                set.shape = Shape{{{"frame", 2}}};

                std::vector<Data> s;
                s.push_back(Data{std::move(set)});

                Pipeline p;
                p.AddStage(Registry::Get().Create("detect_sift"), "detect_sift",
                           {{-1, 0}}, 1, 1);
                auto m = Registry::Get().Create("match_brute");
                if (ParamBase* pb = m->FindParam("ratio")) {
                    std::string e; pb->SetFromScript(Value(0.95), &e);
                }
                p.AddStage(std::move(m), "match_brute", {{0, 0}}, 1, 2);
                p.AddStage(Registry::Get().Create("align_features"),
                           "align_features", {{1, 0}}, 1, 3);

                std::string e;
                // A CHAINED frame that cannot be placed is a hard ERROR, not a
                // note appended to a successful-looking report.
                //
                // Tim hit the out-of-order case twice. The second time the run
                // still said "solved 6 of 7 frames, 90% inliers" with the
                // explanation after it -- which reads as success, and was
                // clipped off the end of the info panel entirely. A chain
                // cannot continue past a broken link: every later frame is
                // positioned relative to this one, so what follows is not
                // slightly worse, it is unrelated.
                //
                // This fixture takes the too-few-matches path rather than the
                // shift guard -- reaching the guard needs plenty of GOOD
                // matches at an impossible displacement, which is a
                // contradiction on a synthetic fixture and ordinary on real
                // frames (65 matches at 91% inliers, 10908 px shift on 5796 px
                // frames). Either way the run must not claim success.
                const bool ran = p.Execute(&s, nullptr, &e);
                if (!ran) {
                    std::printf("        [dbg] %s\n", e.c_str());
                    Check(e.find("out of order") != std::string::npos ||
                          e.find("does not follow") != std::string::npos ||
                          e.find("no matches") != std::string::npos,
                          "an unplaceable chained frame is an error that says "
                          "why: \"" + e + "\"");
                } else if (p.Stages().size() > 2) {
                    const std::string rep = p.Stages()[2].algo->RunReport();
                    std::printf("        [dbg] %s\n", rep.c_str());
                    const bool refused =
                        rep.find("REJECTED") != std::string::npos ||
                        rep.find("solved 0 of") != std::string::npos;
                    Check(refused,
                          "a frame displaced most of a frame width is refused, "
                          "not attached: \"" + rep + "\"");

                    // ...and nothing was attached to it, so a merge downstream
                    // treats it as unaligned rather than warping it into the
                    // wrong place.
                    const Data* out = p.Resolve({2, 0}, &s);
                    const auto* os = out ? std::get_if<ImageSet>(out) : nullptr;
                    if (os && os->images.size() == 2)
                        Check(TransformOf(os->images[1]).IsIdentity(),
                              "...and no transform was attached to it");
                }
            }

            // BUNDLE ADJUSTMENT: does it actually reduce the reprojection
            // error, and does it leave a good solution alone?
            //
            // Both matter. A refiner that improves a bad fit but degrades a
            // good one is worse than none, and the second property is the one
            // an over-eager solver breaks.
            {
                auto bundled = [&](int chainLen, float* rms0, float* rms1,
                                   std::string* err) {
                    // A chain of translations rather than rotations. The
                    // adjuster models rotations, so this is deliberately the
                    // case its model does NOT match exactly -- which is the
                    // honest test of "does it make things worse". A fixture
                    // built from the same rotation model it solves would
                    // flatter it.
                    ImageSet set;
                    set.images.push_back(photo.Clone());
                    for (int i = 1; i < chainLen; ++i) {
                        Affine step;
                        step.m[2] = 9.0f * float(i);
                        step.m[5] = -4.0f * float(i);
                        set.images.push_back(warpBy(photo, step));
                    }
                    set.shape = Shape{{{"frame", chainLen}}};

                    std::vector<Data> s;
                    s.push_back(Data{std::move(set)});

                    Pipeline p;
                    p.AddStage(Registry::Get().Create("detect_sift"), "detect_sift",
                               {{-1, 0}}, 1, 1);
                    auto m = Registry::Get().Create("match_brute");
                    if (ParamBase* pb = m->FindParam("chain")) {
                        std::string e; pb->SetFromScript(Value(1.0), &e);
                    }
                    if (ParamBase* pb = m->FindParam("window")) {
                        std::string e; pb->SetFromScript(Value(2.0), &e);
                    }
                    if (ParamBase* pb = m->FindParam("ratio")) {
                        std::string e; pb->SetFromScript(Value(0.95), &e);
                    }
                    p.AddStage(std::move(m), "match_brute", {{0, 0}}, 1, 2);
                    auto a = Registry::Get().Create("align_features");
                    if (ParamBase* pb = a->FindParam("model")) {
                        std::string e; pb->SetFromScript(Value(2.0), &e);
                    }
                    p.AddStage(std::move(a), "align_features", {{1, 0}}, 1, 3);
                    p.AddStage(Registry::Get().Create("bundle_adjust"),
                               "bundle_adjust", {{2, 0}}, 1, 4);

                    if (!p.Execute(&s, nullptr, err)) return false;
                    const std::string rep = p.Stages()[3].algo->RunReport();
                    std::printf("        [dbg] %s\n", rep.c_str());
                    // "rms A -> B px"
                    const size_t k = rep.find("rms ");
                    if (k == std::string::npos) return false;
                    *rms0 = float(atof(rep.c_str() + k + 4));
                    const size_t a2 = rep.find("-> ", k);
                    if (a2 == std::string::npos) return false;
                    *rms1 = float(atof(rep.c_str() + a2 + 3));
                    return true;
                };

                // A five-frame chain: long enough for drift, short enough that
                // the fixture's frames still overlap.
                float r0 = 0.0f, r1 = 0.0f;
                std::string e;
                if (bundled(5, &r0, &r1, &e)) {
                    Check(r1 <= r0 + 0.01f,
                          "bundle_adjust does not make the fit worse (" +
                              std::to_string(r0) + " -> " + std::to_string(r1) + " px)");
                    // A few pixels is healthy; tens means it did not converge.
                    Check(r1 < 5.0f,
                          "bundle_adjust converges to a few pixels (" +
                              std::to_string(r1) + " px)");
                } else {
                    Check(false, "the bundle pipeline ran: " + e);
                }

                // BUNDLE ADJUSTMENT MUST USE ONLY THE GEOMETRIC INLIERS.
                //
                // Huber down-weights a large residual; it does not remove one,
                // and it cannot when the outliers are a large enough fraction
                // to move the fit before the weighting settles. Measured on a
                // real panorama: a detector whose matches were 45% outliers
                // left the solve stalled at 25 px RMS with the focal length run
                // away to 28022 px against a true ~4800, while a detector at 8%
                // outliers converged to 4.7 px on the same frames. Filtering to
                // the inliers took BOTH to 1.4 px.
                //
                // The test asserts the PLUMBING rather than the outcome: after
                // align_features runs, every match set carries an inlier
                // verdict and some matches are rejected.
                //
                // It does not catch BA ignoring that verdict, and it is worth
                // being explicit rather than implying more coverage than
                // exists: removing the filter from bundle_adjust leaves the
                // convergence checks above passing, because this fixture's
                // outliers are not severe enough to stall the solve. What
                // stalled was 45% outliers on 45 MP frames, which no synthetic
                // fixture here reproduces. The marking is testable; the
                // consequence was measured on real data.
                {
                    ImageSet set;
                    set.images.push_back(photo.Clone());
                    for (int i = 1; i < 4; ++i) {
                        Affine step;
                        step.m[2] = 9.0f * float(i);
                        step.m[5] = -4.0f * float(i);
                        set.images.push_back(warpBy(photo, step));
                    }
                    set.shape = Shape{{{"frame", 4}}};

                    std::vector<Data> s;
                    s.push_back(Data{std::move(set)});

                    Pipeline p;
                    p.AddStage(Registry::Get().Create("detect_sift"), "detect_sift",
                               {{-1, 0}}, 1, 1);
                    auto m = Registry::Get().Create("match_brute");
                    if (ParamBase* pb = m->FindParam("chain")) {
                        std::string e3; pb->SetFromScript(Value(1.0), &e3);
                    }
                    if (ParamBase* pb = m->FindParam("window")) {
                        std::string e3; pb->SetFromScript(Value(2.0), &e3);
                    }
                    // Loose, so there ARE outliers for the geometry to reject.
                    if (ParamBase* pb = m->FindParam("ratio")) {
                        std::string e3; pb->SetFromScript(Value(0.99), &e3);
                    }
                    p.AddStage(std::move(m), "match_brute", {{0, 0}}, 1, 2);
                    p.AddStage(Registry::Get().Create("align_features"),
                               "align_features", {{1, 0}}, 1, 3);

                    std::string e3;
                    if (p.Execute(&s, nullptr, &e3)) {
                        const Data* out = p.Resolve({2, 0}, &s);
                        const auto* os = out ? std::get_if<ImageSet>(out) : nullptr;
                        int marked = 0, total = 0, kept = 0;
                        if (os)
                            for (const Image& im : os->images) {
                                const MatchSidecar* ms2 = MatchesOf(im);
                                if (!ms2) continue;
                                for (const MatchSet& st : ms2->sets) {
                                    total  += int(st.matches.size());
                                    marked += int(st.inlier.size());
                                    for (size_t k = 0; k < st.matches.size(); ++k)
                                        if (st.IsInlier(k)) ++kept;
                                }
                            }
                        Check(total > 0 && marked == total,
                              "align_features marks every match set with an "
                              "inlier verdict (" + std::to_string(marked) +
                              " of " + std::to_string(total) + ")");
                        Check(kept > 0 && kept < total,
                              "...and it rejects some without rejecting all (" +
                                  std::to_string(kept) + " of " +
                                  std::to_string(total) + " kept)");
                    } else {
                        Check(false, "the inlier-marking pipeline ran: " + e3);
                    }
                }

                // Without matches it is an error, not a silent success -- the
                // same contract align_features has.
                {
                    ImageSet plain;
                    for (int f = 0; f < 2; ++f) {
                        Image im;
                        im.Alloc({32, 32, Format::RGBA8});
                        plain.images.push_back(std::move(im));
                    }
                    plain.shape = Shape{{{"frame", 2}}};
                    std::vector<Data> s;
                    s.push_back(Data{std::move(plain)});

                    Pipeline p;
                    p.AddStage(Registry::Get().Create("bundle_adjust"),
                               "bundle_adjust", {{-1, 0}}, 1, 1);
                    std::string e2;
                    const bool ran = p.Execute(&s, nullptr, &e2);
                    Check(!ran && e2.find("no matches") != std::string::npos,
                          "bundling without matches is an error: \"" + e2 + "\"");
                }
            }

            // Aligning without matches is an error rather than a silent no-op:
            // a script missing its matcher otherwise looks like frames that
            // happen to need no alignment.
            {
                ImageSet plain;
                for (int f = 0; f < 2; ++f) {
                    Image im;
                    im.Alloc({32, 32, Format::RGBA8});
                    plain.images.push_back(std::move(im));
                }
                plain.shape = Shape{{{"frame", 2}}};
                std::vector<Data> s;
                s.push_back(Data{std::move(plain)});

                Pipeline p;
                p.AddStage(Registry::Get().Create("align_features"), "align_features",
                           {{-1, 0}}, 1, 1);
                std::string e;
                const bool ran = p.Execute(&s, nullptr, &e);
                Check(!ran && e.find("no matches") != std::string::npos,
                      "aligning without matches is an error: \"" + e + "\"");
            }
        }
    }

    // crop: the two modes, and the property that makes the toggle trustworthy.
    {
        auto run = [&](bool preview, double l, double r, double t, double b,
                       double angle, Image* out) {
            auto algo = Registry::Get().Create("crop");
            if (!algo) return false;
            auto set = [&](const char* n, double v) {
                if (ParamBase* pb = algo->FindParam(n)) {
                    std::string e; pb->SetFromScript(Value(v), &e);
                }
            };
            set("preview", preview ? 1.0 : 0.0);
            set("left", l); set("right", r);
            set("top", t);  set("bottom", b);
            set("angle", angle);

            // A gradient with a hard mark in a known place, so a crop can be
            // checked by WHERE the mark ends up rather than by pixel counts.
            Image src;
            ImageDesc d{100, 100, Format::RGBA32F};
            src.Alloc(d);
            ImageView v = src.MapCpuWrite();
            for (int y = 0; y < 100; ++y)
                for (int x = 0; x < 100; ++x) {
                    float* p = v.At<float>(x, y);
                    p[0] = float(x) / 100.0f;
                    p[1] = float(y) / 100.0f;
                    p[2] = 0.25f;
                    p[3] = 1.0f;
                }

            Pipeline p;
            std::vector<Data> s;
            s.push_back(Data{std::move(src)});
            p.AddStage(std::move(algo), "crop", {{-1, 0}}, 1, 1);
            std::string e;
            if (!p.Execute(&s, nullptr, &e)) return false;
            const Data* dd = p.Resolve({0, 0}, &s);
            const auto* im = dd ? std::get_if<Image>(dd) : nullptr;
            if (!im) return false;
            *out = const_cast<Image*>(im)->Clone();
            return true;
        };

        // APPLIED: the output really is a different raster.
        {
            Image got;
            const bool ok = run(false, 0.10, 0.20, 0.25, 0.05, 0.0, &got);
            Check(ok, "crop runs");
            if (ok) {
                // 100 wide, 10% off the left and 20% off the right -> 70.
                // 100 tall, 25% off the top and 5% off the bottom -> 70.
                Check(got.Desc().width == 70 && got.Desc().height == 70,
                      "crop resizes its output (" +
                          std::to_string(got.Desc().width) + "x" +
                          std::to_string(got.Desc().height) + ", expected 70x70)");

                // ...and it is the RIGHT part of the image. The red channel is
                // x/100, so the first output column must read 0.10.
                ImageView gv = got.MapCpuRead();
                const float first = gv.At<float>(0, 0)[0];
                const float last  = gv.At<float>(69, 0)[0];
                Check(std::fabs(first - 0.10f) < 0.02f &&
                      std::fabs(last - 0.79f) < 0.02f,
                      "crop takes the requested rectangle (x runs " +
                          std::to_string(first) + ".." + std::to_string(last) + ")");
            }
        }

        // PREVIEW: same size as the input, and it draws something.
        {
            Image prev;
            const bool ok = run(true, 0.10, 0.20, 0.25, 0.05, 0.0, &prev);
            Check(ok && prev.Desc().width == 100 && prev.Desc().height == 100,
                  "crop preview keeps the input's size (" +
                      std::to_string(prev.Desc().width) + "x" +
                      std::to_string(prev.Desc().height) + ")");

            if (ok) {
                // The rectangle is drawn in green, and the outside is dimmed.
                // Both are checked, because either alone can pass while the
                // other is broken -- and a uniformly dimmed frame with NO
                // rectangle is exactly what a sign error in the inside test
                // produces. That happened: the first version had the winding
                // backwards and every pixel scored as outside.
                ImageView pv = prev.MapCpuRead();

                // Counted as "green DOMINATES", not as a fixed brightness: the
                // line's value is scaled to the image's own range, so asserting
                // an absolute level would be testing that scaling rather than
                // the drawing. The fixture's own green channel is y/100 and its
                // red is x/100, so a pixel where green greatly exceeds both
                // others can only be a drawn line.
                int greenPixels = 0;
                for (int y = 0; y < 100; ++y)
                    for (int x = 0; x < 100; ++x) {
                        const float* p = pv.At<float>(x, y);
                        if (p[1] > 0.4f && p[0] < p[1] * 0.2f && p[2] < p[1] * 0.2f)
                            ++greenPixels;
                    }
                Check(greenPixels > 100,
                      "crop preview draws the rectangle (" +
                          std::to_string(greenPixels) + " green pixels)");

                // A pixel well outside the rectangle is dimmed; one well inside
                // is not. Blue is a constant 0.25 in the fixture, so it reads
                // the dimming directly without the gradient confusing it.
                const float outside = pv.At<float>(2, 2)[2];
                const float inside  = pv.At<float>(50, 50)[2];
                Check(outside < inside * 0.8f,
                      "crop preview dims outside the rectangle (" +
                          std::to_string(outside) + " vs " +
                          std::to_string(inside) + ")");
            }
        }

        // THE TWO MODES AGREE. This is the property the whole design rests on:
        // the rectangle drawn in preview has to be the rectangle that comes out
        // when preview is turned off, or the toggle is a lie.
        //
        // Checked through the ROTATED case, because that is where they could
        // most easily disagree -- rotating about the frame centre instead of
        // the rectangle centre would leave both modes individually plausible
        // and showing different parts of the image.
        {
            Image applied;
            if (run(false, 0.15, 0.15, 0.15, 0.15, 8.0, &applied)) {
                ImageView av = applied.MapCpuRead();
                const int cw = av.desc.width, chh = av.desc.height;
                // The centre of the cropped result must be the centre of the
                // rectangle in the source: a symmetric crop of a 100x100 frame
                // centres at (50, 50), where the fixture reads r = g = 0.5.
                const float* c = av.At<float>(cw / 2, chh / 2);
                Check(std::fabs(c[0] - 0.5f) < 0.03f && std::fabs(c[1] - 0.5f) < 0.03f,
                      "crop rotates about the rectangle's centre, so the modes "
                      "agree (centre reads " + std::to_string(c[0]) + ", " +
                      std::to_string(c[1]) + ")");
            } else {
                Check(false, "the rotated crop ran");
            }
        }

        // A crop that trims nothing is a pass-through the pipeline can alias
        // away -- but only with preview OFF, since a preview genuinely changes
        // the pixels.
        {
            auto algo = Registry::Get().Create("crop");
            if (ParamBase* pb = algo->FindParam("preview")) {
                std::string e; pb->SetFromScript(Value(0.0), &e);
            }
            Check(algo->IsNoOp(),
                  "an untrimmed crop reports itself as a no-op");
            if (ParamBase* pb = algo->FindParam("preview")) {
                std::string e; pb->SetFromScript(Value(1.0), &e);
            }
            Check(!algo->IsNoOp(),
                  "...but not in preview, where it draws a rectangle");
        }
    }

    // draw_matches turns the sidecar into something visible, and says so when
    // there is nothing to draw.
    {
        // A two-frame group with real structure, so the detector finds
        // something and the matcher has work to do.
        ImageSet set;
        for (int f = 0; f < 2; ++f) {
            Image im;
            ImageDesc d{96, 96, Format::RGBA32F};
            im.Alloc(d);
            ImageView v = im.MapCpuWrite();
            for (int y = 0; y < 96; ++y)
                for (int x = 0; x < 96; ++x) {
                    // A few blobs, shifted between the frames.
                    float a = 0.2f;
                    const float ox = float(x) - float(f) * 5.0f;
                    const float oy = float(y);
                    for (int b = 0; b < 4; ++b) {
                        const float bx = 20.0f + 18.0f * float(b);
                        const float by = 30.0f + 12.0f * float(b % 3);
                        const float ex = ox - bx, ey = oy - by;
                        a += 0.6f * std::exp(-(ex * ex + ey * ey) / 32.0f);
                    }
                    float* p = v.At<float>(x, y);
                    p[0] = p[1] = p[2] = a;
                    p[3] = 1.0f;
                }
            set.images.push_back(std::move(im));
        }
        set.shape = Shape{{{"frame", 2}}};

        std::vector<Data> s;
        s.push_back(Data{std::move(set)});

        Pipeline p;
        p.AddStage(Registry::Get().Create("detect_sift"), "detect_sift", {{-1, 0}}, 1, 1);
        p.AddStage(Registry::Get().Create("match_brute"), "match_brute", {{0, 0}}, 1, 2);
        p.AddStage(Registry::Get().Create("draw_matches"), "draw_matches", {{1, 0}}, 1, 3);

        std::string e;
        const bool ok = p.Execute(&s, nullptr, &e);
        Check(ok, "detect -> match -> draw runs on a group" + (ok ? "" : ": " + e));

        if (ok) {
            // The drawn frame must DIFFER from what the matcher produced, or
            // the visualiser ran and drew nothing -- which a "3 stages ran"
            // check would not distinguish.
            const Data* before = p.Resolve({1, 0}, &s);
            const Data* after  = p.Resolve({2, 0}, &s);
            const auto* bs = before ? std::get_if<ImageSet>(before) : nullptr;
            const auto* as = after  ? std::get_if<ImageSet>(after)  : nullptr;
            if (bs && as && bs->images.size() == 2 && as->images.size() == 2) {
                ImageView bv = const_cast<Image&>(bs->images[1]).MapCpuRead();
                ImageView av = const_cast<Image&>(as->images[1]).MapCpuRead();
                // Counting CHANGED pixels would pass on the dim pass alone --
                // it touches every pixel, so the whole frame differs whether or
                // not a single line was drawn. Count pixels that got BRIGHTER
                // instead: dimming can only darken, so anything brighter than
                // the input is ink.
                int brighter = 0;
                for (int y = 0; y < 96; ++y)
                    for (int x = 0; x < 96; ++x)
                        if (av.At<float>(x, y)[1] > bv.At<float>(x, y)[1] + 1e-3f)
                            ++brighter;
                Check(brighter > 0, "draw_matches actually drew lines (" +
                                        std::to_string(brighter) + " pixels)");
            }
        }

        // Without a matcher upstream it must SAY there is nothing rather than
        // silently passing the group through.
        std::vector<Data> s2;
        {
            ImageSet plain;
            for (int f = 0; f < 2; ++f) {
                Image im;
                im.Alloc({32, 32, Format::RGBA8});
                plain.images.push_back(std::move(im));
            }
            plain.shape = Shape{{{"frame", 2}}};
            s2.push_back(Data{std::move(plain)});
        }
        Pipeline p2;
        p2.AddStage(Registry::Get().Create("draw_matches"), "draw_matches", {{-1, 0}}, 1, 1);
        std::string e2;
        p2.Execute(&s2, nullptr, &e2);
        if (!p2.Stages().empty() && p2.Stages()[0].algo) {
            const std::string r = p2.Stages()[0].algo->RunReport();
            Check(r.find("no matches") != std::string::npos,
                  "and reports when nothing is matched: \"" + r + "\"");
        }
    }

    // Matching a group whose frames used DIFFERENT detectors is an error, not
    // something to paper over: the descriptors are not comparable and the
    // resulting pairs would mean nothing.
    {
        UiState ui; Pipeline p; std::vector<Data> src; std::string err;
        RunScript("src = image(\"test\")\n"
                  "out = match_brute(src)\n"
                  "display(out)\n", &ui, &p, &err, &src);
        Check(!err.empty(),
              "matching a single image is an error: \"" + err + "\"");
    }

    // draw_features marks the image where the features are, and says so when
    // there are none to draw.
    {
        UiState ui; Pipeline p; std::vector<Data> src; std::string err;
        const bool ok = RunScript(
            "src = image(\"test\")\n"
            "src => detect_sift() => draw_features() => display(\"features\")\n",
            &ui, &p, &err, &src);
        Check(ok && p.Stages().size() == 2,
              "detect then draw runs" + (ok ? "" : ": " + err));

        // Drawing without a detector must be reported rather than silently
        // producing the input -- a script missing its detector otherwise looks
        // like a detector that found nothing.
        UiState ui2; Pipeline p2; std::vector<Data> src2;
        const bool ok2 = RunScript(
            "src = image(\"test\")\n"
            "src => draw_features() => display(\"none\")\n", &ui2, &p2, &err, &src2);
        Check(ok2 && p2.Stages().size() == 1, "draw without a detector still runs");
        if (ok2 && !p2.Stages().empty() && p2.Stages()[0].algo) {
            const std::string r = p2.Stages()[0].algo->RunReport();
            Check(r.find("no features") != std::string::npos,
                  "and says there were none: \"" + r + "\"");
        }
    }

    std::printf("\n%s\n", g_fail == 0 ? "all checks passed" : "FAILURES PRESENT");
    return g_fail == 0 ? 0 : 1;
}
