// Reproduces the GPU-display device hang: load a script, click through the
// view tabs (several docked viewers is a precondition -- one alone never
// hangs), end on "adjusted", then nudge the brightness slider 40 times.
//
//   build/Debug/dbg_hang.exe scripts/hello.tgl assets/test.png
//
// Both arguments are required. Measured 3/3 device removals with
// TGLAB_GPUDISPLAY=1 and 0/3 without, which is what pinned the fault on
// GpuTexture::UpdateFromGpu. See todo.txt for the open diagnosis.
//
// It #includes main.cpp to drive the real App through real frames -- the hang
// only appears with genuine present/worker interleaving, so a stub will not do.
#include <cstdio>
#include <cstdlib>
#include <crtdbg.h>
#include <string>
#include <windows.h>

#define main tglab_app_main
#include "../src/app/main.cpp"
#undef main

int main(int argc, char** argv) {
    using namespace tglab;

    if (argc < 3) {
        std::printf("usage: dbg_hang <script.tgl> <image>\n"
                    "  e.g. dbg_hang scripts/hello.tgl assets/test.png\n");
        return 2;
    }

    SetUnhandledExceptionFilter(CrashHandler);

    // Never pop a dialog. This is a headless harness: a Windows error box or a
    // CRT assert window blocks until somebody clicks it, which turns "the test
    // crashed" into "the test hangs forever and a human has to dismiss it".
    // The crash handler above still prints the stack, which is the part worth
    // having.
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);

    WNDCLASSEXA wc = {sizeof(wc), CS_CLASSDC, WndProc, 0, 0,
                      GetModuleHandle(nullptr), nullptr, nullptr, nullptr, nullptr,
                      "tglab_hang", nullptr};
    RegisterClassExA(&wc);
    HWND hwnd = CreateWindowA(wc.lpszClassName, "hang", WS_OVERLAPPEDWINDOW,
                              100, 100, 1280, 800, nullptr, nullptr, wc.hInstance, nullptr);


    // Generate distinct images for the palette-reallocation check.
    //
    // Distinct NAMES matter: InstallLoadedImage replaces an entry whose name
    // already exists and only appends for a new one, so dropping the same file
    // twice never grows the vector -- which is exactly why an earlier version
    // of this test passed against the unfixed code and proved nothing.
    //
    // Written to the build tree rather than committed: five near-identical
    // fixtures in the repository would be pure weight.
    std::string dropList;
    if (getenv("TGLAB_PALETTE_TEST")) {
        for (int i = 0; i < 12; ++i) {
            Image img;
            img.Alloc({64, 64, Format::RGBA8});
            ImageView v = img.MapCpuWrite();
            for (int y = 0; y < 64; ++y)
                for (int x = 0; x < 64; ++x) {
                    uint8_t* p = v.At<uint8_t>(x, y);
                    p[0] = uint8_t(x * 4 + i * 40);
                    p[1] = uint8_t(y * 4);
                    p[2] = uint8_t(i * 50);
                    p[3] = 255;
                }
            const std::string path = "palette_fixture_" + std::to_string(i) + ".png";
            std::string e;
            if (!SavePng(path, img, &e)) {
                std::printf("could not write %s: %s\n", path.c_str(), e.c_str());
                return 1;
            }
            if (!dropList.empty()) dropList += ";";
            dropList += path;
        }
        SetEnvironmentVariableA("TGLAB_DROPTEST",
                                ("900,400," + dropList).c_str());
    }

    App app;
    g_app = &app;
    if (!app.Init(hwnd)) { std::printf("init failed\n"); return 1; }
    ShowWindow(hwnd, SW_SHOWDEFAULT);
    app.LoadImageIntoPalette(argv[2]);
    app.SetScriptPath(argv[1]);

    auto pump = [&] {
        MSG msg;
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    };
    auto settle = [&](int n, const char* what) {
        for (int i = 0; i < n; ++i) {
            pump();
            app.Frame();
        }
        std::printf("  ok: %s\n", what);
        std::fflush(stdout);
    };

    settle(120, "settled");

    if (!getenv("NOTABS")) {
        for (const char* v : {"original", "greyscale", "adjusted"}) {
            ImGui::SetWindowFocus(v);
            settle(60, v);
        }
    }


    // The About dialog: open it and draw real frames.
    //
    // ImGui reports a mismatched Begin/End or a bad style push as an assert at
    // draw time, so a dialog that is never drawn is never checked. Closing it
    // again confirms the modal releases properly rather than wedging the UI.
    app.OpenAboutForTest();
    settle(30, "about opened");
    if (!app.AboutOpenForTest()) { std::printf("  FAIL: about closed itself\n"); return 1; }
    std::printf("  ok: about dialog drew 30 frames\n");

    // Then drag the slider.
    for (int i = 0; i < 40; ++i) {
        app.NudgeControlForTest(std::sin(double(i) * 0.3) * 0.6);
        pump();
        app.Frame();
    }
    std::printf("  ok: slider moved 40 times\n");

    app.Shutdown();
    DestroyWindow(hwnd);
    UnregisterClassA(wc.lpszClassName, wc.hInstance);
    std::printf("survived\n");
    return 0;
}
