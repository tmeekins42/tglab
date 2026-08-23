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

    WNDCLASSEXA wc = {sizeof(wc), CS_CLASSDC, WndProc, 0, 0,
                      GetModuleHandle(nullptr), nullptr, nullptr, nullptr, nullptr,
                      "tglab_hang", nullptr};
    RegisterClassExA(&wc);
    HWND hwnd = CreateWindowA(wc.lpszClassName, "hang", WS_OVERLAPPEDWINDOW,
                              100, 100, 1280, 800, nullptr, nullptr, wc.hInstance, nullptr);

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
