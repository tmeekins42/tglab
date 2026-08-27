# tglab

A computer-vision / computer-graphics algorithm research lab. C++20, Dear ImGui
(docking), D3D12/HLSL via DXC, CMake. Images load into a palette; a glue-only
`.tgl` script wires C++ algorithms together and declares the sliders.

## Building

Use the script, not cmake directly:

    .\build.ps1                 configure, build, run the tests
    .\build.ps1 -NoTest         build only
    .\build.ps1 -Target tglab   one target
    .\build.ps1 -Fresh          delete build/ first
    .\build.ps1 -KeepApp        do not close a running tglab.exe

**It closes a running tglab.exe first, deliberately.** A running app holds a
lock on its own binary, so the link fails with LNK1104 while everything else in
the build reports success -- and ctest then passes against the OLD executable.
That failure mode is quiet and has cost several rounds of "I closed it, try
again". If a build is being run the previous session is over, so an open viewer
is stale rather than precious.

Only processes launched from this directory are closed, matched by path, so a
different tglab elsewhere is left alone. `dbg_hang` is included since it links
the whole app and can be left behind by an interrupted test.

## Verification

Beyond ctest, three checks worth running on anything touching the GPU or the
display path:

    build\Release\gpu_audit.exe      per-algorithm GPU hygiene, expect "N clean"
    build\Release\gpu_leak.exe       VRAM growth over repeated runs
    build\Release\dbg_hang.exe scripts\hello.tgl assets\test.png

`dbg_hang` drives the real app through real frames; set `TGLAB_GPUDISPLAY=1` to
exercise the GPU display path, which is where a fallback would otherwise hide.

## Things this codebase has learned the hard way

- **A fixture that cannot express a failure cannot test for it.** Several tests
  here passed identically with a fix present and removed. When adding one,
  break the implementation deliberately and confirm the test notices.
- **Aggregate statistics disagree with the rendered image, in both
  directions.** Score localized defects at edge sites and as a high percentile,
  and look at a crop.
- **Removing a clamp is not the same as the value surviving.** Check the store,
  not just the computation.
- **CPU and GPU paths must be checked against each other**, not each against
  intuition. `tglab_demosaic_tests` does this per CFA phase and has caught real
  divergence twice.
