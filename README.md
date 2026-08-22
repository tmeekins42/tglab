# tglab

A lab bench for computer-vision / computer-graphics **algorithm research**.

Drop images in, write a short script that wires algorithms together, expose the
interesting parameters as sliders, and watch results change as you drag them.
Algorithms are written in C++, optionally with an HLSL compute kernel. The
scripting language is deliberately *glue only* — it never contains algorithm
logic.

Adding an experiment costs one `.cpp` file and a few lines of script.

---

## Build

Requires Visual Studio 2022 (or newer) with the Windows SDK. CMake and Ninja
ship with VS, so nothing extra to install.

Dear ImGui and LibRaw are submodules, so clone recursively:

```sh
git clone --recursive https://github.com/tmeekins42/tglab.git
cd tglab
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

**Build Release for real work.** Debug is roughly 30x slower, which on a
multi-megapixel scan is the difference between a second and a minute -- long
enough to look like the app has hung. Build Debug only when you need the
debugger or the D3D12 validation layer:

```sh
cmake --build build --config Debug
```

(In an existing clone, `git submodule update --init` fetches both. LibRaw is
built from source and adds about a minute to a clean build; it is only
recompiled when the submodule moves.)

## Run

Arguments are a `.tgl` script plus any images to preload. Paths are relative to
the repository root, which is where the app expects to run from.

```sh
./build/Release/tglab.exe scripts/thresholds.tgl assets/test.png
```

Images can also be **dragged in from Explorer** at any time.

| Panel | What it does |
|---|---|
| **Images** | The palette. `image("name")` refers to these by filename without extension. |
| **Scripts** | Every `.tgl` beside the current script — click to switch. |
| **Algorithms** | Every registered algorithm, with its ports and parameters. |
| **Controls** | Sliders and dropdowns the script declared. |
| **Status** | Errors, run time, and how many stages used the GPU. |
| **Image Info** | Opened from the View menu: size, RGB histogram, and EXIF capture settings. |
| **Compare CPU / GPU** | Opened from the Compute menu. See below. |

Editing the script file and saving re-runs it (or press **F5**). Slider values
survive the reload, so tuning is not lost when you edit. A parse error leaves
the last good result on screen and reports the line in **Status** and on
stderr.

Double-click any control to restore its scripted default, or use **Reset all**.



### The image palette

Each row shows a thumbnail, the **name scripts refer to**, and the file it was
loaded from. Those last two are deliberately separate:

- **Drop a file onto an existing row** to swap what is behind it while keeping
  the name. `image("test")` keeps working, now backed by the new file — which
  is the fast way to run the same script over a series of scans.
- **Drop onto empty space** to add a new entry, named after the file.
- **Double-click a name** to rename it, or **right-click a row** for rename,
  reload from disk, and remove.

Names are case-sensitive and are the filename without its extension, so
`IMG_2369.jpg` arrives as `image("IMG_2369")`. Renaming a slot re-runs the
script, so a script referring to the old name will report the missing image
rather than failing silently.

### Camera raw

`.CR3`, `.CR2`, `.NEF`, `.ARW`, `.DNG` and the other common raw formats load
like any other image, decoded by LibRaw. Dispatch is by extension, so nothing
else in the app knows raw exists.

**The palette holds the sensor mosaic**, not a finished image. That is where
the dynamic range lives: a Canon CR2 measures ~13,400 distinct levels and a
Sony ARW ~15,600, against the 256 an 8-bit conversion leaves. Adjusting exposure
on the 8-bit version recovers nothing, because the highlight information is
already gone.

`image("name")` demosaics automatically using the default method, so **a script
never has to mention demosaicing** — the same script works whether a PNG or a
CR3 is dropped on the slot. `mosaic("name")` returns the undemosaiced sensor
data, which is what [scripts/demosaic.tgl](scripts/demosaic.tgl) uses to compare
two methods side by side.

Three methods, all with GPU kernels:

| Method | What it is |
|---|---|
| `demosaic_passthrough` | The mosaic as it actually is — zoom in and the colour filter array is visible, one colour per pixel, before any interpolation. |
| `demosaic_bilinear` | The standard baseline. Averages the nearest neighbours of each missing colour; fringes visibly at edges. |
| `demosaic_malvar` | Malvar-He-Cutler (2004), the default. Corrects bilinear using the centre channel's second derivative, which predicts how the other channels change. Roughly **half** the colour fringing on fine detail (measured 0.078 against 0.136) for barely more arithmetic. |

Malvar's gains are exposed as parameters rather than baked in: set `alpha`,
`beta` and `gamma` to zero and it reduces to exactly bilinear, which makes the
correction term something you can test rather than take on trust.

`TGLAB_RAW_RGB=1` falls back to LibRaw's own conversion, as an escape hatch if a
camera's mosaic cannot be read.

Two things worth knowing:

- **Decoding is slow** — about 6 s for a 45 MP CR3, most of it demosaicing. It
  runs on the loader thread, so the UI stays responsive and the Images panel
  says "loading...", but the wait is real.
- **Auto-brightening is on.** Turning it off does *not* give unscaled data:
  LibRaw still applies a gamma curve, just against a fixed white point of
  0x2000, and a correctly-exposed frame then comes out at roughly 20/255 —
  essentially black. With it on, the white point is chosen by percentile, which
  preserves relative exposure between frames rather than normalising them all
  to the same level. `TGLAB_RAW_NOBRIGHT=1` selects the fixed white point for
  anyone who wants it.

### Performance note

Algorithms run on the worker thread, so the UI stays responsive no matter how
long a run takes — but a **Debug build is roughly 30x slower than Release**.
On an 8 MP scan (3504x2336) the difference is stark:

| | Debug | Release |
|---|---|---|
| `threshold_otsu` | 1.3 s | 0.17 s |
| `threshold_sauvola` | 3.5 s | 0.33 s |
| `gaussian_blur` (sigma 8) | ~60 s | 3.1 s |

If a large image feels like it has hung, check the Status panel: it shows
elapsed time and a spinner while the worker is busy. The first thing to verify
is that you are running `build/Release/tglab.exe` and not the Debug build.

**A run in progress is abandoned when you change something.** Moving a slider
or switching algorithm mid-run cancels the old value rather than queueing
behind it, so the wait is always for the *current* settings and never for a
stale one. That matters most where it costs most: dragging a slider on a filter
taking a minute would otherwise mean a minute per nudge.

Cancellation is cooperative — algorithms check between rows — so an
unresponsive one finishes its current pass and is then discarded. Nothing
partial is ever displayed or cached.

Non-power-of-2 image sizes are fine, as are spaces in file paths. Palette
names are the filename without extension and are **case-sensitive**, so
`IMG_2369.jpg` is referenced as `image("IMG_2369")`.

## Tests

```sh
./build/Release/tglab_tests.exe          # language and pipeline semantics (fast)
./build/Release/tglab_filter_tests.exe   # filter behaviour: noise, edges, flat-field
./build/Release/tglab_demosaic_tests.exe # demosaic: reconstruction, CFA phase, colour
./build/Release/tglab_runtime_tests.exe  # worker thread, shaders, GPU (needs a device)
```

---

## Script reference

A script is a sequence of statements. Comments run from `#` to end of line.
There is no control flow and no user-defined functions — the language exists to
wire algorithms together, not to compute.

### Values

Numbers, strings, images (really a reference to one output of one stage),
algorithms, matrices, and lists.

```
sigma  = 2.5                     # number
name   = "lena"                  # string
kernel = [[-1,0,1],[-2,0,2],[-1,0,1]]   # matrix literal
```

### Builtins

| Builtin | Meaning |
|---|---|
| `image("name")` | An image from the palette. A raw file is demosaiced automatically. |
| `mosaic("name")` | The undemosaiced sensor mosaic. Errors on a non-raw image. |
| `slider("label", min, max, default)` | Declares a slider; returns its current value. |
| `check("label", default)` | Declares a checkbox; returns 0 or 1. |
| `choose("label", [a, b, c])` | Dropdown of algorithms; returns the selected one. |
| `choose("label", "category")` | Same, but offers every algorithm in a category. |
| `choose("label", opts, default)` | An optional third argument names which option starts selected, and which a reset returns to. |
| `params(algo)` | Declares a control for each of `algo`'s parameters; returns `algo`. |
| `params(algo, "name")` | The same, as an independently-controlled instance, so one algorithm can appear twice. |
| `display(data)` / `display(data, "name")` | Opens a viewer panel. |

### Calling algorithms

Any registered algorithm is callable by name. Named arguments set parameters:

```
blurred = gaussian_blur(src, sigma = 2.0)
```

Multi-output algorithms bind positionally, and `_` discards what you don't
need:

```
gx, gy, mag = sobel(blurred)     # gx/gy are signed R32F
_,  _,  mag = sobel(blurred)     # magnitude only
out         = sobel(blurred)     # a single target takes the first output
```

`_` is write-only, as in Rust and Go — reading it back is an error, since it
would just return whichever output landed last. Discarded outputs are still
computed and cached, so adding a viewer for one later costs nothing. (`_tmp`
and similar are ordinary names; only a bare `_` is special.)

### Selecting algorithms at run time

`choose()` returns an algorithm as a value, which can then be called:

```
op  = choose("operator", "threshold")   # dropdown of every threshold method
out = op(src)
```

Passing a *category* means a newly written algorithm declaring that category
appears in the dropdown with no script edit.

### Automatic parameter controls

`params()` declares a control for every parameter the given algorithm has,
using its own name, range, and default:

```
op   = choose("method", "threshold")
mask = params(op)(src)
```

Switching the dropdown swaps the entire control set, because controls a run
does not re-declare are dropped. This is how one script serves a whole category
of algorithms with different parameters — no conditionals required.

Explicit named arguments still win over `params()`:

```
mask = params(op)(src, window = 31)     # window is fixed, the rest are sliders
```

**Naming an instance.** A second argument names the control set, which is what
lets the *same* algorithm appear twice with independent settings:

```
a = choose("filter A", "filter")
b = choose("filter B", "filter")

outA = params(a, "A")(src)
outB = params(b, "B")(src)
```

Without the name, both calls key their controls by algorithm name alone, so
picking the same algorithm in both dropdowns would give one shared set — and
comparing it against itself at two settings would be impossible.

Each named set becomes its own collapsing group in the Controls panel, showing
short parameter names (`k`, `window`) since the algorithm is already named in
the group header. Hovering any control shows its full identifier, range, and
default.

### Example

```
src = image("test")

op    = choose("method", "threshold")
sigma = slider("pre-blur", 0.0, 8.0, 1.0)

blurred = gaussian_blur(src, sigma = sigma)
mask    = params(op)(blurred)

display(src,     "source")
display(blurred, "blurred")
display(mask,    "mask")
```

More in [scripts/](scripts/).

---

## Writing an algorithm

Create one `.cpp` under the matching category folder in
[src/algorithms/](src/algorithms/). Nothing else is edited — the build globs
that tree and registration is automatic.

```cpp
#include "../../algo_util/histogram.h"
#include "../../core/algorithm.h"

namespace tglab {

class MyThreshold : public AlgorithmBase {
public:
    const char* Name()     const override { return "threshold_mine"; }
    const char* Category() const override { return "threshold"; }

    PortList Inputs()  const override { return {{"src",  DataType::Image, FormatSpec::Any}}; }
    PortList Outputs() const override { return {{"mask", DataType::Image, FormatSpec::R32F}}; }

    void RunCPU(RunCtx& ctx) override {
        const ImageView src = ctx.In(0);
        ImageView       dst = ctx.Out(0);
        if (!src.Valid() || !dst.Valid()) return;

        const float level = m_level;    // reads as a plain float
        /* ... */
    }

private:
    Param<float> m_level{this, "level", 0.5f, 0.0f, 1.0f};
};

REGISTER_ALGORITHM(MyThreshold);

} // namespace tglab
```

`Param<T>` is the single storage location for a parameter — the script writes
it, the UI writes it, the algorithm reads it. There is nothing to keep in sync,
and `params()` exposes it automatically.

**Document every parameter.** `help` is one line saying what the parameter
actually does, shown at the top of its tooltip. A name and a range do not tell
you what `k` or `eps` controls, and for an unfamiliar algorithm that is the
whole question. Say what changes and in which direction:

```cpp
Param<float> m_k{this, "k", 0.05f, 0.001f, 1.0f,
                 {.help = "Edge threshold, as a fraction of the intensity "
                          "range. Note this runs OPPOSITE to a blur radius: "
                          "LOWER k preserves more."}};
```

A test asserts that every parameter of every `"filter"` algorithm has help
text, so an undocumented one fails the build rather than reaching the panel.

**Tuning the widget.** The same `ParamOpts` says how a parameter wants to be
driven, which matters when the declared range is far wider than the range worth
dragging through:

```cpp
// Sigma is legal up to 20 but tuned around 0.5..3, so the slider covers the
// useful part and larger values stay reachable by typing.
Param<float> m_sigma{this, "sigma", 2.0f, 0.1f, 20.0f,
                     {.step = 0.1, .softMin = 0.1, .softMax = 5.0}};

// Only odd window sizes are meaningful, so step past the even ones.
Param<int> m_window{this, "window", 15, 3, 201,
                    {.step = 2, .softMin = 3, .softMax = 51}};
```

- `step` — quantum for the arrow keys and for snapping. Applied in
  `Param::set()`, so a typed or script-assigned value lands on the same grid as
  a dragged one.
- `softMin`/`softMax` — the slider's extent. The full `[lo, hi]` range stays
  reachable by ctrl+click; `ImGuiSliderFlags_AlwaysClamp` keeps a typed value
  inside it.

Omit the argument and the parameter behaves exactly as before.

Every numeric row supports **ctrl+click to type an exact value**, **left/right
arrows to step** while the slider is focused, and **right-click to reset** to
the declared default. None of these add chrome, so the panel stays one row per
parameter.

**Categories must hold interchangeable algorithms.** `choose(label, category)`
offers every member in one dropdown, so they need the same call shape. That is
why Canny's internal stages sit in `"edge stage"` rather than `"edge"` — they
take three inputs, or require `R32F`, so selecting them from an `"edge"`
dropdown could only ever fail.

### Shared utilities

Common building blocks live in [src/algo_util/](src/algo_util/) rather than
being algorithms themselves. Currently:

- `Histogram` — 256-bin intensity histogram with `Mean()`, `Median()`,
  `StdDev()`, `Percentile()`, plus `OtsuThreshold()`, `TriangleThreshold()`,
  and `IsoDataThreshold()`.
- `WindowStats()` — exact mean/stddev/min/max over a square window.
- `IntegralImage` — summed-area table, making windowed mean and stddev O(1)
  per pixel instead of O(r²). The difference between a usable and an unusable
  local thresholder at large window sizes.
- `MinMaxFilter` — windowed min/max in two separable passes, O(2r) per pixel.
- `PixelBuffer` — unpacks an image to flat floats and packs it back, so a
  spatial filter gets branch-free neighbour access and correct handling of all
  three formats without repeating the format switch. `AtClamped()` gives the
  edge-clamped reads every windowed filter needs at the border, and
  `ValueScale()` reports 255 or 1 so a parameter expressed as a fraction of the
  intensity range means the same thing whatever the source format.

### Photographic adjustments

`basic_adjust` is the standard raw-developer control set — temperature, tint,
exposure, contrast, highlights, shadows, whites, blacks, vibrance, saturation
— in **one** algorithm rather than ten. [scripts/develop.tgl](scripts/develop.tgl)
uses it.

That is against this lab's usual grain, and the reason is measured. Chained GPU
stages are **bandwidth-bound, not compute-bound**: at 36 MP a *trivial*
one-fetch-per-pixel kernel still costs ~50 ms per stage, because each stage
reads and writes a full 144 MB intermediate, while a kernel doing nine times
the arithmetic costs only ~20% more.

| stages | 9 MP | 36 MP |
|---|---|---|
| 1 | 29 ms | 74 ms |
| 4 | 67 ms | 190 ms |
| 12 | 188 ms | 664 ms |

Ten chained adjustments at 36 MP would move ~2.9 GB and cost roughly half a
second per slider nudge. Fused, the image is read once and written once and
every adjustment happens in registers, where it is effectively free.

Dispatches are also **batched**: `Dispatch()` only records, and the command list
is submitted when something actually needs the pixels (a readback, or the end of
a run). Nothing needs them between two GPU stages, so the old submit-and-block
per stage bought nothing. That takes a 12-stage chain from 831 ms to 543 ms —
worth having, but not a substitute for fusing, since the bandwidth cost remains.

Iterative stages are the exception: each pass reads what the previous one wrote
through an SRV, and a UAV barrier does not order that. They flush between
passes. (A state-transition barrier was tried instead and does not work —
transitioning out and back gives the driver nothing to synchronise against.)

Everything is computed in **linear light**: the shader decodes sRGB on read and
re-encodes on write. Exposure is a multiply, and multiplying gamma-encoded
values gives the wrong answer — highlights roll off incorrectly and colours
shift as they brighten.

The line to draw: settled operations that always run together belong fused;
anything worth *experimenting* with stays a separate algorithm, where
`choose()` and the compare panel can reach it.

### Filters

Everything in `Category()=="filter"` is interchangeable, so
[scripts/filters.tgl](scripts/filters.tgl) puts two of them side by side from
dropdowns. Roughly in order of cost, measured at 1 MP in Release:

CPU cost is per megapixel in Release; the GPU column is the measured speedup
where a compute kernel exists.

| Filter | CPU | GPU | Preserves edges | Notes |
|---|---|---|---|---|
| `box_blur` | ~75 ms | 3x | no | Running sum, O(1) in radius. The baseline. |
| `gaussian_blur` | ~140 ms | 5x | no | Separable on the CPU, single-pass on the GPU. |
| `guided_filter` | ~230 ms | — | yes | O(1) in radius, no gradient reversal. Usually the best default. |
| `median_blur` | ~680 ms | — | yes | Sliding 256-bin histogram for 8-bit. The impulse-noise filter. |
| `symmetric_nearest` | ~590 ms | 13x | yes | Cheap, no preferred orientation. |
| `kuwahara` | ~810 ms | 15x | yes | Four quadrants; painterly, slightly blocky. |
| `anisotropic_diffusion` | ~920 ms | 17x | yes | Perona-Malik. Iterative; `k` is the edge/noise boundary. |
| `kuwahara_generalized` | ~2.4 s | — | yes | Papari-Petkov-Campisi; smooth sector weighting. |
| `bilateral` | ~3.1 s | **108x** | yes | The classic; O(r²) and prone to halos. |
| `nonlocal_means` | ~11 s | — | yes | Patch-similarity denoiser. Best on repeating texture; by far the slowest. |

The remaining three resist a straightforward kernel: `median_blur` and
`nonlocal_means` need per-pixel sorting or a large search, and
`kuwahara_generalized` needs its precomputed sector weight maps as a second
input, which the one-input dispatch shape does not yet express.

A GPU path is used only within the radius each kernel declares safe. Beyond
that the stage falls back to the CPU, because a single dispatch doing millions
of fetches per pixel trips the GPU watchdog and takes the whole device down.

Two behaviours worth knowing before tuning, both asserted in the tests:

- **Perona-Malik's `k` runs backwards from a blur radius.** Small `k` preserves
  more, because anything above it counts as an edge. A ±255 impulse is a very
  strong gradient, so small `k` *keeps* impulse noise rather than removing it.
- **`nonlocal_means` is not an impulse filter.** Patch distances around an
  impulse are enormous, so it treats one as structure. It is for broad-band
  noise on repeating texture — scanned paper grain, weave, halftone.
  The equivalent trick for statistics an integral image cannot express: at
  8 MP the direct O(r²) version took over two minutes.
- `SampleValue()` — reads any format as a double, with clamp-to-edge.

### Adding a GPU kernel

An algorithm opts into the GPU by returning HLSL. The framework handles
compilation (cached, so slider drags don't recompile), residency, descriptor
binding, and dispatch:

```cpp
bool HasGPU() const override { return true; }

const char* GpuSource() const override {
    return R"(
Texture2D<float4>   Src : register(t0);
RWTexture2D<float4> Dst : register(u0);
cbuffer Params : register(b0) { uint Width; uint Height; uint LevelBits; };

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    float level = asfloat(LevelBits);
    Dst[tid.xy] = step(level, Src[tid.xy]);
}
)";
}

std::vector<uint32_t> GpuConstants(int iteration) const override {
    const float level = m_level;
    uint32_t bits; std::memcpy(&bits, &level, sizeof(bits));
    return {bits};
}
```

Bindings are fixed by convention: `t0..t3` inputs, `u0..u3` outputs, and `b0`
holding `Width`, `Height`, then whatever `GpuConstants()` returns (floats
bit-cast through `uint`, read back with `asfloat`). Thread groups are 8×8.

**Iterative kernels.** Some schemes cannot be one dispatch: each step must see
the *completed* previous step, and threads within a dispatch have no ordering
relative to each other. Returning more than 1 from `GpuIterations()` makes the
framework run the kernel that many times, ping-ponging between two GPU images
and feeding each pass's output back in:

```cpp
int GpuIterations() const override { return m_iterations; }
```

The algorithm still writes a single pass. The framework picks the starting
buffer so the last pass lands in the real output, allocates the scratch image
once, and caches it across runs. This is what makes Perona-Malik interactive:
its ~17x speedup is on top of running every iteration.

Only single-input, single-output stages may iterate — anything else has no
obvious pair of buffers to alternate between.

A kernel that fails to compile falls back to the CPU rather than failing the
run, with the DXC diagnostics reported — a shader you are mid-way through
writing should degrade to slow-and-correct, not to a blank viewer.

### Verifying a GPU kernel

**Compute → Compare CPU / GPU** runs the pipeline twice, forced to each
backend, and reports max/mean/RMSE difference, the count of pixels beyond
tolerance, both timings, and an amplified diff image showing *where* they
disagree. A kernel that merely looks right is not verified.

---

## Layout

```
src/script/     lexer, parser, AST, interpreter   (knows nothing about algorithms)
src/core/       Image, Param<T>, AlgorithmBase, Registry, Pipeline, worker, compare
src/gpu/        D3D12 device, descriptor heap, DXC, compute dispatch, residency
src/app/        win32 window, ImGui panels, views, file watching
src/algo_util/  shared building blocks (histogram, window stats, integral image)
src/algorithms/ one folder per category, one .cpp per algorithm
```

Built on Dear ImGui (docking branch) + D3D12, and `stb_image`, both vendored
under `third_party/`.

## Design notes

- **Execution is a linear stage list, not a DAG.** Dirty detection is by
  parameter *hash*, so there is no invalidation flag to forget to set.
- **Re-parsing happens on every change.** Parsing is microseconds; the cost is
  the algorithms. Results are cached, ASTs are not.
- **Ports carry a `Data` variant**, currently holding only `Image`. Feature
  sets, matrices and point clouds become new alternatives without touching any
  existing algorithm.
- **Parse and interpret stay on the UI thread** (they own the control state);
  the worker runs the pipeline. Slider events coalesce to the newest request,
  so dragging never builds a backlog.
- **The worker owns the stage cache.** `Execute()` moves cached outputs out of
  the previous run, so that pipeline must not be reachable from the UI thread.
- **The compute queue is created on the worker thread**, separate from the
  direct queue ImGui submits on. Sharing one queue across threads is the
  classic source of intermittent corruption.
- **`Image` owns its residency.** Writing on one side invalidates the other,
  and a read transfers only when stale — which is what keeps a chain of GPU
  stages free of intermediate transfers.
- **Views render to offscreen targets** behind a `View` interface, so a 3D
  viewport (SfM, gaussian splats) can dock alongside 2D panels later.

### Known limitations

- **Display still round-trips through the CPU.** A viewer calls `MapCpuRead()`
  and re-uploads as a display texture; rendering straight from the compute
  output's SRV would remove that. The residency API already supports it.
- **One dispatch per stage**, so genuinely multi-pass GPU algorithms either
  fuse into one pass or stay on the CPU.

## Licence

MIT — see [LICENSE](LICENSE). Dear ImGui is MIT; `stb` is public domain / MIT.
