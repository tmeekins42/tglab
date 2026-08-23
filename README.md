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

### GPU validation

Tools for the classes of GPU bug that ordinary testing does not see: a binding
that is wrong when the command list *executes* rather than when it records, and
a resource allocated per run and never freed.

```sh
./build/Debug/gpu_audit.exe            # every GPU algorithm under full validation
./build/Debug/gpu_audit.exe --verbose  # print the text of each distinct message
./build/Debug/gpu_audit.exe --strict   # include warnings inherent to the design
./build/Release/gpu_leak.exe           # VRAM growth across repeated runs
```

`gpu_audit` runs every registered GPU-capable algorithm through the real script
and pipeline layers with the D3D12 debug layer and **GPU-based validation**
enabled, then attributes each validation message to the algorithm that provoked
it. GBV patches shaders to check descriptors and resource state on the GPU, so
it catches faults the CPU-side layer cannot see. It is 10–100× slower, which is
why this is a separate tool rather than part of `ctest`.

`gpu_leak` re-runs one algorithm's pipeline many times, varying a parameter
each run exactly as a slider drag does, and reports whether video memory is
still climbing after warm-up:

```sh
./build/Release/gpu_leak.exe                       # threshold_sauvola, 40 runs
./build/Release/gpu_leak.exe gaussian_blur 60      # any GPU algorithm
```

It compares the two halves of the run rather than first-to-last, because the
first few runs legitimately allocate — kernels compile, outputs and scratch are
created — and a two-point comparison reads that warm-up as a leak.

This caught a real one: `GpuImage` had a manual `Release()` and no destructor,
so a `shared_ptr<GpuImage>` freed the wrapper and leaked the texture. The
iterative scratch is held exactly that way, so every slider tick leaked a
full-size texture — 16 MB a move for an RGBA32F scratch at 1024×1024, climbing
until the card ran out.

The app itself takes `TGLAB_GBV=1` to enable the same validation, and dumps DRED
auto-breadcrumbs (naming the command list and the exact op that stalled) on any
device removal.

`--strict` also reports "Incompatible texture barrier layout" on UAV bindings.
That one is inherent: a resource created with `ALLOW_SIMULTANEOUS_ACCESS` is
permanently in `LAYOUT_COMMON`, so every UAV binding reads as a layout mismatch.
That flag is what lets the UI sample a compute result with no cross-queue
transition, so the warning is the price of the zero-readback display path.

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
sigma = slider("blur sigma", 0.0, 8.0, 1.0)

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
  spatial filter gets branch-free neighbour access and correct handling of every
  format without repeating the format switch. `AtClamped()` gives the
  edge-clamped reads every windowed filter needs at the border, and
  `ValueScale()` reports 255 or 1 so a parameter expressed as a fraction of the
  intensity range means the same thing whatever the source format.

  Its switch names every format and **complains about an unknown one** rather
  than falling through to RGBA8. That switch sits behind thirteen algorithms, so
  it is the single place a newly added format would otherwise be misread by all
  of them at once — and 16-bit data reinterpreted as bytes looks like noise, not
  like a missing case.

  The one algorithm that did not go through it, `brightness`, handled RGBA8 and
  R32F by hand and silently produced **zeros** for anything else: a demosaiced
  raw came out black, with no error anywhere. Worth stating as a rule — an `if`
  chain that handles the formats it knows and does nothing otherwise looks
  correct until a new format arrives, then fails in the way that is hardest to
  trace.

`ComputeContext::BuildHistogram()` is the same thing on the GPU, used by the
info panel. Two dispatches — an order-preserving float-to-uint min/max, then
atomic binning — and only the bins come back: 4 KB rather than the whole image.
It is checked bin-for-bin against `Histogram` for all three formats.

**A follow-up worth taking:** `threshold_otsu`, `threshold_triangle` and
`threshold_isodata` have no GPU path today, and the histogram is precisely why.
Plain `threshold` has a kernel because its level is a parameter passed as a
constant; the automatic variants must *derive* the level from a histogram
first, which pins the whole algorithm to the CPU and drags a full-resolution
image back across the bus to pick a single number.

With `BuildHistogram()` they become: build on the GPU, read back 4 KB, run
Otsu on those 256 bins (microseconds), then dispatch the *existing* threshold
kernel with the level as a constant. Two things to settle first, which is why
it is not done here: the display histogram subsamples to ~262k pixels, and
whether Otsu on a sample picks the same level as Otsu on every pixel needs
measuring rather than assuming (the stride is already a shader constant, so
forcing it to 1 is a one-line change with a cost). And the level has to reach
the CPU between the two dispatches, so such a stage cannot batch with its
neighbours the way a pure kernel does.

`median_blur` also uses a histogram but gains nothing here: it maintains a
sliding window histogram per pixel, which is a different algorithm from binning
one image once.

### Photographic adjustments

`basic_adjust` is the standard raw-developer control set — kelvin, tint,
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

Everything is computed in **linear light**. A gamma-encoded image (a JPEG) is
decoded on read and re-encoded on write; a demosaiced raw is *already* linear
and is passed through untouched. Exposure is a multiply, and multiplying
gamma-encoded values gives the wrong answer — highlights roll off incorrectly
and colours shift as they brighten.

The distinction matters more than it sounds. Applying the sRGB decode to
already-linear data is simply wrong, and the encode on the way out clamps to
1.0 — which discards exactly the highlight headroom that made shooting raw
worthwhile. Measured on a CR2, a peak of 1.77 came back as 0.9995. Images
carry a `linear` flag, set by the pipeline when a demosaic widens a mosaic to
RGB, and tonal algorithms branch on it.

The tonal *bands* have to move with it too. `highlights` targets a band that
ran to a hardcoded 1.0, which is the definitional maximum for gamma-encoded
data but not for a raw: measured across six files from two bodies, peak
luminance after demosaic ran 0.25 to 0.75, so on every one of them the band
never engaged and the slider did nothing. The band now tops out at 0.70 for
linear input.

Two things that fell out of measuring this, both counter-intuitive:

- A single saturated *channel* is not a bright pixel. One frame peaks at 1.82
  in blue while its luminance never exceeds 0.25 — a colour clip, not a blown
  highlight, and a luminance-driven band correctly leaves it alone.
- The band edge cannot be derived from the camera. Three frames sharing
  identical white balance and colour matrix peaked at 0.39, 0.86 and 1.06: the
  peak is a property of the scene. Nor can it be the image's own peak, which
  would make the same slider position mean something different on every shot.

#### Point operations on the GPU

`brightness` and `grayscale` are one fetch and a few instructions per pixel, so
they belong on the device even though they are the two simplest algorithms here.
At 21 MP, measured:

| | CPU | GPU |
|---|---|---|
| `brightness` | 847 ms | 91 ms |
| `grayscale` | 839 ms | 92 ms |
| chained, as `hello.tgl` runs them | 1854 ms | **209 ms** |

The chained figure is the one that matters: the intermediate never leaves the
device, so two stages cost barely more than one.

**The units differ between the two paths, deliberately.** `brightness` is an
offset expressed as a fraction of the intensity range, so the CPU — which works
in the source's own units — multiplies it by 255 for an RGBA8 image. A UNORM
SRV hands the shader 0..1 whatever the storage format, so the shader must *not*.
Scaling in both places would apply the offset 255 times over, and nothing but
the CPU/GPU agreement test would notice. That test now covers both algorithms;
`brightness` agrees exactly, `grayscale` to within one LSB of rounding.

#### Colour temperature

Two axes, which is what white balance actually has: a colour temperature along
the Planckian locus, and a green/magenta offset perpendicular to it that no
temperature can express.

`kelvin` names the illuminant the scene was under, and setting it to the actual
light neutralises the cast — a tungsten photograph needs ~2800 there, not a
cooler number. It **opens at the temperature the camera chose**, recovered from
the file, so the control describes this photograph before anyone touches it.
`tint` opens at the offset the camera applied alongside.

A third relative "temperature" nudge existed briefly and was removed. Once
kelvin could start at the camera's own value, a small warm/cool adjustment was
just a small move in Kelvin — and having two controls for one axis meant neither
said clearly what it did.

Both directions come from the same two references, both in the raw file: the
gains the camera chose for this shot (`cam_mul`) and the camera's own daylight
reference (`pre_mul`). The demosaic has already applied the first, so a request
is applied *relative* to it — undo the camera's choice, apply the asked-for
illuminant — and the two cancel exactly when they agree. That cancellation is
what makes the number mean something rather than being an arbitrary curve, and
it is asserted in the tests.

A JPEG has no such record, so `kelvin` does nothing there rather than guessing.
`tint` still works: a green/magenta push does not need to know where neutral was.

Why a relative control was not enough on its own: it scaled red and blue by at
most ±40%, while a tungsten frame measured here sits a factor of **0.65 in red
and 1.67 in blue** from daylight — outside its reach entirely, which is why a
warm image could not be brought back to neutral however far the slider went.

Three things worth recording from building it:

- The chromaticity comes from Kim et al.'s cubic fit to the Planckian locus. It
  matches Illuminant A to within 0.0005 in xy. D50 and D65 sit about 0.006
  *below* the locus, which is correct rather than error — daylight illuminants
  are not black bodies.
- Below about 1900 K the locus leaves the sRGB gamut and the blue component goes
  negative (measured: −0.036 at 1700 K). Clamping it to a small epsilon produced
  a blue gain of **32×** at 2000 K and broke monotonicity, so dragging the
  slider one way moved the colour back the other. The range starts at 2000 K.
- **The two axes are not independent.** Tint shifts `y`, which moves the R/B
  ratio as well as the green level — measured, tint +0.6 changes R/B by 12% at
  4000 K. Solving for temperature and then for tint therefore does not land on
  the right pair; doing so left the round trip 8% off. Recovering the camera's
  choice alternates between the two until they settle, which brings it inside
  1.5%.

The line to draw: settled operations that always run together belong fused;
anything worth *experimenting* with stays a separate algorithm, where
`choose()` and the compare panel can reach it.

### Getting pixels onto the screen

Optimising the pipeline is only half the latency. Dragging a slider on a 21 MP
raw felt like about half a second while the status bar reported 53 ms, and the
gap was entirely in the display path — which the status bar does not time.

Measured per viewer, at 5634×3752:

| | before | after |
|---|---|---|
| readback (`Clone()` on a GPU-resident image) | 86 ms | 86 ms |
| RGBA16F → RGBA8 conversion | 323 ms | 47 ms |
| **per viewer, per frame** | **409 ms** | **133 ms** |

`develop.tgl` shows two viewers, so all of it doubled.

The conversion was the surprise — larger than running the pipeline. It is 85
million components per frame, and `HalfToFloat` is an out-of-line call in
another translation unit, so none of it inlined. A half has only 65536 possible
values, so the whole operation — decode, clamp, scale — collapses into a 64 KB
lookup table and one indexed load. That is the 9× above.

The second half is not making the work faster but *not doing it*. Viewers now
carry a per-viewer content version, bumped only when that viewer's own pixels
changed, which the stage cache already knows from `FirstDirtyStage()`. In
`display(src, ...)` next to a slider-driven stage, the source is unchanged, so
its texture upload is skipped entirely rather than re-converting identical
pixels every frame. A single global version cannot express this: it either
never changes, or changes for everything at once.

The worker still *sends* unchanged viewers rather than omitting them. It holds
one outcome and a newer run overwrites an unfetched one, so "the UI already has
this" is not something that side can know — and once an image is CPU-resident
the clone is a memcpy, since the expensive parts were the readback and the
conversion, both of which the version check skips.

The readback is now gone too, which removes the last of it. The compute context
and ImGui share one `ID3D12Device`, and compute images already carried
`ALLOW_SIMULTANEOUS_ACCESS`, so the UI's direct queue can sample the worker's
texture while the compute queue owns it — no copy and no cross-queue
transition. The header said this was the plan in M3; it was simply never wired
through to the viewers.

So a result that is still on the device is handed to the UI as a reference
rather than a copy, and the display conversion becomes a compute dispatch that
writes the RGBA8 texture ImGui samples. Per viewer, per frame, at 5634×3752:

| | originally | after the table | now |
|---|---|---|---|
| readback | 86 ms | 86 ms | **0** |
| RGBA8 conversion | 323 ms | 47 ms | **0** (a dispatch) |
| histogram | (rode on the readback) | | ~1 ms |

Lifetime is the part that needs care. The worker frees a stage's outputs on its
own thread whenever the cache is replaced, possibly while the UI is still
drawing the previous frame from one — so the shared texture holds its own
reference rather than relying on the device's deferred-release list, which is a
UI-thread facility.

Two things still need real CPU pixels, and both are handled rather than
regressed. The **histogram** is measured on the GPU (above) and only its bins
come back. The **1:1 loupe** genuinely needs the numbers — it point-samples
individual pixels and prints their values — so it reads back just its own 15×15
window, a few kilobytes, and only while it is switched on. A CPU-only stage has
no shared texture at all and takes the original path, so a script mixing the two
works.

**A trap worth recording:** the conversion binds its own descriptor heap, and
descriptor-heap binding is command-list state rather than per-dispatch. Leaving
it bound made every later ImGui draw reference handles from a heap that was no
longer set — the debug layer says "descriptor heap … is different from currently
set descriptor heap", and the app crashes. Restoring ImGui's heap afterwards is
the whole fix. It only reproduced in Debug; the Release build ran happily.

### Filters

Everything in `Category()=="filter"` is interchangeable, so
[scripts/filters.tgl](scripts/filters.tgl) puts two of them side by side from
dropdowns. Roughly in order of cost, measured at 1 MP in Release:

CPU cost is per megapixel in Release; the GPU column is the measured speedup
where a compute kernel exists.

| Filter | CPU | GPU | Preserves edges | Notes |
|---|---|---|---|---|
| `box_blur` | ~75 ms | 3x | no | Running sum, O(1) in radius. The baseline. |
| `gaussian_blur` | ~140 ms | 5x | no | Separable on both, so a large sigma stays on the GPU. |
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

That applies to the O(r²) kernels — `bilateral`, `kuwahara`, `symmetric_nearest`
— whose weights genuinely do not separate. The two blurs have **no ceiling**:
they run as two separable passes, so radius 60 costs 120 fetches per pixel
rather than 14,600. They previously did have one, at sigma 4, and it meant the
fast path gave up exactly where a blur is most expensive: dragging the sigma
slider past 4 dropped GPU usage to zero and pinned a core. Measured at sigma 12,
the GPU time barely moves (22 → 25 ms) while the CPU goes 111 → 701 ms.

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

**A richer intermediate than the output.** The scratch defaults to the output's
format, which suits a filter whose intermediate is the same kind of thing as
its result. A threshold is where that breaks: its output is a single-channel
`R32F` mask, but the value it must carry between passes is wider — Bernsen
needs a window minimum *and* maximum, the Niblack family a running sum, a sum
of squares, and a count. `GpuScratchFormat()` declares that independently:

```cpp
FormatSpec GpuScratchFormat() const override { return FormatSpec::RGBA32F; }
```

Without it those extra channels are silently dropped. The first Bernsen kernel
lost its maximum that way and got 50% of pixels wrong with nothing reported,
which is why the CPU/GPU agreement test checks the *mean* difference: a
threshold that flips is wrong by the full range, so a max-difference check
cannot distinguish one bad pixel from half the image.

This is also what made the local thresholds tractable on the GPU at all. Their
windowed mean and stddev come from an integral image on the CPU — a 2D prefix
sum, inherently sequential, and the reason they looked like they needed a
multi-dispatch scan. But the window is a *rectangle*, and a rectangular sum
separates: horizontal pass, then vertical, carrying the moments through the
scratch. Measured at 9.4 MP with a 51-pixel window: `threshold_bernsen` 16.3x,
`threshold_adaptive_mean` 5.3x, `threshold_sauvola` 1.9x, `threshold_niblack`
1.5x. The two smaller numbers are honest — an integral image is O(1) per pixel
whatever the window, so the CPU is genuinely hard to beat there, and at the
1024px test size `niblack` is slightly *slower* on the GPU.

A scratch format that differs from the output is only coherent with exactly two
passes: intermediate in scratch, result in output. More would ping-pong the
rich intermediate back into the narrow output and lose it, so the framework
refuses rather than silently truncating.

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
- **A GPU result is never read back to be displayed.** The worker hands the UI a
  refcounted reference to the compute texture and the display conversion is a
  dispatch, so the pixels stay on the device from demosaic to screen. The
  histogram is measured there too. The only thing that still pulls pixels across
  is the 1:1 loupe, which needs the actual numbers and fetches just its own
  15×15 window.
- **Views render to offscreen targets** behind a `View` interface, so a 3D
  viewport (SfM, gaussian splats) can dock alongside 2D panels later.

### Known limitations

- **Intermediates are never freed until the pipeline is replaced.** A 12-stage
  45 MP RGBA16F chain holds roughly 8 GB of them, and nothing collects one that
  no later stage or viewer reads. The stage cache is what makes dragging a
  slider cheap, so collecting eagerly would trade one problem for another; the
  likely shape is a setting that engages when VRAM is tight, which the status
  bar already reports.
- **A GPU stage binds one input.** The root signature has four SRV slots, but no
  multi-input algorithm has a kernel yet — `non_max_suppression` (gx, gy, mag)
  and `kuwahara_generalized` (image plus precomputed sector weights) are the two
  that want one.
- **The automatic thresholds have no GPU path**, because deriving a level from a
  histogram pins the whole algorithm to the CPU. `BuildHistogram()` now makes
  this tractable; see *Shared utilities* for what still needs settling.
- **Tonal control ranges do not follow the pixel format.** `threshold`'s level
  is fixed at 0..255, so on a float image everything falls below it and the
  result is black. Correct for that level, but the level should track the
  format.

## Licence

MIT — see [LICENSE](LICENSE). Dear ImGui is MIT; `stb` is public domain / MIT.

**LibRaw is LGPL-2.1 or CDDL-1.0** and is linked statically, so it carries
obligations the permissive licences do not: the notice must be reproduced, and
recipients must be able to relink against a modified LibRaw. Its complete
source is vendored under [third_party/libraw/](third_party/libraw/) with both
licence texts alongside it, and building tglab rebuilds it from that source —
which satisfies both. Anyone distributing a compiled tglab needs to carry the
same source and notice.

`Help → About tglab` shows the version, every dependency with its licence and
copyright, and the LibRaw notice. Adding a dependency means adding an entry to
`kLibraries` in [src/app/about.cpp](src/app/about.cpp) — the versions there are
read from each library's own headers rather than typed in, so they cannot drift.
