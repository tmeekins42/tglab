# tglab

A lab bench for computer-vision / computer-graphics **algorithm research**.

Drop images in, write a short script that wires algorithms together, expose the
interesting parameters as sliders, and watch results change as you drag them.
Algorithms are written in C++, optionally with an HLSL compute kernel. The
scripting language is deliberately *glue only* — it never contains algorithm
logic.

Adding an experiment costs one `.cpp` file and a few lines of script.

---

## Install

Prebuilt Windows archives are on the [releases
page](https://github.com/tmeekins42/tglab/releases) — unzip anywhere and run
`tglab.exe`. Keep the folder together: the app looks for `scripts/` and
`assets/` beside the executable, and the two DXC DLLs are what compile the GPU
shaders at run time.

Needs 64-bit Windows 10/11, a Direct3D 12 GPU, and the [Visual C++
Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) if the app
reports a missing `VCRUNTIME140.dll`.

### Building from source

Requires Visual Studio 2022 with the Windows SDK. CMake and Ninja ship with VS.

```sh
git clone --recursive https://github.com/tmeekins42/tglab.git
cd tglab
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

**Build Release for real work.** Debug is roughly 30x slower — on a
multi-megapixel image that is the difference between a second and a minute,
which looks exactly like a hang. Build Debug only when you want the debugger or
the D3D12 validation layer.

`cmake --build build --config Release --target package_release` produces
`build/dist/tglab-<version>-win64.zip`.

---

## Run

```sh
./build/Release/tglab.exe scripts/hdr.tgl
```

Arguments are a `.tgl` script plus any images to preload; both are optional.
Images can be **dragged in from Explorer** at any time — hold **Alt** while
dropping several to make them a group.

| Panel | What it does |
|---|---|
| **Images** | The palette. `image("name")` refers to these by filename without extension. |
| **Controls** | Sliders and dropdowns the script declared. |
| **Status** | Errors, run time, CPU/GPU split, and how many stages were skipped. |
| **Image Info** | From the View menu: size, RGB histogram, EXIF capture settings. |
| **Algorithm reference** | From the View menu: every registered algorithm, with ports and parameters. |
| **Compare CPU / GPU** | From the Compute menu — runs both paths and diffs them. |

The **script picker** is on the menu bar: it lists every `.tgl` beside the
current one, so switching is one click.

Editing the script and saving re-runs it (or press **F5**). Slider values
survive the reload, so tuning is not lost when you edit. A parse error leaves
the last good result on screen and reports the line in **Status**.

Double-click any control to restore its default, or use **Reset all**.

### The image palette

Each row shows a thumbnail, the **name scripts refer to**, and the file behind
it. Those last two are deliberately separate:

- **Drop a file onto an existing row** to swap what backs it while keeping the
  name. `image("test")` keeps working with the new file — the fast way to run
  one script over a series of images.
- **Drop onto empty space** to add a new entry, named after the file.
- **Alt-drop several files** to add them as one group.
- **Double-click a name** to rename; **right-click a row** for rename, reload,
  remove, and (on a group) sort by filename.

Names are the filename without extension and are case-sensitive, so
`IMG_2369.jpg` is `image("IMG_2369")`.

A group holds several images in one slot and expands to show its members.
Collapsed, it draws as a stack. That is what feeds the merge algorithms.

### Camera raw

`.CR3`, `.CR2`, `.NEF`, `.ARW`, `.DNG` and the other common formats load
through LibRaw. `image("name")` gives you a **demosaiced** image — the
hot-pixel repair and demosaic are inserted automatically, so a script works
unchanged whether a PNG or a CR3 is dropped on the slot. `mosaic("name")` gives
the raw sensor data instead, for writing a demosaic.

Raw thumbnails are developed for the preview (white balance, auto-exposure,
sRGB), since a sensor mosaic drawn as-is is a dark grey smear.

### While it runs

Algorithms run on a worker thread, so the UI stays responsive however long a
run takes. **Status** shows elapsed time, the CPU/GPU split, and what stage is
running.

Changing anything mid-run **abandons** the run in progress rather than queueing
behind it, so the wait is always for the current settings. The finished part of
the pipeline is kept, so only the stages after your change re-run.

---

## Scripting

A script is a sequence of statements. Comments run from `#` to end of line.
There is no control flow and no user-defined functions — the language exists to
wire algorithms together, not to compute.

### Calling algorithms

Any registered algorithm is callable by name. Named arguments set parameters:

```
blurred = gaussian_blur(src, sigma = 2.0)
```

Multi-output algorithms bind positionally, and `_` discards what you don't need:

```
gx, gy, mag = sobel(blurred)     # gx/gy are signed R32F
_,  _,  mag = sobel(blurred)     # magnitude only
out         = sobel(blurred)     # a single target takes the first output
```

### Pipes

`=>` passes the value on its left as the first argument of the call on its
right, so a chain reads in the order the work happens:

```
src => gaussian_blur(sigma = 2) => grayscale() => display("edges")
```

is exactly `display(grayscale(gaussian_blur(src, sigma = 2)), "edges")`.

Both line-break styles work, and `display()` hands its image back, so it can
sit mid-chain as a tap:

```
out = src
    => gaussian_blur(sigma = 2) => display("blurred")
    => grayscale()
```

A pipe carries one value, so a multi-output algorithm ends the chain — mix the
two forms:

```
gx, gy, mag = sobel(src => gaussian_blur(sigma = 2))
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
| `choose("label", opts, default)` | A third argument names which option starts selected. |
| `params(algo)` | Declares a control for each of `algo`'s parameters; returns `algo`. |
| `params(algo, "name")` | The same, as an independent instance, so one algorithm can appear twice. |
| `shape(group, axis=n, ...)` | Gives a group named axes (see *Groups*). |
| `display(data)` / `display(data, "name")` | Opens a viewer panel; returns its input. |

### Choosing algorithms at run time

`choose()` returns an algorithm as a value, which can then be called:

```
op  = choose("operator", "threshold")   # dropdown of every threshold method
out = op(src)
```

Passing a *category* means a newly written algorithm declaring that category
appears in the dropdown with no script edit.

### Automatic parameter controls

`params()` declares a control for every parameter an algorithm has, using its
own name, range, and default:

```
op   = choose("method", "threshold")
mask = params(op)(src)
```

Switching the dropdown swaps the whole control set, because controls a run does
not re-declare are dropped. One script serves a whole category, no conditionals.

Explicit arguments still win over `params()`:

```
mask = params(op)(src, window = 31)     # window is fixed, the rest are sliders
```

A second argument names the control set, which lets the *same* algorithm appear
twice with independent settings:

```
outA = params(a, "A")(src)
outB = params(b, "B")(src)
```

### Turning stages off

Every algorithm has an **`enabled`** checkbox, first in its group. Unchecking it
skips the stage entirely — no allocation, no dispatch, no copy — and passes the
image straight through, **keeping every other setting** so switching back
returns to what you had.

A stage whose settings would change nothing is skipped the same way
automatically: `gaussian_blur` at sigma 0, `brightness` at 0 / gain 1,
`wavelet_denoise` at both thresholds 0. **Status** reports how many stages were
skipped.

This is what makes a long stack practical — twenty effects with three in use
allocates three intermediates, not twenty.

### Groups

A palette group is several images in one slot. Most algorithms know nothing
about groups: hand one to a single-image algorithm and the framework runs it on
every frame ("broadcasting").

A **merge** reduces a group to one image. `over=` names the axis:

```
frames = image("group")
merged = merge_hdr(align(frames))
```

`shape()` gives a group named axes, so a shoot of 2 positions x 3 exposures can
be reduced one axis at a time:

```
shot   = shape(frames, position=2, exposure=3)
merged = merge_mean(shot, over="exposure")   # -> [position=2]
final  = merge_mean(merged, over="position") # -> one image
```

A chain of broadcast stages feeding a merge runs **one frame at a time** rather
than materialising every intermediate, which is what keeps a seven-frame 45 MP
bracket inside VRAM.

### Example

```
src = image("test")

op    = choose("method", "threshold")
sigma = slider("blur sigma", 0.0, 8.0, 1.0)

src => gaussian_blur(sigma = sigma) => display("blurred")
    => params(op)()                 => display("mask")
```

More in [scripts/](scripts/) — `hdr.tgl`, `stack.tgl`, `tonemap_compare.tgl`
and `pipe.tgl` are good starting points.

---

## Algorithms

39 registered. `choose("x", "category")` offers every algorithm in a category,
so these names are the ones that matter in a script.

| Category | Algorithms |
|---|---|
| **adjust** | `basic_adjust` (exposure, contrast, highlights, shadows, whites, blacks, vibrance, saturation, white balance), `brightness` |
| **tonemap** | `tonemap` (global), `tonemap_local` (illumination/detail split) |
| **merge** | `merge_hdr`, `merge_mean`, `align`, `reshape` |
| **demosaic** | `demosaic_ahd`, `demosaic_consistent`, `demosaic_malvar`, `demosaic_bilinear`, `demosaic_passthrough`, `hot_pixel_repair` |
| **denoise** | `wavelet_denoise` |
| **filter** | `gaussian_blur`, `box_blur`, `median_blur`, `bilateral`, `guided_filter`, `nonlocal_means`, `anisotropic_diffusion`, `kuwahara`, `kuwahara_generalized`, `symmetric_nearest` |
| **edge** | `sobel`, `canny`, `non_max_suppression`, `hysteresis` |
| **threshold** | `threshold`, `threshold_otsu`, `threshold_isodata`, `threshold_triangle`, `threshold_adaptive_mean`, `threshold_adaptive_gaussian`, `threshold_niblack`, `threshold_sauvola`, `threshold_bernsen` |
| **color** | `grayscale` |

Most have a GPU kernel and fall back to the CPU if it fails — **always saying
so** in Status rather than merely running slower.

### Notes on a few

- **`align`** solves each frame of a group against a reference and attaches the
  transform as a sidecar; it warps nothing itself. Merges sample through it if
  present. A correction below half a pixel is skipped when there is no other
  transform to fold into, because the resample costs more than it gains.
- **`merge_hdr`** reads shutter, aperture and ISO from EXIF and divides them
  out, producing scene-linear radiance with real headroom.
- **`tonemap` vs `tonemap_local`** — the global operator applies one curve to
  every pixel: no halos, but it cannot give the sky more of the display without
  taking the same from the land. The local one splits illumination from detail
  and compresses only the illumination. Watch strong edges for halos.
- **`demosaic_consistent`** recovers detail by checking the reconstruction
  against the samples the sensor actually took, rather than sharpening.

---

## Tests

```sh
./build/Release/tglab_tests.exe          # language and pipeline semantics
./build/Release/tglab_filter_tests.exe   # filter behaviour
./build/Release/tglab_demosaic_tests.exe # demosaic reconstruction, CFA phase, colour
./build/Release/tglab_denoise_tests.exe  # denoise
./build/Release/tglab_tone_tests.exe     # tone curve
./build/Release/tglab_autodev_tests.exe  # auto-exposure measurement
./build/Release/tglab_runtime_tests.exe  # worker thread, shaders, GPU
```

Or `ctest --test-dir build -C Release` for all of them. `build.ps1` configures,
builds and tests in one step.

---

## Extending

Writing an algorithm, adding a GPU kernel, and the shared utilities available
to both are documented in [CONTRIBUTING.md](CONTRIBUTING.md). The short
version: one `.cpp` under the matching folder in
[src/algorithms/](src/algorithms/), nothing else edited — the build globs that
tree and registration is automatic.

```
src/
  script/      lexer, parser, interpreter   (knows nothing about algorithms)
  core/        Image, Param, AlgorithmBase, Registry, Pipeline, worker
  gpu/         D3D12 device, descriptor heap, DXC, compute dispatch
  app/         win32 window, ImGui panels, drag-drop
  algorithms/  one .cpp per algorithm
```

---

## Known limitations

- **Intermediates are not freed until the pipeline is replaced.** A 12-stage
  45 MP RGBA16F chain holds roughly 8 GB. The stage cache is what makes
  dragging a slider cheap, so collecting eagerly would trade one problem for
  another.
- **A GPU stage binds one input.** The root signature has four SRV slots, but
  no multi-input algorithm has a kernel yet.
- **The automatic thresholds have no GPU path**, because deriving a level from
  a histogram pins the algorithm to the CPU.
- **Tonal control ranges do not follow the pixel format.** `threshold`'s level
  is fixed at 0..255, so on a float image everything falls below it.

---

## Licence

MIT — see [LICENSE](LICENSE). Dear ImGui is MIT; `stb` is public domain / MIT.

**LibRaw is LGPL-2.1 or CDDL-1.0** and is linked statically, so it carries
obligations the permissive licences do not: the notice must be reproduced, and
recipients must be able to relink against a modified LibRaw. Its complete
source is vendored under [third_party/libraw/](third_party/libraw/) with both
licence texts, and building tglab rebuilds it from that source — which
satisfies both. Anyone distributing a compiled tglab needs to carry the same
source and notice.

`Help → About tglab` shows the version and every dependency with its licence.
