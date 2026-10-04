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
A group decoded from a video expands to a player instead, with Play/Pause,
Stop and a scrubber over the frames that were kept.

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

### Video memory

A 45 MP raw is ~358 MB in RGBA16F, and every palette image a pipeline touches
gains a GPU copy on first use. Half a dozen loaded frames is two gigabytes
before any intermediate exists, and past the card's budget the driver starts
paging — which presents as everything being slow for no visible reason.

So the app reclaims. When usage crosses **80%** of the driver's reported budget
it frees GPU copies until back under **65%**, picking **least recently used**
first. Two thresholds rather than one, or freeing a single texture would drop
usage just below the line and the next allocation would cross it again, with
the collector running every frame while the picture re-uploads.

What it frees is only ever a *cache*: an image resident on both the CPU and the
GPU keeps its pixels in system memory, so releasing the texture loses nothing
and the next use uploads again. An image resident **only** on the GPU — a stage
output written and never read back — is refused, because its pixels exist
nowhere else and freeing it would destroy them silently.

Collection runs on the UI thread and only while the worker is idle. The
palette's sources are shared with the worker, which touches their residency as
it runs, so freeing one mid-run would be a use-after-free rather than merely a
wrong picture. **Status** reports what was freed, since the cost of a
collection is a re-upload on the next use.

### Previews while you drag

A 45 MP frame cannot be re-processed at 60 Hz, so while a slider is moving the
pipeline computes **less of the image** rather than a worse version of it. Two
independent reductions, both dropped the moment you let go — the settled result
is always the full frame at full resolution.

- **Scale.** The run drops to roughly the resolution you are actually looking
  at. Fitted in a window, that is a small fraction of the pixels; the status bar
  says which (`preview at 15% scale (2% of the pixels)`).
- **Region.** Zoomed in, scale saves nothing — at 1:1 the correct scale is 1.0 —
  but most of the frame is off screen, so the run is cropped to what the viewers
  can see. The crop is the **union across every visible viewer**: a second panel
  showing the whole frame means the whole frame is needed.

Algorithms declare how they behave under this, and the declarations are
audited by the test suite rather than inferred:

- **Pixel-unit parameters scale with the image.** A blur of sigma 20 runs at
  sigma 5 on a quarter-scale proxy, or the preview would not resemble the
  result. `ImageDesc::proxyScale` carries the factor and `ScaledPx`/
  `ScaledRadius` apply it.
- **Some algorithms must not be proxied at all.** Demosaicing is the clear case
  — the CFA pattern *is* the data, and downscaling destroys it. Feature
  detectors and anything carrying sidecar coordinates are the same: nothing
  rescales a keypoint list.
- **Some cannot run on a region.** Anything measuring the whole frame to decide
  something global: a threshold picking a level from the histogram, or `dehaze`
  estimating its airlight. Given a crop they would measure whatever happens to
  be on screen, so they decline the region and run whole.
- **Neighbourhood algorithms get a margin.** A blur reading 20 pixels out needs
  those pixels to exist, so the crop is grown by the summed reach of every stage
  in the dirty range and **trimmed back afterwards** — the margin is context,
  not content. Handing it to the viewer instead makes the picture shift as the
  radius changes.

A region result carries its own placement (`originX/originY`, `fullW/fullH`) so
the viewer can draw a window onto the picture without it jumping to the corner
or changing size.

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
| `slider("label", min, max, default[, "help"])` | Declares a slider; returns its current value. The optional help string becomes its tooltip. |
| `check("label", default)` | Declares a checkbox; returns 0 or 1. |
| `pick("label", ["a", "b"][, default])` | Dropdown of names; returns the selected **index**. For a parameter that selects a mode with an integer. |
| `choose("label", [a, b, c])` | Dropdown of algorithms; returns the selected one. |
| `choose("label", "category")` | Same, but offers every algorithm in a category. |
| `choose("label", opts, default)` | A third argument names which option starts selected. |
| `params(algo)` | Declares a control for each of `algo`'s parameters; returns `algo`. |
| `params(algo, "name")` | The same, as an independent instance, so one algorithm can appear twice. |
| `shape(group, axis=n, ...)` | Gives a group named axes (see *Groups*). |
| `display(data)` / `display(data, "name")` | Opens a viewer panel; returns its input. |
| `save(data, "path"[, format=, quality=, existing=])` | Declares an export; returns its input, so it sits in a pipe. |

`include "name.tgl"` is a statement rather than a builtin — see below.

**`save()` writes nothing while you tune.** A save that fired on every slider
tick would fill a directory, so saves are written when you ask — **File → Run
script saves**. For a one-off, right-click an image viewer and choose *Save
image...* instead.

- **Format** comes from the extension, or `format = "jpg"`. PNG and BMP are
  lossless 8-bit, JPEG takes `quality = 1..100`, and `.hdr` keeps linear values
  — the one for a merged bracket. An 8-bit save goes through the viewer's
  display curve, so the file matches the screen.
- **A reconstruction** saves as `.ply`: Gaussian splats in the standard 3DGS
  layout, or coloured points (see *Dense depth and Gaussian splatting*).
- **Existing files** are kept: `existing = "increment"` (the default) writes
  `out_1.png` beside `out.png`; `"overwrite"` and `"skip"` are the others.
- **A group** writes one numbered file per frame: `out_001.png`,
  `out_002.png`, ...

`scripts/save.tgl` shows each of these.

### Sharing a chain between scripts

`include "develop.tgl"` splices another script's statements in where the
include appears. They share one namespace, so the included file reads variables
the caller set and the caller reads what it defined:

```
# _develop.tgl — a fragment, not a script to open
developed = developed
         => params( tonemap_op )()
         => params( basic_adjust, auto_exposure = 1 )()
         => params( vignette )()
```

```
# hdr.tgl
developed = image("group") => params( align )() => params( merge_hdr )()
include "_develop.tgl"
display(developed, "final")
```

That is the point of it: a develop chain written once and used by the HDR and
panorama scripts, which previously each carried their own copy and had already
drifted apart. Shipped fragments are named with a leading underscore, since a
file that calls no `image()` and no `display()` does nothing on its own.

- The path is relative to the **including** file, so a fragment that includes
  another works wherever the pair is dropped. A name with no extension gets
  `.tgl`.
- Included files are **watched for changes** like the main script, so editing a
  shared fragment re-runs whatever includes it.
- Including the same fragment twice is allowed and its **controls are shared** —
  one set of sliders drives both copies.
- A cycle is an error naming the whole chain, not a hang. So is a missing file.
- Errors inside an included file **name that file**, because a bare line number
  would point into whichever script happens to be open.

`include` is recognised by its shape — the word followed by a string — rather
than reserved, so it remains usable as an ordinary variable name.

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

**A stage that cannot pass its input through switches its branch off.** A
stage with several inputs, or one that turns a group into a cloud
(`init_splats`, `plane_sweep`, `solve_cameras`), has nothing to hand on when
unticked. It produces nothing, and **every stage reading it goes off too**, so
one box turns off a whole branch. Their viewers say *off* instead of showing a
result. `sfm.tgl` starts with `init_splats` unticked for exactly this: tune the
solve in seconds, then tick one box to train.

**Effects start switched off.** `orton`, `bloom`, `vignette`, `film_grain` and
`dehaze` are looks rather than corrections, so they do nothing until you tick
their `enabled` box; their control group starts collapsed to match, and opens
if you want to look inside. Corrections — `basic_adjust`, `tonemap`,
`wavelet_denoise` and the rest — start on and doing something sensible.

The distinction is what makes a shared develop chain safe to `include`: a
script picks up the controls for a dozen stages without picking up a glow and a
vignette on every photograph. The categories do not line up with it (`orton` is
a *filter* like `gaussian_blur`; `dehaze` is an *adjust* like `brightness`), so
the list is explicit in the code and audited by name in the tests.

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
and `pipe.tgl` are good starting points; `panorama.tgl`, `features.tgl`,
`matching.tgl`, `align_features.tgl` and `crop.tgl` cover the feature pipeline;
`bloom.tgl` is glow and halation, `vignette.tgl` corner falloff,
`film_grain.tgl` grain with a real grain size, `orton.tgl` the darkroom
sandwich, `lut.tgl` film emulation through a .cube LUT, and
`demosaic_stages.tgl` opens up a demosaic one step at a time. `sfm.tgl` is
the whole 3D chain — frames to cameras, dense depth, Gaussian splats — and
`view_ply.tgl` opens a splat or point cloud made elsewhere.
Each carries its reasoning in comments, including the measurements behind the
defaults.

---

## Algorithms

79 registered. `choose("x", "category")` offers every algorithm in a category,
so these names are the ones that matter in a script.

| Category | Algorithms |
|---|---|
| **adjust** | `basic_adjust` (exposure, contrast, highlights, shadows, whites, blacks, vibrance, saturation, white balance), `brightness`, `crop` (trim and straighten, with a preview), `vignette`, `film_grain`, `dehaze` (dark channel prior), `resize` (area-average on minify, bilinear on magnify) |
| **tonemap** | `tonemap` (global), `tonemap_local` (illumination/detail split) |
| **features** | `detect_sift`, `detect_surf`, `detect_akaze`, `detect_orb`, `detect_brisk`, `detect_dad` and `describe_dedode` (learned; see below), `draw_features`, `draw_matches`, `match_guided` (a second pass along epipolar lines) |
| **match** | `match_brute` (exact), `match_ann` (k-d forest / LSH) |
| **merge** | `merge_hdr`, `merge_mean`, `align`, `align_features`, `bundle_adjust`, `stitch_panorama`, `reshape` |
| **demosaic** | `demosaic_ahd` (default), `demosaic_consistent`, `demosaic_malvar`, `demosaic_ppg`, `demosaic_vng`, `demosaic_bilinear`, `demosaic_passthrough`, `demosaic_stages` (every intermediate as its own image, for debugging), `hot_pixel_repair` |
| **denoise** | `wavelet_denoise` |
| **filter** | `gaussian_blur`, `box_blur`, `median_blur`, `bilateral`, `guided_filter`, `nonlocal_means`, `anisotropic_diffusion`, `kuwahara`, `kuwahara_generalized`, `symmetric_nearest`, `bloom` (glow and halation), `orton` |
| **edge** | `sobel`, `canny`, `non_max_suppression`, `hysteresis` |
| **threshold** | `threshold`, `threshold_otsu`, `threshold_isodata`, `threshold_triangle`, `threshold_adaptive_mean`, `threshold_adaptive_gaussian`, `threshold_niblack`, `threshold_sauvola`, `threshold_bernsen` |
| **color** | `grayscale`, `apply_lut` (3D .cube LUTs) |
| **sfm** | `solve_cameras` (the whole chain, focal found automatically), `relative_pose`, `build_tracks`, `rotation_average`, `global_position`, `triangulate`, `bundle_adjust_sfm`; dense: `plane_sweep` (depth maps), `fuse_depth`; splats: `init_splats`, `train_splats`, `carve_splats`, `render_splats`, `load_ply` |

Most have a GPU kernel and fall back to the CPU if it fails — **always saying
so** in Status rather than merely running slower. `demosaic_vng` is CPU-only.

**Detectors are a different shape.** A detector's product is a *sidecar* — a
list of keypoints — not an image, so the image-in/image-out GPU path cannot
express one: there is nothing for it to write. They therefore stay CPU
algorithms that offload only their inner loops, taking the device from
`RunCtx::Gpu()` and reading the result back. Force CPU withholds the device, so
the toggle still means what it says.

What each one offloads differs, because their costs differ — measured with
`bench_detect` before writing any kernel:

| detector | offloaded | 22 MP CPU → GPU | |
|---|---|---|---|
| `detect_sift` | Gaussian pyramid (65% of runtime) | 8,937 → 2,332 ms | **3.8×** |
| `detect_orb` | FAST + Harris response map | 5,718 → 2,114 ms | **2.7×** |
| `detect_akaze` | nonlinear diffusion chain | 11,306 → 5,367 ms | **2.1×** |
| `detect_brisk` | FAST + bisected score map | 5,241 → 3,721 ms | **1.4×** |

BRISK gains least, and not for want of trying: roughly half its time is the
512-bit ring descriptor, which is a per-keypoint gather rather than a per-pixel
test and stays on the CPU. `detect_surf` is not converted — its integral image
is a sequential prefix sum, the wrong shape for this dispatch model.

The offloaded stages agree with the CPU exactly: ORB and BRISK are bit-identical,
SIFT and AKAZE within 3.1e-05 (`tglab_pyramid_tests` checks all four, and that
they find the same keypoints either way).

**Verify a kernel against a real file, not a synthetic one.** Compute → Compare
CPU / GPU is the tool for this, and the reason it matters is that `gpu_audit`
runs everything on a synthetic gradient: for a demosaic that exercises neither
clipping nor out-of-gamut colour, so the highlight clamp and the in-gamut solve
never fire. Three shipped demosaicers had shaders applying the bare camera
matrix where their CPU paths used the in-gamut solve — a 0.85 divergence on
exactly the pixels the solve exists for — and the audit called them clean
throughout.

### Features, alignment and panoramas

Five detectors, two matchers, and the stages that turn matches into a stitched
panorama. They connect through **sidecars** rather than ports: a detector
attaches keypoints to the image it was given, a matcher reads those and attaches
match sets, and an aligner reads those. Nothing in the chain has to be told what
came before it.

```
frames => detect_orb()
       => match_brute(chain = 1, window = 2)
       => align_features(model = 2)
       => bundle_adjust()
       => stitch_panorama(projection = 1)
       => display("panorama")
```

**Which detector.** The scale-space three cost 8–10 s per 45 MP frame; the
binary two cost about 3. Measured end to end on one pair of a real sweep,
through matching and the RANSAC solve:

| detector | time | matched | inliers |
|---|---|---|---|
| `detect_orb` | 12.7 s | 11% | 90% |
| `detect_brisk` | 17.0 s | 36% | 91% |
| `detect_sift` | 23.8 s | 52% | 86% |
| `detect_surf` | 26.7 s | 57% | 85% |
| `detect_akaze` | 28.1 s | 21% | 86% |

ORB and BRISK are not merely the fast pair, they produce the *highest* inlier
rates here. A low match count is not a problem when a homography needs four
points and gets hundreds. Prefer ORB for speed, BRISK when frames differ in
scale (it refines scale continuously; ORB can only name a pyramid level).

**All of them are accurate enough for alignment.** Through the full pipeline on
the same 15-frame sweep, the final reprojection RMS is 1.38 px (ORB), 1.36
(BRISK), 1.37 (AKAZE), 1.97 (SIFT), with the focal estimates agreeing within
4%. Detector choice is a speed and match-count decision, not an accuracy one —
so the pairwise inlier rates above are worth reading as "how much work the
solver has to do", not "how good the answer will be".

SURF is **patented** (ETH Zurich) and included anyway for a research tool that
is not being sold — check your position before shipping anything built on it.

**Descriptors travel with the data.** SIFT and SURF produce floats compared by
L2; AKAZE, ORB and BRISK produce bit strings compared by Hamming. A matcher
given the wrong distance still returns matches and they are garbage, so the kind
is carried on the descriptor set and the matcher reads it rather than being
told. Nothing in a script names a descriptor type.

**`chain = 1` is what a panorama needs.** Every other group script matches each
frame against frame 0, which is right for a bracket and wrong for a sweep:
measured on 15 frames, the fraction of candidates kept against frame 0 decayed
45% → 32% → 17% → 5% → 2%, and the last frame failed to solve. Consecutive
neighbours hold 40–94% inliers throughout. `window` extends that to several
predecessors, which is what gives `bundle_adjust` constraints beyond the chain.

**`model = 2` (homography) for a pan.** A rotation about the camera centre is a
homography and nothing simpler. From the same 158 matches on one pair:
similarity 41% inliers, affine 41%, homography 93%. On a tripod bracket the
opposite holds — see `scripts/align_features.tgl`.

**`bundle_adjust` removes what the chain accumulates.** Solving links
independently and composing them in sequence lets each link's residual pile up;
by mid-panorama that is tens of pixels, which reads as a doubled ridgeline when
zoomed in. Bundle adjustment re-solves every rotation simultaneously against
every match. Measured: reprojection RMS 26 px → 1.4 px, and the ghosting goes.

It uses only the matches RANSAC verified. That distinction is worth knowing:
a robust loss down-weights an outlier but does not *remove* it, and cannot
rescue a fit the outliers already moved — the weights come from residuals
against the current estimate, so if that estimate is wrong the wrong points look
right. Feeding it every match left one detector stalled at 25 px where filtering
took it to 1.37.

Three angles per frame plus a shared focal length, so 46 unknowns for 15 frames
— a dense 46×46 normal-equation system, which is why there is no dependency on
a sparse solver library.

**The projection is not decoration.** Chaining homographies multiplies their
perspective terms, so a wide pan on a flat canvas diverges: along one sweep the
width each frame mapped to ran 5796 → 7059 → 12650, and by frame 14 the canvas
wanted 169842 megapixels. `stitch_panorama` extracts the *rotation* from each
transform and projects onto a cylinder (0 plane, 1 cylindrical, 2 spherical).
Same frames, same links, 11372×3912.

**The cylinder turns about the axis the camera actually panned around**
(`straighten`, on). A camera tilted down while it pans also rolls slightly, in
its own view, with every step. A 100-frame phone video of a desk measured
0.2–2° per link, all the same sign, about 70° in all. About frame 0's own
vertical, the far end of that panorama ran off at 45°. The true axis comes from
the frames themselves: panning keeps every camera's sideways axis level, so
vertical is the one direction perpendicular to all of them. This is what
OpenCV calls wave correction and Hugin calls straighten. Pans under about 20°
leave that direction undetermined, so they are left as they were.

Its report gives the focal length it estimated — from the geometry, not from
EXIF, so it needs no sensor size — and how much the frames **disagree** where
they overlap. That last number is the direct measure of stitch quality;
sharpness is not, because sharpness also moves with how much the canvas was
scaled.

**When ghosting survives the bundle, read `bundle_adjust`'s residual-by-band
figures before changing anything.** They split the leftover error into the top,
middle and bottom thirds of the frame, and the two failures they distinguish
want opposite fixes.

Roughly even means the solve is imperfect — more overlap or another detector may
help. **Bottom-heavy means the camera translated**, which it does whenever a pan
pivots around the photographer instead of the lens's entrance pupil. The
leftover displacement is then `f·T/Z`: it grows as things get *closer*, so near
content misaligns while the distance is fine. **No homography can fix that** —
one 8-parameter warp cannot give near and far content different displacements at
once.

For that case, `winner_takes_all` takes each output pixel from a single frame
instead of blending. Feathering *averages* the overlapping frames, and averaging
two offset copies of a branch is exactly what makes the doubled image; choosing
one frame cannot ghost because there is nothing to average. It also avoids the
softening that blending causes even at perfect alignment, since each frame is
independently resampled and the average of two softened copies is softer still.

The trade is that exposure steps stop being hidden. `gain_compensation` solves
one gain per frame from the overlaps, in log space so what gets averaged is a
brightness *ratio*. It constrains the **seam** rather than the whole overlap,
and compares local patches rather than single pixels — two frames sampled at one
canvas point are looking at different scene points wherever registration is
imperfect, and in a high-contrast scene that difference is half a stop for
reasons having nothing to do with exposure. Measured on a 15-frame sweep, the
seam gap went 0.299 → 0.180 stops; patch comparison was worth 3× over
single-pixel.

**A per-frame gain cannot fix a difference that varies across the frame.** A
per-frame *ramp* was built for that and removed: every constraint is a
difference between two frames, so a tilt applied identically to all of them is
invisible to the fit and drifts until it hits its limit. Measured, the ramps
railed with the same sign on all fifteen frames — which tilts the whole panorama
and corrects no seam. The remaining residual is within-frame variation (lens
falloff plus the scene's own gradient), and fixing it needs either a measured
flat field or a seam-aware blend, not a better global fit.

### Learned features

**`sfm.tgl` uses learned features by default.** tglab is building up LoMa
(Edstedt, Nordström et al., *LoMa: Local Feature Matching Revisited*, 2026),
the matcher behind several recent splat solvers: a learned detector, a learned
descriptor and a transformer matcher. Each arrives as its own stage, so it can
be compared with the designed one it replaces. The networks run on tglab's own
compute kernels (`src/algo_util/nn.h`): no inference runtime and no extra DLLs.
Each is checked layer by layer against the published PyTorch model
(`bench_nn`), and agrees to about one part in a million.

- **`detect_dad`** — DaD, LoMa's keypoint detector, trained to pick points that
  are found again in other views. It has no notion of a corner or a blob, and
  it spreads its points far more evenly than AKAZE, which crowds onto the few
  high-contrast spots.
- **`describe_dedode`** — LoMa-B128's DeDoDe descriptor, 128 floats a point.
  It describes *any* detector's keypoints, so `detect_akaze => describe_dedode`
  is a fair comparison of a designed descriptor against a learned one on the
  same points. Unit length by default, for `match_ann`.
- **`match_ann`** searches float descriptors *exactly* on the GPU when there is
  one — every pair compared, the eight nearest kept, then the same ratio test,
  separation rule and cross check as the k-d forest. Exhaustive is faster than
  approximate there, and finds about twice the matches.
- **`match_ann(method = 1)`** — LoMa's learned matcher, a nine-layer
  transformer (LightGlue's design): every point's description is rewritten by
  attention to the rest of its own image, positions included, then to the
  other image, before all are compared at once through a dual softmax. Needs
  `describe_dedode(normalise = 0)`. Matches PyTorch's output exactly on a
  2048-point pair (1561 of 1561), about 55 ms a pair at that size.

```
feat = small => detect_dad(max_features = 10000) => describe_dedode()
```

Measured in `sfm.tgl`'s sparse chain (images at 0.4 scale), AKAZE against the
learned pair:

| | fountain-P11, AKAZE | fountain-P11, learned | castle-P19, AKAZE | castle-P19, learned |
|---|---|---|---|---|
| inliers per pair | 437 | **3998** | 299 | **2044** |
| pairs rejected | 6 of 39 | **0 of 30** | 1 of 70 | **0 of 61** |
| points | 1,998 | **10,131** | 2,175 | **14,450** |
| mean ray residual | 0.171° | **0.035°** | 0.982° | **0.130°** |
| reprojection RMS | 0.80 px | **0.45 px** | 0.78 px | **0.45 px** |
| focal (calibrated 58.2°) | 57.9° | **58.1°** | 57.9° | **58.2°** |

On a hand-held video of a chrome tape measure, the weakest link in the
sequence went from 39 triangulated tracks to 389.

On an RTX 4070 Ti the two networks take about 0.2 s a frame (DaD 55 ms at
1024 px, the descriptor 130 ms at 784), and matching a 169-frame video's 638
pairs takes 20 s. Without a device they run on the CPU, at several seconds a
frame.

**The learned matcher against nearest neighbour**, on the same 4096 DaD
points a frame:

| | points | reprojection | worst rotation edge | matching |
|---|---|---|---|---|
| castle-P19, nearest neighbour | 7,967 | 0.38 px | — | 1.2 s |
| castle-P19, LoMa | **8,765** | 0.44 px | — | 10.9 s |
| IMG_1535 (169 frames), nearest neighbour | **26,297** | 0.83 px | 10.8° | 17 s |
| IMG_1535, LoMa | 25,385 | 0.88 px | **4.8°** | 91 s |

Better pairwise geometry, about as many points, five to ten times the
matching time — so `sfm.tgl` keeps nearest neighbour (at 10,000 points, which
reconstructs more still) and shows LoMa as a one-line swap. Two settings
matter. `threshold` defaults to 0.9 rather than LoMa's own 0.1: SfM chains
matches into tracks, and a low-confidence match breaks one (castle-P19 goes
from 6,227 points at 0.1 to 8,765 at 0.9). `layers = 3` keeps 95% of the
points at half the cost.

The weights are files beside the executable rather than part of it. The
release archive includes them; from source, make them from the published
checkpoints (see [third_party/models](third_party/models/) for licences):

```sh
python tools/nn_convert.py dad models/dad.tgw                                        # 13 MB
python tools/nn_convert.py loma_b128 models/dedode_b128.tgw --prefix _descriptor.    # 27 MB
python tools/nn_convert.py loma_b128 models/loma_b128.tgw --exclude _detector. --exclude _descriptor.   # 24 MB
```

Python and PyTorch are needed for that step only.

### Structure from Motion

Where a panorama recovers camera *rotation*, SfM recovers where the camera
**was** — a full 3D reconstruction with translation and a sparse point cloud.
The chain is longer because every stage solves a genuinely different problem,
and each one is a separate algorithm so it can be swapped:

```
frames => detect_akaze(max_features = 10000)
       => match_ann(chain = 1, window = 2, revisit = 3)
       => relative_pose()                  # essential matrix per pair
       => build_tracks()                   # union-find over the matches
       => rotation_average()               # one orientation per camera
       => global_position()                # where the cameras are
       => triangulate()
       => bundle_adjust_sfm()
       => display("reconstruction")        # a 3D viewport, orbit camera
```

**`solve_cameras` runs that chain for you**, from `relative_pose` to a second
round of triangulation and bundle adjustment, and finds the focal length on
the way (below). `scripts/sfm.tgl` uses it, so a capture reconstructs with
nothing to set; the stages remain separate algorithms for when one is being
compared against an alternative.

The reconstruction is a `PointCloud` rather than an image, so `display()`
routes it to a 3D viewport with its own graphics pipeline — depth testing on,
points drawn as camera-facing quads, mouse-orbit camera. **Drag** orbits,
**right-, middle- or shift-drag** pans (the scene follows the mouse), the
**wheel** zooms toward whatever is under the cursor, and **double-click**
makes the point clicked the centre of the orbit. **Frame** resets the view.

**`relative_pose` has two estimators.** The eight-point algorithm is a linear
solve, and `method = 1` selects Nistér's five-point instead. The difference is
not academic, measured on synthetic fixtures with known ground truth:

| condition | 8-point | 5-point |
|---|---|---|
| general, exact | 0.000° | 0.000° |
| general, 1 px noise, 30% outliers | 1.321° | **0.458°** |
| planar, exact | 6.547° | **0.005°** |
| planar, 0.3 px noise | **4.180°** | 11.434° |

Five-point needs five correspondences per RANSAC sample rather than eight, and
the chance of a clean draw is the inlier rate raised to that power — on a pair
with a 30% inlier rate, roughly 35× as many usable samples. Neither method
should be trusted on a near-planar pair: eight-point is degenerate there, and
five-point is degenerate *in a different way* (the essential matrix is not
unique on a plane, so noise makes the solve pick an arbitrary member of a
one-parameter family). The stage reports a measured planarity per pair so you
can tell when you are in that situation.

**Run triangulate and bundle adjustment twice.** The first triangulation judges
every track against cameras nothing has refined yet, and bundle adjustment then
moves them a long way. Re-running the pair re-judges the rejects against the
refined cameras. Measured on castle-P19, a 19-frame courtyard walk: 1669 points
at a 2.5 px reprojection median after one round, **2168 at 0.9 px** after two.

**The field of view is found, not asked for.** It is the most consequential
number in the chain — a wrong focal biases every recovered rotation, and on a
walk around a subject the errors add up: a video circling a toy cat, solved at
42° where the lens was about 33°, failed to close its circle by 31° and broke
apart. So `relative_pose` *estimates* it from the matches when `fov_deg` is 0
(the default): the right focal lets the most correspondences fit an essential
matrix at a tight threshold, as measured with the five-point solver, which
builds the essential constraints in. Bundle adjustment then *measures* it
against reprojection error, and `solve_cameras` solves again with that until
the two agree. With nothing given:

| capture | estimated | settled at | truth |
|---|---|---|---|
| fountain-P11 | 56.5° | 57.8° | 58.2° (calibrated) |
| castle-P19 | 58.5° | 57.9° | 58.2° (calibrated) |
| cat video | 30.5° | 33.3° | ~33° (where its loop closes) |

`build_tracks` takes the focal `relative_pose` used rather than holding its
own. It once did, and two settings that had to agree were, in practice, set
apart — a capture solved at 36° and positioned at 58° came out as three copies
of its subject.

(This paragraph used to say the stock set converges to **50.0°** from every
starting guess. That agreement was the symptom of a bug, not evidence: camera
positioning stopped far short of converging, and bundle adjustment settled on
whatever focal fit the mis-placed cameras. The fix is in `global_position`,
and so is the measurement that found it.)

**Why not the rotation residual, which looks like the obvious measure.** It falls monotonically as
the assumed field of view rises, all the way to physically impossible values,
because too long a focal shrinks every recovered rotation toward the identity
and near-identity rotations agree with each other trivially. The residual
rewards the collapse. A closed walk-around also gives a free ground truth with
no ground-truth file: the relative rotations should sum to about 360°.

### Dense depth and Gaussian splatting

`scripts/sfm.tgl` carries on past the sparse reconstruction, to a depth map per
frame, a dense point cloud, and a trained set of 3D Gaussians (Kerbl et al.,
SIGGRAPH 2023) that renders like the photographs:

```
dense   = params( plane_sweep, planes = 48 )( cloud, small )    # depth per frame
fused   = params( fuse_depth, min_views = 3 )( cloud, dense )   # dense cloud
splats  = params( init_splats, enabled = 0 )( fused )           # a flat Gaussian per point
trained = params( train_splats, iterations = 1000 )( splats, small, dense )
carved  = params( carve_splats )( trained, dense )              # floaters removed
display( carved, "splats" )
```

A viewer showing a group (the frames, the sweep's maps, `render_splats`'s
renders) has **Play/Pause, Stop and a frame slider** across its top. Viewers
with the same number of frames step together, so "frames" and "render" always
show the same camera side by side. Changing frames re-runs the script, which
is entirely cached and so takes milliseconds. A choice made during a long run
waits for that run to finish rather than cancelling it.

`render_splats` is also **off by default**. It is for checking a fit, not
making one, and on a 100-frame video it took 44 s. Tick its `enabled` to
validate.

The splat branch **starts switched off**. Training takes minutes where the
solve takes seconds, so tune the cameras first, then tick `enabled` under
`init_splats`. Training, carving and rendering all read from it, so they switch
on with it.

**`plane_sweep`** sweeps inverse-depth planes through each frame's neighbours
and scores them by NCC; it emits a depth, confidence and colour map per frame,
which the ordinary image viewers show. On the GPU it is one fused kernel per
plane: 2 s for fountain-P11's eleven frames at script resolution, 18 s at full
resolution. **`fuse_depth`** keeps a depth only where other cameras measured
the same surface (`min_views`), which is the filter that matters — a wrong
depth is wrong in one view's own way.

**Only what the sweep can bracket is measured.** The planes span the sparse
points' depths, plus a few guard planes, so a background far behind the
subject lies past the far end. Its score keeps rising toward that end, and
every view would put it there. Those views then *agree*: on a selfie video a
wall six feet back came through fusion as a shell of points around the head.
So a pixel is measured only when its best score stands `min_bracket` (0.15)
above the scores at both ends of the range. A plain wall that correlates
equally at every depth fails the same test. That cleared the selfie's halo
entirely, and cost fountain-P11 and castle-P19 about a tenth of their dense
points at unchanged accuracy.

**`train_splats`** is the paper's optimiser: a differentiable rasteriser with a
hand-derived backward pass (checked against finite differences, and the GPU
against the CPU), Adam at the paper's rates, and densification. The whole
state lives on the GPU: an iteration on half a million Gaussians is about
50 ms, so the default 1000 iterations take under a minute. The CPU fallback is
about 1 s per iteration.

**Eleven photographs is few, and the defaults are tuned for it.** Training
scores on the photographs it fits, which rewards memorising them; `holdout = 5`
leaves cameras out and scores them separately. Measured on fountain-P11 with
cameras 2 and 7 held out:

| iterations | depth | training | held out |
|---|---|---|---|
| 300 | no | 18.0 dB | 17.1 dB |
| 1000 | no | 25.0 dB | 22.8 dB |
| 1000 | **yes** | 24.4 dB | **24.0 dB** |
| 3000 | no | 31.2 dB | 23.3 dB |
| 3000 | **yes** | 30.7 dB | **24.6 dB** |

Past about 1000 iterations the training score climbs six more decibels while
the held-out one gains half of one — the extra fit is to the photographs
themselves, and shows between them as false colour and smears. What pushes back:

- **Depth supervision** — the sweep's maps as the third input. Rendered depth
  is pulled toward them (`depth_weight`, 0.3), pinning Gaussians to the
  surfaces the cameras measured. Compared against the target scaled by each
  pixel's coverage: comparing with the raw target made partly transparent
  pixels read as too near, and the loss pushed Gaussians *behind* the walls.
- **`carve_splats`** removes Gaussians that other cameras see straight through
  (free-space carving against the depth maps), and large ones too few cameras
  have in view. The first is what clears the floating paint over the blank
  wall at the right of the fountain.
- **Pruning outlives growth.** Oversized Gaussians are still removed after
  densification stops; the paper stops both together, and with eleven views a
  Gaussian edge-on to every camera grew into a sheet across the whole scene.
- **Colour depends on the viewpoint** (`sh_degree`, 3). Each Gaussian's
  colour is its base colour plus a spherical-harmonic expansion in the viewing
  direction, the paper's model, so a sheen or broad highlight can change as the
  camera moves. Before this a Gaussian had one colour from everywhere, and a
  highlight could only be baked in or faked with floaters. Bands switch on one
  at a time (`sh_every`), after the base colour has settled. The paper's
  learning rate for the coefficients is for 30000 iterations and barely moved
  them in 1000, so `sh_lr` uses colour's own. Measured with cameras held out:

  | capture | degree 0 | degree 3 | training time |
  |---|---|---|---|
  | fountain-P11, every 5th held out | 26.03 dB | 27.04 dB | 47 → 69 s |
  | cat video, every 10th held out | 16.58 dB | 16.96 dB | 148 → 208 s |

  The gain holds on the held-out views, so it is real view dependence rather
  than memorising the photographs. It is small on the cat, whose book cover
  reflects sharply: a degree-3 expansion is too smooth to draw a mirror image.
  That is the next step, a reflection model rather than a smoother colour.
  `.ply` export writes the coefficients as the reference `f_rest_*` fields, so
  other splat viewers show the view dependence too. Import keeps theirs.
- **Reflections** (`reflect`, off by default). Deferred reflection after
  3DGS-DR: each Gaussian also has a reflectivity and a normal, the scene a
  learned environment cube map (`env_res` per face), and each pixel mixes its
  colour with the environment seen in the mirror direction,
  `(1 − R)·C + R·Env(reflect(ray, N))`. Switched on at `reflect_from`, once
  geometry and colour have settled. On a synthetic mirror, held out: 17.9 dB
  with harmonics alone, 42.9 with reflections.
  **Reflecting has a price** (`reflect_sparsity`, 0.01), or matte surfaces
  use the environment as spare colour capacity: on fountain-P11 mean
  reflectivity reached 0.60 by 3000 iterations with none of it real, and
  each Gaussian's own colour was pushed ever further from the photograph to
  compensate. With the price it stays at 0.27 and the fit is 0.3 dB *better*;
  see train_splats.cpp for the sweep. Leave `reflect` off for scenes without
  mirrors or gloss — it costs time and gains little there.
  The 3D viewer shades each splat with its reflection rather than each pixel,
  as training does, so a strongly reflective splat's colour there is close
  but not exact.
- **Only the reconstructed part of each photograph is fitted.** A photo shows
  more than the dense cloud holds: the wall behind a face, sky over a
  building. Training used to paint that in with Gaussians stretched around the
  subject, which from any other angle are smears and blobs. Now `mask` (on
  when the depth input is given) fits only the pixels the starting Gaussians
  cover, grown by `mask_grow` pixels to close thin gaps. Everything else
  targets a background colour drawn at random each step, which only
  transparency can match, so leftovers there fade and are pruned.
  `init_splats` also drops isolated points first (`isolated`), since each
  stray would otherwise mark its own patch of background as subject.
  *The cost:* a real surface the sweep could not measure is left out too.
  Castle-P19's cobbled courtyard, seen at a grazing angle, now renders
  empty like its sky. Set `mask = 0` for a scene where that matters more
  than clean edges.
- **Colour is clamped to 0..1.** Unclamped, the optimiser cancelled
  over-bright Gaussians against darker ones in front — right from the training
  views, bright red patches from anywhere else.

**From a video.** Drop an `.mp4` or `.mov` on the palette — onto `sfm.tgl`'s
`group` row to run the chain on it — and it arrives as a group of frames chosen
**by how far the view moves**, decoded by Windows' own Media Foundation and
turned upright if the phone recorded in portrait. Corners are tracked from
frame to frame, and a frame is kept each time the view has moved 4% of the
frame's shorter side, or lost track of too much of what it saw — the sharpest
frame near that point, since blur comes exactly when the camera speeds up. A
fast stretch gives many frames and a slow one few, with at least 60 over the
whole clip; past 250 the set is thinned evenly. Equal time slots, which this
replaced, spread thinner the longer the clip:

| clip | length | 100 time slots | by motion |
|---|---|---|---|
| face, close orbit | 8 s | 100 / 100 | 64 / 64 |
| walk-around | 25 s | 100 / 100 | 191 / 191 |
| walk-around, mirror | 39 s | **45 / 100** | 241 / 241 |
| walk-around | 57 s | **40 / 100** | 245 / 245 |
| walk-around | 59 s | **34 / 100** | 180 / 180 |

(cameras reconstructed / frames kept). The long clips cost more — two to four
times the SfM time of 100 frames — because they need the frames. Nothing needs
setting: the field of view is found, and `match_ann`'s `revisit` also matches
frames that see the same view from far apart in the clip, which is what closes
the loop of a walk-around. For a good
capture: walk slowly *around* the subject rather than turning on the spot,
since sideways movement is the parallax; lock exposure and focus; use plenty
of light so frames are not blurred. iPhones record HEVC by default, which
Windows decodes only with the free *HEVC Video Extensions* from the Microsoft
Store — or set Camera → Formats → *Most Compatible* for H.264.

**`.ply` in and out.** `save( carved, "scene.ply" )` writes the Gaussians in the
standard 3DGS layout that SuperSplat and other splat viewers read; a cloud
without them saves as coloured points for MeshLab, CloudCompare or Blender.
`load_ply( file = "scene.ply" )` reads either back in — including splats from
other tools, which `scripts/view_ply.tgl` opens in the 3D viewer.

### Notes on a few

- **`dehaze`** inverts the atmospheric scattering model, `I = J·t + A·(1−t)`,
  using He's **dark channel prior**: in almost any patch of a haze-free outdoor
  photograph some pixel is nearly black in some channel, so where that local
  minimum is *bright*, the brightness was added by haze. Since transmission
  falls off with distance, the map it recovers is really a rough **depth map**
  — which is why the correction grows toward the horizon by itself.

  Put it **before the tone curve**: it inverts a physical mixing of light, so it
  needs scene-linear values. Applied to display values it lifts the shadows
  instead and the picture washes out.

  `sky_protect` covers the prior's known failure. A large bright region with no
  dark pixel in it — sky, snow, a white wall — looks exactly like haze and comes
  back darkened and grey. Set it to 0 to see the raw prior; over sky that is
  usually the most visible thing the algorithm does.

  The **airlight is bounded at 1.0**, which matters on scene-linear raw. Values
  above white live in the highlights, and an unbounded search lands on a blown
  cloud edge — after which every sky pixel sits *below* the airlight, `I − A` is
  negative, and the recovery drives it negative fastest in whichever channel has
  the widest gap. Clamping the channels independently then removes them at
  different rates and a grey cloud comes out **purple**. An airlight is a colour,
  and a colour brighter than white is not one.

  The GPU path is a deliberate approximation of the CPU one, not a translation
  of it: the transmission map runs at full resolution rather than quarter scale,
  a box smooth replaces the guided filter, and the airlight comes from a strided
  sample of ~300k pixels. Each trade buys interactivity; the two agree to a mean
  of about 0.6 on a 0–255 scale. Because the airlight depends only on the
  *photograph* and not on any slider, it is **cached across a drag** — the
  measurement is the whole remaining cost, and skipping it is what makes the
  sliders live. `patch` is the exception, since it is the window the dark
  channel is minimised over, so changing it re-measures.
- **`align`** solves each frame of a group against a reference and attaches the
  transform as a sidecar; it warps nothing itself. Merges sample through it if
  present. A correction below half a pixel is skipped when there is no other
  transform to fold into, because the resample costs more than it gains.
- **`crop`** trims and straightens in one stage, because they are one
  operation: rotating leaves empty corner wedges that have to be cropped, and
  cropping first puts them back. `preview` draws the crop rectangle on the full
  frame instead of returning the cropped raster — which is what makes it usable
  interactively, since the part being cut away is otherwise off screen the
  moment a slider moves.

  `preview_grid` draws division lines inside the rectangle, and they **rotate
  with the crop** — a horizon is level exactly when it lies along a line, which
  turns straightening into a comparison instead of a guess. Screen-aligned lines
  would keep their own angle while the rectangle turned under them, and matching
  the two would mean nothing.
- **`merge_hdr`** reads shutter, aperture and ISO from EXIF and divides them
  out, producing scene-linear radiance with real headroom.
- **`tonemap` vs `tonemap_local`** — the global operator applies one curve to
  every pixel: no halos, but it cannot give the sky more of the display without
  taking the same from the land. The local one splits illumination from detail
  and compresses only the illumination. Watch strong edges for halos.
- **`demosaic_ahd`** is the default. It interpolates green *along* edges rather
  than across them, so thin high-contrast detail keeps its colour instead of
  breaking into speckles.
- **`demosaic_consistent`** recovers detail by checking the reconstruction
  against the samples the sensor actually took, rather than sharpening. It
  still measures the best of the four on detail, but it steers on luminance and
  so preserves whatever colour its bilinear starting point got wrong — which at
  the steep edge of a highlight is a lot. `demosaic_stages` shows why. It was
  the default until that was measured.
- **`demosaic_ppg`** (Patterned Pixel Grouping) never interpolates red
  directly. It interpolates R−G, which varies far more slowly than R does — a
  colour difference tracks the *material* while the absolute value tracks the
  material *and* the light on it — then adds back the green it already knows.
  That is why it produces so little false colour for how simple it is, and it
  runs 3.4× faster than AHD.
- **`demosaic_vng`** (Variable Number of Gradients) measures the gradient in
  all eight directions and averages over however many look smooth, so the
  number varies per pixel. **Verified against LibRaw's own `vng_interpolate`**
  — 0.26 of 255 mean difference on a real frame, with the residual traceable to
  this pipeline's highlight handling, which dcraw has no equivalent of. It is
  here as a conformance reference rather than a recommendation: it carries the
  most luminance detail of the six and the most false colour, which is the
  trade its design implies. Deciding from gradients is exactly what noise and
  clipping corrupt.
- **`bloom`** thresholds the highlights on **luminance**, blurs them with a
  **separate radius per channel**, and adds the result back. Equal radii give an
  ordinary glow; a wider red gives halation, which is a spread that differs per
  channel rather than a red tint — real halation is light reflecting off a
  film base, and red penetrates deepest. `intensity` is compensated for spread,
  so widening the glow no longer dims it and the two controls stay independent.
- **`vignette`** uses Lightroom's sign convention — negative darkens the
  corners, positive lightens them. The two directions are not one operation
  mirrored: darkening **scales** the pixel, which is what a transmission loss
  is and which preserves black, while lightening **lerps toward white**, since
  multiplying up would blow the corner highlights while barely moving its
  shadows. The falloff is normalised so 1.0 is the corner rather than the edge,
  and `roundness` blends between an ellipse following the frame and a true
  circle.
- **`film_grain`** has a real **grain size**, not just an amount: the noise is
  sampled on a lattice whose spacing is the grain size in pixels, so grain
  clumps rather than staying per-pixel — a bigger enlargement of the same
  negative shows bigger grains, not more of them. Strength is compensated for
  the variance interpolation removes, so size changes the texture and strength
  changes the loudness, independently. The grain is **multiplicative**, so
  black stays black and the midtones carry the texture; additive noise instead
  lifts the shadows into the grey haze that reads as digital sensor noise.
- **`orton`** is the darkroom sandwich: a sharp frame **screened** over a
  blurred, brightened copy. Screen rather than an average is the whole effect —
  it only brightens, saturates toward white, and leaves black alone, where
  averaging pulls the highlights down to meet the shadows and gives a flat haze.
  The brightening happens *before* the blur, as Orton overexposed his slides in
  camera, so highlights bleed outward rather than the frame washing out.
- **`apply_lut`** runs the image through a 3D `.cube` lookup table — the format
  film emulations and creative grades ship in, and what Resolve and Lightroom
  export. Interpolation is **tetrahedral** rather than trilinear, so a sharp
  transition in the table stays sharp and a LUT looks the same here as where it
  was authored.

  A LUT is **display-referred**: its domain is almost always 0..1, so it says
  nothing about a scene-linear highlight at 6.0, and sampling clamps rather than
  extrapolating. **Tone map first.** The stage reports what fraction of the
  frame was above the table's domain, so a pipeline in the wrong order says so
  rather than merely looking flat.

  A 33³ table is 35,937 entries and cannot be decomposed back into sliders —
  contrast and saturation are patterns spread across all of them, not numbers
  inside the file. So this is load-and-apply, with a `strength` blend; stack the
  ordinary adjustments after it.

  It runs on the GPU. The table is too big for a constant buffer (431 KB against
  a 64 KB limit) so it rides in as a stage-owned texture — see
  `AlgorithmBase::GpuExtraInputs` — packed 2D as `size` wide by `size²` tall.
  Deliberately not a 3D texture: hardware filtering there is *trilinear*, which
  would visibly disagree with the CPU's tetrahedral on any sharp LUT. Doing
  tetrahedral by hand keeps the two bit-identical. The dispatch measures under a
  millisecond at 1 MP.

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
./build/Release/tglab_sfm_tests.exe      # SfM, depth, splat gradients, GPU == CPU
./build/Release/tglab_pyramid_tests.exe  # detector GPU paths == CPU, same keypoints
./build/Release/tglab_nn_tests.exe       # network ops: CPU vs PyTorch, GPU vs CPU
```

`tglab_pyramid_tests` and `tglab_nn_tests` need a device but *skip* rather than
fail without one, so they stay in the gating suite on a machine with no GPU.

**Measurement tools**, none of them gating tests — they report a number and
leave the judgement to you:

| Tool | Answers |
|---|---|
| `bench_detect <w> <h>` | Per-detector CPU/GPU timings and feature counts. |
| `bench_pyramid <w> <h>` | The SIFT blur pyramid alone, CPU vs GPU. |
| `bench_dehaze <image>` | Dehaze on a real image. `BENCH_ALGO=brightness` times a trivial algorithm instead, which gives the framework's own unpack/pack floor — worth knowing before optimising anything. |
| `bench_stitch <raw>...` | The whole panorama chain headlessly, printing every stage's report. `--gain`, `--wta`. |
| `bench_sfm <dir> --script s.tgl` | A 3D script headlessly, printing every stage's report. `--gpu` gives it a device; `--dump DIR` saves every image viewer as PNG and renders splats from orbit viewpoints beside the middle camera — where floaters show. |
| `bench_nn dad\|dedode\|loma <weights> <reference>` | A learned network against PyTorch, layer by layer, on the CPU and the GPU, and timed. The reference comes from `tools/nn_reference.py`. |
| `frame_exposure <raw>...` | What actually differs across a sweep: EXIF settings, measured brightness, centre vs surround. Says whether seams want a per-frame gain or something spatial. |
| `vignette_profile <raw>...` | Lens falloff measured by averaging frames in sensor coordinates. **Reads its own verdict**: it refuses to call a result a lens profile when the variation is not radial. |

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
- **A chained match is by list order.** `chain = 1` pairs frame *i* with frame
  *i−1* as the group holds them, so frames out of order pair views that were
  never adjacent. That fails *confidently* — a spurious pairing was measured at
  91% inliers, indistinguishable from a real one by rate alone — so the aligner
  rejects any solve displacing a frame more than 1.5 frame widths and says why.
  Sort a group by name or date before stitching.
- **The overlap-disagreement metric measures more than alignment.** Two frames
  with a 3.9 px reprojection RMS still report ~10%, because it is luminance
  variance and a handheld pan has real exposure and vignetting differences
  between frames. Useful for comparing runs on the same data; not an absolute
  scale.
- **A coloured halo survives around blown highlights.** Bilinear interpolation
  across the one-pixel brightness cliff at the edge of a blown light averages
  one channel from saturated neighbours and takes another from a single dim
  centre, producing a channel ratio the scene never had; the camera matrix then
  rotates that false direction into a visible fringe. Blown highlight *cores*
  develop neutral, and the matrix no longer drives any channel negative, but
  the shoulder is still wrong. It needs an edge-aware initial estimate — a
  colour-space repair cannot recover detail the interpolation discarded.
  `scripts/demosaic_stages.tgl` shows each step; AHD and Malvar are markedly
  better here than bilinear or `demosaic_consistent`.
- **A few pixels of residual misalignment remain**, visible when zoomed in on a
  stitched panorama. Not pursued further yet: everything measured so far comes
  from one handheld sweep, and a residual that small could as easily be that
  sequence's parallax — the camera centre does move a little when handheld —
  as anything in the solver. Worth revisiting against a tripod set and a
  second scene, where the two can be told apart.
- **Repetitive texture still produces some wrong pairs.** On castle-P19 three
  of the 35 relative poses, all across frames 11–17 where the walk faces a
  wall of near-identical window bays, disagree with the rest by 15–30°:
  matches land on the wrong bay and are self-consistent enough that RANSAC
  keeps them. Robust rotation averaging down-weights them and the scene
  reconstructs (0.8° mean ray residual), but nothing yet *removes* them.
  (This entry once said those seven frames could not be reconstructed at
  all, blaming the imagery. That was the positioning bug — see
  `global_position`. A limitation attributed to the data deserves the same
  suspicion as any other conclusion.)
- **Sequential matching caps mean track length near 2.** Measured across five
  configurations — feature count 4×, resolution 1.5×, ratio loosened,
  cross-check disabled — the mean never left 2.2, and 83–87% of tracks are
  bare pairs. That is the arithmetic of independent pairwise matching rather
  than a defect: per-link survival is ~16%, so length-4 tracks should be
  ~0.16² ≈ 2.6% and are measured at 2.7%. Disabling cross-check tripled the
  chaining rate and still did not move it, because the extra merges were wrong
  (1156 physically impossible tracks, ray residual doubled).
- **Reflections do not leave the app.** `.ply` export writes base colour and
  harmonics, which other viewers understand; reflectivity, normals and the
  environment map have no standard fields, so a reflective splat exports as
  its colour before the environment is mixed in.
- **Splats only know what the cameras measured.** A textureless surface gets
  no depth from the sweep, so neither depth supervision nor carving can pin
  it, and colour alone places the Gaussians painting it. Views well outside
  the capture — beyond the first or last camera — extrapolate and look it.

---

## Licence

MIT — see [LICENSE](LICENSE). Dear ImGui is MIT; `stb` is public domain / MIT.
The network weights the release ships in `models/` are MIT (DaD, and LoMa's
DeDoDe descriptor) and Apache-2.0 (LoMa's matcher, from LightGlue); their
notices are in [third_party/models](third_party/models/).

**LibRaw is LGPL-2.1 or CDDL-1.0** and is linked statically, so it carries
obligations the permissive licences do not: the notice must be reproduced, and
recipients must be able to relink against a modified LibRaw. Its complete
source is vendored under [third_party/libraw/](third_party/libraw/) with both
licence texts, and building tglab rebuilds it from that source — which
satisfies both. Anyone distributing a compiled tglab needs to carry the same
source and notice.

`Help → About tglab` shows the version and every dependency with its licence.
