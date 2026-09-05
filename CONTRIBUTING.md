# Contributing to tglab

How to add an algorithm, give it a GPU kernel, and use the shared utilities.
For what tglab is and how to drive it, see [README.md](README.md).

Adding an algorithm touches no central file: create one `.cpp` under the
matching category folder in [src/algorithms/](src/algorithms/) and the build
globs that tree. Registration is automatic.

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

**An integer that selects a MODE gets names, not a slider.** A slider shows a
number, so the mapping ends up spelled out in the label and picking a value
means dragging onto an exact integer:

```cpp
// Before: the label carries the documentation, the widget shows "1".
Param<int> m_projection{this, "projection", 1, 0, 2,
    {.help = "0 plane, 1 cylindrical, 2 spherical. ..."}};

// After: a dropdown reading "cylindrical", and the label is just a label.
static constexpr const char* kProjectionNames[] = {
    "plane", "cylindrical", "spherical"};

Param<int> m_projection{this, "projection", 1, 0, 2,
    {.help = "The surface the frames are projected onto. ...",
     .choices = kProjectionNames, .choiceCount = 3}};
```

Names map to `lo`, `lo+1`, … in order, so the range must cover them. The
**value is still the integer** — the algorithm reads its parameter exactly as
before, and nothing downstream changes. Both the inspector and `params()` pick
this up automatically; there is no script change and no second declaration.

Non-consecutive modes are not supported on purpose: they are almost always a
sign the enum wants renumbering, and the restriction keeps the declaration one
line.

For a value the SCRIPT owns rather than an algorithm, `pick()` is the same
control:

```
proj = pick("projection", ["plane", "cylindrical", "spherical"], 1)
```

It returns the index, so it drops straight into a parameter that expects the
integer. And a two-state mode is a `check()`, not a two-entry `pick()`.

**A path, not a number.** `Param<std::string>` exists for the one thing a
numeric parameter cannot express — `apply_lut`'s `.cube` file. It deliberately
has *no widget*: `DescribeControl()` returns false, so `params()` skips it and
the inspector shows the value without offering to edit it. A path is not
something to drag, and a text field in the controls panel would be a worse way
to choose a file than the script line that already names it.

```cpp
Param<std::string> m_path{this, "file", {}, "Path to a .cube LUT."};
```

Its hash folds the string, so changing the file re-runs the stage like any
other parameter change.

Every numeric row supports **ctrl+click to type an exact value**, **left/right
arrows to step** while the slider is focused, and **right-click to reset** to
the declared default. None of these add chrome, so the panel stays one row per
parameter.

**Categories must hold interchangeable algorithms.** `choose(label, category)`
offers every member in one dropdown, so they need the same call shape. That is
why Canny's internal stages sit in `"edge stage"` rather than `"edge"` — they
take three inputs, or require `R32F`, so selecting them from an `"edge"`
dropdown could only ever fail.

### Attaching data to an image: sidecars

Some algorithms produce something that is *about* an image rather than a
replacement for it — keypoints, matches, a transform. Those travel as
**sidecars**: the algorithm passes the pixels through unchanged and attaches a
named, immutable payload.

```cpp
auto sc = std::make_shared<FeatureSidecar>();
sc->detector = "orb";
/* ... fill in keypoints and descriptors ... */
if (Image* im = ctx.OutImage(0)) im->Sidecars().Set(kFeatureSidecar, sc);
```

and downstream, reading something it did not ask for and may not find:

```cpp
const FeatureSidecar* fs = FeaturesOf(img);
if (!fs) { /* no detector ran; say so or carry on */ }
```

Two rules make this safe rather than a loose bag of state:

- **Name it with a constant.** `kFeatureSidecar`, `kMatchSidecar`,
  `kTransformSidecar`. A typo in a literal means "not found", which looks like
  a detector that found nothing rather than a bug.
- **Say whether it derives from pixels.** `DerivedFromPixels()` returning true
  means the framework drops the sidecar when the pixels change — an alignment
  solved against an image is wrong the moment that image is re-developed, and
  silently keeping it produces a subtly wrong merge rather than an error.

The gain is that a merge samples through a transform it was never told about,
and works identically when there is none.

### Algorithms that see the whole group

`RunCPU` is handed one image at a time, which is wrong for anything inherently
about *pairs* or *sets* — matching, alignment, bundle adjustment. Those return
true from `IsAligner()` and implement `RunAlign` instead:

```cpp
bool IsAligner() const override { return true; }
bool RunAlign(std::vector<Image>* images, std::string* err) override;
```

`RunCPU` is then never called (leave it empty). A reduction — many images in,
one out — is a different thing again: see `Reducer` in
[src/core/reduction.h](src/core/reduction.h), which streams `Begin`/`Accept`/
`Finish` so an accumulator never holds the whole group. `stitch_panorama` is
the one reduction that *must* hold every frame, because its canvas size depends
on where the last one lands.

### Changing the output's size

The pipeline allocates each output from input 0 before the algorithm runs,
which is right for the overwhelming majority: a blur, a tone map and a demosaic
all produce one output pixel per input pixel. An algorithm that changes the
raster says so:

```cpp
ImageDesc OutputDesc(int port, const ImageDesc& in) const override {
    ImageDesc d = in;
    d.width  = /* ... */;
    d.height = /* ... */;
    return d;
}
```

A hook rather than a port declaration, because the size is not a property of
the algorithm the way its format is — the same crop produces a different
descriptor for every setting of its sliders, where a format never varies with a
parameter. Returning `in` unchanged is the default and is what every other
algorithm wants.

The bypass path checks this: an algorithm whose `OutputDesc` differs from its
input cannot be aliased away even when `IsNoOp()` is true, because downstream
would get a raster of the wrong dimensions.

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
- `features.h` — `Keypoint`, `DescriptorSet` (which carries its own *kind*, so
  a matcher picks L2 or Hamming from the data rather than being told),
  `FeatureSidecar`, `MatchSidecar`, and `Percentile99()`. That last one is
  worth knowing about even outside the detectors: `ValueScale()` converts the
  *format*, and a scene-referred raw sits in the bottom sixth of 0..1 whatever
  its format, so any threshold expressed as "a fraction of the range" has to be
  measured against the image's own level or it means nothing on a raw.
- `transform.h` — `Affine` (eight parameters: a homography with its
  bottom-right entry pinned at 1), `SampleBilinear`, and the transform sidecar.
  Note that pinning: a stored rotation-induced homography does **not** have
  determinant 1, because `From3x3` normalises by h22. Do not use the
  determinant as a health check.
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

**A parameter measured in PIXELS must be scaled for the proxy.** While a slider
is dragged the pipeline may run at reduced resolution, where a radius stated in
pixels covers a different fraction of the picture — a blur of σ=20 on a
quarter-scale image must run at σ=5 to look the same.

```cpp
// CPU, from RunCtx:
const float sigma  = std::max(0.01f, ctx.ScaledPx(float(m_sigma)));
const int   radius = ctx.ScaledRadius(int(m_radius));

// GPU, from the base — GpuConstants() has no RunCtx and the input
// descriptors are already out of reach by the time it runs:
return {bits(GpuScaledPx(float(m_sigma)))};
```

Both return the value unchanged at full resolution, so they are always safe to
apply. **Scale both paths or neither**: a GPU kernel and its CPU twin blurring
by different amounts is a divergence that only appears mid-drag.

**A value DERIVED from a pixel-unit parameter usually must not be scaled.**
This is the subtle one, and it produced a real bug.

`bloom` compensates its intensity by `spread² / 64`, so that "intensity 1"
means one thing at every radius. Scaling the spread for the proxy scaled that
compensation along with it — at a third scale, spread 24 becomes 8 and the gain
drops **9×**. The glow vanished for the length of a drag and reappeared on
release, which read as "the spread sliders do nothing until I let go".

The test: *what is this expression a statement about?* The blur radius is about
the raster, so it scales. The compensation is about the user's **setting** —
the user changed nothing, only the resolution did — so it must be computed from
the unscaled value.

Three cases that are *not* a plain multiply, and each looks like an oversight
until you know why:

- **Screen furniture** — `crop`'s preview rectangle and grid. These are drawn
  on the proxy and displayed at 1:1 with it, so scaling them down would make
  the overlay nearly invisible during exactly the drag it exists to guide.
- **Derived from the image** — `dehaze` computes its radii from the frame
  dimensions, so it adapts on its own. Scaling again would apply the factor
  twice.
- **`film_grain`** — the multiply is right but for a different reason. Grain is
  a random *field*, not a function of the image, so there is no full-resolution
  answer to approximate. What the preview owes you is grain the same size *on
  screen*, which works out to the same arithmetic — and degrades below size 1,
  where the lattice cannot go finer than a pixel.

`ProxyBehaviour::Never` is for algorithms where no parameter adjustment helps:
demosaicers (downsampling destroys the CFA pattern that *is* the data) and
anything touching a sidecar (keypoint coordinates are in image pixels and
nothing rescales them).

**Check every curve above 1.0 before shipping it.** A polynomial fitted to
behave on `[0, 1]` says nothing about what it does past the end, and the usual
ones misbehave badly. `smoothstep` is the trap: `t²(3−2t)` is a step only on
`[0, 1]`, and beyond it the cubic turns over — `t=1.5` gives exactly 0, `t=2`
gives −4, `t=3` gives −27.

That shipped in `orton`. At its default contrast of 0.2 it drove anything above
about 2.4× white *negative*, so overexposed cloud came back **black** — and
scene-linear highlights are routinely 4–8× white, so the failing case was a
bright sky, not an edge case.

Two things made it survive review. The existing HDR test passed `contrast = 0`,
which switches the S-curve off — **a test of an HDR path must use the default
settings, not a configuration that avoids the arithmetic under suspicion**. And
the fix wants a *weight* that fades the curve out approaching white rather than
a hard switch at 1.0: switching also stops the black clouds, but leaves a 25%
slope jump exactly at the white point, and a slope jump on a smooth sky gradient
is a mach band — in the same content that exposed the original bug.

The general check is monotonicity: feed values through the white point and
assert the output keeps rising. That tests the shape of the curve rather than
one sampled value.

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

### Denoise

Several filters above already denoise — `bilateral`, `nonlocal_means`,
`anisotropic_diffusion` and `median_blur` are all reasonable choices. What
`Category()=="denoise"` adds is the multi-scale approach, which is a different
idea rather than another neighbourhood shape.

| Algorithm | CPU | GPU | Notes |
|---|---|---|---|
| `wavelet_denoise` | ~170 ms/MP | **14x** | À-trous wavelet shrinkage. Separate luma and chroma thresholds. |

**Why multi-scale.** A spatial filter has one neighbourhood and must trade
grain against texture inside it. A wavelet decomposition splits the image into
bands first, so fine grain is removed at the scale it occupies while edges —
which carry energy at every scale — survive. The transform is *à trous* ("with
holes"): undecimated, so every level stays full resolution. That costs memory
against a decimated transform, but avoids the shift-dependence that makes
decimated wavelets ring around edges — the artefact that makes naive wavelet
denoising look worse than the noise it removed.

Two details that matter more than the transform:

- **Soft thresholding, not hard.** Zeroing coefficients below a threshold
  leaves a discontinuity, and a coefficient sitting either side of it from one
  pixel to the next flips between kept and discarded. That is the mottled
  "wavelet blotch" that gives the method its bad reputation.
- **Luma and chroma are shrunk separately,** with chroma defaulting four times
  harder. Sensor noise is far worse in chroma, and chroma carries little fine
  detail, so it tolerates much heavier smoothing before anything shows. That
  asymmetry is most of what makes ISO denoising work, and it is invisible in an
  RGB basis where all three channels get the same treatment.

Measured on an ISO 1000 CR2 to confirm this is the right tool for that noise:
sigma is ~53 sensor levels at ISO 1000 and ~33 at ISO 400, the tails stop
around 4.5 sigma (broad-band, no impulse component — impulse noise would want
a median instead), and sigma tracks √signal, meaning it is shot-noise dominated
rather than a constant read-noise floor. Both point at coefficient shrinkage.

**Denoise after develop, not before.** This has a measured answer rather than
a stylistic one. On an under-exposed ARW pushed hard (exposure 1.85, shadows
0.6), measuring robust sigma in the darkest 40% of the frame for noise and the
same statistic in the brightest 40% for detail:

| | shadow noise | detail kept |
|---|---|---|
| develop only | 0.02481 | 0.04903 (100%) |
| **denoise after develop** | **0.00903** (−64%) | **0.04325 (88%)** |
| denoise before develop | 0.00004 | 0.00283 (6%) |

Denoising first looks spectacular on the noise column and is catastrophic: it
keeps 6% of the detail. Develop applies a strong non-linear tone curve, so
denoising ahead of it means thresholding in linear scene space where shadow
detail is compressed into a tiny value range — a fixed threshold eats nearly
all of it, and the curve then stretches what little survives, amplifying the
damage. After develop, the threshold is applied in the space the eye actually
sees.

The general rule: a shrinkage threshold is a fixed distance in the value
domain, so it belongs on the same side of a tone curve as the values it was
tuned against.

**Zoom into a pushed raw and you will see coloured dots on a 2-pixel grid.**
That is not a demosaic bug, and it is worth knowing before chasing it. On a
Bayer sensor green is sampled at half of all sites while red and blue get a
quarter each, so red and blue are interpolated from samples twice as far apart
and carry roughly twice the noise — at exactly the sensor's own spacing.
Measured on `_DSC0162.ARW` after a 1.85-stop push, the chroma speckle is
0.0752 against 0.0210 of luma detail: the visible noise is overwhelmingly
chroma. Both `demosaic_malvar` and `demosaic_bilinear` score the same parity
swing, which is the check that says the demosaic is not at fault — a real
demosaic bug would separate them.

The defaults follow from that: chroma 0.08, luma 0.005. The luma default was
originally 0.02, taken from a synthetic fixture far noisier than a real photo,
and since the entire luma detail of a developed image is around 0.02 that
threshold removed 62% of the picture. Chroma at 0.08 removes 88% of the
speckle and costs nothing visible, because chroma carries almost no fine
detail.

[scripts/denoise.tgl](scripts/denoise.tgl) puts the original and the denoised
result side by side, with develop before the denoise for this reason.

**On the GPU** it is one dispatch per level, ping-ponging the accumulator
between the output and the scratch. Measured on a 24 MP Sony ARW through the
full develop-and-denoise pipeline: **7.8 s → 0.54 s (14x)**, which is the
difference between a slider you nudge and one you drag.

Two things the kernel does differently from the obvious version, both forced
by the two-buffer ping-pong:

- **The blur is one 5×5 dilated pass, not two separable 1D passes.** Splitting
  it would need three buffers — accumulator, horizontal partial, destination —
  and the vertical pass would no longer have the accumulator it must subtract
  from. 25 taps against 10 is more arithmetic but it all comes from cache, and
  it removes a full-image round trip per level.
- **The luma/chroma rotation happens inside each dispatch,** rather than once
  around the whole pyramid. The ping-pong buffers take the *output's* format,
  and chroma is a signed difference: on an 8-bit image a UNORM target clamps
  every negative chroma value to zero the moment an intermediate is written.
  That agreed perfectly on float images and was wrong by 182/255 on 8-bit ones
  — caught by the CPU/GPU agreement test, not by looking at a picture. The CPU
  path was changed to match, so both compute the same thing.

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

**Data the algorithm supplies itself.** Constants go in the constant buffer and
images arrive through ports, which covers almost everything. `apply_lut` is
where both run out: a 33³ colour LUT is 431 KB against the constant buffer's
64 KB limit, it comes from a *file* rather than from another stage, and it
cannot be a scratch plane because those are allocated at image size.
`GpuExtraInputs()` returns textures the algorithm owns, bound as SRVs after the
port inputs:

```cpp
std::vector<const Image*> GpuExtraInputs() const override {
    if (!m_table.Valid()) return {};
    return {&m_table};
}
```

The images must outlive the dispatch, so hold them as members. Returning the
same ones every call is expected — they upload once and stay resident, so
rebuilding them per run would upload per run. Rebuild only when the thing they
describe changes.

Indexing follows `GpuPass::reads`: with one input port the extras start at
`-2`, so a pass reading `{-1, -2}` gets the image then the first extra.

Worth noting what `apply_lut` does *not* do: bind the LUT as a 3D texture. That
would bring hardware trilinear filtering for free, and free is the wrong price
here — the CPU path is *tetrahedral*, so the two would visibly disagree on any
LUT with a sharp transition, and the audit would rightly flag it. Packing the
cube into a 2D texture and doing tetrahedral by hand keeps the paths
bit-identical and matches what Resolve does, so a LUT looks the same in all
three.

A kernel that fails to compile falls back to the CPU rather than failing the
run, with the DXC diagnostics reported — a shader you are mid-way through
writing should degrade to slow-and-correct, not to a blank viewer.

### GPU work in a CPU algorithm

`RunGPU` is image-in/image-out: the pipeline binds the inputs, dispatches, and
leaves the result on the device. Some algorithms do not fit that shape at all.
A **feature detector's product is a sidecar** — a keypoint list — and its image
output is the input passed through untouched, so there is nothing for `RunGPU`
to write, and the interesting work (extrema, orientation, descriptors) is
branchy and belongs on the CPU anyway.

Such an algorithm stays a CPU algorithm and reaches the device directly:

```cpp
if (ComputeContext* dev = ctx.Gpu()) {
    // dispatch, read back, use the result
}
```

`ctx.Gpu()` is null when there is no device, the device was lost, or the user
picked Force CPU. **Null is ordinary, not an error** — every caller needs its
CPU path regardless, so treat this as an optimisation that may decline. See
[gpu_pyramid.h](src/algorithms/features/gpu_pyramid.h), which `detect_sift`
uses for its scale space.

**Measure before writing the kernel.** SIFT's pyramid was 65% of its runtime, so
offloading it paid at 3.8×. Reasoning by analogy from that would have been wrong
for all three of the others: ORB's pyramid is a cheap box average and its cost is
the FAST+Harris scan; BRISK's is neither, being half descriptor; AKAZE's is the
diffusion. `bench_detect` exists to answer this question in about a minute.

The sharpest version of the trap: BRISK's `IsScaleMax` looked like the obvious
target — up to ~160 ring walks per candidate — so a full-image score map per
layer seemed clearly right. It made BRISK *slower* (1.0× to 0.9×), because those
lookups run per CANDIDATE and there are thousands of those against millions of
pixels: the map computed ~75× more than was ever read. **Dense dispatch only pays
for dense demand.**

Two things to get right, both learned the hard way here:

- **Do not pass width and height in your constants.** `Dispatch` already fills
  `b0`'s first two slots from the *output* descriptor, so passing them again
  shifts every later field by two. The pyramid shader read the image width as
  its blur radius and produced a plausible-looking wrong blur — no crash, no
  validation error, just a silently different result.
- **Keep chained work resident.** Reading each intermediate back and uploading
  it again costs more than the compute at any real size. `GpuBlurStack` does a
  whole octave in one call for exactly this reason.

Offloading only pays if the two paths agree *exactly*. A detector that finds
different keypoints depending on whether a device was present is worse than a
slow one, because every comparison built on it is then meaningless — which is
what `tglab_pyramid_tests` exists to prevent.

### Verifying a GPU kernel

**Compute → Compare CPU / GPU** runs the pipeline twice, forced to each
backend, and reports max/mean/RMSE difference, the count of pixels beyond
tolerance, both timings, and an amplified diff image showing *where* they
disagree. A kernel that merely looks right is not verified.

**Compare on a REAL file, not a synthetic one.** `gpu_audit` runs every
algorithm on a synthetic gradient, which is enough to catch a wholly broken
kernel and not enough to catch a subtly wrong one. For a demosaic it exercises
neither clipping nor out-of-gamut colour, so the highlight clamp and the
in-gamut solve never fire — and three shipped demosaicers had shaders applying
the bare camera matrix where their CPU paths used `CameraMatrixInGamut`, a 0.85
divergence on precisely the pixels that solve exists for, with the audit
reporting clean throughout. Two other things it cannot see: an algorithm with
**no** kernel compares perfectly with itself (`demosaic_vng` reads 0.000000
because both runs fall back to the CPU), and any code path the fixture does not
reach is untested rather than correct.

**Put shared shader code in one string.** Duplicating a helper into each kernel
is how the above happened: four demosaicers each carried their own copy of the
colour step, and three drifted from the CPU. `clip_repair.h` now exports
`kClipRepairHlsl`, concatenated into every demosaic kernel, so there is one
definition to fix.

**Beware exact float comparisons in a kernel.** CPU and GPU do not order
floating-point operations identically, so a branch on `a == b` can be taken
differently on each. `demosaic_ppg` chose a green interpolation direction that
way; 0.262% of sites hit the tie exactly and the two paths disagreed on 0.072%
of samples. A relative band (`1e-4 * max(a, b)`) applied identically in both
paths brought that to 0.008%.

