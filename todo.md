# tglab — to do

A shared list. Tim adds; Claude removes items as they land and notes anything
found along the way. Nothing here is a commitment to an order — it is a record
of what is known to be missing or wrong.

---

## Bugs

*(none currently known — the drop crash, the display corruption, the
"computing..." freeze and the highlight colour crushing are all fixed and
pushed)*

## Features

### Memory

- **Garbage-collect intermediate textures.** A 12-stage 45 MP RGBA16F pipeline
  holds ~8 GB of intermediates and nothing currently frees them. Tim is wary of
  breaking the stage cache that makes slider drags cheap, so the likely shape is
  a setting rather than always-on: automatic when VRAM is tight, off otherwise.
  The status bar already reports VRAM, so the trigger exists.

- **Proxy resolution for interactive edits.** Work at view resolution while a
  slider moves, then re-run at full resolution when idle. Benched deliberately:
  local-neighbourhood algorithms give different results at lower resolution, so
  the proxy is only honest for the preview. Lightroom tiles rather than
  downsamples.

### Colour and tone

- **`basic_adjust` ranges scaling with bit depth.** Raised once and partly
  addressed by the linear-light work, but not resolved: different raws have
  different bit depths, and the controls do not currently know.

- **Exposure and the histogram.** Tim: exposure does not visibly stretch the
  histogram. Suspected to be the SDR display window rather than the data —
  values pushed above 1.0 leave the visible range instead of stretching it.
  Related: showing where the SDR window sits within the full range, which Tim
  suggested as a way to make the headroom legible.

- **Blue-channel clipping at high exposure.** Observed on a warm test image
  before Kelvin white balance existed. Worth re-checking now that the cast can
  be corrected first — the blue gain needed may simply be much smaller.

### GPU

- **Otsu and friends on the GPU.** `threshold_otsu`, `threshold_triangle` and
  `threshold_isodata` have no GPU path, and the histogram is exactly why:
  deriving a level from one pins the whole algorithm to the CPU. With
  `ComputeContext::BuildHistogram()` they become build-on-GPU → read 4 KB →
  compute the level → dispatch the existing threshold kernel. Two things to
  settle first: whether Otsu on the subsampled histogram picks the same level as
  Otsu on every pixel (the stride is already a shader constant), and that such a
  stage cannot batch with its neighbours because the level must reach the CPU
  mid-stream. See the README's "Shared utilities" section.

## Answered, no action

- **`mosaic()` erroring on a non-raw image.** Recommended keeping the error: a
  script asking for sensor data on a JPEG has made a mistake worth reporting.

## Done

Kept briefly so a returning reader can see what recently changed, then pruned.

- Malvar-He-Cutler demosaic, now the default
- Scene-linear handling in `basic_adjust` (no double gamma, no clamp)
- Highlight recovery that desaturates rather than crushing
- Absolute Kelvin white balance
- GPU display path: no readback, GPU histogram, per-viewer versions
