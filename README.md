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

Dear ImGui is a submodule, so clone recursively:

```sh
git clone --recursive https://github.com/tmeekins42/tglab.git
cd tglab
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Debug
```

(In an existing clone, `git submodule update --init` fetches ImGui.)

## Run

Arguments are a `.tgl` script plus any images to preload. Paths are relative to
the repository root, which is where the app expects to run from.

```sh
./build/Debug/tglab.exe scripts/thresholds.tgl assets/test.png
```

Images can also be **dragged in from Explorer** at any time.

| Panel | What it does |
|---|---|
| **Images** | The palette. `image("name")` refers to these by filename without extension. |
| **Scripts** | Every `.tgl` beside the current script — click to switch. |
| **Algorithms** | Every registered algorithm, with its ports and parameters. |
| **Controls** | Sliders and dropdowns the script declared. |
| **Status** | Errors, run time, and how many stages used the GPU. |
| **Compare CPU / GPU** | Opened from the Compute menu. See below. |

Editing the script file and saving re-runs it (or press **F5**). Slider values
survive the reload, so tuning is not lost when you edit. A parse error leaves
the last good result on screen and reports the line in **Status** and on
stderr.

Double-click any control to restore its scripted default, or use **Reset all**.

## Tests

```sh
./build/Debug/tglab_tests.exe          # language and pipeline semantics (fast)
./build/Debug/tglab_runtime_tests.exe  # worker thread, shaders, GPU (needs a device)
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
| `image("name")` | An image from the palette. |
| `slider("label", min, max, default)` | Declares a slider; returns its current value. |
| `check("label", default)` | Declares a checkbox; returns 0 or 1. |
| `choose("label", [a, b, c])` | Dropdown of algorithms; returns the selected one. |
| `choose("label", "category")` | Same, but offers every algorithm in a category. |
| `params(algo)` | Declares a control for each of `algo`'s parameters; returns `algo`. |
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

std::vector<uint32_t> GpuConstants() const override {
    const float level = m_level;
    uint32_t bits; std::memcpy(&bits, &level, sizeof(bits));
    return {bits};
}
```

Bindings are fixed by convention: `t0..t3` inputs, `u0..u3` outputs, and `b0`
holding `Width`, `Height`, then whatever `GpuConstants()` returns (floats
bit-cast through `uint`, read back with `asfloat`). Thread groups are 8×8.

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
