# tglab

A lab bench for computer-vision / computer-graphics **algorithm research**.

Drop images in, write a short script that wires algorithms together, expose the
interesting parameters as sliders, and watch results change as you drag them.
Algorithms are written in C++ (and, from M3, as HLSL compute shaders). The
scripting language is deliberately *glue only* — it never contains algorithm
logic.

Adding a new experiment costs one `.cpp` file and a few lines of script.

## Status

Working end to end: script → registry → ports → parameters → algorithm →
display, with hot reload, live sliders, algorithm dropdowns, and multi-output
algorithms.

| Milestone | State |
|---|---|
| **M1** vertical slice | **done** |
| **M2** `choose()`, edge/gradient algorithms, drag-drop | **done** (worker thread still pending) |
| M3 GPU path (DXC, compute queue, zero-readback display) | next |
| M4 compare mode, benchmarks, pixel inspector | planned |

Algorithms so far: `brightness`, `grayscale`, `gaussian_blur`, `sobel`,
`non_max_suppression`, `hysteresis`, `canny`.

Pipeline execution still runs on the UI thread. That is fine at 512×512, but
a slow algorithm on a large image will block the window until the worker
thread lands.

## Build

Requires Visual Studio 2022 (or 18) with the Windows SDK. CMake and Ninja ship
with VS, so no separate install is needed.

Dear ImGui is a submodule, so clone recursively (or run
`git submodule update --init` in an existing clone):

```sh
git clone --recursive <repo-url>
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Debug
```

Run — arguments are a `.tgl` script plus any images to preload:

```sh
./build/Debug/tglab.exe scripts/hello.tgl assets/test.png
```

Headless checks for the script engine (no window required):

```sh
./build/Debug/tglab_tests.exe
```

## Scripting

The script is a plain file on disk; edit it in your own editor and **save to
re-run** (or press F5). The title bar shows which script is loaded, with an
`[error]` marker when it is failing. Slider values survive a reload, so tuning
is not lost when you edit. A parse error leaves the last good result on screen
and reports the line in the Status panel and on stderr.

To get back to the script's declared values, **double-click any control**, or
use **Reset all** in the Controls panel (also in the File menu). The button is
greyed out when nothing has been changed, so it doubles as an indicator.

```
src = image("test")                  # by palette name

amount = slider("brightness", -1.0, 1.0, 0.0)
gain   = slider("gain", 0.0, 4.0, 1.0)

grey     = grayscale(src)
adjusted = brightness(grey, amount = amount, gain = gain)

display(src,      "original")        # each display() is a docked viewer
display(grey,     "greyscale")
display(adjusted, "adjusted")
```

Builtins: `image(name)`, `slider(label, min, max, default)`,
`check(label, default)`, `choose(label, options)`, `display(data [, name])`.
Any registered algorithm is callable by name; named arguments set its
parameters.

**Multi-output algorithms** bind positionally, and `_` discards outputs you
don't need:

```
gx, gy, mag = sobel(blurred)         # gx/gy are signed R32F
_,  _,  mag = sobel(blurred)         # only the magnitude
out         = sobel(blurred)         # one target takes the first output
```

`_` is write-only, as in Rust and Go — reading it back is an error, since it
would just return whichever output was assigned last. Discarded outputs are
still computed and cached, so adding a viewer for one later costs nothing.
(`_tmp` and similar are ordinary names; only a bare `_` is special.)

**`choose()`** puts a dropdown in the Controls panel and returns the selected
algorithm as a callable value — swap implementations without editing the
script:

```
prep   = choose("preprocess", [gaussian_blur, grayscale])   # explicit list
edgeOp = choose("edge operator", "edge")                    # whole category

result = edgeOp(prep(src))
```

Passing a *category* means a newly written algorithm declaring that
`Category()` appears in the dropdown automatically, with no script change.

The grammar also accepts matrix literals (`[[-1,0,1],[-2,0,2],[-1,0,1]]`),
which are in place for kernel parameters.

Example scripts: [scripts/hello.tgl](scripts/hello.tgl) (basics),
[scripts/edges.tgl](scripts/edges.tgl) (Canny stage by stage),
[scripts/compare.tgl](scripts/compare.tgl) (`choose()` dropdowns).

### Canny, both ways

The stages are registered individually *and* wrapped by `canny()`. Call the
stages when you want to see what each one did; call `canny()` when you just
want edges.

```
blurred     = gaussian_blur(src, sigma = 1.4)
gx, gy, mag = sobel(blurred)
thin        = non_max_suppression(gx, gy, mag)
edges       = hysteresis(thin, low = 0.1, high = 0.3)

edges2 = canny(src, sigma = 1.4, low = 0.1, high = 0.3)   # same thing
```

R32F viewers auto-normalise min/max, so signed gradients are visible (mid-grey
is zero). Note this means brightness is *not* comparable between two viewers,
since each scales independently.

## Adding an algorithm

Create one `.cpp` in `src/algorithms/` and add it to the `tglab_algorithms`
list in `CMakeLists.txt`. Nothing else is edited — registration is automatic.

```cpp
class Threshold : public AlgorithmBase {
public:
    const char* Name()     const override { return "threshold"; }
    const char* Category() const override { return "segment"; }

    PortList Inputs()  const override { return {{"src", DataType::Image, FormatSpec::Any}}; }
    PortList Outputs() const override { return {{"out", DataType::Image, FormatSpec::SameAsInput}}; }

    void RunCPU(RunCtx& ctx) override {
        const ImageView src = ctx.In(0);
        ImageView       dst = ctx.Out(0);
        const float level = m_level;   // reads as a plain float
        /* ... */
    }
private:
    Param<float> m_level{this, "level", 0.5f, 0.0f, 1.0f};
};
REGISTER_ALGORITHM(Threshold);
```

`Param<T>` is the single storage location for a parameter — the script writes
it, the UI writes it, the algorithm reads it. There is nothing to keep in sync.

> `src/algorithms/` **must** stay a CMake `OBJECT` library. Algorithms
> self-register through a static initializer that nothing references, so a
> `STATIC` library would be stripped by the linker and they would silently
> vanish from the registry.

## Layout

```
src/script/     lexer, parser, AST, interpreter   (knows nothing about algorithms)
src/core/       Image, Param<T>, AlgorithmBase, Registry, Pipeline
src/gpu/        D3D12 device, descriptor heap, textures
src/app/        win32 window, ImGui panels, views, file watching
src/algorithms/ one .cpp per algorithm
```

Built on Dear ImGui (docking branch) + D3D12, and `stb_image`. Both are
vendored under `third_party/`.

## Design notes

- **Execution** is a linear stage list, not a DAG. Dirty detection is by
  parameter *hash*, so there is no invalidation flag to forget to set.
- **Re-parsing happens on every change.** Parsing is microseconds; the cost is
  the algorithms. Results are cached, ASTs are not.
- **Ports carry a `Data` variant**, currently holding only `Image`. Feature
  sets, matrices and point clouds become new alternatives without touching any
  existing algorithm.
- **Views render to offscreen targets** behind a `View` interface, so a 3D
  viewport (SfM, gaussian splats) can dock alongside 2D panels later.

## Licence

MIT — see [LICENSE](LICENSE). Dear ImGui is MIT; `stb` is public domain / MIT.
