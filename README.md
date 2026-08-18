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
| **M2** `choose()`, edge/gradient algorithms, drag-drop | **done** |
| **M3** worker thread, DXC, GPU compute path | **done** |
| **M4** GPU residency, CPU/GPU compare mode | **done** |

Algorithms so far: `brightness`, `grayscale`, `gaussian_blur` (CPU + GPU),
`sobel`, `non_max_suppression`, `hysteresis`, `canny`.

**Execution runs off the UI thread.** Parse and interpret stay on the UI thread
(microseconds, and they own the control state); the worker runs the pipeline
and publishes results. A burst of slider events coalesces to the newest
request, so dragging never builds a backlog, and the window keeps drawing
while a slow algorithm runs.

**GPU compute** is available per algorithm. `gaussian_blur` has an HLSL kernel;
under **Compute → Auto** it runs there, and Force CPU / Force GPU let you
compare. Images carry **GPU residency**, so a chain of GPU stages uploads once
at the head and never round-trips through the CPU in between — pixels come back
only when something asks for them.

**Compare mode** (**Compute → Compare CPU / GPU**) runs the pipeline twice,
forced to each backend, and reports max/mean/RMSE difference, the count of
pixels beyond tolerance, both timings, and an amplified diff image showing
*where* the two disagree. A kernel that merely looks right is not verified;
on the test image `gaussian_blur` agrees to within 1/255 at ~9× the speed.

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

Tests — the first is fast and needs no window; the second needs a D3D12 device
and takes a few seconds:

```sh
./build/Debug/tglab_tests.exe          # language, pipeline semantics
./build/Debug/tglab_runtime_tests.exe  # worker thread, shader compilation, GPU
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

### Adding a GPU kernel

An algorithm opts into the GPU by returning HLSL. The framework handles
compilation (cached, so slider drags don't recompile), residency, descriptor
binding and dispatch:

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
run, with the DXC diagnostics reported — a shader you're mid-way through
writing should degrade to slow-and-correct, not to a blank viewer.

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
- **The worker owns the stage cache.** `Execute()` moves cached outputs out of
  the previous run, so that pipeline must not be reachable from the UI thread;
  the worker keeps it and hands the UI only the images its viewers draw.
- **The compute queue is created on the worker thread**, separate from the
  direct queue ImGui submits on. Sharing one queue across threads is the
  classic source of intermittent corruption.

- **`Image` owns its residency.** `AcquireGpuWrite()` clears the CPU bit,
  `AcquireGpuRead()` uploads only when the GPU copy is stale, and
  `MapCpuRead()` is the single blocking sync point. That one rule is what keeps
  a chain of GPU stages free of intermediate transfers.

### Known limitations

- **Display still goes through the CPU.** A viewer calls `MapCpuRead()`, which
  reads the image back and re-uploads it as a display texture. Rendering
  straight from the compute output's SRV would remove that round trip; the
  residency API already supports it, the viewer does not yet use it.
- **One dispatch per stage.** A GPU algorithm gets a single kernel invocation,
  so genuinely multi-pass algorithms (a separable blur, say) either fuse into
  one pass or stay on the CPU.

## Licence

MIT — see [LICENSE](LICENSE). Dear ImGui is MIT; `stb` is public domain / MIT.
