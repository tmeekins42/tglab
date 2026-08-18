# tglab

A lab bench for computer-vision / computer-graphics **algorithm research**.

Drop images in, write a short script that wires algorithms together, expose the
interesting parameters as sliders, and watch results change as you drag them.
Algorithms are written in C++ (and, from M3, as HLSL compute shaders). The
scripting language is deliberately *glue only* — it never contains algorithm
logic.

Adding a new experiment costs one `.cpp` file and a few lines of script.

## Status — M1 (vertical slice) complete

Working end to end: script → registry → ports → parameters → algorithm →
display, with hot reload and live sliders.

| Milestone | State |
|---|---|
| **M1** vertical slice | **done** |
| M2 drag-drop palette, worker thread, `choose()`, more algorithms | next |
| M3 GPU path (DXC, compute queue, zero-readback display) | planned |
| M4 compare mode, benchmarks, pixel inspector | planned |

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
re-run** (or press F5). Slider values survive a reload, so tuning is not lost
when you edit. A parse error leaves the last good result on screen and reports
the line in the Status panel.

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
`check(label, default)`, `display(data [, name])`. Any registered algorithm is
callable by name; named arguments set its parameters.

Multi-output algorithms bind positionally:

```
gx, gy, mag = sobel(blurred)
```

The grammar also accepts matrix literals (`[[-1,0,1],[-2,0,2],[-1,0,1]]`) and
calling a *variable* that holds an algorithm — both are in place for M2's
`choose()` dropdowns and matrix parameters.

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
