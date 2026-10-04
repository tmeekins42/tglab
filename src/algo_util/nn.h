// nn — the few neural-network operations the learned feature matchers need,
// on the GPU and on the CPU.
//
// WHY WRITE OUR OWN rather than link an inference runtime. tglab is one small
// executable that starts in milliseconds; ONNX Runtime would add tens of
// megabytes of DLLs and a dependency on DirectML, which Microsoft has put into
// maintenance. And the networks in question (LoMa: DaD, DeDoDe, a LightGlue-
// style matcher) are built from a short list of operations -- convolutions,
// pooling, resizing, attention -- each a page of HLSL.
//
// ONE INTERFACE, TWO BACKENDS. A network is written once, against Engine;
// Engine runs each operation on the device when it has one and on the CPU
// when it does not. The CPU path is the reference: plain loops, transcribed
// from the PyTorch definitions, which is what the GPU kernels are tested
// against and what runs on a machine without a device.
//
// TENSORS ARE C x H x W floats, one image at a time -- PyTorch's NCHW with
// N = 1, so offsets and weights carry over without transposition.
//
// WEIGHTS come from a .tgw file, which tools/nn_convert.py writes from a
// PyTorch checkpoint with every BatchNorm already folded into the convolution
// before it. Inference needs no BatchNorm operation at all.
#pragma once

#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "../gpu/compute.h"

namespace tglab {
namespace nn {

// --- weight files ---------------------------------------------------------------
//
// .tgw: "TGW1", a uint32 count, then per tensor: uint16 name length, the
// name, uint8 dtype (0 = float32, 1 = float16), uint8 rank, int32 dims, and
// the data, little-endian. Float16 halves the file and is well inside the
// precision these networks were trained at (bfloat16 autocast).
struct Weight {
    std::vector<int>   shape;
    std::vector<float> data;
    size_t Count() const;
};

class WeightFile {
public:
    bool Load(const std::string& path, std::string* err);
    // Writes float32: for the reference dumps tests compare against.
    bool Save(const std::string& path, std::string* err) const;

    const Weight* Find(const std::string& name) const;
    void Put(const std::string& name, Weight w) { m_t[name] = std::move(w); }
    const std::map<std::string, Weight>& All() const { return m_t; }

private:
    std::map<std::string, Weight> m_t;
};

// Where a model's weights live: the first of $TGLAB_MODELS/<file>, then
// models/<file> beside the executable or up to three directories above it
// (the repository root, when running from build/Release), then
// models/<file> under the working directory. Empty when none exists.
std::string FindModelFile(const std::string& file);

// --- tensors --------------------------------------------------------------------

struct Tensor {
    int c = 0, h = 0, w = 0;
    std::vector<float> cpu;   // the CPU backend's storage
    GpuBuffer          gpu;   // the GPU backend's
    // Where `gpu` goes back to when the tensor is done with it; see
    // ComputeContext::AcquireBuffer.
    ComputeContext*    pool = nullptr;

    size_t Count() const { return size_t(c) * size_t(h) * size_t(w); }
    size_t Plane() const { return size_t(h) * size_t(w); }

    // Hands the device buffer back to the pool rather than freeing it.
    void Recycle() {
        if (pool && gpu.Valid()) pool->RecycleBuffer(std::move(gpu));
        gpu.Release();
    }

    Tensor() = default;
    ~Tensor() { Recycle(); }
    Tensor(const Tensor&)            = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& o) noexcept
        : c(o.c), h(o.h), w(o.w), cpu(std::move(o.cpu)), gpu(std::move(o.gpu)), pool(o.pool) {}
    Tensor& operator=(Tensor&& o) noexcept {
        if (this != &o) {
            Recycle();
            c = o.c; h = o.h; w = o.w;
            cpu = std::move(o.cpu);
            gpu = std::move(o.gpu);
            pool = o.pool;
        }
        return *this;
    }
};

// One convolution's parameters, BatchNorm folded in. `groups` is 1 or, for a
// depthwise convolution, the channel count. Stride 1, padding k / 2 -- the
// only form these networks use.
struct Conv {
    int cout = 0, cin = 0, k = 1, groups = 1;
    std::vector<float> w;   // [cout][cin / groups][k][k], PyTorch's order
    std::vector<float> b;   // [cout]

    // The device copy, made on first GPU use, in the layout the kernel reads
    // (see nn_gpu.cpp). Mutable because uploading is a cache, not a change.
    mutable GpuBuffer gw, gb;

    // From a .tgw: "<name>.weight" and "<name>.bias".
    bool From(const WeightFile& f, const std::string& name, std::string* err);

    // A LINEAR layer, out x in, as the 1x1 convolution it is on a channel-
    // major tensor of points (c x 1 x n): see Engine.
    bool FromLinear(const WeightFile& f, const std::string& name, std::string* err);
};

// A parameter vector: a LayerNorm's scale or shift, a frequency table.
struct Vec {
    std::vector<float> v;
    mutable GpuBuffer  g;   // the device copy, made on first GPU use
    bool From(const WeightFile& f, const std::string& name, std::string* err);
};

// How Resize interpolates. Both follow PyTorch's align_corners=False.
enum class Interp { Bilinear, Bicubic };

class GpuKernels;   // nn_gpu.cpp

class Engine {
public:
    // Null runs everything on the CPU.
    explicit Engine(ComputeContext* gpu = nullptr);
    ~Engine();

    bool OnGpu() const { return m_gpu != nullptr; }
    const std::string& Error() const { return m_err; }

    // Shapes a tensor for the active backend. Contents are unspecified.
    bool Alloc(Tensor* t, int c, int h, int w);
    bool Upload(const std::vector<float>& v, int c, int h, int w, Tensor* t);
    bool Download(const Tensor& t, std::vector<float>* v);

    // out[outOff + o] = act(conv(in)), for o in [0, cout). `out` must already
    // have room: writing into a channel offset is how concatenation is done.
    bool Conv2d(const Tensor& in, const Conv& cv, bool relu, Tensor* out, int outOff = 0);
    // 2x2, stride 2, rounding the size down.
    bool MaxPool2(const Tensor& in, Tensor* out);
    // Channels [inOff, inOff + count) of `in`, resized to out's h x w, into
    // channels [outOff, outOff + count) of `out`.
    bool Resize(const Tensor& in, int inOff, int count, Interp mode, Tensor* out, int outOff);
    // dst[dstOff + i] (+)= src[srcOff + i] for i in [0, count), same h x w.
    bool Copy(const Tensor& src, int srcOff, int count, Tensor* dst, int dstOff);
    bool Accumulate(const Tensor& src, int srcOff, int count, Tensor* dst, int dstOff);
    // out = (a + b) * s, elementwise, all the same shape; `out` must be a
    // third tensor.
    bool AddScaled(const Tensor& a, const Tensor& b, float s, Tensor* out);

    // The map's c channels at each of n points, as PyTorch's grid_sample
    // (bilinear, align_corners=False, zero padding): `xy` holds n (x, y)
    // pairs in normalised coordinates, -1..1 across the map's extent. `out`
    // becomes 1 x n x c -- a row per point.
    bool SampleBilinear(const Tensor& map, const std::vector<float>& xy, Tensor* out);

    // --- for transformers -----------------------------------------------------
    //
    // A set of n points with d features each is a d x 1 x n tensor -- channel
    // major, like an image one pixel tall -- so that a linear layer is
    // exactly Conv2d with a 1x1 kernel (Conv::FromLinear) and every image
    // operation above applies unchanged.

    // GELU(LayerNorm(in)) across the channels of each point, with the norm's
    // scale and shift: eps 1e-5 and the exact (erf) GELU, PyTorch's defaults.
    bool LayerNormGelu(const Tensor& in, const Vec& gamma, const Vec& beta, Tensor* out);

    // Rotary position encoding, in place on channels [off, off + count) of
    // `t`, in heads of `headDim`: within each head, channel pair (2j, 2j+1)
    // is rotated by the angle freq[j] . (x, y) of the point's position. `pos`
    // is 2 x 1 x n; `freq` is headDim/2 rows of (fx, fy). LightGlue's
    // apply_cached_rotary_emb, the same angles for every head.
    bool Rotary(Tensor* t, int off, int count, int headDim, const Tensor& pos, const Vec& freq);

    // Multi-head scaled dot-product attention: for each head h and query
    // point i, out[h, :, i] = sum over key points j of softmax_j(q.k / sqrt
    // (headDim)) v[h, :, j]. q, k and v are channel ranges (heads * headDim
    // wide, from the given offsets) of points tensors; k and v share a count,
    // which may differ from q's (cross attention).
    bool Attention(const Tensor& q, int qOff, const Tensor& k, int kOff, const Tensor& v,
                   int vOff, int heads, int headDim, Tensor* out, int outOff);

    // out (1 x m x n) = scale * a^T b, for points tensors a (d x 1 x m) and
    // b (d x 1 x n): every point of one set against every point of the other.
    bool MatMulTN(const Tensor& a, const Tensor& b, float scale, Tensor* out);

    // On a similarity matrix sim (1 x m x n), the DUAL SOFTMAX
    //   P = softmax over j (sim) * softmax over i (sim)
    // and each row's and each column's best: best0[i] = argmax_j P[i][j],
    // p0[i] its value, best1[j] = argmax_i P[i][j]. P itself is never
    // stored -- only the row and column normalisers it needs.
    bool DualSoftmaxBest(const Tensor& sim, std::vector<int>* best0, std::vector<float>* p0,
                         std::vector<int>* best1);

    // Submits recorded GPU work and waits. Download does this itself.
    bool Flush();

    // PROFILING: with a map given, every GPU operation is submitted and
    // waited for on its own, and its milliseconds added under its kind
    // ("conv3x3", "dw5x5", "bilinear", ...). Serialising costs throughput, so
    // the total runs slower than unprofiled -- the split is what it is for.
    void Profile(std::map<std::string, double>* ms) { m_profile = ms; }

private:
    bool Fail(const std::string& e) { m_err = e; return false; }

    template <class F>
    bool Timed(const char* op, F&& run) {
        if (!m_profile) return run();
        if (!Flush()) return false;
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = run() && Flush();
        (*m_profile)[op] += std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t0).count();
        return ok;
    }
    std::map<std::string, double>* m_profile = nullptr;

    ComputeContext*             m_gpu = nullptr;
    std::unique_ptr<GpuKernels> m_k;
    std::string                 m_err;
};

} // namespace nn
} // namespace tglab
