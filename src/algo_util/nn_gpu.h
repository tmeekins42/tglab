// nn_gpu — the device side of nn::Engine. Internal to nn.cpp and nn_gpu.cpp.
#pragma once

#include <string>
#include <vector>

#include "nn.h"

namespace tglab {
namespace nn {

// The compute kernels, compiled on first use. Every method mirrors the
// Engine operation of the same name and has the same contract.
class GpuKernels {
public:
    explicit GpuKernels(ComputeContext* gpu);
    ~GpuKernels();

    bool Conv2d(const Tensor& in, const Conv& cv, bool relu, Tensor* out, int outOff,
                std::string* err);
    bool MaxPool2(const Tensor& in, Tensor* out, std::string* err);
    bool Resize(const Tensor& in, int inOff, int count, Interp mode, Tensor* out,
                int outOff, std::string* err);
    bool CopyAcc(const Tensor& src, int srcOff, int count, Tensor* dst, int dstOff,
                 bool accumulate, std::string* err);
    bool AddScaled(const Tensor& a, const Tensor& b, float s, Tensor* out, std::string* err);
    bool LayerNormGelu(const Tensor& in, const Vec& gamma, const Vec& beta, Tensor* out,
                       std::string* err);
    bool Rotary(Tensor* t, int off, int count, int headDim, const Tensor& pos, const Vec& freq,
                std::string* err);
    bool Attention(const Tensor& q, int qOff, const Tensor& k, int kOff, const Tensor& v,
                   int vOff, int heads, int headDim, Tensor* out, int outOff, std::string* err);
    bool MatMulTN(const Tensor& a, const Tensor& b, float scale, Tensor* out, std::string* err);
    bool DualSoftmaxBest(const Tensor& sim, std::vector<int>* best0, std::vector<float>* p0,
                         std::vector<int>* best1, std::string* err);

    // `pts` holds n (x, y) pairs; see Engine::SampleBilinear.
    bool SampleBilinear(const Tensor& map, const Tensor& pts, int n, Tensor* out,
                        std::string* err);

private:
    struct Impl;
    Impl* m;
};

} // namespace nn
} // namespace tglab
