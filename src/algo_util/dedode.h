// dedode — the network shape shared by DaD and the DeDoDe descriptor
// (Edstedt, Bökman, Wadenbäck & Felsberg, "DeDoDe: Detect, Don't Describe --
// Describe, Don't Detect", 3DV 2024). DaD is built from DeDoDe's code, so the
// detector and the descriptor LoMa uses are one architecture with different
// sizes and a different number of output channels.
//
//   encoder   the front of a VGG classifier: 3x3 convolutions, BatchNorm and
//             ReLU, a 2x2 max-pool between blocks. The features at the end of
//             each block are kept -- full, 1/2, 1/4 and 1/8 resolution.
//   decoder   coarse to fine. At each scale a "refiner" reads that scale's
//             features, concatenated with a context passed down from the
//             coarser one, and outputs `outDim` channels of result plus the
//             next context. The results are SUMMED across scales, each sum
//             upsampled to the next: the coarse levels say roughly, the fine
//             ones refine.
//   refiner   a 1x1 convolution block; N blocks of a 5x5 depthwise and a 1x1
//             convolution; a residual connection scaled by 1/1.4; a 1x1 out.
//
// The two members of the family:
//
//                 encoder       refiner blocks   out    accumulated by
//   DaD           VGG11, 6 conv       3            1    bicubic upsampling
//   DeDoDe-B      VGG19, 12 conv      5          128    bilinear
//
// Neither the VGG depth nor the block count is configured: both are read
// from the weight file. Conv layers sit at "encoder.layers.<i>" with each
// conv-BN-ReLU taking three indices, so a gap of four between two conv
// indices is a max-pool -- the end of a block.
#pragma once

#include <chrono>
#include <cstdio>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "nn.h"

namespace tglab {

using NnTaps = std::map<std::string, std::vector<float>>;

class DedodeNet {
public:
    // `outDim` result channels, summed across scales with `accumulate`.
    bool Load(const nn::WeightFile& f, int outDim, nn::Interp accumulate, std::string* err);
    bool Loaded() const { return !m_blocks.empty(); }
    int  OutDim() const { return m_outDim; }

    // The outDim x h x w result for a 3 x h x w input, left on the engine's
    // backend. h and w must divide by 8. `taps` receives feat1..feat8 and
    // delta8..delta1 (each scale's contribution) when given.
    bool Run(nn::Engine& e, const nn::Tensor& input, nn::Tensor* out, std::string* err,
             NnTaps* taps = nullptr) const;

private:
    struct Refiner {
        nn::Conv in1, in2, out;
        std::vector<nn::Conv> dw, pw;
    };
    bool LoadRefiner(const nn::WeightFile& f, const std::string& name, Refiner* r,
                     std::string* err);
    bool RunRefiner(nn::Engine& e, const Refiner& r, const nn::Tensor& in, nn::Tensor* out) const;

    std::vector<std::vector<nn::Conv>> m_blocks;   // encoder, one entry per scale
    Refiner    m_dec[4];                           // scales 8, 4, 2, 1
    int        m_outDim = 0;
    nn::Interp m_accumulate = nn::Interp::Bilinear;
};

// A network loaded once from models/<file> and shared by every stage
// instance and every frame -- megabytes that never change. Thread-safe;
// the first call loads, and a failure is remembered (with `howToMake` in the
// message) rather than retried on every frame. One slot per network type.
template <class Net>
const Net* SharedNetwork(const char* file, const char* howToMake, std::string* err) {
    struct Slot {
        std::mutex  mtx;
        bool        tried = false;
        std::string error;
        Net         net;
    };
    static Slot s;
    std::lock_guard<std::mutex> lock(s.mtx);
    if (!s.tried) {
        s.tried = true;
        const std::string path = nn::FindModelFile(file);
        nn::WeightFile f;
        if (path.empty())
            s.error = std::string("models/") + file + " not found -- make it with '" + howToMake + "'";
        else if (f.Load(path, &s.error))
            s.net.Load(f, &s.error);
    }
    if (!s.net.Loaded()) {
        *err = s.error;
        return nullptr;
    }
    return &s.net;
}

// Where a network stage's time went, for its run report: preparing the image
// on the CPU, waiting for the device (frames run in parallel and share one
// queue -- see GpuLock), and the network itself.
struct NnTiming {
    using Clock = std::chrono::steady_clock;
    Clock::time_point start = Clock::now(), mark = start;
    double prep = 0.0, wait = 0.0, net = 0.0;

    // Milliseconds since the previous Lap, added to `slot`.
    void Lap(double* slot) {
        const Clock::time_point now = Clock::now();
        *slot += std::chrono::duration<double, std::milli>(now - mark).count();
        mark = now;
    }
    std::string Text() const {
        char b[96];
        std::snprintf(b, sizeof b, "prep %.0f ms, waited %.0f ms, network %.0f ms", prep, wait, net);
        return b;
    }
};

// --- preprocessing, shared by the family -------------------------------------

class PixelBuffer;

// An image as the networks were trained to see it: planar RGB in 0..1,
// display-referred. A scene-linear image (a raw) is scaled so its 99th
// percentile reaches 1 -- Percentile99, as the designed detectors normalise --
// and given the sRGB curve; fed linear, a network would see a dark, flat
// picture unlike anything in its training data. Grey is replicated to RGB.
std::vector<float> NetworkRgb(const PixelBuffer& in, bool linear);

// Planar channels resampled the way PIL's Image.resize(BICUBIC) does it,
// which is what the networks saw in training: a separable cubic (a = -0.5)
// whose support WIDENS by the scale factor when shrinking, so a downscale is
// antialiased rather than aliased. The result is clamped to 0..1.
std::vector<float> ResizePil(const std::vector<float>& planar, int channels, int w, int h,
                             int nw, int nh);

// LoMa-B128's descriptor: DeDoDe-B with 128 channels, trained by the LoMa
// authors. Describes points from ANY detector -- it reads a dense map of
// descriptions and samples it where it is told.
class DedodeDescriptor {
public:
    // From a .tgw: tools/nn_convert.py loma_b128 ... --prefix _descriptor.
    bool Load(const nn::WeightFile& f, std::string* err);
    bool Loaded() const { return m_net.Loaded(); }
    int  Dim() const { return m_net.OutDim(); }

    // `rgb` is 3 x h x w in 0..1, NOT normalised: LoMa feeds this network the
    // raw image (unlike DaD). `xy` holds n points in normalised coordinates,
    // -1..1 across the image. `desc` receives n x Dim() descriptions, raw --
    // LoMa's matcher projects them itself, and L2-normalising is the
    // consumer's choice.
    bool Describe(nn::Engine& e, const std::vector<float>& rgb, int h, int w,
                  const std::vector<float>& xy, std::vector<float>* desc, std::string* err,
                  NnTaps* taps = nullptr) const;

private:
    DedodeNet m_net;
};

} // namespace tglab
