// loma — LoMa's matcher (Edstedt, Nordström et al., "LoMa: Local Feature
// Matching Revisited", 2026): a LightGlue-style transformer that decides
// which keypoints of two images are the same point, reading every keypoint's
// descriptor AND where it sits AND what the other points around it look like.
//
// WHY A NETWORK RATHER THAN NEAREST NEIGHBOURS. match_ann compares two
// descriptors in isolation: a window corner matches the most similar window
// corner, and on a facade of identical windows that is a coin toss the ratio
// test then throws away. Here each point's description is rewritten, layer by
// layer, by attention -- first to the other points of its own image (self),
// which tells it which window it is, then to the points of the other image
// (cross), which tells it which of the candidates there tell the same story.
// After nine layers the descriptions are compared once, all against all.
//
// THE NETWORK, LoMa-B128 (12M weights):
//
//   input      a linear layer, 128 descriptor floats -> 256
//   9 layers   SELF: a linear layer makes a query, key and value per point;
//              the queries and keys are rotated by the point's position
//              (rotary encoding -- so attention sees RELATIVE positions);
//              4-head attention; a linear layer; then x + MLP([x, message])
//              with the MLP Linear 512 -> LayerNorm -> GELU -> Linear 256.
//              CROSS: as self, between the images, with one projection
//              shared by queries and keys and no positions.
//   assignment a linear layer on both sides, every point against every
//              point (a similarity matrix), the DUAL SOFTMAX -- a row
//              softmax times a column softmax, so a pair scores high only if
//              each is the other's clear favourite -- and mutual best
//              matches above 0.1 kept.
//
// Each layer has its own assignment head, trained so the matcher can stop
// early: `layers` below 9 trades accuracy for time, as LoMa's paper does.
#pragma once

#include <string>
#include <vector>

#include "dedode.h"
#include "nn.h"

namespace tglab {

class LomaMatcher {
public:
    // From a .tgw: tools/nn_convert.py loma_b128 ... --exclude _detector.
    // --exclude _descriptor.
    bool Load(const nn::WeightFile& f, std::string* err);
    bool Loaded() const { return !m_layers.empty(); }
    int  Layers() const { return int(m_layers.size()); }
    int  InputDim() const { return m_inProj.cin; }

    struct Pair {
        int   i = -1, j = -1;   // keypoint in image 0, keypoint in image 1
        float score = 0.0f;     // the dual-softmax probability
    };

    // kp0 / kp1: n x 2 positions NORMALISED to -1..1 across each image.
    // desc0 / desc1: n x InputDim() descriptors, RAW (not unit length --
    // the network was trained on DeDoDe's own scale). Points-major.
    // `taps` receives proj0/1 and layer<i>_0/1 (points-major, as the
    // reference saves them) and sim, for comparing against PyTorch.
    bool Match(nn::Engine& e, const std::vector<float>& kp0, const std::vector<float>& desc0,
               const std::vector<float>& kp1, const std::vector<float>& desc1, int layers,
               float threshold, std::vector<Pair>* out, std::string* err,
               NnTaps* taps = nullptr) const;

private:
    struct Ffn {
        nn::Conv l0, l3;
        nn::Vec  gamma, beta;
    };
    struct Layer {
        nn::Conv qkv, outProj;   // qkv's rows reordered to [q | k | v]
        Ffn      selfFfn;
        nn::Conv toQk, toV, toOut;
        Ffn      crossFfn;
    };

    bool LoadFfn(const nn::WeightFile& f, const std::string& name, Ffn* ffn,
                 std::string* err) const;
    bool RunFfn(nn::Engine& e, const Ffn& ffn, const nn::Tensor& x, const nn::Tensor& message,
                const nn::Conv& msgProj, nn::Tensor* out) const;
    bool SelfBlock(nn::Engine& e, const Layer& L, const nn::Tensor& x, const nn::Tensor& pos,
                   nn::Tensor* out) const;
    bool CrossBlock(nn::Engine& e, const Layer& L, const nn::Tensor& x0, const nn::Tensor& x1,
                    nn::Tensor* out0, nn::Tensor* out1) const;

    nn::Conv              m_inProj;
    nn::Vec               m_freq;   // rotary frequencies, headDim/2 rows of (fx, fy)
    std::vector<Layer>    m_layers;
    std::vector<nn::Conv> m_assign; // per layer: the final projection
    int m_dim = 0, m_heads = 0, m_headDim = 0;
};

} // namespace tglab
