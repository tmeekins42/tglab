// dad — the DaD keypoint detector (Edstedt, Bökman, Wadenbäck & Felsberg,
// "DaD: Distilled Reinforcement Learning for Diverse Keypoint Detection",
// 2025), the detector LoMa uses.
//
// WHAT IT LEARNED. Classic detectors decide what a keypoint is by formula --
// a corner (FAST, Harris), a blob (SIFT's difference of Gaussians). DaD was
// trained by reinforcement learning to pick points that are found AGAIN in
// another view of the same scene, then distilled into one network. It has no
// notion of corner or blob; it scores every pixel by how reliably it expects
// that pixel to be re-detected, and the points are the peaks of that score.
//
// THE NETWORK is small by today's standards, 6.5M weights: VGG11's front end
// and a coarse-to-fine decoder whose one output channel is the score, the
// scales' contributions SUMMED -- the coarse levels say where, the fine ones
// say exactly where. It is the DeDoDe architecture; see dedode.h.
//
// THEN, not learned: a softmax over every pixel, non-maximum suppression in
// 3x3, the top N, and a sub-pixel refinement from the 3x3 of scores around
// each peak. See DadSample.
#pragma once

#include <map>
#include <string>
#include <vector>

#include "dedode.h"

namespace tglab {

class DadNet {
public:
    // From a .tgw written by tools/nn_convert.py dad.
    bool Load(const nn::WeightFile& f, std::string* err);
    bool Loaded() const { return m_net.Loaded(); }

    // The per-pixel score map (logits) for an image given as NORMALISED RGB:
    // 3 x h x w, ImageNet mean subtracted and divided by its deviation, h and
    // w multiples of 8. `taps`, when given, receives intermediate tensors by
    // the names tools/nn_reference.py uses, for comparing against PyTorch.
    bool Logits(nn::Engine& e, const std::vector<float>& input, int h, int w,
                std::vector<float>* logits, std::string* err,
                NnTaps* taps = nullptr) const;

private:
    DedodeNet m_net;
};

// One detected point, in the pixel coordinates of the image the network saw.
struct DadPoint {
    float x = 0.0f, y = 0.0f;
    float prob = 0.0f;   // its share of the softmax over the whole image
};

// The published sampling: softmax over all pixels, 3x3 non-maximum
// suppression, the `n` strongest, each refined by the softmax-weighted mean
// of the 3x3 offsets around it (temperature 0.5). Strongest first.
std::vector<DadPoint> DadSample(const std::vector<float>& logits, int h, int w, int n);

// RGB in 0..1 (display-referred, as the network was trained on) to the
// normalised input Logits expects.
void DadNormalise(std::vector<float>* rgbPlanar);

// The size the published code feeds the network: the long side scaled to
// `longSide`, then each side rounded DOWN to a multiple of 8 on its own --
// so the aspect ratio is kept only approximately, exactly as there.
void DadNetworkSize(int w, int h, int longSide, int* nw, int* nh);


} // namespace tglab
