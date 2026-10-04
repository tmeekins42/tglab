// detect_dad — the DaD keypoint detector (Edstedt et al., 2025): LEARNED
// rather than designed. See algo_util/dad.h for how the network works.
//
// WHY IT IS HERE. It is the detector of LoMa (Edstedt, Nordström et al.,
// "LoMa: Local Feature Matching Revisited", 2026), which matches better than
// any designed pipeline on every benchmark that paper reports -- and whose
// claim is that the gap is mostly training data, not architecture. Each of
// LoMa's three parts arrives as its own stage, so each can be compared with
// the designed one it replaces: this against detect_akaze and detect_sift,
// with draw_features to see where the points land.
//
// KEYPOINTS ONLY. DaD finds points; describing them is DeDoDe's job, a
// separate network and a separate stage. So this sidecar carries no
// descriptors yet, and a matcher downstream of it has nothing to compare.
//
// THE NETWORK SEES A RESIZED IMAGE -- the long side at `resolution`, 1024 by
// default as published -- and positions are scaled back to the full image.
// Detection resolution is a real choice: the network was trained around 1024
// pixels, and a 6000-pixel raw at full size would ask it to find structure at
// a scale it never saw, at 35 times the cost.
//
// THE WEIGHTS are a file, not part of the executable: models/dad.tgw, written
// by tools/nn_convert.py from the published checkpoint (see nn.h).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "../../algo_util/dad.h"
#include "../../algo_util/features.h"
#include "../../algo_util/nn.h"
#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"
#include "gpu_pyramid.h"

namespace tglab {
class DetectDad : public AlgorithmBase {
public:
    const char* Name()     const override { return "detect_dad"; }
    const char* Category() const override { return "features"; }

    PortList Inputs()  const override { return {{"src", DataType::Image, FormatSpec::Any}}; }
    PortList Outputs() const override {
        return {{"out", DataType::Image, FormatSpec::SameAsInput}};
    }

    void RunCPU(RunCtx& ctx) override {
        const ImageView src = ctx.In(0);
        ImageView       dst = ctx.Out(0);
        if (!src.Valid() || !dst.Valid()) return;

        PixelBuffer in;
        in.Unpack(src);
        if (!in.Valid()) return;
        {
            PixelBuffer out;
            out.Unpack(dst);
            if (out.Valid()) {
                out.Data() = in.Data();
                out.PackInto(dst);
            }
        }
        const int w = in.Width(), h = in.Height();
        if (w < 32 || h < 32) return;

        std::string err;
        const DadNet* net = SharedNetwork<DadNet>(
            "dad.tgw", "python tools/nn_convert.py dad models/dad.tgw", &err);
        if (!net) {
            ctx.SetReport("detect_dad: " + err);
            return;
        }

        // Display-referred RGB in 0..1, as the network was trained on: see
        // NetworkRgb.
        NnTiming tm;
        const std::vector<float> rgb = NetworkRgb(in, src.desc.linear);

        int nw = 0, nh = 0;
        DadNetworkSize(w, h, int(m_resolution), &nw, &nh);
        if (nw < 16 || nh < 16) return;
        std::vector<float> input = ResizePil(rgb, 3, w, h, nw, nh);
        DadNormalise(&input);
        tm.Lap(&tm.prep);

        // On the device when there is one. The lock covers the whole network
        // -- one logical operation -- and nothing else, so frames still
        // prepare and sample concurrently. See GpuLock.
        std::vector<float> logits;
        bool ok = false;
        bool onGpu = false;
        if (ComputeContext* dev = ctx.Gpu()) {
            GpuLock lock(dev);
            tm.Lap(&tm.wait);
            nn::Engine eng(dev);
            ok = net->Logits(eng, input, nh, nw, &logits, &err);
            onGpu = ok;
            tm.Lap(&tm.net);
        }
        if (!ok) {
            nn::Engine eng;
            ok = net->Logits(eng, input, nh, nw, &logits, &err);
            tm.Lap(&tm.net);
        }
        if (!ok) {
            ctx.SetReport("detect_dad: " + err);
            return;
        }
        if (ctx.Cancelled()) return;

        const std::vector<DadPoint> pts = DadSample(logits, nh, nw, int(m_maxFeatures));

        auto sidecar = std::make_shared<FeatureSidecar>();
        sidecar->detector = "dad";
        sidecar->descriptors.kind = DescriptorKind::None;
        const float sx = float(w) / float(nw), sy = float(h) / float(nh);
        sidecar->keypoints.reserve(pts.size());
        for (const DadPoint& d : pts) {
            Keypoint k;
            k.x = d.x * sx;
            k.y = d.y * sy;
            // DaD has no scale. Nominally one network pixel, for drawing.
            k.scale = 0.5f * (sx + sy);
            k.response = d.prob;
            sidecar->keypoints.push_back(k);
        }

        char rbuf[256];
        std::snprintf(rbuf, sizeof rbuf, "%zu DaD keypoints at %dx%d%s (%s)", pts.size(), nw, nh,
                      onGpu ? "" : " on the CPU", tm.Text().c_str());
        ctx.SetReport(rbuf);
        if (Image* im = ctx.OutImage(0)) im->Sidecars().Set(kFeatureSidecar, sidecar);
    }

    // Image-pixel coordinates; see detect_orb.
    ProxyBehaviour Proxy() const override { return ProxyBehaviour::Never; }
    bool HasGPU() const override { return false; }
    bool UsesGpuInRunCPU() const override { return true; }
    bool LocksGpuInRunCPU() const override { return true; }   // the network only, under GpuLock

private:
    Param<int> m_maxFeatures{this, "max_features", 4096, 64, 20000,
        {.help = "How many keypoints to keep: the strongest, after 3x3 "
                 "non-maximum suppression. LoMa matches 4096 per image when "
                 "it is evaluated, and trains on 2048."}};

    Param<int> m_resolution{this, "resolution", 1024, 256, 2048,
        {.help = "The long side of the image the network sees, in pixels; "
                 "keypoints are scaled back to the full image. 1024 is what "
                 "it was published at. Higher finds finer structure at the "
                 "square of the cost, and drifts from what it was trained "
                 "on."}};
};

REGISTER_ALGORITHM(DetectDad);

} // namespace tglab
