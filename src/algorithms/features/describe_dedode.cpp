// describe_dedode — LEARNED descriptors for keypoints found by any detector:
// LoMa-B128's DeDoDe-B, 128 floats per point. See algo_util/dedode.h.
//
// DESCRIBING IS ITS OWN STAGE, which is the point of DeDoDe's title ("Detect,
// Don't Describe -- Describe, Don't Detect"): the network computes a
// description for every pixel and the keypoints only say where to read it.
// So it describes whatever is upstream -- detect_dad, as LoMa pairs them, or
// AKAZE's or SIFT's points, which is a comparison the lab can now make: the
// same points, a designed descriptor against a learned one.
//
// THE IMAGE IS SQUASHED TO A SQUARE, `resolution` on a side, 784 by default --
// exactly what LoMa does. The network was trained on squares at 560 and
// evaluated at 784, so it has learned to describe a stretched picture;
// keeping the aspect ratio would hand it one it has never seen. Positions
// are normalised, so the stretch does not move them.
//
// NORMALISED BY DEFAULT. DeDoDe's own matcher compares descriptors by
// cosine, and their raw lengths vary by half again across one image (median
// 14.8, p10 12.4, p90 18.0, measured on fountain-P11), so match_ann's L2 on
// raw vectors would rank by length as much as by direction. LoMa's matcher,
// in contrast, was trained on the RAW vectors -- turn `normalise` off for it.
//
// THE WEIGHTS: models/dedode_b128.tgw, 27 MB at float16, from
//   python tools/nn_convert.py loma_b128 models/dedode_b128.tgw --prefix _descriptor.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "../../algo_util/dedode.h"
#include "../../algo_util/features.h"
#include "../../algo_util/nn.h"
#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"
#include "gpu_pyramid.h"

namespace tglab {

class DescribeDedode : public AlgorithmBase {
public:
    const char* Name()     const override { return "describe_dedode"; }
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

        const Image* im = ctx.InImage(0);
        const FeatureSidecar* fs = im ? FeaturesOf(*im) : nullptr;
        if (!fs) {
            ctx.SetReport("no keypoints attached -- run a detector first");
            return;
        }
        const int w = in.Width(), h = in.Height();
        const int side = std::max(16, int(m_resolution) / 8 * 8);

        std::string err;
        const DedodeDescriptor* net = SharedNetwork<DedodeDescriptor>(
            "dedode_b128.tgw",
            "python tools/nn_convert.py loma_b128 models/dedode_b128.tgw --prefix _descriptor.",
            &err);
        if (!net) {
            ctx.SetReport("describe_dedode: " + err);
            return;
        }

        // Positions normalised to -1..1 across the image, pixel centres at
        // +0.5 (the convention the camera model and DaD share).
        const size_t n = fs->keypoints.size();
        std::vector<float> xy(2 * n);
        for (size_t i = 0; i < n; ++i) {
            xy[2 * i]     = 2.0f * fs->keypoints[i].x / float(w) - 1.0f;
            xy[2 * i + 1] = 2.0f * fs->keypoints[i].y / float(h) - 1.0f;
        }

        NnTiming tm;
        const std::vector<float> rgb =
            ResizePil(NetworkRgb(in, src.desc.linear), 3, w, h, side, side);
        tm.Lap(&tm.prep);

        std::vector<float> desc;
        bool ok = false, onGpu = false;
        if (n > 0) {
            if (ComputeContext* dev = ctx.Gpu()) {
                GpuLock lock(dev);   // the whole network, nothing else: see GpuLock
                tm.Lap(&tm.wait);
                nn::Engine eng(dev);
                ok = onGpu = net->Describe(eng, rgb, side, side, xy, &desc, &err);
                tm.Lap(&tm.net);
            }
            if (!ok) {
                nn::Engine eng;
                ok = net->Describe(eng, rgb, side, side, xy, &desc, &err);
                tm.Lap(&tm.net);
            }
            if (!ok) {
                ctx.SetReport("describe_dedode: " + err);
                return;
            }
        }
        if (ctx.Cancelled()) return;

        const int dim = net->Dim();
        if (m_normalise) {
            for (size_t i = 0; i < n; ++i) {
                float* d = &desc[i * size_t(dim)];
                double s = 0.0;
                for (int k = 0; k < dim; ++k) s += double(d[k]) * d[k];
                const float inv = s > 0.0 ? float(1.0 / std::sqrt(s)) : 0.0f;
                for (int k = 0; k < dim; ++k) d[k] *= inv;
            }
        }

        // The upstream keypoints, unchanged, with these descriptors in place
        // of whatever they carried.
        auto sidecar = std::make_shared<FeatureSidecar>();
        sidecar->keypoints = fs->keypoints;
        sidecar->detector = fs->detector + "+dedode";
        sidecar->descriptors.kind = DescriptorKind::Float;
        sidecar->descriptors.dim  = dim;
        sidecar->descriptors.f    = std::move(desc);

        char rbuf[256];
        std::snprintf(rbuf, sizeof rbuf, "described %zu %s keypoints, %d floats each, at %dx%d%s (%s)",
                      n, fs->detector.c_str(), dim, side, side, onGpu || n == 0 ? "" : " on the CPU",
                      tm.Text().c_str());
        ctx.SetReport(rbuf);
        if (Image* out = ctx.OutImage(0)) out->Sidecars().Set(kFeatureSidecar, sidecar);
    }

    // Image-pixel coordinates; see detect_orb.
    ProxyBehaviour Proxy() const override { return ProxyBehaviour::Never; }
    bool HasGPU() const override { return false; }
    bool UsesGpuInRunCPU() const override { return true; }
    bool LocksGpuInRunCPU() const override { return true; }   // the network only, under GpuLock

private:
    Param<int> m_resolution{this, "resolution", 784, 256, 1568,
        {.help = "The side of the square the network describes, in pixels. "
                 "784 is what LoMa evaluates at; the network was trained at "
                 "560. Higher resolves finer detail at the square of the cost."}};

    Param<bool> m_normalise{this, "normalise", true,
        "Scale each descriptor to unit length, so matching compares "
                 "direction only -- what DeDoDe's own matcher does, and what "
                 "match_ann's distance wants. LoMa's matcher was trained on "
                 "the raw vectors, so turn this off for it."};
};

REGISTER_ALGORITHM(DescribeDedode);

} // namespace tglab
