// render_splats — the splats as each camera would see them.
//
// The other half of judging a trained splat set. The 3D viewer shows it from
// wherever you orbit to; this renders it from exactly the cameras that took
// the photographs, so a render sits beside its photograph and the difference
// is what training has and has not learned.
//
// SAME RASTERISER AS TRAINING, deliberately. The viewer draws splats through
// its own vertex and pixel shaders, which is fast and approximate; training
// scores against SplatRaster. A render that came from anywhere else would be
// judging the training by a different picture than the one it optimised.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "../../algo_util/splat_raster.h"
#include "../../core/algorithm.h"

namespace tglab {
namespace {

class RenderSplats : public AlgorithmBase {
public:
    const char* Name()     const override { return "render_splats"; }
    const char* Category() const override { return "sfm"; }

    PortList Inputs() const override {
        return {{"src", DataType::PointCloud, FormatSpec::Any, ShapeSpec::Any}};
    }
    PortList Outputs() const override {
        return {{"out", DataType::ImageSet, FormatSpec::RGBA8, ShapeSpec::Any}};
    }

    void RunCPU(RunCtx&) override {}
    bool IsReconstruct() const override { return true; }
    ProxyBehaviour Proxy() const override { return ProxyBehaviour::Never; }
    bool HasGPU() const override { return false; }

    bool RunDense(const std::vector<Image>*, const PointCloud& cloud,
                  ImageSet* out, std::string* err) override {
        if (cloud.splats.empty()) {
            *err = "render_splats: no splats -- run init_splats first";
            return false;
        }

        std::vector<SplatParam> params;
        params.reserve(cloud.splats.size());
        for (const Splat& s : cloud.splats) params.push_back(ToParam(s));

        RasterOptions opt;
        const double bg = double(m_background);
        opt.background = Vec3{bg, bg, bg};
        const double scale = std::clamp(double(m_scale), 0.05, 1.0);

        out->images.clear();
        int rendered = 0;
        for (const Camera& c : cloud.cameras) {
            const int w = std::max(1, int(std::lround(c.width * scale)));
            const int h = std::max(1, int(std::lround(c.height * scale)));
            Image im;
            im.Alloc(ImageDesc{w, h, Format::RGBA8});
            ImageView v = im.MapCpuWrite();

            if (c.solved && v.Valid()) {
                SplatRaster r;
                r.SetGpu(GroupGpu());
                std::vector<double> rgb;
                r.Forward(params, SplatCamFrom(c, w, h), opt, &rgb);
                for (int y = 0; y < h; ++y)
                    for (int x = 0; x < w; ++x) {
                        uint8_t* p = v.At<uint8_t>(x, y);
                        const double* s = &rgb[(size_t(y) * size_t(w) + size_t(x)) * 3];
                        for (int ch = 0; ch < 3; ++ch)
                            p[ch] = uint8_t(std::clamp(s[ch], 0.0, 1.0) * 255.0 + 0.5);
                        p[3] = 255;
                    }
                ++rendered;
            } else if (v.Valid()) {
                // An unsolved camera has no pose to render from. Kept in the
                // set as black so frame i here is still camera i.
                std::fill(v.data, v.data + size_t(w) * size_t(h) * 4, uint8_t(0));
            }
            out->images.push_back(std::move(im));
        }
        out->shape = Shape::Of("frame", int(out->images.size()));

        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "render_splats: %d Gaussians from %d camera%s",
                      int(cloud.splats.size()), rendered, rendered == 1 ? "" : "s");
        m_note = buf;
        return true;
    }

    std::string RunReport() const override { return m_note; }

private:
    Param<float> m_scale{this, "scale", 1.0f, 0.05f, 1.0f,
        {.help = "Render size as a fraction of the resolution the cameras were "
                 "solved at. The rasteriser is the CPU reference used for "
                 "training, so a full-size render of a large splat set takes a "
                 "while.",
         .step = 0.05}};

    Param<float> m_background{this, "background", 0.0f, 0.0f, 1.0f,
        {.help = "Grey level shown wherever no Gaussian covers a pixel. Should "
                 "match what training used, or uncovered regions compare "
                 "against the wrong colour.",
         .step = 0.05}};

    std::string m_note;
};

REGISTER_ALGORITHM(RenderSplats);

}  // namespace
}  // namespace tglab
