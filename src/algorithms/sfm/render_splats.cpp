// render_splats — the splats as each camera would see them.
//
// The other half of judging a trained splat set. The 3D viewer shows it from
// wherever you orbit to; this renders it from exactly the cameras that took
// the photographs, so a render sits beside its photograph and the difference
// is what training has and has not learned.
//
// SAME RASTERISER AS TRAINING, deliberately. The viewer draws splats through
// its own vertex and pixel shaders, which is fast and approximate; training
// scores against its own forward pass. A render that came from anywhere else
// would be judging the training by a different picture than the one it
// optimised. So on a device this IS training's forward pass
// (SplatTrainerGpu::Render), and without one the CPU SplatRaster it is
// checked against in test_sfm.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "../../algo_util/splat_raster.h"
#include "../../algo_util/splat_reflect.h"
#include "../../algo_util/splat_train_gpu.h"
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
        std::vector<ReflParam> refl;
        EnvMap env;
        ReflFromCloud(cloud, &refl, &env);   // empty when it has no reflections
        const double bg = double(m_background);
        opt.background = Vec3{bg, bg, bg};
        const double scale = std::clamp(double(m_scale), 0.05, 1.0);

        // ON THE DEVICE, through training's own forward pass, when there is
        // one: the splats go up once and every camera is a projection, a
        // tile sort and a composite. The path below instead projects every
        // Gaussian on the CPU for every camera -- 440 ms a camera for a
        // million Gaussians, 44 s for a hundred-frame video -- and stays as
        // the fallback, and as the reference the device path must match.
        std::unique_ptr<SplatTrainerGpu> dev;
        std::string devNote;
        if (ComputeContext* g = GroupGpu()) {
            auto t = std::make_unique<SplatTrainerGpu>(g);
            const size_t n = params.size();
            const std::vector<double> z14(n * 14, 0.0), z1(n, 0.0);
            const std::vector<int> zc(n, 0);
            bool ok = t->Upload(params, z14, z14, z1, zc, z1, &devNote);
            if (ok && cloud.shDegree > 0) {
                const std::vector<double> sh(cloud.splatSh.begin(), cloud.splatSh.end());
                const std::vector<double> zs(sh.size(), 0.0);
                ok = t->UploadSh(sh, zs, zs, &devNote);
            }
            if (ok && !refl.empty()) {
                const std::vector<double> zr(n * 4, 0.0);
                ok = t->UploadRefl(refl, zr, zr, &devNote);
            }
            if (ok) dev = std::move(t);
        }
        int onDevice = 0;

        out->images.clear();
        int rendered = 0;
        for (const Camera& c : cloud.cameras) {
            const int w = std::max(1, int(std::lround(c.width * scale)));
            const int h = std::max(1, int(std::lround(c.height * scale)));
            Image im;
            im.Alloc(ImageDesc{w, h, Format::RGBA8});
            ImageView v = im.MapCpuWrite();

            if (c.solved && v.Valid()) {
                std::vector<double> rgb;
                // Each camera sees each Gaussian's colour for its own
                // direction, and reflections along its own mirror rays.
                const SplatCam sc = SplatCamFrom(c, w, h);
                bool done = false;
                if (dev) {
                    done = dev->Render(sc, opt, cloud.shDegree, refl.empty() ? nullptr : &env,
                                       &rgb, &devNote);
                    if (done) ++onDevice; else dev.reset();   // the rest on the CPU
                }
                if (!done) {
                    SplatRaster r;
                    r.SetGpu(GroupGpu());
                    RenderShaded(r, params, cloud.splatSh, cloud.shDegree, refl, env, sc, opt,
                                 &rgb);
                }
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

        char buf[320];
        std::snprintf(buf, sizeof(buf),
                      "render_splats: %d Gaussians from %d camera%s, %d on the GPU%s%s",
                      int(cloud.splats.size()), rendered, rendered == 1 ? "" : "s",
                      onDevice, devNote.empty() ? "" : " -- ", devNote.c_str());
        m_note = buf;
        return true;
    }

    std::string RunReport() const override { return m_note; }

private:
    Param<float> m_scale{this, "scale", 1.0f, 0.05f, 1.0f,
        {.help = "Render size as a fraction of the resolution the cameras were "
                 "solved at. Rendered through training's own forward pass, on "
                 "the GPU when there is one; without one it falls back to the "
                 "CPU reference, where a full-size render of a large splat set "
                 "takes a while.",
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
