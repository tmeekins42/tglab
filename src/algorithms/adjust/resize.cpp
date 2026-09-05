// resize — scale an image, with the filtering the direction requires.
//
// Written for PROXY RESOLUTION: the pipeline inserts one of these to shrink
// before an interactive run and another to grow the result back. It is a
// general resize and usable directly, but the proxy path is what it exists for
// and what the defaults suit.
//
// DOWN AND UP ARE NOT THE SAME OPERATION, and using one filter for both is the
// classic mistake:
//
//   DOWN needs an AREA average. Point-sampling or bilinear-sampling a
//   minification reads a few pixels and ignores the rest, so detail finer than
//   the new sampling rate aliases -- and aliasing does not merely look wrong,
//   it MOVES when the image does. On a proxy that is worse than softness: a
//   preview that shimmers as a slider drags is unusable for judging anything.
//   Averaging the whole source footprint is what band-limits it properly.
//
//   UP needs interpolation, and bilinear is the right amount here. Something
//   sharper (bicubic, Lanczos) would ring on edges, and since the upsample is
//   only ever showing a preview that is already an approximation, the extra
//   sharpness would be inventing detail the proxy does not have.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "../../algo_util/pixel_buffer.h"
#include "../../algo_util/transform.h"
#include "../../core/algorithm.h"

namespace tglab {
namespace {

class Resize : public AlgorithmBase {
public:
    const char* Name()     const override { return "resize"; }
    const char* Category() const override { return "adjust"; }

    PortList Inputs()  const override {
        return {{"src", DataType::Image, FormatSpec::Any}};
    }
    PortList Outputs() const override {
        return {{"out", DataType::Image, FormatSpec::SameAsInput}};
    }

    ImageDesc OutputDesc(int, const ImageDesc& in) const override {
        ImageDesc d = in;
        const float s = Factor();
        d.width  = std::max(1, int(std::lround(double(in.width)  * double(s))));
        d.height = std::max(1, int(std::lround(double(in.height) * double(s))));

        // The scale factor COMPOUNDS. An image already at 0.5 resized by 0.5 is
        // at 0.25 of the original, and a downstream algorithm scaling its pixel
        // radii needs the total, not the last step.
        d.proxyScale = in.proxyScale * s;

        // THE PLACEMENT SCALES WITH THE RASTER, because origin and full extent
        // are stated in the image's OWN pixels (see ImageDesc). Copying them
        // through unscaled leaves a proxy claiming to be a window onto an
        // 8191 px frame while its own pixels are 0.68 of that size -- and the
        // viewer, which lays out against the full extent, then drew the drag at
        // the wrong scale AND the wrong offset. Only visible in the region
        // path, since without a crop both are zero and nothing is wrong.
        if (in.fullW > 0)
            d.fullW = std::max(1, int(std::lround(double(in.fullW) * double(s))));
        if (in.fullH > 0)
            d.fullH = std::max(1, int(std::lround(double(in.fullH) * double(s))));
        d.originX = int(std::lround(double(in.originX) * double(s)));
        d.originY = int(std::lround(double(in.originY) * double(s)));
        return d;
    }

    bool IsNoOp() const override { return Factor() == 1.0f; }

    // Resizing IS the scale change, so it must not also be scaled by one.
    ProxyBehaviour Proxy() const override { return ProxyBehaviour::Exact; }

    void RunCPU(RunCtx& ctx) override {
        const ImageView src = ctx.In(0);
        ImageView       dst = ctx.Out(0);
        if (!src.Valid() || !dst.Valid()) return;

        PixelBuffer in, out;
        in.Unpack(src);
        out.Unpack(dst);
        if (!in.Valid() || !out.Valid()) return;

        const int sw = in.Width(),  sh = in.Height();
        const int dw = out.Width(), dh = out.Height();
        const int ch = in.Channels();
        if (sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0) return;

        m_w = dw;
        m_h = dh;

        const double xr = double(sw) / double(dw);
        const double yr = double(sh) / double(dh);

        if (dw < sw || dh < sh) {
            // MINIFY: average the source footprint each output pixel covers.
            //
            // The footprint is computed from the output pixel's EDGES rather
            // than its centre plus a fixed radius, so a non-integer ratio still
            // partitions the source exactly -- no source pixel counted twice,
            // none skipped. That is what keeps the average unbiased when the
            // ratio is something like 3.7.
            for (int y = 0; y < dh; ++y) {
                if (ctx.Cancelled()) return;
                const int y0 = std::clamp(int(std::floor(double(y)       * yr)), 0, sh - 1);
                const int y1 = std::clamp(int(std::ceil ((double(y) + 1) * yr)), y0 + 1, sh);

                for (int x = 0; x < dw; ++x) {
                    const int x0 = std::clamp(int(std::floor(double(x)       * xr)), 0, sw - 1);
                    const int x1 = std::clamp(int(std::ceil ((double(x) + 1) * xr)), x0 + 1, sw);

                    double acc[4] = {0, 0, 0, 0};
                    int n = 0;
                    for (int sy = y0; sy < y1; ++sy)
                        for (int sx = x0; sx < x1; ++sx) {
                            const float* p = in.At(sx, sy);
                            for (int c = 0; c < ch; ++c) acc[c] += double(p[c]);
                            ++n;
                        }
                    float* q = out.At(x, y);
                    const double inv = n ? 1.0 / double(n) : 0.0;
                    for (int c = 0; c < ch; ++c) q[c] = float(acc[c] * inv);
                }
            }
        } else {
            // MAGNIFY: bilinear.
            //
            // The half-pixel offsets map pixel CENTRES to pixel centres. Without
            // them the result is shifted by half an output pixel, which on a
            // proxy round trip shows up as the preview sliding as the scale
            // changes -- subtle, and maddening to chase later.
            float sm[4] = {0, 0, 0, 0};
            for (int y = 0; y < dh; ++y) {
                if (ctx.Cancelled()) return;
                const float sy = float((double(y) + 0.5) * yr - 0.5);
                for (int x = 0; x < dw; ++x) {
                    const float sx = float((double(x) + 0.5) * xr - 0.5);
                    SampleBilinear(in, sx, sy, sm);
                    float* q = out.At(x, y);
                    for (int c = 0; c < ch; ++c) q[c] = sm[c];
                }
            }
        }

        out.PackInto(dst);

        // Sidecars do NOT survive a resize.
        //
        // A keypoint list carries coordinates in the pixels of the image it was
        // found on. Passing it through unchanged would leave every position
        // wrong by the scale factor, and silently -- the sidecar would still be
        // there and still look valid. Dropping it makes the loss explicit at
        // the point a downstream stage looks for it.
        //
        // This is also why the detectors are ProxyBehaviour::Never: it is not
        // only that they would find different keypoints, it is that their
        // output cannot be rescaled after the fact.
    }

    std::string RunReport() const override {
        if (m_w <= 0) return {};
        char buf[64];
        std::snprintf(buf, sizeof buf, "resized to %dx%d", m_w, m_h);
        return buf;
    }

    bool HasGPU() const override { return false; }

private:
    float Factor() const {
        return std::clamp(float(m_scale), 0.01f, 8.0f);
    }

    // NO STEP, deliberately.
    //
    // Param snaps to a grid measured from the LOW END, so a step of 0.05 on a
    // range starting at 0.01 puts the grid at 0.01, 0.06, 0.11... -- which
    // never includes 1.0. That would make IsNoOp() unreachable and a requested
    // scale of 0.25 silently become 0.26, and the pipeline's proxy factor is
    // computed from a viewport size rather than dragged, so there is nothing a
    // step would help.
    Param<float> m_scale{this, "scale", 1.0f, 0.01f, 8.0f,
        {.help = "Output size as a fraction of the input. Below 1 the source "
                 "footprint is AREA-AVERAGED, which band-limits it properly; "
                 "sampling instead would alias, and aliasing moves when the "
                 "image does. Above 1 it is bilinear.",
         .softMin = 0.1, .softMax = 2.0}};

    int m_w = 0, m_h = 0;
};

REGISTER_ALGORITHM(Resize);

}  // namespace
}  // namespace tglab
