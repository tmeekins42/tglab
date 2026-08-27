// tonemap: fits a scene-linear image into the range the display curve expects.
//
// A merged bracket carries enormous range -- Tim's seven-frame bracket reaches
// 913 linear with a median near 2.0. The display tone curve in tone_curve.h has
// a shoulder, but its asymptote sits at 1.39, so anything above roughly 19
// linear still hits the clamp. Handing it a 913 means a white image.
//
// So this is a SEPARATE STAGE rather than something merge_hdr does, and rather
// than an operator passed into the merge:
//
//   - it is not only for HDR. A single backlit raw, a focus stack and a
//     panorama all produce scene-linear data that may need compressing.
//   - the merge's job is to produce radiance; deciding how to SHOW radiance is
//     a different question, and fusing them would stop merge_hdr from emitting
//     the linear data that is worth keeping for editing.
//   - swapping this for a local operator should be a one-line script change,
//     which is what the lab is for. As a parameter it would be buried.
//
// The operator itself is global: one curve, every pixel. That means no halos
// and a trivially parallel kernel, at the cost of not recovering LOCAL contrast
// -- a bright window in a dark room stays flat. A local operator is a separate
// algorithm, and comparing the two side by side is the point.
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "../../algo_util/pixel_buffer.h"
#include "../../algo_util/tone_curve.h"
#include "../../core/algorithm.h"

namespace tglab {
namespace {

// Sampled percentiles of the input, for placing the curve.
//
// Measured from the image rather than assumed, because the scale of a merge is
// arbitrary: it depends on the shutter speeds in the bracket, so no fixed
// anchor can be right. Sampling on a stride rather than reading every pixel --
// at 22 MP a full sort costs far more than the answer is worth, and a few
// hundred thousand samples pin a percentile down.
struct Levels {
    float median = 0.0f;
    float high   = 0.0f;   // the brightest real detail, not the brightest pixel
    bool  valid  = false;
};

Levels Measure(const PixelBuffer& pb) {
    Levels out;
    const int w = pb.Width(), h = pb.Height(), ch = pb.Channels();
    if (w <= 0 || h <= 0) return out;

    std::vector<float> lum;
    lum.reserve(400000);
    const int stride = std::max(1, std::min(w, h) / 700);
    for (int y = 0; y < h; y += stride) {
        for (int x = 0; x < w; x += stride) {
            const float* p = pb.At(x, y);
            const float l = (ch >= 3)
                ? 0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2]
                : p[0];
            // Negatives are real -- the demosaic undershoots on near-black
            // pixels and the merge amplifies that -- but they carry no
            // information about where the midtones sit, so they are excluded
            // from the measurement rather than dragging it down.
            if (l > 0.0f) lum.push_back(l);
        }
    }
    if (lum.size() < 1000) return out;

    std::sort(lum.begin(), lum.end());
    auto pct = [&](double f) {
        const size_t i = std::min(lum.size() - 1,
                                  size_t(f * double(lum.size() - 1) + 0.5));
        return lum[i];
    };

    out.median = pct(0.50);
    // p99.5 rather than the maximum: the brightest pixel in a photograph is a
    // specular glint or a light source, and anchoring the shoulder to it would
    // compress the whole picture to protect something that should simply be
    // white.
    out.high  = std::max(pct(0.995), out.median * 1.0001f);
    out.valid = out.median > 0.0f;
    return out;
}

class Tonemap : public AlgorithmBase {
public:
    const char* Name()     const override { return "tonemap"; }
    const char* Category() const override { return "tonemap"; }

    PortList Inputs()  const override { return {{"src"}}; }
    PortList Outputs() const override {
        // Stays float. The result is still linear -- lower-scaled and
        // shoulder-compressed, but linear -- so everything downstream, the
        // display curve included, behaves exactly as it does for a raw.
        return {{"out", DataType::Image, FormatSpec::RGBA32F}};
    }

    void RunCPU(RunCtx& ctx) override {
        const ImageView in = ctx.In(0);
        ImageView out = ctx.Out(0);
        if (!in.data || !out.data) return;

        PixelBuffer src;
        src.Unpack(in);
        if (!src.Valid()) return;

        const Levels lv = Measure(src);

        // Scale so the measured median lands on middle grey.
        //
        // This is the whole trick, and it is why the operator has to measure:
        // a merge's absolute scale is arbitrary, so the only meaningful anchor
        // is where the picture's own midtones sit. After this the image is in
        // the range every other algorithm and the display curve already assume.
        float scale = 1.0f;
        if (lv.valid) {
            const float target = kGreyIn * m_exposure;
            scale = target / lv.median;
        }
        m_scale = scale;

        // How far above grey the highlights reach AFTER scaling, in stops.
        // This is what the shoulder has to absorb, and it is reported because
        // it is the number that says whether the compression is doing anything.
        m_stopsAbove = (lv.valid && lv.high > 0.0f)
            ? std::log2(std::max(lv.high * scale, 1e-6f) / kGreyIn) : 0.0f;

        const float shoulder = std::max(0.1f, float(m_shoulder));
        const int   n  = src.Width() * src.Height();
        const int   ch = src.Channels();
        const std::vector<float>& sp = src.Data();

        PixelBuffer dst;
        dst.Unpack(out);
        std::vector<float>& dp = dst.Data();
        const int dch = dst.Channels();

        for (int i = 0; i < n; ++i) {
            for (int c = 0; c < dch; ++c) {
                if (c == 3 && dch == 4) { dp[size_t(i) * dch + 3] = 1.0f; continue; }
                const int sc = (c < ch) ? c : (ch - 1);
                const float v = sp[size_t(i) * ch + sc] * scale;
                dp[size_t(i) * dch + c] = Compress(v, shoulder);
            }
        }
        dst.PackInto(out);
    }

    std::string RunReport() const override {
        if (m_scale <= 0.0f) return {};
        char buf[96];
        std::snprintf(buf, sizeof buf, "scaled %.4gx, highlights %.1f stops above grey",
                      double(m_scale), double(m_stopsAbove));
        return buf;
    }

    bool HasGPU() const override { return false; }

private:
    // Roll off above middle grey, leaving everything below it alone.
    //
    // Below grey the mapping is EXACTLY linear -- shadows and midtones are not
    // touched at all, so the operator cannot muddy the part of the picture that
    // was already fine. Above grey, a Reinhard-style compression maps an
    // unbounded input onto a finite ceiling:
    //
    //     y = grey + (x - grey) / (1 + (x - grey) / headroom)
    //
    // which is asymptotic to grey + headroom. `shoulder` sets the headroom in
    // units of middle grey, so at the default 4 the ceiling is 5 * 0.18 = 0.9
    // linear -- comfortably inside what the display curve renders without
    // clipping.
    //
    // Negative input passes through unchanged rather than being clamped: it is
    // the demosaic's undershoot, the out-of-gamut overlay exists to show it,
    // and silently swallowing it here would hide a real signal.
    static float Compress(float x, float shoulder) {
        if (x <= kGreyIn) return x;
        const float over     = x - kGreyIn;
        const float headroom = kGreyIn * shoulder;
        return kGreyIn + over / (1.0f + over / headroom);
    }

    // How bright to place the midtones, as a multiple of middle grey. This is
    // the exposure control for the operator: the measurement decides where the
    // median IS, and this decides where it should go.
    Param<float> m_exposure{this, "exposure", 1.0f, 0.1f, 4.0f, {.step = 0.05}};

    // Highlight headroom above grey, in multiples of middle grey. Larger keeps
    // more highlight separation and risks the display curve clipping; smaller
    // compresses harder and flattens the top.
    Param<float> m_shoulder{this, "shoulder", 4.0f, 0.5f, 16.0f, {.step = 0.1}};

    mutable float m_scale      = 0.0f;
    mutable float m_stopsAbove = 0.0f;
};

} // namespace

REGISTER_ALGORITHM(Tonemap);

} // namespace tglab
