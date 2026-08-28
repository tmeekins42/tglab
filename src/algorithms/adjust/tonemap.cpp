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
    float low    = 0.0f;   // the darkest real detail, not the darkest pixel
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

    // Percentiles rather than min/max at BOTH ends. The darkest pixel in a
    // photograph is read noise and the brightest is a specular glint; anchoring
    // to either would let a handful of pixels decide the whole rendering.
    out.low    = pct(0.01);
    out.median = pct(0.50);
    out.high   = std::max(pct(0.995), out.median * 1.0001f);
    out.valid  = out.median > 0.0f;
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

        // Place the curve from BOTH ends of the scene, not from its midpoint.
        //
        // Anchoring the median to middle grey and hoping the ends land well is
        // what the first version did, and it cannot work: where the ends sit
        // relative to the median is a property of the SCENE, and it varies
        // enormously. Measured on two of Tim's brackets:
        //
        //   a sunlit building   3.8 stops below the median, 1.5 above
        //   a valley and sky    2.1 stops below,            3.7 above
        //
        // Fixing the midpoint threw the long end off the display in each case
        // -- the building's shadows to 0.013 linear (black, nothing for the
        // shadow slider to recover), the valley's sky to 2.3 (flat, needing
        // heavy manual contrast). Same operator, opposite failures, because the
        // anchor ignored the thing that differed.
        //
        // So: put the shadows at a visible black point and the highlights near
        // white, and let the median land where the scene puts it. That is what
        // a photographer does by eye with the black and white points, and it is
        // what SuggestExposure already does for raws.
        float scale = 1.0f;
        if (lv.valid) {
            // Where the ends should land, in linear light before the display
            // curve. kBlackAim renders around 0.10 display -- dark, but with
            // detail visible rather than crushed. kWhiteAim is the top of what
            // the display shoulder resolves without flattening.
            constexpr float kBlackAim = 0.004f;
            constexpr float kWhiteAim = 8.0f;

            // The scale that would satisfy each end alone.
            const float byLow  = kBlackAim / std::max(lv.low,  1e-6f);
            const float byHigh = kWhiteAim / std::max(lv.high, 1e-6f);

            // Both cannot generally be satisfied: a scene wider than the range
            // between the aims has to give somewhere. The GEOMETRIC MEAN splits
            // the difference evenly in stops, which is the right currency here
            // -- an arithmetic mean would be dominated by whichever end has the
            // larger number and would drift with the merge's arbitrary scale.
            scale = std::sqrt(byLow * byHigh);

            // Keep the midtones from drifting somewhere absurd on a scene whose
            // ends are lopsided. Nothing in the fit constrains the median, and
            // a frame that is almost all sky or almost all shadow can put it
            // far from grey; bounding it to a couple of stops either side keeps
            // the result recognisable without overriding the fit in the normal
            // case.
            const float midAfter = lv.median * scale;
            const float lo = kGreyIn * 0.25f;   // 2 stops below grey
            const float hi = kGreyIn * 4.0f;    // 2 stops above
            if (midAfter < lo) scale *= lo / midAfter;
            else if (midAfter > hi) scale *= hi / midAfter;

            scale *= m_exposure;
        }
        m_scale = scale;

        // How far above grey the highlights reach AFTER scaling, in stops.
        // This is what the shoulder has to absorb, and it is reported because
        // it is the number that says whether the compression is doing anything.
        m_stopsAbove = (lv.valid && lv.high > 0.0f)
            ? std::log2(std::max(lv.high * scale, 1e-6f) / kGreyIn) : 0.0f;
        m_stopsBelow = (lv.valid && lv.low > 0.0f)
            ? std::log2(kGreyIn / std::max(lv.low * scale, 1e-6f)) : 0.0f;

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
        std::snprintf(buf, sizeof buf,
                      "scaled %.4gx, %.1f stops below grey / %.1f above",
                      double(m_scale), double(m_stopsBelow), double(m_stopsAbove));
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
    // which is asymptotic to grey + headroom.
    //
    // THE CEILING MUST BE EXPRESSED IN THE DISPLAY CURVE'S UNITS, not picked to
    // look safe in linear. The first version capped at 0.9 linear, which sounds
    // conservative until you follow it through tone_curve.h: 0.9 linear renders
    // 0.733 display, so every tone from the midtones to the brightest highlight
    // landed between 0.46 and 0.73. The result was exactly what Tim reported --
    // flat, no whites, and highlight controls with nothing left to act on,
    // because the range they operate in had already been thrown away.
    //
    // The display curve reaches 0.975 at 19.0 linear, which is 6.7 stops above
    // grey. Compressing into 2.3 of them and calling it tone mapping is
    // compressing twice. The default now targets that curve's real headroom, so
    // the operator's job is only to bring an arbitrary merge scale INTO the
    // range the display curve was built for -- not to re-do its work.
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

    // Highlight headroom above grey, in multiples of middle grey.
    //
    // 100 rather than 4, which is the fix for the flat result: the ceiling is
    // grey * (1 + shoulder), so 100 puts it at 18.2 linear -- essentially the
    // 19.0 where the display curve reaches 0.975. The operator now brings the
    // merge into that range and lets the display shoulder do the rolling off it
    // was designed for, instead of pre-compressing into a fifth of it.
    //
    // Still a real control: lower it to compress the highlights harder, which
    // is what a scene with a genuinely blown sky wants.
    Param<float> m_shoulder{this, "shoulder", 100.0f, 1.0f, 400.0f, {.step = 1.0, .softMin = 4.0f, .softMax = 200.0f}};

    mutable float m_scale      = 0.0f;
    mutable float m_stopsAbove = 0.0f;
    mutable float m_stopsBelow = 0.0f;
};

} // namespace

REGISTER_ALGORITHM(Tonemap);

} // namespace tglab
