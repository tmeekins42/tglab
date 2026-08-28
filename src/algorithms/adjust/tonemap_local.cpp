// tonemap_local — compress the illumination, keep the detail.
//
// The global operator does what a global operator can, and the limit is
// structural rather than a bug. It applies ONE curve to every pixel, so
// squeezing 5.8 stops of scene into a display necessarily squeezes every
// region by the same factor. On Tim's hdrtest2 the sky keeps its ordering and
// 1.36 of its 1.89 stops through a full highlight recovery, and still reads
// flat: the sky occupies a small slice of the total range and sits next to
// much darker land. Nothing a single curve does can give it back more of the
// display without taking the same amount from everything else.
//
// The observation this operator is built on -- Oppenheim (1968), and Durand &
// Dorsey (2002) in the form used here -- is that an image is roughly
//
//     image = illumination * reflectance
//
// Illumination is what varies hugely across a scene: sunlit sky against shaded
// land is several stops of ILLUMINATION, not of surface. Reflectance is
// bounded, roughly 0.02 to 0.9, because that is what real materials do.
//
// So: separate them, compress only the illumination, and put the reflectance
// back untouched. The sky's own internal contrast survives at full strength
// while the gap between sky and land shrinks. That is precisely the failure
// the global operator cannot address.
//
// WHY THE SPLIT IS IN LOG SPACE, which is the one thing to get right
// ------------------------------------------------------------------
// The relationship above is a PRODUCT. In log space it becomes a sum:
//
//     log(image) = log(illumination) + log(reflectance)
//
// so a subtraction separates them and scaling the base is a POWER on the
// original value. Local contrast -- a ratio between neighbouring pixels --
// then survives compression exactly, everywhere in the frame.
//
// Doing it in linear instead would make `detail = image - base` a DIFFERENCE,
// which is proportional to brightness: the same 20% local contrast is a large
// number in the sky and a small one in the shadows. Compressing the base then
// scales those differences unevenly and the shadows come back flat. The log is
// not a detail of the implementation; it is the reason the method works.
//
// HALOS ARE THE FAILURE MODE
// --------------------------
// If the base layer blurs ACROSS a strong edge, the detail layer picks up a
// large-scale gradient that does not belong to it, and compressing the base
// leaves a bright rim on the dark side of the edge and a dark rim on the light
// side. That is the classic tone-mapped look, and it is why the global
// operator stays the default rather than being replaced.
//
// The guided filter is the defence: it is edge-preserving and, unlike the
// bilateral, has no gradient reversal (He, Sun & Tang 2010). It also costs
// O(1) per pixel regardless of radius, which matters because a base layer
// wants a LARGE radius -- illumination varies slowly, so a small radius pulls
// real detail into the base and leaves nothing to protect.
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "../../algo_util/pixel_buffer.h"
#include "../../algo_util/tone_curve.h"
#include "../../core/algorithm.h"

namespace tglab {
namespace {

// Percentiles of the log-luminance, which is where this operator works.
//
// Sampled on a stride rather than read in full: at 22 MP a complete sort costs
// far more than the answer is worth, and a few hundred thousand samples pin a
// percentile down. Same reasoning, and the same stride, as the global operator.
struct LogLevels {
    float low    = 0.0f;   // log2 of the darkest real detail
    float median = 0.0f;
    float high   = 0.0f;   // log2 of the brightest real detail
    bool  valid  = false;
};

LogLevels MeasureLog(const std::vector<float>& logLum, int w, int h) {
    LogLevels out;
    if (w <= 0 || h <= 0) return out;

    std::vector<float> s;
    s.reserve(400000);
    const int stride = std::max(1, std::min(w, h) / 700);
    for (int y = 0; y < h; y += stride)
        for (int x = 0; x < w; x += stride)
            s.push_back(logLum[size_t(y) * size_t(w) + size_t(x)]);

    if (s.size() < 1000) return out;
    std::sort(s.begin(), s.end());
    auto pct = [&](double f) {
        return s[std::min(s.size() - 1, size_t(f * double(s.size() - 1) + 0.5))];
    };

    // Percentiles at both ends rather than min/max: the darkest pixel is read
    // noise and the brightest a specular glint, and anchoring to either would
    // let a handful of pixels decide the whole rendering.
    out.low    = pct(0.01);
    out.median = pct(0.50);
    out.high   = pct(0.995);
    out.valid  = out.high > out.low;
    return out;
}

class TonemapLocal : public AlgorithmBase {
public:
    const char* Name()     const override { return "tonemap_local"; }

    // Same category as the global operator, so choose("op", "tonemap") offers
    // both and a script can swap them with one word. Comparing the two is the
    // point of having them separate.
    const char* Category() const override { return "tonemap"; }

    PortList Inputs()  const override { return {{"src"}}; }
    PortList Outputs() const override {
        // Still linear float, exactly like tonemap: lower-scaled and
        // compressed, but linear, so the display curve and every develop
        // control downstream behave as they do for a raw.
        return {{"out", DataType::Image, FormatSpec::RGBA32F}};
    }

    void RunCPU(RunCtx& ctx) override {
        const ImageView in = ctx.In(0);
        ImageView out = ctx.Out(0);
        if (!in.data || !out.data) return;

        m_in.Unpack(in);
        if (!m_in.Valid()) return;

        const int w = m_in.Width(), h = m_in.Height(), ch = m_in.Channels();
        const size_t n = size_t(w) * size_t(h);
        if (n == 0) return;

        // --- log luminance ---------------------------------------------------
        //
        // The split is computed on LUMINANCE alone, not per channel, and the
        // colour is carried through as a ratio at the end. Filtering each
        // channel separately would let the three bases diverge near an edge and
        // shift hue there -- the same reason the develop path scales all three
        // channels by one factor.
        m_log.assign(n, 0.0f);
        const std::vector<float>& sp = m_in.Data();
        for (size_t i = 0; i < n; ++i) {
            const float* p = &sp[i * size_t(ch)];
            const float lum = (ch >= 3) ? 0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2]
                                        : p[0];
            // Floored rather than clamped away. A merged bracket contains real
            // negatives -- the demosaic undershoots near black and the merge
            // amplifies it -- and log is undefined there. The floor is far
            // below any real scene value, so it costs nothing and keeps the
            // arrays the same length as the image.
            m_log[i] = std::log2(std::max(lum, kFloor));
        }

        const LogLevels lv = MeasureLog(m_log, w, h);
        if (!lv.valid) {   // degenerate input: pass it through rather than guess
            PixelBuffer dst;
            dst.Unpack(out);
            CopyThrough(dst, 1.0f);
            dst.PackInto(out);
            return;
        }

        // --- base layer ------------------------------------------------------
        //
        // Radius is a FRACTION of the image, not a pixel count. Illumination
        // varies over a scene, so what counts as "large scale" scales with the
        // frame: a 40-pixel radius that works on a 2 MP preview pulls real
        // detail into the base at 45 MP, and the operator then flattens the
        // texture it exists to protect.
        const int radius = std::max(2, int(float(m_radius) * 0.01f *
                                           float(std::min(w, h))));
        m_base.assign(n, 0.0f);
        GuidedSelf(m_log, m_base, w, h, radius, float(m_detail));
        if (ctx.Cancelled()) return;

        // --- compress the base, keep the detail ------------------------------
        //
        // The base is scaled about the scene's own median so the midtones stay
        // put and the compression pulls both ends inward. Anchoring anywhere
        // else would shift the whole image brighter or darker as a side effect
        // of compressing it.
        //
        // `compress` is a target RANGE in stops rather than a raw factor, which
        // is what makes the control mean something: it says how many stops of
        // scene the base should occupy afterwards. A scene already narrower
        // than the target is left alone rather than being EXPANDED -- stretching
        // a low-contrast frame is a different operation and not one to do by
        // accident.
        const float spanBefore = lv.high - lv.low;
        const float target     = float(m_range);
        const float k = (spanBefore > target) ? (target / spanBefore) : 1.0f;
        m_compression = k;
        m_spanBefore  = spanBefore;

        // Where the compressed result should sit. The median lands on middle
        // grey, which is the anchor the display curve is built around, and the
        // exposure control offsets from there in stops.
        const float greyLog = std::log2(kGreyIn);
        const float offset  = greyLog - k * lv.median + float(m_exposure);

        // --- recombine -------------------------------------------------------
        PixelBuffer dst;
        dst.Unpack(out);
        std::vector<float>& dp = dst.Data();
        const int dch = dst.Channels();

        const float sat = float(m_saturation);

        for (size_t i = 0; i < n; ++i) {
            const float detail = m_log[i] - m_base[i];

            // Base compressed, detail preserved at full strength. This is the
            // whole operator in one line: because these are logs, keeping
            // `detail` unscaled keeps every local RATIO exactly as it was,
            // while k rescales only the slow illumination underneath it.
            const float outLog = k * m_base[i] + detail + offset;
            const float newLum = std::exp2(outLog);

            // Colour carried as a ratio to the old luminance, so hue and
            // saturation are unchanged by the tone mapping itself. Scaling all
            // three channels by one factor is what makes that true -- see the
            // note above about per-channel filtering.
            const float oldLum = std::exp2(m_log[i]);
            const float gain   = newLum / std::max(oldLum, kFloor);

            const float* p = &sp[i * size_t(ch)];
            for (int c = 0; c < dch; ++c) {
                if (c == 3 && dch == 4) { dp[i * size_t(dch) + 3] = 1.0f; continue; }
                const int sc = (c < ch) ? c : (ch - 1);

                float v = p[sc] * gain;

                // Compressing luminance desaturates -- the ratio between a
                // channel and the luminance is preserved, but the eye reads the
                // result as less colourful once the range is squeezed. This
                // pushes the channel away from the new luminance to compensate,
                // and at 1.0 does nothing at all.
                if (sat != 1.0f) v = newLum + (v - newLum) * sat;

                dp[i * size_t(dch) + c] = v;
            }
        }
        dst.PackInto(out);
    }

    std::string RunReport() const override {
        if (m_spanBefore <= 0.0f) return {};
        char buf[128];
        std::snprintf(buf, sizeof buf,
                      "%.1f stops of scene compressed to %.1f (base x%.2f)",
                      double(m_spanBefore), double(m_spanBefore * m_compression),
                      double(m_compression));
        return buf;
    }

    bool HasGPU() const override { return false; }

private:
    // Below any real scene value, and only there to keep log() defined on the
    // negatives a merge legitimately produces.
    static constexpr float kFloor = 1e-6f;

    void CopyThrough(PixelBuffer& dst, float gain) {
        const std::vector<float>& sp = m_in.Data();
        std::vector<float>& dp = dst.Data();
        const int ch = m_in.Channels(), dch = dst.Channels();
        const size_t n = size_t(m_in.Width()) * size_t(m_in.Height());
        for (size_t i = 0; i < n; ++i)
            for (int c = 0; c < dch; ++c) {
                if (c == 3 && dch == 4) { dp[i * size_t(dch) + 3] = 1.0f; continue; }
                dp[i * size_t(dch) + c] = sp[i * size_t(ch) + size_t((c < ch) ? c : ch - 1)] * gain;
            }
    }

    // Self-guided guided filter over one plane (He, Sun & Tang 2010).
    //
    // Same algorithm as the registered guided_filter, deliberately duplicated
    // rather than shared: that one is an image-to-image stage over a
    // PixelBuffer, and reaching into it for a single float plane would either
    // couple the two or force a general "filter one plane" interface that only
    // has one caller. Twenty lines is the cheaper answer, and the comment there
    // carries the derivation.
    //
    // Self-guided means cov(I, p) == var(I), so the algebra collapses to
    // a = var / (var + eps), b = (1 - a) * mean.
    void GuidedSelf(const std::vector<float>& in, std::vector<float>& out,
                    int w, int h, int radius, float epsStops) {
        const size_t n = in.size();
        m_mean.assign(n, 0.0f);
        m_meanSq.assign(n, 0.0f);
        m_a.assign(n, 0.0f);
        m_b.assign(n, 0.0f);
        m_scratch.assign(n, 0.0f);

        // eps compares against a VARIANCE, and the plane is in log2 units, so
        // the control is in stops and squared here. That makes the number mean
        // something physical: "variation smaller than this many stops is
        // texture, not an illumination edge."
        const float eps = epsStops * epsStops;

        BoxMean(in, m_mean, w, h, radius);
        for (size_t i = 0; i < n; ++i) m_scratch[i] = in[i] * in[i];
        BoxMean(m_scratch, m_meanSq, w, h, radius);

        for (size_t i = 0; i < n; ++i) {
            const float var = std::max(0.0f, m_meanSq[i] - m_mean[i] * m_mean[i]);
            const float a   = var / (var + eps);
            m_a[i] = a;
            m_b[i] = (1.0f - a) * m_mean[i];
        }

        // Averaging a and b is what makes the result smooth where windows
        // overlap; without it the output shows window-sized blocking.
        BoxMean(m_a, m_scratch, w, h, radius);
        m_a.swap(m_scratch);
        BoxMean(m_b, m_scratch, w, h, radius);
        m_b.swap(m_scratch);

        for (size_t i = 0; i < n; ++i) out[i] = m_a[i] * in[i] + m_b[i];
    }

    // Separable running-sum box mean: O(1) per pixel, which is what lets the
    // radius be large enough to be a genuine illumination estimate.
    void BoxMean(const std::vector<float>& in, std::vector<float>& out,
                 int w, int h, int radius) {
        m_rowTmp.assign(in.size(), 0.0f);
        const float norm = 1.0f / float(radius * 2 + 1);

        for (int y = 0; y < h; ++y) {
            const size_t row = size_t(y) * size_t(w);
            float sum = 0.0f;
            for (int i = -radius; i <= radius; ++i)
                sum += in[row + size_t(std::clamp(i, 0, w - 1))];
            for (int x = 0; x < w; ++x) {
                m_rowTmp[row + size_t(x)] = sum * norm;
                sum += in[row + size_t(std::clamp(x + radius + 1, 0, w - 1))] -
                       in[row + size_t(std::clamp(x - radius, 0, w - 1))];
            }
        }

        for (int x = 0; x < w; ++x) {
            float sum = 0.0f;
            for (int i = -radius; i <= radius; ++i)
                sum += m_rowTmp[size_t(std::clamp(i, 0, h - 1)) * size_t(w) + size_t(x)];
            for (int y = 0; y < h; ++y) {
                out[size_t(y) * size_t(w) + size_t(x)] = sum * norm;
                sum += m_rowTmp[size_t(std::clamp(y + radius + 1, 0, h - 1)) * size_t(w) + size_t(x)] -
                       m_rowTmp[size_t(std::clamp(y - radius, 0, h - 1)) * size_t(w) + size_t(x)];
            }
        }
    }

    Param<float> m_range{
        this, "range", 4.0f, 1.0f, 12.0f,
        {.help = "How many stops of scene the compressed illumination should "
                 "occupy. Lower compresses harder. A scene already narrower "
                 "than this is left alone rather than stretched.",
         .step = 0.1, .softMin = 2.0, .softMax = 8.0}};

    Param<float> m_radius{
        this, "radius", 8.0f, 1.0f, 40.0f,
        {.help = "Size of the illumination estimate, as a percentage of the "
                 "shorter side. Too small pulls real detail into the base and "
                 "flattens texture; too large stops following the light.",
         .step = 0.5, .softMin = 3.0, .softMax = 20.0}};

    Param<float> m_detail{
        this, "detail", 1.2f, 0.05f, 4.0f,
        {.help = "In stops: local variation smaller than this counts as "
                 "texture to protect, larger as an illumination edge to "
                 "compress. Raising it moves more into the base, which "
                 "preserves more contrast but risks halos at strong edges.",
         .step = 0.05, .softMin = 0.2, .softMax = 1.5}};

    Param<float> m_saturation{
        this, "saturation", 1.0f, 0.0f, 2.0f,
        {.help = "Compressing luminance reads as desaturated even though the "
                 "channel ratios are preserved. Above 1 compensates. 1.0 "
                 "leaves colour exactly as the ratios give it.",
         .step = 0.01, .softMin = 0.8, .softMax = 1.5}};

    Param<float> m_exposure{
        this, "exposure", 0.0f, -5.0f, 5.0f,
        {.help = "Stops of offset applied after compression. The median lands "
                 "on middle grey by default; this moves it.",
         .step = 0.05, .softMin = -2.0, .softMax = 2.0}};

    PixelBuffer        m_in;
    std::vector<float> m_log, m_base;
    std::vector<float> m_mean, m_meanSq, m_a, m_b, m_scratch, m_rowTmp;

    float m_compression = 1.0f;
    float m_spanBefore  = 0.0f;
};

} // namespace

REGISTER_ALGORITHM(TonemapLocal);

} // namespace tglab
