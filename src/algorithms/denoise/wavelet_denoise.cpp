// wavelet_denoise — multi-scale denoising by shrinking wavelet coefficients.
//
// The standard method for photographic sensor noise, and the reason it beats
// the spatial filters already in the registry: noise and detail live at
// different SCALES. A bilateral or non-local-means filter has one neighbourhood
// size and must trade grain against texture within it. A wavelet decomposition
// separates the image into bands first, so fine grain can be removed at the
// scale it occupies while edges -- which have energy at every scale -- survive.
//
// Measured on Tim's ISO 1000 CR2 to check this is the right tool: the noise is
// broad-band with tails stopping around 4.5 sigma (so no impulse component,
// which would want a median instead), and sigma tracks sqrt(signal), meaning it
// is shot-noise dominated rather than a constant read-noise floor. Both point
// at coefficient shrinkage rather than a spatial kernel.
//
// The transform is the a-trous ("with holes") wavelet: no decimation, so every
// level stays full resolution. That costs memory against a decimated transform
// but removes the shift-dependence that makes decimated wavelets ring around
// edges -- the artefact that makes naive wavelet denoising look worse than the
// noise it removed.
#include <algorithm>
#include <cmath>
#include <vector>

#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"

namespace tglab {
namespace {

// B3-spline kernel, the usual a-trous choice: smooth enough that the residual
// bands hold little aliasing, short enough to stay cheap.
constexpr float kB3[5] = {1.0f / 16, 4.0f / 16, 6.0f / 16, 4.0f / 16, 1.0f / 16};

// One a-trous level: convolve with the B3 kernel dilated by `step`, separably.
void ATrousBlur(const std::vector<float>& src, std::vector<float>& dst,
                int w, int h, int ch, int step, std::vector<float>& tmp) {
    tmp.assign(src.size(), 0.0f);

    // The five tap offsets are resolved per pixel and then reused across
    // channels, rather than recomputing a full index per (tap, channel). That
    // ordering was worth 2x: at 17 MP and four levels the naive version spent
    // its time on address arithmetic, not filtering.

    // Horizontal.
    for (int y = 0; y < h; ++y) {
        const size_t row = size_t(y) * size_t(w);
        for (int x = 0; x < w; ++x) {
            const float* t[5];
            for (int k = -2; k <= 2; ++k) {
                const int sx = std::clamp(x + k * step, 0, w - 1);
                t[k + 2] = &src[(row + size_t(sx)) * size_t(ch)];
            }
            float* out = &tmp[(row + size_t(x)) * size_t(ch)];
            for (int c = 0; c < ch; ++c)
                out[c] = kB3[0] * t[0][c] + kB3[1] * t[1][c] + kB3[2] * t[2][c] +
                         kB3[3] * t[3][c] + kB3[4] * t[4][c];
        }
    }
    // Vertical.
    dst.assign(src.size(), 0.0f);
    for (int y = 0; y < h; ++y) {
        // Row bases for the five taps: constant across the whole row, so they
        // are hoisted out of the x loop entirely.
        size_t rb[5];
        for (int k = -2; k <= 2; ++k)
            rb[k + 2] = size_t(std::clamp(y + k * step, 0, h - 1)) * size_t(w);

        for (int x = 0; x < w; ++x) {
            const float* t0 = &tmp[(rb[0] + size_t(x)) * size_t(ch)];
            const float* t1 = &tmp[(rb[1] + size_t(x)) * size_t(ch)];
            const float* t2 = &tmp[(rb[2] + size_t(x)) * size_t(ch)];
            const float* t3 = &tmp[(rb[3] + size_t(x)) * size_t(ch)];
            const float* t4 = &tmp[(rb[4] + size_t(x)) * size_t(ch)];
            float* out = &dst[(size_t(y) * size_t(w) + size_t(x)) * size_t(ch)];
            for (int c = 0; c < ch; ++c)
                out[c] = kB3[0] * t0[c] + kB3[1] * t1[c] + kB3[2] * t2[c] +
                         kB3[3] * t3[c] + kB3[4] * t4[c];
        }
    }
}

class WaveletDenoise : public AlgorithmBase {
public:
    const char* Name()     const override { return "wavelet_denoise"; }
    const char* Category() const override { return "denoise"; }

    PortList Inputs()  const override { return {{"src", DataType::Image, FormatSpec::Any}}; }
    PortList Outputs() const override { return {{"out", DataType::Image, FormatSpec::SameAsInput}}; }

    void RunCPU(RunCtx& ctx) override {
        const ImageView src = ctx.In(0);
        ImageView       dst = ctx.Out(0);
        if (!src.Valid() || !dst.Valid()) return;

        PixelBuffer buf;
        buf.Unpack(src);
        if (!buf.Valid()) return;

        const int w  = buf.Width();
        const int h  = buf.Height();
        const int ch = buf.Channels();
        const float scale = buf.ValueScale();

        const int levels = std::clamp(int(m_levels), 1, 6);
        // Thresholds are expressed as a fraction of the intensity range, so a
        // setting means the same thing on an 8-bit scan and a float raw. The
        // same units trap that made brightness apply its offset 255x over.
        const float lumaT   = float(m_lumaStrength)   * scale;
        const float chromaT = float(m_chromaStrength) * scale;

        std::vector<float> cur = buf.Data();
        std::vector<float> blurred, tmp;
        std::vector<float> residual(cur.size());

        // Working in a luma/chroma basis when the image has colour, because
        // sensor noise is markedly worse in chroma and chroma carries little
        // fine detail -- it can be shrunk far harder than luma without a
        // visible cost. That asymmetry is most of what makes ISO denoising
        // work, and it is invisible in an RGB basis where all three channels
        // get the same treatment.
        const bool colour = (ch >= 3);
        if (colour) ToYcc(cur, w, h, ch);

        for (int lvl = 0; lvl < levels; ++lvl) {
            if (ctx.Cancelled()) return;

            const int step = 1 << lvl;
            ATrousBlur(cur, blurred, w, h, ch, step, tmp);

            // The detail band: what this scale holds that the next coarser one
            // does not.
            for (size_t i = 0; i < cur.size(); ++i) residual[i] = cur[i] - blurred[i];

            // Coarser levels hold progressively more real structure and less
            // grain, so the threshold falls as the scale grows. Halving per
            // level matches how a noise field's energy distributes across an
            // a-trous pyramid.
            const float falloff = 1.0f / float(1 << lvl);

            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    const size_t i = (size_t(y) * size_t(w) + size_t(x)) * size_t(ch);
                    for (int c = 0; c < ch; ++c) {
                        if (c == 3) continue;                 // leave alpha alone
                        const bool isChroma = colour && c > 0;
                        const float t = (isChroma ? chromaT : lumaT) * falloff;
                        if (t <= 0.0f) continue;
                        residual[i + size_t(c)] = Shrink(residual[i + size_t(c)], t);
                    }
                }
            }

            // Reassemble: the shrunk detail plus the coarser approximation.
            for (size_t i = 0; i < cur.size(); ++i) cur[i] = blurred[i] + residual[i];
        }

        if (colour) FromYcc(cur, w, h, ch);

        buf.Data() = std::move(cur);
        buf.PackInto(dst);
    }

private:
    // Soft thresholding rather than hard.
    //
    // Hard thresholding (zero below t, keep above) leaves a discontinuity at
    // the threshold, and a coefficient sitting either side of it from one pixel
    // to the next flips between kept and discarded -- which is what produces
    // the mottled "wavelet blotch" that makes the method look worse than doing
    // nothing. Soft shrinkage is continuous, so a coefficient near the
    // threshold moves smoothly.
    static float Shrink(float v, float t) {
        const float a = std::fabs(v);
        if (a <= t) return 0.0f;
        return (v > 0.0f) ? (a - t) : -(a - t);
    }

    // Reversible luma/chroma rotation. Not a colorimetric space -- a proper
    // YCbCr would need the primaries and transfer function, which vary by
    // source -- but this is only ever a basis to threshold in, and it round
    // trips exactly, which is the property that matters.
    static void ToYcc(std::vector<float>& d, int w, int h, int ch) {
        for (int i = 0; i < w * h; ++i) {
            float* p = &d[size_t(i) * size_t(ch)];
            const float r = p[0], g = p[1], b = p[2];
            p[0] = 0.25f * r + 0.5f * g + 0.25f * b;   // luma
            p[1] = r - g;                              // chroma 1
            p[2] = b - g;                              // chroma 2
        }
    }
    static void FromYcc(std::vector<float>& d, int w, int h, int ch) {
        for (int i = 0; i < w * h; ++i) {
            float* p = &d[size_t(i) * size_t(ch)];
            const float y = p[0], c1 = p[1], c2 = p[2];
            const float g = y - 0.25f * c1 - 0.25f * c2;
            p[0] = c1 + g;
            p[1] = g;
            p[2] = c2 + g;
        }
    }

    Param<int> m_levels{this, "levels", 4, 1, 6,
        {.help = "How many scales to process. Each level is twice as coarse as "
                 "the last, so more levels remove larger-scale blotchiness at "
                 "the cost of time. 4 covers grain through to mottling.",
         .step = 1}};

    Param<float> m_lumaStrength{this, "luma", 0.02f, 0.0f, 0.25f,
        {.help = "Shrinkage threshold for the luminance detail bands, as a "
                 "fraction of the intensity range. Raise it to remove more "
                 "grain; too high and fine texture goes with it. 0 leaves "
                 "luminance untouched.",
         .step = 0.002, .softMin = 0.0, .softMax = 0.08}};

    Param<float> m_chromaStrength{this, "chroma", 0.08f, 0.0f, 0.5f,
        {.help = "Shrinkage threshold for the colour detail bands. Defaults "
                 "well above luma because sensor noise is far worse in chroma "
                 "and colour carries little fine detail, so it tolerates much "
                 "harder smoothing before anything shows.",
         .step = 0.005, .softMin = 0.0, .softMax = 0.25}};
};

REGISTER_ALGORITHM(WaveletDenoise);

} // namespace
} // namespace tglab
