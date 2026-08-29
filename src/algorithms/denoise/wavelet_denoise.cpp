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
#include <cstdlib>
#include <vector>

#include <windows.h>

#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"

namespace tglab {
namespace {

// B3-spline kernel, the usual a-trous choice: smooth enough that the residual
// bands hold little aliasing, short enough to stay cheap.
constexpr float kB3[5] = {1.0f / 16, 4.0f / 16, 6.0f / 16, 4.0f / 16, 1.0f / 16};

// Clamp bounds for the level-dependent threshold, overridable so they can be
// swept without a rebuild between runs. Diagnostic only -- the defaults are
// what ship, and TGLAB_LEVELDEP_CLAMP exists so the choice can be re-measured
// rather than taken on trust.
inline float ClampEnv(const char* name, float def) {
    char buf[32] = {};
    if (GetEnvironmentVariableA(name, buf, sizeof buf) == 0) return def;
    const float v = float(std::atof(buf));
    return (v > 0.0f) ? v : def;
}
inline float ClampLo() { static const float v = ClampEnv("TGLAB_CLAMP_LO", 0.20f); return v; }
inline float ClampHi() { static const float v = ClampEnv("TGLAB_CLAMP_HI", 4.0f);  return v; }

// How much to scale the threshold for a neighbourhood at this brightness.
//
// One fixed threshold assumes the noise is the same everywhere, and MEASURED on
// developed images that is roughly true for most frames and badly wrong for a
// pushed high-ISO one. tools/noise_profile.exe reports sigma per brightness
// band; on three frames it found:
//
//   _DSC0037  ISO 6400, pushed +1.8   0.0031 .. 0.0292   9.6x
//   _MG_9673  ISO 1000                0.0024 .. 0.0075   3.1x
//   _DSC0162  ISO 125                 0.0017 .. 0.0045   2.7x
//
// So the shadows want a much lower threshold than the highlights on a noisy
// frame, and about the same one on a clean frame. `amount` at 0 is the old
// behaviour exactly, which is what makes this comparable rather than a silent
// change to everything.
//
// The relationship is sigma ~ L^0.5 -- shot noise, which the tone curve
// flattens but does not remove. Measured on the same three frames: raising
// every band's sigma to L^-0.5 takes _DSC0037's 9.6x spread to 2.1x, while
// L^-0.75 leaves 4.5x and L^-0.25 leaves 6.6x. So the exponent is 0.5, and
// `amount` interpolates toward it rather than picking a different one.
//
// (An earlier note in todo.txt reasoned the opposite -- that develop had
// already stabilised the noise and L^0.75 was the better exponent, quoting
// 1.2x against sqrt's 2.0x. Re-measured with an estimator that self-tests
// against injected noise of known sigma, sqrt wins on every frame. The tool is
// checked in so the disagreement can be settled by running it.)
// `mid` is middle grey IN THE BUFFER'S UNITS, which is why it is passed in
// rather than being a constant here. On an RGBA8 image PixelBuffer holds
// 0..255 and on a float image 0..1, so a hardcoded 0.18 means middle grey on
// one and near-black on the other -- the CPU/GPU parity test caught exactly
// that, reporting a max difference of 78 against the usual 8. The thresholds
// above already scale for this reason; this has to as well.
inline float LevelScale(float level, float amount, float mid) {
    if (amount <= 0.0f) return 1.0f;

    // Normalised to middle grey, so `amount` scales the threshold DOWN in the
    // shadows and UP in the highlights while leaving midtones alone -- the
    // existing luma/chroma defaults keep meaning what they meant.
    const float rel = std::sqrt(std::max(level, 1e-4f * mid) / mid);

    // Bounded, so a near-black neighbourhood cannot drive the threshold to
    // nearly zero (denoising nothing exactly where a noisy shadow needs it)
    // and a specular highlight cannot drive it up until it erases texture.
    //
    // 0.20/4.0 from a sweep rather than from reasoning. Denoised spread on
    // _DSC0037, the pushed ISO 6400 frame:
    //
    //   1.00/1.0  31.1x    (clamped shut: level_dep does nothing)
    //   0.60/1.7  17.4x
    //   0.35/2.5  14.5x
    //   0.20/4.0  13.7x    <- default
    //   0.10/8.0  13.5x
    //
    // So the bound matters until about 0.2/4.0 and is flat after -- loosening
    // to 0.10/8.0 buys 0.2x. The clean frames agree: _MG_9673 improves 3.2x ->
    // 2.6x and then stops, _DSC0162 does not move at all. The first value here
    // was 0.35/2.5, chosen as "roughly where a photograph stops having usable
    // detail" -- reasonable, and slightly too tight.
    const float scaled = std::clamp(rel, ClampLo(), ClampHi());
    return 1.0f + amount * (scaled - 1.0f);
}

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
        const float levelDep = float(m_levelDep);
        const float midGrey  = 0.18f * scale;   // middle grey in the buffer's units

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

        for (int lvl = 0; lvl < levels; ++lvl) {
            if (ctx.Cancelled()) return;

            // Rotate into luma/chroma for THIS level only, and back at the end
            // of it, rather than once around the whole pyramid.
            //
            // The GPU has to work this way -- its ping-pong buffers take the
            // output.s format, and chroma is a signed difference that an RGBA8
            // UNORM target clamps to zero -- so the CPU follows suit to keep
            // the two paths computing the same thing. The cost is a rotation
            // pair per level; the benefit is that neither path can silently
            // destroy chroma on an 8-bit image.
            if (colour) ToYcc(cur, w, h, ch);

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

                    // How bright this neighbourhood is, from the BLURRED value
                    // rather than the pixel's own: a noisy pixel would
                    // otherwise scale its own threshold by its own noise.
                    //
                    // Channel 0 is luma after the rotation, and the plain value
                    // when there is no rotation -- either way it is the level
                    // that matters. Chroma is a signed difference around zero
                    // and says nothing about brightness.
                    const float lvlScale = LevelScale(blurred[i], levelDep, midGrey);

                    for (int c = 0; c < ch; ++c) {
                        if (c == 3) continue;                 // leave alpha alone
                        const bool isChroma = colour && c > 0;
                        const float t = (isChroma ? chromaT : lumaT) * falloff * lvlScale;
                        if (t <= 0.0f) continue;
                        residual[i + size_t(c)] = Shrink(residual[i + size_t(c)], t);
                    }
                }
            }

            // Reassemble: the shrunk detail plus the coarser approximation.
            for (size_t i = 0; i < cur.size(); ++i) cur[i] = blurred[i] + residual[i];

            if (colour) FromYcc(cur, w, h, ch);
        }

        buf.Data() = std::move(cur);
        buf.PackInto(dst);
    }

    // --- GPU implementation -------------------------------------------------
    //
    // One dispatch per level, ping-ponging the accumulator between the output
    // and the scratch. Each pass computes the level's blur with a 2D 5x5
    // dilated kernel, takes the detail band against its own input, shrinks it,
    // and writes blur + shrunk detail.
    //
    // The blur is NOT split into separable horizontal and vertical passes here,
    // although the CPU does exactly that. Separating would need three buffers:
    // the accumulator, the horizontal partial, and the destination -- and the
    // ping-pong has two, so the vertical pass would no longer have the
    // accumulator it must subtract from. 25 taps against 10 is more arithmetic
    // but it all comes from cache, and it removes a full-image round trip per
    // level, which on a GPU is the term that actually costs.
    //
    // No custom scratch format: the output is SameAsInput, which for a
    // developed photograph is already four channels, so the default scratch is
    // the right shape.
    // Both thresholds at zero shrinks nothing: every coefficient survives and
    // the reconstruction returns the input. Worth skipping rather than running
    // four levels of a-trous to arrive back where it started.
    bool IsNoOp() const override {
        return float(m_lumaStrength) <= 0.0f && float(m_chromaStrength) <= 0.0f;
    }

    bool HasGPU() const override { return true; }

    int GpuIterations() const override { return std::clamp(int(m_levels), 1, 6); }

    const char* GpuSource() const override {
        return R"(
Texture2D<float4>   Src  : register(t0);   // ping-pong: the accumulator
Texture2D<float4>   Orig : register(t1);   // the untouched input, every pass
RWTexture2D<float4> Dst  : register(u0);

cbuffer Params : register(b0) {
    uint  Width;
    uint  Height;
    uint  LumaBits;
    uint  ChromaBits;
    uint  Level;        // which scale this pass is working at
    uint  LastLevel;    // so the final pass can rotate back to RGB
    uint  LevelDepBits;
    uint  Colour;       // 1 when there are three channels to rotate
};

static const float kB3[5] = { 0.0625, 0.25, 0.375, 0.25, 0.0625 };

// Reversible luma/chroma rotation, matching the CPU exactly. Not colorimetric
// -- only ever a basis to threshold in -- but it round trips, which is what
// matters. Chroma tolerates far harder shrinkage than luma, and that asymmetry
// is most of what makes sensor-noise denoising work.
float4 ToYcc(float4 p) {
    return float4(0.25 * p.r + 0.5 * p.g + 0.25 * p.b, p.r - p.g, p.b - p.g, p.a);
}
float4 FromYcc(float4 p) {
    float g = p.x - 0.25 * p.y - 0.25 * p.z;
    return float4(p.y + g, g, p.z + g, p.a);
}

// Soft shrinkage. Hard thresholding leaves a discontinuity, and a coefficient
// sitting either side of it from one pixel to the next flips between kept and
// discarded -- the mottled "wavelet blotch" that gives the method its bad name.
float Shrink(float v, float t) {
    float a = abs(v);
    return (a <= t) ? 0.0 : (v > 0.0 ? a - t : -(a - t));
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;

    int  x = int(tid.x), y = int(tid.y);
    int  step = 1 << Level;
    int2 hi = int2(Width - 1, Height - 1);

    // Rotate into luma/chroma for THIS level and back out at the end of it,
    // rather than once around the whole pyramid.
    //
    // Not a stylistic choice: the ping-pong buffers take the OUTPUT's format,
    // and chroma is a signed difference. On an RGBA8 image the UNORM target
    // clamps every negative chroma value to zero the moment an intermediate is
    // written, which silently destroyed half the colour information -- it
    // agreed perfectly on float images and was wrong by 182/255 on 8-bit ones.
    // Keeping the rotation inside one dispatch means the buffers only ever hold
    // RGB, which is non-negative and survives. The CPU does the same, so the
    // two paths compute the same thing.
    float4 cur = Src[int2(x, y)];
    if (Colour != 0) cur = ToYcc(cur);

    float4 blur = 0.0;
    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            int2 p = clamp(int2(x + dx * step, y + dy * step), int2(0, 0), hi);
            float4 s = Src[p];
            if (Colour != 0) s = ToYcc(s);
            blur += (kB3[dx + 2] * kB3[dy + 2]) * s;
        }
    }

    float4 detail = cur - blur;

    // Coarser levels hold more real structure and less grain, so the threshold
    // falls as the scale grows -- matching how a noise field's energy
    // distributes across an a-trous pyramid.
    float falloff = 1.0 / float(1 << Level);
    // Threshold scaled by how bright this neighbourhood is, from the BLURRED
    // value rather than the pixel's own -- a noisy pixel would otherwise scale
    // its own threshold by its own noise. Matches LevelScale() on the CPU; the
    // agreement test is what keeps them the same.
    //
    // 0.18 literally, unlike the CPU which scales it. A UNORM texture reads as
    // 0..1 here whatever its storage, which is the same reason the thresholds
    // need no 255 -- see GpuConstants. The CPU's PixelBuffer does NOT do that:
    // it holds 0..255 for an 8-bit image, so its midpoint is scaled and this
    // one is not. Getting that backwards is what the parity test caught,
    // reporting a max difference of 78 against the usual 8.
    float lvl = 1.0;
    float amt = asfloat(LevelDepBits);
    if (amt > 0.0) {
        float rel = sqrt(max(blur.x, 1.8e-5) / 0.18);
        // 0.20/4.0, matching ClampLo/ClampHi on the CPU. Literal here because a
        // shader cannot read the environment -- the override exists for
        // sweeping, which is done on the CPU path, and the parity test is what
        // catches these two drifting apart.
        lvl = 1.0 + amt * (clamp(rel, 0.20, 4.0) - 1.0);
    }

    float lt = asfloat(LumaBits)   * falloff * lvl;
    float ct = (Colour != 0 ? asfloat(ChromaBits) : asfloat(LumaBits)) * falloff * lvl;

    detail.x = Shrink(detail.x, lt);
    detail.y = Shrink(detail.y, ct);
    detail.z = Shrink(detail.z, ct);

    float4 outv = blur + detail;
    outv.a = cur.a;                      // alpha is carried, never shrunk

    if (Colour != 0) outv = FromYcc(outv);
    Dst[tid.xy] = outv;
}
)";
    }

    std::vector<uint32_t> GpuConstants(int iteration) const override {
        auto bits = [](float f) {
            uint32_t u;
            std::memcpy(&u, &f, sizeof(u));
            return u;
        };
        const int levels = std::clamp(int(m_levels), 1, 6);
        // Thresholds are a fraction of the intensity range on both paths. A
        // float image reaches the shader in its own units and ValueScale() is
        // 1, so unlike the RGBA8 kernels there is no 255 to divide out -- and
        // the CPU/GPU agreement test is what catches getting that backwards.
        // Order must match the cbuffer declaration exactly -- LevelDepBits
        // sits before Colour, so it goes here rather than appended.
        return {bits(float(m_lumaStrength)),
                bits(float(m_chromaStrength)),
                uint32_t(iteration),
                uint32_t(levels - 1),
                bits(float(m_levelDep)),
                1u};
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

    Param<float> m_levelDep{this, "level_dep", 0.0f, 0.0f, 1.0f,
        {.help = "How much the threshold follows local brightness. Sensor "
                 "noise grows with signal, so on a pushed high-ISO frame the "
                 "shadows need a much lower threshold than the highlights. 0 "
                 "is one fixed threshold everywhere; 1 follows sqrt(level).",
         .step = 0.05}};

    Param<int> m_levels{this, "levels", 4, 1, 6,
        {.help = "How many scales to process. Each level is twice as coarse as "
                 "the last, so more levels remove larger-scale blotchiness at "
                 "the cost of time. 4 covers grain through to mottling.",
         .step = 1}};

    // 0.005, from measurement rather than taste.
    //
    // This defaulted to 0.02, taken from a synthetic fixture whose noise sigma
    // was 0.05 -- far noisier than a developed photograph. On a real one
    // (_DSC0162.ARW, pushed 1.85 stops) the entire luma detail measures 0.021,
    // so a 0.02 threshold is the SIZE of the signal and soft shrinkage removed
    // 62% of it. That is what "denoise destroyed the image" looked like.
    //
    // Luma also needs far less help than chroma on a Bayer sensor: green is
    // sampled at half of all sites while red and blue get a quarter each, so
    // the visible speckle is overwhelmingly chroma. Defaulting luma low and
    // chroma high matches where the noise actually is.
    Param<float> m_lumaStrength{this, "luma", 0.005f, 0.0f, 0.25f,
        {.help = "Shrinkage threshold for the luminance detail bands, as a "
                 "fraction of the intensity range. Raise it to remove more "
                 "grain; too high and fine texture goes with it -- on a "
                 "developed photo the whole luma detail is around 0.02, so "
                 "thresholds near that erase it. 0 leaves luminance untouched.",
         .step = 0.001, .softMin = 0.0, .softMax = 0.03}};

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
