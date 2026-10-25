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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <windows.h>

#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"
#include "../../core/parallel.h"

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

// THE PER-LEVEL THRESHOLD, and why it is measured.
//
// Each detail band w_l = c_l - c_{l+1} holds part of the noise, and the
// threshold for a band should follow how much. For WHITE noise that is fixed
// by the kernel -- a B3 a-trous band keeps 0.891, 0.200, 0.086, 0.041, 0.020
// of sigma, so relative to the finest 1, 0.22, 0.10, 0.05, 0.02 -- and the
// first rewrite used exactly that. It was wrong for photographs, measured:
// a developed frame's noise is not white. Demosaicing interpolates every
// colour from its neighbours, which spreads each sample's noise over two or
// three pixels, so the coarser bands carry far more of it than white noise
// would. In the flattest tenth of 32x32 blocks, as a multiple of the white
// prediction:
//
//                      level 1   level 2
//   _DSC0037 luma        1.8       2.2      (ISO 6400, pushed)
//            chroma      3.9       5.9
//   _MG_9673 luma        2.0       2.9      (ISO 1000)
//            chroma      3.4       4.8
//
// Chroma's level 1 holds 0.87 of its level 0 -- not 0.22. Thresholded by the
// white table, the fine grain went and the colour mottling one and two levels
// up stayed. So `adaptive` measures each level (NoiseModel below), and the
// fixed thresholds fall by half per level, which is the old rule and much the
// closer of the two to real noise.
constexpr int kMaxLevels = 6;

// The noise in each band, in normalised (0..1) units, luma and chroma, each
// twice: as measured, and with every coefficient divided by the
// sqrt-of-brightness scale level_dep applies, so a threshold that follows
// brightness starts from the noise at middle grey.
struct NoiseModel {
    float luma[kMaxLevels] = {}, chroma[kMaxLevels] = {};
    float lumaN[kMaxLevels] = {}, chromaN[kMaxLevels] = {};
    bool  valid = false;
};

// MEASURED IN THE FLAT PARTS OF THE IMAGE. Tiles spread over the frame are
// decomposed on the CPU (the same transform as the denoise), and each band's
// noise is the robust sigma, median(|w|)/0.6745, of its 32x32 blocks --
// taken over the flattest tenth of them, where there is nothing but noise to
// measure. Over the whole image the coarse bands measure the photograph's
// own structure: 25-65x the white prediction at level 3-4 rather than noise.
// Even the flattest blocks of a coarse band still hold gradients and
// vignetting, so a band is never allowed more noise than the one below it.
//
// `sample(x, y, rgb)` returns the pixel in normalised units.
template <typename Sample>
NoiseModel MeasureNoise(int w, int h, bool colour, Sample&& sample) {
    NoiseModel nm;
    constexpr int kB = 32;
    // A margin of the coarsest kernel's reach keeps the tile edge's clamping
    // out of the measured blocks; a small image gets less, and its coarse
    // bands are then capped by the finer ones anyway.
    const int tile = std::min({640, w, h});
    const int margin = std::min(4 * (1 << (kMaxLevels - 1)), tile / 4);
    const int inner = tile - 2 * margin;
    if (inner < kB) return nm;                                // too small to tell
    const int grid = 3;
    struct Stat { std::vector<float> s[4][kMaxLevels]; };     // luma, chroma, lumaN, chromaN
    std::vector<Stat> per(size_t(grid * grid));

    ParallelFor(per.size(), [&](size_t ti) {
        const int tx = int(ti) % grid, ty = int(ti) / grid;
        const int x0 = (w - tile) * tx / std::max(1, grid - 1);
        const int y0 = (h - tile) * ty / std::max(1, grid - 1);
        const int ch = 3;
        std::vector<float> c(size_t(tile) * size_t(tile) * ch), next, tmp;
        for (int y = 0; y < tile; ++y)
            for (int x = 0; x < tile; ++x) {
                float p[3];
                sample(x0 + x, y0 + y, p);
                float* q = &c[(size_t(y) * size_t(tile) + size_t(x)) * ch];
                if (colour) {
                    q[0] = 0.25f * p[0] + 0.5f * p[1] + 0.25f * p[2];
                    q[1] = p[0] - p[1];
                    q[2] = p[2] - p[1];
                } else {
                    q[0] = p[0]; q[1] = q[2] = 0.0f;
                }
            }
        Stat& st = per[ti];
        const int nb = inner / kB;
        std::vector<float> bl[4];
        for (int l = 0; l < kMaxLevels; ++l) {
            ATrousBlur(c, next, tile, tile, ch, 1 << l, tmp);
            for (int by = 0; by < nb; ++by)
                for (int bx = 0; bx < nb; ++bx) {
                    for (auto& v : bl) v.clear();
                    for (int y = 0; y < kB; ++y)
                        for (int x = 0; x < kB; ++x) {
                            const size_t i = (size_t(margin + by * kB + y) * size_t(tile) +
                                              size_t(margin + bx * kB + x)) * ch;
                            const float s = std::clamp(
                                std::sqrt(std::max(next[i], 1e-4f * 0.18f) / 0.18f), ClampLo(), ClampHi());
                            const float d0 = std::fabs(c[i] - next[i]);
                            bl[0].push_back(d0);
                            bl[2].push_back(d0 / s);
                            for (int k = 1; k < 3 && colour; ++k) {
                                const float d = std::fabs(c[i + size_t(k)] - next[i + size_t(k)]);
                                bl[1].push_back(d);
                                bl[3].push_back(d / s);
                            }
                        }
                    for (int q = 0; q < 4; ++q) {
                        if (bl[q].empty()) continue;
                        auto mid = bl[q].begin() + std::ptrdiff_t(bl[q].size() / 2);
                        std::nth_element(bl[q].begin(), mid, bl[q].end());
                        st.s[q][l].push_back(*mid / 0.6745f);
                    }
                }
            c.swap(next);
        }
    });

    float* outs[4] = {nm.luma, nm.chroma, nm.lumaN, nm.chromaN};
    auto median = [](std::vector<float> v) {
        if (v.empty()) return 0.0f;
        auto mid = v.begin() + std::ptrdiff_t(v.size() / 2);
        std::nth_element(v.begin(), mid, v.end());
        return *mid;
    };
    for (int q = 0; q < 4; ++q) {
        std::vector<std::vector<float>> lv(kMaxLevels);
        for (const Stat& st : per)
            for (int l = 0; l < kMaxLevels; ++l)
                lv[size_t(l)].insert(lv[size_t(l)].end(), st.s[q][l].begin(), st.s[q][l].end());
        const size_t nb = lv[0].size();
        if (nb == 0) continue;

        // HOW MUCH noise: the finest band over EVERY block. Taken from only
        // the flattest blocks it came out three times too low on _DSC0037 --
        // the blocks with the least going on are the crushed blacks and the
        // darkest shadows, which have the least noise of all.
        const float s0 = median(lv[0]);

        // ITS SHAPE across the levels: each band as a fraction of the finest,
        // from the blocks whose coarse band is smallest RELATIVE TO THEIR OWN
        // NOISE -- the ones where the coarse band is noise, not picture.
        // Blocks with no noise at all (clipped) say nothing and are skipped.
        std::vector<std::pair<float, size_t>> rel;
        for (size_t b = 0; b < nb; ++b)
            if (lv[0][b] > 1e-6f && b < lv[2].size()) rel.push_back({lv[2][b] / lv[0][b], b});
        if (rel.empty()) continue;
        std::sort(rel.begin(), rel.end());
        const size_t keep = std::max<size_t>(1, rel.size() / 10);
        outs[q][0] = s0;
        for (int l = 1; l < kMaxLevels; ++l) {
            std::vector<float> ratio;
            for (size_t k = 0; k < keep; ++k) {
                const size_t b = rel[k].second;
                if (b < lv[size_t(l)].size()) ratio.push_back(lv[size_t(l)][b] / lv[0][b]);
            }
            outs[q][l] = std::min(s0 * median(ratio), outs[q][l - 1]);
        }
    }
    nm.valid = true;
    return nm;
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

        // At a preview scale, the finest levels are of the full image's
        // coarser ones -- see ProxyDrop.
        const int drop   = ProxyDrop(ctx.PixelScale());
        const int levels = EffectiveLevels(drop);
        // Thresholds are expressed as a fraction of the intensity range, so a
        // setting means the same thing on an 8-bit scan and a float raw. The
        // same units trap that made brightness apply its offset 255x over.
        const bool colourIn = ch >= 3;
        // Measured into a LOCAL: one instance serves every frame of a group,
        // possibly at once, so nothing measured per frame may live on it.
        NoiseModel noise;
        if (bool(m_adaptive)) {
            const std::vector<float>& d = buf.Data();
            noise = MeasureNoise(w, h, colourIn, [&](int x, int y, float* p) {
                const float* q = &d[(size_t(y) * size_t(w) + size_t(x)) * size_t(ch)];
                for (int c = 0; c < 3; ++c) p[c] = q[colourIn ? c : 0] / scale;
            });
        }
        ctx.SetReport(Report(noise));
        const float levelDep = float(m_levelDep);
        const float midGrey  = 0.18f * scale;   // middle grey in the buffer's units

        // Working in a luma/chroma basis when the image has colour, because
        // sensor noise is markedly worse in chroma and chroma carries little
        // fine detail -- it can be shrunk far harder than luma without a
        // visible cost. That asymmetry is most of what makes ISO denoising
        // work, and it is invisible in an RGB basis where all three channels
        // get the same treatment.
        const bool colour = (ch >= 3);

        // THE STANDARD A-TROUS CASCADE: each level blurs the previous level's
        // APPROXIMATION, the band is their difference, and the result is the
        // coarsest approximation plus every shrunk band:
        //
        //     c_{l+1} = B_l * c_l,   w_l = c_l - c_{l+1},   out = c_L + sum S(w_l)
        //
        // It used to blur the previous level's RECONSTRUCTION instead -- the
        // denoised image so far -- so each "band" held every finer band again,
        // already shrunk once, and was shrunk again: the finest grain was cut
        // `levels` times over, and no threshold meant what it said. Without
        // shrinkage both forms return the input exactly, which is why the
        // zero-strength identity test never noticed.
        std::vector<float> approx = buf.Data();
        if (colour) ToYcc(approx, w, h, ch);
        std::vector<float> next, tmp;
        std::vector<float> acc(approx.size(), 0.0f);

        for (int lvl = 0; lvl < levels; ++lvl) {
            if (ctx.Cancelled()) return;
            ATrousBlur(approx, next, w, h, ch, 1 << lvl, tmp);

            // This level's thresholds, in the buffer's units.
            float lumaT = 0.0f, chromaT = 0.0f;
            LevelThresholds(noise, lvl, drop, &lumaT, &chromaT);
            lumaT *= scale;
            chromaT *= scale;

            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    const size_t i = (size_t(y) * size_t(w) + size_t(x)) * size_t(ch);

                    // How bright this neighbourhood is, from the coarser
                    // approximation rather than the pixel's own value: a noisy
                    // pixel would otherwise scale its own threshold by its own
                    // noise. Channel 0 is luma after the rotation, and the
                    // plain value when there is none.
                    const float lvlScale = LevelScale(next[i], levelDep, midGrey);

                    for (int c = 0; c < ch; ++c) {
                        if (c == 3) continue;                 // leave alpha alone
                        const bool isChroma = colour && c > 0;
                        const float t = (isChroma ? chromaT : lumaT) * lvlScale;
                        const float d = approx[i + size_t(c)] - next[i + size_t(c)];
                        acc[i + size_t(c)] += (t > 0.0f) ? Shrink(d, t) : d;
                    }
                }
            }
            approx.swap(next);
        }

        // Reassemble: the coarsest approximation plus every shrunk band.
        std::vector<float>& out = buf.Data();
        for (size_t i = 0; i < out.size(); ++i) {
            if (ch == 4 && (i % 4) == 3) continue;            // alpha as it was
            out[i] = approx[i] + acc[i];
        }
        if (colour) {
            // FromYcc rotates channels 0..2 in place and leaves alpha.
            FromYcc(out, w, h, ch);
        }
        buf.PackInto(dst);
    }

    // --- GPU implementation -------------------------------------------------
    //
    // The same cascade as the CPU, as explicit passes over four scratch
    // planes: two for the approximation and two for the running sum of shrunk
    // bands, each ping-ponged, since a pass may not read the buffer it writes.
    //
    //   start      input -> A0 (rotated to luma/chroma), C0 = 0
    //   level l    A, C  -> A' = B_l * A,  C' = C + S(A - A')
    //   finish     A, C  -> output = rotate back (A + C), alpha from the input
    //
    // HALF-FLOAT PLANES, signed. The old single-kernel version ping-ponged in
    // the OUTPUT's format, which on an RGBA8 image is UNORM: chroma is a signed
    // difference, so it had to rotate in and out of luma/chroma inside every
    // pass and round to 8 bits between levels -- the reason its CPU/GPU
    // tolerance was loose. Four RGBA16F planes cost 32 bytes a pixel, and hold
    // chroma, the cascade, and 11 bits of mantissa between levels.
    //
    // The blur is a 2D 5x5 dilated kernel rather than separable passes: a
    // separable pair would need another plane and another full-image round
    // trip per level, which on a GPU is the term that actually costs.
    bool IsNoOp() const override {
        if (bool(m_adaptive)) return float(m_lumaK) <= 0.0f && float(m_chromaK) <= 0.0f;
        return float(m_lumaStrength) <= 0.0f && float(m_chromaStrength) <= 0.0f;
    }

    bool HasGPU() const override { return true; }

    // The noise is measured from the input, on the CPU, before the passes:
    // a readback (~86 ms at 21 MP) plus a strided sample. A pure function of
    // the pixels, so the pipeline caches it while a slider is dragged.
    bool GpuNeedsInputPixels() const override { return bool(m_adaptive); }
    bool MeasurementDependsOnlyOnInput() const override { return true; }
    std::vector<uint32_t> SaveMeasurement() const override {
        if (!m_noise.valid) return {};
        std::vector<uint32_t> v;
        for (const float* a : {m_noise.luma, m_noise.chroma, m_noise.lumaN, m_noise.chromaN})
            for (int l = 0; l < kMaxLevels; ++l) v.push_back(FloatBits(a[l]));
        return v;
    }
    bool RestoreMeasurement(const std::vector<uint32_t>& v) override {
        if (v.size() != size_t(4 * kMaxLevels)) return false;
        float* outs[4] = {m_noise.luma, m_noise.chroma, m_noise.lumaN, m_noise.chromaN};
        for (int q = 0; q < 4; ++q)
            for (int l = 0; l < kMaxLevels; ++l)
                std::memcpy(&outs[q][l], &v[size_t(q * kMaxLevels + l)], 4);
        m_noise.valid = true;
        m_gpuValid = true;
        return true;
    }
    void MeasureForGpu(const std::vector<const Image*>& inputs) override {
        m_noise = NoiseModel{};
        m_gpuValid = false;
        if (inputs.empty() || !inputs[0]) return;
        ImageView v = const_cast<Image*>(inputs[0])->MapCpuRead();
        if (!v.data) return;
        const ImageDesc& vd = v.desc;
        const bool colour = vd.format != Format::R32F;
        // The sampled pixels read directly, not through PixelBuffer, whose
        // Unpack converts the whole image -- see dehaze's MeasureForGpu.
        m_noise = MeasureNoise(vd.width, vd.height, colour, [&](int x, int y, float* p) {
            switch (vd.format) {
                case Format::RGBA8: {
                    const uint8_t* q = v.At<uint8_t>(x, y);
                    for (int c = 0; c < 3; ++c) p[c] = float(q[c]) / 255.0f;
                    break;
                }
                case Format::RGBA32F: {
                    const float* q = v.At<float>(x, y);
                    for (int c = 0; c < 3; ++c) p[c] = q[c];
                    break;
                }
                case Format::RGBA16F: {
                    const uint16_t* q = v.At<uint16_t>(x, y);
                    for (int c = 0; c < 3; ++c) p[c] = HalfToFloat(q[c]);
                    break;
                }
                case Format::R32F: {
                    p[0] = p[1] = p[2] = *v.At<float>(x, y);
                    break;
                }
                default: p[0] = p[1] = p[2] = 0.0f;
            }
        });
        m_gpuValid = m_noise.valid;
    }
    // ONLY FOR THE GPU PATH, whose measurement is made once per stage in
    // MeasureForGpu; the CPU path reports per call (see RunCPU), because one
    // instance serves every frame of a group.
    std::string RunReport() const override { return m_gpuValid ? Report(m_noise) : std::string(); }

    void PrepareGpu(const std::vector<ImageDesc>& inputs) override {
        m_gpuColour = inputs.empty() || inputs[0].format != Format::R32F;
    }

    int        GpuScratchCount()  const override { return 4; }
    FormatSpec GpuScratchPlanes() const override { return FormatSpec::RGBA16F; }

    std::vector<GpuPass> GpuPasses() const override {
        const int levels = EffectiveLevels(ProxyDrop(GpuScaledPx(1.0f)));
        std::vector<GpuPass> p;
        p.push_back({kStart, "wavelet.start", {-1}, {0, 2}});
        for (int l = 0; l < levels; ++l) {
            const int a = l % 2, c = 2 + l % 2;
            p.push_back({kLevel, "wavelet.level", {a, c}, {1 - a, 5 - c}});
        }
        p.push_back({kFinish, "wavelet.finish", {levels % 2, 2 + levels % 2, -1}, {-1}});
        return p;
    }

    std::vector<uint32_t> GpuPassConstants(int pass) const override {
        const int drop   = ProxyDrop(GpuScaledPx(1.0f));
        const int levels = EffectiveLevels(drop);
        const int l = std::clamp(pass - 1, 0, levels - 1);
        // Thresholds are a fraction of the intensity range on both paths: a
        // texture reads 0..1 whatever its storage, so unlike the CPU's
        // PixelBuffer there is no 255 to apply.
        float lt = 0.0f, ct = 0.0f;
        LevelThresholds(m_noise, l, drop, &lt, &ct);
        return {FloatBits(lt),
                FloatBits(ct),
                uint32_t(1 << l),
                FloatBits(float(m_levelDep)),
                m_gpuColour ? 1u : 0u};
    }

private:
    bool m_gpuColour = true;
    NoiseModel m_noise;          // GPU path only: see RunReport
    bool       m_gpuValid = false;

    // Level l's thresholds, in normalised (0..1) units: k times the noise
    // measured in that band when adaptive, else the fixed luma/chroma halved
    // per level.
    //
    // With level_dep the threshold is later scaled by brightness relative to
    // middle grey, so the base is the noise AT middle grey -- the normalised
    // estimate -- blended by how much brightness dependence is asked for.
    // AT A PREVIEW SCALE the image is smaller, so each a-trous level covers
    // more of the picture: at 1/4 scale, level 0 is the scale full-resolution
    // level 2 works at. Run unchanged, a 4-level denoise of the preview took
    // out structure up to 4x coarser than the final render does -- the preview
    // looked smoother than what it previewed. So the `drop` finest levels go
    // (they are below the preview's pixels), the rest run at the thresholds
    // of the full-resolution levels they stand for, and at least one level
    // always runs. With adaptive thresholds nothing more is needed: the noise
    // is measured on the preview itself, level by level.
    static int ProxyDrop(float scale) {
        if (!(scale > 0.0f) || scale >= 1.0f) return 0;
        return std::max(0, int(std::lround(std::log2(1.0 / double(scale)))));
    }
    int EffectiveLevels(int drop) const {
        return std::max(1, std::clamp(int(m_levels), 1, kMaxLevels) - drop);
    }

    // How far the a-trous cascade reads: 2 * (2^L - 1) pixels of THIS run.
    // Region rendering needs it to crop with a margin -- without one, a view
    // zoomed in denoised its edge against a clamped border.
    int ReachPixels() const override {
        return 2 * ((1 << std::clamp(int(m_levels), 1, kMaxLevels)) - 1);
    }

    // The measured noise is the WHOLE picture's. Measured on a region it is
    // that region's -- a patch of sky against a patch of shadow -- and the
    // thresholds would change as the view pans. Fixed thresholds do not care.
    bool RegionSafe() const override { return !bool(m_adaptive); }

    void LevelThresholds(const NoiseModel& nm, int l, int drop, float* luma, float* chroma) const {
        l = std::clamp(l, 0, kMaxLevels - 1);
        if (!bool(m_adaptive)) {
            const float f = 1.0f / float(1 << std::min(l + drop, 30));
            *luma = float(m_lumaStrength) * f;
            *chroma = float(m_chromaStrength) * f;
            return;
        }
        const float a = std::clamp(float(m_levelDep), 0.0f, 1.0f);
        const float sl = nm.luma[l] + a * (nm.lumaN[l] - nm.luma[l]);
        const float sc = nm.chroma[l] + a * (nm.chromaN[l] - nm.chroma[l]);
        *luma = float(m_lumaK) * sl;
        *chroma = float(m_chromaK) * sc;
    }

    std::string Report(const NoiseModel& nm) const {
        const int levels = std::clamp(int(m_levels), 1, kMaxLevels);
        std::string r;
        char b[64];
        if (bool(m_adaptive) && nm.valid) {
            r = "noise by level, luma";
            for (int l = 0; l < levels; ++l) { std::snprintf(b, sizeof b, " %.4f", double(nm.luma[l])); r += b; }
            r += ", chroma";
            for (int l = 0; l < levels; ++l) { std::snprintf(b, sizeof b, " %.4f", double(nm.chroma[l])); r += b; }
            std::snprintf(b, sizeof b, "; thresholds %.2g / %.2g sigma", double(m_lumaK), double(m_chromaK));
            r += b;
        } else {
            float l0 = 0, c0 = 0;
            LevelThresholds(nm, 0, 0, &l0, &c0);
            std::snprintf(b, sizeof b, "thresholds %.4f / %.4f (fixed)", double(l0), double(c0));
            r = b;
        }
        return r;
    }


    // Each pass declares the same constant buffer and helpers: the framework
    // compiles each pass on its own, and one that omitted a field would
    // misread the ones after it.
#define WAVELET_COMMON R"(
cbuffer Params : register(b0) {
    uint  Width;
    uint  Height;
    uint  LumaBits;     // this level's luma threshold
    uint  ChromaBits;   // ...and chroma's
    uint  Step;         // the kernel's dilation at this level
    uint  LevelDepBits;
    uint  Colour;       // 1 when there are three channels to rotate
};
// Reversible luma/chroma rotation, matching the CPU exactly.
float4 ToYcc(float4 p) {
    return float4(0.25 * p.r + 0.5 * p.g + 0.25 * p.b, p.r - p.g, p.b - p.g, p.a);
}
float4 FromYcc(float4 p) {
    float g = p.x - 0.25 * p.y - 0.25 * p.z;
    return float4(p.y + g, g, p.z + g, p.a);
}
)"

    static constexpr const char* kStart = WAVELET_COMMON R"(
Texture2D<float4>   Src : register(t0);
RWTexture2D<float4> A   : register(u0);
RWTexture2D<float4> C   : register(u1);
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    float4 v = Src[tid.xy];
    A[tid.xy] = (Colour != 0) ? ToYcc(v) : v;
    C[tid.xy] = 0.0;
}
)";

    static constexpr const char* kLevel = WAVELET_COMMON R"(
Texture2D<float4>   A    : register(t0);
Texture2D<float4>   C    : register(t1);
RWTexture2D<float4> ADst : register(u0);
RWTexture2D<float4> CDst : register(u1);
static const float kB3[5] = { 0.0625, 0.25, 0.375, 0.25, 0.0625 };
// Soft shrinkage: continuous at the threshold, so a coefficient near it moves
// smoothly instead of flipping between kept and discarded (the "blotch").
float Shrink(float v, float t) {
    float a = abs(v);
    return (a <= t) ? 0.0 : (v > 0.0 ? a - t : -(a - t));
}
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    int x = int(tid.x), y = int(tid.y), step = int(Step);
    int2 hi = int2(Width - 1, Height - 1);
    float4 blur = 0.0;
    for (int dy = -2; dy <= 2; ++dy)
        for (int dx = -2; dx <= 2; ++dx) {
            int2 p = clamp(int2(x + dx * step, y + dy * step), int2(0, 0), hi);
            blur += (kB3[dx + 2] * kB3[dy + 2]) * A[p];
        }
    float4 d = A[tid.xy] - blur;

    // Threshold scaled by how bright this neighbourhood is, from the coarser
    // approximation -- matches LevelScale() on the CPU, 0.20/4.0 bounds and
    // all; the agreement test keeps them the same.
    float lvl = 1.0;
    float amt = asfloat(LevelDepBits);
    if (amt > 0.0) {
        float rel = sqrt(max(blur.x, 1.8e-5) / 0.18);
        lvl = 1.0 + amt * (clamp(rel, 0.20, 4.0) - 1.0);
    }
    float lt = asfloat(LumaBits) * lvl;
    float ct = (Colour != 0 ? asfloat(ChromaBits) : asfloat(LumaBits)) * lvl;
    float4 s = d;
    if (lt > 0.0) s.x = Shrink(d.x, lt);
    if (ct > 0.0) { s.y = Shrink(d.y, ct); s.z = Shrink(d.z, ct); }
    s.w = 0.0;
    ADst[tid.xy] = blur;
    CDst[tid.xy] = C[tid.xy] + s;
}
)";

    static constexpr const char* kFinish = WAVELET_COMMON R"(
Texture2D<float4>   A   : register(t0);
Texture2D<float4>   C   : register(t1);
Texture2D<float4>   Src : register(t2);
RWTexture2D<float4> Dst : register(u0);
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    float4 v = A[tid.xy] + C[tid.xy];
    if (Colour != 0) v = FromYcc(v);
    v.a = Src[tid.xy].a;                 // alpha is carried, never shrunk
    Dst[tid.xy] = v;
}
)";
#undef WAVELET_COMMON

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


    Param<bool> m_adaptive{this, "adaptive", true,
        "Measure the noise in this image and set the thresholds as multiples "
        "of it (luma_k, chroma_k), instead of the fixed luma and chroma "
        "values. A fixed threshold is right for one noise level: too strong "
        "on a clean ISO 100 frame, too weak on a pushed ISO 6400 one. The "
        "report gives the measured noise."};

    Param<float> m_lumaK{this, "luma_k", 0.75f, 0.0f, 5.0f,
        {.help = "With adaptive on: the luma threshold, in multiples of the "
                 "measured luma noise, band by band. 0.75 rather than 1 because "
                 "low-contrast texture -- fur, foliage -- sits at about the "
                 "noise's own size: compared on an ISO 1000 cat and an ISO 6400 "
                 "pushed frame, 0.75 kept visibly more hair for a little more "
                 "grain. 0 leaves luma untouched.",
         .step = 0.05}};

    Param<float> m_chromaK{this, "chroma_k", 3.0f, 0.0f, 10.0f,
        {.help = "With adaptive on: the chroma threshold, in multiples of the "
                 "measured chroma noise. Higher than luma's for the same reason "
                 "the fixed chroma is: colour noise is worse on a Bayer sensor "
                 "and colour carries little fine detail.",
         .step = 0.1}};

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
        {.help = "With adaptive off: the shrinkage threshold for the finest "
                 "luminance band, as a "
                 "fraction of the intensity range. Raise it to remove more "
                 "grain; too high and fine texture goes with it -- on a "
                 "developed photo the whole luma detail is around 0.02, so "
                 "thresholds near that erase it. 0 leaves luminance untouched.",
         .step = 0.001, .softMin = 0.0, .softMax = 0.03}};

    Param<float> m_chromaStrength{this, "chroma", 0.08f, 0.0f, 0.5f,
        {.help = "With adaptive off: the threshold for the finest colour "
                 "band. Defaults "
                 "well above luma because sensor noise is far worse in chroma "
                 "and colour carries little fine detail, so it tolerates much "
                 "harder smoothing before anything shows.",
         .step = 0.005, .softMin = 0.0, .softMax = 0.25}};
};

REGISTER_ALGORITHM(WaveletDenoise);

} // namespace
} // namespace tglab
