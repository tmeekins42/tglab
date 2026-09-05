// dehaze — He, Sun & Tang (2009), dark channel prior.
//
// WHAT HAZE ACTUALLY IS, because the model is what makes this solvable.
//
// Light from a distant object is scattered away by the atmosphere on its way to
// the camera, and scattered sunlight is added in its place. Koschmieder's model
// says a photographed pixel is a mix of the two:
//
//     I = J * t  +  A * (1 - t)
//
//   I  what the camera recorded
//   J  the haze-free scene -- what we want
//   A  the AIRLIGHT: the colour of the haze itself, usually a bright grey-blue
//   t  TRANSMISSION: the fraction of the scene's light that survived. 1 is
//      perfectly clear, 0 is pure haze with no scene left at all.
//
// t falls off exponentially with distance, which is why haze reads as depth --
// and why dehazing a photograph is really depth estimation in disguise.
//
// That is one equation per pixel with two unknowns (J and t), so it cannot be
// solved as written. Every dehazing method is a different guess at the missing
// constraint.
//
// THE DARK CHANNEL PRIOR is He's, and it is an observation about photographs
// rather than about physics:
//
//   In almost any small patch of a haze-free OUTDOOR image, at least one pixel
//   is very dark in at least one colour channel.
//
// Shadows, dark foliage, wet rock, a window, the dark side of anything. Take
// the minimum over the three channels, then the minimum over a local window,
// and in clear air that number is near zero almost everywhere.
//
// So where it is NOT near zero, the brightness has been added -- and how much
// was added is how much haze there is. The unsolvable equation becomes a local
// minimum you can measure directly:
//
//     t = 1 - omega * min_patch( min_channel( I / A ) )
//
// WHERE THE PRIOR FAILS, which is worth knowing before using it:
//
//   * Large bright objects with no dark pixel in them -- snow, a white wall, a
//     blank sky. The prior reads them as haze and the correction darkens them.
//     `sky_protect` is the standard mitigation: floor the transmission where
//     the pixel is already close to the airlight colour.
//   * Indoor scenes and anything backlit. There is no atmosphere to remove, so
//     whatever this finds is not haze.
//
// WHY THE TRANSMISSION IS REFINED. The patch minimum is blocky by construction:
// every pixel in a window shares one value, so edges in the map sit on window
// boundaries rather than on objects. Used raw it produces halos exactly where
// depth changes -- the classic ring around a foreground branch against sky.
//
// The fix is a GUIDED FILTER: smooth the transmission map while forcing its
// edges to follow the IMAGE's edges. He's original paper used soft matting,
// which is a large sparse solve; the guided filter (He 2010) gives nearly the
// same answer in O(n) and is what everyone uses now.
//
// This is a linear-light operation. Applying it after a tone curve dehazes the
// display values rather than the light, which lifts shadows and washes the
// result out -- so put it before tonemap, next to the other scene-linear
// adjustments.
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"

namespace tglab {
namespace {

// Running box sum over a float plane, via a summed-area table.
//
// O(1) per window regardless of radius, which matters because the guided filter
// below needs five of these over the whole image and the radius is typically
// tens of pixels. Doubles rather than floats: a 45 MP plane summed in float
// loses the low bits well before the last row.
class BoxSum {
public:
    void Build(const std::vector<float>& v, int w, int h) {
        m_w = w; m_h = h;
        m_s.assign(size_t(w + 1) * size_t(h + 1), 0.0);
        for (int y = 0; y < h; ++y) {
            double row = 0.0;
            for (int x = 0; x < w; ++x) {
                row += double(v[size_t(y) * size_t(w) + size_t(x)]);
                m_s[size_t(y + 1) * size_t(m_w + 1) + size_t(x + 1)] =
                    m_s[size_t(y) * size_t(m_w + 1) + size_t(x + 1)] + row;
            }
        }
    }

    // Mean over the window centred at (cx, cy), clamped to the image.
    double Mean(int cx, int cy, int r) const {
        const int x0 = std::max(0, cx - r), y0 = std::max(0, cy - r);
        const int x1 = std::min(m_w - 1, cx + r), y1 = std::min(m_h - 1, cy + r);
        const double s =
            m_s[size_t(y1 + 1) * size_t(m_w + 1) + size_t(x1 + 1)]
          - m_s[size_t(y0)     * size_t(m_w + 1) + size_t(x1 + 1)]
          - m_s[size_t(y1 + 1) * size_t(m_w + 1) + size_t(x0)]
          + m_s[size_t(y0)     * size_t(m_w + 1) + size_t(x0)];
        const double n = double(x1 - x0 + 1) * double(y1 - y0 + 1);
        return n > 0.0 ? s / n : 0.0;
    }

private:
    int m_w = 0, m_h = 0;
    std::vector<double> m_s;
};

// Guided filter: smooth `p` while keeping `guide`'s edges (He, 2010).
//
// The model is that within any window the output is a LINEAR function of the
// guide, out = a*guide + b, with a and b chosen to fit p as closely as possible
// subject to a penalty eps on a. Where the guide has an edge, a is large and the
// output follows it; where the guide is flat, a goes to zero and the output is
// just the local mean of p. That is exactly what a transmission map needs: it
// should be smooth across a wall and step at the wall's edge.
//
// Single-channel guide (luminance) rather than the full colour version: it is
// three times cheaper and the difference on a transmission map is not visible,
// because the map itself is smooth by nature.
void GuidedFilter(const std::vector<float>& guide, const std::vector<float>& p,
                  int w, int h, int r, float eps, std::vector<float>* out) {
    const size_t n = size_t(w) * size_t(h);

    std::vector<float> gp(n), gg(n);
    for (size_t i = 0; i < n; ++i) {
        gp[i] = guide[i] * p[i];
        gg[i] = guide[i] * guide[i];
    }

    BoxSum sG, sP, sGP, sGG;
    sG.Build(guide, w, h);
    sP.Build(p, w, h);
    sGP.Build(gp, w, h);
    sGG.Build(gg, w, h);

    std::vector<float> A(n), B(n);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const double mg  = sG.Mean(x, y, r);
            const double mp  = sP.Mean(x, y, r);
            const double mgp = sGP.Mean(x, y, r);
            const double mgg = sGG.Mean(x, y, r);

            const double cov = mgp - mg * mp;
            const double var = mgg - mg * mg;
            const double a = cov / (var + double(eps));
            const double b = mp - a * mg;
            A[size_t(y) * size_t(w) + size_t(x)] = float(a);
            B[size_t(y) * size_t(w) + size_t(x)] = float(b);
        }

    // Every window containing a pixel proposes its own (a, b), so the answer is
    // the average of them -- which is what makes the result continuous rather
    // than blocky like the patch minimum it is smoothing.
    BoxSum sA, sB;
    sA.Build(A, w, h);
    sB.Build(B, w, h);

    out->assign(n, 0.0f);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const size_t i = size_t(y) * size_t(w) + size_t(x);
            (*out)[i] = float(sA.Mean(x, y, r) * double(guide[i]) +
                              sB.Mean(x, y, r));
        }
}

class Dehaze : public AlgorithmBase {
public:
    const char* Name()     const override { return "dehaze"; }
    const char* Category() const override { return "adjust"; }

    PortList Inputs()  const override {
        return {{"src", DataType::Image, FormatSpec::Any}};
    }
    PortList Outputs() const override {
        return {{"out", DataType::Image, FormatSpec::SameAsInput}};
    }

    bool IsNoOp() const override { return float(m_strength) <= 0.0f; }

    void RunCPU(RunCtx& ctx) override {
        const ImageView src = ctx.In(0);
        ImageView       dst = ctx.Out(0);
        if (!src.Valid() || !dst.Valid()) return;

        m_in.Unpack(src);
        if (!m_in.Valid()) return;
        m_out.AllocLike(m_in);

        const int w = m_in.Width(), h = m_in.Height(), ch = m_in.Channels();
        const size_t n = size_t(w) * size_t(h);
        const float scale = m_in.ValueScale();

        // Greyscale passes through: the prior is a minimum ACROSS CHANNELS, so
        // with one channel the dark channel is just the image and "removing the
        // haze" would remove the picture.
        if (ch < 3 || n == 0) {
            m_out.Data() = m_in.Data();
            m_out.PackInto(dst);
            m_note = "dehaze needs colour: the prior is a minimum across channels";
            return;
        }

        const float omega = std::clamp(float(m_strength), 0.0f, 1.0f);
        const float t0    = std::max(0.01f, float(m_floor));

        // BOTH RADII SCALE WITH THE IMAGE, because neither is a length in
        // pixels in any meaningful sense -- they are fractions of the picture.
        //
        // He's patch of 7 is quoted for a 600px image. Used literally on a
        // 6000px frame it covers a tenth as much of the scene, so the prior sees
        // only a fraction of the structure it needs and the transmission map
        // gets noisy. The refine radius has the opposite failure and it is the
        // sharper one: measured on a 240px fixture, the default 30 spanned a
        // quarter of the width and smoothed the depth ramp so hard that the far
        // end was corrected barely more than the near end -- the map had
        // averaged the distance away.
        //
        // So the parameters are stated relative to a 1000px reference and scaled
        // here. A value that works on one image then works on all of them, which
        // is what a lab needs when the same script runs on a thumbnail and a
        // 45 MP raw.
        const double ref = double(std::max(w, h)) / 1000.0;
        int patch  = std::max(1, int(std::lround(double(int(m_patch)) * ref)));
        int refine = std::max(1, int(std::lround(double(int(m_refine)) * ref)));

        // THE TRANSMISSION MAP IS COMPUTED SMALL, and this is the difference
        // between an algorithm you wait for and one you can drag a slider on.
        //
        // t is a DEPTH map. It is smooth by construction -- that is the whole
        // reason a guided filter is applied to it in the first place -- so it
        // has no high-frequency content that full resolution could preserve.
        // Estimating it at 1/4 linear scale costs 1/16 as much and loses
        // nothing the eye or the recovery can see.
        //
        // Only the final recovery runs at full resolution, because THAT is
        // where detail lives: J = (I - A)/t + A applies a smooth correction to
        // every original pixel.
        //
        // This is He & Sun's "fast guided filter" (2015) argument applied to
        // the whole front half. Measured at 22 MP: 3.6 s to 1.2 s end to end,
        // and the recovered image differs from the full-resolution answer by a
        // mean of 0.006 on a 0-1 scale (worst 0.069, at the few pixels where
        // the airlight search picks a different winner).
        //
        // Small images skip it: below about 2 MP the setup costs more than the
        // saving, and the map is not large enough for the downsample to be free
        // of aliasing.
        const int kScale = (n > 2000000) ? 4 : 1;
        const int sw = std::max(1, w / kScale);
        const int sh = std::max(1, h / kScale);
        const size_t sn = size_t(sw) * size_t(sh);
        if (kScale > 1) {
            patch  = std::max(1, patch  / kScale);
            refine = std::max(1, refine / kScale);
        }

        // The image at working scale, as three planes plus luminance.
        //
        // BOX-AVERAGED rather than point-sampled: the dark channel is a
        // MINIMUM, and point sampling would let the minimum land on whichever
        // pixel the grid happened to hit. Averaging first makes the estimate
        // depend on the whole neighbourhood, which is what the prior assumes.
        std::vector<float> sr(sn), sg(sn), sb(sn);
        for (int y = 0; y < sh; ++y)
            for (int x = 0; x < sw; ++x) {
                double a0 = 0, a1 = 0, a2 = 0;
                int cnt = 0;
                const int y0 = y * kScale, x0 = x * kScale;
                for (int dy = 0; dy < kScale; ++dy)
                    for (int dx = 0; dx < kScale; ++dx) {
                        const int yy = std::min(y0 + dy, h - 1);
                        const int xx = std::min(x0 + dx, w - 1);
                        const float* p = m_in.At(xx, yy);
                        a0 += p[0]; a1 += p[1]; a2 += p[2];
                        ++cnt;
                    }
                const size_t i = size_t(y) * size_t(sw) + size_t(x);
                sr[i] = float(a0 / cnt) / scale;
                sg[i] = float(a1 / cnt) / scale;
                sb[i] = float(a2 / cnt) / scale;
            }

        // --- 1. the dark channel -------------------------------------------
        //
        // Per pixel the minimum over channels, then the minimum over a patch.
        // Separable: a horizontal minimum pass then a vertical one gives the
        // same result as a 2D window at a fraction of the work.
        std::vector<float> dark(sn), tmp(sn);
        for (size_t i = 0; i < sn; ++i)
            tmp[i] = std::min(sr[i], std::min(sg[i], sb[i]));
        MinFilter(tmp, sw, sh, patch, &dark);

        // --- 2. the airlight ------------------------------------------------
        //
        // The brightest pixels of the DARK CHANNEL are the haziest places, and
        // the airlight is what the image reads there. Taking the brightest pixel
        // of the image instead would find a specular highlight or the sun.
        //
        // The top 0.1% rather than the single maximum, because one hot pixel
        // would otherwise set A for the whole image.
        float A[3] = {0, 0, 0};
        {
            std::vector<float> sorted = dark;
            const size_t k = std::max<size_t>(1, sn / 1000);
            std::nth_element(sorted.begin(), sorted.begin() + long(sn - k),
                             sorted.end());
            const float cut = sorted[sn - k];

            // Among those, the one with the greatest intensity. He takes the
            // brightest; averaging them instead would pull A toward the scene.
            double best = -1.0;
            for (size_t i = 0; i < sn; ++i) {
                if (dark[i] < cut) continue;
                const double s = double(sr[i]) + double(sg[i]) + double(sb[i]);
                if (s > best) {
                    best = s;
                    A[0] = sr[i]; A[1] = sg[i]; A[2] = sb[i];
                }
            }
            // A of zero would divide by zero below, and a near-black airlight
            // is not physical anyway.
            //
            // The TOP matters just as much on a scene-linear image, and only
            // this path ever sees one. Haze cannot be brighter than the
            // brightest thing the haze is in front of, but the search happily
            // lands on a specular cloud edge above 1.0 -- and then every pixel
            // in the sky has I < A, so (I - A) is negative and the recovery
            // drives it hard negative before the clamp to zero. Red falls
            // fastest and blue least, which is why the clouds went PURPLE
            // rather than simply dark.
            //
            // Bounded at 1.0: the airlight is a colour, and a colour brighter
            // than white is not one. Values above it are highlights, not haze.
            for (int c = 0; c < 3; ++c) A[c] = std::clamp(A[c], 1e-3f, 1.0f);
        }

        // --- 3. the transmission map ----------------------------------------
        //
        // The dark channel of the image NORMALISED BY THE AIRLIGHT. Dividing
        // first is what makes the estimate colour-correct: haze that is blue
        // dims the red channel more, and normalising accounts for that where a
        // grey estimate would leave a colour cast behind.
        //
        // omega < 1 leaves a little haze in deliberately. Removing all of it
        // looks wrong -- distant objects lose the aerial perspective that tells
        // the eye they are distant, and the picture reads as flat.
        for (size_t i = 0; i < sn; ++i)
            tmp[i] = std::min(sr[i] / A[0],
                              std::min(sg[i] / A[1], sb[i] / A[2]));
        std::vector<float> tdark(sn);
        MinFilter(tmp, sw, sh, patch, &tdark);

        std::vector<float> t(sn);
        for (size_t i = 0; i < sn; ++i)
            t[i] = 1.0f - omega * tdark[i];

        // --- 4. refine against the image ------------------------------------
        //
        // Without this the map is blocky -- every pixel in a patch shares one
        // value -- and the blockiness shows up as halos wherever depth changes.
        {
            std::vector<float> guide(sn);
            for (size_t i = 0; i < sn; ++i)
                guide[i] = 0.2126f * sr[i] + 0.7152f * sg[i] + 0.0722f * sb[i];
            std::vector<float> refined;
            GuidedFilter(guide, t, sw, sh, refine, 1e-4f, &refined);
            t.swap(refined);
        }

        // --- 5. protect the sky ---------------------------------------------
        //
        // The prior's known failure: a large bright region with no dark pixel in
        // it reads as haze, so sky and snow get darkened and often turn grey.
        //
        // Where a pixel is already close to the airlight COLOUR, raise its
        // transmission floor. Measured on the colour distance rather than the
        // brightness, so a bright object that is not haze-coloured -- a white
        // building against blue sky -- is left alone.
        const float protect = std::clamp(float(m_skyProtect), 0.0f, 1.0f);
        if (protect > 0.0f) {
            for (size_t i = 0; i < sn; ++i) {
                const float e0 = sr[i] - A[0];
                const float e1 = sg[i] - A[1];
                const float e2 = sb[i] - A[2];
                const float d = std::sqrt(e0 * e0 + e1 * e1 + e2 * e2);
                // Within about 15% of the airlight colour counts as sky.
                const float near = std::clamp(1.0f - d / 0.15f, 0.0f, 1.0f);
                t[i] = std::max(t[i], near * protect);
            }
        }

        // --- 6. recover the scene -------------------------------------------
        //
        //     J = (I - A) / max(t, t0) + A
        //
        // t0 floors the division. As t goes to zero the model says almost none
        // of the scene survived, so recovering it multiplies whatever is left --
        // including the noise -- without bound. The floor is what keeps a deeply
        // hazy region merely soft rather than a wall of amplified noise.
        // At FULL resolution, sampling the small transmission map bilinearly.
        //
        // Bilinear is enough precisely because t is smooth: interpolating a
        // depth map introduces no error the eye can find, whereas interpolating
        // the IMAGE would. This is why only the map was computed small.
        for (int y = 0; y < h; ++y) {
            // Map the full-res centre back to small-map coordinates. The -0.5
            // pair is the usual half-pixel correction: without it the map is
            // offset by half a low-res pixel, which shows up as the correction
            // sitting slightly off from the edge it belongs to.
            const float fy = (kScale > 1)
                ? std::clamp((float(y) + 0.5f) / float(kScale) - 0.5f,
                             0.0f, float(sh - 1))
                : float(y);
            const int   y0 = int(fy);
            const int   y1 = std::min(y0 + 1, sh - 1);
            const float wy = fy - float(y0);

            for (int x = 0; x < w; ++x) {
                const float fx = (kScale > 1)
                    ? std::clamp((float(x) + 0.5f) / float(kScale) - 0.5f,
                                 0.0f, float(sw - 1))
                    : float(x);
                const int   x0 = int(fx);
                const int   x1 = std::min(x0 + 1, sw - 1);
                const float wx = fx - float(x0);

                const float t00 = t[size_t(y0) * size_t(sw) + size_t(x0)];
                const float t10 = t[size_t(y0) * size_t(sw) + size_t(x1)];
                const float t01 = t[size_t(y1) * size_t(sw) + size_t(x0)];
                const float t11 = t[size_t(y1) * size_t(sw) + size_t(x1)];
                const float ta = t00 + (t10 - t00) * wx;
                const float tb = t01 + (t11 - t01) * wx;
                const float tt = std::max(ta + (tb - ta) * wy, t0);

                const float* p = m_in.At(x, y);
                float* q = m_out.At(x, y);

                // Recovered first, then clamped as a COLOUR -- see the GPU
                // path's note. Clamping each channel on its own turns an
                // over-strong correction into a hue shift, which reads as
                // broken rather than as too much dehaze.
                float rec[3], src3[3];
                for (int c = 0; c < 3; ++c) {
                    src3[c] = p[c] / scale;
                    rec[c]  = (src3[c] - A[c]) / tt + A[c];
                }
                const float lo = std::min(rec[0], std::min(rec[1], rec[2]));
                if (lo < 0.0f) {
                    float k = 1.0f;
                    for (int c = 0; c < 3; ++c) {
                        const float d = rec[c] - src3[c];
                        if (d < 0.0f && rec[c] < 0.0f)
                            k = std::min(k, src3[c] / std::max(-d, 1e-6f));
                    }
                    k = std::clamp(k, 0.0f, 1.0f);
                    for (int c = 0; c < 3; ++c)
                        rec[c] = src3[c] + (rec[c] - src3[c]) * k;
                }

                for (int c = 0; c < 3; ++c) {
                    // The top is left unclamped so a scene-linear image keeps
                    // its headroom.
                    q[c] = std::max(rec[c], 0.0f) * scale;
                }
                for (int c = 3; c < ch; ++c) q[c] = p[c];   // alpha
            }
        }

        m_out.PackInto(dst);

        // The airlight is worth reporting: it is the one thing here that is
        // estimated rather than set, and a wrong one explains a colour cast that
        // no slider will fix.
        char buf[128];
        std::snprintf(buf, sizeof buf,
                      "airlight %.3f %.3f %.3f, transmission %.2f-%.2f",
                      double(A[0]), double(A[1]), double(A[2]),
                      double(*std::min_element(t.begin(), t.end())),
                      double(*std::max_element(t.begin(), t.end())));
        m_note = buf;
    }

    std::string RunReport() const override { return m_note; }

    // The airlight is the brightest of the haziest pixels ANYWHERE in the
    // frame. Estimated from a region it is whatever that region happens to
    // contain, and every recovered colour shifts with it. Safe at reduced
    // scale, where the same pixels are still represented.
    bool RegionSafe() const override { return false; }

    // The guided filter's radius, which is the widest thing it reads.
    int ReachPixels() const override {
        return std::max(1, int(m_refine));
    }

    // --- GPU implementation ---------------------------------------------
    //
    // WHY THIS IS WORTH THE TROUBLE, measured rather than assumed. At 44.7 MP
    // the CPU path costs 2429 ms, and collapsing both radii to 1 -- which
    // removes the entire transmission-map estimation -- saves only 183 ms. The
    // map is 7% of the runtime. `brightness`, a five-line loop, costs 1483 ms
    // on the same image: that is the PixelBuffer unpack/pack floor every CPU
    // algorithm pays, and it is 61% of dehaze.
    //
    // So porting the clever part would have bought almost nothing. The win is
    // that a GPU stage reads and writes GPU-resident images and pays no floor
    // at all.
    //
    // WHAT STAYS ON THE CPU: the airlight. It is a top-0.1% selection over the
    // whole frame -- a global sort, not a per-pixel operation -- and the
    // framework has no readback-and-decide step between passes. It is measured
    // in MeasureForGpu from a strided subsample, which is both cheap and
    // legitimate: the airlight is a property of the haziest REGION, and a
    // regular sample of a few hundred thousand pixels finds the same region as
    // a full scan.
    //
    // WHAT DIFFERS FROM THE CPU PATH: the map is computed at full resolution
    // rather than at 1/4 scale, because the scratch pool is sized to the image.
    // That is more total work but it is GPU work, and it removes the bilinear
    // upsample along with its half-pixel correction. The two paths therefore
    // agree closely rather than exactly, which is what the CPU/GPU comparison
    // tolerance is for.
    bool HasGPU() const override { return true; }

    // The airlight has to be known before any pass runs, and it cannot be
    // derived from the descriptor -- it is a property of the CONTENT.
    bool GpuNeedsInputPixels() const override { return true; }

    // THE WORKING RADII, which come from the descriptor rather than the pixels.
    //
    // Here rather than in MeasureForGpu because that is skipped on a cache
    // hit, and these must track their sliders every frame -- they are what
    // dragging `patch` and `refine` actually changes. Both are stated relative
    // to a 1000px reference and scaled to the image, exactly as the CPU path
    // does, so a value that works on a thumbnail works on a 45 MP raw.
    // ALREADY IN THIS IMAGE'S OWN PIXELS, so the passes must NOT scale them
    // again.
    //
    // Both radii are fractions of the picture, derived here from the extent of
    // the image actually in hand. On a proxy that extent is already reduced,
    // so the answer is already a proxy-pixel radius. Passing it through
    // GpuScaledRadius as well applied the factor twice: at a 15% preview the
    // patch came out ~6.7x too small, the dark channel saw far less structure
    // than it should, and the preview therefore read MORE haze than the full
    // run and corrected harder. That is why the proxy dehazed more strongly
    // than the settled image.
    void PrepareGpu(const std::vector<ImageDesc>& inputs) override {
        if (inputs.empty()) return;
        const double ref =
            double(std::max(inputs[0].width, inputs[0].height)) / 1000.0;
        m_gpuPatch  = std::max(1, int(std::lround(double(int(m_patch))  * ref)));
        m_gpuRefine = std::max(1, int(std::lround(double(int(m_refine)) * ref)));
    }

    // Stated in the image's own pixels by PrepareGpu above, so the framework
    // must not rescale anything for this algorithm.
    ProxyBehaviour Proxy() const override { return ProxyBehaviour::Exact; }

    void MeasureForGpu(const std::vector<const Image*>& inputs) override {
        m_gpuValid = false;
        if (inputs.empty() || !inputs[0]) return;

        ImageView v = const_cast<Image*>(inputs[0])->MapCpuRead();
        if (!v.data) return;

        const ImageDesc& vd = v.desc;
        const int w = vd.width, h = vd.height;
        if (w <= 0 || h <= 0) return;
        if (vd.format == Format::R32F) return;    // greyscale: nothing to do

        // READ THE SAMPLED PIXELS DIRECTLY, rather than through PixelBuffer.
        //
        // Unpack() converts the WHOLE image to float -- that is the very cost
        // this GPU path exists to avoid, and paying it here to look at 0.3% of
        // the pixels would have made the measurement dominate the algorithm.
        // Measured at 9 MP: unpacking first put the stage at ~190 ms against
        // 4 ms of actual GPU work.
        const float scale = (vd.format == Format::RGBA8) ? 255.0f : 1.0f;
        auto sample = [&](int x, int y, float* out) {
            switch (vd.format) {
                case Format::RGBA8: {
                    const uint8_t* p = v.At<uint8_t>(x, y);
                    out[0] = float(p[0]); out[1] = float(p[1]); out[2] = float(p[2]);
                    return true;
                }
                case Format::RGBA32F: {
                    const float* p = v.At<float>(x, y);
                    out[0] = p[0]; out[1] = p[1]; out[2] = p[2];
                    return true;
                }
                case Format::RGBA16F: {
                    const uint16_t* p = v.At<uint16_t>(x, y);
                    for (int c = 0; c < 3; ++c) out[c] = HalfToFloat(p[c]);
                    return true;
                }
                default: return false;
            }
        };

        // STRIDED, like tonemap_local's measurement and for the same reason:
        // this runs on every slider tick, and a full scan of 45 MP would cost
        // more than the passes it prepares. Aim at a few hundred thousand
        // samples however large the image is.
        //
        // Legitimate because the airlight is a property of the haziest REGION,
        // not of one pixel: a regular sample finds the same region, and the
        // patch minimum below is what makes it robust to which pixel within it
        // happens to be hit.
        const size_t target = 300000;
        const size_t total  = size_t(w) * size_t(h);
        const int step = std::max(1, int(std::lround(
            std::sqrt(double(total) / double(target)))));

        const int sw = std::max(1, (w + step - 1) / step);
        const int sh = std::max(1, (h + step - 1) / step);
        const size_t sn = size_t(sw) * size_t(sh);

        std::vector<float> sr(sn), sg(sn), sb(sn), dc(sn);
        for (int y = 0; y < sh; ++y)
            for (int x = 0; x < sw; ++x) {
                const int xx = std::min(x * step, w - 1);
                const int yy = std::min(y * step, h - 1);
                float p[3];
                if (!sample(xx, yy, p)) return;
                const size_t i = size_t(y) * size_t(sw) + size_t(x);
                sr[i] = p[0] / scale;
                sg[i] = p[1] / scale;
                sb[i] = p[2] / scale;
                dc[i] = std::min(sr[i], std::min(sg[i], sb[i]));
            }

        // The patch minimum, at the sample grid's own scale. Without it the
        // dark channel is a per-pixel minimum and a single dark pixel in a hazy
        // region would keep that region from ever being chosen.
        const double ref = double(std::max(w, h)) / 1000.0;
        const int patch = std::max(1, int(std::lround(
            double(int(m_patch)) * ref)) / step);
        std::vector<float> dark(sn);
        MinFilter(dc, sw, sh, patch, &dark);

        // Top 0.1% of the dark channel, then the brightest among those --
        // exactly the CPU path's rule, on the sample.
        std::vector<float> sorted = dark;
        const size_t k = std::max<size_t>(1, sn / 1000);
        std::nth_element(sorted.begin(), sorted.begin() + long(sn - k), sorted.end());
        const float cut = sorted[sn - k];

        double best = -1.0;
        float A[3] = {0, 0, 0};
        for (size_t i = 0; i < sn; ++i) {
            if (dark[i] < cut) continue;
            const double s = double(sr[i]) + double(sg[i]) + double(sb[i]);
            if (s > best) { best = s; A[0] = sr[i]; A[1] = sg[i]; A[2] = sb[i]; }
        }
        // Bounded above as well as below -- see the CPU path's note. A
        // scene-linear frame puts values over 1.0 in the highlights, the
        // search lands on one, and every sky pixel then recovers negative in
        // red before blue. That is the purple cast on the clouds.
        for (int c = 0; c < 3; ++c) m_gpuA[c] = std::clamp(A[c], 1e-3f, 1.0f);

        // The radii are PrepareGpu's job, not this function's: they must track
        // their sliders on every frame, and this function is skipped whenever
        // the airlight comes from the cache.
        m_gpuScale  = scale;
        m_gpuValid  = true;

        char buf[128];
        std::snprintf(buf, sizeof buf,
                      "airlight %.3f %.3f %.3f (GPU, sampled 1/%d)",
                      m_gpuA[0], m_gpuA[1], m_gpuA[2], step);
        m_note = buf;
    }

    // THE AIRLIGHT IS A PROPERTY OF THE PHOTOGRAPH, not of any slider.
    //
    // strength, floor and sky_protect all consume the airlight; none of them
    // changes it. So dragging any of them re-measures an answer that cannot
    // have moved, and the measurement is the whole remaining cost of the
    // stage -- the five GPU passes are ~3 ms at 9 MP.
    //
    // patch DOES feed the measurement (it is the window the dark channel is
    // minimised over), and so would a change of input. Both are covered: the
    // patch radius is part of the saved blob and checked on restore, and the
    // input identity is the pipeline's cache key.
    bool MeasurementDependsOnlyOnInput() const override { return true; }

    std::vector<uint32_t> SaveMeasurement() const override {
        if (!m_gpuValid) return {};   // nothing worth caching yet
        auto bits = [](float f) {
            uint32_t u; std::memcpy(&u, &f, sizeof(u)); return u;
        };
        // ONLY THE AIRLIGHT, plus the patch radius it was measured with.
        //
        // Deliberately NOT the two working radii. Both are trivial to
        // recompute and both are read by the passes -- m_gpuRefine is the
        // refine pass's window -- so restoring them would freeze those
        // sliders: dragging `refine` would change nothing at all. Caching only
        // what is expensive keeps the live parameters live.
        //
        // The patch radius rides along so a restore can REFUSE when it
        // changed. Without that, dragging `patch` would reuse an airlight
        // measured over a different window -- the exact failure this cache has
        // to be careful about, because a slightly wrong airlight is a
        // plausible-looking picture rather than an obvious error.
        return {bits(m_gpuA[0]), bits(m_gpuA[1]), bits(m_gpuA[2]),
                bits(m_gpuScale), uint32_t(int(m_patch))};
    }

    bool RestoreMeasurement(const std::vector<uint32_t>& b) override {
        if (b.size() != 5) return false;
        // Measured over a different window: not this measurement.
        if (b[4] != uint32_t(int(m_patch))) return false;

        auto flt = [](uint32_t u) {
            float f; std::memcpy(&f, &u, sizeof(f)); return f;
        };
        m_gpuA[0]   = flt(b[0]);
        m_gpuA[1]   = flt(b[1]);
        m_gpuA[2]   = flt(b[2]);
        m_gpuScale  = flt(b[3]);
        m_gpuValid  = true;

        char buf[128];
        std::snprintf(buf, sizeof buf, "airlight %.3f %.3f %.3f (GPU, cached)",
                      m_gpuA[0], m_gpuA[1], m_gpuA[2]);
        m_note = buf;
        return true;
    }

    // Five planes: the normalised dark channel, its patch minimum, the
    // transmission map, and two for the guided filter's coefficients.
    int        GpuScratchCount()  const override { return 5; }
    FormatSpec GpuScratchPlanes() const override { return FormatSpec::R32F; }

    std::vector<GpuPass> GpuPasses() const override {
        return {
            // I/A per pixel, minimised across channels.
            {kNormSrc,   "dehaze.norm",    {-1},     {0}},
            // Patch minimum, separable: horizontal then vertical.
            {kMinH,      "dehaze.minh",    {0},      {1}},
            {kMinV,      "dehaze.minv",    {1},      {2}},
            // t = 1 - omega*dark, refined against the image, sky protected.
            {kRefine,    "dehaze.refine",  {-1, 2},  {3}},
            // J = (I - A)/t + A.
            {kRecover,   "dehaze.recover", {-1, 3},  {-1}},
        };
    }

    std::vector<uint32_t> GpuPassConstants(int pass) const override {
        auto bits = [](float f) {
            uint32_t u; std::memcpy(&u, &f, sizeof(u)); return u;
        };
        const float omega   = std::clamp(float(m_strength), 0.0f, 1.0f);
        const float t0      = std::max(0.01f, float(m_floor));
        const float protect = std::clamp(float(m_skyProtect), 0.0f, 1.0f);

        // The radius each pass uses. Only the two minimum passes and the
        // refine pass care, and they want different ones.
        int radius = 0;
        if (pass == 1 || pass == 2) radius = m_gpuPatch;
        else if (pass == 3)         radius = m_gpuRefine;

        return {bits(m_gpuA[0]), bits(m_gpuA[1]), bits(m_gpuA[2]),
                bits(omega), bits(t0), bits(protect),
                uint32_t(std::max(0, radius))};
    }

private:
    // Each pass declares the same constant buffer. Repeated rather than
    // shared, because the framework compiles each pass's HLSL on its own and a
    // pass that omitted a field would silently misread the ones after it.
    //
    // min over channels of I/A -- the dark channel's per-pixel part.
    static constexpr const char* kNormSrc = R"(
Texture2D<float4>   Src : register(t0);
RWTexture2D<float>  Dst : register(u0);

cbuffer Constants : register(b0) {
    uint Width; uint Height;
    uint A0Bits; uint A1Bits; uint A2Bits;
    uint OmegaBits; uint FloorBits; uint ProtectBits;
    uint Radius;
};

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    float3 A = float3(asfloat(A0Bits), asfloat(A1Bits), asfloat(A2Bits));
    float3 c = Src[int2(tid.xy)].rgb / max(A, 1e-3);
    Dst[tid.xy] = min(c.r, min(c.g, c.b));
}
)";

    // Separable patch minimum. Two passes rather than a 2D window: the same
    // answer for 2r+1 reads instead of (2r+1)^2.
    //
    // A straight loop rather than the CPU path's van Herk trick, which is a
    // sequential scan and does not map onto independent threads. At the radii
    // this uses the GPU's parallelism more than covers the difference.
    static constexpr const char* kMinH = R"(
Texture2D<float>    Src : register(t0);
RWTexture2D<float>  Dst : register(u0);
cbuffer Constants : register(b0) {
    uint Width; uint Height;
    uint A0Bits; uint A1Bits; uint A2Bits;
    uint OmegaBits; uint FloorBits; uint ProtectBits;
    uint Radius;
};

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    int r = int(Radius);
    float m = 1e30;
    for (int i = -r; i <= r; ++i) {
        int x = clamp(int(tid.x) + i, 0, int(Width) - 1);
        m = min(m, Src[int2(x, tid.y)]);
    }
    Dst[tid.xy] = m;
}
)";

    static constexpr const char* kMinV = R"(
Texture2D<float>    Src : register(t0);
RWTexture2D<float>  Dst : register(u0);
cbuffer Constants : register(b0) {
    uint Width; uint Height;
    uint A0Bits; uint A1Bits; uint A2Bits;
    uint OmegaBits; uint FloorBits; uint ProtectBits;
    uint Radius;
};

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    int r = int(Radius);
    float m = 1e30;
    for (int i = -r; i <= r; ++i) {
        int y = clamp(int(tid.y) + i, 0, int(Height) - 1);
        m = min(m, Src[int2(tid.x, y)]);
    }
    Dst[tid.xy] = m;
}
)";

    // t = 1 - omega*dark, smoothed against the image, with the sky floor.
    //
    // A BOX SMOOTH of the transmission rather than the CPU path's guided
    // filter. The guided filter needs four box sums and two more passes over
    // coefficient planes; a box smooth of the same radius keeps the map from
    // being blocky, which is what the refinement is for, at a fraction of the
    // dispatch count. The cost is that edges follow the window rather than the
    // picture, so a hard depth edge keeps a little of the halo the guided
    // filter removes -- visible on a branch against bright sky and nowhere
    // else. Worth it here because this path exists to be dragged.
    static constexpr const char* kRefine = R"(
Texture2D<float4>   Img  : register(t0);
Texture2D<float>    Dark : register(t1);
RWTexture2D<float>  Dst  : register(u0);
cbuffer Constants : register(b0) {
    uint Width; uint Height;
    uint A0Bits; uint A1Bits; uint A2Bits;
    uint OmegaBits; uint FloorBits; uint ProtectBits;
    uint Radius;
};

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;

    float3 A = float3(asfloat(A0Bits), asfloat(A1Bits), asfloat(A2Bits));
    float omega   = asfloat(OmegaBits);
    float protect = asfloat(ProtectBits);

    // Box average of the dark channel, subsampled so a large radius does not
    // cost a large loop: the map is smooth, so every fourth sample describes
    // it as well as every sample.
    int r = int(Radius);
    int stepp = max(1, r / 4);
    float sum = 0.0; float cnt = 0.0;
    for (int dy = -r; dy <= r; dy += stepp)
        for (int dx = -r; dx <= r; dx += stepp) {
            int x = clamp(int(tid.x) + dx, 0, int(Width) - 1);
            int y = clamp(int(tid.y) + dy, 0, int(Height) - 1);
            sum += Dark[int2(x, y)];
            cnt += 1.0;
        }
    float dark = (cnt > 0.0) ? sum / cnt : Dark[int2(tid.xy)];

    float t = 1.0 - omega * dark;

    // Sky protection: where the pixel is already close to the airlight
    // COLOUR, raise its floor. Measured on colour distance so a white
    // building against blue sky is left alone.
    if (protect > 0.0) {
        float3 c = Img[int2(tid.xy)].rgb;
        float d = length(c - A);
        float near_ = saturate(1.0 - d / 0.15);
        t = max(t, near_ * protect);
    }
    Dst[tid.xy] = t;
}
)";

    // J = (I - A)/max(t, t0) + A, at full resolution.
    static constexpr const char* kRecover = R"(
Texture2D<float4>   Src : register(t0);
Texture2D<float>    T   : register(t1);
RWTexture2D<float4> Dst : register(u0);
cbuffer Constants : register(b0) {
    uint Width; uint Height;
    uint A0Bits; uint A1Bits; uint A2Bits;
    uint OmegaBits; uint FloorBits; uint ProtectBits;
    uint Radius;
};

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    float3 A  = float3(asfloat(A0Bits), asfloat(A1Bits), asfloat(A2Bits));
    float  t0 = asfloat(FloorBits);
    float4 c  = Src[int2(tid.xy)];
    float  t  = max(T[int2(tid.xy)], t0);
    float3 j  = (c.rgb - A) / t + A;

    // CLAMPED WITHOUT SHIFTING THE HUE.
    //
    // Clamping each channel at zero on its own is what turns an over-strong
    // correction into a COLOUR change: a pixel below the airlight recovers
    // negative fastest in the channel where the gap is widest, so red hits the
    // clamp while blue is still positive and a grey cloud comes out purple.
    // That is a far worse artefact than the darkening it is standing in for,
    // because it does not look like too much dehaze -- it looks broken.
    //
    // So when any channel goes negative, scale the whole pixel back toward
    // black along its own colour instead. The result darkens, which is what
    // the model is actually saying, and keeps its hue while doing it.
    float m = min(j.r, min(j.g, j.b));
    if (m < 0.0) {
        // How far along the line from the original colour to the recovered one
        // we can travel before any channel crosses zero.
        float3 d = j - c.rgb;
        float k = 1.0;
        [unroll] for (int i = 0; i < 3; ++i) {
            if (d[i] < 0.0 && j[i] < 0.0) k = min(k, c.rgb[i] / max(-d[i], 1e-6));
        }
        j = c.rgb + d * saturate(k);
    }
    // The top is left unclamped so a scene-linear image keeps its headroom.
    Dst[tid.xy] = float4(max(j, 0.0), c.a);
}
)";


    // Running minimum over one line, in O(1) per pixel REGARDLESS OF RADIUS
    // (van Herk 1992 / Gil-Werman 1993).
    //
    // The idea: cut the line into blocks of one window width. Build a running
    // minimum forward within each block (prefix) and backward within each block
    // (suffix). Any full window of that width straddles exactly one block
    // boundary, so its minimum is min(suffix at its left end, prefix at its
    // right end) -- two reads, whatever the radius.
    //
    // This matters here more than anywhere else in the codebase because the
    // radius SCALES WITH THE IMAGE: on a 45 MP frame the patch is ~98 px, so a
    // window is 197 samples and the straightforward version does 197 reads per
    // pixel per axis. Measured at that size: 20.5 s against 1.3 s, a 16x
    // difference, bit-for-bit identical output.
    //
    // The line is PADDED by r on each side, replicating the edge sample. The
    // prefix/suffix pair only answers a FULL window; a clamped window at an edge
    // is shorter and can straddle two block boundaries, which it cannot express.
    // The first version handled edges as a special case and got 433k of 45M
    // pixels wrong. Padding makes every window full width, so the special case
    // disappears rather than needing to be right.
    static void MinLine(const float* in, int n, int stride, int r, float* out,
                        std::vector<float>* scratch) {
        const int win = 2 * r + 1;
        const int m = n + 2 * r;
        scratch->assign(size_t(m) * 3, 0.0f);
        float* buf = scratch->data();
        float* pre = buf + m;
        float* suf = pre + m;

        for (int i = 0; i < m; ++i)
            buf[i] = in[size_t(std::clamp(i - r, 0, n - 1)) * size_t(stride)];

        for (int i = 0; i < m; ++i)
            pre[i] = (i % win == 0) ? buf[i] : std::min(pre[i - 1], buf[i]);
        for (int i = m - 1; i >= 0; --i)
            suf[i] = (i == m - 1 || (i + 1) % win == 0)
                         ? buf[i] : std::min(suf[i + 1], buf[i]);

        for (int i = 0; i < n; ++i) {
            const int c = i + r;   // centre, in padded coordinates
            out[size_t(i) * size_t(stride)] = std::min(suf[c - r], pre[c + r]);
        }
    }

    // Minimum over a square window, separably: min over a square IS
    // min-of-rows then min-of-columns.
    static void MinFilter(const std::vector<float>& in, int w, int h, int r,
                          std::vector<float>* out) {
        std::vector<float> mid(in.size());
        std::vector<float> scratch;
        for (int y = 0; y < h; ++y)
            MinLine(&in[size_t(y) * size_t(w)], w, 1, r,
                    &mid[size_t(y) * size_t(w)], &scratch);

        out->assign(in.size(), 0.0f);
        // Column pass walks with a stride of w. Cache-hostile, but the same is
        // true of the version it replaces, and it is no longer the bottleneck.
        for (int x = 0; x < w; ++x)
            MinLine(&mid[size_t(x)], h, w, r, &(*out)[size_t(x)], &scratch);
    }

    Param<float> m_strength{this, "strength", 0.0f, 0.0f, 1.0f,
        {.help = "How much of the estimated haze to remove. 0 is off. Values "
                 "near 1 remove all of it, which usually looks wrong -- distant "
                 "objects lose the aerial perspective that tells the eye they "
                 "are distant, and the picture goes flat. 0.7-0.9 is the useful "
                 "range; He's paper uses 0.95 on deliberately hazy scenes.",
         .step = 0.01f}};

    Param<int> m_patch{this, "patch", 12, 1, 60,
        {.help = "Radius of the window the dark channel is minimised over, per "
                 "1000 px of image -- SCALED to the picture, so one value works "
                 "on a thumbnail and a 45 MP raw. Larger makes the prior more "
                 "reliable, because a bigger patch is likelier to contain "
                 "something genuinely dark, but coarser: thin bright structures "
                 "against sky start being read as haze.",
         .step = 1}};

    Param<int> m_refine{this, "refine", 20, 1, 200,
        {.help = "Radius of the guided filter that snaps the transmission map "
                 "to the image's edges, per 1000 px. The raw map is blocky -- "
                 "every pixel in a patch shares one value -- so unrefined it "
                 "puts halos wherever depth changes. Too large is the opposite "
                 "failure and is easy to miss: it smooths ACROSS depth, so "
                 "distant haze stops being corrected more than near haze. "
                 "Several times the patch radius.",
         .step = 1, .softMax = 60}};

    Param<float> m_floor{this, "floor", 0.1f, 0.01f, 0.5f,
        {.help = "Smallest transmission the recovery will divide by. Where the "
                 "model says almost nothing of the scene survived, recovering "
                 "it amplifies whatever is left -- including the noise -- "
                 "without bound. Raising this keeps deep haze soft instead of "
                 "noisy; lowering it digs harder into the distance.",
         .step = 0.01f}};

    Param<float> m_skyProtect{this, "sky_protect", 0.8f, 0.0f, 1.0f,
        {.help = "Floors the transmission where a pixel is already close to the "
                 "airlight COLOUR. The dark channel prior fails on large bright "
                 "regions that contain no dark pixel -- sky, snow, a white wall "
                 "-- and reads them as haze, so they come out darkened and "
                 "often grey. Set to 0 to see the raw prior.",
         .step = 0.05f}};

    PixelBuffer m_in, m_out;
    std::string m_note;

    // Measured in MeasureForGpu and handed to the passes as root constants.
    // Separate from the CPU path's locals because the GPU path never runs
    // RunCPU, so there is nothing on the stack to carry them.
    bool  m_gpuValid  = false;
    float m_gpuA[3]   = {0.5f, 0.5f, 0.5f};
    int   m_gpuPatch  = 12;
    int   m_gpuRefine = 20;
    float m_gpuScale  = 1.0f;
};

REGISTER_ALGORITHM(Dehaze);

}  // namespace
}  // namespace tglab
