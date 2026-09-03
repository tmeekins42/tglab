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
            for (int c = 0; c < 3; ++c) A[c] = std::max(A[c], 1e-3f);
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
                for (int c = 0; c < 3; ++c) {
                    const float in = p[c] / scale;
                    const float out = (in - A[c]) / tt + A[c];
                    // Negatives are physically meaningless and clip badly
                    // downstream; the top is left unclamped so a scene-linear
                    // image keeps its headroom.
                    q[c] = std::max(out, 0.0f) * scale;
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

    bool HasGPU() const override { return false; }

private:
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
};

REGISTER_ALGORITHM(Dehaze);

}  // namespace
}  // namespace tglab
