// merge_hdr: combines an exposure bracket into one scene-linear image.
//
// The physically correct merge, not an average. Each frame records
//
//     value = radiance * exposure
//
// so dividing by that frame's own exposure recovers radiance, and every frame
// then agrees about the same scene up to noise and clipping. What differs is
// WHICH frames are trustworthy where: a dark frame resolves the highlights and
// buries the shadows in read noise, a bright one does the reverse. So each
// sample is weighted by how much it can be trusted, and the estimate is
//
//     radiance = sum(w_i * v_i / e_i) / sum(w_i)
//
// The result carries real headroom -- a bracket spanning six stops produces
// values well above 1.0 -- which is the point of shooting one.
//
// This is Debevec-style merging without the response-curve solve: a raw sensor
// is linear by construction, so there is no curve to recover. That step exists
// in the literature because the input there is usually JPEG.
//
// WHAT CHANGED, AND WHY (each measured on Tim's hdrtest1/hdrtest2 brackets):
//
//  * ONE WEIGHT PER PIXEL, NOT PER CHANNEL. The old weight was a hat function
//    of each channel on its own, so where red neared clipping and blue did not,
//    red came from the darker frames and blue from the brighter -- a pixel
//    assembled from different exposures, which is the pink and cyan that showed
//    in hdrtest2's clouds. Now every channel of a sample shares one weight,
//    taken from its BRIGHTEST channel, so a merged pixel is always a blend of
//    whole pixels and the hue cannot drift.
//
//  * THE MERGE HAPPENS BEFORE THE CAMERA MATRIX. A demosaiced raw arrives as
//    sRGB primaries: white balanced, clamped at the brightest neutral the
//    sensor can describe (clip_repair.h), then rotated by the camera matrix.
//    The clamp is what says a sample is clipped, and it is only visible in
//    balanced camera units -- after the matrix a clamped pixel is just a
//    colour. So each frame is rotated BACK by the matrix's inverse, merged
//    there with clipping read directly against the clamp, and the matrix and
//    its in-gamut solve are applied once to the merged result. For the 99%+ of
//    pixels the solve never touches, the round trip is exact.
//
//    What this does NOT recover: the demosaic's clamp already discarded what a
//    channel held above the neutral ceiling, so the darkest frame's top stop
//    of colour is gone before the merge sees it. Moving the merge in front of
//    the demosaic's colour step would keep it; that needs the demosaic to hand
//    over unbalanced camera RGB, and is a separate change.
//
//  * THE WEIGHT IS NOISE-OPTIMAL, NOT A HAT. Shot noise makes a sample's
//    variance, in radiance units, proportional to 1/exposure -- so the longest
//    exposure that is not clipped is the best estimate, and the weight is the
//    exposure itself. That replaces the hat's arbitrary dark-end falloff. The
//    bright end rolls off smoothly from 80% to 96% of the clip level, measured
//    on the 3x3 neighbourhood's brightest channel: the demosaic interpolates
//    across a clipped neighbour, so a pixel next to one is already suspect.
//
//  * EXPOSURE RATIOS ARE MEASURED. EXIF shutter speeds are nominal -- a
//    "1/250" is whatever the shutter actually delivered -- and a ratio that is
//    off shows as a brightness step where one frame takes over from the next.
//    The ratio between each pair of neighbouring exposures is measured from
//    the pixels both expose well, and the chain is anchored on the middle
//    frame's EXIF so the overall level matches what it was.
//
//  * DEGHOSTING. Anything that moved between frames -- clouds, leaves, water --
//    is merged from frames that disagree about it, which is a ghost. At each
//    pixel every frame is compared with an anchor -- the middle frame of the
//    bracket wherever it exposes the pixel usably -- and left out where they
//    disagree by more than ghost_threshold. And where the middle frame is
//    clipped but a darker frame shows something much darker, that is a
//    highlight that MOVED: the darker frames are left out too, rather than
//    pasting their sky into the cloud. Measured: hdrtest2's drifting clouds
//    lose their double edges, hdrtest1's wind-blown sage its overlaid
//    branches.
//
//  * SAMPLES OUTSIDE A FRAME ARE NOT DATA. An aligned frame sampled past its
//    own edge used to return the edge pixel, smeared; now it contributes
//    nothing there, and the warp samples Lanczos-3 rather than bilinear.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "../../algo_util/pixel_buffer.h"
#include "../../algo_util/transform.h"
#include "../../core/algorithm.h"
#include "../../core/parallel.h"
#include "../demosaic/clip_repair.h"

namespace tglab {
namespace {

// A half float NaN: "this frame has no sample here". Tested by exponent, so
// any NaN the conversion might produce counts too.
constexpr uint16_t kNoSample = 0x7E00;

// Deghosting holds every frame's sample of a pixel on the stack.
constexpr int kMaxDeghost = 32;
inline bool IsNoSample(uint16_t h) { return (h & 0x7C00) == 0x7C00; }

inline float SrgbToLinear(float v) {
    return v <= 0.04045f ? v * (1.0f / 12.92f)
                         : std::pow((v + 0.055f) * (1.0f / 1.055f), 2.4f);
}

inline float Smoothstep(float a, float b, float x) {
    const float t = std::clamp((x - a) / (b - a), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// FILLS WHAT NO FRAME SAW, from what surrounds it.
//
// Where the reference is clipped and every darker frame saw something dimmer
// than the clip -- a cloud that drifted over a patch of sky between frames --
// the scene's value there is unknown, only bounded below by the clip. The
// merged neighbours that WERE measured (clipped in the reference too, so the
// same bright material) are spread in by a pull-push pyramid at half
// resolution, and each pixel goes over to that by `amount`, never below the
// level it already had. Leaving the clipped reference's flat value there
// instead made grey blotches in a bright cloud; pasting in the darker frames
// made holes of sky.
void FillUnseen(std::vector<float>& px, int ch, int w, int h,
                const std::vector<float>& amount, const std::vector<uint8_t>& source) {
    const int w2 = std::max(1, (w + 1) / 2), h2 = std::max(1, (h + 1) / 2);
    struct Lv { int w, h; std::vector<float> c; };   // r, g, b, weight
    std::vector<Lv> lv(1);
    lv[0].w = w2; lv[0].h = h2;
    lv[0].c.assign(size_t(w2) * size_t(h2) * 4, 0.0f);
    ParallelFor(size_t(h2), [&](size_t cy) {
        for (int cx = 0; cx < w2; ++cx) {
            float* o = &lv[0].c[(cy * size_t(w2) + size_t(cx)) * 4];
            int k = 0;
            for (int dy = 0; dy < 2; ++dy)
                for (int dx = 0; dx < 2; ++dx) {
                    const int x = 2 * cx + dx, y = 2 * int(cy) + dy;
                    if (x >= w || y >= h) continue;
                    ++k;
                    const size_t i = size_t(y) * size_t(w) + size_t(x);
                    if (!source[i]) continue;
                    for (int c = 0; c < 3; ++c) o[c] += px[i * size_t(ch) + size_t(c)];
                    o[3] += 1.0f;
                }
            if (k) for (int c = 0; c < 4; ++c) o[c] /= float(k);
        }
    });
    while (lv.back().w > 1 || lv.back().h > 1) {
        const Lv& f = lv.back();
        Lv g;
        g.w = std::max(1, (f.w + 1) / 2); g.h = std::max(1, (f.h + 1) / 2);
        g.c.assign(size_t(g.w) * size_t(g.h) * 4, 0.0f);
        for (int y = 0; y < g.h; ++y)
            for (int x = 0; x < g.w; ++x) {
                float* o = &g.c[(size_t(y) * size_t(g.w) + size_t(x)) * 4];
                int k = 0;
                for (int dy = 0; dy < 2; ++dy)
                    for (int dx = 0; dx < 2; ++dx) {
                        const int sx = 2 * x + dx, sy = 2 * y + dy;
                        if (sx >= f.w || sy >= f.h) continue;
                        ++k;
                        const float* q = &f.c[(size_t(sy) * size_t(f.w) + size_t(sx)) * 4];
                        for (int c = 0; c < 4; ++c) o[c] += q[c];
                    }
                for (int c = 0; c < 4; ++c) o[c] /= float(k);
            }
        lv.push_back(std::move(g));
    }
    // Normalise the top, then push down: each level's own (weighted) colour
    // blended with the coarser estimate where its weight is thin.
    {
        float* t = lv.back().c.data();
        if (t[3] > 0.0f) for (int c = 0; c < 3; ++c) t[c] /= t[3];
        t[3] = 1.0f;
    }
    auto sample = [](const Lv& l, int x, int y, float* out) {
        const float fx = std::clamp((float(x) + 0.5f) * 0.5f - 0.5f, 0.0f, float(l.w - 1));
        const float fy = std::clamp((float(y) + 0.5f) * 0.5f - 0.5f, 0.0f, float(l.h - 1));
        const int x0 = int(fx), y0 = int(fy);
        const int x1 = std::min(x0 + 1, l.w - 1), y1 = std::min(y0 + 1, l.h - 1);
        const float tx = fx - float(x0), ty = fy - float(y0);
        for (int c = 0; c < 3; ++c) {
            auto at = [&](int xx, int yy) { return l.c[(size_t(yy) * size_t(l.w) + size_t(xx)) * 4 + size_t(c)]; };
            const float a = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * tx;
            const float b = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * tx;
            out[c] = a + (b - a) * ty;
        }
    };
    constexpr float kPrior = 0.05f;
    for (int li = int(lv.size()) - 2; li >= 0; --li) {
        Lv& f = lv[size_t(li)];
        const Lv& g = lv[size_t(li + 1)];
        ParallelFor(size_t(f.h), [&](size_t y) {
            for (int x = 0; x < f.w; ++x) {
                float* o = &f.c[(y * size_t(f.w) + size_t(x)) * 4];
                float e[3];
                sample(g, x, int(y), e);
                const float d = o[3] + kPrior;
                for (int c = 0; c < 3; ++c) o[c] = (o[c] + kPrior * e[c]) / d;
                o[3] = 1.0f;
            }
        });
    }
    const Lv& l0 = lv[0];
    ParallelFor(size_t(h), [&](size_t y) {
        for (int x = 0; x < w; ++x) {
            const size_t i = y * size_t(w) + size_t(x);
            const float a = amount[i];
            if (a <= 0.0f) continue;
            float e[3];
            sample(l0, x, int(y), e);
            float* q = &px[i * size_t(ch)];
            // Never below the level the pixel already had: the clip is a floor.
            const float have = q[0] + q[1] + q[2], fill = e[0] + e[1] + e[2];
            const float lift = (fill > 1e-9f && fill < have) ? have / fill : 1.0f;
            for (int c = 0; c < 3; ++c) q[c] += (e[c] * lift - q[c]) * a;
        }
    });
}

// 3x3 inverse. False when the matrix is (near) singular.
bool Invert3(const float* m, float* inv) {
    const double a = m[0], b = m[1], c = m[2], d = m[3], e = m[4], f = m[5],
                 g = m[6], h = m[7], k = m[8];
    const double det = a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g);
    if (std::abs(det) < 1e-9) return false;
    const double r = 1.0 / det;
    inv[0] = float((e * k - f * h) * r); inv[1] = float((c * h - b * k) * r); inv[2] = float((b * f - c * e) * r);
    inv[3] = float((f * g - d * k) * r); inv[4] = float((a * k - c * g) * r); inv[5] = float((c * d - a * f) * r);
    inv[6] = float((d * h - e * g) * r); inv[7] = float((b * g - a * h) * r); inv[8] = float((a * e - b * d) * r);
    return true;
}

bool IsIdentity3(const float* m) {
    for (int i = 0; i < 9; ++i)
        if (std::abs(m[i] - ((i % 4) == 0 ? 1.0f : 0.0f)) > 1e-6f) return false;
    return true;
}

class MergeHdr : public AlgorithmBase, public Reducer {
public:
    const char* Name()     const override { return "merge_hdr"; }
    const char* Category() const override { return "merge"; }

    PortList Inputs() const override {
        return {{"src", DataType::ImageSet, FormatSpec::Any, ShapeSpec::Any}};
    }
    PortList Outputs() const override {
        // RGBA32F rather than the input format. The whole point is a result
        // with more range than any single frame, and half float would clip the
        // top of a wide bracket straight back off.
        return {{"out", DataType::Image, FormatSpec::RGBA32F, ShapeSpec::Reduced}};
    }

    void RunCPU(RunCtx&) override {}

    bool     IsReduction() const override { return true; }
    Reducer* AsReducer()         override { return this; }

    // Accepting a frame is the warp and the clip map; the merge itself, the
    // exposure measurement and the deghost all run in Finish.
    double AcceptShare() const override { return 0.5; }

    bool Begin(int count, const std::string&, std::string* err) override {
        if (count <= 0) { *err = "merge_hdr: nothing to merge"; return false; }
        m_frames.clear();
        m_desc = ImageDesc{};
        m_noExposure  = false;
        m_unexposed   = 0;
        m_warped      = 0;
        m_spanStops   = 0.0f;
        m_exifOff     = -1.0f;
        m_unmeasured  = 0;
        m_inconsistent = 0;
        m_ghostPct    = -1.0f;
        m_movedPct    = 0.0f;
        m_noSamplePct = 0.0f;
        m_reported    = 0;
        return true;
    }

    bool Accept(int, const Image& img, std::string* err) override {
        ImageView v = const_cast<Image&>(img).MapCpuRead();
        if (!v.data) { *err = "merge_hdr: a frame has no pixels"; return false; }

        const ImageDesc& d = img.Desc();
        if (!m_frames.empty() && (d.width != m_desc.width || d.height != m_desc.height)) {
            *err = "merge_hdr: every frame must be the same size (frame " +
                   std::to_string(m_frames.size()) + " is " + std::to_string(d.width) + "x" +
                   std::to_string(d.height) + ", expected " + std::to_string(m_desc.width) +
                   "x" + std::to_string(m_desc.height) + ")";
            return false;
        }

        PixelBuffer buf;
        buf.Unpack(v);
        if (!buf.Valid()) { *err = "merge_hdr: unsupported format"; return false; }
        const int w = buf.Width(), h = buf.Height(), ch = buf.Channels();
        const int C = ch >= 3 ? 3 : 1;

        if (m_frames.empty()) {
            m_desc = d;
            m_C    = C;
            // The camera matrix, and whether there is one to undo. A JPEG or
            // a non-raw TIFF carries identity, and skips both rotations.
            m_hasMatrix = C == 3 && !IsIdentity3(d.rgbCam) && Invert3(d.rgbCam, m_camInv);
        }
        if (C != m_C) { *err = "merge_hdr: frames differ in channel count"; return false; }

        Frame f;
        f.exif = d.RelativeExposure();
        if (f.exif <= 0.0f) { f.exif = 0.0f; m_noExposure = true; }

        // The level at which this frame's samples stop meaning anything. A
        // demosaiced raw is clamped at the brightest balanced neutral its
        // sensor can describe, which is the smallest white-balance gain (see
        // clip_repair.h); anything else clips at 1.0.
        f.ceiling = 1.0f;
        if (m_hasMatrix) {
            const float c = std::min(d.camMul[0], std::min(d.camMul[1], d.camMul[2]));
            if (c > 0.0f) f.ceiling = c;
        }

        // Into the space the merge runs in: linear, balanced camera RGB.
        std::vector<float>& px = buf.Data();
        const bool decode = !d.linear;
        ParallelFor(size_t(h), [&](size_t y) {
            for (int x = 0; x < w; ++x) {
                float* p = &px[(y * size_t(w) + size_t(x)) * size_t(ch)];
                if (decode)
                    for (int c = 0; c < C; ++c) p[c] = SrgbToLinear(p[c]);
                if (m_hasMatrix) {
                    const float* m = m_camInv;
                    const float r = p[0], g = p[1], b = p[2];
                    p[0] = m[0] * r + m[1] * g + m[2] * b;
                    p[1] = m[3] * r + m[4] * g + m[5] * b;
                    p[2] = m[6] * r + m[7] * g + m[8] * b;
                }
            }
        });

        // The transform an aligner attached, or identity when none did.
        //
        // This is the whole point of the sidecar mechanism: merge_hdr reads a
        // transform it did not ask for and may not find. Applied DIRECTLY, not
        // inverted: T maps a reference coordinate to the position in this
        // frame, which is precisely the question a loop over the reference grid
        // asks. See transform.h -- inverting it here negated the correction and
        // made an aligned merge blurrier than an unaligned one.
        const Affine warp = TransformOf(img);
        const bool warped = !warp.IsIdentity();
        if (warped) ++m_warped;

        const size_t N = size_t(w) * size_t(h);
        f.px.assign(N * size_t(C), kNoSample);
        ParallelFor(size_t(h), [&](size_t y) {
            float sm[4] = {0, 0, 0, 0};
            for (int x = 0; x < w; ++x) {
                const size_t o = (y * size_t(w) + size_t(x)) * size_t(C);
                const float* s;
                if (!warped) {
                    s = buf.At(x, int(y));
                } else {
                    float sx, sy;
                    warp.MapPoint(float(x), float(y), &sx, &sy);
                    // Past the frame's own edge there is no sample, and the
                    // edge pixel smeared outward is not one.
                    if (!(sx >= -0.5f && sy >= -0.5f &&
                          sx <= float(w) - 0.5f && sy <= float(h) - 0.5f))
                        continue;
                    SampleLanczos3(buf, sx, sy, sm);
                    s = sm;
                }
                for (int c = 0; c < C; ++c) f.px[o + size_t(c)] = FloatToHalf(s[c]);
            }
        });

        // The clip map: how far each sample is from clipping, by its BRIGHTEST
        // channel over the 3x3 neighbourhood, as a weight in 0..255.
        {
            std::vector<float> mx(N, -1.0f), tmp(N, -1.0f);
            ParallelFor(size_t(h), [&](size_t y) {
                for (int x = 0; x < w; ++x) {
                    const size_t i = y * size_t(w) + size_t(x);
                    if (IsNoSample(f.px[i * size_t(C)])) continue;
                    float m = -1.0f;
                    for (int c = 0; c < C; ++c)
                        m = std::max(m, HalfToFloat(f.px[i * size_t(C) + size_t(c)]));
                    mx[i] = m;
                }
            });
            ParallelFor(size_t(h), [&](size_t y) {
                for (int x = 0; x < w; ++x) {
                    float m = -1.0f;
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int xx = std::clamp(x + dx, 0, w - 1);
                        m = std::max(m, mx[y * size_t(w) + size_t(xx)]);
                    }
                    tmp[y * size_t(w) + size_t(x)] = m;
                }
            });
            f.wclip.assign(N, 0);
            const float lo = 0.80f * f.ceiling, hi = 0.96f * f.ceiling;
            ParallelFor(size_t(h), [&](size_t y) {
                for (int x = 0; x < w; ++x) {
                    const size_t i = y * size_t(w) + size_t(x);
                    if (mx[i] < -0.5f) continue;            // no sample
                    float m = -1.0f;
                    for (int dy = -1; dy <= 1; ++dy) {
                        const int yy = std::clamp(int(y) + dy, 0, h - 1);
                        m = std::max(m, tmp[size_t(yy) * size_t(w) + size_t(x)]);
                    }
                    const float wt = 1.0f - Smoothstep(lo, hi, m);
                    f.wclip[i] = uint8_t(std::lround(wt * 255.0f));
                }
            });
        }

        // A half-resolution, 3x3-blurred plane of summed channels: what the
        // deghost compares. Single samples were too literal -- on hdrtest1 a
        // sub-pixel misregistration and the demosaic's edge handling made 58%
        // of pixels "disagree" along gravel and wood grain, none of it motion.
        {
            const int w2 = (w + 1) / 2, h2 = (h + 1) / 2;
            std::vector<float> lo(size_t(w2) * size_t(h2), -1.0f);
            ParallelFor(size_t(h2), [&](size_t y2) {
                for (int x2 = 0; x2 < w2; ++x2) {
                    float s = 0.0f; int m = 0;
                    for (int dy = 0; dy < 2; ++dy)
                        for (int dx = 0; dx < 2; ++dx) {
                            const int x = 2 * x2 + dx, y = 2 * int(y2) + dy;
                            if (x >= w || y >= h) continue;
                            const size_t i = size_t(y) * size_t(w) + size_t(x);
                            if (IsNoSample(f.px[i * size_t(C)])) continue;
                            for (int c = 0; c < C; ++c) s += HalfToFloat(f.px[i * size_t(C) + size_t(c)]);
                            ++m;
                        }
                    if (m) lo[y2 * size_t(w2) + size_t(x2)] = s / float(m);
                }
            });
            f.local.assign(lo.size(), -1.0f);
            ParallelFor(size_t(h2), [&](size_t y2) {
                for (int x2 = 0; x2 < w2; ++x2) {
                    float s = 0.0f; int m = 0;
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx) {
                            const int xx = x2 + dx, yy = int(y2) + dy;
                            if (xx < 0 || yy < 0 || xx >= w2 || yy >= h2) continue;
                            const float v = lo[size_t(yy) * size_t(w2) + size_t(xx)];
                            if (v < 0.0f) continue;
                            s += v; ++m;
                        }
                    if (m) f.local[y2 * size_t(w2) + size_t(x2)] = s / float(m);
                }
            });
        }

        // Mean brightness, for ordering a bracket that has no EXIF.
        {
            double s = 0.0; size_t n = 0;
            const size_t step = std::max<size_t>(1, N / 100000);
            for (size_t i = 0; i < N; i += step) {
                const uint16_t hv = f.px[i * size_t(C) + (C == 3 ? 1 : 0)];
                if (IsNoSample(hv)) continue;
                s += HalfToFloat(hv); ++n;
            }
            f.meanLevel = n ? float(s / double(n)) : 0.0f;
        }

        m_frames.push_back(std::move(f));
        return true;
    }

    bool Finish(Image* out, std::string* err) override {
        const int n = int(m_frames.size());
        if (n == 0) { *err = "merge_hdr: no frames were accepted"; return false; }
        const int w = m_desc.width, h = m_desc.height, C = m_C;
        const size_t N = size_t(w) * size_t(h);

        // ORDER by exposure, darkest first: EXIF when every frame has it, the
        // measured brightness otherwise.
        std::vector<int> order(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) order[size_t(i)] = i;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            const Frame& fa = m_frames[size_t(a)];
            const Frame& fb = m_frames[size_t(b)];
            if (!m_noExposure) return fa.exif < fb.exif;
            return fa.meanLevel < fb.meanLevel;
        });
        const int refPos = n / 2;
        const int ref    = order[size_t(refPos)];

        // MEASURE each neighbouring pair's exposure ratio: the median of
        // bright/dark over samples both expose well -- above 5% of the dark
        // frame's clip level, where a black-level error no longer biases the
        // ratio, and below the bright frame's roll-off.
        //
        // TRUSTED ONLY WHEN IT IS ONE NUMBER. A true exposure difference scales
        // every pixel alike, so the log-ratios cluster tightly: on hdrtest1
        // their interquartile range is 0.03-0.05 stops. On hdrtest2 the light
        // changed between two shots -- cloud shadows crossing the mountains --
        // and the two pairs touching that frame spread 0.36-0.51 stops, with a
        // median that moved by a quarter stop depending on which pixels were
        // admitted. No single ratio is right for a frame like that, so EXIF is
        // kept and the deghost deals with the regions that differ.
        GroupProgress(0.5, "measuring exposures");
        constexpr double kMaxSpreadStops = 0.25;
        std::vector<double> ratio(size_t(std::max(n - 1, 0)), 1.0);
        for (int k = 0; k + 1 < n; ++k) {
            const Frame& a = m_frames[size_t(order[size_t(k)])];
            const Frame& b = m_frames[size_t(order[size_t(k + 1)])];
            const double exifRatio = (a.exif > 0.0f && b.exif > 0.0f) ? double(b.exif) / double(a.exif) : 0.0;
            double measured = 0.0;
            bool spread = false;
            if (m_measure) {
                std::vector<float> logs;
                const size_t step = std::max<size_t>(1, N / 400000);
                const int gc = C == 3 ? 1 : 0;
                for (size_t i = 0; i < N; i += step) {
                    if (a.wclip[i] < 255 || b.wclip[i] < 255) continue;
                    const uint16_t ha = a.px[i * size_t(C) + size_t(gc)];
                    const uint16_t hb = b.px[i * size_t(C) + size_t(gc)];
                    if (IsNoSample(ha) || IsNoSample(hb)) continue;
                    const float va = HalfToFloat(ha), vb = HalfToFloat(hb);
                    if (va < 0.05f * a.ceiling || vb <= 0.0f) continue;
                    logs.push_back(std::log(vb / va));
                }
                if (logs.size() >= 500) {
                    std::sort(logs.begin(), logs.end());
                    const double iqr = double(logs[logs.size() * 3 / 4] - logs[logs.size() / 4]) / 0.6931472;
                    if (iqr <= kMaxSpreadStops || exifRatio <= 0.0)
                        measured = std::exp(double(logs[logs.size() / 2]));
                    else
                        spread = true;
                }
            }
            if (measured > 0.0) {
                ratio[size_t(k)] = measured;
                if (exifRatio > 0.0)
                    m_exifOff = std::max(m_exifOff, float(std::abs(std::log2(measured / exifRatio))));
            } else {
                ratio[size_t(k)] = exifRatio > 0.0 ? exifRatio : 1.0;
                if (spread) ++m_inconsistent;
                else if (m_measure) ++m_unmeasured;
            }
        }

        // FACTORS: value -> radiance, anchored on the middle frame's EXIF so
        // the overall level stays where an EXIF-only merge put it.
        std::vector<double> fac(size_t(n), 1.0);
        {
            const Frame& r = m_frames[size_t(ref)];
            fac[size_t(ref)] = r.exif > 0.0f ? 1.0 / double(r.exif) : 1.0;
            for (int k = refPos + 1; k < n; ++k)
                fac[size_t(order[size_t(k)])] = fac[size_t(order[size_t(k - 1)])] / ratio[size_t(k - 1)];
            for (int k = refPos - 1; k >= 0; --k)
                fac[size_t(order[size_t(k)])] = fac[size_t(order[size_t(k + 1)])] * ratio[size_t(k)];
            m_spanStops = float(std::log2(fac[size_t(order.front())] / fac[size_t(order.back())]));
        }
        // Noise weight: exposure, relative to the longest.
        std::vector<float> wn(size_t(n), 1.0f);
        {
            const double fmin = fac[size_t(order.back())];
            for (int i = 0; i < n; ++i) wn[size_t(i)] = float(fmin / fac[size_t(i)]);
        }

        // DEGHOST, per pixel. A frame is left out of a pixel where its
        // radiance disagrees with the ANCHOR's by more than ghost_threshold.
        //
        // The anchor is the reference wherever the reference exposes the pixel
        // usably, which keeps it the same from one pixel to the next. Where it
        // does not, the walk goes toward the exposures that do -- darker when
        // the reference is clipped, brighter when it is in the noise -- and
        // takes the first frame that agrees with the one beyond it.
        //
        // Both halves of that were measured on Tim's brackets. Deciding per 8x8
        // block, with the anchor chosen by majority per block, cut hdrtest1's
        // wind-blown sage into squares from different moments: with every
        // frame of a moving bush different, the majority flips from block to
        // block. And simply taking the next frame along, unchecked, failed in
        // hdrtest2's clouds: where the reference neared clipping, that frame was
        // the one shot first, before the cloud moved, and the four frames that
        // agreed with each other were thrown out for disagreeing with it.
        const bool deghost = m_deghost && n > 1 && n <= kMaxDeghost;
        const int w2 = (w + 1) / 2, h2 = (h + 1) / 2;
        auto localAt = [&](const Frame& f, int x, int y) -> float {
            const float gx = std::clamp((float(x) + 0.5f) * 0.5f - 0.5f, 0.0f, float(w2 - 1));
            const float gy = std::clamp((float(y) + 0.5f) * 0.5f - 0.5f, 0.0f, float(h2 - 1));
            const int x0 = int(gx), y0 = int(gy);
            const int x1 = std::min(x0 + 1, w2 - 1), y1 = std::min(y0 + 1, h2 - 1);
            const float fx = gx - float(x0), fy = gy - float(y0);
            float s = 0.0f, ws = 0.0f;
            const float wts[4] = {(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy};
            const size_t idx[4] = {size_t(y0) * size_t(w2) + size_t(x0), size_t(y0) * size_t(w2) + size_t(x1),
                                   size_t(y1) * size_t(w2) + size_t(x0), size_t(y1) * size_t(w2) + size_t(x1)};
            for (int q = 0; q < 4; ++q) {
                const float v = f.local[idx[q]];
                if (v < 0.0f || wts[q] <= 0.0f) continue;
                s += wts[q] * v; ws += wts[q];
            }
            return ws > 0.0f ? s / ws : -1.0f;
        };
        const float T = float(m_ghostStops) * 0.6931472f;               // stops -> ln

        // MERGE.
        GroupProgress(0.6, "merging");
        ImageDesc od = m_desc;
        od.format = Format::RGBA32F;
        od.linear = true;     // scene-linear radiance, not display encoded
        od.rawClamp = 0.0f;   // radiance runs past the frames' clamp by design
        out->Alloc(od);
        ImageView ov = out->MapCpuWrite();
        if (!ov.data) { *err = "merge_hdr: could not allocate the result"; return false; }

        PixelBuffer ob;
        ob.Unpack(ov);
        std::vector<float>& op = ob.Data();
        const int oc = ob.Channels();

        std::vector<int> unexposedRows(size_t(h), 0), noSampleRows(size_t(h), 0);
        std::vector<int> ghostRows(size_t(h), 0), movedRows(size_t(h), 0);
        std::vector<float> unseenW(deghost ? N : 0, 0.0f);
        std::vector<uint8_t> fillSource(deghost ? N : 0, 0);
        std::atomic<int> rowsDone{0};
        ParallelFor(size_t(h), [&](size_t yy) {
            const int y = int(yy);
            // Per pixel, in exposure order (k = 0 darkest).
            float v[kMaxDeghost][3];
            float rad[kMaxDeghost];       // summed channels, in radiance
            float own[kMaxDeghost];       // brightest channel, as recorded
            float wc[kMaxDeghost];        // clip weight 0..1
            bool  ok[kMaxDeghost];
            bool  keepK[kMaxDeghost];
            float gate[3][kMaxDeghost];   // per anchor: how much each frame agrees
            float low[kMaxDeghost];       // consistent with the reference's clip floor
            const int nk = std::min(n, kMaxDeghost);
            for (int x = 0; x < w; ++x) {
                const size_t i = size_t(y) * size_t(w) + size_t(x);
                double num[3] = {0, 0, 0}, den = 0.0;
                float rgb[3] = {0, 0, 0};

                if (deghost) {
                    int anyValid = -1;
                    for (int k = 0; k < nk; ++k) {
                        const int fi = order[size_t(k)];
                        const Frame& f = m_frames[size_t(fi)];
                        ok[k] = !IsNoSample(f.px[i * size_t(C)]);
                        keepK[k] = true;
                        if (!ok[k]) continue;
                        float mx = -1e30f;
                        for (int c = 0; c < C; ++c) {
                            v[k][c] = HalfToFloat(f.px[i * size_t(C) + size_t(c)]);
                            mx = std::max(mx, v[k][c]);
                        }
                        rad[k] = float(double(localAt(f, x, y)) * fac[size_t(fi)]);
                        own[k] = mx;
                        wc[k]  = float(f.wclip[i]) * (1.0f / 255.0f);
                    }
                    // Above the noise floor: a sample there can be compared.
                    auto floorOk = [&](int k) {
                        const int fi = order[size_t(k)];
                        return ok[k] && rad[k] > 0.0f &&
                               double(rad[k]) >= 0.01 * double(m_frames[size_t(fi)].ceiling) * C * fac[size_t(fi)];
                    };
                    auto usable = [&](int k) { return floorOk(k) && wc[k] >= 0.5f; };
                    auto agree  = [&](int a, int b) {
                        return std::abs(std::log(rad[a] / rad[b])) <= T;
                    };
                    // SOFT, not a cut. How much frame k counts beside anchor a:
                    // fully within half the threshold, not at all past one and
                    // a half. A hard cut at the threshold switched a frame on
                    // and off along every contour where a moving cloud's
                    // frames disagreed by about that much, and the merge
                    // became a patchwork of different moments with an edge at
                    // each switch.
                    auto weigh = [&](int a, float* g) {
                        for (int k = 0; k < nk; ++k) {
                            g[k] = 1.0f;
                            if (k == a || !floorOk(k)) continue;
                            const float d = std::abs(std::log(rad[k] / rad[a]));
                            const float t = std::clamp((d - 0.5f * T) / T, 0.0f, 1.0f);
                            g[k] = 1.0f - t * t * (3.0f - 2.0f * t);
                        }
                    };

                    // TWO ANCHORS, BLENDED. The reference where it exposes the
                    // pixel well; where it does not, the walk below. Choosing
                    // one or the other at a clip weight of 0.5 switched the
                    // anchor along every brightness contour through a cloud
                    // near the reference's clip level -- and the two anchors,
                    // shot at different moments, disagree about a moving cloud.
                    // So across the reference's clip ramp the two merges are
                    // blended by its clip weight.
                    // THE CLIP IS A FLOOR. Where the reference is clipped the
                    // scene was at least its clip level, so a darker frame
                    // reading well below that was shot at another moment --
                    // the sky a cloud later drifted over. Each such frame fades
                    // out (the same soft ramp as the agreement), and none of
                    // them can be the anchor. Only "under half" used to count,
                    // and bright sky passed, pasting holes of blue into cloud.
                    const Frame& rf = m_frames[size_t(ref)];
                    const bool refClip = ok[refPos] && own[refPos] >= 0.96f * rf.ceiling;
                    const double floorRad = 0.96 * double(rf.ceiling) * fac[size_t(ref)];
                    float bestLow = 0.0f;
                    bool anyDarker = false;
                    for (int k = 0; k < nk; ++k) {
                        low[k] = 1.0f;
                        if (!refClip || k >= refPos || !ok[k]) continue;
                        const double got = double(own[k]) * fac[size_t(order[size_t(k)])];
                        if (got > 0.0) {
                            const float d = float(std::log(floorRad / got));
                            const float t = std::clamp((d - 0.5f * T) / T, 0.0f, 1.0f);
                            low[k] = 1.0f - t * t * (3.0f - 2.0f * t);
                        } else {
                            low[k] = 0.0f;
                        }
                        if (floorOk(k)) { anyDarker = true; bestLow = std::max(bestLow, low[k]); }
                    }
                    // How far no darker frame could see this pixel at all.
                    const float unseen = (refClip && anyDarker) ? 1.0f - bestLow : 0.0f;
                    unseenW[i] = unseen;
                    fillSource[i] = (refClip && unseen <= 0.0f) ? 1 : 0;

                    // SUPPORT: how many other frames, each counted by how well it
                    // exposes the pixel, agree with frame a. The anchor should be
                    // the frame the bracket agrees with, not just the first
                    // usable one: on hdrtest2 an extra frame shot moments
                    // earlier, of a thinner cloud, sat next to the reference,
                    // and where the frame beyond it neared its own clip the old
                    // check ("agrees with the next one along") was skipped --
                    // so it anchored unchecked, threw out the four frames that
                    // agreed with each other, and left holes in the cloud.
                    auto support = [&](int a) {
                        float sp = 0.0f;
                        for (int k = 0; k < nk; ++k) {
                            if (k == a || !floorOk(k) || low[k] < 0.5f) continue;
                            const float d = std::abs(std::log(rad[k] / rad[a]));
                            const float t = std::clamp((d - 0.5f * T) / T, 0.0f, 1.0f);
                            sp += std::min(1.0f, 2.0f * wc[k]) * (1.0f - t * t * (3.0f - 2.0f * t));
                        }
                        return sp;
                    };

                    const bool refOk = floorOk(refPos);
                    const float tRef = refOk ? std::clamp((wc[refPos] - 0.3f) / 0.4f, 0.0f, 1.0f) : 0.0f;
                    const float bRef = tRef * tRef * (3.0f - 2.0f * tRef);
                    int alt = -1;
                    bool moved = false;
                    if (bRef < 1.0f && ok[refPos]) {
                        const bool clipped = wc[refPos] < 0.5f;
                        const int step = clipped ? -1 : 1;
                        float best = -1.0f;
                        for (int p = refPos + step; p >= 0 && p < nk; p += step) {
                            if (!usable(p) || low[p] < 0.5f) continue;
                            const float sp = support(p);
                            if (sp > best + 1e-3f) { best = sp; alt = p; }   // ties: the nearer
                        }
                    }
                    // A highlight that MOVED: no darker frame is consistent with
                    // the clip floor. FillUnseen settles its value afterwards.
                    moved = unseen >= 0.5f;
                    // The anchors in play and their shares.
                    int anc[3] = {-1, -1, -1};
                    float share[3] = {0.0f, 0.0f, 0.0f};
                    if (refOk && bRef > 0.0f) { anc[0] = refPos; share[0] = bRef; }
                    if (alt >= 0 && !moved && bRef < 1.0f) { anc[1] = alt; share[1] = 1.0f - bRef; }
                    if (anc[0] < 0 && anc[1] >= 0) share[1] = 1.0f;
                    if (anc[1] < 0 && anc[0] >= 0) share[0] = 1.0f;
                    // THE REFERENCE OUTVOTED. Where at least two more frames
                    // agree with some other frame than with the reference, the
                    // reference is the one that moved, and its share goes over
                    // to that frame -- gradually, between one and two votes, so
                    // the hand-over has no edge. Where every frame differs (a
                    // bush in the wind) no frame has that support and the
                    // reference keeps the pixel, which is what keeps one moment
                    // from being stitched out of many.
                    if (anc[0] >= 0) {
                        const float sr = support(refPos);
                        int cons = -1; float sc = sr;
                        for (int k = 0; k < nk; ++k) {
                            if (k == refPos || !usable(k) || low[k] < 0.5f) continue;
                            const float sk = support(k);
                            if (sk > sc + 1e-3f) { sc = sk; cons = k; }
                        }
                        if (cons >= 0) {
                            const float t = std::clamp(sc - sr - 1.0f, 0.0f, 1.0f);
                            const float over = t * t * (3.0f - 2.0f * t);
                            if (over > 0.0f) {
                                anc[2] = cons;
                                share[2] = share[0] * over;
                                share[0] *= 1.0f - over;
                            }
                        }
                    }

                    bool dropped = moved;
                    for (int j = 0; j < 3; ++j) {
                        if (anc[j] < 0) continue;
                        weigh(anc[j], gate[j]);
                        const int a = anc[j];
                        const float wa = wc[a] * wn[size_t(order[size_t(a)])];
                        for (int k = 0; k < nk; ++k) {
                            if (gate[j][k] >= 0.5f || !floorOk(k)) continue;
                            // Counted only when it would have mattered: a dark
                            // frame's shadow sample carries almost no weight.
                            if (wc[k] * wn[size_t(order[size_t(k)])] >= 0.1f * wa) dropped = true;
                        }
                    }
                    ghostRows[yy] += dropped ? 1 : 0;
                    movedRows[yy] += moved ? 1 : 0;

                    // Each anchor's weighted merge, then the blend.
                    double sum[3] = {0, 0, 0}, shareSum = 0.0;
                    int rescue = -1; float rescueW = 0.0f;
                    for (int k = 0; k < nk; ++k) {
                        if (!ok[k]) continue;
                        if (anyValid < 0) anyValid = k;
                        const float w0 = wc[k] * wn[size_t(order[size_t(k)])];
                        if (w0 > rescueW && low[k] >= 0.5f) { rescueW = w0; rescue = k; }
                    }
                    for (int j = 0; j < 3; ++j) {
                        if (anc[j] < 0 || share[j] <= 0.0f) continue;
                        double nj[3] = {0, 0, 0}, dj = 0.0;
                        for (int k = 0; k < nk; ++k) {
                            if (!ok[k] || !keepK[k]) continue;
                            const int fi = order[size_t(k)];
                            const float w0 = wc[k] * wn[size_t(fi)] * gate[j][k] * low[k];
                            if (w0 <= 0.0f) continue;
                            const double sc = double(w0) * fac[size_t(fi)];
                            for (int c = 0; c < C; ++c) nj[c] += sc * double(v[k][c]);
                            dj += double(w0);
                        }
                        if (dj <= 1e-7) continue;
                        for (int c = 0; c < C; ++c) sum[c] += share[j] * nj[c] / dj;
                        shareSum += share[j];
                    }
                    if (shareSum > 0.0) {
                        for (int c = 0; c < C; ++c) num[c] = sum[c] / shareSum;
                        den = 1.0;
                    } else if (!moved) {
                        // No anchor (nothing exposes the pixel well enough to
                        // judge by): every frame, weighted, as before.
                        for (int k = 0; k < nk; ++k) {
                            if (!ok[k]) continue;
                            const int fi = order[size_t(k)];
                            const float w0 = wc[k] * wn[size_t(fi)] * low[k];
                            if (w0 <= 0.0f) continue;
                            const double sc = double(w0) * fac[size_t(fi)];
                            for (int c = 0; c < C; ++c) num[c] += sc * double(v[k][c]);
                            den += double(w0);
                        }
                    }
                    if (anyValid < 0) {
                        ++noSampleRows[yy];
                    } else {
                        int one = -1;
                        if (den > 1e-7) {
                            for (int c = 0; c < C; ++c) rgb[c] = float(num[c] / den);
                        } else if (moved) {
                            one = refPos;          // the clipped reference itself
                            ++unexposedRows[yy];
                        } else if (rescue >= 0) {
                            // Deghosting left this pixel only frames that clip
                            // here; one unclipped frame on its own cannot ghost,
                            // and beats the reference's clipped grey.
                            one = rescue;
                        } else {
                            one = anyValid;        // the darkest: every frame clips
                            ++unexposedRows[yy];
                        }
                        if (one >= 0) {
                            const double fo = fac[size_t(order[size_t(one)])];
                            for (int c = 0; c < C; ++c) rgb[c] = float(double(v[one][c]) * fo);
                        }
                    }
                } else {
                    // No deghost: every frame, weighted, nothing to decide.
                    int fallback = -1;
                    for (int k = 0; k < n; ++k) {
                        const int fi = order[size_t(k)];
                        const Frame& f = m_frames[size_t(fi)];
                        if (IsNoSample(f.px[i * size_t(C)])) continue;
                        if (fallback < 0) fallback = fi;     // the darkest with a sample
                        const float wt = float(f.wclip[i]) * (1.0f / 255.0f) * wn[size_t(fi)];
                        if (wt <= 0.0f) continue;
                        const double s = double(wt) * fac[size_t(fi)];
                        for (int c = 0; c < C; ++c)
                            num[c] += s * double(HalfToFloat(f.px[i * size_t(C) + size_t(c)]));
                        den += double(wt);
                    }
                    if (fallback < 0) {
                        ++noSampleRows[yy];
                    } else if (den > 1e-7) {
                        for (int c = 0; c < C; ++c) rgb[c] = float(num[c] / den);
                    } else {
                        const Frame& f = m_frames[size_t(fallback)];
                        for (int c = 0; c < C; ++c)
                            rgb[c] = float(double(HalfToFloat(f.px[i * size_t(C) + size_t(c)])) * fac[size_t(fallback)]);
                        ++unexposedRows[yy];
                    }
                }

                if (C == 1) rgb[1] = rgb[2] = rgb[0];
                else if (m_hasMatrix) CameraMatrixInGamut(m_desc, rgb);
                float* o = &op[i * size_t(oc)];
                for (int c = 0; c < oc; ++c) o[c] = (c == 3) ? 1.0f : rgb[std::min(c, 2)];
            }
            const int done = ++rowsDone;
            if ((done & 63) == 0) GroupProgress(0.6 + 0.4 * double(done) / double(h), "merging");
        });
        if (deghost && oc >= 3) {
            bool any = false;
            for (float u : unseenW) if (u > 0.0f) { any = true; break; }
            if (any) FillUnseen(op, oc, w, h, unseenW, fillSource);
        }
        if (deghost) {
            size_t g = 0, m = 0;
            for (int y = 0; y < h; ++y) { g += size_t(ghostRows[size_t(y)]); m += size_t(movedRows[size_t(y)]); }
            m_ghostPct = float(100.0 * double(g) / double(N));
            m_movedPct = float(100.0 * double(m) / double(N));
        }
        ob.PackInto(ov);
        for (int y = 0; y < h; ++y) { m_unexposed += unexposedRows[size_t(y)]; }
        size_t noSample = 0;
        for (int y = 0; y < h; ++y) noSample += size_t(noSampleRows[size_t(y)]);
        m_noSamplePct = float(100.0 * double(noSample) / double(N));

        m_reported = n;
        m_frames.clear();
        m_frames.shrink_to_fit();
        return true;
    }

    std::string RunReport() const override {
        if (m_reported == 0) return {};
        char buf[160];
        std::string r = "merged " + std::to_string(m_reported) + " exposures";
        if (m_spanStops > 0.0f) {
            std::snprintf(buf, sizeof buf, " spanning %.1f stops", double(m_spanStops));
            r += buf;
        }
        if (m_noExposure) {
            r += m_measure && m_unmeasured == 0
                ? " (NO EXIF exposure: ratios measured from the frames)"
                : " (NO EXIF exposure: frames treated as equal where it could not be measured)";
        } else if (m_exifOff >= 0.0f) {
            std::snprintf(buf, sizeof buf, ", ratios measured (EXIF off by up to %.2f stops)",
                          double(m_exifOff));
            r += buf;
        }
        if (m_unmeasured && !m_noExposure)
            r += ", " + std::to_string(m_unmeasured) + " pair(s) too little overlap to measure, EXIF used";
        if (m_inconsistent)
            r += ", " + std::to_string(m_inconsistent) +
                 " pair(s) not one consistent ratio (did the light change?), EXIF used";
        if (m_ghostPct >= 0.0f) {
            std::snprintf(buf, sizeof buf, ", deghost: %.1f%% of pixels dropped a frame (%.2f%% moved highlights)",
                          double(m_ghostPct), double(m_movedPct));
            r += buf;
        }
        if (m_unexposed) {
            r += ", " + std::to_string(m_unexposed) + " pixels outside every exposure";
        }
        if (m_noSamplePct > 0.0f) {
            std::snprintf(buf, sizeof buf, ", %.2f%% outside every aligned frame", double(m_noSamplePct));
            r += buf;
        }

        // Whether alignment applied, ALWAYS said out loud.
        //
        // Tim compared an aligned and an unaligned merge and got identical
        // images, because merge_hdr did not read the transform at all. The
        // sidecar being optional means its absence is silent by design, so the
        // consumer has to be the one that says whether it found anything.
        r += m_warped ? (", " + std::to_string(m_warped) + " frames aligned")
                      : ", not aligned";
        return r;
    }

private:
    struct Frame {
        std::vector<uint16_t> px;      // C halves per pixel, balanced camera RGB
        std::vector<uint8_t>  wclip;   // 255 = far from clipping, 0 = clipped or no sample
        std::vector<float>    local;   // half-res blurred channel sum, < 0 = none
        float exif      = 0.0f;        // relative exposure from EXIF, 0 = unknown
        float ceiling   = 1.0f;        // where this frame's samples clip
        float meanLevel = 0.0f;        // for ordering without EXIF
    };

    Param<bool> m_measure{this, "measure_exposure", true,
        "Measure the exposure ratio between neighbouring frames from the pixels "
        "both expose well, instead of trusting EXIF. EXIF shutter speeds are "
        "nominal, and a ratio that is off shows as a brightness step where one "
        "frame takes over from the next. The report says how far EXIF was off. "
        "The overall level is still anchored on the middle frame's EXIF."};

    Param<bool> m_deghost{this, "deghost", true,
        "Leave a frame out wherever it disagrees with the middle of the bracket "
        "-- clouds, leaves, water, people -- instead of averaging the two into a "
        "ghost. Also catches a highlight that moved: where the middle frame is "
        "clipped and a darker one is far darker, the darker one saw something "
        "else."};

    Param<float> m_ghostStops{this, "ghost_threshold", 0.5f, 0.1f, 3.0f,
        {.help = "How far, in stops, a frame may disagree with the reference "
                 "(locally averaged over a few pixels) before it is left out "
                 "there. Lower catches "
                 "fainter motion and starts to drop frames for noise in the "
                 "shadows; higher lets soft motion through as a ghost.",
         .step = 0.05}};

    std::vector<Frame> m_frames;

    ImageDesc m_desc{};
    int       m_C = 3;
    bool      m_hasMatrix = false;
    float     m_camInv[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    bool      m_noExposure = false;
    int       m_unexposed  = 0;
    int       m_reported   = 0;
    float     m_spanStops  = 0.0f;
    float     m_exifOff    = -1.0f;
    int       m_unmeasured = 0;
    int       m_inconsistent = 0;   // pairs whose ratios did not agree: EXIF kept
    float     m_ghostPct   = -1.0f, m_movedPct = 0.0f;
    float     m_noSamplePct = 0.0f;

    // The count is for the run report, because "did the alignment actually
    // apply" is exactly the question a user has when an aligned and unaligned
    // merge look identical.
    int       m_warped = 0;
};

} // namespace

REGISTER_ALGORITHM(MergeHdr);

} // namespace tglab
