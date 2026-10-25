// highlight_reconstruct: rebuilds partly clipped highlights from nearby colour.
//
// WHAT IT FIXES. A demosaiced raw is clamped at the brightest neutral its
// sensor can describe (clip_repair.h): every white-balanced channel at or above
// min(camMul) is cut there. That stops a blown light rendering magenta, and it
// throws away what the unclipped channels knew. Where only SOME channels
// clipped -- the warm halo round a streetlight, a sunlit cloud's edge, skin in
// hard sun -- the pixel is still mostly measured: one or two channels are real,
// and only the clipped one is unknown. Clamped, the pixel goes paler and
// flatter than the scene was, and the halo turns into a pale disc; and every
// darkening control afterwards (highlights, negative exposure) shows it.
//
// HOW. In white-balanced camera RGB, recovered by undoing the camera matrix:
//
//   1. Clipped channels are the ones at the clamp.
//   2. The COLOUR of the surroundings is learned from reliable pixels -- bright
//      enough to describe a highlight (dark foliage says nothing about the
//      colour of a lamp), not on a strong edge (edge pixels mix two surfaces
//      and carry lens fringes), and not next to clipping (those were
//      demosaiced partly from clipped samples). Their chromaticity is spread
//      into the clipped areas by a pull-push pyramid at half resolution, so a
//      clipped region takes the colour of what surrounds it at whatever scale
//      reaches it, and fades to neutral where nothing reliable is near.
//   3. A partly clipped pixel keeps its unclipped channels; the clipped ones
//      are rebuilt at that chromaticity, at the level the unclipped channels
//      imply, and never below the clamp they were cut at.
//   4. A pixel with every channel clipped carries no colour and no level: it
//      becomes neutral, from the clamp at the rim of a clipped area rising to
//      the brightest level the sensor allows at its heart, and the partly
//      clipped pixels around fade into that same level -- so the core of a
//      light is never darker than its halo, and never steps up from it.
//
// Then the camera matrix goes back on. The approach is lightcraft's
// (crates/raw/src/highlight.rs), simplified: no "rim" colour blending, and a
// single trust fade instead of its agreement tests.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "../../algo_util/color.h"
#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"
#include "../../core/parallel.h"
#include "clip_repair.h"

namespace tglab {
namespace {

inline float Smoothstep(float a, float b, float x) {
    const float t = std::clamp((x - a) / (b - a), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

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

// Reliable pixels lend colour in proportion to how bright they are, relative
// to the clamp: from nothing below kSourceLo to fully above kSourceHi.
constexpr float kSourceLo = 0.25f, kSourceHi = 0.6f;
// A source on an edge (a 3x3 neighbour brighter or darker by this many stops)
// fades out between these.
constexpr float kEdgeLo = 0.15f, kEdgeHi = 0.4f;
// Pixels within this many pixels of a clipped one are not sources.
constexpr int kMargin = 2;
// How much a coarser estimate counts against a cell's own sources, in units
// of source density.
constexpr float kPrior = 0.01f;
// The rebuilt colour fades to neutral where reliable content covers less than
// kSupport of the surroundings at every scale up to kReach of the long edge.
constexpr float kReach = 0.25f, kSupport = 0.05f;

// One pyramid level: per cell, red and blue chromaticity (weighted, as
// densities) and the weight density.
struct Level {
    int w = 0, h = 0;
    std::vector<float> r, b, wt;
};

void Sample(const Level& l, float scale, int x, int y, float* r, float* b, float* wt) {
    const float fx = std::clamp((float(x) + 0.5f) / scale - 0.5f, 0.0f, float(l.w - 1));
    const float fy = std::clamp((float(y) + 0.5f) / scale - 0.5f, 0.0f, float(l.h - 1));
    const int x0 = int(fx), y0 = int(fy);
    const int x1 = std::min(x0 + 1, l.w - 1), y1 = std::min(y0 + 1, l.h - 1);
    const float tx = fx - float(x0), ty = fy - float(y0);
    auto bil = [&](const std::vector<float>& v) {
        const float a = v[size_t(y0) * size_t(l.w) + size_t(x0)];
        const float c = v[size_t(y0) * size_t(l.w) + size_t(x1)];
        const float d = v[size_t(y1) * size_t(l.w) + size_t(x0)];
        const float e = v[size_t(y1) * size_t(l.w) + size_t(x1)];
        const float top = a + (c - a) * tx, bot = d + (e - d) * tx;
        return top + (bot - top) * ty;
    };
    if (r)  *r  = bil(l.r);
    if (b)  *b  = bil(l.b);
    if (wt) *wt = bil(l.wt);
}

Level Pull(const Level& l) {
    Level o;
    o.w = std::max(1, (l.w + 1) / 2);
    o.h = std::max(1, (l.h + 1) / 2);
    const size_t n = size_t(o.w) * size_t(o.h);
    o.r.assign(n, 0.0f); o.b.assign(n, 0.0f); o.wt.assign(n, 0.0f);
    for (int y = 0; y < o.h; ++y)
        for (int x = 0; x < o.w; ++x) {
            float sr = 0, sb = 0, sw = 0; int k = 0;
            for (int dy = 0; dy < 2; ++dy)
                for (int dx = 0; dx < 2; ++dx) {
                    const int sx = 2 * x + dx, sy = 2 * y + dy;
                    if (sx >= l.w || sy >= l.h) continue;
                    const size_t i = size_t(sy) * size_t(l.w) + size_t(sx);
                    sr += l.r[i]; sb += l.b[i]; sw += l.wt[i]; ++k;
                }
            const size_t i = size_t(y) * size_t(o.w) + size_t(x);
            o.r[i] = sr / float(k); o.b[i] = sb / float(k); o.wt[i] = sw / float(k);
        }
    return o;
}

class HighlightReconstruct : public AlgorithmBase {
public:
    const char* Name()     const override { return "highlight_reconstruct"; }
    const char* Category() const override { return "adjust"; }

    PortList Inputs()  const override { return {{"src", DataType::Image, FormatSpec::Any}}; }
    PortList Outputs() const override { return {{"out", DataType::Image, FormatSpec::SameAsInput}}; }

    // The colour a clipped region takes comes from anywhere within a quarter
    // of the frame: on a region it would be whatever the region holds.
    bool RegionSafe() const override { return false; }

    void RunCPU(RunCtx& ctx) override {
        const ImageView src = ctx.In(0);
        ImageView dst = ctx.Out(0);
        if (!src.Valid() || !dst.Valid()) return;
        PixelBuffer buf;
        buf.Unpack(src);
        if (!buf.Valid()) return;
        const ImageDesc& d = src.desc;
        const int w = buf.Width(), h = buf.Height(), ch = buf.Channels();
        const float scale = buf.ValueScale();

        // Only a demosaiced raw has a clamp to undo: it carries white balance
        // gains and a camera matrix. Anything else passes through.
        float minv[9];
        const float ceiling = std::min(d.camMul[0], std::min(d.camMul[1], d.camMul[2]));
        const float maxMul  = std::max(d.camMul[0], std::max(d.camMul[1], d.camMul[2]));
        const bool raw = d.linear && ch >= 3 && ceiling > 0.0f && maxMul > ceiling * 1.0001f &&
                         Invert3(d.rgbCam, minv);
        if (!raw) {
            buf.PackInto(dst);
            ctx.SetReport("not a demosaiced raw: nothing to reconstruct");
            return;
        }

        const size_t n = size_t(w) * size_t(h);
        std::vector<float> bal(n * 3);          // white-balanced camera RGB
        std::vector<uint8_t> clip(n, 0);        // bit c: channel c at the clamp
        std::vector<float>& px = buf.Data();
        ParallelFor(size_t(h), [&](size_t y) {
            for (int x = 0; x < w; ++x) {
                const size_t i = y * size_t(w) + size_t(x);
                const float* p = &px[i * size_t(ch)];
                const float r = p[0] / scale, g = p[1] / scale, b = p[2] / scale;
                float* q = &bal[i * 3];
                q[0] = minv[0] * r + minv[1] * g + minv[2] * b;
                q[1] = minv[3] * r + minv[4] * g + minv[5] * b;
                q[2] = minv[6] * r + minv[7] * g + minv[8] * b;
                uint8_t m = 0;
                for (int c = 0; c < 3; ++c) if (q[c] >= 0.995f * ceiling) m |= uint8_t(1 << c);
                clip[i] = m;
            }
        });
        size_t clipped = 0, full = 0;
        for (uint8_t m : clip) { clipped += m ? 1 : 0; full += (m == 7) ? 1 : 0; }
        if (clipped == 0) {
            buf.PackInto(dst);
            ctx.SetReport("no clipped pixels");
            return;
        }

        // Near clipping: dilated by kMargin, separably.
        std::vector<uint8_t> nearH(n, 0), nearC(n, 0);
        ParallelFor(size_t(h), [&](size_t y) {
            for (int x = 0; x < w; ++x) {
                uint8_t v = 0;
                for (int k = -kMargin; k <= kMargin && !v; ++k) {
                    const int xx = std::clamp(x + k, 0, w - 1);
                    v = clip[y * size_t(w) + size_t(xx)] ? 1 : 0;
                }
                nearH[y * size_t(w) + size_t(x)] = v;
            }
        });
        ParallelFor(size_t(h), [&](size_t y) {
            for (int x = 0; x < w; ++x) {
                uint8_t v = 0;
                for (int k = -kMargin; k <= kMargin && !v; ++k) {
                    const int yy = std::clamp(int(y) + k, 0, h - 1);
                    v = nearH[size_t(yy) * size_t(w) + size_t(x)];
                }
                nearC[y * size_t(w) + size_t(x)] = v;
            }
        });

        // The source level, at half resolution.
        Level l0;
        l0.w = std::max(1, (w + 1) / 2);
        l0.h = std::max(1, (h + 1) / 2);
        const size_t cn = size_t(l0.w) * size_t(l0.h);
        l0.r.assign(cn, 0.0f); l0.b.assign(cn, 0.0f); l0.wt.assign(cn, 0.0f);
        const float edgeLo = std::exp2(kEdgeLo), edgeHi = std::exp2(kEdgeHi);
        auto lumAt = [&](int x, int y) {
            const float* q = &bal[(size_t(y) * size_t(w) + size_t(x)) * 3];
            return std::max(1e-6f, std::max(0.0f, q[0]) + std::max(0.0f, q[1]) + std::max(0.0f, q[2]));
        };
        ParallelFor(size_t(l0.h), [&](size_t cy) {
            for (int cx = 0; cx < l0.w; ++cx) {
                float sr = 0, sb = 0, sw = 0; int k = 0;
                for (int dy = 0; dy < 2; ++dy)
                    for (int dx = 0; dx < 2; ++dx) {
                        const int x = 2 * cx + dx, y = 2 * int(cy) + dy;
                        if (x >= w || y >= h) continue;
                        ++k;
                        const size_t i = size_t(y) * size_t(w) + size_t(x);
                        if (nearC[i]) continue;
                        const float* q = &bal[i * 3];
                        const float top = std::max(q[0], std::max(q[1], q[2])) / ceiling;
                        if (top <= kSourceLo) continue;
                        const float s = lumAt(x, y);
                        float lo = s, hi = s;
                        for (int ey = -1; ey <= 1; ++ey)
                            for (int ex = -1; ex <= 1; ++ex) {
                                const float v = lumAt(std::clamp(x + ex, 0, w - 1), std::clamp(y + ey, 0, h - 1));
                                lo = std::min(lo, v); hi = std::max(hi, v);
                            }
                        const float keep = 1.0f - Smoothstep(edgeLo, edgeHi, std::max(hi / s, s / lo));
                        const float wgt = Smoothstep(kSourceLo, kSourceHi, top) * keep;
                        if (wgt <= 0.0f) continue;
                        sr += wgt * std::max(0.0f, q[0]) / s;
                        sb += wgt * std::max(0.0f, q[2]) / s;
                        sw += wgt;
                    }
                const size_t ci = cy * size_t(l0.w) + size_t(cx);
                const float inv = 1.0f / float(std::max(k, 1));
                l0.r[ci] = sr * inv; l0.b[ci] = sb * inv; l0.wt[ci] = sw * inv;
            }
        });

        // Pull to a single cell, then push the estimates back down: each level
        // its own sources blended with the coarser estimate. The top is its own
        // sources' mean, or neutral when it has none.
        std::vector<Level> levels{std::move(l0)};
        while (levels.back().w > 1 || levels.back().h > 1) levels.push_back(Pull(levels.back()));
        // Support (the raw densities) kept for the trust fade before the push
        // overwrites them.
        std::vector<std::vector<float>> density;
        for (const Level& l : levels) density.push_back(l.wt);
        {
            Level& top = levels.back();
            for (size_t i = 0; i < top.wt.size(); ++i) {
                if (top.wt[i] > 0.0f) { top.r[i] /= top.wt[i]; top.b[i] /= top.wt[i]; }
                else { top.r[i] = top.b[i] = 1.0f / 3.0f; }
            }
        }
        for (int li = int(levels.size()) - 2; li >= 0; --li) {
            Level& fine = levels[size_t(li)];
            const Level& coarse = levels[size_t(li + 1)];
            ParallelFor(size_t(fine.h), [&](size_t y) {
                for (int x = 0; x < fine.w; ++x) {
                    const size_t i = y * size_t(fine.w) + size_t(x);
                    float pr, pb;
                    Sample(coarse, 2.0f, x, int(y), &pr, &pb, nullptr);
                    const float dd = fine.wt[i] + kPrior;
                    fine.r[i] = (fine.r[i] + kPrior * pr) / dd;
                    fine.b[i] = (fine.b[i] + kPrior * pb) / dd;
                }
            });
        }
        // Trust: how much reliable content lies around, at any scale up to
        // kReach of the long edge (level i's cells span 2^(i+1) pixels).
        const float reachPx = kReach * float(std::max(w, h));
        int widest = int(levels.size()) - 1;
        for (int i = 0; i < int(levels.size()); ++i)
            if (float(2 << i) >= reachPx) { widest = i; break; }

        // HOW DEEP INTO A CLIPPED AREA each pixel is: the fraction of fully
        // clipped pixels in a 7x7 window. After the demosaic's clamp a channel
        // the sensor clipped and one the clamp merely cut both read exactly at
        // the clamp, so "every channel at the clamp" means only "every channel
        // unknown, and at least this". The least the scene can have been there
        // is the clamp itself; the most plausible core of a light is brighter.
        // So a fully clipped pixel's level runs from the clamp at the rim of
        // the area to the brightest neutral the sensor allows at its heart,
        // and the partly clipped pixels around fade into the same level. Set
        // to the brightest level outright, the core stood out from its halo in
        // a hard, speckled edge as soon as the image was darkened.
        std::vector<float> depth(n, 0.0f);
        {
            constexpr int R = 3;
            std::vector<float> rowSum(n, 0.0f);
            ParallelFor(size_t(h), [&](size_t y) {
                for (int x = 0; x < w; ++x) {
                    int c = 0;
                    for (int k = -R; k <= R; ++k)
                        c += clip[y * size_t(w) + size_t(std::clamp(x + k, 0, w - 1))] == 7 ? 1 : 0;
                    rowSum[y * size_t(w) + size_t(x)] = float(c);
                }
            });
            ParallelFor(size_t(h), [&](size_t y) {
                for (int x = 0; x < w; ++x) {
                    float c = 0.0f;
                    for (int k = -R; k <= R; ++k)
                        c += rowSum[size_t(std::clamp(int(y) + k, 0, h - 1)) * size_t(w) + size_t(x)];
                    depth[y * size_t(w) + size_t(x)] = c / float((2 * R + 1) * (2 * R + 1));
                }
            });
        }
        auto clippedLevel = [&](size_t i) {
            return ceiling + (maxMul - ceiling) * Smoothstep(0.0f, 1.0f, depth[i]);
        };

        // Rebuild.
        size_t rebuiltPartial = 0;
        std::vector<size_t> partRows(size_t(h), 0);
        ParallelFor(size_t(h), [&](size_t y) {
            for (int x = 0; x < w; ++x) {
                const size_t i = y * size_t(w) + size_t(x);
                const uint8_t m = clip[i];
                if (!m) continue;
                float* q = &bal[i * 3];
                float out[3];
                if (m == 7) {
                    const float L = clippedLevel(i);         // neutral; see depth above
                    for (float& v : out) v = L;
                } else {
                    float er, eb;
                    Sample(levels[0], 2.0f, x, int(y), &er, &eb, nullptr);
                    float trust = 0.0f;
                    for (int li = 0; li <= widest && trust < 1.0f; ++li) {
                        const int lw = levels[size_t(li)].w, lh = levels[size_t(li)].h;
                        const float scaleL = float(2 << li);
                        const float fx = std::clamp((float(x) + 0.5f) / scaleL - 0.5f, 0.0f, float(lw - 1));
                        const float fy = std::clamp((float(y) + 0.5f) / scaleL - 0.5f, 0.0f, float(lh - 1));
                        const float dv = density[size_t(li)][size_t(int(fy)) * size_t(lw) + size_t(int(fx))];
                        trust = std::max(trust, Smoothstep(0.0f, kSupport, dv));
                    }
                    const float local[3] = {er, std::max(0.0f, 1.0f - er - eb), eb};
                    const float plain[3] = {1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f};
                    auto rebuild = [&](const float* r, float* o) {
                        float sum = 0.0f; int k = 0;
                        for (int c = 0; c < 3; ++c)
                            if (!(m & (1 << c)) && r[c] > 1e-3f) { sum += q[c] / r[c]; ++k; }
                        if (k == 0) { for (int c = 0; c < 3; ++c) o[c] = q[c]; return; }
                        float floorL = 0.0f, topR = 1e-3f;
                        for (int c = 0; c < 3; ++c) {
                            topR = std::max(topR, r[c]);
                            if (m & (1 << c)) floorL = std::max(floorL, q[c] / std::max(r[c], 1e-3f));
                        }
                        const float level = std::max(sum / float(k), std::min(floorL, maxMul / topR));
                        for (int c = 0; c < 3; ++c)
                            o[c] = (m & (1 << c)) ? std::min(maxMul, std::max(q[c], level * r[c])) : q[c];
                    };
                    float a[3], b[3];
                    rebuild(local, a);
                    rebuild(plain, b);
                    for (int c = 0; c < 3; ++c) out[c] = b[c] + (a[c] - b[c]) * trust;

                    // ON THE WAY TO FULLY CLIPPED, fade into it: as the channels
                    // still measured approach their own clip, the pixel goes
                    // over to the clipped neutral. Without this the core of a
                    // light -- every channel clipped, neutral at the brightest
                    // allowed level -- stood out from the rebuilt halo round it
                    // in a hard, pixel-stepped edge, visible as soon as the
                    // image was darkened.
                    float nearest = 0.0f;
                    for (int c = 0; c < 3; ++c)
                        if (!(m & (1 << c))) nearest = std::max(nearest, q[c] / d.camMul[c]);
                    const float fade = std::max(Smoothstep(0.75f, 1.0f, nearest), depth[i]);
                    const float L = clippedLevel(i);
                    for (int c = 0; c < 3; ++c) out[c] += (std::max(L, out[c]) - out[c]) * fade;
                    ++partRows[y];
                }
                q[0] = out[0]; q[1] = out[1]; q[2] = out[2];
            }
        });
        for (size_t v : partRows) rebuiltPartial += v;

        // The camera matrix back on, with its in-gamut solve.
        ParallelFor(size_t(h), [&](size_t y) {
            for (int x = 0; x < w; ++x) {
                const size_t i = y * size_t(w) + size_t(x);
                if (!clip[i]) continue;
                float rgb[3] = {bal[i * 3], bal[i * 3 + 1], bal[i * 3 + 2]};
                CameraMatrixInGamut(d, rgb);
                float* p = &px[i * size_t(ch)];
                for (int c = 0; c < 3; ++c) p[c] = rgb[c] * scale;
            }
        });
        buf.PackInto(dst);
        char r[160];
        std::snprintf(r, sizeof r, "%zu partly clipped pixels rebuilt from nearby colour, %zu fully "
                                   "clipped set neutral (%.2f%% of the frame)",
                      rebuiltPartial, full, 100.0 * double(clipped) / double(n));
        ctx.SetReport(r);
    }
};

REGISTER_ALGORITHM(HighlightReconstruct);

} // namespace
} // namespace tglab
