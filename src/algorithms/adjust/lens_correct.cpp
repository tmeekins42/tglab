// lens_correct — undo a lens's distortion, lateral chromatic aberration and
// vignetting, in one resample.
//
// THE MODEL. Every quantity is relative to the frame, so a proxy and the full
// image are corrected identically: positions are measured from the centre in
// units of the HALF DIAGONAL, so r = 1 at a corner.
//
//   distortion  the lens put a scene point at radius r at r * (1 + k1 r^2 + k2 r^4)
//               (k1 < 0 is barrel: straight lines bow outward; > 0 pincushion)
//   lateral CA  it put the red and blue images at (1 + a) times the green one's
//               radius: one magnification per colour, which is what lateral
//               CA is -- the lens's focal length differs slightly by
//               wavelength, so the colour planes are concentric scalings
//   vignetting  it passed 2^-(stops * r^power) of the light at radius r
//
// Correcting is the inverse map, evaluated per output pixel: where in the
// photograph did the lens put the light that belongs HERE? Output radius r
// samples green at r * (1 + k1 r^2 + k2 r^4), red and blue at that times their
// magnification, and the vignetting gain is applied for the radius sampled. One
// bicubic resample per channel, so the image is interpolated once however many
// corrections are on.
//
// FILL. Undoing pincushion pulls the frame's edges inward, and undoing red
// magnification does the same to one channel: the output's border then
// samples outside the photograph. `fill` scales the map so that every edge of
// the output lands inside it -- the largest such scale, so undoing barrel
// (which leaves the source's edges unused) also zooms out to use them.
//
// AUTOMATIC LATERAL CA (auto_ca). Measured, not looked up: strong edges that
// run across the radius are found in green, spread over the frame, and the red
// and blue profiles across each are matched to green's to a twentieth of a
// pixel. Lateral CA moves each plane radially in proportion to radius, so a
// robust least-squares fit of displacement = a * r over all of them is the
// magnification. The approach is lightcraft's (crates/pipeline/src/optics.rs,
// estimate_lateral_ca). Manual ca_red / ca_blue add to what was measured.
//
// LENS PROFILES (profile). When the lensfun database is installed
// (core/lensfun.h) and the raw names its lens, the lens's measured distortion,
// TCA and vignetting at this focal length and aperture are looked up and
// applied, with no slider touched. The profile widens the model above: the
// distortion becomes a quartic in r, f(r) = d0 + d1 r + d2 r^2 + d3 r^3 +
// d4 r^4, and each colour's magnification a quadratic in green's radius. k1
// and k2 add to the profile's r^2 and r^4 terms and ca_red / ca_blue to its
// magnifications, so the sliders stay a trim on top. A profile with TCA
// replaces auto_ca's measurement rather than adding to it -- both are the
// whole aberration, and both together would correct it twice.
//
// WHERE IT GOES: right after the demosaic, on the linear image, before any
// merge -- a panorama or a stack aligns better on undistorted frames, and the
// vignetting gain is a transmission and belongs to linear light. On an encoded
// image the gain is applied through the sRGB curve.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../../algo_util/color.h"
#include "../../algo_util/pixel_buffer.h"
#include "../../algo_util/resample.h"
#include "../../core/algorithm.h"
#include "../../core/lensfun.h"
#include "../../core/parallel.h"

namespace tglab {
namespace {

// The inverse map, in full-frame pixel coordinates (pixel centres at +0.5).
struct LensMap {
    double cx = 0, cy = 0, hd = 1;              // centre and half diagonal
    double d[5] = {1, 0, 0, 0, 0};              // green's source radius over r
    double scale = 1;                           // the fill scale, on the output radius
    double tca[2][3] = {{1, 0, 0}, {1, 0, 0}};  // red, blue magnification at green's radius
    double vig[3] = {0, 0, 0};                  // the profile's falloff, k1 r^2 + k2 r^4 + k3 r^6

    bool Identity() const {
        return d[0] == 1 && d[1] == 0 && d[2] == 0 && d[3] == 0 && d[4] == 0 &&
               tca[0][0] == 1 && tca[0][1] == 0 && tca[0][2] == 0 &&
               tca[1][0] == 1 && tca[1][1] == 0 && tca[1][2] == 0;
    }
    double F(double r) const { return d[0] + r * (d[1] + r * (d[2] + r * (d[3] + r * d[4]))); }
    double Mag(int c, double rg) const {
        if (c == 1) return 1.0;
        const double* t = tca[c == 0 ? 0 : 1];
        return t[0] + rg * (t[1] + rg * t[2]);
    }
    void Source(double px, double py, int c, double* sx, double* sy) const {
        const double ux = (px - cx) / hd * scale, uy = (py - cy) / hd * scale;
        const double r = std::sqrt(ux * ux + uy * uy);
        const double f = F(r);
        const double m = f * Mag(c, r * f);
        *sx = cx + ux * m * hd;
        *sy = cy + uy * m * hd;
    }
    // The profile's gain at a source radius: what the lens lost, given back.
    // Held to 8x so a profile evaluated past its corner cannot blow up.
    double VigGain(double rs) const {
        const double r2 = rs * rs;
        const double kept = 1.0 + r2 * (vig[0] + r2 * (vig[1] + r2 * vig[2]));
        return 1.0 / std::max(kept, 0.125);
    }
    bool HasVig() const { return vig[0] != 0 || vig[1] != 0 || vig[2] != 0; }
};

// The largest scale at which every point of the output's border samples
// inside the w x h frame, for every channel. 1 when nothing moves.
double FitScale(LensMap m, double w, double h) {
    if (m.Identity()) return 1.0;
    constexpr int kSteps = 64;
    auto inside = [&](double s) {
        m.scale = s;
        const double tol = 1e-3;
        for (int i = 0; i <= kSteps; ++i) {
            const double t = double(i) / kSteps;
            const double pts[4][2] = {{t * w, 0}, {t * w, h}, {0, t * h}, {w, t * h}};
            for (const auto& p : pts)
                for (int c = 0; c < 3; ++c) {
                    double sx, sy;
                    m.Source(p[0], p[1], c, &sx, &sy);
                    if (sx < -tol || sy < -tol || sx > w + tol || sy > h + tol) return false;
                }
        }
        return true;
    };
    double lo = 0.25, hi = 4.0;
    if (!inside(lo)) return lo;
    if (inside(hi)) return hi;
    for (int it = 0; it < 40; ++it) {
        const double mid = 0.5 * (lo + hi);
        (inside(mid) ? lo : hi) = mid;
    }
    return lo;
}


// --- automatic lateral CA ---------------------------------------------------

struct CaEstimate {
    double a[2] = {0, 0};   // red, blue magnification - 1
    int n[2] = {0, 0};      // edges each fit used
};

// From float RGB(A) pixels; cx, cy the optical centre in this raster's pixels.
CaEstimate EstimateLateralCa(const float* px, int w, int h, int ch, float valueScale,
                             double cx, double cy) {
    CaEstimate out;
    if (w < 64 || h < 64 || ch < 3) return out;
    const double hd = 0.5 * std::hypot(double(w), double(h));
    auto at = [&](int x, int y, int c) {
        return px[(size_t(y) * size_t(w) + size_t(x)) * size_t(ch) + size_t(c)];
    };
    auto bil = [&](double x, double y, int c) {
        x = std::clamp(x, 0.0, double(w - 1)); y = std::clamp(y, 0.0, double(h - 1));
        const int x0 = int(x), y0 = int(y);
        const int x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
        const float tx = float(x - x0), ty = float(y - y0);
        const float a = at(x0, y0, c) + (at(x1, y0, c) - at(x0, y0, c)) * tx;
        const float b = at(x0, y1, c) + (at(x1, y1, c) - at(x0, y1, c)) * tx;
        return a + (b - a) * ty;
    };

    // CANDIDATES: the strongest radial green edges in each cell of a grid, so
    // the fit sees the whole frame rather than the one busiest corner. Relative
    // gradient, so the threshold means the same in linear and encoded data.
    // Clipped edges are left out: a blown light's rim is the same in every
    // channel because every channel stopped, not because the lens agreed.
    constexpr int kCells = 24, kPerCell = 12, kBorder = 8;
    struct Cand { float rel; int x, y; };
    std::vector<std::vector<Cand>> cells(size_t(kCells) * kCells);
    const float eps = 0.02f * valueScale;
    float clipAt = 0;
    for (int y = 0; y < h; y += 7)
        for (int x = 0; x < w; x += 7) clipAt = std::max(clipAt, at(x, y, 1));
    clipAt *= 0.95f;
    ParallelFor(size_t(kCells), [&](size_t cyc) {
        const int y0 = std::max(kBorder, int(cyc) * h / kCells);
        const int y1 = std::min(h - kBorder, (int(cyc) + 1) * h / kCells);
        for (int y = y0 + (y0 & 1); y < y1; y += 2)
            for (int x = kBorder; x < w - kBorder; x += 2) {
                const float gx = at(x + 1, y, 1) - at(x - 1, y, 1);
                const float gy = at(x, y + 1, 1) - at(x, y - 1, 1);
                const float mag = std::hypot(gx, gy);
                const float mean = 0.25f * (at(x + 1, y, 1) + at(x - 1, y, 1) +
                                            at(x, y + 1, 1) + at(x, y - 1, 1)) + eps;
                const float rel = mag / mean;
                if (rel < 0.2f) continue;
                const double vx = x - cx, vy = y - cy, rr = std::hypot(vx, vy);
                if (rr < 0.2 * hd) continue;
                if (std::fabs((gx * vx + gy * vy) / (mag * rr)) < 0.85) continue;
                bool clipped = false;
                for (int k = -2; k <= 2 && !clipped; ++k)
                    for (int c = 0; c < 3; ++c)
                        if (at(std::clamp(x + k, 0, w - 1), y, c) >= clipAt ||
                            at(x, std::clamp(y + k, 0, h - 1), c) >= clipAt) clipped = true;
                if (clipped) continue;
                auto& v = cells[cyc * kCells + size_t(std::min(kCells - 1, x * kCells / w))];
                if (int(v.size()) < kPerCell) { v.push_back({rel, x, y}); continue; }
                auto mn = std::min_element(v.begin(), v.end(),
                                           [](const Cand& a, const Cand& b) { return a.rel < b.rel; });
                if (mn->rel < rel) *mn = {rel, x, y};
            }
    });
    std::vector<Cand> pts;
    for (auto& v : cells) pts.insert(pts.end(), v.begin(), v.end());

    // DISPLACEMENT along the radius of red and blue against green: the shift of
    // the normalised 9-sample profile that matches best, in 1/20 px steps.
    constexpr int kHalf = 4, kSub = 20, kMaxD = 3;
    struct Obs { double r, d, wt; };
    std::vector<Obs> obsAll[2];
    auto znorm = [](float* v, int n) {
        float m = 0; for (int i = 0; i < n; ++i) m += v[i];
        m /= float(n);
        float sd = 0; for (int i = 0; i < n; ++i) sd += (v[i] - m) * (v[i] - m);
        sd = std::sqrt(sd / float(n));
        if (sd < 1e-6f) return false;
        for (int i = 0; i < n; ++i) v[i] = (v[i] - m) / sd;
        return true;
    };
    std::vector<Obs> found(pts.size() * 2, Obs{0, 0, -1});
    ParallelFor(pts.size(), [&](size_t k) {
        const Cand& p = pts[k];
        const double vx = p.x - cx, vy = p.y - cy, rr = std::hypot(vx, vy);
        const double ux = vx / rr, uy = vy / rr;
        constexpr int n = 2 * kHalf + 1;
        float g[n];
        for (int t = -kHalf; t <= kHalf; ++t) g[t + kHalf] = bil(p.x + ux * t, p.y + uy * t, 1);
        if (!znorm(g, n)) return;
        constexpr int reach = kHalf + kMaxD;
        std::vector<float> fine(size_t(2 * reach * kSub + 1));
        for (int ci = 0; ci < 2; ++ci) {
            const int c = ci == 0 ? 0 : 2;
            for (int i = -reach * kSub; i <= reach * kSub; ++i)
                fine[size_t(i + reach * kSub)] =
                    bil(p.x + ux * i / double(kSub), p.y + uy * i / double(kSub), c);
            float best = 3.4e38f; int bestD = 0;
            for (int d = -kMaxD * kSub; d <= kMaxD * kSub; ++d) {
                float prof[n];
                for (int t = -kHalf; t <= kHalf; ++t)
                    prof[t + kHalf] = fine[size_t((t + reach) * kSub + d)];
                if (!znorm(prof, n)) continue;
                float ssd = 0;
                for (int i = 0; i < n; ++i) ssd += (prof[i] - g[i]) * (prof[i] - g[i]);
                if (ssd < best) { best = ssd; bestD = d; }
            }
            // Normalised SSD is 2n(1 - correlation): only a good match counts.
            if (best < 0.3f * n && std::abs(bestD) < kMaxD * kSub)
                found[k * 2 + size_t(ci)] = Obs{rr, double(bestD) / kSub, double(p.rel)};
        }
    });
    for (size_t k = 0; k < found.size(); ++k)
        if (found[k].wt > 0) obsAll[k & 1].push_back(found[k]);

    // FIT d = a * r, weighted, then twice refit on what lies within 2.5 MADs.
    auto fit = [](const std::vector<Obs>& o, int* used) {
        *used = 0;
        if (o.size() < 12) return 0.0;
        auto solve = [](const std::vector<const Obs*>& s) {
            double num = 0, den = 0;
            for (const Obs* q : s) { num += q->wt * q->d * q->r; den += q->wt * q->r * q->r; }
            return den > 0 ? num / den : 0.0;
        };
        std::vector<const Obs*> keep;
        for (const Obs& q : o) keep.push_back(&q);
        double a = solve(keep);
        for (int it = 0; it < 2; ++it) {
            std::vector<double> res;
            for (const Obs& q : o) res.push_back(std::fabs(q.d - a * q.r));
            std::nth_element(res.begin(), res.begin() + res.size() / 2, res.end());
            const double mad = std::max(res[res.size() / 2], 0.05);
            std::vector<const Obs*> next;
            for (const Obs& q : o) if (std::fabs(q.d - a * q.r) <= 2.5 * mad) next.push_back(&q);
            if (next.size() < 8) break;
            keep.swap(next);
            a = solve(keep);
        }
        *used = int(keep.size());
        return a;
    };
    out.a[0] = fit(obsAll[0], &out.n[0]);
    out.a[1] = fit(obsAll[1], &out.n[1]);
    return out;
}

class LensCorrect : public AlgorithmBase {
public:
    const char* Name()     const override { return "lens_correct"; }
    const char* Category() const override { return "adjust"; }

    PortList Inputs()  const override { return {{"src", DataType::Image, FormatSpec::Any}}; }
    PortList Outputs() const override { return {{"out", DataType::Image, FormatSpec::SameAsInput}}; }

    // A warp reads wherever the lens moved the light, and the CA measurement
    // and the fill scale are properties of the whole frame.
    bool RegionSafe() const override { return false; }

    bool IsNoOp() const override {
        return float(m_k1) == 0 && float(m_k2) == 0 && float(m_caRed) == 0 &&
               float(m_caBlue) == 0 && !bool(m_autoCa) && float(m_vignette) == 0 &&
               !bool(m_profile);
    }

    void RunCPU(RunCtx& ctx) override {
        const ImageView src = ctx.In(0);
        ImageView dst = ctx.Out(0);
        if (!src.Valid() || !dst.Valid()) return;
        PixelBuffer in, out;
        in.Unpack(src);
        if (!in.Valid()) return;
        out.AllocLike(in);
        const ImageDesc& d = ctx.InDesc(0);
        const int w = in.Width(), h = in.Height(), ch = in.Channels();
        const float vs = in.ValueScale();
        const float* px = in.Data().data();

        const LensProfile prof = Profile(d);
        CaEstimate est;
        if (WantAutoCa(prof) && ch >= 3)
            est = EstimateLateralCa(px, w, h, ch, vs,
                                    0.5 * d.FullWidth() - d.originX, 0.5 * d.FullHeight() - d.originY);
        const LensMap m = Map(d, prof, est);
        const float stops = float(m_vignette), power = float(m_vigPower);
        const bool profVig = m.HasVig();
        const bool linear = d.linear;
        const int colours = std::min(ch, 3);
        float* o = out.Data().data();
        ParallelFor(size_t(h), [&](size_t y) {
            for (int x = 0; x < w; ++x) {
                const double fx = x + d.originX + 0.5, fy = double(y) + d.originY + 0.5;
                float* q = &o[(y * size_t(w) + size_t(x)) * size_t(ch)];
                double gx = 0, gy = 0;
                for (int c = 0; c < ch; ++c) {
                    const int mc = (c < 3 && colours == 3) ? c : 1;   // alpha follows green
                    double sx, sy;
                    m.Source(fx, fy, mc, &sx, &sy);
                    if (mc == 1) { gx = sx; gy = sy; }
                    q[c] = SampleCubicClamped(px, w, h, ch, c, sx - d.originX - 0.5, sy - d.originY - 0.5);
                }
                if (stops != 0.0f || profVig) {
                    const double r = std::hypot(gx - m.cx, gy - m.cy) / m.hd;
                    const float gain = std::exp2(stops * std::pow(float(r), power)) *
                                       float(m.VigGain(r));
                    for (int c = 0; c < colours; ++c)
                        q[c] = linear ? q[c] * gain
                                      : LinearToSrgb(SrgbToLinear(q[c] / vs) * gain) * vs;
                }
            }
        });
        out.PackInto(dst);
        ctx.SetReport(Report(d, prof, m, est));
    }

    // --- GPU ----------------------------------------------------------------
    // The same map, per pixel, sixteen loads per channel.
    bool HasGPU() const override { return true; }
    bool GpuNeedsInputPixels() const override { return WantAutoCa(Profile(m_desc)); }

    void PrepareGpu(const std::vector<ImageDesc>& inputs) override {
        if (inputs.empty()) return;
        m_desc = inputs[0];
        m_gpuRan = true;
    }

    void MeasureForGpu(const std::vector<const Image*>& inputs) override {
        m_est = CaEstimate{};
        m_measured = false;
        if (inputs.empty() || !inputs[0]) return;
        if (!WantAutoCa(Profile(inputs[0]->Desc()))) return;
        ImageView v = const_cast<Image*>(inputs[0])->MapCpuRead();
        if (!v.data) return;
        PixelBuffer buf;
        buf.Unpack(v);
        if (!buf.Valid() || buf.Channels() < 3) return;
        const ImageDesc& d = v.desc;
        m_est = EstimateLateralCa(buf.Data().data(), buf.Width(), buf.Height(), buf.Channels(),
                                  buf.ValueScale(), 0.5 * d.FullWidth() - d.originX,
                                  0.5 * d.FullHeight() - d.originY);
        m_measured = true;
    }

    // The CA is a property of the photograph: no slider changes it.
    bool MeasurementDependsOnlyOnInput() const override { return true; }
    std::vector<uint32_t> SaveMeasurement() const override {
        if (!m_measured) return {};
        return {FloatBits(float(m_est.a[0])), FloatBits(float(m_est.a[1])),
                uint32_t(m_est.n[0]), uint32_t(m_est.n[1])};
    }
    bool RestoreMeasurement(const std::vector<uint32_t>& b) override {
        if (b.size() != 4) return false;
        auto flt = [](uint32_t u) { float f; std::memcpy(&f, &u, sizeof f); return f; };
        m_est.a[0] = flt(b[0]); m_est.a[1] = flt(b[1]);
        m_est.n[0] = int(b[2]); m_est.n[1] = int(b[3]);
        m_measured = true;
        return true;
    }

    // The GPU path's note; the CPU path reports per call.
    std::string RunReport() const override {
        if (!m_gpuRan) return {};
        const LensProfile prof = Profile(m_desc);
        const CaEstimate est = WantAutoCa(prof) ? m_est : CaEstimate{};
        return Report(m_desc, prof, Map(m_desc, prof, est), est);
    }

    const char* GpuSource() const override {
        // The shared sampler (algo_util/resample.h) between the declarations
        // it uses and the kernel that calls it.
        static const std::string src = std::string(kLensHead) + kSampleCubicHlsl + kLensMain;
        return src.c_str();
    }
    static constexpr const char* kLensHead = R"(
Texture2D<float4>   Src : register(t0);
RWTexture2D<float4> Dst : register(u0);

cbuffer Params : register(b0) {
    uint Width;
    uint Height;
    uint FullWidth;
    uint FullHeight;
    uint OffsetX;
    uint OffsetY;
    uint D0Bits, D1Bits, D2Bits, D3Bits, D4Bits;   // green's source radius over r
    uint ScaleBits;
    uint R0Bits, R1Bits, R2Bits;                   // red's magnification at green's radius
    uint B0Bits, B1Bits, B2Bits;                   // ... blue's
    uint V1Bits, V2Bits, V3Bits;                   // the profile's falloff
    uint StopsBits, PowerBits;
    uint Linear;
};

float3 SrgbToLinear3(float3 c) {
    return select(c <= 0.04045, c / 12.92, pow((max(c, 0.0) + 0.055) / 1.055, 2.4));
}
float3 LinearToSrgb3(float3 c) {
    c = saturate(c);
    return select(c <= 0.0031308, c * 12.92, 1.055 * pow(c, 1.0 / 2.4) - 0.055);
}

)";
    static constexpr const char* kLensMain = R"([numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    float cx = 0.5 * float(FullWidth), cy = 0.5 * float(FullHeight);
    float hd = 0.5 * length(float2(FullWidth, FullHeight));
    float s = asfloat(ScaleBits);
    float2 off = float2(OffsetX, OffsetY);

    float2 p = float2(tid.xy) + off + 0.5;
    float2 u = (p - float2(cx, cy)) / hd * s;
    float r = length(u);
    float f = asfloat(D0Bits) + r * (asfloat(D1Bits) + r * (asfloat(D2Bits) +
              r * (asfloat(D3Bits) + r * asfloat(D4Bits))));
    float rg = r * f;
    float3 mag = float3(asfloat(R0Bits) + rg * (asfloat(R1Bits) + rg * asfloat(R2Bits)), 1.0,
                        asfloat(B0Bits) + rg * (asfloat(B1Bits) + rg * asfloat(B2Bits)));
    float2 g = float2(cx, cy) + u * f * hd;              // green's source
    float4 o;
    [unroll] for (uint c = 0; c < 3; ++c) {
        float2 sp = float2(cx, cy) + u * f * mag[c] * hd;
        o[c] = SampleCubic(sp - off - 0.5, c);
    }
    o.a = SampleCubic(g - off - 0.5, 3);

    float stops = asfloat(StopsBits);
    float3 v = float3(asfloat(V1Bits), asfloat(V2Bits), asfloat(V3Bits));
    if (stops != 0.0 || any(v != 0.0)) {
        float rs = length(g - float2(cx, cy)) / hd;
        float r2 = rs * rs;
        float kept = 1.0 + r2 * (v.x + r2 * (v.y + r2 * v.z));
        float gain = exp2(stops * pow(rs, asfloat(PowerBits))) / max(kept, 0.125);
        o.rgb = Linear != 0 ? o.rgb * gain : LinearToSrgb3(SrgbToLinear3(o.rgb) * gain);
    }
    Dst[tid.xy] = o;
}
)";

    std::vector<uint32_t> GpuConstants(int) const override {
        const LensProfile prof = Profile(m_desc);
        const LensMap m = Map(m_desc, prof, WantAutoCa(prof) ? m_est : CaEstimate{});
        std::vector<uint32_t> k = {uint32_t(m_desc.FullWidth()), uint32_t(m_desc.FullHeight()),
                                   uint32_t(m_desc.originX), uint32_t(m_desc.originY)};
        for (double c : m.d) k.push_back(FloatBits(float(c)));
        k.push_back(FloatBits(float(m.scale)));
        for (const auto& t : m.tca)
            for (double c : t) k.push_back(FloatBits(float(c)));
        for (double c : m.vig) k.push_back(FloatBits(float(c)));
        k.push_back(FloatBits(float(m_vignette)));
        k.push_back(FloatBits(float(m_vigPower)));
        k.push_back(m_desc.linear ? 1u : 0u);
        return k;
    }

private:
    // The database's profile for the lens that took this image, when
    // `profile` is on and one matches; cached in core/lensfun, so cheap.
    LensProfile Profile(const ImageDesc& d) const {
        if (!bool(m_profile) || d.focalMm <= 0.0f) return {};
        const LensTag& t = LensTagOf(d.lensTag);
        return FindLensProfile(t.make.c_str(), t.model.c_str(), t.lens.c_str(), d.focalMm,
                               d.aperture, d.crop35);
    }
    bool WantAutoCa(const LensProfile& p) const { return bool(m_autoCa) && !p.hasTca; }

    LensMap Map(const ImageDesc& d, const LensProfile& prof, const CaEstimate& est) const {
        LensMap m;
        const double W = std::max(1, d.FullWidth()), H = std::max(1, d.FullHeight());
        m.cx = 0.5 * W; m.cy = 0.5 * H;
        m.hd = 0.5 * std::hypot(W, H);
        if (prof.hasDistortion)
            for (int i = 0; i < 5; ++i) m.d[i] = prof.dist[i];
        m.d[2] += float(m_k1);
        m.d[4] += float(m_k2);
        const bool colour = !(d.format == Format::R32F);
        if (colour) {
            if (prof.hasTca)
                for (int c = 0; c < 2; ++c)
                    for (int i = 0; i < 3; ++i) m.tca[c][i] = prof.tca[c][i];
            m.tca[0][0] += est.a[0] + 1e-3 * double(float(m_caRed));
            m.tca[1][0] += est.a[1] + 1e-3 * double(float(m_caBlue));
        }
        if (prof.hasVignetting && bool(m_profileVig))
            for (int i = 0; i < 3; ++i) m.vig[i] = prof.vig[i];
        if (bool(m_fill)) m.scale = FitScale(m, W, H);
        return m;
    }

    std::string Report(const ImageDesc& d, const LensProfile& prof, const LensMap& m,
                       const CaEstimate& est) const {
        char b[512];
        int k = 0;
        if (bool(m_profile)) {
            if (prof.valid)
                k += std::snprintf(b + k, sizeof b - size_t(k),
                                   "profile: %s at %.0f mm f/%.1f (crop %.2f, calibrated at %.2f): %s%s%s; ",
                                   prof.lens.c_str(), prof.focal, prof.aperture, prof.cropFactor,
                                   prof.calibCrop, prof.hasDistortion ? "distortion " : "",
                                   prof.hasTca ? "TCA " : "",
                                   prof.hasVignetting ? (bool(m_profileVig) ? "vignetting" : "(vignetting off)") : "");
            else if (!LensTagOf(d.lensTag).lens.empty())
                k += std::snprintf(b + k, sizeof b - size_t(k), "profile: none for \"%s\"%s; ",
                                   LensTagOf(d.lensTag).lens.c_str(), LensfunLensCount() == 0 ? " (no lensfun database found)" : "");
            else
                k += std::snprintf(b + k, sizeof b - size_t(k), "profile: the file names no lens; ");
        }
        if (WantAutoCa(prof))
            k += std::snprintf(b + k, sizeof b - size_t(k),
                               "auto CA: red %+.2f, blue %+.2f per mille (%d / %d edges); ",
                               1e3 * est.a[0], 1e3 * est.a[1], est.n[0], est.n[1]);
        std::snprintf(b + k, sizeof b - size_t(k),
                      "corner: green from r %.4f, red x%.5f, blue x%.5f; fill scale %.4f",
                      m.F(1.0), m.Mag(0, m.F(1.0)), m.Mag(2, m.F(1.0)), m.scale);
        return b;
    }

    Param<bool> m_profile{this, "profile", true,
        "Look the lens up in the lensfun database (when installed) by the name the "
        "raw gives it, and correct its measured distortion, chromatic aberration "
        "and vignetting at this focal length and aperture. The sliders below add "
        "to the profile; the info panel names the lens it found."};
    Param<bool> m_profileVig{this, "profile_vignetting", true,
        "Use the profile's vignetting too. Off keeps the lens's own darkening of "
        "the corners while still straightening and de-fringing."};
    Param<float> m_k1{this, "k1", 0.0f, -0.5f, 0.5f,
        {.help = "The lens's radial distortion, r^2 term, with r = 1 at the corner: "
                 "negative is barrel (straight lines bow outward), positive is "
                 "pincushion. The correction undoes what is set here, on top of "
                 "any profile.",
         .step = 0.001}};
    Param<float> m_k2{this, "k2", 0.0f, -0.5f, 0.5f,
        {.help = "The r^4 term, for a moustache-shaped distortion that changes "
                 "direction towards the corners.",
         .step = 0.001}};
    Param<bool> m_autoCa{this, "auto_ca", true,
        "Measure lateral chromatic aberration -- red and blue fringes that grow "
        "toward the corners -- from the photograph's own edges and correct it. "
        "ca_red and ca_blue add to what is measured. Not used when the lens "
        "profile has its own chromatic aberration."};
    Param<float> m_caRed{this, "ca_red", 0.0f, -5.0f, 5.0f,
        {.help = "Red's magnification against green, in parts per thousand: "
                 "positive means the lens drew red larger, so red fringes sit "
                 "outside edges towards the corners.",
         .step = 0.01}};
    Param<float> m_caBlue{this, "ca_blue", 0.0f, -5.0f, 5.0f,
        {.help = "Blue's magnification against green, in parts per thousand.",
         .step = 0.01}};
    Param<float> m_vignette{this, "vignette", 0.0f, -2.0f, 3.0f,
        {.help = "Light the lens lost at the corners, in stops, given back. "
                 "Applied in linear light.",
         .step = 0.01}};
    Param<float> m_vigPower{this, "vignette_power", 2.0f, 1.0f, 8.0f,
        {.help = "How the loss grows with radius (stops * r^power): 2 for a "
                 "gentle, wide falloff, higher for one confined to the corners.",
         .step = 0.1}};
    Param<bool> m_fill{this, "fill", true,
        "Scale the result so its edges are always inside the photograph: no "
        "empty corners after removing pincushion, and the whole width used "
        "after removing barrel."};

    // Set by PrepareGpu and MeasureForGpu for the GPU path.
    ImageDesc m_desc{};
    CaEstimate m_est;
    bool m_measured = false;
    bool m_gpuRan = false;
};

REGISTER_ALGORITHM(LensCorrect);

}  // namespace
}  // namespace tglab
