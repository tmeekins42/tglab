// upright — perspective correction from the photograph's own lines.
//
// A camera tilted up at a building makes its verticals converge toward a
// vanishing point above the frame; rolled, it makes the horizon slope. Both
// are rotations of the camera, and a rotation is undone exactly by a
// homography, H = K R K^-1 -- the picture a camera turned back by R would have
// taken from the same spot. So: find the vanishing points, choose the rotation
// that puts them where an upright camera would see them, resample once.
//
//   1. LINE SEGMENTS, on a copy downscaled to ~1200 px: pixels whose gradient
//      is strong are grown into regions whose level-line direction agrees
//      (LSD-style, von Gioi et al.), each region fitted by PCA, and the long,
//      thin ones kept.
//   2. VANISHING POINTS by RANSAC over pairs of near-vertical (and, separately,
//      near-horizontal) segments -- each pair's lines meet at a candidate, and
//      the candidate the most segment length points at wins -- refined as the
//      point closest to all its inliers' lines (the smallest eigenvector of
//      sum w l l^T).
//   3. THE FOCAL LENGTH from the two vanishing points when both are finite (an
//      orthogonal pair satisfies v . h = -f^2 about the centre), else a
//      normal field of view. The verticals come out parallel whatever f is --
//      the vertical vanishing point goes to infinity either way -- and f only
//      sets how much the frame is stretched getting there.
//   4. THE ROTATION, by `mode`: level (roll only, so verticals are vertical on
//      average and nothing else moves), vertical (the vertical vanishing point
//      to infinity: converging verticals made parallel), or full (and the
//      dominant horizontal one too). `amount` scales it. `crop` fills the
//      frame with the largest view that has no empty corners.
//
// The detection is the expensive part and depends only on the photograph, so
// it is measured once and cached: dragging `amount` or switching `mode`
// resamples only. (lightcraft's Upright is the same pipeline:
// crates/pipeline/src/upright.rs.)
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "../../algo_util/color.h"
#include "../../algo_util/linalg.h"
#include "../../algo_util/pixel_buffer.h"
#include "../../algo_util/resample.h"
#include "../../core/algorithm.h"
#include "../../core/parallel.h"

namespace tglab {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr int    kDetectLong = 1200;     // the detector's working size
constexpr float  kGradMin = 0.02f;       // gradient below this is noise (display units)
constexpr double kTau = 22.5 * kPi / 180.0;   // region-growing angle tolerance
constexpr double kCandidate = 30.0 * kPi / 180.0;   // "near vertical/horizontal"
constexpr double kInlier = 1.5 * kPi / 180.0;       // VP inlier angle
constexpr double kDefaultF = 1.0;        // focal, in long edges, when not measured

struct Segment {
    double x0, y0, x1, y1;   // normalised: (px - centre) / long edge
    double len;
    double line[3];          // a x + b y + c = 0, (a, b) unit
};

// What the photograph says, cached: the vanishing points (homogeneous,
// normalised coordinates) and the focal they imply.
struct UprightFit {
    bool   hasV = false, hasH = false;
    double v[3] = {0, 1, 0}, h[3] = {1, 0, 0};
    double f = kDefaultF;
    bool   fMeasured = false;
    int    segments = 0, vIn = 0, hIn = 0;
};

// --- 3x3 helpers ----------------------------------------------------------

void Cross(const double* a, const double* b, double* o) {
    o[0] = a[1] * b[2] - a[2] * b[1];
    o[1] = a[2] * b[0] - a[0] * b[2];
    o[2] = a[0] * b[1] - a[1] * b[0];
}
void MatMul(const double* a, const double* b, double* o) {
    double t[9];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) {
            double s = 0;
            for (int k = 0; k < 3; ++k) s += a[r * 3 + k] * b[k * 3 + c];
            t[r * 3 + c] = s;
        }
    std::memcpy(o, t, sizeof t);
}
void Transpose(const double* a, double* o) {
    for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) o[c * 3 + r] = a[r * 3 + c];
}
// Rotation of `angle` about unit `axis` (Rodrigues).
void AxisAngle(const double* axis, double angle, double* R) {
    const double c = std::cos(angle), s = std::sin(angle), t = 1 - c;
    const double x = axis[0], y = axis[1], z = axis[2];
    const double m[9] = {t * x * x + c,     t * x * y - s * z, t * x * z + s * y,
                         t * x * y + s * z, t * y * y + c,     t * y * z - s * x,
                         t * x * z - s * y, t * y * z + s * x, t * z * z + c};
    std::memcpy(R, m, sizeof m);
}
// The rotation's axis and angle.
void ToAxisAngle(const double* R, double* axis, double* angle) {
    const double tr = std::clamp((R[0] + R[4] + R[8] - 1.0) * 0.5, -1.0, 1.0);
    *angle = std::acos(tr);
    double a[3] = {R[7] - R[5], R[2] - R[6], R[3] - R[1]};
    const double n = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    if (n < 1e-12) { axis[0] = 0; axis[1] = 0; axis[2] = 1; *angle = 0; return; }
    for (int i = 0; i < 3; ++i) axis[i] = a[i] / n;
}
// --- 1. line segments -----------------------------------------------------

std::vector<Segment> DetectSegments(const float* px, int w, int h, int ch, float vs, bool linear) {
    // Downscale by area to the detector's size, as display-like grey.
    const int L = std::max(w, h);
    const int k = std::max(1, (L + kDetectLong - 1) / kDetectLong);
    const int dw = std::max(1, w / k), dh = std::max(1, h / k);
    std::vector<float> g(size_t(dw) * size_t(dh));
    ParallelFor(size_t(dh), [&](size_t y) {
        for (int x = 0; x < dw; ++x) {
            double s = 0;
            for (int j = 0; j < k; ++j)
                for (int i = 0; i < k; ++i) {
                    const float* p = &px[((y * size_t(k) + size_t(j)) * size_t(w) + size_t(x * k + i)) * size_t(ch)];
                    const float Y = ch >= 3 ? Luma(p[0], p[1], p[2]) / vs : p[0] / vs;
                    s += linear ? LinearToSrgb(std::min(Y, 1.0f)) : Y;
                }
            g[y * size_t(dw) + size_t(x)] = float(s / double(k * k));
        }
    });
    // Gradient (Sobel) and level-line angle.
    const size_t n = size_t(dw) * size_t(dh);
    std::vector<float> mag(n, 0.0f), ang(n, 0.0f);
    ParallelFor(size_t(dh), [&](size_t yy) {
        const int y = int(yy);
        if (y == 0 || y == dh - 1) return;
        for (int x = 1; x < dw - 1; ++x) {
            auto G = [&](int xx, int yy2) { return g[size_t(yy2) * size_t(dw) + size_t(xx)]; };
            const float gx = (G(x + 1, y - 1) + 2 * G(x + 1, y) + G(x + 1, y + 1) -
                              G(x - 1, y - 1) - 2 * G(x - 1, y) - G(x - 1, y + 1)) * 0.25f;
            const float gy = (G(x - 1, y + 1) + 2 * G(x, y + 1) + G(x + 1, y + 1) -
                              G(x - 1, y - 1) - 2 * G(x, y - 1) - G(x + 1, y - 1)) * 0.25f;
            const size_t i = size_t(y) * size_t(dw) + size_t(x);
            mag[i] = std::hypot(gx, gy);
            ang[i] = std::atan2(gx, -gy);
        }
    });
    std::vector<uint32_t> order;
    for (size_t i = 0; i < n; ++i) if (mag[i] > kGradMin) order.push_back(uint32_t(i));
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return mag[a] > mag[b]; });

    std::vector<uint8_t> used(n, 0);
    std::vector<uint32_t> region;
    std::vector<Segment> segs;
    const double minLen = std::max(20.0, 0.04 * double(std::max(dw, dh)));
    auto angDiff = [](double a, double b) {
        double d = std::fmod(a - b, 2 * kPi);
        if (d > kPi) d -= 2 * kPi;
        if (d < -kPi) d += 2 * kPi;
        return std::fabs(d);
    };
    for (uint32_t seed : order) {
        if (used[seed]) continue;
        region.clear();
        region.push_back(seed);
        used[seed] = 1;
        double sx = std::cos(ang[seed]), sy = std::sin(ang[seed]);
        double ra = ang[seed];
        for (size_t qi = 0; qi < region.size(); ++qi) {
            const int x = int(region[qi] % uint32_t(dw)), y = int(region[qi] / uint32_t(dw));
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    const int xx = x + dx, yy = y + dy;
                    if (xx < 1 || yy < 1 || xx >= dw - 1 || yy >= dh - 1) continue;
                    const uint32_t j = uint32_t(yy) * uint32_t(dw) + uint32_t(xx);
                    if (used[j] || mag[j] <= kGradMin || angDiff(ang[j], ra) > kTau) continue;
                    used[j] = 1;
                    region.push_back(j);
                    sx += std::cos(ang[j]); sy += std::sin(ang[j]);
                    ra = std::atan2(sy, sx);
                }
        }
        if (region.size() < 12) continue;
        // PCA, weighted by gradient.
        double W = 0, mx = 0, my = 0;
        for (uint32_t j : region) {
            const double wgt = mag[j];
            W += wgt; mx += wgt * double(j % uint32_t(dw)); my += wgt * double(j / uint32_t(dw));
        }
        mx /= W; my /= W;
        double cxx = 0, cyy = 0, cxy = 0;
        for (uint32_t j : region) {
            const double wgt = mag[j], ux = double(j % uint32_t(dw)) - mx, uy = double(j / uint32_t(dw)) - my;
            cxx += wgt * ux * ux; cyy += wgt * uy * uy; cxy += wgt * ux * uy;
        }
        const double th = 0.5 * std::atan2(2 * cxy, cxx - cyy);
        const double ex = std::cos(th), ey = std::sin(th);
        double tmin = 1e30, tmax = -1e30, wmax = 0;
        for (uint32_t j : region) {
            const double ux = double(j % uint32_t(dw)) - mx, uy = double(j / uint32_t(dw)) - my;
            const double t = ux * ex + uy * ey, pdist = std::fabs(-ux * ey + uy * ex);
            tmin = std::min(tmin, t); tmax = std::max(tmax, t); wmax = std::max(wmax, pdist);
        }
        const double len = tmax - tmin;
        if (len < minLen || 2 * wmax > std::max(3.0, 0.08 * len)) continue;
        // To normalised coordinates of the full frame.
        auto norm = [&](double x, double y, double* ox, double* oy) {
            *ox = ((x + 0.5) * double(w) / double(dw) - 0.5 * double(w)) / double(L);
            *oy = ((y + 0.5) * double(h) / double(dh) - 0.5 * double(h)) / double(L);
        };
        Segment s;
        norm(mx + ex * tmin, my + ey * tmin, &s.x0, &s.y0);
        norm(mx + ex * tmax, my + ey * tmax, &s.x1, &s.y1);
        s.len = std::hypot(s.x1 - s.x0, s.y1 - s.y0);
        const double p0[3] = {s.x0, s.y0, 1}, p1[3] = {s.x1, s.y1, 1};
        Cross(p0, p1, s.line);
        const double ln = std::hypot(s.line[0], s.line[1]);
        if (ln < 1e-12) continue;
        for (double& v : s.line) v /= ln;
        segs.push_back(s);
    }
    return segs;
}

// --- 2. vanishing points --------------------------------------------------

// The angle between segment s and the direction from its midpoint to v.
double VpAngle(const Segment& s, const double* v) {
    const double mx = 0.5 * (s.x0 + s.x1), my = 0.5 * (s.y0 + s.y1);
    double dx, dy;
    const double vn = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (std::fabs(v[2]) < 1e-9 * vn) { dx = v[0]; dy = v[1]; }
    else { dx = v[0] / v[2] - mx; dy = v[1] / v[2] - my; }
    const double sx = s.x1 - s.x0, sy = s.y1 - s.y0;
    const double c = std::fabs(dx * sx + dy * sy) / (std::hypot(dx, dy) * std::hypot(sx, sy) + 1e-30);
    return std::acos(std::min(1.0, c));
}

bool FindVp(const std::vector<Segment>& segs, bool vertical, double* vp, int* inliers) {
    std::vector<int> cand;
    for (size_t i = 0; i < segs.size(); ++i) {
        const double dx = segs[i].x1 - segs[i].x0, dy = segs[i].y1 - segs[i].y0;
        const double off = vertical ? std::atan2(std::fabs(dx), std::fabs(dy))
                                    : std::atan2(std::fabs(dy), std::fabs(dx));
        if (off < kCandidate) cand.push_back(int(i));
    }
    if (cand.size() < 3) return false;
    std::vector<double> cum(cand.size());
    double tot = 0;
    for (size_t i = 0; i < cand.size(); ++i) { tot += segs[size_t(cand[i])].len; cum[i] = tot; }
    std::mt19937 rng(12345u);
    std::uniform_real_distribution<double> U(0.0, tot);
    auto pick = [&] { return int(std::lower_bound(cum.begin(), cum.end(), U(rng)) - cum.begin()); };
    double best = 0, bestV[3] = {0, 0, 0};
    for (int it = 0; it < 800; ++it) {
        const int a = pick(), b = pick();
        if (a == b) continue;
        double v[3];
        Cross(segs[size_t(cand[size_t(a)])].line, segs[size_t(cand[size_t(b)])].line, v);
        if (std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) < 1e-12) continue;
        double score = 0;
        for (int i : cand) if (VpAngle(segs[size_t(i)], v) < kInlier) score += segs[size_t(i)].len;
        if (score > best) { best = score; std::memcpy(bestV, v, sizeof v); }
    }
    if (best <= 0) return false;
    // Refine on the inliers: the point closest to all their lines.
    double M[9] = {0};
    int nIn = 0;
    double inLen = 0;
    for (int i : cand) {
        const Segment& s = segs[size_t(i)];
        if (VpAngle(s, bestV) >= kInlier) continue;
        ++nIn; inLen += s.len;
        for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) M[r * 3 + c] += s.len * s.line[r] * s.line[c];
    }
    if (nIn < 3 || inLen < 0.3) return false;
    linalg::SmallestEigenvector<3>(M, vp);
    *inliers = nIn;
    return true;
}

UprightFit Fit(const float* px, int w, int h, int ch, float vs, bool linear) {
    UprightFit f;
    const std::vector<Segment> segs = DetectSegments(px, w, h, ch, vs, linear);
    f.segments = int(segs.size());
    f.hasV = FindVp(segs, true, f.v, &f.vIn);
    f.hasH = FindVp(segs, false, f.h, &f.hIn);
    // The focal from an orthogonal pair -- but only when both points are near
    // enough to be well placed. A vertical vanishing point many frames away
    // (verticals that barely converge) is placed poorly, and f^2 = -v.h
    // amplifies that: on a courtyard shot it read 1.84 long edges for a lens
    // of 0.9, and the correction then overstated the tilt twofold.
    if (f.hasV && f.hasH && std::fabs(f.v[2]) > 1e-9 && std::fabs(f.h[2]) > 1e-9) {
        const double vx = f.v[0] / f.v[2], vy = f.v[1] / f.v[2];
        const double hx = f.h[0] / f.h[2], hy = f.h[1] / f.h[2];
        const double f2 = -(vx * hx + vy * hy);
        if (std::hypot(vx, vy) < 3.0 && std::hypot(hx, hy) < 3.0 && f2 > 0.4 * 0.4 && f2 < 2.5 * 2.5) {
            f.f = std::sqrt(f2);
            f.fMeasured = true;
        }
    }
    return f;
}

// --- 3/4. the rotation and the map ----------------------------------------

// The rotation to apply (source rays -> output rays) for `mode`, scaled by
// `amount`. Identity when there is nothing to correct by.
void Rotation(const UprightFit& fit, int mode, double amount, double* R) {
    const double I[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    std::memcpy(R, I, sizeof I);
    if (mode == 0) return;
    const double fl = fit.f;
    auto dir = [&](const double* p, double* d) {   // a vanishing point as a ray
        d[0] = p[0]; d[1] = p[1]; d[2] = p[2] * fl;
        const double n = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        for (int i = 0; i < 3; ++i) d[i] /= n;
    };
    if (mode == 1) {
        // Level: roll only, from the verticals if there are any, else the
        // horizontals.
        double ang = 0;
        if (fit.hasV) {
            double d[3]; dir(fit.v, d);
            if (d[1] < 0) for (double& x : d) x = -x;
            ang = std::atan2(d[0], d[1]);   // how far "down" leans toward +x
        } else if (fit.hasH) {
            double d[3]; dir(fit.h, d);
            if (d[0] < 0) for (double& x : d) x = -x;
            ang = -std::atan2(d[1], d[0]);
        } else {
            return;
        }
        const double z[3] = {0, 0, 1};
        AxisAngle(z, ang * amount, R);
        return;
    }
    if (!fit.hasV) return;
    // Vertical: the smallest rotation taking the vertical vanishing ray onto
    // the image's own vertical axis.
    double d[3]; dir(fit.v, d);
    const double t[3] = {0, d[1] >= 0 ? 1.0 : -1.0, 0};
    double ax[3]; Cross(d, t, ax);
    const double an = std::sqrt(ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2]);
    double R1[9];
    std::memcpy(R1, I, sizeof I);
    if (an > 1e-12) {
        for (double& x : ax) x /= an;
        AxisAngle(ax, std::acos(std::clamp(d[0] * t[0] + d[1] * t[1] + d[2] * t[2], -1.0, 1.0)), R1);
    }
    double Rt[9];
    std::memcpy(Rt, R1, sizeof Rt);
    if (mode == 3 && fit.hasH) {
        // Full: then about the (now true) vertical axis, so the dominant
        // horizontal vanishing ray lies in the image plane -- its lines parallel.
        double hd[3]; dir(fit.h, hd);
        double h2[3] = {R1[0] * hd[0] + R1[1] * hd[1] + R1[2] * hd[2],
                        R1[3] * hd[0] + R1[4] * hd[1] + R1[5] * hd[2],
                        R1[6] * hd[0] + R1[7] * hd[1] + R1[8] * hd[2]};
        if (h2[0] < 0) for (double& x : h2) x = -x;
        const double y[3] = {0, 1, 0};
        double Ry[9];
        AxisAngle(y, std::atan2(h2[2], h2[0]), Ry);
        MatMul(Ry, R1, Rt);
    }
    // Scale by amount.
    double axis[3], angle;
    ToAxisAngle(Rt, axis, &angle);
    AxisAngle(axis, angle * amount, R);
}

// Output pixel (full frame, pixel centres at +0.5) -> source pixel, as a 3x3:
// to normalised, scaled by s, RE-CENTRED, through K R^T K^-1, back to pixels.
//
// Re-centred because a rotation of the camera also moves the picture: turned
// 10 degrees, the source's centre lands f tan(10) from the output's -- a sixth
// of the frame -- and most of the view is then off the photograph. The output
// is shifted so the source's centre stays at the centre.
void BuildMap(const double* R, double f, double s, int W, int H, double* M) {
    const double L = double(std::max(W, H)), cx = 0.5 * W, cy = 0.5 * H;
    // Where the source's centre ray (0, 0, 1) lands in the output, normalised.
    const double ox = R[8] > 1e-9 ? f * R[2] / R[8] : 0.0, oy = R[8] > 1e-9 ? f * R[5] / R[8] : 0.0;
    const double toN[9] = {s / L, 0, -s * cx / L + ox, 0, s / L, -s * cy / L + oy, 0, 0, 1};
    const double Kinv[9] = {1 / f, 0, 0, 0, 1 / f, 0, 0, 0, 1};
    const double K[9] = {f, 0, 0, 0, f, 0, 0, 0, 1};
    const double toP[9] = {L, 0, cx, 0, L, cy, 0, 0, 1};
    double Rt[9]; Transpose(R, Rt);
    double a[9];
    MatMul(Kinv, toN, a);
    MatMul(Rt, a, a);
    MatMul(K, a, a);
    MatMul(toP, a, M);
}

bool MapPoint(const double* M, double x, double y, double* sx, double* sy) {
    const double z = M[6] * x + M[7] * y + M[8];
    if (z <= 1e-9) return false;
    *sx = (M[0] * x + M[1] * y + M[2]) / z;
    *sy = (M[3] * x + M[4] * y + M[5]) / z;
    return true;
}

// The largest scale at which the output's whole border lands in the source.
double CropScale(const double* R, double f, int W, int H) {
    bool identity = true;
    for (int i = 0; i < 9; ++i) if (std::fabs(R[i] - ((i % 4 == 0) ? 1.0 : 0.0)) > 1e-12) identity = false;
    if (identity) return 1.0;
    auto inside = [&](double s) {
        double M[9];
        BuildMap(R, f, s, W, H, M);
        constexpr int kSteps = 64;
        for (int i = 0; i <= kSteps; ++i) {
            const double t = double(i) / kSteps;
            const double pts[4][2] = {{t * W, 0}, {t * W, double(H)}, {0, t * H}, {double(W), t * H}};
            for (const auto& p : pts) {
                double sx, sy;
                if (!MapPoint(M, p[0], p[1], &sx, &sy)) return false;
                if (sx < -1e-3 || sy < -1e-3 || sx > W + 1e-3 || sy > H + 1e-3) return false;
            }
        }
        return true;
    };
    double lo = 0.1, hi = 1.0;
    if (inside(hi)) return hi;
    if (!inside(lo)) return lo;
    for (int it = 0; it < 40; ++it) { const double m = 0.5 * (lo + hi); (inside(m) ? lo : hi) = m; }
    return lo;
}

const char* const kUprightHead = R"(
Texture2D<float4>   Src : register(t0);
RWTexture2D<float4> Dst : register(u0);
cbuffer Params : register(b0) {
    uint Width;
    uint Height;
    uint M0, M1, M2, M3, M4, M5, M6, M7, M8;   // output px -> source px
    uint OffsetX, OffsetY;
};
)";
const char* const kUprightMain = R"(
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    float2 off = float2(OffsetX, OffsetY);
    float2 p = float2(tid.xy) + off + 0.5;
    float z = asfloat(M6) * p.x + asfloat(M7) * p.y + asfloat(M8);
    float4 o = float4(0, 0, 0, 1);
    if (z > 1e-9) {
        float2 s = float2(asfloat(M0) * p.x + asfloat(M1) * p.y + asfloat(M2),
                          asfloat(M3) * p.x + asfloat(M4) * p.y + asfloat(M5)) / z;
        // The source frame is this raster (upright runs whole, never on a region).
        if (s.x >= -0.5 && s.y >= -0.5 && s.x <= float(Width) + 0.5 && s.y <= float(Height) + 0.5) {
            float2 q = s - off - 0.5;
            [unroll] for (uint c = 0; c < 4; ++c) o[c] = SampleCubic(q, c);
        }
    }
    Dst[tid.xy] = o;
}
)";

class Upright : public AlgorithmBase {
public:
    const char* Name()     const override { return "upright"; }
    const char* Category() const override { return "adjust"; }

    PortList Inputs()  const override { return {{"src", DataType::Image, FormatSpec::Any}}; }
    PortList Outputs() const override { return {{"out", DataType::Image, FormatSpec::SameAsInput}}; }

    // A warp, chosen from lines anywhere in the frame.
    bool RegionSafe() const override { return false; }
    bool IsNoOp() const override { return int(m_mode) == 0 || float(m_amount) == 0.0f; }

    void RunCPU(RunCtx& ctx) override {
        const ImageView src = ctx.In(0);
        ImageView dst = ctx.Out(0);
        if (!src.Valid() || !dst.Valid()) return;
        PixelBuffer in, out;
        in.Unpack(src);
        if (!in.Valid()) return;
        out.AllocLike(in);
        const int w = in.Width(), h = in.Height(), ch = in.Channels();
        const float* px = in.Data().data();
        const UprightFit fit = Fit(px, w, h, ch, in.ValueScale(), ctx.InDesc(0).linear);
        double R[9], M[9];
        Rotation(fit, int(m_mode), double(float(m_amount)), R);
        const double s = bool(m_crop) ? CropScale(R, fit.f, w, h) : 1.0;
        BuildMap(R, fit.f, s, w, h, M);
        float* o = out.Data().data();
        ParallelFor(size_t(h), [&](size_t y) {
            for (int x = 0; x < w; ++x) {
                float* q = &o[(y * size_t(w) + size_t(x)) * size_t(ch)];
                double sx, sy;
                const bool ok = MapPoint(M, x + 0.5, double(y) + 0.5, &sx, &sy) &&
                                sx >= -0.5 && sy >= -0.5 && sx <= w + 0.5 && sy <= h + 0.5;
                for (int c = 0; c < ch; ++c)
                    q[c] = ok ? SampleCubicClamped(px, w, h, ch, c, sx - 0.5, sy - 0.5)
                              : (c == 3 ? in.ValueScale() : 0.0f);
            }
        });
        out.PackInto(dst);
        ctx.SetReport(Report(fit, R, s));
    }

    // --- GPU: the resample, from a measurement made once --------------------
    bool HasGPU() const override { return true; }
    bool GpuNeedsInputPixels() const override { return true; }
    void PrepareGpu(const std::vector<ImageDesc>& inputs) override {
        if (!inputs.empty()) m_desc = inputs[0];
        m_gpuRan = true;
    }
    void MeasureForGpu(const std::vector<const Image*>& inputs) override {
        m_fit = UprightFit{};
        if (inputs.empty() || !inputs[0]) return;
        ImageView v = const_cast<Image*>(inputs[0])->MapCpuRead();
        if (!v.data) return;
        PixelBuffer buf;
        buf.Unpack(v);
        if (!buf.Valid()) return;
        m_fit = Fit(buf.Data().data(), buf.Width(), buf.Height(), buf.Channels(), buf.ValueScale(),
                    v.desc.linear);
        m_measured = true;
    }
    bool MeasurementDependsOnlyOnInput() const override { return true; }
    std::vector<uint32_t> SaveMeasurement() const override {
        if (!m_measured) return {};
        std::vector<uint32_t> b = {m_fit.hasV ? 1u : 0u, m_fit.hasH ? 1u : 0u, m_fit.fMeasured ? 1u : 0u,
                                   uint32_t(m_fit.segments), uint32_t(m_fit.vIn), uint32_t(m_fit.hIn)};
        for (double d : {m_fit.v[0], m_fit.v[1], m_fit.v[2], m_fit.h[0], m_fit.h[1], m_fit.h[2], m_fit.f})
            b.push_back(FloatBits(float(d)));
        return b;
    }
    bool RestoreMeasurement(const std::vector<uint32_t>& b) override {
        if (b.size() != 13) return false;
        auto fl = [&](size_t i) { float f; std::memcpy(&f, &b[i], 4); return double(f); };
        m_fit.hasV = b[0] != 0; m_fit.hasH = b[1] != 0; m_fit.fMeasured = b[2] != 0;
        m_fit.segments = int(b[3]); m_fit.vIn = int(b[4]); m_fit.hIn = int(b[5]);
        for (int i = 0; i < 3; ++i) { m_fit.v[i] = fl(6 + size_t(i)); m_fit.h[i] = fl(9 + size_t(i)); }
        m_fit.f = fl(12);
        m_measured = true;
        return true;
    }

    const char* GpuSource() const override {
        static const std::string src = std::string(kUprightHead) + kSampleCubicHlsl + kUprightMain;
        return src.c_str();
    }
    std::vector<uint32_t> GpuConstants(int) const override {
        double R[9], M[9];
        Rotation(m_fit, int(m_mode), double(float(m_amount)), R);
        const int W = std::max(1, m_desc.width), H = std::max(1, m_desc.height);
        const double s = bool(m_crop) ? CropScale(R, m_fit.f, W, H) : 1.0;
        BuildMap(R, m_fit.f, s, W, H, M);
        std::vector<uint32_t> c;
        for (double v : M) c.push_back(FloatBits(float(v)));
        c.push_back(uint32_t(m_desc.originX));
        c.push_back(uint32_t(m_desc.originY));
        return c;
    }
    std::string RunReport() const override {
        if (!m_gpuRan) return {};
        double R[9];
        Rotation(m_fit, int(m_mode), double(float(m_amount)), R);
        const int W = std::max(1, m_desc.width), H = std::max(1, m_desc.height);
        return Report(m_fit, R, bool(m_crop) ? CropScale(R, m_fit.f, W, H) : 1.0);
    }

private:
    std::string Report(const UprightFit& fit, const double* R, double s) const {
        char b[256];
        // The correction as the tilt (about x), swing (about y) and roll (about
        // z) a photographer would name.
        const double tilt = std::atan2(R[7], R[8]) * 180 / kPi;
        const double swing = std::asin(std::clamp(-R[6], -1.0, 1.0)) * 180 / kPi;
        const double roll = std::atan2(R[3], R[0]) * 180 / kPi;
        std::snprintf(b, sizeof b,
                      "%d segments; vertical VP %s (%d), horizontal VP %s (%d); focal %.2f long edges%s; "
                      "corrected tilt %.1f, swing %.1f, roll %.1f deg; crop %.0f%%",
                      fit.segments, fit.hasV ? "found" : "none", fit.vIn, fit.hasH ? "found" : "none", fit.hIn,
                      fit.f, fit.fMeasured ? " (measured)" : " (assumed)", tilt, swing, roll, 100.0 * s);
        return b;
    }

    // Off until chosen: a photograph with no architecture in it has no
    // trustworthy verticals, and a correction made from the wrong lines is a
    // warp nobody asked for.
    static constexpr const char* kModeNames[] = {"off", "level", "vertical", "full"};
    Param<int> m_mode{this, "mode", 0, 0, 3,
        {.help = "off; level (roll only); vertical (converging verticals made "
                 "parallel, and level); full (the dominant horizontals too).",
         .step = 1,
         .choices = kModeNames, .choiceCount = 4}};
    Param<float> m_amount{this, "amount", 1.0f, 0.0f, 1.0f,
        {.help = "How much of the correction to apply. A full vertical correction "
                 "can look stretched; architects often leave a little convergence.",
         .step = 0.01}};
    Param<bool> m_crop{this, "crop", true,
        "Fill the frame with the largest view that has no empty corners. Off: "
        "the whole corrected frame, with blank wedges where the camera saw nothing."};

    ImageDesc m_desc{};       // set by PrepareGpu
    UprightFit m_fit;         // measured or restored for the GPU path
    bool m_measured = false, m_gpuRan = false;
};

REGISTER_ALGORITHM(Upright);

}  // namespace
}  // namespace tglab
