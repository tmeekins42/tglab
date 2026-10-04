#include "splat_raster.h"

#include <algorithm>
#include <chrono>
#include <cmath>

#include "../core/parallel.h"

namespace tglab {
namespace {

using Clock = std::chrono::steady_clock;
inline double MsSince(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}


// Row-major 3x3 helpers. Small enough to write out rather than pull in a
// matrix type the rest of the file would have to convert to and from.
inline void Mul33(const double* A, const double* B, double* C) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double s = 0.0;
            for (int k = 0; k < 3; ++k) s += A[i * 3 + k] * B[k * 3 + j];
            C[i * 3 + j] = s;
        }
}
inline void Transpose33(const double* A, double* T) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) T[j * 3 + i] = A[i * 3 + j];
}

}  // namespace

SplatParam ToParam(const Splat& s) {
    SplatParam p;
    p.mean[0] = s.mean.x; p.mean[1] = s.mean.y; p.mean[2] = s.mean.z;
    p.logScale[0] = std::log(std::max(1e-12, s.scale.x));
    p.logScale[1] = std::log(std::max(1e-12, s.scale.y));
    p.logScale[2] = std::log(std::max(1e-12, s.scale.z));
    for (int i = 0; i < 4; ++i) p.quat[i] = s.rot[i];
    const double o = std::clamp(s.opacity, 1e-4, 1.0 - 1e-4);
    p.opacity = Logit(o);
    p.color[0] = s.color.x; p.color[1] = s.color.y; p.color[2] = s.color.z;
    return p;
}

Splat FromParam(const SplatParam& p) {
    Splat s;
    s.mean = Vec3{p.mean[0], p.mean[1], p.mean[2]};
    s.scale = Vec3{std::exp(p.logScale[0]), std::exp(p.logScale[1]),
                   std::exp(p.logScale[2])};
    double n = 0.0;
    for (int i = 0; i < 4; ++i) n += p.quat[i] * p.quat[i];
    n = std::sqrt(n);
    for (int i = 0; i < 4; ++i) s.rot[i] = (n > 1e-12) ? p.quat[i] / n : (i == 0);
    s.opacity = Sigmoid(p.opacity);
    s.color = Vec3{p.color[0], p.color[1], p.color[2]};
    return s;
}

// --- projection ----------------------------------------------------------------

void SplatRaster::Project(const std::vector<SplatParam>& splats,
                          const SplatCam& cam, const RasterOptions& opt) {
    const size_t n = splats.size();
    m_proj.assign(n, Proj{});
    m_tilesX = (cam.w + kTile - 1) / kTile;
    m_tilesY = (cam.h + kTile - 1) / kTile;
    const double* W = cam.R.m;

    ParallelFor(n, [&](size_t i) {
        const SplatParam& s = splats[i];
        Proj& P = m_proj[i];

        // Camera-space centre. Too close or behind: not drawn. The near limit
        // is absolute rather than scene-relative because the Jacobian's 1/z^2
        // terms, not the scene's scale, are what go wrong near zero.
        for (int r = 0; r < 3; ++r)
            P.tc[r] = W[r * 3 + 0] * s.mean[0] + W[r * 3 + 1] * s.mean[1] +
                      W[r * 3 + 2] * s.mean[2] + (&cam.t.x)[r];
        const double x = P.tc[0], y = P.tc[1], z = P.tc[2];
        if (z < 1e-2) return;

        P.opac = Sigmoid(s.opacity);
        if (P.opac < opt.minAlpha) return;

        // Rotation from the normalised quaternion, as Splat::Rotation.
        double len = 0.0;
        for (int k = 0; k < 4; ++k) len += s.quat[k] * s.quat[k];
        len = std::sqrt(len);
        if (len < 1e-12) { P.qn[0] = 1; P.qn[1] = P.qn[2] = P.qn[3] = 0; len = 1.0; }
        else for (int k = 0; k < 4; ++k) P.qn[k] = s.quat[k] / len;
        P.qlen = len;
        const double qw = P.qn[0], qx = P.qn[1], qy = P.qn[2], qz = P.qn[3];
        double* R = P.Rq;
        R[0] = 1 - 2 * (qy * qy + qz * qz); R[1] = 2 * (qx * qy - qw * qz);     R[2] = 2 * (qx * qz + qw * qy);
        R[3] = 2 * (qx * qy + qw * qz);     R[4] = 1 - 2 * (qx * qx + qz * qz); R[5] = 2 * (qy * qz - qw * qx);
        R[6] = 2 * (qx * qz - qw * qy);     R[7] = 2 * (qy * qz + qw * qx);     R[8] = 1 - 2 * (qx * qx + qy * qy);

        for (int k = 0; k < 3; ++k) P.scale[k] = std::exp(s.logScale[k]);

        // Sigma = R S S^T R^T.
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) {
                double v = 0.0;
                for (int k = 0; k < 3; ++k)
                    v += R[a * 3 + k] * P.scale[k] * P.scale[k] * R[b * 3 + k];
                P.sigma[a * 3 + b] = v;
            }

        // M = W Sigma W^T: the covariance in camera space.
        double WS[9], Wt[9];
        Mul33(W, P.sigma, WS);
        Transpose33(W, Wt);
        Mul33(WS, Wt, P.M);

        // The Jacobian of (fx x/z + cx, fy y/z + cy).
        const double iz = 1.0 / z, iz2 = iz * iz;
        double* J = P.J;
        J[0] = cam.fx * iz; J[1] = 0.0;         J[2] = -cam.fx * x * iz2;
        J[3] = 0.0;         J[4] = cam.fy * iz; J[5] = -cam.fy * y * iz2;

        // cov2 = J M J^T + lowPass I.
        double JM[6];
        for (int r = 0; r < 2; ++r)
            for (int c = 0; c < 3; ++c) {
                double v = 0.0;
                for (int k = 0; k < 3; ++k) v += J[r * 3 + k] * P.M[k * 3 + c];
                JM[r * 3 + c] = v;
            }
        auto jmjt = [&](int r, int c) {
            double v = 0.0;
            for (int k = 0; k < 3; ++k) v += JM[r * 3 + k] * J[c * 3 + k];
            return v;
        };
        const double A = jmjt(0, 0) + opt.lowPass;
        const double B = jmjt(0, 1);
        const double C = jmjt(1, 1) + opt.lowPass;
        const double det = A * C - B * B;
        if (!(det > 1e-12)) return;
        P.cov2[0] = A; P.cov2[1] = B; P.cov2[2] = C;
        P.conic[0] = C / det; P.conic[1] = -B / det; P.conic[2] = A / det;

        P.u = cam.fx * x * iz + cam.cx;
        P.v = cam.fy * y * iz + cam.cy;
        P.depth = z;

        // HOW FAR OUT IT CAN MATTER: where opacity * G falls below minAlpha,
        // along the ellipse's long axis. Tied to the opacity rather than a
        // fixed three sigma, so the skip test in the compositing loop -- not
        // an arbitrary cut-off -- is the only place a Gaussian ends. A fixed
        // cut-off inside that radius would make the rendered image jump as a
        // Gaussian moves, which is a discontinuity no gradient can describe.
        const double mid = 0.5 * (A + C);
        const double lmax = mid + std::sqrt(std::max(0.0, mid * mid - det));
        P.radius = 3.0 * std::sqrt(lmax);
        const double reach2 = 2.0 * std::log(P.opac / opt.minAlpha);
        if (!(reach2 > 0.0)) return;
        const double r = std::sqrt(reach2 * lmax) + 1.0;
        P.reach = r;
        if (P.u + r < 0 || P.v + r < 0 || P.u - r >= cam.w || P.v - r >= cam.h)
            return;
        P.x0 = std::clamp(int(std::floor((P.u - r) / kTile)), 0, m_tilesX - 1);
        P.x1 = std::clamp(int(std::floor((P.u + r) / kTile)), 0, m_tilesX - 1);
        P.y0 = std::clamp(int(std::floor((P.v - r) / kTile)), 0, m_tilesY - 1);
        P.y1 = std::clamp(int(std::floor((P.v + r) / kTile)), 0, m_tilesY - 1);
        P.ok = true;
    });
}

void SplatRaster::BinTiles(const SplatCam&) {
    m_tiles.assign(size_t(m_tilesX) * size_t(m_tilesY), {});
    m_visible = 0;
    m_screenRadius.assign(m_proj.size(), -1.0);
    for (size_t i = 0; i < m_proj.size(); ++i)
        if (m_proj[i].ok) m_screenRadius[i] = m_proj[i].radius;
    for (size_t i = 0; i < m_proj.size(); ++i) {
        const Proj& P = m_proj[i];
        if (!P.ok) continue;
        ++m_visible;
        for (int ty = P.y0; ty <= P.y1; ++ty)
            for (int tx = P.x0; tx <= P.x1; ++tx)
                m_tiles[size_t(ty) * size_t(m_tilesX) + size_t(tx)].push_back(int(i));
    }
    // Near to far within each tile. Stable, so equal depths composite in a
    // fixed order and a render is reproducible to the bit.
    ParallelFor(m_tiles.size(), [&](size_t t) {
        std::stable_sort(m_tiles[t].begin(), m_tiles[t].end(), [&](int a, int b) {
            return m_proj[size_t(a)].depth < m_proj[size_t(b)].depth;
        });
    });
}

// --- forward ---------------------------------------------------------------------

void SplatRaster::Forward(const std::vector<SplatParam>& splats,
                          const SplatCam& cam, const RasterOptions& opt,
                          std::vector<double>* rgb, std::vector<double>* depth) {
    auto clk = Clock::now();
    Project(splats, cam, opt);
    m_time.project += MsSince(clk);
    clk = Clock::now();
    BinTiles(cam);
    m_time.bin += MsSince(clk);
    clk = Clock::now();

    // On the device when there is one; the CPU loop below otherwise, or if
    // the device fails -- in which case nothing it wrote is used.
    m_usedGpu = false;
    if (m_gpu && CompositeGpu(splats, cam, opt, rgb, depth)) {
        m_usedGpu = true;
        m_time.composite += MsSince(clk);
        return;
    }

    const size_t np = size_t(cam.w) * size_t(cam.h);
    rgb->assign(np * 3, 0.0);
    if (depth) depth->assign(np, 0.0);
    m_finalT.assign(np, 1.0);
    m_lastIdx.assign(np, 0);

    ParallelFor(m_tiles.size(), [&](size_t t) {
        const std::vector<int>& list = m_tiles[t];
        const int tx = int(t % size_t(m_tilesX)), ty = int(t / size_t(m_tilesX));
        for (int py = ty * kTile; py < std::min(cam.h, (ty + 1) * kTile); ++py)
            for (int px = tx * kTile; px < std::min(cam.w, (tx + 1) * kTile); ++px) {
                double T = 1.0, c0 = 0, c1 = 0, c2 = 0, cd = 0;
                int last = 0;
                for (int k = 0; k < int(list.size()); ++k) {
                    const Proj& P = m_proj[size_t(list[size_t(k)])];
                    const double dx = double(px) - P.u, dy = double(py) - P.v;
                    const double power = -0.5 * (P.conic[0] * dx * dx +
                                                 P.conic[2] * dy * dy) -
                                         P.conic[1] * dx * dy;
                    if (power > 0.0) continue;
                    const double alpha = std::min(opt.maxAlpha, P.opac * std::exp(power));
                    if (alpha < opt.minAlpha) continue;
                    const double nextT = T * (1.0 - alpha);
                    // Stop BEFORE blending the one that would take T below
                    // the threshold, as the paper does -- and the backward
                    // pass relies on this entry not having contributed.
                    if (nextT < opt.tStop) break;
                    const double* col = splats[size_t(list[size_t(k)])].color;
                    c0 += col[0] * alpha * T;
                    c1 += col[1] * alpha * T;
                    c2 += col[2] * alpha * T;
                    cd += P.depth * alpha * T;
                    T = nextT;
                    last = k + 1;
                }
                const size_t pi = size_t(py) * size_t(cam.w) + size_t(px);
                (*rgb)[pi * 3 + 0] = c0 + T * opt.background.x;
                (*rgb)[pi * 3 + 1] = c1 + T * opt.background.y;
                (*rgb)[pi * 3 + 2] = c2 + T * opt.background.z;
                m_finalT[pi] = T;
                m_lastIdx[pi] = last;
                if (depth) (*depth)[pi] = cd;   // background depth is 0
            }
    });
    m_time.composite += MsSince(clk);
}

// --- backward --------------------------------------------------------------------

void SplatRaster::Backward(const std::vector<SplatParam>& splats,
                           const SplatCam& cam, const RasterOptions& opt,
                           const std::vector<double>& dRgb,
                           std::vector<SplatParam>* grad,
                           const std::vector<double>* dDepth,
                           const std::vector<float>* depthShift) {
    auto clk = Clock::now();
    // PER-PIXEL PASS, one slot per tile-list entry. Each tile is handled by
    // one thread and writes only its own entries, so there is no contention
    // and no atomics; the entries are summed per Gaussian afterwards. Cheaper
    // in memory than a gradient buffer per thread, because a Gaussian appears
    // in only as many entries as tiles it touches.
    std::vector<size_t> offset(m_tiles.size() + 1, 0);
    for (size_t t = 0; t < m_tiles.size(); ++t)
        offset[t + 1] = offset[t] + m_tiles[t].size();
    std::vector<Grad2> entry(offset.back());

    // The device's version produces the same per-entry gradients. Only tried
    // if the forward pass also ran there: its transmittance and last-used
    // index live on the device.
    const bool onGpu = m_gpu && m_usedGpu &&
                       BackPixelGpu(cam, opt, dRgb, dDepth, depthShift, offset, &entry);

    if (!onGpu) ParallelFor(m_tiles.size(), [&](size_t t) {
        const std::vector<int>& list = m_tiles[t];
        Grad2* eg = entry.data() + offset[t];
        const int tx = int(t % size_t(m_tilesX)), ty = int(t / size_t(m_tilesX));
        for (int py = ty * kTile; py < std::min(cam.h, (ty + 1) * kTile); ++py)
            for (int px = tx * kTile; px < std::min(cam.w, (tx + 1) * kTile); ++px) {
                const size_t pi = size_t(py) * size_t(cam.w) + size_t(px);
                const double dC[3] = {dRgb[pi * 3 + 0], dRgb[pi * 3 + 1],
                                      dRgb[pi * 3 + 2]};

                // WALKING BACK TO FRONT. `accum` is the colour of everything
                // behind the current Gaussian, as a fraction of what reaches
                // it: the background at first, then each Gaussian folded in
                // as it is passed. That is what an alpha's gradient needs --
                // raising alpha shows more of this Gaussian and less of
                // whatever was behind it -- and it is why no per-pixel list
                // has to be stored: T is recovered by dividing back through
                // (1 - alpha), which maxAlpha keeps away from zero.
                double T = m_finalT[pi];
                double acc[3] = {opt.background.x, opt.background.y,
                                 opt.background.z};
                // Depth is a fourth channel: each Gaussian's "colour" in it
                // is its own depth less the per-pixel shift, and the
                // background's is 0. The gradient of its depth itself is
                // unshifted: d(value)/dz is 1 either way.
                const double dD = dDepth ? (*dDepth)[pi] : 0.0;
                const double shift = depthShift ? double((*depthShift)[pi]) : 0.0;
                double accD = 0.0;
                for (int k = m_lastIdx[pi] - 1; k >= 0; --k) {
                    const int j = list[size_t(k)];
                    const Proj& P = m_proj[size_t(j)];
                    const double dx = double(px) - P.u, dy = double(py) - P.v;
                    const double power = -0.5 * (P.conic[0] * dx * dx +
                                                 P.conic[2] * dy * dy) -
                                         P.conic[1] * dx * dy;
                    if (power > 0.0) continue;
                    const double G = std::exp(power);
                    const double raw = P.opac * G;
                    const double alpha = std::min(opt.maxAlpha, raw);
                    if (alpha < opt.minAlpha) continue;

                    const double Ti = T / (1.0 - alpha);
                    const double* col = splats[size_t(j)].color;
                    Grad2& g = eg[k];

                    // dC/dcolour = alpha T -- unless the pixel's measured
                    // depth says it shows another surface: see
                    // RasterOptions::colourGate.
                    const bool gated = opt.colourGate > 0.0 && shift > 0.0 &&
                                       std::fabs(P.depth - shift) > opt.colourGate * shift;
                    double dAlpha = 0.0;
                    for (int ch = 0; ch < 3; ++ch) {
                        if (!gated) g.color[ch] += alpha * Ti * dC[ch];
                        dAlpha += Ti * (col[ch] - acc[ch]) * dC[ch];
                        acc[ch] = alpha * col[ch] + (1.0 - alpha) * acc[ch];
                    }
                    if (dD != 0.0) {
                        g.depth += alpha * Ti * dD;
                        // Value (z - shift), background 0: see DepthLossGrad.
                        const double zs = P.depth - shift;
                        dAlpha += Ti * (zs - accD) * dD;
                        accD = alpha * zs + (1.0 - alpha) * accD;
                    }
                    T = Ti;

                    // Clamped at maxAlpha: alpha no longer moves with anything.
                    if (raw > opt.maxAlpha) continue;

                    // alpha = sigmoid(o) * G, and G = exp(power).
                    g.opacity += dAlpha * G * P.opac * (1.0 - P.opac);
                    const double dPower = dAlpha * alpha;

                    // power = -a dx^2/2 - b dx dy - c dy^2/2, with dx = px - u.
                    g.conic[0] += dPower * (-0.5 * dx * dx);
                    g.conic[1] += dPower * (-dx * dy);
                    g.conic[2] += dPower * (-0.5 * dy * dy);
                    g.u += dPower * (P.conic[0] * dx + P.conic[1] * dy);
                    g.v += dPower * (P.conic[1] * dx + P.conic[2] * dy);
                }
            }
    });

    m_time.backPixel += MsSince(clk);
    clk = Clock::now();
    // Sum the entries per Gaussian.
    std::vector<Grad2> g2(m_proj.size());
    for (size_t t = 0; t < m_tiles.size(); ++t)
        for (size_t k = 0; k < m_tiles[t].size(); ++k)
            g2[size_t(m_tiles[t][k])].Add(entry[offset[t] + k]);

    m_screenGrad.assign(m_proj.size(), -1.0);
    for (size_t i = 0; i < m_proj.size(); ++i)
        if (m_proj[i].ok) m_screenGrad[i] = std::hypot(g2[i].u, g2[i].v);

    m_time.backSum += MsSince(clk);
    clk = Clock::now();
    // PER-GAUSSIAN PASS: from the 2D quantities back to the parameters.
    const double* W = cam.R.m;
    ParallelFor(m_proj.size(), [&](size_t i) {
        const Proj& P = m_proj[i];
        if (!P.ok) return;
        const Grad2& g = g2[i];
        SplatParam& out = (*grad)[i];

        for (int ch = 0; ch < 3; ++ch) out.color[ch] += g.color[ch];
        out.opacity += g.opacity;

        // --- conic -> 2D covariance ------------------------------------------
        // The conic is Q = cov2^-1, and power uses its off-diagonal b ONCE
        // with the factor 2 folded in; as a symmetric matrix the gradient is
        // therefore [[ga, gb/2], [gb/2, gc]]. Then d(inv X) = -X^-1 dX X^-1
        // gives G_cov = -Q G_Q Q.
        const double Q[4] = {P.conic[0], P.conic[1], P.conic[1], P.conic[2]};
        const double GQ[4] = {g.conic[0], 0.5 * g.conic[1], 0.5 * g.conic[1],
                              g.conic[2]};
        double QG[4], GS2[4];
        for (int r = 0; r < 2; ++r)
            for (int c = 0; c < 2; ++c)
                QG[r * 2 + c] = Q[r * 2 + 0] * GQ[0 * 2 + c] + Q[r * 2 + 1] * GQ[1 * 2 + c];
        for (int r = 0; r < 2; ++r)
            for (int c = 0; c < 2; ++c)
                GS2[r * 2 + c] = -(QG[r * 2 + 0] * Q[0 * 2 + c] + QG[r * 2 + 1] * Q[1 * 2 + c]);

        // --- cov2 = J M J^T ---------------------------------------------------
        // G_M = J^T G J, and G_J = 2 G J M since G and M are symmetric.
        const double* J = P.J;
        double GM[9];
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) {
                double v = 0.0;
                for (int r = 0; r < 2; ++r)
                    for (int c = 0; c < 2; ++c)
                        v += J[r * 3 + a] * GS2[r * 2 + c] * J[c * 3 + b];
                GM[a * 3 + b] = v;
            }
        double GJ[6];
        for (int r = 0; r < 2; ++r)
            for (int c = 0; c < 3; ++c) {
                double v = 0.0;
                for (int s = 0; s < 2; ++s)
                    for (int k = 0; k < 3; ++k)
                        v += GS2[r * 2 + s] * J[s * 3 + k] * P.M[k * 3 + c];
                GJ[r * 3 + c] = 2.0 * v;
            }

        // --- M = W Sigma W^T  ->  G_Sigma = W^T G_M W --------------------------
        double Wt[9], WtG[9], GSig[9];
        Transpose33(W, Wt);
        Mul33(Wt, GM, WtG);
        Mul33(WtG, W, GSig);

        // --- Sigma = (R S)(R S)^T ---------------------------------------------
        // With M3 = R S: G_M3 = 2 G_Sigma M3. Then M3_ik = R_ik s_k gives the
        // scale and rotation gradients directly.
        const double* R = P.Rq;
        double GM3[9];
        for (int a = 0; a < 3; ++a)
            for (int k = 0; k < 3; ++k) {
                double v = 0.0;
                for (int b = 0; b < 3; ++b) v += GSig[a * 3 + b] * R[b * 3 + k] * P.scale[k];
                GM3[a * 3 + k] = 2.0 * v;
            }
        double GR[9];
        for (int k = 0; k < 3; ++k) {
            double gs = 0.0;
            for (int a = 0; a < 3; ++a) {
                gs += GM3[a * 3 + k] * R[a * 3 + k];
                GR[a * 3 + k] = GM3[a * 3 + k] * P.scale[k];
            }
            // s = exp(logScale).
            out.logScale[k] += gs * P.scale[k];
        }

        // --- rotation matrix -> quaternion ------------------------------------
        // Each entry of R differentiated by w, x, y and z; see Splat::Rotation
        // for the matrix itself.
        const double w = P.qn[0], x = P.qn[1], y = P.qn[2], z = P.qn[3];
        const double gw = GR[1] * (-2 * z) + GR[2] * (2 * y) + GR[3] * (2 * z) +
                          GR[5] * (-2 * x) + GR[6] * (-2 * y) + GR[7] * (2 * x);
        const double gx = GR[1] * (2 * y) + GR[2] * (2 * z) + GR[3] * (2 * y) +
                          GR[4] * (-4 * x) + GR[5] * (-2 * w) + GR[6] * (2 * z) +
                          GR[7] * (2 * w) + GR[8] * (-4 * x);
        const double gy = GR[0] * (-4 * y) + GR[1] * (2 * x) + GR[2] * (2 * w) +
                          GR[3] * (2 * x) + GR[5] * (2 * z) + GR[6] * (-2 * w) +
                          GR[7] * (2 * z) + GR[8] * (-4 * y);
        const double gz = GR[0] * (-4 * z) + GR[1] * (-2 * w) + GR[2] * (2 * x) +
                          GR[3] * (2 * w) + GR[4] * (-4 * z) + GR[5] * (2 * y) +
                          GR[6] * (2 * x) + GR[7] * (2 * y);
        // Through the normalisation q = q_raw / |q_raw|.
        const double gq[4] = {gw, gx, gy, gz};
        double dot = 0.0;
        for (int k = 0; k < 4; ++k) dot += gq[k] * P.qn[k];
        for (int k = 0; k < 4; ++k)
            out.quat[k] += (gq[k] - P.qn[k] * dot) / P.qlen;

        // --- the mean: through the centre (u, v) and through J ------------------
        const double tx = P.tc[0], ty = P.tc[1], tz = P.tc[2];
        const double iz = 1.0 / tz, iz2 = iz * iz, iz3 = iz2 * iz;
        double gt[3] = {0, 0, 0};
        // The depth a Gaussian contributes IS its camera-space z.
        gt[2] += g.depth;
        gt[0] += g.u * cam.fx * iz;
        gt[2] += g.u * (-cam.fx * tx * iz2);
        gt[1] += g.v * cam.fy * iz;
        gt[2] += g.v * (-cam.fy * ty * iz2);
        // J00 = fx/z, J02 = -fx x/z^2, J11 = fy/z, J12 = -fy y/z^2.
        gt[2] += GJ[0] * (-cam.fx * iz2) + GJ[2] * (2.0 * cam.fx * tx * iz3) +
                 GJ[4] * (-cam.fy * iz2) + GJ[5] * (2.0 * cam.fy * ty * iz3);
        gt[0] += GJ[2] * (-cam.fx * iz2);
        gt[1] += GJ[5] * (-cam.fy * iz2);
        // Camera space to world: t = W p + t0, so dL/dp = W^T dL/dt.
        for (int a = 0; a < 3; ++a)
            out.mean[a] += W[0 * 3 + a] * gt[0] + W[1 * 3 + a] * gt[1] +
                           W[2 * 3 + a] * gt[2];
    });
    m_time.backGaussian += MsSince(clk);
}

}  // namespace tglab
