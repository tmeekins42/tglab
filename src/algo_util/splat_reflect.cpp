#include "splat_reflect.h"

#include <algorithm>
#include <cmath>

#include "../core/parallel.h"

namespace tglab {
namespace {

// Direction -> (face, u, v) with u, v in 0..res in texel units, the cube
// map's usual major-axis mapping. `du`, `dv`, when given, receive the
// derivatives of u and v with respect to the direction: each of sc, tc and
// ma is plus or minus one of d's components, so the quotient rule on sc/ma
// and tc/ma is all it takes.
void FaceUV(const Vec3& d, int res, int* face, double* u, double* v,
            Vec3* du = nullptr, Vec3* dv = nullptr) {
    const double c[3] = {d.x, d.y, d.z};
    const double ax = std::fabs(d.x), ay = std::fabs(d.y), az = std::fabs(d.z);
    // Which component, and with what sign, each of sc, tc, ma is.
    int as, at, am;
    double ss, st, sm;
    if (ax >= ay && ax >= az) {
        am = 0; sm = d.x > 0 ? 1 : -1;
        if (d.x > 0) { *face = 0; as = 2; ss = -1; at = 1; st = -1; }
        else         { *face = 1; as = 2; ss =  1; at = 1; st = -1; }
    } else if (ay >= az) {
        am = 1; sm = d.y > 0 ? 1 : -1;
        if (d.y > 0) { *face = 2; as = 0; ss =  1; at = 2; st =  1; }
        else         { *face = 3; as = 0; ss =  1; at = 2; st = -1; }
    } else {
        am = 2; sm = d.z > 0 ? 1 : -1;
        if (d.z > 0) { *face = 4; as = 0; ss =  1; at = 1; st = -1; }
        else         { *face = 5; as = 0; ss = -1; at = 1; st = -1; }
    }
    const double ma = std::max(sm * c[am], 1e-12);
    const double sc = ss * c[as], tc = st * c[at];
    *u = (sc / ma + 1.0) * 0.5 * res;
    *v = (tc / ma + 1.0) * 0.5 * res;
    if (du && dv) {
        double gu[3] = {0, 0, 0}, gv[3] = {0, 0, 0};
        const double k = 0.5 * res / ma;
        gu[as] += k * ss;
        gu[am] -= k * sc / ma * sm;
        gv[at] += k * st;
        gv[am] -= k * tc / ma * sm;
        *du = Vec3{gu[0], gu[1], gu[2]};
        *dv = Vec3{gv[0], gv[1], gv[2]};
    }
}

// The four texels and weights a bilinear read at (u, v) uses, clamped to the
// face.
struct Taps { size_t at[4]; double w[4]; };
Taps TapsOf(int face, double u, double v, int res) {
    const double x = std::clamp(u - 0.5, 0.0, double(res - 1));
    const double y = std::clamp(v - 0.5, 0.0, double(res - 1));
    const int x0 = int(x), y0 = int(y);
    const int x1 = std::min(x0 + 1, res - 1), y1 = std::min(y0 + 1, res - 1);
    const double fx = x - x0, fy = y - y0;
    auto idx = [&](int xx, int yy) {
        return ((size_t(face) * size_t(res) + size_t(yy)) * size_t(res) + size_t(xx)) * 3;
    };
    return Taps{{idx(x0, y0), idx(x1, y0), idx(x0, y1), idx(x1, y1)},
                {(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy}};
}

// The pixel's viewing ray in world space, unit length, from the eye.
Vec3 RayOf(const SplatCam& cam, int x, int y) {
    const Vec3 c{(double(x) - cam.cx) / cam.fx, (double(y) - cam.cy) / cam.fy, 1.0};
    return (cam.R.Transpose() * c).Normalized();
}

Vec3 Reflect(const Vec3& v, const Vec3& n) { return v - n * (2.0 * v.Dot(n)); }

}  // namespace

Vec3 EnvMap::Sample(const Vec3& d) const {
    int face = 0;
    double u = 0, v = 0;
    FaceUV(d, res, &face, &u, &v);
    const Taps t = TapsOf(face, u, v, res);
    double c[3] = {0, 0, 0};
    for (int k = 0; k < 4; ++k)
        for (int ch = 0; ch < 3; ++ch) c[ch] += t.w[k] * texels[t.at[k] + size_t(ch)];
    return Vec3{c[0], c[1], c[2]};
}

Vec3 EnvMap::DirectionGrad(const Vec3& d, const Vec3& dColour) const {
    int face = 0;
    double u = 0, v = 0;
    Vec3 du, dv;
    FaceUV(d, res, &face, &u, &v, &du, &dv);
    // The bilinear read's slope across the face, zero where it is clamped
    // to the edge (the value does not change there).
    const double x = u - 0.5, y = v - 0.5;
    const bool clampX = x < 0.0 || x > double(res - 1);
    const bool clampY = y < 0.0 || y > double(res - 1);
    const Taps t = TapsOf(face, u, v, res);
    const double xs = std::clamp(x, 0.0, double(res - 1)), ys = std::clamp(y, 0.0, double(res - 1));
    const double fx = xs - std::floor(xs), fy = ys - std::floor(ys);
    const double g[3] = {dColour.x, dColour.y, dColour.z};
    double gx = 0.0, gy = 0.0;
    for (int ch = 0; ch < 3; ++ch) {
        const double T0 = texels[t.at[0] + size_t(ch)], T1 = texels[t.at[1] + size_t(ch)];
        const double T2 = texels[t.at[2] + size_t(ch)], T3 = texels[t.at[3] + size_t(ch)];
        gx += g[ch] * ((1 - fy) * (T1 - T0) + fy * (T3 - T2));
        gy += g[ch] * ((1 - fx) * (T2 - T0) + fx * (T3 - T1));
    }
    if (clampX) gx = 0.0;
    if (clampY) gy = 0.0;
    return du * gx + dv * gy;
}

void EnvMap::Splat(const Vec3& d, const Vec3& dColour, std::vector<double>* grad) const {
    int face = 0;
    double u = 0, v = 0;
    FaceUV(d, res, &face, &u, &v);
    const Taps t = TapsOf(face, u, v, res);
    const double g[3] = {dColour.x, dColour.y, dColour.z};
    for (int k = 0; k < 4; ++k)
        for (int ch = 0; ch < 3; ++ch) (*grad)[t.at[k] + size_t(ch)] += t.w[k] * g[ch];
}

void ReflPayload(const std::vector<SplatParam>& params, const std::vector<ReflParam>& refl,
                 std::vector<SplatParam>* out) {
    *out = params;
    for (size_t i = 0; i < params.size(); ++i) {
        const double r = Sigmoid(refl[i].reflLogit);
        for (double& c : (*out)[i].color) c = r;
    }
}

void NormalPayload(const std::vector<SplatParam>& params, const std::vector<ReflParam>& refl,
                   const Vec3& eye, std::vector<SplatParam>* out, std::vector<double>* flip) {
    *out = params;
    flip->assign(params.size(), 1.0);
    for (size_t i = 0; i < params.size(); ++i) {
        Vec3 n{refl[i].normal[0], refl[i].normal[1], refl[i].normal[2]};
        const double len = n.Norm();
        n = (len > 1e-12) ? n * (1.0 / len) : Vec3{0, 0, 1};
        // Turned toward the camera: a disc has no front, and a normal facing
        // away would reflect the ray into the surface.
        const Vec3 toEye = eye - Vec3{params[i].mean[0], params[i].mean[1], params[i].mean[2]};
        if (n.Dot(toEye) < 0.0) { n = n * -1.0; (*flip)[i] = -1.0; }
        (*out)[i].color[0] = n.x;
        (*out)[i].color[1] = n.y;
        (*out)[i].color[2] = n.z;
    }
}

void ShadeDeferred(const std::vector<double>& Cd, const std::vector<double>& Rm,
                   const std::vector<double>& Nm, const SplatCam& cam, const EnvMap& env,
                   std::vector<double>* out) {
    const int w = cam.w, h = cam.h;
    out->assign(size_t(w) * size_t(h) * 3, 0.0);
    ParallelFor(size_t(h), [&](size_t yy) {
        const int y = int(yy);
        for (int x = 0; x < w; ++x) {
            const size_t p = (size_t(y) * size_t(w) + size_t(x)) * 3;
            const double R = Rm[p];
            const Vec3 N{Nm[p], Nm[p + 1], Nm[p + 2]};
            Vec3 e{0, 0, 0};
            const double nl = N.Norm();
            if (R > 1e-9 && nl > 1e-9)
                e = env.Sample(Reflect(RayOf(cam, x, y), N * (1.0 / nl)));
            (*out)[p]     = (1.0 - R) * Cd[p]     + R * e.x;
            (*out)[p + 1] = (1.0 - R) * Cd[p + 1] + R * e.y;
            (*out)[p + 2] = (1.0 - R) * Cd[p + 2] + R * e.z;
        }
    });
}

void ShadeDeferredBackward(const std::vector<double>& Cd, const std::vector<double>& Rm,
                           const std::vector<double>& Nm, const SplatCam& cam,
                           const EnvMap& env, const std::vector<double>& dOut,
                           std::vector<double>* dCd, std::vector<double>* dRm,
                           std::vector<double>* dNm, std::vector<double>* envGrad) {
    const int w = cam.w, h = cam.h;
    const size_t np = size_t(w) * size_t(h) * 3;
    dCd->assign(np, 0.0);
    dRm->assign(np, 0.0);
    dNm->assign(np, 0.0);
    // The environment's gradient is gathered per BLOCK of rows and summed
    // after, so blocks run in parallel without contending for texels -- and
    // without a whole map's worth of accumulator per row.
    const size_t nBlocks = std::min<size_t>(size_t(h), 16);
    std::vector<std::vector<double>> rowEnv(nBlocks);
    ParallelFor(nBlocks, [&](size_t blk) {
        std::vector<double>& eg = rowEnv[blk];
        const int y0 = int(blk * size_t(h) / nBlocks), y1 = int((blk + 1) * size_t(h) / nBlocks);
        for (int y = y0; y < y1; ++y)
        for (int x = 0; x < w; ++x) {
            const size_t p = (size_t(y) * size_t(w) + size_t(x)) * 3;
            const double R = Rm[p];
            const Vec3 g{dOut[p], dOut[p + 1], dOut[p + 2]};
            (*dCd)[p]     = (1.0 - R) * g.x;
            (*dCd)[p + 1] = (1.0 - R) * g.y;
            (*dCd)[p + 2] = (1.0 - R) * g.z;
            const Vec3 N{Nm[p], Nm[p + 1], Nm[p + 2]};
            const double nl = N.Norm();
            if (R <= 1e-9 || nl <= 1e-9) {
                // No reflection drawn here, so only R's own term: dC/dR is
                // (e - Cd) with e = 0.
                (*dRm)[p] = -(g.x * Cd[p] + g.y * Cd[p + 1] + g.z * Cd[p + 2]);
                continue;
            }
            const Vec3 n = N * (1.0 / nl);
            const Vec3 v = RayOf(cam, x, y);
            const Vec3 om = Reflect(v, n);
            const Vec3 e = env.Sample(om);
            (*dRm)[p] = g.x * (e.x - Cd[p]) + g.y * (e.y - Cd[p + 1]) + g.z * (e.z - Cd[p + 2]);

            if (eg.empty()) eg.assign(env.texels.size(), 0.0);
            env.Splat(om, g * R, &eg);

            // dLoss/d(reflected direction), exactly: the face coordinates'
            // change with direction times the bilinear read's slope.
            const Vec3 gOm = env.DirectionGrad(om, g * R);

            // om = v - 2 (v.n) n  ->  dL/dn_j = -2 (v_j (gOm.n) + (v.n) gOm_j)
            const double vn = v.Dot(n), gn = gOm.Dot(n);
            const Vec3 gUnit = (v * gn + gOm * vn) * -2.0;
            // n = N / |N|  ->  dL/dN = (gUnit - n (n.gUnit)) / |N|
            const Vec3 gN = (gUnit - n * n.Dot(gUnit)) * (1.0 / nl);
            (*dNm)[p]     = gN.x;
            (*dNm)[p + 1] = gN.y;
            (*dNm)[p + 2] = gN.z;
        }
    });
    for (const std::vector<double>& eg : rowEnv)
        if (!eg.empty())
            for (size_t i = 0; i < eg.size(); ++i) (*envGrad)[i] += eg[i];
}

}  // namespace tglab
