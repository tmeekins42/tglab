#include "gpu_sweep.h"

#include <cstring>
#include <vector>

#include "../../gpu/compute.h"

namespace tglab {
namespace {

// Every plane is RGBA32F, which is what the framework's descriptor formats
// are. What each channel holds is stated where the texture is filled.
ImageDesc PlaneDesc(int w, int h) {
    ImageDesc d;
    d.width  = w;
    d.height = h;
    d.format = Format::RGBA32F;
    return d;
}

// --- the kernel --------------------------------------------------------------
//
// One thread per reference pixel, one dispatch per plane. Bindings:
//
//   t0..t3  the neighbours, R32F luma (read as .x)
//   u0      running state: best, prev, next, rival
//   u1      running state: bestPlane, rivalPlane, lastScore, firstScore
//   u2      the reference: luma, sA, sAA, -     (read only; a UAV slot
//                                                because the SRVs are full)
//   u3      homographies: 3 texels per (plane, neighbour), row = plane*4+nb
//
// The reference and the homography table are bound as UAVs purely for want
// of SRV slots -- the root signature allows four of each, and four
// neighbours take every SRV. A UAV load is a plain cached load on any
// hardware this will run on, and neither is written.
//
// The window is re-read per neighbour rather than cached in registers: 49
// floats with dynamic indexing would go to scratch memory, which is slower
// than the L1 hits the re-read gets.
constexpr const char* kSweepHlsl = R"(
Texture2D<float4>   T0 : register(t0);
Texture2D<float4>   T1 : register(t1);
Texture2D<float4>   T2 : register(t2);
Texture2D<float4>   T3 : register(t3);
RWTexture2D<float4> U0 : register(u0);   // best, prev, next, rival
RWTexture2D<float4> U1 : register(u1);   // bestPlane, rivalPlane, lastScore, firstScore
RWTexture2D<float4> U2 : register(u2);   // reference: luma, sA, sAA, -
RWTexture2D<float4> U3 : register(u3);   // homographies

cbuffer Params : register(b0) {
    uint Width;  uint Height;
    uint NumNb;  uint PlaneIdx;  uint SkipDist;  uint Radius;
    uint NbW0;   uint NbH0;   uint NbW1;   uint NbH1;
    uint NbW2;   uint NbH2;   uint NbW3;   uint NbH3;
};

float LoadNb(uint k, int2 p) {
    switch (k) {
        case 0:  return T0[p].x;
        case 1:  return T1[p].x;
        case 2:  return T2[p].x;
        default: return T3[p].x;
    }
}

// A 16x16 TILE PER GROUP, its warped samples shared. Each window position
// is a sample every pixel whose window covers it would otherwise compute
// again -- 49 homography divides and 196 texture loads per pixel per
// neighbour at the default 7x7 window. Here each sample of the tile and its
// apron is warped ONCE, by whichever thread draws it, into group-shared
// memory; the windows are then summed from there. The arithmetic per sample
// and the order of the sums are the old kernel's -- but the compiler rounds
// the warp a little differently, so a pixel whose best two planes were all
// but tied can now pick the other. Measured on fountain-P11 (sfm.tgl at 0.4):
// 90.2% measured at a 0.2% median error either way, and fusion kept 475459
// points either way. On a 179-frame video the planes took 6.3 s, from 8.2.
#define TS 16
#define MAXT 46          // TS + 2 * the largest radius, 15
groupshared float gA[MAXT * MAXT];
groupshared float gB[MAXT * MAXT];
groupshared uint  gOk[MAXT * MAXT];

[numthreads(TS, TS, 1)]
void main(uint3 gid : SV_GroupID, uint3 gt : SV_GroupThreadID) {
    int r  = int(Radius);
    int T  = TS + 2 * r;
    int ox = int(gid.x) * TS - r, oy = int(gid.y) * TS - r;
    int li = int(gt.y) * TS + int(gt.x);
    int px = int(gid.x) * TS + int(gt.x), py = int(gid.y) * TS + int(gt.y);
    bool live = px < int(Width) && py < int(Height);

    // The reference's luma over the tile and its apron.
    for (int i = li; i < T * T; i += TS * TS) {
        int wx = ox + i % T, wy = oy + i / T;
        gA[i] = (wx >= 0 && wy >= 0 && wx < int(Width) && wy < int(Height))
                    ? U2[int2(wx, wy)].x : 0.0;
    }

    // Only a complete window is scored, matching the CPU: a partially
    // sampled window compares different amounts of image at different depths
    // and biases one plane against another. Margin pixels still run the fold
    // below with a score of -2, so their lastScore stays correct.
    float combined = -2.0;
    bool inside = live && px >= r && py >= r &&
                  px < int(Width) - r && py < int(Height) - r;
    float full = float((2 * r + 1) * (2 * r + 1));
    float sA = 0.0, sAA = 0.0;
    if (inside) {
        float4 rv = U2[int2(px, py)];
        sA = rv.y; sAA = rv.z;
    }

    float v[4];
    int n = 0;

    for (uint k = 0; k < NumNb; ++k) {
        uint row = PlaneIdx * 4 + k;
        float4 ha = U3[int2(0, row)];
        float4 hb = U3[int2(1, row)];
        float4 hc = U3[int2(2, row)];
        float h00 = ha.x, h01 = ha.y, h02 = ha.z;
        float h10 = ha.w, h11 = hb.x, h12 = hb.y;
        float h20 = hb.z, h21 = hb.w, h22 = hc.x;
        uint nw = (k == 0) ? NbW0 : (k == 1) ? NbW1 : (k == 2) ? NbW2 : NbW3;
        uint nh = (k == 0) ? NbH0 : (k == 1) ? NbH1 : (k == 2) ? NbH2 : NbH3;

        GroupMemoryBarrierWithGroupSync();   // the last neighbour's samples are read
        for (int i = li; i < T * T; i += TS * TS) {
            float fx = float(ox + i % T), fy = float(oy + i / T);
            float qx = h00 * fx + h01 * fy + h02;
            float qy = h10 * fx + h11 * fy + h12;
            float qz = h20 * fx + h21 * fy + h22;
            uint ok = 0;
            float b = 0.0;
            if (abs(qz) > 1e-12) {
                float sx = qx / qz, sy = qy / qz;
                // Strictly inside, so x0+1 and y0+1 are in range and no
                // clamping is needed. A clamped edge sample would invent
                // an agreement with the border pixel.
                if (sx >= 0.0 && sy >= 0.0 && sx < float(nw - 1) && sy < float(nh - 1)) {
                    int x0 = int(sx), y0 = int(sy);
                    float ax = sx - float(x0), ay = sy - float(y0);
                    float p00 = LoadNb(k, int2(x0,     y0));
                    float p10 = LoadNb(k, int2(x0 + 1, y0));
                    float p01 = LoadNb(k, int2(x0,     y0 + 1));
                    float p11 = LoadNb(k, int2(x0 + 1, y0 + 1));
                    b = lerp(lerp(p00, p10, ax), lerp(p01, p11, ax), ay);
                    ok = 1;
                }
            }
            gB[i] = b;
            gOk[i] = ok;
        }
        GroupMemoryBarrierWithGroupSync();

        if (inside) {
            float sB = 0, sBB = 0, sAB = 0;
            bool ok = true;
            int cx = px - ox, cy = py - oy;
            for (int dy = -r; dy <= r && ok; ++dy) {
                for (int dx = -r; dx <= r; ++dx) {
                    int t = (cy + dy) * T + (cx + dx);
                    if (gOk[t] == 0) { ok = false; break; }
                    float b = gB[t];
                    float a = gA[t];
                    sB += b; sBB += b * b; sAB += a * b;
                }
            }
            if (ok) {
                float num = sAB - sA * sB / full;
                float da  = sAA - sA * sA / full;
                float db  = sBB - sB * sB / full;
                float den = sqrt(da * db);
                // A flat window has no structure to correlate: the variance
                // is zero and the correlation is 0/0. Left unmeasured.
                if (den > 1e-6) v[n++] = num / den;
            }
        }
    }
    if (!live) return;

    if (inside) {
        // THE BEST HALF, NOT THE MEAN. On repeating texture a wrong depth
        // that shifts the pattern by one period correlates well in every
        // neighbour at once, so the mean does not penalise it; and a pixel
        // occluded in one view scores badly there however right the depth
        // is, which the mean charges against the correct answer.
        if (n > 0) {
            for (int i = 1; i < n; ++i) {
                float key = v[i];
                int j = i - 1;
                while (j >= 0 && v[j] < key) { v[j + 1] = v[j]; --j; }
                v[j + 1] = key;
            }
            int take = (n + 1) / 2;
            if (n >= 2 && take < 2) take = 2;
            float acc = 0.0;
            for (int t = 0; t < take; ++t) acc += v[t];
            combined = acc / float(take);
        }
    }

    // --- fold into the running state, mirroring the CPU's PixState -----------
    float4 s  = U0[uint2(px, py)];
    float4 pi = U1[uint2(px, py)];
    int   bestP     = int(pi.x);
    int   rivalP    = int(pi.y);
    float lastScore = pi.z;
    int   p         = int(PlaneIdx);
    int   skip      = int(SkipDist);

    // This plane is the one after the current best: its right shoulder.
    if (bestP == p - 1) s.z = combined;

    if (combined > s.x) {
        // A new winner. The old best becomes a rival only if it is far enough
        // away to be a separate peak rather than the same one's shoulder.
        if (bestP >= 0 && abs(bestP - p) > skip && s.x > s.w) {
            s.w = s.x;
            rivalP = bestP;
        }
        s.x = combined;
        bestP = p;
        s.y = lastScore;          // the left shoulder: last plane's score
        s.z = -2.0;               // the right shoulder, filled next plane
    } else if (abs(p - bestP) > skip && combined > s.w) {
        s.w = combined;
        rivalP = p;
    }

    U0[uint2(px, py)] = s;
    // w: the FIRST plane's score, kept so the winner can be checked for a
    // peak that falls away at both ends of the range; z ends up holding the
    // last plane's.
    U1[uint2(px, py)] = float4(float(bestP), float(rivalP), combined, p == 0 ? combined : pi.w);
}
)";

struct SweepKernel {
    ComputeKernel k;
    bool ready  = false;
    bool failed = false;

    bool Ensure(ComputeContext* gpu, std::string* err) {
        if (ready)  return true;
        if (failed) { *err = "sweep kernel failed to compile"; return false; }
        if (!gpu->CreateKernel(kSweepHlsl, "main", "plane_sweep", &k, err)) {
            failed = true;
            return false;
        }
        ready = true;
        return true;
    }
};

SweepKernel& TheKernel() {
    static SweepKernel k;
    return k;
}

ImageView ViewOf(std::vector<float>& buf, int w, int h) {
    ImageView v;
    v.desc = PlaneDesc(w, h);
    v.data = reinterpret_cast<uint8_t*>(buf.data());
    return v;
}

// The reference plane as the kernel wants it: luma in .x, and the window
// sums sA and sAA in .y and .z. The sums are the same for every plane, so
// they are computed once here with the CPU's O(1) sliding form and never
// again.
void ReferenceWithSums(const SweepPlane& ref, int radius,
                       std::vector<float>* out) {
    const int w = ref.w, h = ref.h, win = 2 * radius + 1;
    const size_t n = size_t(w) * size_t(h);
    out->assign(n * 4, 0.0f);
    for (size_t i = 0; i < n; ++i) (*out)[i * 4] = ref.v[i];

    std::vector<float> rowA, rowAA;
    rowA.assign(n, 0.0f);
    rowAA.assign(n, 0.0f);

    for (int y = 0; y < h; ++y) {
        const float* a = &ref.v[size_t(y) * size_t(w)];
        float sA = 0.0f, sAA = 0.0f;
        for (int x = 0; x < win && x < w; ++x) { sA += a[x]; sAA += a[x] * a[x]; }
        for (int x = radius; x < w - radius; ++x) {
            rowA[size_t(y) * size_t(w) + size_t(x)]  = sA;
            rowAA[size_t(y) * size_t(w) + size_t(x)] = sAA;
            const int add = x + radius + 1, sub = x - radius;
            if (add < w) { sA += a[add]; sAA += a[add] * a[add]; }
            sA -= a[sub]; sAA -= a[sub] * a[sub];
        }
    }
    for (int x = radius; x < w - radius; ++x) {
        float sA = 0.0f, sAA = 0.0f;
        for (int y = 0; y < win && y < h; ++y) {
            sA  += rowA[size_t(y) * size_t(w) + size_t(x)];
            sAA += rowAA[size_t(y) * size_t(w) + size_t(x)];
        }
        for (int y = radius; y < h - radius; ++y) {
            const size_t i = size_t(y) * size_t(w) + size_t(x);
            (*out)[i * 4 + 1] = sA;
            (*out)[i * 4 + 2] = sAA;
            const int add = y + radius + 1, sub = y - radius;
            if (add < h) {
                sA  += rowA[size_t(add) * size_t(w) + size_t(x)];
                sAA += rowAA[size_t(add) * size_t(w) + size_t(x)];
            }
            sA  -= rowA[size_t(sub) * size_t(w) + size_t(x)];
            sAA -= rowAA[size_t(sub) * size_t(w) + size_t(x)];
        }
    }
}

}  // namespace

struct GpuSweepSession::Impl {
    ComputeContext* gpu = nullptr;
    int w = 0, h = 0, radius = 0, skip = 2, nPlanes = 0;

    GpuImage state{}, planeIdx{}, ref{}, homog{};
    std::vector<GpuImage> nbs;
    std::vector<uint32_t> nbDims;   // w0, h0, w1, h1, ... padded to four

    std::vector<float> staging;
    std::vector<float> readback;
};

GpuSweepSession::GpuSweepSession() : m(std::make_unique<Impl>()) {}
GpuSweepSession::~GpuSweepSession() = default;

bool GpuSweepReady(ComputeContext* gpu) {
    if (!gpu || !gpu->Ready()) return false;
    std::string err;
    return TheKernel().Ensure(gpu, &err);
}

bool GpuSweepSession::Begin(ComputeContext* gpu, const SweepPlane& ref,
                            const std::vector<SweepPlane>& neighbours,
                            int radius, int nPlanes, int skip,
                            const double* H, std::string* err) {
    if (!gpu || !gpu->Ready()) { *err = "no device"; return false; }
    if (!TheKernel().Ensure(gpu, err)) return false;
    if (!ref.v || ref.w <= 0 || ref.h <= 0) { *err = "empty reference"; return false; }
    if (neighbours.empty() || neighbours.size() > 4) {
        *err = "the GPU sweep takes one to four neighbours";
        return false;
    }
    if (nPlanes <= 0 || !H) { *err = "no planes"; return false; }

    m->gpu = gpu;
    m->w = ref.w;
    m->h = ref.h;
    m->radius = radius;
    m->skip = skip;
    m->nPlanes = nPlanes;

    const size_t n = size_t(m->w) * size_t(m->h);
    const ImageDesc desc = PlaneDesc(m->w, m->h);
    if (!gpu->CreateImage(desc, &m->state) ||
        !gpu->CreateImage(desc, &m->planeIdx) ||
        !gpu->CreateImage(desc, &m->ref)) {
        *err = "could not allocate the sweep planes";
        return false;
    }

    ReferenceWithSums(ref, radius, &m->staging);
    if (!gpu->Upload(ViewOf(m->staging, m->w, m->h), &m->ref)) {
        *err = "could not upload the reference";
        return false;
    }

    // The running state starts empty: no winner, no rival, and a last score
    // of -2 so the first plane's left shoulder is correctly "nothing".
    m->staging.assign(n * 4, -2.0f);
    if (!gpu->Upload(ViewOf(m->staging, m->w, m->h), &m->state)) {
        *err = "could not initialise the sweep state";
        return false;
    }
    for (size_t i = 0; i < n; ++i) {
        m->staging[i * 4 + 0] = -1.0f;   // bestPlane
        m->staging[i * 4 + 1] = -1.0f;   // rivalPlane
        m->staging[i * 4 + 2] = -2.0f;   // lastScore
        m->staging[i * 4 + 3] = -2.0f;   // first plane's score
    }
    if (!gpu->Upload(ViewOf(m->staging, m->w, m->h), &m->planeIdx)) {
        *err = "could not initialise the plane indices";
        return false;
    }

    // The whole homography table, once. Three texels per (plane, neighbour),
    // rows strided by four so the shader's indexing does not depend on how
    // many neighbours there are.
    {
        const int hw = 3, hh = nPlanes * 4;
        if (!gpu->CreateImage(PlaneDesc(hw, hh), &m->homog)) {
            *err = "could not allocate the homography table";
            return false;
        }
        m->staging.assign(size_t(hw) * size_t(hh) * 4, 0.0f);
        const size_t nNb = neighbours.size();
        for (int p = 0; p < nPlanes; ++p)
            for (size_t k = 0; k < nNb; ++k) {
                const double* Hp = H + (size_t(p) * nNb + k) * 9;
                const size_t row = size_t(p) * 4 + k;
                float* t0 = &m->staging[(row * size_t(hw) + 0) * 4];
                float* t1 = &m->staging[(row * size_t(hw) + 1) * 4];
                float* t2 = &m->staging[(row * size_t(hw) + 2) * 4];
                t0[0] = float(Hp[0]); t0[1] = float(Hp[1]);
                t0[2] = float(Hp[2]); t0[3] = float(Hp[3]);
                t1[0] = float(Hp[4]); t1[1] = float(Hp[5]);
                t1[2] = float(Hp[6]); t1[3] = float(Hp[7]);
                t2[0] = float(Hp[8]);
            }
        if (!gpu->Upload(ViewOf(m->staging, hw, hh), &m->homog)) {
            *err = "could not upload the homography table";
            return false;
        }
    }

    // Every neighbour, uploaded once.
    m->nbs.resize(neighbours.size());
    m->nbDims.assign(8, 1);
    for (size_t i = 0; i < neighbours.size(); ++i) {
        const SweepPlane& nb = neighbours[i];
        if (!nb.v || nb.w <= 0 || nb.h <= 0) { *err = "empty neighbour"; return false; }
        // ONE CHANNEL: the kernel reads only luma, and a four-channel copy
        // was four times the upload -- most of what preparing a frame cost.
        ImageDesc nd = PlaneDesc(nb.w, nb.h);
        nd.format = Format::R32F;
        if (!gpu->CreateImage(nd, &m->nbs[i])) {
            *err = "could not allocate a neighbour plane";
            return false;
        }
        ImageView nv;
        nv.desc = nd;
        nv.data = reinterpret_cast<uint8_t*>(const_cast<float*>(nb.v));
        if (!gpu->Upload(nv, &m->nbs[i])) {
            *err = "could not upload a neighbour plane";
            return false;
        }
        m->nbDims[i * 2 + 0] = uint32_t(nb.w);
        m->nbDims[i * 2 + 1] = uint32_t(nb.h);
    }

    m->readback.assign(n * 4, 0.0f);
    return true;
}

bool GpuSweepSession::Plane(int planeIndex, std::string* err) {
    if (!m->gpu) { *err = "session not started"; return false; }
    if (planeIndex < 0 || planeIndex >= m->nPlanes) { *err = "bad plane"; return false; }

    std::vector<const GpuImage*> srv;
    for (size_t i = 0; i < m->nbs.size(); ++i) srv.push_back(&m->nbs[i]);
    while (srv.size() < 4) srv.push_back(&m->nbs[0]);   // unused slots

    std::vector<uint32_t> c{uint32_t(m->nbs.size()), uint32_t(planeIndex),
                            uint32_t(m->skip), uint32_t(m->radius)};
    for (uint32_t d : m->nbDims) c.push_back(d);

    if (!m->gpu->Dispatch(TheKernel().k, srv,
                          {&m->state, &m->planeIdx, &m->ref, &m->homog},
                          c, err, uint32_t((m->w + 15) / 16), uint32_t((m->h + 15) / 16)))
        return false;

    // SUBMITTED ON ITS OWN, every time. ComputeContext batches dispatches
    // into one command list until a pixel budget fills, and that batching
    // is where this machine hangs: several dispatches in one list, device
    // gone, desktop stalled. One dispatch per submission has never failed
    // here at any size, and the fence round trip it costs is a fraction of
    // a millisecond against a dispatch that does real work. It also lets the
    // UI's direct queue interleave, so the spinner keeps moving.
    return m->gpu->Flush(err);
}

bool GpuSweepSession::Finish(std::vector<float>* best, std::vector<float>* prev,
                             std::vector<float>* next, std::vector<float>* rival,
                             std::vector<int>* bestPlane, std::vector<float>* edge,
                             std::string* err) {
    if (!m->gpu) { *err = "session not started"; return false; }
    const size_t n = size_t(m->w) * size_t(m->h);

    ImageView rv = ViewOf(m->readback, m->w, m->h);
    if (!m->gpu->Readback(m->state, &rv)) {
        *err = "could not read back the sweep state";
        return false;
    }
    best->resize(n); prev->resize(n); next->resize(n); rival->resize(n);
    for (size_t i = 0; i < n; ++i) {
        (*best)[i]  = m->readback[i * 4 + 0];
        (*prev)[i]  = m->readback[i * 4 + 1];
        (*next)[i]  = m->readback[i * 4 + 2];
        (*rival)[i] = m->readback[i * 4 + 3];
    }

    if (!m->gpu->Readback(m->planeIdx, &rv)) {
        *err = "could not read back the plane indices";
        return false;
    }
    bestPlane->resize(n);
    for (size_t i = 0; i < n; ++i)
        (*bestPlane)[i] = int(m->readback[i * 4 + 0]);
    edge->resize(n);
    for (size_t i = 0; i < n; ++i)
        (*edge)[i] = std::max(m->readback[i * 4 + 2], m->readback[i * 4 + 3]);
    return true;
}

}  // namespace tglab
