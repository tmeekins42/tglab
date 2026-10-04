#include "gpu_knn.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace tglab {

namespace {

// One thread per query, its descriptor held in registers; the group walks the
// database through group-shared memory a tile at a time, so every database
// vector is read from memory once per 64 queries and then broadcast.
//
// The distance is the sum of squared DIFFERENCES, as DistanceL2Sq computes it
// on the CPU -- not |q|^2 + |d|^2 - 2 q.d, which is cheaper and loses the
// small distances that matter most to cancellation.
const char* kKnnHlsl = R"(
ByteAddressBuffer   Q    : register(t0);
ByteAddressBuffer   D    : register(t1);
RWByteAddressBuffer OIdx : register(u0);
RWByteAddressBuffer ODst : register(u1);
#ifdef BACK
// Per group of 64 queries, each database vector's nearest among them:
// [group][database vector], index and distance. See kColMergeHlsl.
RWByteAddressBuffer BIdx : register(u2);
RWByteAddressBuffer BDst : register(u3);
#endif

cbuffer P : register(b0) { uint NQ, ND, SliceLen, Slices; };

#define V    (DIM / 4)
#define TILE (1024 / V)    // database vectors per tile: 16 KB of float4
#ifndef KMAX
#define KMAX 8
#endif

groupshared float4 sD[TILE * V];
#ifdef BACK
groupshared float sDist[TILE * 64];   // this tile's distances, [vector][query]
#endif

[numthreads(64, 1, 1)]
void main(uint3 g : SV_GroupID, uint gi : SV_GroupIndex) {
    const uint qi = g.x * 64 + gi;
    const bool live = qi < NQ;
    float4 q[V];
    [unroll] for (uint k = 0; k < V; ++k)
        q[k] = live ? asfloat(Q.Load4((qi * V + k) * 16)) : float4(0, 0, 0, 0);

    float bd[KMAX];
    int   bi[KMAX];
    [unroll] for (uint j = 0; j < KMAX; ++j) { bd[j] = 3.0e38; bi[j] = -1; }

    const uint begin = g.z * SliceLen, end = min(ND, begin + SliceLen);
    for (uint base = begin; base < end; base += TILE) {
        const uint cnt = min(uint(TILE), end - base);
        for (uint i = gi; i < TILE * V; i += 64) {
            const uint r = i / V;
            sD[i] = r < cnt ? asfloat(D.Load4(((base + r) * V + i % V) * 16)) : float4(0, 0, 0, 0);
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint r = 0; r < cnt; ++r) {
            float4 acc = float4(0, 0, 0, 0);
            [unroll] for (uint k = 0; k < V; ++k) {
                const float4 d = q[k] - sD[r * V + k];
                acc += d * d;
            }
            const float dist = (acc.x + acc.y) + (acc.z + acc.w);
#ifdef BACK
            sDist[r * 64 + gi] = live ? dist : 3.0e38;
#endif
            if (dist < bd[KMAX - 1]) {
                bd[KMAX - 1] = dist;
                bi[KMAX - 1] = int(base + r);
                [unroll] for (int j = KMAX - 1; j > 0; --j) {
                    if (bd[j] < bd[j - 1]) {
                        const float td = bd[j]; bd[j] = bd[j - 1]; bd[j - 1] = td;
                        const int   ti = bi[j]; bi[j] = bi[j - 1]; bi[j - 1] = ti;
                    }
                }
            }
        }
#ifdef BACK
        // The other direction, from the same distances: each of this tile's
        // database vectors, its nearest of the group's queries -- the lowest
        // query index on a tie, as a search the other way round keeps.
        GroupMemoryBarrierWithGroupSync();
        for (uint c = gi; c < cnt; c += 64) {
            float bd2 = sDist[c * 64];
            uint  bj  = 0;
            for (uint j = 1; j < 64; ++j) {
                const float d = sDist[c * 64 + j];
                if (d < bd2) { bd2 = d; bj = j; }
            }
            const uint o = (g.x * ND + base + c) * 4;
            BIdx.Store(o, g.x * 64 + bj < NQ ? g.x * 64 + bj : 0xffffffffu);
            BDst.Store(o, asuint(bd2));
        }
#endif
        GroupMemoryBarrierWithGroupSync();
    }

    if (!live) return;
    const uint o = (qi * Slices + g.z) * KMAX;
    [unroll] for (uint j = 0; j < KMAX; ++j) {
        OIdx.Store((o + j) * 4, asuint(bi[j]));
        ODst.Store((o + j) * 4, asuint(bd[j]));
    }
}
)";

// THE MERGE, one thread per query: every slice's list -- each sorted, KP
// long -- folded into the K best by (distance, index), the same order the
// CPU merge's partial_sort of pairs gave, ties to the lower index. Written
// as K indices then K distances per query, at OutBase in a buffer the whole
// batch shares, so a batch reads back once.
const char* kMergeHlsl = R"(
ByteAddressBuffer   PIdx : register(t0);
ByteAddressBuffer   PDst : register(t1);
RWByteAddressBuffer Out  : register(u0);

cbuffer P : register(b0) { uint NQ, Slices, KP, K, OutBase; };

#define KMAX 8

bool Before(float d, int i, float bd, int bi) {
    return bi < 0 || d < bd || (d == bd && i < bi);
}

[numthreads(64, 1, 1)]
void main(uint3 t : SV_DispatchThreadID) {
    const uint qi = t.x;
    if (qi >= NQ) return;
    float bd[KMAX];
    int   bi[KMAX];
    [unroll] for (uint j = 0; j < KMAX; ++j) { bd[j] = 0; bi[j] = -1; }
    for (uint s = 0; s < Slices; ++s) {
        for (uint j = 0; j < KP; ++j) {
            const uint o = ((qi * Slices + s) * KP + j) * 4;
            const int i = asint(PIdx.Load(o));
            if (i < 0) continue;
            const float d = asfloat(PDst.Load(o));
            if (!Before(d, i, bd[K - 1], bi[K - 1])) continue;
            bd[K - 1] = d;
            bi[K - 1] = i;
            for (int m = int(K) - 1; m > 0; --m) {
                if (!Before(bd[m], bi[m], bd[m - 1], bi[m - 1])) break;
                const float td = bd[m]; bd[m] = bd[m - 1]; bd[m - 1] = td;
                const int   ti = bi[m]; bi[m] = bi[m - 1]; bi[m - 1] = ti;
            }
        }
    }
    const uint base = (OutBase + qi * K * 2) * 4;
    for (uint j = 0; j < K; ++j) {
        Out.Store(base + j * 4, asuint(bi[j]));
        Out.Store(base + (K + j) * 4, bi[j] < 0 ? 0x7f800000u : asuint(bd[j]));
    }
}
)";

// The other direction's merge: for each database vector, the nearest query
// over every group's candidate -- by (distance, index), so a tie goes to the
// lower query as before. Written as [index, distance] at OutBase.
const char* kColMergeHlsl = R"(
ByteAddressBuffer   BIdx : register(t0);
ByteAddressBuffer   BDst : register(t1);
RWByteAddressBuffer Out  : register(u0);

cbuffer P : register(b0) { uint ND, Groups, OutBase; };

[numthreads(64, 1, 1)]
void main(uint3 t : SV_DispatchThreadID) {
    const uint r = t.x;
    if (r >= ND) return;
    float bd = 0;
    int   bi = -1;
    for (uint g = 0; g < Groups; ++g) {
        const uint o = (g * ND + r) * 4;
        const int i = asint(BIdx.Load(o));
        if (i < 0) continue;
        const float d = asfloat(BDst.Load(o));
        if (bi < 0 || d < bd || (d == bd && i < bi)) { bd = d; bi = i; }
    }
    const uint base = (OutBase + r * 2) * 4;
    Out.Store(base, asuint(bi));
    Out.Store(base + 4, bi < 0 ? 0x7f800000u : asuint(bd));
}
)";

} // namespace

const ComputeKernel* GpuKnn::ColMergeKernel(std::string* err) {
    if (!m_colMerge) {
        std::string e;
        m_colMerge = m_gpu->SharedKernel("knn_colmerge", kColMergeHlsl, &e);
        if (!m_colMerge) *err = "knn column merge: " + e;
    }
    return m_colMerge;
}

const ComputeKernel* GpuKnn::Kernel(int dim, int kp, bool back, std::string* err) {
    const ComputeKernel*& kern = m_kernel[back ? 1 : 0][kp == 1 ? 1 : 0][dim / 4];
    if (!kern) {
        const std::string src = "#define DIM " + std::to_string(dim) + "\n#define KMAX " +
                                std::to_string(kp) + "\n" + (back ? "#define BACK 1\n" : "") +
                                kKnnHlsl;
        std::string e;
        kern = m_gpu->SharedKernel("knn_dim" + std::to_string(dim) + "_k" + std::to_string(kp) +
                                       (back ? "_back" : ""),
                                   src, &e);
        if (!kern) *err = "knn: " + e;
    }
    return kern;
}

const ComputeKernel* GpuKnn::MergeKernel(std::string* err) {
    if (!m_merge) {
        std::string e;
        m_merge = m_gpu->SharedKernel("knn_merge", kMergeHlsl, &e);
        if (!m_merge) *err = "knn merge: " + e;
    }
    return m_merge;
}

bool GpuKnn::Upload(const float* v, int n, int dim, GpuBuffer* out, std::string* err) {
    const uint64_t bytes = uint64_t(std::max(n, 1)) * uint64_t(dim) * 4u;
    if (!m_gpu->CreateBuffer(bytes, out) ||
        (n > 0 && !m_gpu->UploadBuffer(v, uint64_t(n) * uint64_t(dim) * 4u, out))) {
        *err = "could not upload descriptors";
        return false;
    }
    return true;
}

bool GpuKnn::Search(const GpuBuffer& q, int nq, const GpuBuffer& db, int nd, int dim, int k,
                    std::vector<int>* idx, std::vector<float>* d2, std::string* err) {
    Job j;
    j.q = &q;
    j.nq = nq;
    j.db = &db;
    j.nd = nd;
    j.k = k;
    j.idx = idx;
    j.d2 = d2;
    return SearchBatch({j}, dim, err);
}

bool GpuKnn::SearchBatch(const std::vector<Job>& jobs, int dim, std::string* err) {
    if (!Supports(dim)) { *err = "unsupported descriptor length"; return false; }
    const ComputeKernel* merge = MergeKernel(err);
    if (!merge) return false;

    // Each job's place in the shared output, in uints: K indices and K
    // distances per query.
    std::vector<uint64_t> at(jobs.size(), 0), atBack(jobs.size(), 0);
    uint64_t total = 0;
    for (size_t n = 0; n < jobs.size(); ++n) {
        const Job& j = jobs[n];
        const int k = std::clamp(j.k, 1, kKnnMax);
        j.idx->assign(size_t(std::max(j.nq, 0)) * size_t(k), -1);
        j.d2->assign(size_t(std::max(j.nq, 0)) * size_t(k), std::numeric_limits<float>::infinity());
        at[n] = total;
        if (j.nq > 0 && j.nd > 0) total += uint64_t(j.nq) * uint64_t(k) * 2;
        if (j.back) {
            j.back->assign(size_t(std::max(j.nd, 0)), -1);
            atBack[n] = total;
            if (j.nq > 0 && j.nd > 0) total += uint64_t(j.nd) * 2;
        }
    }
    const ComputeKernel* colMerge = nullptr;
    for (const Job& j : jobs)
        if (j.back && !(colMerge = ColMergeKernel(err))) return false;
    if (total == 0) return true;
    GpuBuffer out;
    if (!m_gpu->AcquireBuffer(total * 4, &out)) { *err = "could not allocate knn output"; return false; }

    bool ok = true;
    for (size_t n = 0; n < jobs.size() && ok; ++n) {
        const Job& j = jobs[n];
        if (j.nq <= 0 || j.nd <= 0) continue;
        const int k = std::clamp(j.k, 1, kKnnMax);
        // The slices keep a list as long as needed: one for a nearest-only
        // search, which is a smaller kernel and an eighth of the output.
        const int kp = k == 1 ? 1 : kKnnMax;
        const ComputeKernel* kern = Kernel(dim, kp, j.back != nullptr, err);
        if (!kern) { ok = false; break; }

        // Slices of the database across z, enough that the grid fills the
        // device even when the queries alone would not: ~1000 groups.
        const uint32_t groups = uint32_t((j.nq + 63) / 64);
        const int slices = std::clamp(int(1000 / std::max(1u, groups)), 1, std::max(1, j.nd / 256));
        const uint32_t sliceLen = uint32_t((j.nd + slices - 1) / slices);
        const uint64_t partN = uint64_t(j.nq) * uint64_t(slices) * uint64_t(kp);

        // Scratch handed back as soon as it is recorded: reusing it later in
        // the same batch is ordered by the barriers each bind records.
        GpuBuffer pi, pd, bi, bd;
        const uint64_t backN = j.back ? uint64_t(groups) * uint64_t(j.nd) : 0;
        if (!m_gpu->AcquireBuffer(partN * 4, &pi) || !m_gpu->AcquireBuffer(partN * 4, &pd) ||
            (j.back && (!m_gpu->AcquireBuffer(backN * 4, &bi) || !m_gpu->AcquireBuffer(backN * 4, &bd)))) {
            *err = "could not allocate knn output";
            ok = false;
            break;
        }
        std::vector<GpuBuffer*> outs{&pi, &pd};
        if (j.back) {
            outs.push_back(&bi);
            outs.push_back(&bd);
        }
        ok = m_gpu->DispatchBuffers(*kern, {j.q, j.db}, outs,
                                    {uint32_t(j.nq), uint32_t(j.nd), sliceLen, uint32_t(slices)},
                                    groups, 1, uint32_t(slices), err) &&
             (!j.back ||
              m_gpu->DispatchBuffers(*colMerge, {&bi, &bd}, {&out},
                                     {uint32_t(j.nd), groups, uint32_t(atBack[n])},
                                     (uint32_t(j.nd) + 63) / 64, 1, 1, err)) &&
             m_gpu->DispatchBuffers(*merge, {&pi, &pd}, {&out},
                                    {uint32_t(j.nq), uint32_t(slices), uint32_t(kp), uint32_t(k),
                                     uint32_t(at[n])},
                                    groups, 1, 1, err);
        m_gpu->RecycleBuffer(std::move(pi));
        m_gpu->RecycleBuffer(std::move(pd));
        if (j.back) {
            m_gpu->RecycleBuffer(std::move(bi));
            m_gpu->RecycleBuffer(std::move(bd));
        }
    }

    std::vector<uint32_t> host;
    if (ok) {
        host.resize(size_t(total));
        ok = m_gpu->ReadbackBuffer(out, host.data(), total * 4);
        if (!ok && err->empty()) *err = "knn readback failed";
    }
    m_gpu->RecycleBuffer(std::move(out));
    if (!ok) return false;

    for (size_t n = 0; n < jobs.size(); ++n) {
        const Job& j = jobs[n];
        if (j.nq <= 0 || j.nd <= 0) continue;
        const size_t k = size_t(std::clamp(j.k, 1, kKnnMax));
        const uint32_t* h = host.data() + at[n];
        for (size_t qi = 0; qi < size_t(j.nq); ++qi) {
            const uint32_t* row = h + qi * k * 2;
            for (size_t m = 0; m < k; ++m) {
                (*j.idx)[qi * k + m] = int(row[m]);
                float f;
                std::memcpy(&f, &row[k + m], 4);
                (*j.d2)[qi * k + m] = f;
            }
        }
        if (j.back) {
            const uint32_t* hb = host.data() + atBack[n];
            for (size_t r = 0; r < size_t(j.nd); ++r) (*j.back)[r] = int(hb[r * 2]);
        }
    }
    return true;
}

} // namespace tglab
