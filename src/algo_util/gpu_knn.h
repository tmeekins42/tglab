// gpu_knn — exact k-nearest-neighbour search over float descriptors, on the
// GPU: every query against every database vector, the k smallest squared L2
// distances kept.
//
// WHY BRUTE FORCE. match_ann's k-d forest exists because brute force on the
// CPU is O(n*m*dim): two images of 10,000 128-float descriptors are 13 billion
// multiply-adds a pair. On a GPU that is a few milliseconds -- and the answer
// is EXACT, where the forest trades recall for speed. Measured on a 169-frame
// video with learned descriptors, the forest took 168 s for 638 pairs.
//
// k is at most kKnnMax. The search runs the database in slices across the
// grid's z (each slice keeps its own k best) so that ten thousand queries are
// enough threads to fill the device; a second kernel merges the slices.
//
// BATCHED. A pair's search is a few milliseconds of device time, and it used
// to be run one at a time: dispatch, wait, read back every slice's list --
// 3.8 MB for ten thousand queries -- and merge them on one CPU thread. On a
// 179-frame video that was 9.6 s of waiting and copying and 7.6 s of
// merging, for 19.5 s of matching. SearchBatch records many searches, merges
// on the device, and reads back only the final k per query, once.
#pragma once

#include <string>
#include <vector>

#include "../gpu/compute.h"

namespace tglab {

constexpr int kKnnMax = 8;

class GpuKnn {
public:
    explicit GpuKnn(ComputeContext* gpu) : m_gpu(gpu) {}

    // Whether descriptors of this length can be searched: a multiple of 4,
    // at most 256.
    static bool Supports(int dim) { return dim > 0 && dim % 4 == 0 && dim <= 256; }

    // n descriptors of `dim` floats, onto the device.
    bool Upload(const float* v, int n, int dim, GpuBuffer* out, std::string* err);

    // For each of `nq` queries, the `k` nearest of `nd` database vectors:
    // idx and d2 receive nq * k entries, nearest first, -1 / +inf where the
    // database has fewer than k. Ties go to the lower index.
    bool Search(const GpuBuffer& q, int nq, const GpuBuffer& db, int nd, int dim, int k,
                std::vector<int>* idx, std::vector<float>* d2, std::string* err);

    // Search() for many query/database pairs in one submission, each job's
    // answer exactly what Search() would give it.
    struct Job {
        const GpuBuffer* q = nullptr;
        int nq = 0;
        const GpuBuffer* db = nullptr;
        int nd = 0;
        int k = 1;
        std::vector<int>*   idx = nullptr;
        std::vector<float>* d2 = nullptr;
        // The other direction too, from the same distances: for each of the
        // nd database vectors, its nearest query (-1 if none) -- exactly
        // what a k = 1 search with the roles swapped returns, for half the
        // arithmetic. match_ann's cross check is this.
        std::vector<int>*   back = nullptr;
    };
    bool SearchBatch(const std::vector<Job>& jobs, int dim, std::string* err);

private:
    const ComputeKernel* Kernel(int dim, int kp, bool back, std::string* err);
    const ComputeKernel* MergeKernel(std::string* err);
    const ComputeKernel* ColMergeKernel(std::string* err);

    ComputeContext* m_gpu = nullptr;
    // [back][k == 1 ? 1 : 0][dim / 4], from SharedKernel.
    const ComputeKernel* m_kernel[2][2][65] = {};
    const ComputeKernel* m_merge = nullptr;
    const ComputeKernel* m_colMerge = nullptr;
};

} // namespace tglab
