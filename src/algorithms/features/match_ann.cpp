// match_ann — approximate nearest neighbour matching.
//
// Same job as match_brute, trading exactness for speed. The trade is the whole
// point, so it is measured rather than asserted: `recall` in the run report is
// the fraction of brute force's matches this reproduces, and a script can raise
// `checks` until that is where it wants it.
//
// TWO STRUCTURES, because one does not cover both descriptor kinds:
//
//   FLOAT -> a randomised k-d FOREST. A single k-d tree is the textbook answer
//   and is nearly useless at 128 dimensions: the curse of dimensionality means
//   backtracking visits most of the tree, so an "exact" search costs about what
//   the linear scan costs. Muja & Lowe's fix, and FLANN's: build SEVERAL trees
//   whose splits are randomised among the highest-variance dimensions, then
//   search them together under one shared budget of leaf visits. Each tree is
//   wrong in a different direction, so a point missed by one is usually found
//   by another, and the budget bounds the total work regardless.
//
//   BINARY -> LSH. A k-d tree over Hamming space is not merely inefficient, it
//   is ill-defined: there is no ordering along a single bit to split on.
//   Locality-sensitive hashing instead -- key each descriptor by a random
//   subset of its bits, so two descriptors differing in few bits usually land
//   in the same bucket. Several independent tables, because any one of them can
//   split a true pair apart on a bit where they happen to differ.
//
// The ratio test and cross check work exactly as in match_brute, on whatever
// candidates come back. That matters: the approximation is in WHICH candidates
// are considered, not in how they are judged, so a match this returns is as
// trustworthy as the same match from brute force. What approximation costs is
// RECALL -- matches missed entirely -- not precision.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../algo_util/features.h"
#include "../../algo_util/gpu_knn.h"
#include "../../algo_util/loma.h"
#include "../../core/algorithm.h"
#include "../../core/parallel.h"
#include "gpu_pyramid.h"

namespace tglab {
namespace {

struct Candidate {
    int   index = -1;
    float dist  = std::numeric_limits<float>::max();
};

// Best and second-best over a candidate list, with the same "runner-up must be
// a different feature" rule match_brute uses -- see there for why.
struct Best {
    int   index = -1;
    float d1 = std::numeric_limits<float>::max();
    float d2 = std::numeric_limits<float>::max();
};

bool FarEnough(const std::vector<Keypoint>& kps, int a, int b, float sepSq) {
    if (sepSq <= 0.0f) return true;
    if (a < 0 || b < 0 || size_t(a) >= kps.size() || size_t(b) >= kps.size()) return true;
    const float dx = kps[size_t(a)].x - kps[size_t(b)].x;
    const float dy = kps[size_t(a)].y - kps[size_t(b)].y;
    return dx * dx + dy * dy >= sepSq;
}

Best PickBest(const std::vector<Candidate>& cands,
              const std::vector<Keypoint>& kps, float sepSq) {
    Best b;
    for (const Candidate& c : cands)
        if (c.dist < b.d1) { b.d1 = c.dist; b.index = c.index; }
    if (b.index < 0) return b;
    for (const Candidate& c : cands) {
        if (c.index == b.index) continue;
        if (!FarEnough(kps, b.index, c.index, sepSq)) continue;
        if (c.dist < b.d2) b.d2 = c.dist;
    }
    return b;
}

// --- randomised k-d forest, for float descriptors ---------------------------

struct KdNode {
    int   dim   = -1;      // split dimension; -1 marks a leaf
    float value = 0.0f;
    int   left  = -1;
    int   right = -1;
    int   begin = 0, end = 0;   // leaf: range into that tree's index array
};

// A branch not taken, kept to come back to. `dist` is how far the query is from
// the split plane, which is a lower bound on how much better anything down that
// branch could be -- so exploring the smallest first is exploring the most
// promising first.
struct Branch {
    int   node = -1;
    int   tree = 0;
    float dist = 0.0f;
};

class KdForest {
public:
    void Build(const DescriptorSet& set, int trees, int leafSize, uint32_t seed) {
        m_set  = &set;
        m_dim  = set.dim;
        m_n    = int(set.Count());
        m_leaf = std::max(1, leafSize);
        m_roots.clear();
        m_nodes.clear();
        m_indices.clear();
        if (m_n == 0 || m_dim == 0) return;

        std::mt19937 rng(seed);

        // Variance per dimension, from a sample. Exact variance over the whole
        // set is not worth it: the split only has to be reasonable, and the
        // randomisation below deliberately does not take the best one anyway.
        std::vector<float> var;
        var.assign(size_t(m_dim), 0.0f);
        {
            const int sample = std::min(m_n, 128);
            std::vector<float> mean;
            mean.assign(size_t(m_dim), 0.0f);
            for (int i = 0; i < sample; ++i) {
                const float* p = set.FloatAt(size_t(i));
                for (int d = 0; d < m_dim; ++d) mean[size_t(d)] += p[d];
            }
            for (float& m : mean) m /= float(sample);
            for (int i = 0; i < sample; ++i) {
                const float* p = set.FloatAt(size_t(i));
                for (int d = 0; d < m_dim; ++d) {
                    const float e = p[d] - mean[size_t(d)];
                    var[size_t(d)] += e * e;
                }
            }
        }

        // Splits are drawn at random from the top few dimensions by variance.
        //
        // Randomised rather than always the best, and that is the whole idea:
        // identical trees would make extra trees pure cost, since they would
        // all miss the same points. Five candidates is FLANN's value -- fewer
        // and the trees repeat each other, more and the splits stop separating
        // anything.
        // resize() rather than a sized constructor: the latter parses as a
        // function declaration and the errors land on the uses.
        std::vector<int> cand;
        cand.resize(size_t(m_dim));
        std::iota(cand.begin(), cand.end(), 0);
        const size_t keep = std::min<size_t>(5, cand.size());
        std::partial_sort(cand.begin(), cand.begin() + keep, cand.end(),
                          [&](int a, int b) { return var[size_t(a)] > var[size_t(b)]; });
        cand.resize(keep);

        m_indices.resize(size_t(trees) * size_t(m_n));
        for (int t = 0; t < trees; ++t) {
            const int base = t * m_n;
            for (int i = 0; i < m_n; ++i) m_indices[size_t(base + i)] = i;
            m_roots.push_back(BuildNode(base, base + m_n, cand, rng));
        }
    }

    // Collects candidates until `checks` points have been examined.
    //
    // ONE shared budget across all trees, rather than checks/trees each. That
    // is what lets a tree whose split happened to separate the query from its
    // neighbour spend nothing more while another tree finds it.
    void Search(const float* q, int checks, std::vector<Candidate>* out) const {
        out->clear();
        if (m_n == 0) return;

        std::vector<Branch> queue;
        std::vector<uint8_t> seen(size_t(m_n), 0);
        int visited = 0;

        for (size_t t = 0; t < m_roots.size(); ++t)
            Descend(m_roots[t], int(t), q, &queue, &seen, &visited, checks, out);

        while (visited < checks && !queue.empty()) {
            const auto it = std::min_element(
                queue.begin(), queue.end(),
                [](const Branch& a, const Branch& b) { return a.dist < b.dist; });
            const Branch b = *it;
            queue.erase(it);
            Descend(b.node, b.tree, q, &queue, &seen, &visited, checks, out);
        }
    }

private:
    int BuildNode(int begin, int end, const std::vector<int>& cand, std::mt19937& rng) {
        const int id = int(m_nodes.size());
        m_nodes.push_back(KdNode{});

        if (end - begin <= m_leaf) {
            m_nodes[size_t(id)].dim   = -1;
            m_nodes[size_t(id)].begin = begin;
            m_nodes[size_t(id)].end   = end;
            return id;
        }

        std::uniform_int_distribution<size_t> pick(0, cand.size() - 1);
        const int d = cand[pick(rng)];

        // The MEDIAN, not the midpoint of the value range: a midpoint split on
        // skewed data puts nearly everything on one side and the tree
        // degenerates into a list.
        const int mid = begin + (end - begin) / 2;
        std::nth_element(m_indices.begin() + begin, m_indices.begin() + mid,
                         m_indices.begin() + end,
                         [&](int a, int b) {
                             return m_set->FloatAt(size_t(a))[d] <
                                    m_set->FloatAt(size_t(b))[d];
                         });

        const float value = m_set->FloatAt(size_t(m_indices[size_t(mid)]))[d];
        const int l = BuildNode(begin, mid, cand, rng);
        const int r = BuildNode(mid, end, cand, rng);

        // Written AFTER the recursive calls: BuildNode pushes to m_nodes, which
        // can reallocate, so a reference taken before them would dangle.
        m_nodes[size_t(id)].dim   = d;
        m_nodes[size_t(id)].value = value;
        m_nodes[size_t(id)].left  = l;
        m_nodes[size_t(id)].right = r;
        return id;
    }

    void Descend(int node, int tree, const float* q, std::vector<Branch>* queue,
                 std::vector<uint8_t>* seen, int* visited, int checks,
                 std::vector<Candidate>* out) const {
        while (node >= 0 && *visited < checks) {
            const KdNode& n = m_nodes[size_t(node)];
            if (n.dim < 0) {
                for (int i = n.begin; i < n.end; ++i) {
                    const int idx = m_indices[size_t(i)];
                    // Deduplicated across trees: the same point in several
                    // trees would otherwise burn the budget several times over.
                    if ((*seen)[size_t(idx)]) continue;
                    (*seen)[size_t(idx)] = 1;
                    out->push_back({idx, DistanceL2Sq(q, m_set->FloatAt(size_t(idx)),
                                                      m_dim)});
                    ++(*visited);
                    if (*visited >= checks) return;
                }
                return;
            }

            const float diff = q[n.dim] - n.value;
            const int nearSide = (diff < 0.0f) ? n.left : n.right;
            const int farSide  = (diff < 0.0f) ? n.right : n.left;
            // The far branch cannot hold anything closer than |diff|, so that
            // is its priority when it is reconsidered.
            queue->push_back({farSide, tree, diff * diff});
            node = nearSide;
        }
    }

    const DescriptorSet* m_set = nullptr;
    int m_dim = 0, m_n = 0, m_leaf = 8;
    std::vector<int>    m_roots;
    std::vector<KdNode> m_nodes;
    std::vector<int>    m_indices;
};

// --- LSH, for binary descriptors --------------------------------------------

class LshTables {
public:
    // `tables` hash tables, each keyed by `keyBits` bit positions drawn at
    // random from the descriptor.
    //
    // Two descriptors differing in few bits agree on a random subset with high
    // probability -- (1 - p)^keyBits for a per-bit disagreement rate p -- so
    // they collide. Several tables because any one can pick a bit where a true
    // pair happens to differ; the chance ALL of them do falls off fast.
    void Build(const DescriptorSet& set, int tables, int keyBits, uint32_t seed) {
        m_set = &set;
        m_bytes = set.BytesPerBinary();
        m_tables.clear();
        m_bits.clear();
        const size_t n = set.Count();
        if (n == 0 || set.dim <= 0) return;

        // 20 bits caps the key at a million buckets, which is already far more
        // than any realistic feature count -- beyond that the tables are mostly
        // empty and every query misses.
        keyBits = std::clamp(keyBits, 4, 20);

        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> bit(0, set.dim - 1);

        for (int t = 0; t < tables; ++t) {
            std::vector<int> chosen;
            chosen.resize(size_t(keyBits));
            for (int& b : chosen) b = bit(rng);
            m_bits.push_back(chosen);

            std::unordered_map<uint32_t, std::vector<int>> table;
            for (size_t i = 0; i < n; ++i)
                table[Key(set.BinaryAt(i), chosen)].push_back(int(i));
            m_tables.push_back(std::move(table));
        }
    }

    void Search(const uint8_t* q, int checks, std::vector<Candidate>* out) const {
        out->clear();
        if (m_tables.empty()) return;

        std::vector<uint8_t> seen(m_set->Count(), 0);
        int visited = 0;

        // MULTI-PROBE: the exact bucket, then the buckets one bit away.
        //
        // Probing only the exact bucket is what made this matcher unusable.
        // Measured: 262 matches of which 138 were correct, against brute
        // force's 155 of 153 -- and the damage was concentrated in one band,
        // 121 matches at ratio 0.7-0.8 with exactly 1 correct among them.
        //
        // The mechanism is worth understanding because it is the one way an
        // approximate matcher can lose PRECISION rather than only recall: when
        // the true neighbour is not in the bucket, some mediocre candidate
        // becomes the "best" and an even worse one becomes the runner-up. The
        // ratio between two wrong answers can look perfectly respectable, so
        // the ratio test cannot catch it -- the test assumes the best candidate
        // is at least a serious contender.
        //
        // Flipping one key bit reaches the buckets holding descriptors that
        // differ from the query in exactly that bit, which is where a true
        // neighbour differing in a handful of bits most often ends up. That
        // costs keyBits extra lookups per table and recovers the neighbour in
        // most of the cases the exact probe missed.
        for (size_t t = 0; t < m_tables.size() && visited < checks; ++t) {
            const uint32_t base = Key(q, m_bits[t]);

            auto probe = [&](uint32_t key) {
                const auto it = m_tables[t].find(key);
                if (it == m_tables[t].end()) return;
                for (int idx : it->second) {
                    if (seen[size_t(idx)]) continue;
                    seen[size_t(idx)] = 1;
                    out->push_back({idx,
                                    float(DistanceHamming(q,
                                            m_set->BinaryAt(size_t(idx)), m_bytes))});
                    if (++visited >= checks) return;
                }
            };

            probe(base);
            for (size_t b = 0; b < m_bits[t].size() && visited < checks; ++b)
                probe(base ^ (1u << b));
        }
    }

private:
    static uint32_t Key(const uint8_t* d, const std::vector<int>& bits) {
        uint32_t k = 0;
        for (size_t i = 0; i < bits.size(); ++i) {
            const int b = bits[i];
            if (d[b / 8] & (1u << (b % 8))) k |= (1u << i);
        }
        return k;
    }

    const DescriptorSet* m_set = nullptr;
    int m_bytes = 0;
    std::vector<std::unordered_map<uint32_t, std::vector<int>>> m_tables;
    std::vector<std::vector<int>> m_bits;
};

class MatchAnn : public AlgorithmBase {
public:
    const char* Name()     const override { return "match_ann"; }
    const char* Category() const override { return "match"; }

    PortList Inputs() const override {
        return {{"src", DataType::ImageSet, FormatSpec::Any, ShapeSpec::Any}};
    }
    PortList Outputs() const override {
        return {{"out", DataType::ImageSet, FormatSpec::SameAsInput, ShapeSpec::SameAsInput}};
    }

    void RunCPU(RunCtx&) override {}
    bool IsAligner() const override { return true; }

    bool RunAlign(std::vector<Image>* images, std::string* err) override {
        m_pairs = 0;
        m_total = 0;
        m_kept  = 0;
        m_note.clear();

        if (!images || images->size() < 2) {
            *err = "match_ann needs a group of at least two images";
            return false;
        }

        const int fixedRef = std::clamp(int(m_reference), 0, int(images->size()) - 1);
        const bool chain   = bool(m_chain);

        if (!chain) {
            const FeatureSidecar* r0 = FeaturesOf((*images)[size_t(fixedRef)]);
            if (!r0) {
                *err = "match_ann: the reference image has no features -- "
                       "run a detector before matching";
                return false;
            }
            if (r0->keypoints.empty()) {
                m_note = "the reference image has no features to match against";
                return true;
            }
        }

        // An index is built ONCE per frame that serves as a reference and
        // queried by every frame matching against it: with a fixed reference
        // that is one build over N-1 searches, and in a chain each frame's
        // index serves its `window` successors and any revisits.
        bool isFloat = false;

        // See match_brute's `window`: a chain alone holds no constraint
        // relating frame 8 to frame 5, so bundle adjustment has nothing to
        // correct drift with unless each frame matches several neighbours.
        const int window = chain ? std::max(1, int(m_window)) : 1;

        // REVISITS: frames far apart in the sequence that see the same thing,
        // matched as well as the neighbours. See SelectRevisits.
        std::vector<std::vector<int>> revisitFor;
        revisitFor.resize(images->size());
        m_revisits = 0;
        if (chain && int(m_revisit) > 0)
            m_revisits = SelectRevisits(*images, window, int(m_revisit), &revisitFor);

        // Which pairs, decided up front so the indexes can be built for
        // exactly the frames that need them.
        const size_t nImg = images->size();
        const int nNeighbours = chain ? window : 1;
        std::vector<std::vector<std::pair<int, bool>>> refsOf(nImg);   // (reference, revisit)
        std::vector<char> isRef(nImg, 0), isQuery(nImg, 0);
        for (size_t i = 0; i < nImg; ++i) {
            const FeatureSidecar* fs = FeaturesOf((*images)[i]);
            if (!fs || fs->keypoints.empty()) continue;
            const int nTotal = nNeighbours + int(revisitFor[i].size());
            for (int w = 1; w <= nTotal; ++w) {
                const int refIdx = w > nNeighbours ? revisitFor[i][size_t(w - nNeighbours - 1)]
                                 : chain ? int(i) - w : fixedRef;
                if (refIdx < 0 || int(i) == refIdx) continue;
                const FeatureSidecar* ref = FeaturesOf((*images)[size_t(refIdx)]);
                if (!ref || ref->keypoints.empty()) continue;
                if (fs->descriptors.kind != ref->descriptors.kind ||
                    fs->descriptors.dim  != ref->descriptors.dim) {
                    *err = "match_ann: frame " + std::to_string(i) +
                           " has a different descriptor from the reference (" +
                           fs->detector + " vs " + ref->detector +
                           ") -- one detector for the whole group";
                    return false;
                }
                isFloat = ref->descriptors.kind == DescriptorKind::Float;
                refsOf[i].push_back({refIdx, w > nNeighbours});
                isRef[size_t(refIdx)] = 1;
                isQuery[i] = 1;
                if (!chain) break;
            }
        }

        // EXACT, ON THE GPU, for float descriptors when there is a device:
        // every candidate compared, the k nearest kept, then judged by the
        // same rules as below. See GpuKnn for why brute force wins there.
        std::vector<std::shared_ptr<MatchSidecar>> made(nImg);
        m_loma = int(m_method) == 1;
        if (m_loma) {
            if (!MatchAllLoma(*images, refsOf, &made, err)) return false;
            m_exact = false;
        } else {
            m_exact = isFloat && GroupGpu() && MatchAllOnGpu(*images, refsOf, &made);
        }
        if (!m_exact && !m_loma) {
            for (auto& m : made) m.reset();

            // EACH FRAME'S INDEX BUILT ONCE, in parallel, rather than once per
            // pair: a chain of window 2 with 3 revisits rebuilt every frame's
            // index about five times, plus a reverse index per pair for the cross
            // check. The forward and reverse indexes keep their own seeds, so the
            // matches are exactly what the pair-by-pair loop produced -- in about
            // a sixth of the time on a 100-frame video (49 s to 8 s).
            std::vector<KdForest>  fwdF(nImg), revF(nImg);
            std::vector<LshTables> fwdL(nImg), revL(nImg);
            const bool cross = bool(m_crossCheck);
            ParallelFor(nImg, [&](size_t i) {
                const FeatureSidecar* fs = FeaturesOf((*images)[i]);
                if (!fs) return;
                const int trees = std::max(1, int(m_trees));
                if (isRef[i]) {
                    if (isFloat) fwdF[i].Build(fs->descriptors, trees, 8, 12345u);
                    else         fwdL[i].Build(fs->descriptors, trees, int(m_keyBits), 12345u);
                }
                if (isQuery[i] && cross) {
                    if (isFloat) revF[i].Build(fs->descriptors, trees, 8, 54321u);
                    else         revL[i].Build(fs->descriptors, trees, int(m_keyBits), 54321u);
                }
            });

            ParallelFor(nImg, [&](size_t i) {
                if (refsOf[i].empty()) return;
                const FeatureSidecar* fs = FeaturesOf((*images)[i]);
                auto ms = std::make_shared<MatchSidecar>();
                for (const auto& [refIdx, revisit] : refsOf[i]) {
                    const FeatureSidecar* ref = FeaturesOf((*images)[size_t(refIdx)]);
                    MatchSet set;
                    set.reference = refIdx;
                    set.revisit = revisit;
                    MatchPair(*ref, *fs, fwdF[size_t(refIdx)], fwdL[size_t(refIdx)],
                              revF[i], revL[i], isFloat, &set);
                    ms->considered += set.considered;
                    ms->sets.push_back(std::move(set));
                }
                made[i] = std::move(ms);
            });
        }

        for (size_t i = 0; i < nImg; ++i) {
            if (!made[i] || made[i]->sets.empty()) continue;
            for (const MatchSet& set : made[i]->sets) {
                m_total += set.considered;
                m_kept  += int(set.matches.size());
                ++m_pairs;
            }
            made[i]->matcher = m_loma ? "loma" : m_exact ? "exact (GPU)"
                             : isFloat ? "ann (kd-forest)" : "ann (lsh)";
            (*images)[i].Sidecars().Set(kMatchSidecar, made[i]);
        }
        return true;
    }

    std::string RunReport() const override {
        if (!m_note.empty()) return m_note;
        if (m_pairs == 0) return {};
        char how[64];
        if (m_loma)
            std::snprintf(how, sizeof how, "with LoMa (%d layers%s)", m_lomaLayers,
                          m_lomaOnGpu ? "" : ", on the CPU");
        else
            std::snprintf(how, sizeof how, "%s", m_exact ? "exactly, on the GPU" : "approximately");
        char buf[192];
        std::snprintf(buf, sizeof buf, "matched %d pair%s %s: %d of %d candidates kept", m_pairs,
                      m_pairs == 1 ? "" : "s", how, m_kept, m_total);
        std::string s = buf;
        if (m_revisits > 0) {
            s += "; " + std::to_string(m_revisits) + " of them revisits -- far apart "
                 "in the sequence, alike in view";
            if (!m_revisitGaps.empty()) {
                std::vector<int> g = m_revisitGaps;
                std::nth_element(g.begin(), g.begin() + long(g.size() / 2), g.end());
                s += " (typically " + std::to_string(g[g.size() / 2]) + " frames apart)";
            }
        }
        return s;
    }

    // Sidecar coordinates are in IMAGE PIXELS and nothing rescales them, so a
    // proxy run would hand every downstream stage positions that are wrong by
    // the scale factor -- silently, since the sidecar is still present and
    // still looks valid. See AlgorithmBase::Proxy.
    ProxyBehaviour Proxy() const override { return ProxyBehaviour::Never; }

    bool HasGPU() const override { return false; }

private:
    void MatchPair(const FeatureSidecar& ref, const FeatureSidecar& other,
                   const KdForest& forest, const LshTables& lsh,
                   const KdForest& revForest, const LshTables& revLsh, bool isFloat,
                   MatchSet* out) const {
        const int checks = std::max(1, int(m_checks));

        // The reverse direction, for the cross check.
        //
        // Omitting this is what left 122 matches at ratio 0.7-0.8 with one
        // correct among them, while brute force -- which cross checks by
        // default -- had a single match in that band. Those are features with
        // NO true partner in the other image: they still have a nearest
        // neighbour, and the ratio between two equally wrong candidates can
        // look respectable, so the ratio test cannot see the problem. Requiring
        // the pairing to be mutual is what catches it.
        //
        // Built by searching the same structures in reverse, so it is another
        // N approximate queries rather than an exact scan -- the point of this
        // matcher is that nothing in it is O(n*m). The reverse index over
        // `other` is built once per frame by the caller.
        std::vector<int> backBest;
        if (bool(m_crossCheck)) {
            std::vector<Candidate> rc;
            backBest.assign(ref.descriptors.Count(), -1);

            for (size_t i = 0; i < ref.descriptors.Count(); ++i) {
                if (isFloat) revForest.Search(ref.descriptors.FloatAt(i), checks, &rc);
                else         revLsh.Search(ref.descriptors.BinaryAt(i), checks, &rc);
                float best = std::numeric_limits<float>::max();
                for (const Candidate& c : rc)
                    if (c.dist < best) { best = c.dist; backBest[i] = c.index; }
            }
        }

        Judge(ref, other, isFloat, backBest, out, [&](size_t i, std::vector<Candidate>* cands) {
            if (isFloat) forest.Search(other.descriptors.FloatAt(i), checks, cands);
            else         lsh.Search(other.descriptors.BinaryAt(i), checks, cands);
        });
    }

    // THE JUDGING, shared by every way of finding candidates -- the forest,
    // LSH, or the exact GPU search -- so that how a match is accepted never
    // depends on how its candidates were found. `candidatesFor(i, &c)` fills
    // the candidates for query i; `backBest` is the reverse direction's best,
    // for the cross check (empty for none).
    template <class CandidatesFor>
    void Judge(const FeatureSidecar& ref, const FeatureSidecar& other, bool isFloat,
               const std::vector<int>& backBest, MatchSet* out,
               CandidatesFor&& candidatesFor) const {
        const float maxRatio = float(m_ratio);
        const float sepSq    = float(m_minSeparation) * float(m_minSeparation);
        std::vector<Candidate> cands;
        const size_t n = other.descriptors.Count();

        for (size_t i = 0; i < n; ++i) {
            candidatesFor(i, &cands);
            if (cands.empty()) continue;

            const Best b = PickBest(cands, ref.keypoints, sepSq);
            if (b.index < 0) continue;

            // NO RUNNER-UP MEANS NO RATIO TEST, so the match is dropped rather
            // than kept.
            //
            // This is the one place approximation can cost PRECISION rather
            // than only recall, and it bit hard: an LSH bucket holding a single
            // candidate leaves d2 at infinity, the ratio computes as ~0, and
            // every such match sails through unexamined. Measured before this
            // check: 285 matches of which 157 were correct, against brute
            // force's 155 of 157 -- nearly twice as many at half the precision.
            //
            // Dropping them is right rather than merely safe: a match that
            // cannot be shown to be unambiguous has not earned the same
            // confidence as one that has, and the whole value of this matcher
            // is that its output is as trustworthy as brute force's.
            if (b.d2 >= std::numeric_limits<float>::max() * 0.5f) continue;

            ++out->considered;

            // Same units trap as match_brute: L2 is stored squared, so the
            // ratio of the stored values is the SQUARE of the ratio of the
            // distances, and comparing it to 0.8 would silently test 0.64.
            float ratio;
            if (isFloat) ratio = (b.d2 > 0.0f) ? std::sqrt(b.d1 / b.d2) : 0.0f;
            else         ratio = (b.d2 > 0.0f) ? (b.d1 / b.d2) : 0.0f;
            if (ratio > maxRatio) continue;

            // Mutual best, as in match_brute.
            if (!backBest.empty() && backBest[size_t(b.index)] != int(i)) continue;

            Match m;
            m.a = b.index;
            m.b = int(i);
            m.distance = isFloat ? std::sqrt(b.d1) : b.d1;
            m.ratio = ratio;
            out->matches.push_back(m);
        }
    }

    static constexpr const char* kMethodNames[] = {"nearest neighbour", "LoMa (learned)"};

    Param<int> m_method{this, "method", 0, 0, 1,
        {.help = "Nearest neighbour compares descriptors two at a time, with "
                 "the ratio test and cross check below. LoMa is a transformer "
                 "trained to match: each point's description is rewritten by "
                 "attention to the rest of its own image and then to the "
                 "other image before any comparison, so repeated structure "
                 "-- a facade of identical windows -- is resolved by context "
                 "rather than thrown away as ambiguous. Needs describe_dedode "
                 "with normalise = 0, and is best with a few thousand "
                 "keypoints a frame: its cost grows with their square.",
         .choices = kMethodNames, .choiceCount = 2}};

    Param<int> m_lomaLayersParam{this, "layers", 9, 1, 9,
        {.help = "LoMa: how many of its nine transformer layers to run. Every "
                 "layer has its own trained matching head, so fewer is a "
                 "real, cheaper matcher rather than a broken one -- LoMa's "
                 "paper reports 3 and 5 as useful speed/accuracy points."}};

    // 0.9, NOT LoMa's own 0.1, and measured: LoMa's default maximises matches
    // for one pair's pose, but SfM chains matches into multi-view TRACKS, and
    // a low-confidence match -- a point paired with a neighbour of its true
    // partner, which the network will do when the partner was not detected --
    // conflicts with the same point's matches in other pairs and breaks the
    // track. At 4096 DaD points a frame, sfm.tgl's sparse chain:
    //
    //   threshold     castle-P19 points   fountain-P11 points   IMG_1535 points
    //     0.1               6227                3038                   --
    //     0.5               7887                3809                 13555
    //     0.8               8621                4479                 22685
    //     0.9               8765                4742                 25385
    //   (nearest neighbour  7967                4557                 26297)
    Param<float> m_lomaThreshold{this, "threshold", 0.9f, 0.0f, 1.0f,
        {.help = "LoMa: the dual-softmax probability a mutual best match "
                 "must exceed to be kept. Higher than LoMa's own 0.1 because "
                 "SfM chains matches into tracks across many pairs, and an "
                 "unsure match breaks a track: measured, raising it from 0.1 "
                 "to 0.9 took castle-P19 from 6227 points to 8765."}};

    Param<int> m_reference{this, "reference", 0, 0, 64,
        {.help = "Which frame the others are matched against. Ignored when "
                 "`chain` is on."}};

    // See match_brute's `chain` for the measurement that motivates this: along
    // a 15-frame sweep, overlap with frame 0 decays to 2% and the solve fails,
    // while consecutive neighbours hold throughout.
    //
    // Note the cost asymmetry against brute force. The index build is amortised
    // over N-1 queries with a fixed reference, and not amortised at all in a
    // chain -- so if the group is chained, measure before assuming this matcher
    // is the faster one.
    Param<int> m_window{this, "window", 1, 1, 8,
        {.help = "In chain mode, how many previous frames each one matches "
                 "against. See match_brute's window: 1 is a plain chain, and "
                 "more is what bundle adjustment needs to undo accumulated "
                 "drift. Each extra step costs another index build here, "
                 "since the reference changes every pair."}};

    Param<bool> m_chain{this, "chain", false,
        "Match each frame to the one before it instead of to a fixed "
        "reference. What a panorama needs. Costs this matcher its main "
        "advantage -- the index must be rebuilt for every pair."};

    Param<int> m_checks{this, "checks", 256, 1, 4000,
        {.help = "How many candidates to examine per query. THE speed/accuracy "
                 "control: higher finds more of what brute force would, and "
                 "costs proportionally. Measured against exact matching on two "
                 "45 MP frames with 2000 AKAZE features: 64 checks recovers "
                 "30% of the exact matches, 256 recovers 70%, 1024 recovers "
                 "78%. Raise it until recall stops climbing."}};

    Param<int> m_trees{this, "trees", 4, 1, 16,
        {.help = "Independent search structures -- k-d trees for float "
                 "descriptors, hash tables for binary. Each is wrong in a "
                 "different direction, so a point one misses another usually "
                 "finds."}};

    Param<int> m_keyBits{this, "lsh_bits", 12, 4, 20,
        {.help = "Bits per LSH key, for binary descriptors only. More bits "
                 "means smaller buckets: faster and likelier to miss. Ignored "
                 "for float descriptors, which use the k-d forest."}};

    Param<float> m_ratio{this, "ratio", 0.8f, 0.1f, 1.0f,
        {.help = "Lowe's ratio test, exactly as in match_brute. The "
                 "approximation is in which candidates are CONSIDERED, not in "
                 "how they are judged.",
         .step = 0.01, .softMin = 0.6, .softMax = 0.9}};

    Param<bool> m_crossCheck{this, "cross_check", true,
        "Require the match to be mutual. Catches features with NO true partner "
        "in the other image, which the ratio test cannot: such a feature still "
        "has a nearest neighbour, and the ratio between two equally wrong "
        "candidates can look respectable. Costs a second set of approximate "
        "queries."};

    Param<float> m_minSeparation{this, "min_separation", 3.0f, 0.0f, 20.0f,
        {.help = "How far apart two candidates must be for the second to count "
                 "as the runner-up. See match_brute."},
    };

    // REVISITS, which a chain alone never matches: frames far apart in the
    // sequence that see the same thing. The end of a walk around an object
    // is back where it started; a sweep that doubles back passes the same
    // view twice. Matched only to their neighbours, those frames are joined
    // by a long thin chain, along which scale and position errors
    // accumulate with nothing to close them -- measured on a 100-frame video
    // walking a full circle round a subject, the camera path opened into a
    // spiral whose end landed three radii from its start.
    //
    // FOUND BY VOTING, not by trying every pair: a sample of each frame's
    // descriptors goes into one index over the whole group, each sample
    // votes for the frame its nearest match came from (when that match
    // clearly beats the runner-up), and a pair's score is its votes in both
    // directions. A frame's revisit candidates are the highest-scoring
    // frames outside its neighbour window, kept when they reach a quarter of
    // the typical NEIGHBOUR score -- relative, so the threshold means the same
    // at any feature count or scene. The chosen pairs then go through the
    // same matching as the neighbours, and relative_pose decides whether
    // each one is real.
    int SelectRevisits(const std::vector<Image>& images, int window, int perFrame,
                       std::vector<std::vector<int>>* out) const {
        const int n = int(images.size());
        const size_t kSample = 300;
        DescriptorSet pool;
        std::vector<int> owner;
        std::vector<std::vector<size_t>> mine;   // each frame's rows in the pool
        mine.resize(size_t(n));
        for (int f = 0; f < n; ++f) {
            const FeatureSidecar* fs = FeaturesOf(images[size_t(f)]);
            if (!fs || fs->descriptors.Count() == 0) continue;
            const DescriptorSet& d = fs->descriptors;
            if (pool.kind == DescriptorKind::None) { pool.kind = d.kind; pool.dim = d.dim; }
            if (d.kind != pool.kind || d.dim != pool.dim) return 0;
            // The STRONGEST responses, not every n-th keypoint: a stride picks
            // a different 3% of a 10000-keypoint frame from each frame, so two
            // frames of the same view rarely sampled the same corner -- a
            // frame and its neighbour shared about 5 votes of 600. The
            // strongest corners are the ones every view of a place detects.
            const size_t cnt = d.Count();
            std::vector<size_t> order(cnt);
            for (size_t k = 0; k < cnt; ++k) order[k] = k;
            if (fs->keypoints.size() == cnt && cnt > kSample) {
                std::partial_sort(order.begin(), order.begin() + long(kSample), order.end(),
                                  [&](size_t a, size_t b) {
                                      return fs->keypoints[a].response > fs->keypoints[b].response;
                                  });
            } else {
                const size_t stride = std::max<size_t>(1, cnt / kSample);
                order.clear();
                for (size_t k = 0; k < cnt; k += stride) order.push_back(k);
            }
            for (size_t j = 0; j < order.size() && mine[size_t(f)].size() < kSample; ++j) {
                const size_t k = order[j];
                mine[size_t(f)].push_back(owner.size());
                owner.push_back(f);
                if (d.kind == DescriptorKind::Float)
                    pool.f.insert(pool.f.end(), d.FloatAt(k), d.FloatAt(k) + d.dim);
                else
                    pool.b.insert(pool.b.end(), d.BinaryAt(k),
                                  d.BinaryAt(k) + d.BytesPerBinary());
            }
        }
        if (owner.empty()) return 0;
        const bool isFloat = pool.kind == DescriptorKind::Float;
        KdForest forest;
        LshTables lsh;
        if (isFloat) forest.Build(pool, std::max(1, int(m_trees)), 8, 777u);
        else         lsh.Build(pool, std::max(1, int(m_trees)), int(m_keyBits), 777u);

        // THE RUNNER-UP COMES FROM ANOTHER PLACE. A video's consecutive frames
        // carry nearly the same descriptor for the same corner, so the best
        // and second-best matches of a sample are usually two adjacent frames
        // at almost the same distance, and a plain ratio test rejects nearly
        // every vote: measured on a 1576-frame room scan, a frame and its
        // neighbour shared about 5 votes of 600, and the revisits chosen from
        // scores that small were noise. The runner-up is taken from frames
        // outside the winner's own stretch (and the asking frame's), so the
        // test asks what it should -- is this place clearly the one -- rather
        // than which of two near-identical frames is closer.
        std::vector<float> votes(size_t(n) * size_t(n), 0.0f);
        const float kStretchShare = 0.25f;   // of a frame's score with its neighbour
        const float ratio = float(m_ratio);
        const int checks = std::max(1, int(m_checks));
        const int place = 2 * window + 4;   // before the stretches are known
        auto vote = [&](std::vector<float>* into, std::vector<float>* soft, auto&& excluded,
                        auto&& samePlace) {
            std::fill(into->begin(), into->end(), 0.0f);
            // Each frame writes only its own row of votes.
            ParallelFor(size_t(n), [&](size_t fi) {
                const int f = int(fi);
                if (GroupCancelled()) return;   // superseded: see SetGroupCancel
                std::vector<Candidate> c;
                std::vector<int> hit;
                for (size_t row : mine[size_t(f)]) {
                    if (isFloat) forest.Search(pool.FloatAt(row), checks, &c);
                    else         lsh.Search(pool.BinaryAt(row), checks, &c);
                    float b1 = std::numeric_limits<float>::max();
                    int who = -1;
                    for (const Candidate& cd : c) {
                        if (cd.index < 0) continue;
                        const int g = owner[size_t(cd.index)];
                        if (g == f || excluded(f, g)) continue;
                        if (cd.dist < b1) { b1 = cd.dist; who = g; }
                    }
                    if (who < 0) continue;
                    // SOFT: every frame whose copy of the corner is about as
                    // close as the best -- no competition between frames, so
                    // it falls off with time only as the view changes.
                    if (soft) {
                        const float aboutAsClose = b1 / ratio;
                        hit.clear();
                        for (const Candidate& cd : c) {
                            if (cd.index < 0 || cd.dist > aboutAsClose) continue;
                            const int g = owner[size_t(cd.index)];
                            if (g == f || std::find(hit.begin(), hit.end(), g) != hit.end()) continue;
                            hit.push_back(g);
                            (*soft)[size_t(f) * size_t(n) + size_t(g)] += 1.0f;
                        }
                    }
                    float b2 = std::numeric_limits<float>::max();
                    for (const Candidate& cd : c) {
                        if (cd.index < 0) continue;
                        const int g = owner[size_t(cd.index)];
                        if (g == f || excluded(f, g) || samePlace(f, who, g)) continue;
                        b2 = std::min(b2, cd.dist);
                    }
                    if (b1 < ratio * b2) (*into)[size_t(f) * size_t(n) + size_t(who)] += 1.0f;
                }
            });
        };
        // First over every frame, for the neighbour scale and each frame's
        // own stretch.
        std::vector<float> soft(size_t(n) * size_t(n), 0.0f);
        vote(&votes, &soft, [](int, int) { return false; },
             [&](int f, int w, int g) { return std::abs(g - w) <= place || std::abs(g - f) <= place; });
        auto score = [&](int a, int b) {
            return votes[size_t(a) * size_t(n) + size_t(b)] + votes[size_t(b) * size_t(n) + size_t(a)];
        };

        // The typical NEIGHBOUR score, which every revisit is judged against.
        std::vector<float> nb;
        for (int f = 1; f < n; ++f) nb.push_back(score(f, f - 1));
        if (nb.empty()) return 0;
        std::nth_element(nb.begin(), nb.begin() + long(nb.size() / 2), nb.end());
        const float bar = 0.25f * nb[nb.size() / 2];

        // A frame's OWN STRETCH of the walk: the frames either side of it,
        // contiguously, that still look like it. A revisit is outside that
        // stretch, not merely outside the matching window -- otherwise the
        // best "revisits" are frames 3 or 4 steps away, which the chain
        // already joins, and every slot goes to them. Measured on a 1576-frame
        // room scan with window 2: the revisits joined no two blocks of 100
        // frames that the chain did not, and frames 600-1300 were one long
        // unclosed chain that came out at twice the scale of the rest, a
        // second copy of the walls beside the first.
        auto softScore = [&](int a, int b) {
            return soft[size_t(a) * size_t(n) + size_t(b)] + soft[size_t(b) * size_t(n) + size_t(a)];
        };
        auto stretch = [&](int f) {
            const float self = std::max(f > 0 ? softScore(f, f - 1) : 0.0f,
                                        f < n - 1 ? softScore(f, f + 1) : 0.0f);
            const float keep = kStretchShare * self;
            int lo = f, hi = f;
            while (lo > 0 && softScore(f, lo - 1) >= keep && softScore(f, lo - 1) > 0.0f) --lo;
            while (hi < n - 1 && softScore(f, hi + 1) >= keep && softScore(f, hi + 1) > 0.0f) ++hi;
            return std::pair<int, int>(std::min(lo, f - window), std::max(hi, f + window));
        };
        std::vector<std::pair<int, int>> own(static_cast<size_t>(n));
        ParallelFor(size_t(n), [&](size_t f) { own[f] = stretch(int(f)); });
        // ...and then twice as wide again: what the soft count measures is
        // where a corner still looks almost the same, and a frame a little
        // beyond that is still the same visit, not a return to it.
        for (int f = 0; f < n; ++f) {
            auto& [lo, hi] = own[size_t(f)];
            lo = std::max(0, f - 2 * (f - lo));
            hi = std::min(n - 1, f + 2 * (hi - f));
        }
        auto inside = [&](int g, int f) { return g >= own[size_t(f)].first && g <= own[size_t(f)].second; };

        // Then again, the asking frame's own stretch left out altogether and
        // the runner-up from outside both stretches: what is left are votes
        // for OTHER visits to the place, which the first pass hands to the
        // neighbours that look most like it.
        std::vector<float> farVotes(size_t(n) * size_t(n), 0.0f);
        vote(&farVotes, nullptr, [&](int f, int g) { return std::abs(g - f) <= window || inside(g, f); },
             [&](int f, int w, int g) { return std::abs(g - w) <= window || inside(g, w) || inside(g, f); });
        auto farScore = [&](int a, int b) {
            return farVotes[size_t(a) * size_t(n) + size_t(b)] + farVotes[size_t(b) * size_t(n) + size_t(a)];
        };

        // Each unordered pair once, stored on the later frame with the earlier
        // as its reference -- the chain's own convention. A frame's revisits
        // go to DIFFERENT places: once a candidate is taken, the rest of its
        // stretch is passed over, so three revisits are three loop closures
        // rather than three adjacent frames of one.
        //
        // A RETURN, NOT THE NEXT STRETCH: the score must DIP between the
        // frame's stretch and the candidate -- the camera left the place and
        // came back. Without that, a frame just past the stretch, which the
        // far vote naturally favours, would still win every slot.
        const float revisitBar = std::max(3.0f, 0.4f * bar);
        std::vector<char> taken(size_t(n) * size_t(n), 0);
        int added = 0;
        m_revisitGaps.clear();
        std::vector<float> dip(static_cast<size_t>(n));
        for (int f = 0; f < n; ++f) {
            const auto [lo, hi] = own[size_t(f)];
            // dip[g]: the lowest score strictly between the stretch and g.
            float m = std::numeric_limits<float>::max();
            for (int g = hi + 1; g < n; ++g) { dip[size_t(g)] = m; m = std::min(m, farScore(f, g)); }
            m = std::numeric_limits<float>::max();
            for (int g = lo - 1; g >= 0; --g) { dip[size_t(g)] = m; m = std::min(m, farScore(f, g)); }
            std::vector<std::pair<float, int>> cand;
            for (int g = 0; g < n; ++g) {
                if (std::abs(g - f) <= window || (g >= lo && g <= hi)) continue;
                const float s = farScore(f, g);
                if (s >= revisitBar && dip[size_t(g)] <= 0.25f * s) cand.emplace_back(s, g);
            }
            std::sort(cand.begin(), cand.end(), std::greater<>());
            std::vector<std::pair<int, int>> used;
            int kept = 0;
            for (size_t k = 0; k < cand.size() && kept < perFrame; ++k) {
                const int g = cand[k].second;
                bool sameStretch = false;
                for (const auto& [ulo, uhi] : used) sameStretch = sameStretch || (g >= ulo && g <= uhi);
                if (sameStretch) continue;
                used.push_back({std::min(own[size_t(g)].first, g - window),
                                std::max(own[size_t(g)].second, g + window)});
                ++kept;
                const int a = std::min(f, g), b = std::max(f, g);
                char& t = taken[size_t(a) * size_t(n) + size_t(b)];
                if (t) continue;
                t = 1;
                (*out)[size_t(b)].push_back(a);
                m_revisitGaps.push_back(b - a);
                ++added;
            }
        }
        return added;
    }

    Param<int> m_revisit{this, "revisit", 0, 0, 8,
        {.help = "In a chain, ALSO match each frame against up to this many "
                 "frames elsewhere in the sequence that look most like it -- "
                 "the end of a walk around an object meeting its start, a "
                 "sweep passing the same view twice. What closes the loop, so "
                 "drift has something to correct against. 0 matches "
                 "neighbours only."}};

    int         m_pairs = 0;
    int         m_total = 0;
    int         m_kept  = 0;
    int         m_revisits = 0;
    mutable std::vector<int> m_revisitGaps;   // frames apart, per revisit
    bool        m_exact = false;   // the last run searched exhaustively on the GPU
    bool        m_loma = false;    // ...or ran LoMa's matcher
    bool        m_lomaOnGpu = false;
    int         m_lomaLayers = 0;
    std::string m_note;

    // Every pair through LoMa's matcher. Errors (missing weights, the wrong
    // descriptors) fail the stage rather than falling back: a matcher
    // quietly swapped for another would make every comparison meaningless.
    bool MatchAllLoma(const std::vector<Image>& images,
                      const std::vector<std::vector<std::pair<int, bool>>>& refsOf,
                      std::vector<std::shared_ptr<MatchSidecar>>* made, std::string* err) {
        const LomaMatcher* net = SharedNetwork<LomaMatcher>(
            "loma_b128.tgw",
            "python tools/nn_convert.py loma_b128 models/loma_b128.tgw "
            "--exclude _detector. --exclude _descriptor.",
            err);
        if (!net) { *err = "match_ann: " + *err; return false; }
        m_lomaLayers = std::clamp(int(m_lomaLayersParam), 1, net->Layers());

        // Each frame's inputs once: positions normalised to -1..1 across the
        // image (pixel centres at +0.5, as describe_dedode samples them) and
        // the descriptors as they are.
        const size_t nImg = images.size();
        std::vector<std::vector<float>> kp(nImg);
        for (size_t i = 0; i < nImg; ++i) {
            const FeatureSidecar* fs = FeaturesOf(images[i]);
            if (!fs) continue;
            const DescriptorSet& d = fs->descriptors;
            if (fs->keypoints.empty()) continue;
            if (d.kind != DescriptorKind::Float || d.dim != net->InputDim()) {
                *err = "match_ann: LoMa needs describe_dedode's " + std::to_string(net->InputDim()) +
                       "-float descriptors, and frame " + std::to_string(i) + " has " + fs->detector + "'s";
                return false;
            }
            // Unit-length descriptors are the default for match_ann's
            // distances, and NOT what the network was trained on -- it would
            // still return matches, quietly worse ones. Caught by their norms.
            double mean = 0.0;
            for (size_t k = 0; k < d.Count(); ++k) {
                double s = 0.0;
                for (int c = 0; c < d.dim; ++c) s += double(d.FloatAt(k)[c]) * d.FloatAt(k)[c];
                mean += std::sqrt(s);
            }
            mean /= double(d.Count());
            if (std::fabs(mean - 1.0) < 1e-3) {
                *err = "match_ann: these descriptors are unit length -- LoMa needs them raw, "
                       "so describe with describe_dedode(normalise = 0)";
                return false;
            }
            const ImageDesc& im = images[i].Desc();
            kp[i].resize(fs->keypoints.size() * 2);
            for (size_t k = 0; k < fs->keypoints.size(); ++k) {
                kp[i][2 * k]     = 2.0f * fs->keypoints[k].x / float(im.width) - 1.0f;
                kp[i][2 * k + 1] = 2.0f * fs->keypoints[k].y / float(im.height) - 1.0f;
            }
        }

        ComputeContext* dev = GroupGpu();
        GpuLock lock(dev);
        nn::Engine eng(dev);
        m_lomaOnGpu = dev != nullptr;
        const float threshold = float(m_lomaThreshold);
        std::vector<LomaMatcher::Pair> pairs;
        for (size_t i = 0; i < nImg; ++i) {
            if (refsOf[i].empty()) continue;
            if (GroupCancelled()) { *err = "cancelled"; return false; }
            const FeatureSidecar* fs = FeaturesOf(images[i]);
            auto ms = std::make_shared<MatchSidecar>();
            for (const auto& [refIdx, revisit] : refsOf[i]) {
                const FeatureSidecar* ref = FeaturesOf(images[size_t(refIdx)]);
                if (!net->Match(eng, kp[size_t(refIdx)], ref->descriptors.f, kp[i],
                                fs->descriptors.f, m_lomaLayers, threshold, &pairs, err)) {
                    *err = "match_ann: " + *err;
                    return false;
                }
                MatchSet set;
                set.reference = refIdx;
                set.revisit = revisit;
                set.considered = int(fs->keypoints.size());
                for (const auto& p : pairs) {
                    Match m;
                    m.a = p.i;   // the reference's keypoint
                    m.b = p.j;   // this frame's
                    // Lower is better in both, as for the distance matchers.
                    m.distance = 1.0f - p.score;
                    m.ratio = 1.0f - p.score;
                    set.matches.push_back(m);
                }
                ms->considered += set.considered;
                ms->sets.push_back(std::move(set));
            }
            (*made)[i] = std::move(ms);
        }
        return true;
    }

    // Every pair's matches by exact search on the device: each frame's
    // descriptors uploaded once, each pair's k nearest found by GpuKnn, then
    // the same Judge as the forest's. False -- and nothing in `made` to trust
    // -- when the device cannot do it, so the caller falls back to the forest.
    //
    // k = kKnnMax rather than 2, because of min_separation: the runner-up
    // must be a different place in the reference image, and the second
    // nearest descriptor is often the same corner detected twice. The
    // runner-up is the best of the k that is far enough; when none of the
    // nearest eight is, Judge finds no runner-up and drops the match, which
    // is the conservative direction.
    bool MatchAllOnGpu(const std::vector<Image>& images,
                       const std::vector<std::vector<std::pair<int, bool>>>& refsOf,
                       std::vector<std::shared_ptr<MatchSidecar>>* made) const {
        ComputeContext* dev = GroupGpu();
        GpuLock lock(dev);
        GpuKnn knn(dev);
        std::string err;
        const size_t nImg = images.size();

        std::vector<char> used(nImg, 0);
        for (size_t i = 0; i < nImg; ++i)
            for (const auto& r : refsOf[i]) used[i] = used[size_t(r.first)] = 1;
        std::vector<GpuBuffer> bufs(nImg);
        int dim = 0;
        for (size_t i = 0; i < nImg; ++i) {
            if (!used[i]) continue;
            const DescriptorSet& d = FeaturesOf(images[i])->descriptors;
            dim = d.dim;
            if (!GpuKnn::Supports(dim) ||
                !knn.Upload(d.f.data(), int(d.Count()), dim, &bufs[i], &err))
                return false;
        }

        // Every pair, in the order the sets are reported.
        struct PairJob {
            size_t i;
            int refIdx;
            bool revisit;
            std::vector<int> idx, back;
            std::vector<float> d2;
            MatchSet set;
        };
        std::vector<PairJob> pairs;
        for (size_t i = 0; i < nImg; ++i)
            for (const auto& [refIdx, revisit] : refsOf[i]) {
                PairJob pj;
                pj.i = i;
                pj.refIdx = refIdx;
                pj.revisit = revisit;
                pairs.push_back(std::move(pj));
            }

        // IN BATCHES: a batch's searches are recorded together, merged on
        // the device and read back once, then its pairs are judged on every
        // core. One pair at a time, the device waited on the CPU and the
        // CPU on the device. A batch's output is ~0.4 MB a pair.
        const bool cross = bool(m_crossCheck);
        constexpr size_t kBatch = 64;
        std::vector<GpuKnn::Job> jobs;
        for (size_t b0 = 0; b0 < pairs.size(); b0 += kBatch) {
            GroupProgress(double(b0) / double(pairs.size()));
            if (GroupCancelled()) return false;
            const size_t b1 = std::min(pairs.size(), b0 + kBatch);
            jobs.clear();
            for (size_t p = b0; p < b1; ++p) {
                PairJob& pj = pairs[p];
                const int nq = int(FeaturesOf(images[pj.i])->descriptors.Count());
                const int nr = int(FeaturesOf(images[size_t(pj.refIdx)])->descriptors.Count());
                GpuKnn::Job fwd;
                fwd.q = &bufs[pj.i];
                fwd.nq = nq;
                fwd.db = &bufs[size_t(pj.refIdx)];
                fwd.nd = nr;
                fwd.k = kKnnMax;
                fwd.idx = &pj.idx;
                fwd.d2 = &pj.d2;
                // The cross check's search the other way round, from the
                // same distances rather than a second pass: see GpuKnn::Job.
                if (cross) fwd.back = &pj.back;
                jobs.push_back(fwd);
            }
            if (!knn.SearchBatch(jobs, dim, &err)) return false;

            ParallelFor(b1 - b0, [&](size_t o) {
                PairJob& pj = pairs[b0 + o];
                const FeatureSidecar* fs = FeaturesOf(images[pj.i]);
                const FeatureSidecar* ref = FeaturesOf(images[size_t(pj.refIdx)]);
                pj.set.reference = pj.refIdx;
                pj.set.revisit = pj.revisit;
                Judge(*ref, *fs, true, pj.back, &pj.set, [&](size_t q, std::vector<Candidate>* c) {
                    c->clear();
                    for (int j = 0; j < kKnnMax; ++j) {
                        const size_t at = q * size_t(kKnnMax) + size_t(j);
                        if (pj.idx[at] >= 0) c->push_back({pj.idx[at], pj.d2[at]});
                    }
                });
                // The candidates are spent; free them as the batch goes.
                std::vector<int>().swap(pj.idx);
                std::vector<float>().swap(pj.d2);
                std::vector<int>().swap(pj.back);
            });
        }

        for (PairJob& pj : pairs) {
            auto& ms = (*made)[pj.i];
            if (!ms) ms = std::make_shared<MatchSidecar>();
            ms->considered += pj.set.considered;
            ms->sets.push_back(std::move(pj.set));
        }
        return true;
    }
};

} // namespace

REGISTER_ALGORITHM(MatchAnn);

} // namespace tglab
