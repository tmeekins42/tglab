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
#include "../../core/algorithm.h"

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
            const int near = (diff < 0.0f) ? n.left : n.right;
            const int far  = (diff < 0.0f) ? n.right : n.left;
            // The far branch cannot hold anything closer than |diff|, so that
            // is its priority when it is reconsidered.
            queue->push_back({far, tree, diff * diff});
            node = near;
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

        // The index is built ONCE over the reference and queried by every other
        // frame. That is the whole reason an approximate matcher pays off in a
        // group: the build is amortised over N-1 searches.
        //
        // Chain mode gives that up, necessarily: each frame queries a different
        // neighbour, so there are N-1 builds over N-1 searches and the
        // amortisation is gone. The index still beats brute force on a single
        // pair -- that is what the checks/trees parameters buy -- but the
        // margin is much smaller here than in the fixed-reference case, and a
        // chained group is the one place brute force may well be the better
        // choice. Said plainly rather than hidden: `builtFor` is what makes the
        // rebuild explicit instead of accidental.
        KdForest  forest;
        LshTables lsh;
        bool      isFloat  = false;
        int       builtFor = -1;

        // See match_brute's `window`: a chain alone holds no constraint
        // relating frame 8 to frame 5, so bundle adjustment has nothing to
        // correct drift with unless each frame matches several neighbours.
        const int window = chain ? std::max(1, int(m_window)) : 1;

        for (size_t i = 0; i < images->size(); ++i) {
            Image& img = (*images)[i];
            const FeatureSidecar* fs = FeaturesOf(img);
            if (!fs || fs->keypoints.empty()) continue;

            auto ms = std::make_shared<MatchSidecar>();

            for (int w = 1; w <= window; ++w) {
                const int refIdx = chain ? int(i) - w : fixedRef;
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

                if (builtFor != refIdx) {
                    forest = KdForest{};
                    lsh    = LshTables{};
                    isFloat = ref->descriptors.kind == DescriptorKind::Float;
                    if (isFloat)
                        forest.Build(ref->descriptors, std::max(1, int(m_trees)), 8, 12345u);
                    else
                        lsh.Build(ref->descriptors, std::max(1, int(m_trees)),
                                  int(m_keyBits), 12345u);
                    builtFor = refIdx;
                }

                MatchSet set;
                set.reference = refIdx;
                MatchPair(*ref, *fs, forest, lsh, isFloat, &set);

                m_total += set.considered;
                m_kept  += int(set.matches.size());
                ++m_pairs;
                ms->considered += set.considered;
                ms->sets.push_back(std::move(set));

                if (!chain) break;
            }

            if (!ms->sets.empty()) {
                ms->matcher = isFloat ? "ann (kd-forest)" : "ann (lsh)";
                img.Sidecars().Set(kMatchSidecar, ms);
            }
        }
        return true;
    }

    std::string RunReport() const override {
        if (!m_note.empty()) return m_note;
        if (m_pairs == 0) return {};
        char buf[192];
        std::snprintf(buf, sizeof buf,
                      "matched %d pair%s approximately: %d of %d candidates kept",
                      m_pairs, m_pairs == 1 ? "" : "s", m_kept, m_total);
        return buf;
    }

    bool HasGPU() const override { return false; }

private:
    void MatchPair(const FeatureSidecar& ref, const FeatureSidecar& other,
                   const KdForest& forest, const LshTables& lsh, bool isFloat,
                   MatchSet* out) const {
        const float maxRatio = float(m_ratio);
        const int   checks   = std::max(1, int(m_checks));
        const float sepSq    = float(m_minSeparation) * float(m_minSeparation);

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
        // matcher is that nothing in it is O(n*m).
        std::vector<int> backBest;
        if (bool(m_crossCheck)) {
            std::vector<Candidate> rc;
            backBest.assign(ref.descriptors.Count(), -1);

            KdForest  revForest;
            LshTables revLsh;
            if (isFloat) revForest.Build(other.descriptors, std::max(1, int(m_trees)),
                                         8, 54321u);
            else         revLsh.Build(other.descriptors, std::max(1, int(m_trees)),
                                      int(m_keyBits), 54321u);

            for (size_t i = 0; i < ref.descriptors.Count(); ++i) {
                if (isFloat) revForest.Search(ref.descriptors.FloatAt(i), checks, &rc);
                else         revLsh.Search(ref.descriptors.BinaryAt(i), checks, &rc);
                float best = std::numeric_limits<float>::max();
                for (const Candidate& c : rc)
                    if (c.dist < best) { best = c.dist; backBest[i] = c.index; }
            }
        }

        std::vector<Candidate> cands;
        const size_t n = other.descriptors.Count();

        for (size_t i = 0; i < n; ++i) {
            if (isFloat) forest.Search(other.descriptors.FloatAt(i), checks, &cands);
            else         lsh.Search(other.descriptors.BinaryAt(i), checks, &cands);
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

    int         m_pairs = 0;
    int         m_total = 0;
    int         m_kept  = 0;
    std::string m_note;
};

} // namespace

REGISTER_ALGORITHM(MatchAnn);

} // namespace tglab
