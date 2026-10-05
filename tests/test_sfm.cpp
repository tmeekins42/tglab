// Structure-from-Motion: tracks, relative pose, and rotation averaging.
//
// A binary of its own rather than more of test_script.cpp, for two reasons.
//
// The first is subject matter. These tests are about GEOMETRY -- whether a
// union-find recovers a known track structure, whether an averager recovers
// known orientations -- and they share a vocabulary (synthetic cameras, ground
// truth, gauge freedom) with each other and with nothing else in the suite.
// Phases still to come add triangulation, positioning and bundle adjustment,
// all of which belong here too.
//
// The second is that test_script.cpp had grown past seven thousand lines, and
// adding these to it produced a silent crash in an UNRELATED test some three
// thousand lines earlier -- the classic signature of a memory error surfacing
// at whatever allocates next. Splitting them apart isolates the two, which is
// worth doing on its own terms whether or not it proves to be the cure.
//
// WHAT EVERY TEST HERE HAS IN COMMON: synthetic input with a knowable answer.
// A real detector's correct output is not independently known, so a test built
// on one can only check that nothing crashed. Building the fixture from ground
// truth is what lets these assert that the answer is RIGHT.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "../src/algo_util/features.h"
#include "../src/algo_util/linalg.h"
#include "../src/algo_util/sparse_cholesky.h"
#include "../src/algo_util/view_graph.h"
#include "../src/core/algorithm.h"
#include "../src/app/orbit_camera.h"
#include "../src/core/pipeline.h"
#include "../src/script/interp.h"
#include "../src/script/parser.h"
#include "../src/gpu/compute.h"
#include "../src/algorithms/sfm/gpu_sweep.h"
#include "../src/algo_util/splat_raster.h"
#include "../src/algo_util/splat_reflect.h"
#include "../src/algo_util/splat_train_gpu.h"

#include <d3d12.h>

// d3d12.h drags in windows.h, which defines `near` and `far` as macros for
// 16-bit segmented addressing. They have no business here and they collide
// with ordinary variable names.
#ifdef near
#undef near
#endif
#ifdef far
#undef far
#endif

using namespace tglab;

static int g_fail = 0;

// The device for the GPU sections, or none when TGLAB_TESTS_NO_GPU is set.
// The release workflow sets it: a hosted runner has no GPU, only Microsoft's
// software renderer, and there the splat rasteriser's device was removed
// part-way through (0.19.0's first build) -- which says nothing about the
// code, and every GPU section after it then found no device at all. The
// same checks pass on WARP on a desktop.
static HRESULT TestDevice(IUnknown* adapter, D3D_FEATURE_LEVEL level, REFIID riid,
                          void** out) {
    if (std::getenv("TGLAB_TESTS_NO_GPU")) return E_FAIL;
    return D3D12CreateDevice(adapter, level, riid, out);
}

static void Check(bool cond, const std::string& what) {
    std::printf("%s  %s\n", cond ? "[ ok ]" : "[FAIL]", what.c_str());
    if (!cond) ++g_fail;
}

// Camera-to-world unprojection, written out INDEPENDENTLY of fuse_depth's own
// copy rather than calling into it.
//
// That duplication is the point. A test that called the algorithm's private
// helper would pass for any self-consistent implementation, including a wrong
// one; stating the intended formula separately is what makes the comparison
// meaningful. If these two ever disagree, one of them is a bug.
static Vec3 UnprojectForTest(const Camera& c, double px, double py,
                             double depth) {
    const double f = (std::fabs(c.focal) < 1e-9) ? 1e-9 : c.focal;
    const Vec3 cam{(px - c.cx) / f * depth, (py - c.cy) / f * depth, depth};
    return c.R.Transpose() * (cam - c.t);
}

int main() {
    // Unbuffered, so a crash does not swallow the output that would say where.
    setvbuf(stdout, nullptr, _IONBF, 0);

    // --- block-sparse Cholesky ------------------------------------------------
    //
    // Against the dense Cholesky on the same matrix, shaped like a video's
    // reduced camera system: each node coupled to its next few, a handful of
    // long-range revisits -- what makes the ordering matter -- and one size-1
    // node coupled to everything, as a shared focal is. Built as a sum of
    // rank-3 "points" over two or three nodes each, plus a little diagonal,
    // so it is positive definite and its pattern is exactly the pairs used.
    {
        std::printf("\n--- block-sparse Cholesky ---\n");
        std::mt19937 rng(7);
        std::uniform_real_distribution<double> U(-1.0, 1.0);
        const int nCam = 60;
        std::vector<int> sizes(size_t(nCam), 6);
        sizes.push_back(1);   // the shared node, last
        const int nNode = nCam + 1;
        linalg::BlockSparseCholesky sp;
        std::vector<int> scalar(size_t(nNode) + 1, 0);
        for (int i = 0; i < nNode; ++i) scalar[size_t(i) + 1] = scalar[size_t(i)] + sizes[size_t(i)];
        const int dim = scalar[size_t(nNode)];

        std::vector<std::vector<int>> groups;
        for (int c = 0; c < nCam; ++c)
            for (int d = 1; d <= 3 && c + d < nCam; ++d) groups.push_back({c, c + d, nCam});
        for (int r = 0; r < 8; ++r) groups.push_back({r * 3, nCam - 1 - r * 4, nCam});   // revisits
        std::vector<std::pair<int, int>> pairs;
        for (const auto& g : groups)
            for (int a : g)
                for (int b : g) pairs.push_back({a, b});
        Check(sp.Analyse(sizes, pairs), "analyses a video-shaped pattern");

        std::vector<double> A(size_t(dim) * size_t(dim), 0.0);
        for (const auto& g : groups) {
            std::vector<int> cols;
            for (int node : g)
                for (int a = 0; a < sizes[size_t(node)]; ++a) cols.push_back(scalar[size_t(node)] + a);
            for (int row = 0; row < 3; ++row) {
                std::vector<double> gv(cols.size());
                for (auto& v : gv) v = U(rng);
                for (size_t p = 0; p < cols.size(); ++p)
                    for (size_t q = 0; q < cols.size(); ++q)
                        A[size_t(cols[p]) * size_t(dim) + size_t(cols[q])] += gv[p] * gv[q];
            }
        }
        for (int i = 0; i < dim; ++i) A[size_t(i) * size_t(dim) + size_t(i)] += 0.1;

        // The sparse values, block by block out of the dense matrix.
        std::vector<double> vals(sp.NumValues(), 0.0);
        for (int i = 0; i < nNode; ++i)
            for (int j = 0; j <= i; ++j) {
                if (sp.Offset(i, j) < 0) continue;
                std::vector<double> blk(size_t(sizes[size_t(i)]) * size_t(sizes[size_t(j)]));
                for (int a = 0; a < sizes[size_t(i)]; ++a)
                    for (int b = 0; b < sizes[size_t(j)]; ++b)
                        blk[size_t(a * sizes[size_t(j)] + b)] =
                            A[size_t(scalar[size_t(i)] + a) * size_t(dim) + size_t(scalar[size_t(j)] + b)];
                sp.Add(vals.data(), i, j, blk.data());
            }

        std::vector<double> rhs(static_cast<size_t>(dim));
        for (auto& v : rhs) v = U(rng);
        std::vector<double> xd = rhs, Ad = A;
        const bool denseOk = linalg::CholeskySolve(Ad.data(), xd.data(), dim);
        std::vector<double> xs = rhs;
        const bool sparseOk = sp.Factor(vals.data());
        if (sparseOk) sp.Solve(xs.data());
        double worst = 0.0, scale = 0.0;
        for (int i = 0; i < dim; ++i) {
            worst = std::max(worst, std::fabs(xs[size_t(i)] - xd[size_t(i)]));
            scale = std::max(scale, std::fabs(xd[size_t(i)]));
        }
        Check(denseOk && sparseOk && worst <= 1e-9 * std::max(1.0, scale),
              "solves as the dense Cholesky does (worst difference " + std::to_string(worst) +
                  ", fill " + std::to_string(sp.FactorValues()) + " of dense " +
                  std::to_string(size_t(dim) * size_t(dim + 1) / 2) + ")");

        // Not positive definite: one diagonal entry driven negative.
        std::vector<double> bad = vals;
        bad[size_t(sp.Offset(5, 5))] = -1.0;
        Check(!sp.Factor(bad.data()), "refuses a matrix that is not positive definite");
        // And recovers on the next good one: the pattern is reused.
        std::vector<double> xs2 = rhs;
        const bool again = sp.Factor(vals.data());
        if (again) sp.Solve(xs2.data());
        Check(again && xs2 == xs, "refactors on the same pattern to the same answer");
    }

    // --- build_tracks -------------------------------------------------------
    //
    // SYNTHETIC MATCHES, not a real detector, because the point is whether the
    // union-find recovers a track structure we already know. A real detector's
    // correct answer is not independently knowable, so a test built on one can
    // only check that nothing crashed.
    //
    // The fixture: five frames, and a handful of 3D points each seen by a
    // known subset. Matches are generated from that ground truth, then
    // build_tracks must reconstruct exactly the subsets we started from.
    {
        std::printf("\n--- build_tracks ---\n");

        // Point p is seen by frames `seenBy[p]`, at keypoint index p in each.
        // Keypoint index == point index keeps the fixture readable; the
        // algorithm has no way to exploit that, since it only ever sees
        // (frame, keypoint) pairs joined by matches.
        const std::vector<std::vector<int>> seenBy = {
            {0, 1, 2, 3, 4},   // point 0: every frame -- a length-5 track
            {0, 1, 2},         // point 1: the first three
            {2, 3, 4},         // point 2: the last three
            {1, 2},            // point 3: a bare pair
            {0, 4},            // point 4: non-adjacent, only found via matching
        };
        const int kFrames = 5;
        const int kPoints = int(seenBy.size());

        auto buildSet = [&](bool addConflict, bool markOutlier) {
            ImageSet set;
            for (int f = 0; f < kFrames; ++f) {
                Image im;
                im.Alloc({32, 32, Format::RGBA8});

                auto fs = std::make_shared<FeatureSidecar>();
                fs->detector = "synthetic";
                // One keypoint slot per point, whether or not this frame sees
                // it: keeps indices aligned across frames, and an unseen slot
                // is simply never matched, so it stays a singleton and is
                // dropped.
                for (int p = 0; p < kPoints; ++p) {
                    Keypoint k;
                    k.x = float(4 + p * 5);
                    k.y = float(4 + f * 5);
                    fs->keypoints.push_back(k);
                }
                im.Sidecars().Set(kFeatureSidecar, std::move(fs));
                set.images.push_back(std::move(im));
            }

            // Matches: for each point, join consecutive frames that see it.
            // Deliberately NOT all-pairs -- transitivity is the union-find's
            // job, and a fixture that pre-joins everything would not test it.
            for (int f = 1; f < kFrames; ++f) {
                auto ms = std::make_shared<MatchSidecar>();
                ms->matcher = "synthetic";
                for (int ref = 0; ref < f; ++ref) {
                    MatchSet mset;
                    mset.reference = ref;
                    for (int p = 0; p < kPoints; ++p) {
                        const auto& s = seenBy[size_t(p)];
                        const bool inRef =
                            std::find(s.begin(), s.end(), ref) != s.end();
                        const bool inCur =
                            std::find(s.begin(), s.end(), f) != s.end();
                        if (!inRef || !inCur) continue;
                        // Only CONSECUTIVE sightings, so point 4 (frames 0 and
                        // 4) is joined directly while point 0 relies on the
                        // chain 0-1-2-3-4.
                        bool consecutive = true;
                        for (int mid : s)
                            if (mid > ref && mid < f) { consecutive = false; break; }
                        if (!consecutive) continue;
                        Match m;
                        m.a = p;
                        m.b = p;
                        mset.matches.push_back(m);
                        mset.inlier.push_back(1);
                    }
                    if (addConflict && ref == 0 && f == 1) {
                        // A WRONG match: frame 1's keypoint for point 1 also
                        // claims to be point 0. Union-find merges the two
                        // tracks, and the result sees frame 1 twice.
                        Match bad;
                        bad.a = 0;
                        bad.b = 1;
                        mset.matches.push_back(bad);
                        mset.inlier.push_back(markOutlier ? 0 : 1);
                    }
                    if (!mset.matches.empty()) ms->sets.push_back(std::move(mset));
                }
                if (!ms->sets.empty())
                    set.images[size_t(f)].Sidecars().Set(kMatchSidecar, std::move(ms));
            }
            set.shape = Shape{{{"frame", kFrames}}};
            return set;
        };

        auto runTracks = [&](ImageSet set, int minLen, bool dropConflicts,
                             PointCloud* out, std::string* err) {
            std::vector<Data> s;
            s.push_back(Data{std::move(set)});
            Pipeline p;
            auto bt = Registry::Get().Create("build_tracks");
            if (!bt) { *err = "build_tracks not registered"; return false; }
            if (ParamBase* pb = bt->FindParam("min_length")) {
                std::string e; pb->SetFromScript(Value(double(minLen)), &e);
            }
            if (ParamBase* pb = bt->FindParam("drop_conflicts")) {
                std::string e;
                pb->SetFromScript(Value(dropConflicts ? 1.0 : 0.0), &e);
            }
            p.AddStage(std::move(bt), "build_tracks", {{-1, 0}}, 1, 1);
            if (!p.Execute(&s, nullptr, err)) return false;
            const Data* d = p.Resolve({0, 0}, &s);
            if (!d) { *err = "no output"; return false; }
            const PointCloud* pc = std::get_if<PointCloud>(d);
            if (!pc) { *err = "output is not a PointCloud"; return false; }
            *out = *pc;
            return true;
        };

        // --- the structure is recovered exactly ---------------------------
        {
            PointCloud pc;
            std::string err;
            const bool ok = runTracks(buildSet(false, false), 2, true, &pc, &err);
            Check(ok, "build_tracks runs" + (ok ? "" : ": " + err));
            if (ok) {
                Check(int(pc.tracks.size()) == kPoints,
                      "one track per synthetic point (" +
                          std::to_string(pc.tracks.size()) + " of " +
                          std::to_string(kPoints) + ")");

                // Sorted longest first by the algorithm, so compare as sets of
                // frame lists rather than by index.
                std::vector<std::vector<int>> got;
                for (const Track& t : pc.tracks) {
                    std::vector<int> frames;
                    for (const Observation& o : t.obs) frames.push_back(o.frame);
                    std::sort(frames.begin(), frames.end());
                    got.push_back(frames);
                }
                std::vector<std::vector<int>> want = seenBy;
                std::sort(got.begin(), got.end());
                std::sort(want.begin(), want.end());
                Check(got == want,
                      "and every track holds exactly the frames that saw it");

                // THE TRANSITIVITY CHECK. Point 0's frames were joined only
                // CONSECUTIVELY (0-1, 1-2, 2-3, 3-4); recovering one length-5
                // track from four separate pairwise matches is precisely what
                // union-find is for, and a per-pair implementation would give
                // four length-2 tracks instead.
                //
                // Checked as "a track holding exactly {0,1,2,3,4}" rather than
                // "the longest track is 5". The weaker form passed a
                // deliberately broken build, because point 4 is seen by frames
                // 0 and 4 and joined DIRECTLY -- so a length-5 span exists even
                // with transitivity removed. The span was never the claim.
                bool chained = false;
                for (const Track& t : pc.tracks) {
                    if (t.Length() != 5) continue;
                    std::vector<int> frames;
                    for (const Observation& o : t.obs) frames.push_back(o.frame);
                    std::sort(frames.begin(), frames.end());
                    if (frames == std::vector<int>{0, 1, 2, 3, 4}) chained = true;
                }
                Check(chained,
                      "a CHAIN of consecutive matches becomes one long track");

                Check(int(pc.cameras.size()) == kFrames,
                      "a camera slot per frame (" +
                          std::to_string(pc.cameras.size()) + ")");
                Check(!pc.cameras.empty() && !pc.cameras[0].solved,
                      "...and none is marked solved, because nothing solved one");
            }
        }

        // --- min_length ----------------------------------------------------
        {
            PointCloud pc;
            std::string err;
            if (runTracks(buildSet(false, false), 3, true, &pc, &err)) {
                // Points 3 and 4 are length 2, so three survive.
                Check(int(pc.tracks.size()) == 3,
                      "min_length drops the short tracks (" +
                          std::to_string(pc.tracks.size()) + " of 5)");
                for (const Track& t : pc.tracks)
                    if (t.Length() < 3)
                        Check(false, "a track shorter than min_length survived");
            }
        }

        // --- conflicts ------------------------------------------------------
        //
        // A wrong match merges two points into a track seeing frame 1 twice.
        // That is physically impossible, and keeping it would feed a bad
        // correspondence to every solver downstream.
        {
            PointCloud pc;
            std::string err;
            if (runTracks(buildSet(true, false), 2, true, &pc, &err)) {
                for (const Track& t : pc.tracks) {
                    std::vector<int> frames;
                    for (const Observation& o : t.obs) frames.push_back(o.frame);
                    std::sort(frames.begin(), frames.end());
                    const bool dup =
                        std::adjacent_find(frames.begin(), frames.end()) != frames.end();
                    if (dup) { Check(false, "a conflicting track survived"); break; }
                }
                Check(true, "a track seeing one frame twice is dropped");
            }

            // With the filter off it must SURVIVE -- otherwise the check above
            // proves nothing, since something else might be removing it.
            PointCloud pc2;
            if (runTracks(buildSet(true, false), 2, false, &pc2, &err)) {
                bool anyDup = false;
                for (const Track& t : pc2.tracks) {
                    std::vector<int> frames;
                    for (const Observation& o : t.obs) frames.push_back(o.frame);
                    std::sort(frames.begin(), frames.end());
                    if (std::adjacent_find(frames.begin(), frames.end()) != frames.end())
                        anyDup = true;
                }
                Check(anyDup,
                      "...and with drop_conflicts off it survives, so the "
                      "filter is what removed it");
            }
        }

        // --- outliers are not unioned ---------------------------------------
        //
        // The same bad match, but flagged as a RANSAC outlier. It must be
        // ignored, so the structure comes back clean without the conflict
        // filter having to act.
        {
            PointCloud pc;
            std::string err;
            if (runTracks(buildSet(true, true), 2, false, &pc, &err)) {
                Check(int(pc.tracks.size()) == kPoints,
                      "a match marked outlier is not unioned (" +
                          std::to_string(pc.tracks.size()) + " of " +
                          std::to_string(kPoints) + ")");
            }
        }

        // --- colour is sampled from the source frames ------------------------
        //
        // build_tracks is the LAST stage that holds the pixels, so if it does
        // not sample the colour nothing downstream can. The fixture paints
        // each frame a known flat colour, so the answer is knowable: every
        // track must come back that colour, whichever frames saw it.
        {
            ImageSet set = buildSet(false, false);
            // A distinctive colour no default would produce.
            const uint8_t want[3] = {40, 160, 220};
            for (Image& im : set.images) {
                ImageView v = im.MapCpuWrite();
                for (int y = 0; y < v.desc.height; ++y)
                    for (int x = 0; x < v.desc.width; ++x) {
                        uint8_t* p = v.At<uint8_t>(x, y);
                        p[0] = want[0]; p[1] = want[1]; p[2] = want[2]; p[3] = 255;
                    }
            }

            PointCloud pc;
            std::string err;
            if (runTracks(std::move(set), 2, true, &pc, &err)) {
                int coloured = 0;
                double worst = 0.0;
                for (const Track& t : pc.tracks) {
                    const bool set_ = t.color.x != 0.0 || t.color.y != 0.0 ||
                                      t.color.z != 0.0;
                    if (!set_) continue;
                    ++coloured;
                    worst = std::max(worst,
                        std::max({std::fabs(t.color.x - want[0] / 255.0),
                                  std::fabs(t.color.y - want[1] / 255.0),
                                  std::fabs(t.color.z - want[2] / 255.0)}));
                }
                Check(coloured == int(pc.tracks.size()),
                      "every track is coloured from its frames (" +
                          std::to_string(coloured) + " of " +
                          std::to_string(pc.tracks.size()) + ")");
                Check(worst < 0.01,
                      "...with the colour those frames actually hold (worst "
                      "channel error " + std::to_string(worst) + ")");
            }
        }

        // --- it refuses rather than inventing --------------------------------
        {
            ImageSet bare;
            for (int f = 0; f < 3; ++f) {
                Image im;
                im.Alloc({16, 16, Format::RGBA8});
                bare.images.push_back(std::move(im));
            }
            bare.shape = Shape{{{"frame", 3}}};
            PointCloud pc;
            std::string err;
            const bool ok = runTracks(std::move(bare), 2, true, &pc, &err);
            Check(!ok && err.find("detector") != std::string::npos,
                  "a group with no features is a named error: \"" + err + "\"");
        }
    }

    // --- rotation_average ---------------------------------------------------
    //
    // Synthetic relative rotations with a KNOWN answer, fed straight into the
    // averager. Not through a detector: the point is whether the averaging
    // recovers orientations we already know, and a real detector's correct
    // answer is not independently knowable.
    //
    // GAUGE FREEDOM shapes every assertion here. Rotating every camera by the
    // same R leaves all relative rotations unchanged, so there is no correct
    // absolute orientation to compare against. Every check below is on
    // RELATIVE rotations, which are what the problem actually determines.
    {
        std::printf("\n--- rotation_average ---\n");

        // Eight cameras on an arc, each yawed 10 degrees from the last, plus a
        // little pitch so the rotations do not all commute -- a fixture where
        // every rotation shares an axis is a much easier problem than the real
        // one and would flatter any solver.
        const int kFrames = 8;
        std::vector<Mat3> truth;
        for (int f = 0; f < kFrames; ++f) {
            const double yaw = double(f) * 10.0 * 3.14159265358979 / 180.0;
            const double pitch = double(f % 3) * 4.0 * 3.14159265358979 / 180.0;
            truth.push_back(AxisAngleToMat(Vec3{pitch, yaw, 0.0}));
        }

        // Builds a group whose relative-pose sidecars encode `truth`, with a
        // chosen connectivity and optionally one corrupted edge.
        auto buildGroup = [&](int window, int corruptFrom, int corruptTo,
                              double corruptDeg) {
            ImageSet set;
            for (int f = 0; f < kFrames; ++f) {
                Image im;
                im.Alloc({32, 32, Format::RGBA8});
                set.images.push_back(std::move(im));
            }
            for (int j = 1; j < kFrames; ++j) {
                auto rp = std::make_shared<RelativePoseSidecar>();
                for (int i = std::max(0, j - window); i < j; ++i) {
                    RelativePoseSidecar::Edge e;
                    e.reference = i;
                    // R_ij = R_j * R_i^T, the relation the averager inverts.
                    e.R = truth[size_t(j)] * truth[size_t(i)].Transpose();
                    e.inliers = 200;
                    if (i == corruptFrom && j == corruptTo) {
                        // A grossly wrong edge, as a mismatched pair that
                        // survived RANSAC would produce.
                        const double rad = corruptDeg * 3.14159265358979 / 180.0;
                        e.R = e.R * AxisAngleToMat(Vec3{0.0, 0.0, rad});
                    }
                    rp->edges.push_back(e);
                }
                set.images[size_t(j)].Sidecars().Set(kRelativePoseSidecar,
                                                     std::move(rp));
            }
            set.shape = Shape{{{"frame", kFrames}}};
            return set;
        };

        auto runAverage = [&](ImageSet set, int method, PointCloud* out,
                              std::string* err) {
            std::vector<Data> s;
            s.push_back(Data{std::move(set)});
            Pipeline p;
            auto ra = Registry::Get().Create("rotation_average");
            if (!ra) { *err = "rotation_average not registered"; return false; }
            if (ParamBase* pb = ra->FindParam("method")) {
                std::string e; pb->SetFromScript(Value(double(method)), &e);
            }
            p.AddStage(std::move(ra), "rotation_average", {{-1, 0}}, 1, 1);
            if (!p.Execute(&s, nullptr, err)) return false;
            const Data* d = p.Resolve({0, 0}, &s);
            const PointCloud* pc = d ? std::get_if<PointCloud>(d) : nullptr;
            if (!pc) { *err = "output is not a PointCloud"; return false; }
            *out = *pc;
            return true;
        };

        // Worst relative-rotation error against ground truth, in degrees.
        // Relative rather than absolute: see the gauge note above.
        auto worstRelativeError = [&](const PointCloud& pc) {
            double worst = 0.0;
            // Bounded by what the algorithm actually returned, not by what the
            // fixture built. A stage that solves fewer frames than it was given
            // is a legitimate outcome -- a disconnected graph does exactly that
            // -- and indexing past the end to find out is an out-of-bounds read
            // that Release turns into a silent crash rather than an assertion.
            const int lim = std::min(kFrames, int(pc.cameras.size()));
            for (int a = 0; a < lim; ++a) {
                for (int b = a + 1; b < lim; ++b) {
                    if (!pc.cameras[size_t(a)].solved || !pc.cameras[size_t(b)].solved)
                        continue;
                    const Mat3 got = pc.cameras[size_t(b)].R *
                                     pc.cameras[size_t(a)].R.Transpose();
                    const Mat3 want = truth[size_t(b)] * truth[size_t(a)].Transpose();
                    worst = std::max(worst, got.AngleTo(want) * 180.0 / 3.14159265358979);
                }
            }
            return worst;
        };

        // --- clean data, both methods --------------------------------------
        for (int method = 0; method <= 1; ++method) {
            const char* name = method == 0 ? "L2" : "L1-IRLS";
            PointCloud pc;
            std::string err;
            const bool ok = runAverage(buildGroup(2, -1, -1, 0.0), method, &pc, &err);
            Check(ok, std::string(name) + " runs" + (ok ? "" : ": " + err));
            if (!ok) continue;

            Check(pc.SolvedCameras() == kFrames,
                  std::string(name) + " solves every frame (" +
                      std::to_string(pc.SolvedCameras()) + " of " +
                      std::to_string(kFrames) + ")");

            const double worst = worstRelativeError(pc);
            Check(worst < 0.5,
                  std::string(name) + " recovers the true orientations (worst " +
                      std::to_string(worst) + " deg)");
        }

        // --- the gauge is a convention, not a result ------------------------
        {
            PointCloud pc;
            std::string err;
            if (runAverage(buildGroup(2, -1, -1, 0.0), 1, &pc, &err)) {
                // Frame 0 pinned to the identity is the stated convention.
                Check(pc.cameras[0].R.AngleTo(Mat3::Identity()) < 1e-6,
                      "frame 0 is pinned to the identity by convention");
            }
        }

        // --- THE POINT OF IRLS: one bad edge --------------------------------
        //
        // A 40-degree error on the 2-3 edge, as a mismatched pair would give.
        // L2 has no defence -- least squares gives it the same say as every
        // good edge -- while IRLS should discount it once the rest of the
        // graph disagrees. That CONTRAST is the test; a check on IRLS alone
        // would not show that the robustness is doing anything.
        {
            double err2 = -1.0, errIrls = -1.0;
            PointCloud pc;
            std::string err;
            if (runAverage(buildGroup(3, 2, 3, 40.0), 0, &pc, &err))
                err2 = worstRelativeError(pc);
            if (runAverage(buildGroup(3, 2, 3, 40.0), 1, &pc, &err))
                errIrls = worstRelativeError(pc);

            std::printf("       one bad edge: L2 %.2f deg, IRLS %.2f deg\n",
                        err2, errIrls);
            Check(err2 > 0.0 && errIrls > 0.0, "both methods ran on bad data");
            Check(errIrls < err2,
                  "IRLS beats L2 when an edge is grossly wrong (" +
                      std::to_string(errIrls) + " vs " + std::to_string(err2) +
                      " deg)");
        }

        // --- a chain must not drift ------------------------------------------
        //
        // With window = 1 the graph is a bare chain and averaging can only do
        // what composition does. Widening the window adds loop closures, and
        // THAT is what averaging exploits -- so the wider graph must not be
        // worse, and on noisy data would be better.
        {
            PointCloud chain, looped;
            std::string err;
            if (runAverage(buildGroup(1, -1, -1, 0.0), 1, &chain, &err) &&
                runAverage(buildGroup(3, -1, -1, 0.0), 1, &looped, &err)) {
                Check(chain.SolvedCameras() == kFrames,
                      "a bare chain still solves every frame");
                Check(worstRelativeError(looped) <= worstRelativeError(chain) + 1e-6,
                      "extra loop closures do not make the answer worse");
            }
        }

        // --- it refuses rather than inventing --------------------------------
        {
            ImageSet bare;
            for (int f = 0; f < 3; ++f) {
                Image im;
                im.Alloc({16, 16, Format::RGBA8});
                bare.images.push_back(std::move(im));
            }
            bare.shape = Shape{{{"frame", 3}}};
            PointCloud pc;
            std::string err;
            const bool ok = runAverage(std::move(bare), 1, &pc, &err);
            Check(!ok && err.find("relative_pose") != std::string::npos,
                  "a group with no relative poses is a named error: \"" + err + "\"");
        }
    }


    // --- global_position ----------------------------------------------------
    //
    // A synthetic scene with known camera positions and known 3D points, fed in
    // as a reconstruction that already has its orientations. The question is
    // whether the solver puts the cameras back where they were.
    //
    // TWO GAUGE FREEDOMS have to be handled here, one more than rotation
    // averaging had. The whole reconstruction can be translated, rotated AND
    // SCALED without changing any measurement -- two views cannot see baseline
    // length, so nothing in the input fixes the unit. Comparing raw positions
    // would therefore fail on a perfectly correct answer. Every check below
    // compares SHAPE: distances normalised by their own mean, which is
    // invariant to all three.
    {
        std::printf("\n--- global_position ---\n");

        // Builds a reconstruction with solved rotations, tracks carrying exact
        // observations, and the view graph -- everything positioning consumes.
        // `camCentres` is the ground truth the solver must recover.
        auto makeScene = [](const std::vector<Vec3>& camCentres,
                            const std::vector<Vec3>& worldPts) {
            PointCloud pc;
            const int nCam = int(camCentres.size());
            const double focal = 600.0;

            for (int c = 0; c < nCam; ++c) {
                Camera cam;
                cam.width = 1024; cam.height = 768;
                cam.cx = 512.0; cam.cy = 384.0;
                cam.focal = focal;
                // All cameras look down +Z, which keeps the fixture readable.
                // The rotations are what rotation_average would have supplied.
                cam.R = Mat3::Identity();
                cam.t = cam.R * camCentres[size_t(c)] * -1.0;
                cam.solved = true;
                pc.cameras.push_back(cam);
            }

            for (const Vec3& wp : worldPts) {
                Track tr;
                for (int c = 0; c < nCam; ++c) {
                    double px, py;
                    if (!pc.cameras[size_t(c)].Project(wp, &px, &py)) continue;
                    Observation o;
                    o.frame = c;
                    o.keypoint = 0;
                    o.x = float(px);
                    o.y = float(py);
                    tr.obs.push_back(o);
                }
                if (tr.obs.size() >= 2) pc.tracks.push_back(tr);
            }

            // Edge directions, for the translation-averaging arm. In frame i's
            // coordinates, as two-view geometry produces them.
            for (int i = 0; i < nCam; ++i) {
                for (int j = i + 1; j < nCam; ++j) {
                    PointCloud::ViewEdge e;
                    e.i = i; e.j = j;
                    e.R = Mat3::Identity();
                    e.direction = pc.cameras[size_t(i)].R *
                                  (camCentres[size_t(j)] - camCentres[size_t(i)]);
                    e.direction = e.direction.Normalized();
                    e.inliers = 200;
                    pc.edges.push_back(e);
                }
            }
            return pc;
        };

        auto runPosition = [&](PointCloud in, int method, int iters,
                               PointCloud* out, std::string* err) {
            std::vector<Data> s;
            s.push_back(Data{std::move(in)});
            Pipeline p;
            auto gp = Registry::Get().Create("global_position");
            if (!gp) { *err = "global_position not registered"; return false; }
            if (ParamBase* pb = gp->FindParam("method")) {
                std::string e; pb->SetFromScript(Value(double(method)), &e);
            }
            if (ParamBase* pb = gp->FindParam("iterations")) {
                std::string e; pb->SetFromScript(Value(double(iters)), &e);
            }
            p.AddStage(std::move(gp), "global_position", {{-1, 0}}, 1, 1);
            if (!p.Execute(&s, nullptr, err)) return false;
            const Data* d = p.Resolve({0, 0}, &s);
            const PointCloud* pc = d ? std::get_if<PointCloud>(d) : nullptr;
            if (!pc) { *err = "output is not a PointCloud"; return false; }
            *out = *pc;
            return true;
        };

        // SHAPE ERROR, invariant to translation, rotation and scale.
        //
        // Every pairwise distance, divided by the mean pairwise distance, then
        // compared against the same quantity from ground truth. That ratio is
        // what the measurements actually determine; absolute positions are not.
        auto shapeError = [](const PointCloud& pc, const std::vector<Vec3>& truth) {
            const int n = int(truth.size());
            std::vector<double> got, want;
            for (int a = 0; a < n; ++a)
                for (int b = a + 1; b < n; ++b) {
                    got.push_back((pc.cameras[size_t(b)].Center() -
                                   pc.cameras[size_t(a)].Center()).Norm());
                    want.push_back((truth[size_t(b)] - truth[size_t(a)]).Norm());
                }
            double gm = 0.0, wm = 0.0;
            for (double v : got) gm += v;
            for (double v : want) wm += v;
            if (gm < 1e-12 || wm < 1e-12) return 1e9;   // collapsed
            gm /= double(got.size());
            wm /= double(want.size());
            double worst = 0.0;
            for (size_t i = 0; i < got.size(); ++i)
                worst = std::max(worst, std::fabs(got[i] / gm - want[i] / wm));
            return worst;
        };

        // --- a well-conditioned scene: cameras on an arc --------------------
        const std::vector<Vec3> arc = {
            {-3, 0, 0}, {-1.5, 0.4, 0}, {0, 0.6, 0}, {1.5, 0.4, 0}, {3, 0, 0}};
        std::vector<Vec3> pts;
        for (int i = 0; i < 40; ++i) {
            const double a = double(i) * 0.37;
            pts.push_back(Vec3{std::cos(a) * 2.0, std::sin(a) * 1.5, 8.0 + (i % 5)});
        }

        {
            PointCloud out;
            std::string err;
            const bool ok = runPosition(makeScene(arc, pts), 0, 400, &out, &err);
            Check(ok, "joint positioning runs" + (ok ? "" : ": " + err));
            if (ok) {
                const double e = shapeError(out, arc);
                Check(e < 0.05,
                      "joint recovers the camera arrangement (shape error " +
                          std::to_string(e) + ")");
                Check(out.TriangulatedPoints() == int(out.tracks.size()),
                      "...and places every track in 3D (" +
                          std::to_string(out.TriangulatedPoints()) + " of " +
                          std::to_string(out.tracks.size()) + ")");
            }
        }

        // --- random start, same answer ---------------------------------------
        //
        // The formulation's headline claim is that its bounded error needs no
        // initialisation. Two different seeds must therefore reach the same
        // shape -- otherwise "starts from random" would mean "gives a random
        // answer", which is a very different property.
        {
            PointCloud a, b;
            std::string err;
            PointCloud scene = makeScene(arc, pts);
            bool ok = runPosition(scene, 0, 400, &a, &err);

            // A different seed via a second stage run with the seed changed.
            std::vector<Data> s;
            s.push_back(Data{scene});
            Pipeline p;
            auto gp = Registry::Get().Create("global_position");
            if (ParamBase* pb = gp->FindParam("seed")) {
                std::string e; pb->SetFromScript(Value(77.0), &e);
            }
            if (ParamBase* pb = gp->FindParam("iterations")) {
                std::string e; pb->SetFromScript(Value(400.0), &e);
            }
            p.AddStage(std::move(gp), "global_position", {{-1, 0}}, 1, 1);
            if (ok && p.Execute(&s, nullptr, &err)) {
                const Data* d = p.Resolve({0, 0}, &s);
                if (const PointCloud* pc = d ? std::get_if<PointCloud>(d) : nullptr) {
                    b = *pc;
                    Check(shapeError(b, arc) < 0.05,
                          "a different random start reaches the same shape (" +
                              std::to_string(shapeError(b, arc)) + ")");
                }
            }
        }

        // --- A LONG, WEAKLY LINKED CHAIN: the case that exposed the solver ---
        //
        // Thirty cameras walking a circle and looking inward, each point
        // seen by only FOUR consecutive cameras -- how a walk-around or a
        // video actually links up. The fixtures above are the easy case: five
        // cameras that all see all the points, where anything converges.
        //
        // The alternating distance solve that positioning used to end with
        // is the thing this catches: on castle-P19 it left even the best
        // cameras at a 1 degree median ray residual and the reconstruction
        // unusable, while every test here passed. So the contrast is checked
        // too -- the refinement OFF must fail on this fixture, or the fixture
        // cannot express the bug it is here for.
        {
            const int N = 30;
            std::vector<Vec3> ring;
            PointCloud pc;
            const Vec3 up{0, 1, 0};
            for (int c = 0; c < N; ++c) {
                const double a = 2.0 * 3.14159265358979 * c / N;
                const Vec3 pos{4.0 * std::cos(a), 0.3 * std::sin(3 * a), 4.0 * std::sin(a)};
                ring.push_back(pos);
                Camera cam;
                cam.width = 1024; cam.height = 768;
                cam.cx = 512.0; cam.cy = 384.0;
                cam.focal = 600.0;
                const Vec3 z = (Vec3{0, 0, 0} - pos).Normalized();
                const Vec3 x = up.Cross(z).Normalized();
                const Vec3 y = z.Cross(x);
                cam.R.m[0] = x.x; cam.R.m[1] = x.y; cam.R.m[2] = x.z;
                cam.R.m[3] = y.x; cam.R.m[4] = y.y; cam.R.m[5] = y.z;
                cam.R.m[6] = z.x; cam.R.m[7] = z.y; cam.R.m[8] = z.z;
                cam.t = cam.R * pos * -1.0;
                cam.solved = true;
                pc.cameras.push_back(cam);
            }
            uint32_t seed = 4242u;
            auto rnd = [&]() {
                seed = seed * 1664525u + 1013904223u;
                return double(seed >> 8) / double(1u << 24);
            };
            for (int i = 0; i < 600; ++i) {
                const Vec3 wp{rnd() * 2.0 - 1.0, rnd() * 2.0 - 1.0, rnd() * 2.0 - 1.0};
                Track tr;
                const int first = i % N;
                for (int k = 0; k < 4; ++k) {
                    const int c = (first + k) % N;
                    double px, py;
                    if (!pc.cameras[size_t(c)].Project(wp, &px, &py)) continue;
                    Observation o;
                    o.frame = c;
                    o.x = float(px + (rnd() - 0.5) * 0.6);   // +-0.3 px
                    o.y = float(py + (rnd() - 0.5) * 0.6);
                    tr.obs.push_back(o);
                }
                if (tr.obs.size() >= 2) pc.tracks.push_back(tr);
            }

            auto run = [&](int refine, double* shape) {
                std::vector<Data> s;
                s.push_back(Data{pc});
                Pipeline p;
                auto gp = Registry::Get().Create("global_position");
                std::string e;
                gp->FindParam("refine")->SetFromScript(Value(double(refine)), &e);
                p.AddStage(std::move(gp), "global_position", {{-1, 0}}, 1, 1);
                std::string err;
                if (!p.Execute(&s, nullptr, &err)) return false;
                const Data* d = p.Resolve({0, 0}, &s);
                const PointCloud* out = d ? std::get_if<PointCloud>(d) : nullptr;
                if (!out) return false;
                *shape = shapeError(*out, ring);
                return true;
            };
            double withRefine = 1e9, without = 1e9;
            const bool okA = run(100, &withRefine);
            const bool okB = run(0, &without);
            std::printf("       30-camera ring, 4 views per point: shape error %.4f "
                        "refined, %.4f with the alternating solve alone\n",
                        withRefine, without);
            Check(okA && withRefine < 0.02,
                  "a long weakly-linked ring is positioned (shape error " +
                      std::to_string(withRefine) + ")");
            Check(okB && without > 5.0 * withRefine,
                  "...where the alternating solve alone is not -- so this "
                  "fixture can see the bug it guards against");
        }

        // --- THE COLLINEAR DEGENERACY ----------------------------------------
        //
        // The reason the joint method exists. Five cameras on a straight line:
        // every pairwise direction between them is identical, so the EDGE
        // measurements carry no information about spacing and translation
        // averaging is ill-posed. The rays to off-axis points are still
        // distinct, so the joint method is not.
        //
        // The contrast IS the test. Checking the joint method alone would not
        // show that the degeneracy is real, and checking averaging alone would
        // look like an ordinary failure.
        {
            const std::vector<Vec3> line = {
                {-4, 0, 0}, {-2, 0, 0}, {0, 0, 0}, {2, 0, 0}, {4, 0, 0}};

            PointCloud jointOut, avgOut;
            std::string err;
            double jointErr = 1e9, avgErr = 1e9;

            if (runPosition(makeScene(line, pts), 0, 400, &jointOut, &err))
                jointErr = shapeError(jointOut, line);
            if (runPosition(makeScene(line, pts), 1, 400, &avgOut, &err))
                avgErr = shapeError(avgOut, line);

            std::printf("       collinear cameras: joint %.4f, averaging %.4f\n",
                        jointErr, avgErr);

            Check(jointErr < 0.05,
                  "joint solves collinear cameras (shape error " +
                      std::to_string(jointErr) + ")");
            Check(jointErr < avgErr,
                  "...where translation averaging does worse (" +
                      std::to_string(avgErr) + "), which is the degeneracy "
                      "the joint formulation removes");
        }

        // --- EVERY POINT ENDS UP IN FRONT OF ITS CAMERAS ---------------------
        //
        // A ray is a HALF-line, but the least-squares step that places points
        // measures perpendicular distance to the infinite LINE -- so a point
        // behind the camera fits perfectly and the solve has no reason to
        // prefer the front.
        //
        // The synthetic fixtures here did not catch it because they start from
        // a configuration where the right answer is easy to find. It took
        // castle-P19 to expose it: 3891 of 16237 tracks landed behind a
        // camera, the mean ray residual sat at 39 degrees, and the viewport
        // showed an unreadable haze. With the constraint the residual is 3.4.
        //
        // Asserted on the RECONSTRUCTED geometry rather than on the residual,
        // because "in front of the camera that saw it" is the actual physical
        // claim and a residual is a proxy for it.
        {
            PointCloud out;
            std::string err;
            if (runPosition(makeScene(arc, pts), 0, 400, &out, &err)) {
                int behind = 0, checked = 0;
                for (const Track& t : out.tracks) {
                    if (!t.hasPoint) continue;
                    for (const Observation& o : t.obs) {
                        if (o.frame < 0 || o.frame >= int(out.cameras.size()))
                            continue;
                        const Camera& c = out.cameras[size_t(o.frame)];
                        if (!c.solved) continue;
                        ++checked;
                        double px, py;
                        if (!c.Project(t.point, &px, &py)) ++behind;
                    }
                }
                Check(checked > 100, "there are observations to check (" +
                                         std::to_string(checked) + ")");
                Check(behind == 0,
                      "every point is in front of every camera that saw it (" +
                          std::to_string(behind) + " behind of " +
                          std::to_string(checked) + ")");
            }
        }

        // --- A STARVED EDGE SHOULD NOT DRAG THE RECONSTRUCTION ---------------
        //
        // On castle-P19 the reconstruction split into two clusters, and the
        // per-pair diagnostics put the break at one link: pair 11->12, with
        // 130 inliers at a 30% rate where the healthy pairs carry 558-1060 at
        // 59-78%. A walk around a courtyard turns a corner there and the views
        // barely overlap. Unweighted, that link spoke as loudly as any other.
        //
        // The fixture reproduces the SHAPE of that problem rather than its
        // scale: one camera whose only edge is weakly supported, and whose
        // observations are corrupted accordingly. The others are well
        // supported and correct. A solver that weights by support should let
        // the good cameras hold their ground; one that does not will let the
        // bad camera pull them.
        //
        // Compared against the same scene with the weights flattened, because
        // an absolute threshold here would be a number pulled from this
        // fixture rather than a statement about behaviour.
        {
            const std::vector<Vec3> line = {
                {-3, 0, 0}, {-1, 0, 0}, {1, 0, 0}, {3, 0, 0}};

            auto corruptLast = [&](PointCloud pc, int inliersForLast) {
                const int last = int(pc.cameras.size()) - 1;
                // Displace what the last camera reports seeing, so its rays
                // disagree with everyone else's.
                for (Track& t : pc.tracks)
                    for (Observation& o : t.obs)
                        if (o.frame == last) { o.x += 60.0f; o.y -= 40.0f; }
                // ...and mark its edges as thinly supported, which is the
                // signal the solver is meant to act on.
                for (PointCloud::ViewEdge& e : pc.edges)
                    if (e.i == last || e.j == last) e.inliers = inliersForLast;
                return pc;
            };

            // MEASURED ON THE POINTS, not on the cameras, and that distinction
            // is the whole content of this test.
            //
            // The weight is a property of a camera, and the camera solve is
            // separable -- each camera gets its own 3x3 system from its own
            // rays, so a constant factor cancels exactly and cannot move it.
            // A starved camera therefore still goes wherever its bad rays say.
            // What the weighting can do is stop that camera dragging the
            // POINTS, which every other camera shares.
            //
            // A first version of this test asserted on camera positions and
            // failed with the two arms agreeing to five decimals -- correctly,
            // because it was asking about the one quantity the change is
            // designed not to touch.
            auto pointSpread = [](const PointCloud& pc) {
                double worst = 0.0;
                int n = 0;
                Vec3 mean{0, 0, 0};
                for (const Track& t : pc.tracks) {
                    if (!t.hasPoint) continue;
                    mean = mean + t.point;
                    ++n;
                }
                if (n == 0) return 1e9;
                mean = mean * (1.0 / double(n));
                for (const Track& t : pc.tracks) {
                    if (!t.hasPoint) continue;
                    worst = std::max(worst, (t.point - mean).Norm());
                }
                return worst;
            };

            PointCloud weighted, flat;
            std::string err;
            double weightedSpread = 1e9, flatSpread = 1e9;

            // 12 inliers against 200: the ratio the castle's worst link had,
            // exaggerated so the effect is unambiguous on four cameras.
            if (runPosition(corruptLast(makeScene(line, pts), 12), 0, 400,
                            &weighted, &err))
                weightedSpread = pointSpread(weighted);

            // The control: same corruption, but the starved edge claims full
            // support, so weighting has nothing to act on.
            if (runPosition(corruptLast(makeScene(line, pts), 200), 0, 400,
                            &flat, &err))
                flatSpread = pointSpread(flat);

            std::printf("       starved edge: point spread weak %.4f, "
                        "strong %.4f\n", weightedSpread, flatSpread);

            Check(weightedSpread <= flatSpread * 1.001,
                  "a thinly-supported camera scatters the points no more than "
                  "a well-supported one making the same error (" +
                      std::to_string(weightedSpread) + " vs " +
                      std::to_string(flatSpread) + ")");
        }

        // --- A POINT FAR OUTSIDE THE SCENE IS NOT KEPT ----------------------
        //
        // The three original triangulation tests -- parallax, cheirality,
        // reprojection -- can all pass for a point far beyond the cameras. Two
        // nearly-but-not-quite parallel rays clear the angle threshold and
        // meet at an enormous distance, the point is in front of both, and it
        // reprojects where it was seen because that is what placed it.
        //
        // Measured on castle-P19 this was not hypothetical: the triangulated
        // cloud was 13.3 units across and bundle adjustment returned one 65.9
        // across, from about 45 points of 1129. The viewport frames on the 2nd
        // to 98th percentile, so those strays sat outside the frame and made
        // the real structure look like two small clumps.
        //
        // Asserted by planting a point that is WRONG BY CONSTRUCTION: observed
        // by two cameras, consistent with both, and a hundred times further
        // away than the scene.
        {
            PointCloud pc = makeScene(arc, pts);

            // A track whose observations are the exact projections of a very
            // distant point, so nothing but the distance test can reject it.
            const Vec3 far{0.0, 0.0, 4000.0};
            Track distant;
            for (int c = 0; c < int(pc.cameras.size()); ++c) {
                double px, py;
                if (!pc.cameras[size_t(c)].Project(far, &px, &py)) continue;
                Observation o;
                o.frame = c;
                o.keypoint = 0;
                o.x = float(px);
                o.y = float(py);
                distant.obs.push_back(o);
            }
            const bool planted = distant.obs.size() >= 2;
            if (planted) pc.tracks.push_back(distant);
            const size_t distantIndex = pc.tracks.size() - 1;

            std::vector<Data> s2;
            s2.push_back(Data{std::move(pc)});
            Pipeline p2;
            p2.AddStage(Registry::Get().Create("triangulate"), "triangulate",
                        {{-1, 0}}, 1, 1);
            std::string err2;
            const bool ok2 = p2.Execute(&s2, nullptr, &err2);
            Check(ok2, "triangulate runs with a distant point planted" +
                           (ok2 ? "" : ": " + err2));

            if (ok2 && planted) {
                const Data* d2 = p2.Resolve({0, 0}, &s2);
                const PointCloud* out = d2 ? std::get_if<PointCloud>(d2) : nullptr;
                Check(out && distantIndex < out->tracks.size(),
                      "the planted track survived to the output");
                if (out && distantIndex < out->tracks.size()) {
                    Check(!out->tracks[distantIndex].hasPoint,
                          "a point a hundred times the scene radius away is "
                          "rejected, though it reprojects exactly and is in "
                          "front of every camera");
                }
            }
        }

        // --- it refuses rather than inventing --------------------------------
        {
            PointCloud noTracks = makeScene(arc, pts);
            noTracks.tracks.clear();
            PointCloud out;
            std::string err;
            const bool ok = runPosition(std::move(noTracks), 0, 50, &out, &err);
            Check(!ok && err.find("build_tracks") != std::string::npos,
                  "no tracks is a named error: \"" + err + "\"");
        }
        {
            PointCloud unsolved = makeScene(arc, pts);
            for (Camera& c : unsolved.cameras) c.solved = false;
            PointCloud out;
            std::string err;
            const bool ok = runPosition(std::move(unsolved), 0, 50, &out, &err);
            Check(!ok && err.find("rotation_average") != std::string::npos,
                  "no orientations is a named error: \"" + err + "\"");
        }
    }

    // --- triangulate and bundle_adjust_sfm ----------------------------------
    //
    // These two are tested together because they are only meaningful together:
    // triangulation needs cameras, bundle adjustment needs points, and the
    // quantity both are judged by is the same one -- reprojection error in
    // pixels.
    //
    // THE FIXTURE IS PERTURBED ON PURPOSE. A reconstruction built exactly from
    // ground truth has zero error and nothing to refine, so a bundle adjuster
    // that did nothing at all would pass. Starting from cameras nudged away
    // from the truth is what makes "it got better" a real claim.
    {
        std::printf("\n--- triangulate + bundle_adjust_sfm ---\n");

        const double focal = 700.0;
        const std::vector<Vec3> truthCentres = {
            {-2.0, 0.0, 0.0}, {-1.0, 0.3, 0.2}, {0.0, 0.5, 0.0},
            {1.0, 0.3, -0.2}, {2.0, 0.0, 0.0}};

        std::vector<Vec3> worldPts;
        for (int i = 0; i < 60; ++i) {
            const double a = double(i) * 0.41;
            worldPts.push_back(Vec3{std::cos(a) * 2.5, std::sin(a) * 1.8,
                                    7.0 + double(i % 7) * 0.5});
        }

        // Exact cameras, exact observations. `rotJitter` and `posJitter` then
        // move the cameras AWAY from the truth without touching the
        // observations -- so the reprojection error is real and the solver has
        // somewhere to go.
        auto makeScene = [&](double rotJitter, double posJitter,
                             double focalScale) {
            PointCloud pc;
            const int nCam = int(truthCentres.size());

            std::vector<Camera> truthCams;
            for (int c = 0; c < nCam; ++c) {
                Camera cam;
                cam.width = 1280; cam.height = 960;
                cam.cx = 640.0; cam.cy = 480.0;
                cam.focal = focal;
                // A small yaw per camera, so the views genuinely differ.
                cam.R = AxisAngleToMat(Vec3{0.0, double(c - 2) * 0.05, 0.0});
                cam.t = cam.R * truthCentres[size_t(c)] * -1.0;
                cam.solved = true;
                truthCams.push_back(cam);
            }

            // Observations from the TRUE cameras.
            for (const Vec3& wp : worldPts) {
                Track tr;
                for (int c = 0; c < nCam; ++c) {
                    double px, py;
                    if (!truthCams[size_t(c)].Project(wp, &px, &py)) continue;
                    if (px < 0 || py < 0 || px > 1280 || py > 960) continue;
                    Observation o;
                    o.frame = c; o.keypoint = 0;
                    o.x = float(px); o.y = float(py);
                    tr.obs.push_back(o);
                }
                if (tr.obs.size() >= 2) pc.tracks.push_back(tr);
            }

            // Cameras handed to the solver: the truth, displaced.
            for (int c = 0; c < nCam; ++c) {
                Camera cam = truthCams[size_t(c)];
                const double sign = (c % 2) ? 1.0 : -1.0;
                cam.R = AxisAngleToMat(Vec3{rotJitter * sign, rotJitter * 0.5,
                                            rotJitter * -0.3}) * cam.R;
                const Vec3 moved = truthCentres[size_t(c)] +
                                   Vec3{posJitter * sign, posJitter * 0.4,
                                        posJitter * 0.2};
                cam.t = cam.R * moved * -1.0;
                cam.focal = focal * focalScale;
                pc.cameras.push_back(cam);
            }
            return pc;
        };

        std::string lastReport;
        auto runStage = [&](const char* name, PointCloud in,
                            const std::vector<std::pair<const char*, double>>& params,
                            PointCloud* out, std::string* err) {
            std::vector<Data> s;
            s.push_back(Data{std::move(in)});
            Pipeline p;
            auto a = Registry::Get().Create(name);
            if (!a) { *err = std::string(name) + " not registered"; return false; }
            for (const auto& kv : params)
                if (ParamBase* pb = a->FindParam(kv.first)) {
                    std::string e; pb->SetFromScript(Value(kv.second), &e);
                }
            p.AddStage(std::move(a), name, {{-1, 0}}, 1, 1);
            if (!p.Execute(&s, nullptr, err)) return false;
            lastReport = p.Stages()[0].Report();
            const Data* d = p.Resolve({0, 0}, &s);
            const PointCloud* pc = d ? std::get_if<PointCloud>(d) : nullptr;
            if (!pc) { *err = "output is not a PointCloud"; return false; }
            *out = *pc;
            return true;
        };

        // RMS reprojection error, in pixels: the measure both stages exist to
        // reduce, and the only one that says a reconstruction is right.
        auto rms = [](const PointCloud& pc) {
            double sum = 0.0;
            int n = 0;
            for (const Track& tr : pc.tracks) {
                if (!tr.hasPoint) continue;
                for (const Observation& o : tr.obs) {
                    if (o.frame < 0 || o.frame >= int(pc.cameras.size())) continue;
                    const Camera& c = pc.cameras[size_t(o.frame)];
                    if (!c.solved) continue;
                    double px, py;
                    if (!c.Project(tr.point, &px, &py)) continue;
                    const double dx = px - double(o.x), dy = py - double(o.y);
                    sum += dx * dx + dy * dy;
                    ++n;
                }
            }
            return n ? std::sqrt(sum / double(n)) : 0.0;
        };

        // --- triangulation against exact cameras ----------------------------
        //
        // With cameras at the truth, the points must come back essentially
        // exact. This is the baseline that makes the perturbed cases readable.
        {
            PointCloud out;
            std::string err;
            const bool ok = runStage("triangulate", makeScene(0.0, 0.0, 1.0),
                                     {}, &out, &err);
            Check(ok, "triangulate runs" + (ok ? "" : ": " + err));
            if (ok) {
                Check(out.TriangulatedPoints() > 50,
                      "exact cameras triangulate nearly every track (" +
                          std::to_string(out.TriangulatedPoints()) + " of " +
                          std::to_string(out.tracks.size()) + ")");
                const double e = rms(out);
                Check(e < 0.01,
                      "...to sub-pixel accuracy (rms " + std::to_string(e) +
                          " px)");
            }
        }

        // --- the parallax rejection ------------------------------------------
        //
        // Demanding 30 degrees of parallax from a fixture whose widest baseline
        // subtends far less must reject nearly everything. The test is that the
        // filter ACTS, not that it is lenient: a triangulator that accepts
        // narrow rays fills a reconstruction with points at arbitrary depth.
        {
            PointCloud wide, narrow;
            std::string err;
            runStage("triangulate", makeScene(0.0, 0.0, 1.0),
                     {{"min_angle", 0.5}}, &wide, &err);
            const bool ok = runStage("triangulate", makeScene(0.0, 0.0, 1.0),
                                     {{"min_angle", 30.0}}, &narrow, &err);
            // Rejecting everything is a legitimate outcome here, and the stage
            // reports it as an error rather than returning an empty cloud.
            Check(!ok || narrow.TriangulatedPoints() < wide.TriangulatedPoints(),
                  "a strict parallax requirement rejects points a loose one "
                  "keeps");
        }

        // --- BUNDLE ADJUSTMENT: the central claim ----------------------------
        //
        // Points displaced from where triangulation would put them, cameras
        // left correct. BA must pull the points back, and the reprojection
        // error must fall to near zero -- there is a solution with zero error
        // and the solver should find it.
        //
        // PERTURBING THE POINTS RATHER THAN THE CAMERAS, and that choice is
        // the result of a wrong first attempt worth recording. Displacing the
        // cameras by a small rotation AND a small translation does not give BA
        // a recoverable problem: for a distant scene those two are nearly
        // interchangeable -- 0.004 rad of yaw and 0.02 of sideways translation
        // both move the image by about 2 px at focal 700 -- so the solver
        // finds a different-but-equally-good pose and the error plateaus
        // around 3 px however long it runs. That is a genuine property of the
        // geometry (the rotation/translation ambiguity at depth), not a defect,
        // and a test asserting a large drop there asserts something false.
        {
            PointCloud tri, adj;
            std::string err;
            const bool okT = runStage("triangulate", makeScene(0.0, 0.0, 1.0),
                                      {}, &tri, &err);
            Check(okT, "triangulate runs" + (okT ? "" : ": " + err));
            if (okT) {
                // Shove every point off its correct position. The cameras and
                // the observations still agree, so a zero-error solution
                // exists and BA has somewhere definite to go.
                PointCloud bent = tri;
                for (size_t i = 0; i < bent.tracks.size(); ++i) {
                    const double s = (i % 2) ? 1.0 : -1.0;
                    bent.tracks[i].point = bent.tracks[i].point +
                                           Vec3{0.05 * s, 0.04, 0.06 * s};
                }
                const double before = rms(bent);
                const bool okB = runStage("bundle_adjust_sfm", bent,
                                          {{"iterations", 60.0}}, &adj, &err);
                Check(okB, "bundle_adjust_sfm runs" + (okB ? "" : ": " + err));
                if (okB) {
                    std::printf("       %s\n", lastReport.c_str());
                    const double after = rms(adj);
                    std::printf("       reprojection rms %.4f -> %.4f px\n",
                                before, after);
                    Check(after < before,
                          "bundle adjustment reduces reprojection error (" +
                              std::to_string(before) + " -> " +
                              std::to_string(after) + " px)");
                    // NEAR ZERO, and the bound is tight on purpose.
                    //
                    // A zero-error solution exists here: the cameras and the
                    // observations agree exactly and only the points were
                    // displaced. A loose bound would have passed the bug this
                    // test was written to find -- the camera update read
                    // dCam[0..2] as rotation where the Jacobian wrote the
                    // centre there, so every step applied each correction to
                    // the wrong parameter. The solve still converged, firmly
                    // and repeatably, to 3.4 px. Identical under every loss
                    // and every iteration count, which is exactly what made it
                    // look like a plateau rather than a defect.
                    Check(after < before * 0.01,
                          "...to near zero, since a zero-error solution exists (" +
                              std::to_string(after) + " px)");
                }
            }
        }

        // --- each camera parameter is corrected in its OWN slot --------------
        //
        // Displace ONE camera purely in position, leaving its rotation exact,
        // and check that BA recovers the position. This is the test that pins
        // down the parameter layout: with the rotation and centre blocks
        // swapped in the update, the solver answers a pure translation error
        // with a rotation, which for a distant scene looks almost as good and
        // converges to a confident wrong answer.
        //
        // Checked against the KNOWN centre rather than against reprojection
        // error, because that is the distinction -- a wrong pose that
        // reprojects well is precisely the failure mode.
        {
            PointCloud tri, adj;
            std::string err;
            if (runStage("triangulate", makeScene(0.0, 0.0, 1.0), {}, &tri, &err)) {
                PointCloud bent = tri;
                const Vec3 shove{0.15, -0.1, 0.05};
                const int moved = 1;
                Camera& c = bent.cameras[size_t(moved)];
                c.t = c.R * (truthCentres[size_t(moved)] + shove) * -1.0;

                const double startErr =
                    (bent.cameras[size_t(moved)].Center() -
                     truthCentres[size_t(moved)]).Norm();

                if (runStage("bundle_adjust_sfm", bent,
                             {{"iterations", 80.0}}, &adj, &err)) {
                    const double endErr =
                        (adj.cameras[size_t(moved)].Center() -
                         truthCentres[size_t(moved)]).Norm();
                    std::printf("       camera centre error %.4f -> %.4f\n",
                                startErr, endErr);
                    // A FOURFOLD improvement, not a tenfold one. The bound was
                    // 0.2 when the damping was purely multiplicative; adding
                    // an absolute term -- which is what makes the solve work
                    // at all on real data, where the gauge directions leave
                    // the Hessian singular -- costs a little convergence on an
                    // easy problem. That is the right trade, and pinning the
                    // number here records it rather than letting the next
                    // person wonder why it is not tighter.
                    Check(endErr < startErr * 0.3,
                          "a displaced camera is returned to its true position (" +
                              std::to_string(startErr) + " -> " +
                              std::to_string(endErr) + ")");
                }
            }
        }

        // --- a good solution is not made worse -------------------------------
        //
        // The other half of the claim, and the one an over-eager solver breaks.
        // Starting from the exact answer, BA must leave it alone.
        {
            PointCloud tri, adj;
            std::string err;
            if (runStage("triangulate", makeScene(0.0, 0.0, 1.0), {}, &tri, &err)) {
                const double before = rms(tri);
                if (runStage("bundle_adjust_sfm", tri, {{"iterations", 30.0}},
                             &adj, &err)) {
                    const double after = rms(adj);
                    Check(after <= before + 1e-3,
                          "an already-correct solution is not degraded (" +
                              std::to_string(before) + " -> " +
                              std::to_string(after) + " px)");
                }
            }
        }

        // --- focal refinement --------------------------------------------------
        //
        // A wrong focal trades against depth almost exactly, so leaving it fixed
        // bakes the error into the geometry. Given a focal 12% off, refining it
        // must beat holding it.
        {
            PointCloud tri, refined, held;
            std::string err;
            if (runStage("triangulate", makeScene(0.0, 0.0, 1.12),
                         {{"max_error", 200.0}}, &tri, &err)) {
                const bool a = runStage("bundle_adjust_sfm", tri,
                                        {{"iterations", 80.0}, {"refine_focal", 1.0}},
                                        &refined, &err);
                const bool b = runStage("bundle_adjust_sfm", tri,
                                        {{"iterations", 80.0}, {"refine_focal", 0.0}},
                                        &held, &err);
                if (a && b) {
                    std::printf("       wrong focal: refined %.4f px, held %.4f px\n",
                                rms(refined), rms(held));
                    Check(rms(refined) <= rms(held),
                          "refining focal beats holding it when the guess is "
                          "wrong");

                    // AND IT MUST ACTUALLY MOVE. Comparing rms alone passes a
                    // refinement that does nothing, which is exactly the bug
                    // this caught on real data: the damping's additive term
                    // was scaled by the mean diagonal over ALL parameters, and
                    // the focal's own curvature -- dx/df is the normalised
                    // coordinate, order 1, against pixels-per-radian for
                    // rotation -- was orders of magnitude below it. The focal
                    // was damped to a standstill and reported back its input
                    // as a "measurement".
                    double startF = 0.0, endF = 0.0;
                    int nf = 0;
                    for (size_t i = 0; i < tri.cameras.size(); ++i) {
                        if (!tri.cameras[i].solved) continue;
                        startF += tri.cameras[i].focal;
                        endF   += refined.cameras[i].focal;
                        ++nf;
                    }
                    if (nf > 0) { startF /= double(nf); endF /= double(nf); }
                    const double moved =
                        std::fabs(endF - startF) / std::max(1.0, startF);
                    char fm[200];
                    std::snprintf(fm, sizeof(fm),
                                  "...and the focal MOVES toward the truth "
                                  "(%.1f -> %.1f, %.1f%%)",
                                  startF, endF, moved * 100.0);
                    // The fixture starts 12% off, so a real refinement moves
                    // several percent. Anything under 1% is a stalled solve.
                    Check(moved > 0.01, fm);

                    // A SHARED FOCAL MUST AGREE ACROSS CAMERAS, by
                    // construction. One lens took every frame, and letting
                    // each camera find its own lets each find its own local
                    // minimum -- measured on castle-P19, nineteen frames of
                    // one camera produced focals from 48.9 to 81.9 degrees.
                    double lo = 1e30, hi = -1e30;
                    for (size_t i = 0; i < refined.cameras.size(); ++i) {
                        if (!refined.cameras[i].solved) continue;
                        lo = std::min(lo, refined.cameras[i].focal);
                        hi = std::max(hi, refined.cameras[i].focal);
                    }
                    char sm[200];
                    std::snprintf(sm, sizeof(sm),
                                  "...and is SHARED across cameras "
                                  "(%.3f..%.3f)", lo, hi);
                    Check(hi - lo < 1e-6 * std::max(1.0, hi), sm);
                }
            }
        }

        // --- THE SOLVE MUST NOT STALL ON A SINGULAR GAUGE ---------------------
        //
        // Reprojection error is invariant to rotating, translating and scaling
        // the whole reconstruction, so the Hessian is singular in seven
        // directions however good the data is. Multiplicative damping alone
        // -- scaling the diagonal by (1 + lambda) -- damps a direction in
        // proportion to what is already there, which is nothing along a gauge
        // direction. Cholesky then fails at every lambda and the solver
        // accepts ZERO steps.
        //
        // The synthetic fixtures above did not catch this: a small, clean,
        // well-spread scene has enough curvature for the multiplicative term
        // to carry it. It took castle-P19 -- nineteen real photographs -- to
        // expose it, and the symptom there was bundle adjustment reporting the
        // error unchanged to three decimals after forty iterations.
        //
        // This asserts what that failure looked like: a solver that cannot
        // factor its system takes no steps at all.
        {
            PointCloud tri, adj;
            std::string err;
            if (runStage("triangulate", makeScene(0.0, 0.0, 1.0), {}, &tri, &err)) {
                PointCloud bent = tri;
                for (size_t i = 0; i < bent.tracks.size(); ++i)
                    bent.tracks[i].point = bent.tracks[i].point + Vec3{0.03, 0.02, 0.04};
                if (runStage("bundle_adjust_sfm", bent, {{"iterations", 40.0}},
                             &adj, &err)) {
                    // "in N accepted steps" -- zero means every factorisation
                    // failed, which is the signature of the singular gauge.
                    const size_t k = lastReport.find(" in ");
                    const int steps = (k == std::string::npos)
                                          ? -1 : std::atoi(lastReport.c_str() + k + 4);
                    Check(steps > 0,
                          "the solver takes steps rather than failing to factor "
                          "its system (" + std::to_string(steps) + " accepted)");
                }
            }
        }

        // --- it refuses rather than inventing ----------------------------------
        {
            PointCloud noPoints = makeScene(0.0, 0.0, 1.0);
            for (Track& t : noPoints.tracks) t.hasPoint = false;
            PointCloud out;
            std::string err;
            const bool ok = runStage("bundle_adjust_sfm", std::move(noPoints),
                                     {}, &out, &err);
            Check(!ok && err.find("nothing to refine") != std::string::npos,
                  "no triangulated points is a named error: \"" + err + "\"");
        }
    }

    // --- orbit camera -------------------------------------------------------
    //
    // The view-projection matrix is pure arithmetic and is exactly the kind of
    // thing that is wrong in a way no screenshot reveals: a cloud drawn with a
    // subtly wrong projection still looks like a cloud. Tested here, without a
    // window, by projecting points whose screen positions are knowable.
    {
        std::printf("\n--- orbit camera ---\n");

        // Applies the row-major view-projection and divides through.
        // Returns false when the point is behind the camera (w <= 0), which is
        // itself a thing worth checking.
        auto project = [](const OrbitCamera& c, const Vec3& p,
                          double* sx, double* sy, double* sz) {
            float m[16];
            c.ViewProj(m);
            const double x = p.x * m[0] + p.y * m[4] + p.z * m[8]  + m[12];
            const double y = p.x * m[1] + p.y * m[5] + p.z * m[9]  + m[13];
            const double z = p.x * m[2] + p.y * m[6] + p.z * m[10] + m[14];
            const double w = p.x * m[3] + p.y * m[7] + p.z * m[11] + m[15];
            if (w <= 1e-9) return false;
            *sx = x / w; *sy = y / w; *sz = z / w;
            return true;
        };

        OrbitCamera cam;
        cam.target = Vec3{0, 0, 0};
        cam.distance = 5.0;
        cam.yaw = 0.0;
        cam.pitch = 0.0;

        // The target projects to the centre of the screen. If this is wrong
        // nothing else can be right.
        {
            double sx, sy, sz;
            const bool ok = project(cam, cam.target, &sx, &sy, &sz);
            Check(ok, "the target is in front of the camera");
            Check(ok && std::fabs(sx) < 1e-5 && std::fabs(sy) < 1e-5,
                  "the target projects to the centre of the view (" +
                      std::to_string(sx) + ", " + std::to_string(sy) + ")");
        }

        // DEPTH RANGE IS [0,1], the D3D convention rather than OpenGL's
        // [-1,1]. Getting this wrong costs half the depth buffer's precision
        // and shows up as z-fighting rather than as an error.
        {
            double sx, sy, zNear, zFar;
            const Vec3 eye = cam.Eye();
            const Vec3 fwd = (cam.target - eye).Normalized();
            project(cam, eye + fwd * (cam.nearZ * 1.01), &sx, &sy, &zNear);
            project(cam, eye + fwd * (cam.farZ * 0.99), &sx, &sy, &zFar);
            Check(zNear >= -1e-4 && zNear < 0.2,
                  "the near plane maps near 0 (" + std::to_string(zNear) + ")");
            Check(zFar > 0.8 && zFar <= 1.0 + 1e-4,
                  "the far plane maps near 1 (" + std::to_string(zFar) + ")");
            Check(zNear < zFar, "...and depth increases with distance");
        }

        // A point to the camera's right lands on the right of the screen.
        // This is the handedness check, and the one that catches a mirrored
        // reconstruction -- which otherwise looks entirely plausible.
        {
            // THE EXPECTED DIRECTION IS DERIVED FROM THE WORLD, not from the
            // same cross product the viewer uses.
            //
            // This test previously computed `right = fwd x worldUp` -- the
            // exact expression in OrbitCamera::ViewProj -- and asserted the
            // projection agreed with it. That is circular: it passes for
            // EITHER handedness, because both sides flip together. The viewer
            // was mirroring every reconstruction left-to-right and this test
            // watched it happen.
            //
            // Tim found it with camera tracking markers taped to the wall in
            // fountain-P11: a blue square right of the fountain and a
            // black/white checker further right, which the reconstruction put
            // on the LEFT. Real ground truth in the photograph, which is the
            // only thing that could have caught a self-consistent mirror.
            //
            // So: state where the camera is and what "right" means there.
            // Looking down +Z from -Z, with +Y down (OpenCV), a point at +X
            // is to the viewer's LEFT -- because turning the world's +Y axis
            // downward swaps left and right. That is the whole subtlety, and
            // naming it is what makes the expectation checkable.
            cam.yaw = 0.0;
            cam.pitch = 0.0;
            const Vec3 eye = cam.Eye();
            const Vec3 fwd = (cam.target - eye).Normalized();

            // A right-handed basis with +Y up: right = up x fwd.
            const Vec3 rhRight = Vec3{0, 1, 0}.Cross(fwd).Normalized();

            double sx, sy, sz;
            if (project(cam, cam.target + rhRight * 0.5, &sx, &sy, &sz))
                Check(sx > 0.0,
                      "a point to the camera's right projects RIGHT (" +
                          std::to_string(sx) + ")");

            // ...and the opposite direction lands on the other side, so a
            // mirrored projection cannot satisfy both.
            if (project(cam, cam.target - rhRight * 0.5, &sx, &sy, &sz))
                Check(sx < 0.0,
                      "...and a point to its left projects LEFT (" +
                          std::to_string(sx) + ")");
            // +Y IS DOWN in the reconstruction -- OpenCV's convention, which
            // geometry.h states and COLMAP shares -- so a point at +0.5 y is
            // BELOW the target and must project below centre. Screen y is
            // positive upward, hence the negative expectation.
            //
            // This test previously asserted the opposite and passed, because
            // the viewer was flipping the scene to match: the fountain
            // rendered upside down and nothing caught it, since a point cloud
            // has no obvious top and the test agreed with the bug.
            double ux, uy, uz;
            if (project(cam, cam.target + Vec3{0, 0.5, 0}, &ux, &uy, &uz))
                Check(uy < 0.0,
                      "a point at +Y -- which is DOWN -- projects below centre (" +
                          std::to_string(uy) + ")");
        }

        // --- THE PROJECTION IS RIGID AS THE CAMERA TURNS --------------------
        //
        // Everything above checks ONE orientation. A projection can be right
        // at yaw 0 and wrong everywhere else -- a basis that is not
        // orthonormal, or an aspect applied in the wrong space, shears the
        // scene as it rotates rather than displacing it. Tim saw exactly that:
        // "really weird warping when I try to rotate the camera".
        //
        // Rigidity is testable without knowing what the right picture is: a
        // rotation preserves DISTANCES between points, so if the camera turns
        // and the view-space separation of two fixed points changes, the
        // transform is not a rotation. Checked in view space rather than on
        // screen, because the perspective divide legitimately changes screen
        // distances with depth.
        {
            auto viewSpace = [](const OrbitCamera& c, const Vec3& p) {
                float m[16];
                c.ViewProj(m);
                // The w row is the view-space forward component, and the
                // basis rows are the view axes scaled by the projection
                // terms. Undoing those scales recovers the view-space point.
                const double f = 1.0 / std::tan(c.fovY * 0.5);
                const double w = p.x * m[3] + p.y * m[7] + p.z * m[11] + m[15];
                const double x = (p.x * m[0] + p.y * m[4] + p.z * m[8]  + m[12]) / f;
                const double y = (p.x * m[1] + p.y * m[5] + p.z * m[9]  + m[13]) / f;
                return Vec3{x, y, w};
            };

            const Vec3 a{0.3, -0.2, 0.7}, b{-0.5, 0.4, -0.1};
            const double truth = (a - b).Norm();

            double worst = 0.0;
            double worstYaw = 0.0, worstPitch = 0.0;
            for (double yaw = -3.0; yaw <= 3.0; yaw += 0.37) {
                for (double pitch = -1.2; pitch <= 1.2; pitch += 0.31) {
                    OrbitCamera c;
                    c.target = Vec3{0, 0, 0};
                    c.distance = 5.0;
                    c.yaw = yaw;
                    c.pitch = pitch;

                    const double d = (viewSpace(c, a) - viewSpace(c, b)).Norm();
                    const double err = std::fabs(d - truth);
                    if (err > worst) { worst = err; worstYaw = yaw; worstPitch = pitch; }
                }
            }

            Check(worst < 1e-4,
                  "the view transform is rigid at every orientation (worst "
                  "separation error " + std::to_string(worst) + " at yaw " +
                      std::to_string(worstYaw) + ", pitch " +
                      std::to_string(worstPitch) + ")");
        }

        // Framing: a box must end up fully on screen, whatever its scale.
        // Scale is a gauge freedom in a reconstruction (see global_position),
        // so a viewer that only works at one scale is broken for half of them.
        for (double scale : {0.01, 1.0, 1000.0}) {
            OrbitCamera f;
            const Vec3 lo{-scale, -scale, -scale}, hi{scale, scale, scale};
            f.Frame(lo, hi);

            bool allVisible = true;
            const double corners[8][3] = {
                {-1,-1,-1}, {1,-1,-1}, {-1,1,-1}, {1,1,-1},
                {-1,-1,1},  {1,-1,1},  {-1,1,1},  {1,1,1}};
            for (const auto& c : corners) {
                double sx, sy, sz;
                const Vec3 p{c[0] * scale, c[1] * scale, c[2] * scale};
                if (!project(f, p, &sx, &sy, &sz) ||
                    std::fabs(sx) > 1.0 || std::fabs(sy) > 1.0 ||
                    sz < 0.0 || sz > 1.0)
                    allVisible = false;
            }
            Check(allVisible,
                  "Frame() fits a box of scale " + std::to_string(scale) +
                      " entirely on screen");

            // AND LEAVES THE DEPTH BUFFER USABLE. A perspective depth buffer
            // spends most of its range near the front, so a large near/far
            // ratio costs precision where the scene actually is -- and the
            // symptom is not a wrong picture but an UNSTABLE one: points at
            // similar depth flicker in and out as the camera turns.
            //
            // This was 1e5 (near = distance/10000, far = 100) and presented
            // exactly that way. Bounded rather than fixed, because the right
            // ratio depends on the scene; anything under a few hundred has
            // precision to spare.
            const double ratio = f.farZ / std::max(1e-12, f.nearZ);
            Check(ratio < 500.0,
                  "...with a depth range the buffer can resolve (near/far "
                  "ratio " + std::to_string(ratio) + ")");
        }

        // SCREEN-RELATIVE ROTATION. On an untilted view Turn() must do what
        // Rotate() does -- the mouse feels the same until the view tilts --
        // and on a tilted one a sideways turn must be about the SCREEN's up:
        // the screen's up stays put and the target stays in the middle.
        {
            OrbitCamera a, b;
            a.yaw = b.yaw = 0.4;
            a.pitch = b.pitch = 0.0;
            a.Rotate(0.05, 0.03);
            b.Turn(0.05, 0.03);
            const double d = (a.Eye() - b.Eye()).Norm();
            Check(d < 1e-3 * a.distance,
                  "Turn matches Rotate on an untilted view (eyes " + std::to_string(d) + " apart)");

            OrbitCamera t;
            t.Turn(0.0, 1.2);   // tilt well over
            Vec3 r0, u0, f0;
            t.Basis(&r0, &u0, &f0);
            t.Turn(0.7, 0.0);   // then sideways
            Vec3 r1, u1, f1;
            t.Basis(&r1, &u1, &f1);
            Check(u0.Dot(u1) > 0.9999,
                  "a sideways turn keeps the screen's up on a tilted view (" +
                      std::to_string(u0.Dot(u1)) + ")");
            Check(std::fabs((t.Eye() - t.target).Norm() - t.distance) < 1e-9,
                  "...and the distance to the target");
            for (int i = 0; i < 40; ++i) t.Turn(0.0, 0.2);   // over the top, no pole
            double sx, sy, sz;
            Check(project(t, t.target, &sx, &sy, &sz) && std::fabs(sx) < 1e-4 && std::fabs(sy) < 1e-4,
                  "...and the view stays valid right over the top");

            OrbitCamera m;
            const Vec3 e0 = m.Eye(), g0 = m.target;
            m.Move(0.0, 0.0, 0.5);
            Vec3 r, u, f;
            m.Basis(&r, &u, &f);
            Check(((m.target - g0) - f * (0.5 * m.distance)).Norm() < 1e-9 &&
                      ((m.Eye() - e0) - (m.target - g0)).Norm() < 1e-9,
                  "Move carries eye and target together along the view");
        }

        // Pitch must not reach vertical: there the up vector and the view
        // direction are parallel, the basis collapses, and the view flips.
        {
            OrbitCamera p;
            for (int i = 0; i < 100; ++i) p.Rotate(0.0, 1.0);
            Check(std::fabs(p.pitch) < 1.5708,
                  "pitch is clamped short of vertical (" +
                      std::to_string(p.pitch) + " rad)");
            double sx, sy, sz;
            Check(project(p, p.target, &sx, &sy, &sz) &&
                      std::fabs(sx) < 1e-4 && std::fabs(sy) < 1e-4,
                  "...so the view stays valid at the limit");
        }

        // Dolly is multiplicative, so in and out are inverses.
        {
            OrbitCamera d;
            const double start = d.distance;
            d.Dolly(5.0);
            Check(d.distance < start, "dolly in moves closer");
            d.Dolly(-5.0);
            Check(std::fabs(d.distance - start) < 1e-9,
                  "...and dollying back out returns exactly (" +
                      std::to_string(d.distance) + " vs " +
                      std::to_string(start) + ")");
        }

        // PAN AND ZOOM KEEP WHAT IS UNDER THE CURSOR UNDER THE CURSOR -- the
        // whole point of them. Checked against the RENDERED projection (the
        // ViewProj matrix, through `project`), not against the camera's own
        // helpers, so a flipped axis in the new maths cannot agree with
        // itself: this file has seen that happen once already.
        {
            const double W = 800.0, H = 500.0;
            OrbitCamera c;
            c.target = Vec3{0.3, -0.2, 4.0};
            c.distance = 3.0;
            c.yaw = 0.4;
            c.pitch = -0.3;
            // Pixel (from top-left) where the rendered matrix puts a point.
            auto pixel = [&](const OrbitCamera& cc, const Vec3& p, double* px, double* py) {
                double nx, ny, nz;
                if (!project(cc, p, &nx, &ny, &nz)) return false;
                nx /= W / H;   // the viewport's aspect correction (viewport3d.cpp)
                *px = (nx + 1.0) * 0.5 * W;
                *py = (1.0 - ny) * 0.5 * H;
                return true;
            };

            // Each projection is taken BEFORE its Check: the message is an
            // argument too, and C++ may build it before the condition runs,
            // which printed the previous check's numbers the first time.
            auto at = [](double x, double y) {
                return "(" + std::to_string(x) + ", " + std::to_string(y) + ")";
            };

            const Vec3 grab = c.OnTargetPlane(620.0, 140.0, W, H);
            double px = 0, py = 0;
            bool ok = pixel(c, grab, &px, &py);
            Check(ok && std::hypot(px - 620.0, py - 140.0) < 0.01,
                  "the point found under a pixel renders at that pixel " + at(px, py));

            double tx = 0, ty = 0, tz = 0;
            ok = c.ToScreen(grab, W, H, &tx, &ty, &tz);
            Check(ok && std::hypot(tx - px, ty - py) < 0.01,
                  "ToScreen agrees with the rendered projection " + at(tx, ty));

            OrbitCamera p2 = c;
            p2.Pan(35.0, -20.0, H);
            ok = pixel(p2, grab, &px, &py);
            Check(ok && std::hypot(px - 655.0, py - 120.0) < 0.05,
                  "a pan drags the scene with the mouse, pixel for pixel " + at(px, py) +
                      " vs (655, 120)");

            OrbitCamera z = c;
            z.DollyAt(4.0, 620.0, 140.0, W, H);
            ok = pixel(z, grab, &px, &py);
            Check(z.distance < c.distance && ok && std::hypot(px - 620.0, py - 140.0) < 0.05,
                  "zooming at the cursor keeps the point under it " + at(px, py));
            Check(z.nearZ < z.distance,
                  "...and the near plane follows in, so the target is not clipped");

            OrbitCamera f = c;
            const Vec3 other{1.0, 0.5, 5.0};
            f.FocusOn(other);
            ok = pixel(f, other, &px, &py);
            Check(ok && std::hypot(px - W * 0.5, py - H * 0.5) < 0.05,
                  "focusing on a point puts it in the middle of the view " + at(px, py));
        }
    }

    // --- relative_pose, against known two-view geometry ---------------------
    //
    // THE GAP THIS FILLS. Every other stage here is tested against ground
    // truth, but relative_pose was only ever exercised through the bench on
    // real photographs -- where a wrong answer is indistinguishable from hard
    // data. On castle-P19 it solves 49 of 51 pairs and rotation averaging then
    // reports 15 degrees of mean residual with a 105 degree worst case: every
    // edge individually plausible, the set of them mutually contradictory.
    // That is the signature of a CONVENTION error, which no amount of real
    // data can distinguish from noise.
    //
    // So: two synthetic cameras with a known relative rotation, exact
    // correspondences, and a check that what comes back is what went in.
    {
        std::printf("\n--- relative_pose ---\n");

        const double focal = 800.0;
        const int W = 1600, H = 1200;

        // Two cameras looking at a cloud of points, separated by a known
        // rotation and translation. The rotation is about two axes so a
        // transposed result cannot masquerade as the right one -- a pure yaw
        // and its inverse differ only in sign, which a sloppy test would miss.
        const Mat3 R_a = AxisAngleToMat(Vec3{0.05, -0.12, 0.02});
        const Mat3 R_b = AxisAngleToMat(Vec3{-0.03, 0.18, -0.06});
        const Vec3 C_a{-0.5, 0.1, 0.0};
        const Vec3 C_b{0.6, -0.05, 0.2};

        auto makeCam = [&](const Mat3& R, const Vec3& C) {
            Camera c;
            c.width = W; c.height = H;
            c.cx = 0.5 * W; c.cy = 0.5 * H;
            c.focal = focal;
            c.R = R;
            c.t = R * C * -1.0;
            c.solved = true;
            return c;
        };
        const Camera camA = makeCam(R_a, C_a);
        const Camera camB = makeCam(R_b, C_b);

        // THE ANSWER relative_pose should produce, by its own documented
        // convention: x_b = R * x_a, so R = R_b * R_a^T.
        const Mat3 expected = R_b * R_a.Transpose();

        // A frame pair carrying exact correspondences of a point cloud.
        ImageSet set;
        for (int f = 0; f < 2; ++f) {
            Image im;
            im.Alloc({W, H, Format::RGBA8});
            set.images.push_back(std::move(im));
        }

        auto fsA = std::make_shared<FeatureSidecar>();
        auto fsB = std::make_shared<FeatureSidecar>();
        auto ms  = std::make_shared<MatchSidecar>();
        MatchSet mset;
        mset.reference = 0;

        int n = 0;
        for (int i = 0; i < 200; ++i) {
            // Spread in depth as well as across the frame: a planar scene is
            // the eight-point algorithm's known degeneracy, and a fixture that
            // was accidentally planar would be testing that instead.
            const double a = double(i) * 0.7;
            const Vec3 wp{std::cos(a) * 1.5, std::sin(a * 1.3) * 1.2,
                          6.0 + std::fmod(double(i), 5.0) * 0.8};
            double ax, ay, bx, by;
            if (!camA.Project(wp, &ax, &ay)) continue;
            if (!camB.Project(wp, &bx, &by)) continue;
            if (ax < 0 || ay < 0 || ax >= W || ay >= H) continue;
            if (bx < 0 || by < 0 || bx >= W || by >= H) continue;

            Keypoint ka; ka.x = float(ax); ka.y = float(ay);
            Keypoint kb; kb.x = float(bx); kb.y = float(by);
            fsA->keypoints.push_back(ka);
            fsB->keypoints.push_back(kb);

            Match m;
            m.a = n; m.b = n;
            mset.matches.push_back(m);
            ++n;
        }
        set.images[0].Sidecars().Set(kFeatureSidecar, std::move(fsA));
        set.images[1].Sidecars().Set(kFeatureSidecar, std::move(fsB));
        ms->sets.push_back(std::move(mset));
        set.images[1].Sidecars().Set(kMatchSidecar, std::move(ms));
        set.shape = Shape{{{"frame", 2}}};

        Check(n > 50, "the fixture has correspondences (" +
                          std::to_string(n) + ")");

        // The true horizontal field of view, so the normalised coordinates are
        // right. A wrong focal is a separate failure and this test is not
        // about that one.
        const double trueFov =
            2.0 * std::atan2(0.5 * W, focal) * 180.0 / 3.14159265358979;

        std::vector<Data> s;
        s.push_back(Data{std::move(set)});
        Pipeline p;
        auto rp = Registry::Get().Create("relative_pose");
        if (!rp) { Check(false, "relative_pose is registered"); }
        else {
            if (ParamBase* pb = rp->FindParam("fov_deg")) {
                std::string e; pb->SetFromScript(Value(trueFov), &e);
            }
            p.AddStage(std::move(rp), "relative_pose", {{-1, 0}}, 1, 1);

            std::string err;
            const bool ok = p.Execute(&s, nullptr, &err);
            Check(ok, "relative_pose runs" + (ok ? "" : ": " + err));

            if (ok) {
                const Data* d = p.Resolve({0, 0}, &s);
                const ImageSet* out = d ? std::get_if<ImageSet>(d) : nullptr;
                const RelativePoseSidecar* got =
                    out ? RelativePosesOf(out->images[1]) : nullptr;

                Check(got && !got->edges.empty(),
                      "...and attaches an edge to the second frame");
                if (got && !got->edges.empty()) {
                    const auto& e = got->edges[0];
                    Check(e.reference == 0, "the edge names frame 0 as reference");

                    const double errDeg =
                        e.R.AngleTo(expected) * 180.0 / 3.14159265358979;
                    std::printf("       recovered rotation differs by %.4f deg\n",
                                errDeg);
                    Check(errDeg < 1.0,
                          "the recovered rotation is R_b * R_a^T, as documented (" +
                              std::to_string(errDeg) + " deg)");

                    // THE TRANSPOSE CHECK. If the convention were inverted the
                    // result would be the transpose, which is a perfectly valid
                    // rotation and differs from the right one by twice the
                    // relative angle -- large here, and invisible on a chain
                    // where nothing composes.
                    const double transposedDeg =
                        e.R.AngleTo(expected.Transpose()) * 180.0 / 3.14159265358979;
                    Check(errDeg < transposedDeg,
                          "...and not its transpose (" + std::to_string(errDeg) +
                              " vs " + std::to_string(transposedDeg) + " deg)");

                    // The translation direction, in frame a's coordinates.
                    // SIGNED: translation averaging cannot see a flipped sign
                    // -- the mirrored layout satisfies every direction -- but
                    // anything weighing the directions against rays can.
                    const Vec3 wantDir = (R_a * (C_b - C_a)).Normalized();
                    const double dot = e.direction.Dot(wantDir);
                    Check(dot > 0.99,
                          "the translation direction points from a to b (cos " +
                              std::to_string(dot) + ")");
                }
            }
        }
    }

    // --- relative_pose ESTIMATES the focal, and CHECKS revisits ---------------
    //
    // Six cameras on an arc with a known focal, matched in a chain of window
    // 2, with 0.3 px of noise -- the estimate must land near the truth with
    // no fov given. Then two REVISIT pairs onto the last frame: a true one
    // against frame 0, and a look-alike whose matches come from a different
    // point cloud seen as if camera 5 were turned 25 degrees further. RANSAC
    // accepts both -- each is consistent with some essential matrix -- and
    // only the chain can tell them apart.
    {
        std::printf("\n--- relative_pose: focal estimate and revisit check ---\n");
        const int W = 1600, H = 1200, N = 6;
        const double focal = 1400.0;
        const double trueFov = 2.0 * std::atan2(0.5 * W, focal) * 180.0 / 3.14159265358979;
        std::vector<Camera> cams;
        for (int c = 0; c < N; ++c) {
            Camera cam;
            cam.width = W; cam.height = H; cam.cx = 0.5 * W; cam.cy = 0.5 * H;
            cam.focal = focal;
            cam.R = AxisAngleToMat(Vec3{0.02 * c, 0.09 * c - 0.2, 0.01 * c});
            const Vec3 C{-1.0 + 0.4 * c, 0.05 * c, 0.1 * std::sin(double(c))};
            cam.t = cam.R * C * -1.0;
            cam.solved = true;
            cams.push_back(cam);
        }
        uint32_t seed = 99u;
        auto noise = [&]() {
            seed = seed * 1664525u + 1013904223u;
            return (double(seed >> 8) / double(1u << 24) - 0.5) * 0.6;
        };
        std::vector<Vec3> world;
        for (int i = 0; i < 500; ++i) {
            const double a = double(i) * 0.61;
            world.push_back(Vec3{std::cos(a) * 2.2, std::sin(a * 1.7) * 1.6,
                                 7.0 + std::fmod(double(i) * 0.37, 4.0)});
        }
        std::vector<std::shared_ptr<FeatureSidecar>> fs;
        for (int c = 0; c < N; ++c) fs.push_back(std::make_shared<FeatureSidecar>());
        // Matches between two cameras from a list of points: each visible
        // point becomes a keypoint in both and a match between them.
        auto addPair = [&](int a, int b, const Camera& ca, const Camera& cb,
                           const std::vector<Vec3>& pts, bool revisit) {
            MatchSet ms;
            ms.reference = a;
            ms.revisit = revisit;
            for (const Vec3& wp : pts) {
                double ax, ay, bx, by;
                if (!ca.Project(wp, &ax, &ay) || !cb.Project(wp, &bx, &by)) continue;
                if (ax < 0 || ay < 0 || ax >= W || ay >= H) continue;
                if (bx < 0 || by < 0 || bx >= W || by >= H) continue;
                Keypoint ka; ka.x = float(ax + noise()); ka.y = float(ay + noise());
                Keypoint kb; kb.x = float(bx + noise()); kb.y = float(by + noise());
                Match m;
                m.a = int(fs[size_t(a)]->keypoints.size());
                m.b = int(fs[size_t(b)]->keypoints.size());
                fs[size_t(a)]->keypoints.push_back(ka);
                fs[size_t(b)]->keypoints.push_back(kb);
                ms.matches.push_back(m);
            }
            return ms;
        };
        std::vector<std::shared_ptr<MatchSidecar>> msc;
        for (int c = 0; c < N; ++c) msc.push_back(std::make_shared<MatchSidecar>());
        for (int b = 1; b < N; ++b)
            for (int w = 1; w <= 2 && b - w >= 0; ++w)
                msc[size_t(b)]->sets.push_back(addPair(b - w, b, cams[size_t(b - w)],
                                                       cams[size_t(b)], world, false));
        // The true revisit, and the look-alike.
        msc[N - 1]->sets.push_back(addPair(0, N - 1, cams[0], cams[N - 1], world, true));
        Camera turned = cams[N - 1];
        turned.R = AxisAngleToMat(Vec3{0, 0.44, 0}) * turned.R;   // 25 deg further
        std::vector<Vec3> other;
        for (const Vec3& p : world) other.push_back(Vec3{p.x * 0.8 + 0.5, p.y, p.z + 1.0});
        {
            // Matched against frame 1 so it does not collide with the true
            // revisit's reference.
            MatchSet fake = addPair(1, N - 1, cams[1], turned, other, true);
            msc[N - 1]->sets.push_back(std::move(fake));
        }
        ImageSet set;
        for (int c = 0; c < N; ++c) {
            Image im;
            im.Alloc({W, H, Format::RGBA8});
            im.Sidecars().Set(kFeatureSidecar, fs[size_t(c)]);
            if (!msc[size_t(c)]->sets.empty()) im.Sidecars().Set(kMatchSidecar, msc[size_t(c)]);
            set.images.push_back(std::move(im));
        }
        set.shape = Shape{{{"frame", N}}};
        std::vector<Data> s;
        s.push_back(Data{std::move(set)});
        Pipeline p;
        auto rp = Registry::Get().Create("relative_pose");   // fov_deg left at 0
        p.AddStage(std::move(rp), "relative_pose", {{-1, 0}}, 1, 1);
        std::string err;
        const bool ok = p.Execute(&s, nullptr, &err);
        Check(ok, "relative_pose runs with no fov given" + (ok ? "" : ": " + err));
        const Data* d = ok ? p.Resolve({0, 0}, &s) : nullptr;
        const ImageSet* out = d ? std::get_if<ImageSet>(d) : nullptr;
        const RelativePoseSidecar* last = out ? RelativePosesOf(out->images[N - 1]) : nullptr;
        if (last) {
            char m[160];
            std::snprintf(m, sizeof m, "the field of view is estimated from the matches "
                          "(%.1f deg against a true %.1f)", last->fovDeg, trueFov);
            Check(std::fabs(last->fovDeg - trueFov) < 0.05 * trueFov, m);
            bool trueKept = false, fakeKept = false;
            for (const auto& e : last->edges) {
                if (e.revisit && e.reference == 0) trueKept = true;
                if (e.revisit && e.reference == 1) fakeKept = true;
            }
            Check(trueKept, "a true revisit is kept");
            Check(!fakeKept, "...and a look-alike one is dropped");
            const MatchSidecar* mm = MatchesOf(out->images[N - 1]);
            bool fakeCleared = false;
            if (mm)
                for (const MatchSet& ms : mm->sets)
                    if (ms.revisit && ms.reference == 1) {
                        fakeCleared = !ms.inlier.empty();
                        for (uint8_t v : ms.inlier) fakeCleared &= v == 0;
                    }
            Check(fakeCleared, "...along with its matches, so no track is built through it");
        } else {
            Check(false, "the last frame carries relative poses");
        }
    }

    // --- relative_pose on a PLANAR scene, the eight-point degeneracy --------
    //
    // The test above deliberately spreads its points in depth, because a
    // planar fixture would be testing the degeneracy instead of the estimator.
    // This one IS that fixture, on purpose.
    //
    // WHY IT MATTERS HERE, beyond textbook completeness: castle-P19 splits into
    // two clusters at one specific link (frames 11-12), and those two frames
    // are a courtyard corner -- two flat plastered wings, the camera pivoting
    // between them. The hypothesis is that the pair is near-planar, that E is
    // therefore underdetermined, and that the translation direction which
    // global_position depends on is consequently arbitrary.
    //
    // A hypothesis about real data that cannot be checked on real data, since
    // there is no ground truth for castle-P19 here. So it is checked where the
    // answer IS known: every point on one plane, exact correspondences, no
    // noise. If eight-point recovers the pose anyway, planarity is not the
    // explanation for the split and the search goes elsewhere.
    {
        std::printf("\n--- relative_pose, planar scene ---\n");

        const double focal = 800.0;
        const int W = 1600, H = 1200;

        const Mat3 R_a = AxisAngleToMat(Vec3{0.05, -0.12, 0.02});
        const Mat3 R_b = AxisAngleToMat(Vec3{-0.03, 0.18, -0.06});
        const Vec3 C_a{-0.5, 0.1, 0.0};
        const Vec3 C_b{0.6, -0.05, 0.2};

        auto makeCam = [&](const Mat3& R, const Vec3& C) {
            Camera c;
            c.width = W; c.height = H;
            c.cx = 0.5 * W; c.cy = 0.5 * H;
            c.focal = focal;
            c.R = R;
            c.t = R * C * -1.0;
            c.solved = true;
            return c;
        };
        const Camera camA = makeCam(R_a, C_a);
        const Camera camB = makeCam(R_b, C_b);
        const Mat3 expected = R_b * R_a.Transpose();

        ImageSet set;
        for (int f = 0; f < 2; ++f) {
            Image im;
            im.Alloc({W, H, Format::RGBA8});
            set.images.push_back(std::move(im));
        }

        auto fsA = std::make_shared<FeatureSidecar>();
        auto fsB = std::make_shared<FeatureSidecar>();
        auto ms  = std::make_shared<MatchSidecar>();
        MatchSet mset;
        mset.reference = 0;

        int n = 0;
        for (int i = 0; i < 200; ++i) {
            // EXACTLY PLANAR: z is a fixed function of x and y, so every point
            // lies on one tilted plane. A facade, in other words. The tilt is
            // deliberate -- a fronto-parallel plane is an even weaker case and
            // would overstate the problem.
            const double a = double(i) * 0.7;
            const double x = std::cos(a) * 1.5;
            const double y = std::sin(a * 1.3) * 1.2;
            const Vec3 wp{x, y, 6.0 + 0.3 * x + 0.15 * y};
            double ax, ay, bx, by;
            if (!camA.Project(wp, &ax, &ay)) continue;
            if (!camB.Project(wp, &bx, &by)) continue;
            if (ax < 0 || ay < 0 || ax >= W || ay >= H) continue;
            if (bx < 0 || by < 0 || bx >= W || by >= H) continue;

            Keypoint ka; ka.x = float(ax); ka.y = float(ay);
            Keypoint kb; kb.x = float(bx); kb.y = float(by);
            fsA->keypoints.push_back(ka);
            fsB->keypoints.push_back(kb);

            Match m;
            m.a = n; m.b = n;
            mset.matches.push_back(m);
            ++n;
        }
        set.images[0].Sidecars().Set(kFeatureSidecar, std::move(fsA));
        set.images[1].Sidecars().Set(kFeatureSidecar, std::move(fsB));
        ms->sets.push_back(std::move(mset));
        set.images[1].Sidecars().Set(kMatchSidecar, std::move(ms));
        set.shape = Shape{{{"frame", 2}}};

        Check(n > 50, "the planar fixture has correspondences (" +
                          std::to_string(n) + ")");

        const double trueFov =
            2.0 * std::atan2(0.5 * W, focal) * 180.0 / 3.14159265358979;

        std::vector<Data> s;
        s.push_back(Data{std::move(set)});
        Pipeline p;
        auto rp = Registry::Get().Create("relative_pose");
        if (rp) {
            if (ParamBase* pb = rp->FindParam("fov_deg")) {
                std::string e; pb->SetFromScript(Value(trueFov), &e);
            }
            p.AddStage(std::move(rp), "relative_pose", {{-1, 0}}, 1, 1);

            std::string err;
            const bool ok = p.Execute(&s, nullptr, &err);

            if (ok) {
                const Data* d = p.Resolve({0, 0}, &s);
                const ImageSet* out = d ? std::get_if<ImageSet>(d) : nullptr;
                const RelativePoseSidecar* got =
                    out ? RelativePosesOf(out->images[1]) : nullptr;

                if (got && !got->edges.empty()) {
                    const auto& e = got->edges[0];
                    const double errDeg =
                        e.R.AngleTo(expected) * 180.0 / 3.14159265358979;
                    const Vec3 wantDir = (R_a * (C_b - C_a)).Normalized();
                    const double dot = std::fabs(e.direction.Dot(wantDir));

                    // REPORTED, NOT ASSERTED, and the distinction is the point
                    // of this block. Whether eight-point copes with an exactly
                    // planar pair is the question being asked; writing an
                    // assertion either way would be encoding the answer before
                    // measuring it. The numbers go in the log, and the
                    // assertion below covers only what is not in doubt.
                    std::printf("       planar: rotation off by %.4f deg, "
                                "translation direction cos %.4f\n",
                                errDeg, dot);

                    // The stage's own planarity estimate, against a fixture
                    // that is planar by construction. This is the calibration
                    // the measure needs: on real data a low number is
                    // ambiguous between "not planar" and "the fit is broken",
                    // and only a known-planar case can tell those apart.
                    std::printf("       stage says: %s\n",
                                p.Stages()[0].Report().c_str());
                } else {
                    // Also a legitimate outcome, and arguably the RIGHT one:
                    // refusing a pair it cannot solve beats returning an
                    // arbitrary pose that every later stage trusts.
                    std::printf("       planar: no edge produced "
                                "(the pair was rejected)\n");
                }
                Check(true, "relative_pose survives an exactly planar pair "
                            "without crashing");
            } else {
                std::printf("       planar: pipeline failed: %s\n", err.c_str());
                Check(true, "relative_pose survives an exactly planar pair "
                            "without crashing");
            }
        }
    }

    // --- 8-point against 5-point, on the geometry that separates them -------
    //
    // The planar block above establishes that the linear method fails on an
    // exactly planar pair: a rotation several degrees off and a translation
    // direction tens of degrees off, on EXACT noise-free correspondences. That
    // is the textbook degeneracy, and it is the reason the five-point method
    // exists.
    //
    // This runs both solvers over the same two fixtures -- one planar, one
    // with depth -- and reports each. The general fixture is the control: a
    // five-point implementation that did well on planes and badly on ordinary
    // geometry would be worse than useless, and only running both says so.
    //
    // Reported rather than asserted, except for the one claim that is not in
    // doubt: both must return SOMETHING on well-conditioned input. Which is
    // more accurate on this data is the measurement being taken, and an
    // assertion would be encoding the expected answer in advance -- a mistake
    // that has already cost real time on this pipeline.
    {
        std::printf("\n--- 5-point vs 8-point ---\n");

        const double focal = 800.0;
        const int W = 1600, H = 1200;
        const Mat3 R_a = AxisAngleToMat(Vec3{0.05, -0.12, 0.02});
        const Mat3 R_b = AxisAngleToMat(Vec3{-0.03, 0.18, -0.06});
        const Vec3 C_a{-0.5, 0.1, 0.0};
        const Vec3 C_b{0.6, -0.05, 0.2};

        auto makeCam = [&](const Mat3& R, const Vec3& C) {
            Camera c;
            c.width = W; c.height = H;
            c.cx = 0.5 * W; c.cy = 0.5 * H;
            c.focal = focal;
            c.R = R;
            c.t = R * C * -1.0;
            c.solved = true;
            return c;
        };
        const Camera camA = makeCam(R_a, C_a);
        const Camera camB = makeCam(R_b, C_b);
        const Mat3 expected = R_b * R_a.Transpose();
        const Vec3 wantDir = (R_a * (C_b - C_a)).Normalized();

        // `planar` picks which of the two scene shapes to build; `noisePx`
        // and `outlierFrac` make it resemble real matching output.
        //
        // NOISE IS NOT A REFINEMENT OF THIS TEST, it is the test. On exact
        // correspondences the constraints a five-point solver enforces can be
        // satisfied exactly, and any implementation that converges at all
        // looks perfect. Real matches are a pixel or two off and a third of
        // them are wrong outright, and that is where a minimal solver either
        // earns its complexity or falls apart.
        auto buildSet = [&](bool planar, double noisePx = 0.0,
                            double outlierFrac = 0.0) {
            // Deterministic: a comparison between two estimators must see the
            // same perturbations, or the difference measured is the seed.
            std::mt19937 rng(1234567u);
            std::normal_distribution<double> gauss(0.0, noisePx);
            std::uniform_real_distribution<double> uni(0.0, 1.0);
            std::uniform_real_distribution<double> anywhere(0.0, 1.0);
            ImageSet set;
            for (int f = 0; f < 2; ++f) {
                Image im;
                im.Alloc({W, H, Format::RGBA8});
                set.images.push_back(std::move(im));
            }
            auto fsA = std::make_shared<FeatureSidecar>();
            auto fsB = std::make_shared<FeatureSidecar>();
            auto ms  = std::make_shared<MatchSidecar>();
            MatchSet mset;
            mset.reference = 0;

            int n = 0;
            for (int i = 0; i < 200; ++i) {
                const double a = double(i) * 0.7;
                const double x = std::cos(a) * 1.5;
                const double y = std::sin(a * 1.3) * 1.2;
                const Vec3 wp = planar
                                    ? Vec3{x, y, 6.0 + 0.3 * x + 0.15 * y}
                                    : Vec3{x, y, 6.0 + std::fmod(double(i), 5.0) * 0.8};
                double ax, ay, bx, by;
                if (!camA.Project(wp, &ax, &ay)) continue;
                if (!camB.Project(wp, &bx, &by)) continue;
                if (ax < 0 || ay < 0 || ax >= W || ay >= H) continue;
                if (bx < 0 || by < 0 || bx >= W || by >= H) continue;

                if (noisePx > 0.0) {
                    ax += gauss(rng); ay += gauss(rng);
                    bx += gauss(rng); by += gauss(rng);
                }
                // An OUTLIER here is a match to somewhere else entirely, which
                // is what a repeated window produces: the descriptors agree
                // and the geometry does not. Displacing the point slightly
                // would model a different and much easier failure.
                if (outlierFrac > 0.0 && uni(rng) < outlierFrac) {
                    bx = anywhere(rng) * double(W);
                    by = anywhere(rng) * double(H);
                }

                Keypoint ka; ka.x = float(ax); ka.y = float(ay);
                Keypoint kb; kb.x = float(bx); kb.y = float(by);
                fsA->keypoints.push_back(ka);
                fsB->keypoints.push_back(kb);
                Match m;
                m.a = n; m.b = n;
                mset.matches.push_back(m);
                ++n;
            }
            set.images[0].Sidecars().Set(kFeatureSidecar, std::move(fsA));
            set.images[1].Sidecars().Set(kFeatureSidecar, std::move(fsB));
            ms->sets.push_back(std::move(mset));
            set.images[1].Sidecars().Set(kMatchSidecar, std::move(ms));
            set.shape = Shape{{{"frame", 2}}};
            return set;
        };

        const double trueFov =
            2.0 * std::atan2(0.5 * W, focal) * 180.0 / 3.14159265358979;

        // Returns false when no edge came back at all, which is a legitimate
        // outcome for a solver that declines a pair it cannot handle.
        auto run = [&](bool planar, int method, double noisePx,
                       double outlierFrac, double* rotDeg, double* dirCos) {
            std::vector<Data> s;
            s.push_back(Data{buildSet(planar, noisePx, outlierFrac)});
            Pipeline p;
            auto rp = Registry::Get().Create("relative_pose");
            if (!rp) return false;
            if (ParamBase* pb = rp->FindParam("fov_deg")) {
                std::string e; pb->SetFromScript(Value(trueFov), &e);
            }
            if (ParamBase* pb = rp->FindParam("method")) {
                std::string e; pb->SetFromScript(Value(double(method)), &e);
            }
            p.AddStage(std::move(rp), "relative_pose", {{-1, 0}}, 1, 1);
            std::string err;
            if (!p.Execute(&s, nullptr, &err)) return false;
            const Data* d = p.Resolve({0, 0}, &s);
            const ImageSet* out = d ? std::get_if<ImageSet>(d) : nullptr;
            const RelativePoseSidecar* got =
                out ? RelativePosesOf(out->images[1]) : nullptr;
            if (!got || got->edges.empty()) return false;
            const auto& e = got->edges[0];
            *rotDeg = e.R.AngleTo(expected) * 180.0 / 3.14159265358979;
            *dirCos = std::fabs(e.direction.Dot(wantDir));
            return true;
        };

        // Four conditions: the two scene shapes, each clean and each with the
        // noise and outlier rate real matching produces. castle-P19's weakest
        // pairs run at a 30% inlier rate, so 0.6 here is if anything kind.
        struct Cond { bool planar; double noise; double outliers; const char* name; };
        const Cond conds[] = {
            {false, 0.0, 0.0,  "general, exact"},
            {true,  0.0, 0.0,  "planar,  exact"},
            {false, 1.0, 0.6,  "general, 1px noise + 60% outliers"},
            {true,  1.0, 0.6,  "planar,  1px noise + 60% outliers"},
            {false, 1.0, 0.3,  "general, 1px noise + 30% outliers"},
            {true,  0.3, 0.0,  "planar,  0.3px noise, no outliers"},
            {true,  1.0, 0.0,  "planar,  1px noise, no outliers"},
            {true,  1.0, 0.3,  "planar,  1px noise + 30% outliers"},
        };

        for (const Cond& c : conds) {
            std::printf("       %s:\n", c.name);
            bool anySolved = false;
            for (int method = 0; method <= 1; ++method) {
                double rotDeg = 0.0, dirCos = 0.0;
                if (run(c.planar, method, c.noise, c.outliers, &rotDeg, &dirCos)) {
                    anySolved = true;
                    std::printf("         %-8s rotation %8.4f deg, "
                                "direction cos %.4f\n",
                                method == 0 ? "8-point" : "5-point",
                                rotDeg, dirCos);
                } else {
                    std::printf("         %-8s no edge (declined the pair)\n",
                                method == 0 ? "8-point" : "5-point");
                }
            }
            if (!c.planar && c.noise == 0.0) {
                Check(anySolved,
                      "at least one solver handles a well-conditioned pair");
            }
        }
    }

    // --- a dense stage gets both a cloud and the frames ---------------------
    //
    // A dense reconstruction stage -- plane sweeping, PMVS -- reads PIXELS,
    // and by the time the chain reaches it the pipeline is passing a
    // PointCloud and the source frames are gone. So a second input port may
    // carry them, and this checks the pipeline actually delivers it.
    //
    // WRITTEN TO FAIL IF THE PLUMBING IS ABSENT. A fixture that merely runs
    // the stage would pass whether or not the frames arrived, which is the
    // trap test_script.cpp's shape fixtures document. So the stage RECORDS
    // what it was handed and the assertions read those recordings back.
    {
        std::printf("\n--- dense stage plumbing ---\n");

        // Shared with the fixture below. Static so the algorithm can reach it
        // without the registry needing to hand back the instance it made.
        struct Seen {
            bool  ran        = false;
            bool  hadFrames  = false;
            int   frameCount = 0;
            bool  hadCloud   = false;
            int   cameras    = 0;
        };
        static Seen g_seen;
        g_seen = Seen{};

        struct FakeDense : AlgorithmBase {
            const char* Name()     const override { return "test_dense"; }
            const char* Category() const override { return "sfm"; }

            PortList Inputs() const override {
                return {{"src", DataType::PointCloud, FormatSpec::Any, ShapeSpec::Any},
                        {"frames", DataType::ImageSet, FormatSpec::Any, ShapeSpec::Any}};
            }
            PortList Outputs() const override {
                return {{"out", DataType::PointCloud, FormatSpec::Any, ShapeSpec::Any}};
            }

            bool IsReconstruct() const override { return true; }
            void RunCPU(RunCtx&) override {}

            bool RunReconstruct(const std::vector<Image>* images,
                                PointCloud* cloud, std::string*) override {
                g_seen.ran        = true;
                g_seen.hadFrames  = images != nullptr;
                g_seen.frameCount = images ? int(images->size()) : 0;
                g_seen.hadCloud   = cloud != nullptr;
                g_seen.cameras    = cloud ? int(cloud->cameras.size()) : 0;
                return true;
            }
        };

        Registry::Get().Add("test_dense",
            []() -> std::unique_ptr<AlgorithmBase> {
                return std::make_unique<FakeDense>();
            });

        // Build a two-frame group and a cloud by hand, then run the stage
        // directly against the pipeline. Driving Pipeline rather than a script
        // keeps this a test of the PLUMBING, not of the parser.
        ImageSet set;
        set.shape = Shape::Of("frame", 2);
        for (int i = 0; i < 2; ++i) {
            ImageDesc d{4, 4, Format::RGBA8};
            Image im;
            im.Alloc(d);
            set.images.push_back(std::move(im));
        }

        PointCloud cloud;
        cloud.cameras.resize(3);

        std::vector<Data> sources;
        sources.push_back(Data{std::move(cloud)});
        sources.push_back(Data{std::move(set)});

        Pipeline p;
        auto dense = Registry::Get().Create("test_dense");
        Check(dense != nullptr, "the dense fixture registers");
        if (dense) {
            p.AddStage(std::move(dense), "test_dense",
                       {{-1, 0}, {-1, 1}}, 1, 1);

            std::string err;
            const bool ok = p.Execute(&sources, nullptr, &err);
            Check(ok, "a two-input reconstruct stage runs" +
                          (ok ? std::string() : ": " + err));

            Check(g_seen.ran, "...and the stage was actually called");
            Check(g_seen.hadCloud && g_seen.cameras == 3,
                  "...the cloud arrives on input 0 (" +
                      std::to_string(g_seen.cameras) + " cameras)");

            // THE POINT OF THE WHOLE CHANGE. Before it, `images` was null
            // whenever input 0 was a PointCloud, so a dense stage had no
            // pixels to read and this assertion is what would catch a
            // regression.
            Check(g_seen.hadFrames && g_seen.frameCount == 2,
                  "...and the FRAMES arrive on input 1 (" +
                      std::to_string(g_seen.frameCount) + " frames)");
        }
    }

    // --- plane_sweep recovers a known depth ---------------------------------
    //
    // THE ONLY HONEST TEST OF A DEPTH ESTIMATOR is one where the depth is
    // known, so this builds a scene whose answer is arithmetic: a textured
    // plane at a fixed distance, photographed by two cameras whose poses are
    // set by hand. The sweep must put the plane back where it was.
    //
    // The images are SYNTHESISED BY PROJECTION rather than drawn: for each
    // pixel of each camera, the ray at the known depth hits a point on the
    // plane, and that point's texture value is written. Both views are then
    // genuinely two photographs of one surface, and a correct sweep has
    // something real to find. Drawing the same bitmap into both would make
    // every depth correlate equally and the test would pass on a broken
    // implementation -- the trap the fixtures above are careful about.
    {
        std::printf("\n--- plane_sweep ---\n");

        const int    W = 160, H = 120;
        const double focal = 150.0;
        const double cx = W * 0.5, cy = H * 0.5;
        const double trueDepth = 5.0;
        const double baseline  = 0.6;

        // A deterministic, high-frequency, NON-REPEATING texture. Repetition
        // is what defeats a sweep, so a test on a repeating pattern would be
        // measuring the wrong thing; this is a smooth but aperiodic function
        // of world position.
        auto Texture = [](double wx, double wy) {
            const double v = std::sin(wx * 2.7 + 0.3) * std::cos(wy * 3.1 - 0.7) +
                             0.5 * std::sin(wx * 7.3 - wy * 5.1 + 1.1) +
                             0.25 * std::cos(wx * 13.7 + wy * 11.3);
            return 0.5 + 0.18 * v;       // comfortably inside 0..1
        };

        // Two cameras looking down +Z, offset along X. World-to-camera is
        // identity rotation and t = -C, so a world point's camera-space Z is
        // its world Z.
        auto MakeCam = [&](double cxw) {
            Camera c;
            c.width = W; c.height = H;
            c.cx = cx;   c.cy = cy;
            c.focal = focal;
            c.R = Mat3::Identity();
            c.t = Vec3{-cxw, 0.0, 0.0};
            c.solved = true;
            return c;
        };

        PointCloud pc;
        pc.cameras.push_back(MakeCam(0.0));
        pc.cameras.push_back(MakeCam(baseline));

        // Render each view by back-projecting every pixel to the plane.
        std::vector<Image> frames;
        for (int f = 0; f < 2; ++f) {
            const Camera& c = pc.cameras[size_t(f)];
            const Vec3 centre = c.Center();

            ImageDesc d{W, H, Format::RGBA8};
            Image im;
            im.Alloc(d);
            ImageView v = im.MapCpuWrite();
            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    // The ray through this pixel, hitting Z = trueDepth.
                    const double dx = (double(x) - c.cx) / c.focal;
                    const double dy = (double(y) - c.cy) / c.focal;
                    const double wx = centre.x + dx * trueDepth;
                    const double wy = centre.y + dy * trueDepth;
                    const uint8_t g = uint8_t(std::clamp(
                        Texture(wx, wy) * 255.0, 0.0, 255.0));
                    uint8_t* p = v.At<uint8_t>(x, y);
                    p[0] = p[1] = p[2] = g;
                    p[3] = 255;
                }
            }
            frames.push_back(std::move(im));
        }

        // Sparse points on the plane, which is where the sweep reads its
        // near/far bounds from. Spread so the 2..98 percentile is meaningful.
        for (int i = 0; i < 64; ++i) {
            Track t;
            t.hasPoint = true;
            const double u = double(i % 8) - 3.5, w = double(i / 8) - 3.5;
            t.point = Vec3{u * 0.2, w * 0.2, trueDepth};
            pc.tracks.push_back(t);
        }

        auto algo = Registry::Get().Create("plane_sweep");
        Check(algo != nullptr, "plane_sweep is registered");

        if (algo) {
            if (ParamBase* p = algo->FindParam("planes"))
                { std::string e; p->SetFromScript(Value(64.0), &e); }
            if (ParamBase* p = algo->FindParam("neighbours"))
                { std::string e; p->SetFromScript(Value(1.0), &e); }
            if (ParamBase* p = algo->FindParam("window"))
                { std::string e; p->SetFromScript(Value(9.0), &e); }

            ImageSet out;
            std::string err;
            const bool ok = algo->RunDense(&frames, pc, &out, &err);
            Check(ok, "plane_sweep runs on a synthetic plane" +
                          (ok ? std::string() : ": " + err));

            // THREE maps per frame, in order: depth, confidence, colour. The
            // colour plane is the reference frame's own pixels, carried so
            // fusion can colour a point from the pixel it was measured from
            // rather than from the nearest sparse point.
            Check(ok && out.images.size() == 6,
                  "...and emits depth, confidence and colour per frame (" +
                      std::to_string(out.images.size()) + ")");

            if (ok && out.images.size() == 6) {
                // The metadata that makes it a DEPTH map rather than a
                // greyscale image, and that the existing viewer normalises by.
                const ImageDesc& dd = out.images[0].Desc();
                Check(dd.isDepth, "the depth map declares itself as depth");
                Check(dd.format == Format::R32F,
                      "...and is R32F, so the existing panels can show it");
                Check(dd.depthNear > 0.0f && dd.depthFar > dd.depthNear,
                      "...and carries the range it spans");
                Check(std::fabs(dd.blackLevel - dd.depthNear) < 1e-6f &&
                          std::fabs(dd.whiteLevel - dd.depthFar) < 1e-6f,
                      "...with display levels matching, so it renders unaided");

                // THE MEASUREMENT. Median over the measured pixels, compared
                // against the depth the scene was built with.
                ImageView dv = out.images[0].MapCpuRead();
                ImageView cv = out.images[1].MapCpuRead();
                std::vector<double> got;
                double maxConf = 0.0;
                if (dv.Valid() && cv.Valid()) {
                    for (int y = 0; y < H; ++y)
                        for (int x = 0; x < W; ++x) {
                            const float z = *dv.At<float>(x, y);
                            const float c = *cv.At<float>(x, y);
                            if (c > 0.0f) maxConf = std::max(maxConf, double(c));
                            if (z > 0.0f) got.push_back(double(z));
                        }
                }

                Check(!got.empty(), "the sweep measured some pixels (" +
                                        std::to_string(got.size()) + ")");

                if (!got.empty()) {
                    std::sort(got.begin(), got.end());
                    const double median = got[got.size() / 2];
                    const double relErr =
                        std::fabs(median - trueDepth) / trueDepth;

                    char msg[200];
                    std::snprintf(msg, sizeof(msg),
                                  "the recovered depth matches the truth "
                                  "(%.3f vs %.3f, %.1f%% off)",
                                  median, trueDepth, relErr * 100.0);
                    Check(relErr < 0.05, msg);

                    // Confidence must be a real signal, not a constant. A map
                    // that is uniformly zero would still let the depth check
                    // above pass, and would be useless for fusion.
                    Check(maxConf > 0.0,
                          "the confidence map carries a non-zero margin");
                }

                // --- fusion, on the same known scene ---------------------
                //
                // The plane is at a known depth, so every fused point must
                // lie ON it. That is a stronger check than the depth map's
                // median: fusion unprojects to WORLD space, so an error in
                // the camera-to-world transform -- a transpose the wrong way
                // round, a translation not subtracted -- shows here and is
                // invisible in a depth map, which never leaves camera space.
                // --- carve_splats, on the same known scene -----------------
                //
                // Three Gaussians on the central ray of camera 0: one ON the
                // plane, one floating halfway to it, one behind it. Both
                // cameras measured the plane, so both look straight through
                // the floater -- it must go. The one on the plane agrees with
                // them and must stay. The one BEHIND is occluded, not seen
                // through: nothing measured says it is empty, so it stays too.
                // A carve that removed it would be treating "hidden" as
                // "absent", which would eat every back surface in a scene.
                std::printf("\n--- carve_splats ---\n");
                if (auto carver = Registry::Get().Create("carve_splats")) {
                    const Camera& c0 = pc.cameras[0];
                    const double u = c0.cx, v = c0.cy;
                    PointCloud sp = pc;
                    auto add = [&](double depth, double red) {
                        Splat s;
                        s.mean = UnprojectForTest(c0, u, v, depth);
                        s.scale = Vec3{0.01, 0.01, 0.01};
                        s.color = Vec3{red, 0.5, 0.5};
                        sp.splats.push_back(s);
                    };
                    add(trueDepth, 0.1);          // on the surface
                    add(trueDepth * 0.5, 0.2);    // floating in front
                    add(trueDepth * 1.5, 0.3);    // hidden behind

                    // The fixture has two cameras, so every Gaussian is in
                    // view of fewer than large_min_views' default: off here,
                    // so this checks the see-through rule alone.
                    PointCloud sizeCase = sp;     // before any carving
                    if (ParamBase* p = carver->FindParam("large_min_views"))
                        { std::string e; p->SetFromScript(Value(0.0), &e); }

                    std::string cerr2;
                    const bool cok = carver->RunReconstruct(&out.images, &sp, &cerr2);
                    Check(cok, "carve_splats runs on the swept maps" +
                                   (cok ? std::string() : ": " + cerr2));
                    bool onPlane = false, floater = false, behind = false;
                    for (const Splat& s : sp.splats) {
                        onPlane |= s.color.x == 0.1;
                        floater |= s.color.x == 0.2;
                        behind  |= s.color.x == 0.3;
                    }
                    Check(cok && !floater,
                          "the Gaussian both cameras see through is carved away");
                    Check(cok && onPlane, "...the one on the surface is kept");
                    Check(cok && behind,
                          "...and the one hidden behind it is kept: occluded is "
                          "not empty");

                    // LARGE AND BARELY SEEN. Same surface Gaussian twice, once
                    // small and once huge, with three views required: the
                    // fixture's two cameras are too few to vouch for the big
                    // one, and size is the only thing that condemns it.
                    {
                        PointCloud lc = sizeCase;
                        lc.splats.resize(1);                 // the surface one
                        lc.splats[0].scale = Vec3{1e-6, 1e-6, 1e-6};   // tiny at any extent
                        Splat big = lc.splats[0];
                        big.scale = Vec3{10.0, 10.0, 10.0};
                        big.color.x = 0.4;
                        lc.splats.push_back(big);
                        if (ParamBase* p = carver->FindParam("large_min_views"))
                            { std::string e; p->SetFromScript(Value(3.0), &e); }
                        std::string lerr;
                        const bool lok = carver->RunReconstruct(&out.images, &lc, &lerr);
                        bool keptSmall = false, keptLarge = false;
                        for (const Splat& s : lc.splats) {
                            keptSmall |= s.color.x == 0.1;
                            keptLarge |= s.color.x == 0.4;
                        }
                        Check(lok && !keptLarge,
                              "a LARGE Gaussian too few cameras see is removed");
                        Check(lok && keptSmall,
                              "...while a small one in the same place is kept");
                    }
                } else {
                    Check(false, "carve_splats is registered");
                }

                std::printf("\n--- fuse_depth ---\n");

                auto fuser = Registry::Get().Create("fuse_depth");
                Check(fuser != nullptr, "fuse_depth is registered");

                if (fuser) {
                    // One agreeing view: the fixture has only two cameras.
                    if (ParamBase* p = fuser->FindParam("min_views"))
                        { std::string e; p->SetFromScript(Value(1.0), &e); }
                    if (ParamBase* p = fuser->FindParam("step"))
                        { std::string e; p->SetFromScript(Value(1.0), &e); }

                    PointCloud fused = pc;      // cameras and sparse points
                    std::string ferr;
                    const bool fok =
                        fuser->RunReconstruct(&out.images, &fused, &ferr);
                    Check(fok, "fuse_depth runs on the swept maps" +
                                   (fok ? std::string() : ": " + ferr));

                    if (fok) {
                        Check(fused.tracks.size() > 1000,
                              "...and produces a DENSE cloud (" +
                                  std::to_string(fused.tracks.size()) +
                                  " points from 64 sparse)");

                        // Every point on the plane Z = trueDepth. Measured as
                        // the median, and separately as the spread, because a
                        // correct median with a huge spread would mean the
                        // points straddle the plane rather than lying on it.
                        std::vector<double> zs;
                        zs.reserve(fused.tracks.size());
                        for (const Track& t : fused.tracks)
                            if (t.hasPoint) zs.push_back(t.point.z);

                        Check(!zs.empty(), "the fused points carry positions");

                        if (!zs.empty()) {
                            std::sort(zs.begin(), zs.end());
                            const double med = zs[zs.size() / 2];
                            const double p05 = zs[size_t(0.05 * double(zs.size()))];
                            const double p95 = zs[size_t(0.95 * double(zs.size()))];
                            const double relErr =
                                std::fabs(med - trueDepth) / trueDepth;
                            const double spread = (p95 - p05) / trueDepth;

                            char m1[220];
                            std::snprintf(m1, sizeof(m1),
                                          "the fused points lie on the known "
                                          "plane (median Z %.3f vs %.3f, "
                                          "%.1f%% off)",
                                          med, trueDepth, relErr * 100.0);
                            Check(relErr < 0.05, m1);

                            char m2[220];
                            std::snprintf(m2, sizeof(m2),
                                          "...and form a PLANE, not a cloud "
                                          "around one (5-95%% spread %.1f%% "
                                          "of the depth)",
                                          spread * 100.0);
                            Check(spread < 0.10, m2);
                        }

                        // COLOUR MUST VARY. The first version gave every
                        // dense point the colour of the nearest sparse point,
                        // which at 1.2M points against 3300 samples rendered
                        // the cloud as a few thousand flat patches. A test
                        // that only checked colour was PRESENT would have
                        // passed on that, so this checks it is not constant:
                        // the fixture's texture is aperiodic, so genuinely
                        // per-pixel colour cannot come out uniform.
                        {
                            double lo = 1e9, hi = -1e9;
                            for (const Track& t : fused.tracks) {
                                const double l = t.color.x;
                                lo = std::min(lo, l);
                                hi = std::max(hi, l);
                            }
                            char m4[200];
                            std::snprintf(m4, sizeof(m4),
                                          "dense colour is sampled per pixel, "
                                          "not per patch (range %.3f)",
                                          hi - lo);
                            Check(hi - lo > 0.05, m4);
                        }

                        // The sparse tracks must be GONE, replaced rather
                        // than appended to: keeping both double-counts every
                        // surface the sparse cloud already had.
                        bool anyMultiObs = false;
                        for (const Track& t : fused.tracks)
                            if (t.obs.size() > 1) anyMultiObs = true;
                        Check(!anyMultiObs,
                              "the sparse tracks were replaced, not appended");

                        // THE CAMERA-TO-WORLD TRANSFORM, which the fixture
                        // above CANNOT test. Both its cameras have identity
                        // rotation, and with R = I the correct
                        // R^T (x - t) and the incorrect R^T x - t are the
                        // same expression -- verified by breaking it and
                        // watching every assertion above still pass.
                        //
                        // So: rotate a camera and unproject its principal
                        // point, where the answer is knowable by hand. At
                        // depth d along the optical axis the world point is
                        // the camera centre plus d times the viewing
                        // direction, whatever the rotation is.
                        {
                            Camera rc;
                            rc.width = W; rc.height = H;
                            rc.cx = cx;   rc.cy = cy;
                            rc.focal = focal;
                            rc.solved = true;

                            // 30 degrees about Y, and a centre away from the
                            // origin so a missing translation cannot hide.
                            const double a = 0.5235987756;   // pi/6
                            rc.R = Mat3::Identity();
                            rc.R.m[0] =  std::cos(a); rc.R.m[2] = std::sin(a);
                            rc.R.m[6] = -std::sin(a); rc.R.m[8] = std::cos(a);
                            const Vec3 centre{1.3, -0.7, 2.1};
                            rc.t = rc.R * centre * -1.0;

                            const double dd = 4.0;
                            const Vec3 got = UnprojectForTest(rc, cx, cy, dd);

                            // The optical axis in world space is R^T (0,0,1),
                            // so the expected point is centre + dd * axis.
                            const Vec3 axis = rc.R.Transpose() * Vec3{0, 0, 1};
                            const Vec3 want{centre.x + dd * axis.x,
                                            centre.y + dd * axis.y,
                                            centre.z + dd * axis.z};
                            const double e =
                                std::sqrt((got.x - want.x) * (got.x - want.x) +
                                          (got.y - want.y) * (got.y - want.y) +
                                          (got.z - want.z) * (got.z - want.z));

                            char m3[240];
                            std::snprintf(m3, sizeof(m3),
                                          "unprojection puts a ROTATED "
                                          "camera's point in the right place "
                                          "(off by %.2e)", e);
                            Check(e < 1e-9, m3);
                        }
                    }
                }
            }
        }
    }

    // --- a SLANTED plane, which is where a sign error hides ------------------
    //
    // The fixture above is FRONTO-PARALLEL: every pixel is at the same depth.
    // That validates the homography's translation term but says nothing about
    // its SLOPE, because a plane with no depth gradient has no slope to get
    // backwards. A wall receding across the frame is the case that does, and
    // it is exactly what Tim reported as "180 degrees off" -- a wall that
    // should recede away instead splaying toward the viewer.
    //
    // So: a plane tilted about Y, with depth genuinely varying left to right,
    // and the test asserts the RECOVERED GRADIENT HAS THE RIGHT SIGN. A
    // mirrored reconstruction gets the depths right on average and the slope
    // inverted, which every check in the fixture above would pass.
    {
        std::printf("\n--- plane_sweep on a slanted wall ---\n");

        const int    W = 160, H = 120;
        const double focal = 150.0;
        const double cx = W * 0.5, cy = H * 0.5;
        const double baseline = 0.6;

        // The plane: z = z0 + slope * x_world, so depth INCREASES to the
        // right. Chosen steep enough that the sign is unambiguous and shallow
        // enough to stay well inside the sweep's range.
        const double z0 = 5.0, slope = 0.45;

        auto Texture = [](double wx, double wy) {
            const double v = std::sin(wx * 2.7 + 0.3) * std::cos(wy * 3.1 - 0.7) +
                             0.5 * std::sin(wx * 7.3 - wy * 5.1 + 1.1) +
                             0.25 * std::cos(wx * 13.7 + wy * 11.3);
            return 0.5 + 0.18 * v;
        };

        auto MakeCam = [&](double cxw) {
            Camera c;
            c.width = W; c.height = H;
            c.cx = cx;   c.cy = cy;
            c.focal = focal;
            c.R = Mat3::Identity();
            c.t = Vec3{-cxw, 0.0, 0.0};
            c.solved = true;
            return c;
        };

        PointCloud pc;
        pc.cameras.push_back(MakeCam(0.0));
        pc.cameras.push_back(MakeCam(baseline));

        // Render by intersecting each pixel's ray with the slanted plane.
        // The ray is (u*t, v*t, t) from the camera centre; substituting into
        // z = z0 + slope*(cx_w + u*t) and solving for t gives the depth.
        std::vector<Image> frames;
        for (int f = 0; f < 2; ++f) {
            const Camera& c = pc.cameras[size_t(f)];
            const Vec3 centre = c.Center();

            ImageDesc d{W, H, Format::RGBA8};
            Image im;
            im.Alloc(d);
            ImageView v = im.MapCpuWrite();
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x) {
                    const double u = (double(x) - c.cx) / c.focal;
                    const double w = (double(y) - c.cy) / c.focal;
                    // t - slope*u*t = z0 + slope*centre.x
                    const double den = 1.0 - slope * u;
                    const double t = (den > 1e-6)
                                         ? (z0 + slope * centre.x) / den : z0;
                    const double wx = centre.x + u * t;
                    const double wy = centre.y + w * t;
                    const uint8_t g = uint8_t(std::clamp(
                        Texture(wx, wy) * 255.0, 0.0, 255.0));
                    uint8_t* p = v.At<uint8_t>(x, y);
                    p[0] = p[1] = p[2] = g;
                    p[3] = 255;
                }
            frames.push_back(std::move(im));
        }

        // Sparse points spread across the plane, so the swept range covers it.
        // Across the WHOLE visible wall: at 0.35 apart they spanned only the
        // middle, the right band lay past the far end of the sweep, and this
        // test passed on depths pinned to the boundary plane -- which the
        // sweep now refuses to report.
        for (int i = 0; i < 64; ++i) {
            Track t;
            t.hasPoint = true;
            const double ux = (double(i % 8) - 3.5) * 0.6;
            const double uy = (double(i / 8) - 3.5) * 0.25;
            t.point = Vec3{ux, uy, z0 + slope * ux};
            pc.tracks.push_back(t);
        }

        auto algo = Registry::Get().Create("plane_sweep");
        if (algo) {
            if (ParamBase* p = algo->FindParam("planes"))
                { std::string e; p->SetFromScript(Value(96.0), &e); }
            if (ParamBase* p = algo->FindParam("neighbours"))
                { std::string e; p->SetFromScript(Value(1.0), &e); }
            if (ParamBase* p = algo->FindParam("window"))
                { std::string e; p->SetFromScript(Value(9.0), &e); }

            ImageSet out;
            std::string err;
            const bool ok = algo->RunDense(&frames, pc, &out, &err);
            Check(ok, "plane_sweep runs on a slanted wall" +
                          (ok ? std::string() : ": " + err));

            if (ok && out.images.size() >= 3) {
                ImageView dv = out.images[0].MapCpuRead();

                // Median depth in a left band and a right band. The plane
                // recedes to the right, so right MUST be further.
                std::vector<double> left, right;
                if (dv.Valid()) {
                    for (int y = 20; y < H - 20; ++y) {
                        for (int x = 20; x < 50; ++x) {
                            const float z = *dv.At<float>(x, y);
                            if (z > 0.0f) left.push_back(double(z));
                        }
                        for (int x = W - 50; x < W - 20; ++x) {
                            const float z = *dv.At<float>(x, y);
                            if (z > 0.0f) right.push_back(double(z));
                        }
                    }
                }
                Check(!left.empty() && !right.empty(),
                      "the slanted wall was measured on both sides");

                if (!left.empty() && !right.empty()) {
                    std::sort(left.begin(), left.end());
                    std::sort(right.begin(), right.end());
                    const double lz = left[left.size() / 2];
                    const double rz = right[right.size() / 2];

                    // Ground truth at the two band centres.
                    const double wantL = z0 + slope * ((35.0 - cx) / focal * z0);
                    const double wantR = z0 + slope * ((W - 35.0 - cx) / focal * z0);

                    char m[260];
                    std::snprintf(m, sizeof(m),
                                  "the slant runs the RIGHT WAY (left %.3f, "
                                  "right %.3f; truth slopes %s)",
                                  lz, rz, wantR > wantL ? "up" : "down");
                    // The sign is the assertion. A mirrored reconstruction
                    // gets the mean depth right and this backwards.
                    Check((rz - lz) * (wantR - wantL) > 0.0, m);

                    // ...and by roughly the right amount, so a nearly-flat
                    // result cannot pass on the sign alone.
                    const double got = rz - lz, want = wantR - wantL;
                    char m2[260];
                    std::snprintf(m2, sizeof(m2),
                                  "...and by the right amount (%.3f vs %.3f)",
                                  got, want);
                    Check(std::fabs(got - want) < 0.35 * std::fabs(want), m2);
                }
            }
        }
    }

    // --- the GPU sweep agrees with the CPU one ------------------------------
    //
    // THE ONLY CHECK THAT MATTERS FOR AN OFFLOAD. A GPU path that is merely
    // plausible is worse than none: it runs by default, produces a slightly
    // different answer, and every measurement taken afterwards is quietly
    // against a different algorithm than the one that was tested.
    //
    // So this runs plane_sweep twice on the same synthetic scene -- once with
    // the device and once with ForceCPU -- and compares the depth maps pixel
    // by pixel. Skipped, loudly, where there is no device.
    //
    // The GPU path is one fused kernel per plane, submitted on its own -- see
    // gpu_sweep.h for why that shape and not the three-pass one it replaced.
    // The float summation order differs from the CPU's sliding box sums, so
    // exact equality is not expected; a fraction of a percent is.
    {
        std::printf("\n--- gpu sweep vs cpu ---\n");

        ID3D12Device* dev = nullptr;
        if (FAILED(TestDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
                                     IID_PPV_ARGS(&dev)))) {
            std::printf("       no D3D12 device; skipped\n");
        } else {
            ComputeContext gpu;
            if (!gpu.Init(dev)) {
                std::printf("       compute init failed; skipped\n");
            } else if (!GpuSweepReady(&gpu)) {
                std::printf("       sweep kernels did not compile; skipped\n");
            } else {
                // The same fronto-parallel fixture the CPU test uses, rebuilt
                // here so the two tests cannot drift apart silently.
                const int    W = 160, H = 120;
                const double focal = 150.0, cx = W * 0.5, cy = H * 0.5;
                const double trueDepth = 5.0, baseline = 0.6;

                auto Texture = [](double wx, double wy) {
                    const double v =
                        std::sin(wx * 2.7 + 0.3) * std::cos(wy * 3.1 - 0.7) +
                        0.5 * std::sin(wx * 7.3 - wy * 5.1 + 1.1) +
                        0.25 * std::cos(wx * 13.7 + wy * 11.3);
                    return 0.5 + 0.18 * v;
                };

                PointCloud pc;
                for (int i = 0; i < 2; ++i) {
                    Camera c;
                    c.width = W; c.height = H;
                    c.cx = cx;   c.cy = cy;
                    c.focal = focal;
                    c.R = Mat3::Identity();
                    c.t = Vec3{-(i == 0 ? 0.0 : baseline), 0.0, 0.0};
                    c.solved = true;
                    pc.cameras.push_back(c);
                }

                std::vector<Image> frames;
                for (int f = 0; f < 2; ++f) {
                    const Camera& c = pc.cameras[size_t(f)];
                    const Vec3 centre = c.Center();
                    ImageDesc d{W, H, Format::RGBA8};
                    Image im;
                    im.Alloc(d);
                    ImageView v = im.MapCpuWrite();
                    for (int y = 0; y < H; ++y)
                        for (int x = 0; x < W; ++x) {
                            const double dx = (double(x) - c.cx) / c.focal;
                            const double dy = (double(y) - c.cy) / c.focal;
                            const uint8_t g = uint8_t(std::clamp(
                                Texture(centre.x + dx * trueDepth,
                                        centre.y + dy * trueDepth) * 255.0,
                                0.0, 255.0));
                            uint8_t* p = v.At<uint8_t>(x, y);
                            p[0] = p[1] = p[2] = g;
                            p[3] = 255;
                        }
                    frames.push_back(std::move(im));
                }
                for (int i = 0; i < 64; ++i) {
                    Track t;
                    t.hasPoint = true;
                    const double u = double(i % 8) - 3.5, w2 = double(i / 8) - 3.5;
                    t.point = Vec3{u * 0.2, w2 * 0.2, trueDepth};
                    pc.tracks.push_back(t);
                }

                // The CPU reference, through the stage as a user would get it.
                auto sweep = [&](ComputeContext* device, ImageSet* out) {
                    auto algo = Registry::Get().Create("plane_sweep");
                    if (!algo) return false;
                    std::string e;
                    if (ParamBase* p = algo->FindParam("planes"))
                        p->SetFromScript(Value(32.0), &e);
                    if (ParamBase* p = algo->FindParam("neighbours"))
                        p->SetFromScript(Value(1.0), &e);
                    algo->SetGroupGpu(device);
                    return algo->RunDense(&frames, pc, out, &e);
                };

                ImageSet cpuOut, gpuOut;
                const bool okC = sweep(nullptr, &cpuOut);
                const bool okG = sweep(&gpu,    &gpuOut);
                Check(okC && okG, "plane_sweep runs on both paths");


                if (okC && okG && cpuOut.images.size() >= 1 &&
                    gpuOut.images.size() >= 1) {
                    ImageView a = cpuOut.images[0].MapCpuRead();
                    ImageView b = gpuOut.images[0].MapCpuRead();

                    int both = 0, onlyOne = 0;
                    double worst = 0.0;
                    if (a.Valid() && b.Valid()) {
                        for (int y = 0; y < H; ++y)
                            for (int x = 0; x < W; ++x) {
                                const float za = *a.At<float>(x, y);
                                const float zb = *b.At<float>(x, y);
                                const bool ma = za > 0.0f, mb = zb > 0.0f;
                                if (ma != mb) { ++onlyOne; continue; }
                                if (!ma) continue;
                                ++both;
                                worst = std::max(worst,
                                    std::fabs(double(za) - double(zb)) / double(za));
                            }
                    }

                    Check(both > 1000,
                          "the two paths measure the same region (" +
                              std::to_string(both) + " shared pixels)");

                    char m[220];
                    std::snprintf(m, sizeof(m),
                                  "the GPU depth matches the CPU (worst %.3f%%"
                                  " over %d pixels, %d disagree on coverage)",
                                  worst * 100.0, both, onlyOne);
                    // Float arithmetic on the device against float on the
                    // host, in a different summation order -- exact equality
                    // is not on offer, but a fraction of a percent is.
                    Check(worst < 0.01, m);

                    // Coverage may differ slightly at the flat-window
                    // threshold, where a correlation of exactly zero falls on
                    // one side or the other. A few pixels is rounding; a
                    // large fraction means the paths disagree about what is
                    // measurable.
                    Check(onlyOne < (W * H) / 50,
                          "...and they agree on WHICH pixels are measurable (" +
                              std::to_string(onlyOne) + " differ)");
                }
            }
            dev->Release();
        }
    }

    // --- unticking a stage that cannot pass through turns its branch off ----
    //
    // init_splats turns a cloud into Gaussians, so it cannot alias its input
    // the way an unticked blur does. It used to ignore the box and run; now it
    // produces nothing and the stage reading it goes off too -- render_splats
    // here has no cameras and would fail if it ran, so success means skipped.
    {
        std::printf("\n--- switched-off branches ---\n");
        PointCloud pc;
        for (int i = 0; i < 50; ++i) {
            Track t;
            t.hasPoint = true;
            t.point = Vec3{double(i % 7), double(i / 7), 5.0};
            pc.tracks.push_back(t);
        }
        std::vector<Data> s;
        s.push_back(Data{std::move(pc)});
        Pipeline p;
        auto init = Registry::Get().Create("init_splats");
        std::string perr;
        init->FindParam("enabled")->SetFromScript(Value(0.0), &perr);
        p.AddStage(std::move(init), "init_splats", {{-1, 0}}, 1, 1);
        p.AddStage(Registry::Get().Create("render_splats"), "render_splats", {{0, 0}}, 1, 2);
        std::string err;
        const bool ok = p.Execute(&s, nullptr, &err);
        Check(ok, "an unticked init_splats does not fail the run" + (ok ? "" : ": " + err));
        Check(ok && p.IsOff({0, 0}), "...it is reported off");
        Check(ok && p.IsOff({1, 0}), "...and so is everything reading it");
        Check(!p.IsOff({-1, 0}), "...but not the palette source");
    }

    // --- init_splats drops isolated points -----------------------------------
    //
    // A dense patch and a handful of strays far from it and from each other:
    // the strays are what survived fusion from a background on a selfie
    // video, and each would train into a blob. Exactly they must go.
    {
        std::printf("\n--- init_splats: isolated points ---\n");
        PointCloud pc;
        for (int i = 0; i < 40; ++i)
            for (int j = 0; j < 40; ++j) {
                Track t;
                t.hasPoint = true;
                t.point = Vec3{i * 0.05, j * 0.05, 5.0};
                pc.tracks.push_back(t);
            }
        const int strays = 6;
        for (int s = 0; s < strays; ++s) {
            Track t;
            t.hasPoint = true;
            t.point = Vec3{-3.0 + s * 1.3, 4.0 - s * 0.9, 9.0 + s};
            pc.tracks.push_back(t);
        }
        auto algo = Registry::Get().Create("init_splats");
        std::string err;
        const bool ok = algo && algo->RunReconstruct(nullptr, &pc, &err);
        Check(ok, "init_splats runs with strays" + (ok ? std::string() : ": " + err));
        Check(ok && pc.splats.size() == 1600,
              "the strays are dropped and the patch kept (" +
                  std::to_string(pc.splats.size()) + " of 1606)");
    }

    // --- init_splats: discs lying in a known surface ------------------------
    //
    // Points scattered on a TILTED plane with a known normal. Every Gaussian
    // init_splats makes must be a flat disc whose thin axis is that normal --
    // the whole point of fitting the surface rather than starting from
    // spheres, as the paper does. Tilted rather than axis-aligned so a
    // rotation that silently came out as the identity cannot pass.
    {
        std::printf("\n--- init_splats ---\n");

        const Vec3 nTrue = Vec3{0.3, -0.5, 0.81}.Normalized();
        // Two in-plane directions.
        const Vec3 u = nTrue.Cross(Vec3{1, 0, 0}).Normalized();
        const Vec3 v = nTrue.Cross(u).Normalized();

        PointCloud pc;
        uint32_t seed = 12345u;
        auto rnd = [&]() {
            seed = seed * 1664525u + 1013904223u;
            return double(seed >> 8) / double(1u << 24);
        };
        const int n = 4000;
        for (int i = 0; i < n; ++i) {
            Track t;
            t.hasPoint = true;
            const double a = rnd() * 4.0 - 2.0, b = rnd() * 4.0 - 2.0;
            t.point = Vec3{1.0, 2.0, 5.0} + u * a + v * b;
            t.color = Vec3{0.2 + 0.5 * rnd(), 0.4, 0.6};
            pc.tracks.push_back(t);
        }

        auto algo = Registry::Get().Create("init_splats");
        Check(algo != nullptr, "init_splats is registered");
        if (algo) {
            std::string err;
            const bool ok = algo->RunReconstruct(nullptr, &pc, &err);
            Check(ok, "init_splats runs" + (ok ? std::string() : ": " + err));
            Check(ok && int(pc.splats.size()) == n,
                  "one Gaussian per point (" + std::to_string(pc.splats.size()) +
                      ")");

            if (ok && !pc.splats.empty()) {
                // The thin axis is column 2 of the rotation, by construction.
                // Compared up to sign: a disc facing either way is the same.
                double worstDeg = 0.0, worstFlat = 0.0, worstQ = 0.0;
                bool colourKept = true;
                // Expected spacing: 4000 points on a 4x4 square.
                const double spacing = 4.0 / std::sqrt(double(n));
                double meanWidth = 0.0;
                for (size_t i = 0; i < pc.splats.size(); ++i) {
                    const Splat& s = pc.splats[i];
                    const Mat3 R = s.Rotation();
                    const Vec3 axis{R.m[2], R.m[5], R.m[8]};
                    const double cosA = std::min(1.0, std::fabs(axis.Dot(nTrue)));
                    worstDeg = std::max(worstDeg,
                                        std::acos(cosA) * 180.0 / 3.14159265358979);
                    worstFlat = std::max(worstFlat, s.scale.z / s.scale.x);
                    const double qn = s.rot[0] * s.rot[0] + s.rot[1] * s.rot[1] +
                                      s.rot[2] * s.rot[2] + s.rot[3] * s.rot[3];
                    worstQ = std::max(worstQ, std::fabs(qn - 1.0));
                    if (std::fabs(s.color.x - pc.tracks[i].color.x) > 1e-12)
                        colourKept = false;
                    meanWidth += s.scale.x;
                }
                meanWidth /= double(pc.splats.size());

                char m1[200];
                std::snprintf(m1, sizeof(m1),
                              "every disc lies in the surface (worst normal "
                              "error %.2f deg)", worstDeg);
                // Random points give a noisy local plane; a few degrees is
                // the fit, tens of degrees is a wrong axis.
                Check(worstDeg < 15.0, m1);

                Check(worstFlat < 0.11,
                      "...and is flat through it (thickness/width " +
                          std::to_string(worstFlat) + ")");
                Check(worstQ < 1e-9, "...with a unit quaternion");
                Check(colourKept, "...and the colour of its point");

                // Width tracks the spacing: `size` 0.6 of the neighbour
                // distance, which for k=8 is a little over the grid spacing.
                char m2[200];
                std::snprintf(m2, sizeof(m2),
                              "discs are sized to the point spacing (mean "
                              "width %.4f, spacing %.4f)", meanWidth, spacing);
                Check(meanWidth > 0.3 * spacing && meanWidth < 2.0 * spacing, m2);
            }
        }

        // Splat::Covariance must agree with the scale and rotation it is
        // built from: along the thin axis the variance is scale.z squared.
        {
            Splat s;
            s.scale = Vec3{2.0, 1.0, 0.25};
            const double h = 0.5 * 0.7;   // 40 degrees about z, as a quaternion
            s.rot[0] = std::cos(h); s.rot[1] = 0; s.rot[2] = 0; s.rot[3] = std::sin(h);
            double c[6];
            s.Covariance(c);
            const Mat3 R = s.Rotation();
            const Vec3 ax0{R.m[0], R.m[3], R.m[6]};
            // v^T Sigma v along the first axis should be scale.x squared.
            const double sxx =
                ax0.x * (c[0] * ax0.x + c[1] * ax0.y + c[2] * ax0.z) +
                ax0.y * (c[1] * ax0.x + c[3] * ax0.y + c[4] * ax0.z) +
                ax0.z * (c[2] * ax0.x + c[4] * ax0.y + c[5] * ax0.z);
            Check(std::fabs(sxx - 4.0) < 1e-9 && std::fabs(c[5] - 0.0625) < 1e-9,
                  "Splat::Covariance matches its scale and rotation (" +
                      std::to_string(sxx) + ", " + std::to_string(c[5]) + ")");
        }
    }

    // --- the splat rasteriser's gradients, against finite differences -------
    //
    // THE TEST THE WHOLE TRAINER RESTS ON. A backward pass that is wrong does
    // not crash and does not stop training from reducing the loss a little;
    // it just optimises something other than the image. The only honest check
    // is to perturb every parameter of every Gaussian and compare the change
    // in loss with what the gradient predicted.
    //
    // Set up to be SMOOTH, because finite differences are meaningless across a
    // discontinuity: minAlpha near zero so nothing is skipped at a threshold,
    // tStop zero so compositing never stops early, and opacities well under
    // maxAlpha so nothing clamps. Four Gaussians, overlapping, rotated and
    // anisotropic, so every term of the chain -- ordering, transmittance,
    // the conic, the Jacobian, the quaternion -- is exercised.
    {
        std::printf("\n--- splat rasteriser gradients ---\n");

        SplatCam cam;
        cam.R = Mat3::Identity();
        cam.t = Vec3{0.1, -0.05, 0.0};
        cam.fx = 40.0; cam.fy = 42.0; cam.cx = 15.5; cam.cy = 16.5;
        cam.w = 32; cam.h = 32;

        RasterOptions opt;
        opt.minAlpha = 1e-12;
        opt.tStop = 0.0;
        opt.background = Vec3{0.1, 0.2, 0.3};

        std::vector<SplatParam> sp(4);
        const double means[4][3] = {{0.0, 0.0, 3.0}, {0.3, -0.2, 3.5},
                                    {-0.25, 0.3, 4.0}, {0.1, 0.25, 2.6}};
        for (int i = 0; i < 4; ++i) {
            SplatParam& p = sp[size_t(i)];
            for (int k = 0; k < 3; ++k) p.mean[k] = means[i][k];
            p.logScale[0] = std::log(0.25 + 0.05 * i);
            p.logScale[1] = std::log(0.15 + 0.03 * i);
            p.logScale[2] = std::log(0.08 + 0.02 * i);
            // Deliberately NOT unit: the forward pass normalises, and the
            // gradient has to go back through that normalisation.
            p.quat[0] = 0.9 + 0.1 * i; p.quat[1] = 0.2 * i - 0.3;
            p.quat[2] = 0.15 * i;      p.quat[3] = 0.4 - 0.1 * i;
            p.opacity = -0.8 + 0.4 * i;   // sigmoid: 0.31 .. 0.60
            p.color[0] = 0.2 + 0.2 * i; p.color[1] = 0.8 - 0.15 * i;
            p.color[2] = 0.5;
        }

        // A fixed pseudo-random target.
        std::vector<double> target(size_t(cam.w * cam.h * 3));
        uint32_t seed = 777u;
        for (double& t : target) {
            seed = seed * 1664525u + 1013904223u;
            t = double(seed >> 8) / double(1u << 24);
        }

        auto loss = [&](const std::vector<SplatParam>& s) {
            SplatRaster r;
            std::vector<double> img;
            r.Forward(s, cam, opt, &img);
            double L = 0.0;
            for (size_t i = 0; i < img.size(); ++i) {
                const double d = img[i] - target[i];
                L += 0.5 * d * d;
            }
            return L;
        };

        SplatRaster r;
        std::vector<double> img;
        r.Forward(sp, cam, opt, &img);
        Check(r.Visible() == 4, "all four Gaussians are in view (" +
                                    std::to_string(r.Visible()) + ")");
        std::vector<double> dImg(img.size());
        for (size_t i = 0; i < img.size(); ++i) dImg[i] = img[i] - target[i];
        std::vector<SplatParam> grad(sp.size(), SplatParam::Zero());
        r.Backward(sp, cam, opt, dImg, &grad);

        // Worst relative error per parameter group, over every Gaussian.
        const char* names[5] = {"mean", "log-scale", "quaternion", "opacity",
                                "colour"};
        const int groupOf[14] = {0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 4, 4, 4};
        double worst[5] = {0, 0, 0, 0, 0};
        double biggest[5] = {0, 0, 0, 0, 0};
        const double eps = 1e-6;
        for (size_t i = 0; i < sp.size(); ++i)
            for (int k = 0; k < SplatParam::kCount; ++k) {
                std::vector<SplatParam> a = sp, b = sp;
                a[i].Data()[k] += eps;
                b[i].Data()[k] -= eps;
                const double fd = (loss(a) - loss(b)) / (2.0 * eps);
                const double an = grad[i].Data()[k];
                const double rel = std::fabs(an - fd) /
                                   std::max(1e-4, std::fabs(an) + std::fabs(fd));
                worst[groupOf[k]] = std::max(worst[groupOf[k]], rel);
                biggest[groupOf[k]] = std::max(biggest[groupOf[k]], std::fabs(fd));
            }
        for (int g = 0; g < 5; ++g) {
            char m[200];
            std::snprintf(m, sizeof(m),
                          "%s gradients match finite differences (worst "
                          "relative error %.1e, largest gradient %.2e)",
                          names[g], worst[g], biggest[g]);
            // A gradient that is identically zero would "match" a zero finite
            // difference, so require that each group actually has something
            // to match.
            Check(worst[g] < 1e-4 && biggest[g] > 1e-6, m);
        }

        // THE DEPTH TERM, alone. A loss on the rendered EXPECTED depth only,
        // so every gradient here comes through depth: through each
        // Gaussian's own depth (its mean) and through the blending weights
        // (everything that shapes alpha). Colour must get none -- depth does
        // not depend on it -- which is checked rather than assumed.
        {
            std::vector<double> dTarget(size_t(cam.w * cam.h));
            std::vector<float> shiftF;   // the target, as Backward takes it
            uint32_t ds = 4242u;
            for (double& t : dTarget) {
                ds = ds * 1664525u + 1013904223u;
                t = 2.5 + 1.5 * double(ds >> 8) / double(1u << 24);
            }
            auto dloss = [&](const std::vector<SplatParam>& s) {
                SplatRaster rr;
                std::vector<double> im, dep;
                rr.Forward(s, cam, opt, &im, &dep);
                double L = 0.0;
                for (size_t i = 0; i < dep.size(); ++i) {
                    // DepthLossGrad's form, squared: D against coverage x t.
                    const double d = dep[i] - (1.0 - rr.FinalT()[i]) * dTarget[i];
                    L += 0.5 * d * d;
                }
                return L;
            };
            SplatRaster rd;
            std::vector<double> im, dep;
            rd.Forward(sp, cam, opt, &im, &dep);
            std::vector<double> dDep(dep.size()), zeroRgb(im.size(), 0.0);
            for (size_t i = 0; i < dep.size(); ++i)
                dDep[i] = dep[i] - (1.0 - rd.FinalT()[i]) * dTarget[i];
            shiftF.assign(dTarget.begin(), dTarget.end());
            std::vector<SplatParam> dg(sp.size(), SplatParam::Zero());
            rd.Backward(sp, cam, opt, zeroRgb, &dg, &dDep, &shiftF);

            double dworst[5] = {0, 0, 0, 0, 0}, dbig[5] = {0, 0, 0, 0, 0};
            double colourMax = 0.0;
            for (size_t i = 0; i < sp.size(); ++i)
                for (int k = 0; k < SplatParam::kCount; ++k) {
                    std::vector<SplatParam> a = sp, b = sp;
                    a[i].Data()[k] += eps;
                    b[i].Data()[k] -= eps;
                    const double fd = (dloss(a) - dloss(b)) / (2.0 * eps);
                    const double an = dg[i].Data()[k];
                    const double rel = std::fabs(an - fd) /
                                       std::max(1e-4, std::fabs(an) + std::fabs(fd));
                    dworst[groupOf[k]] = std::max(dworst[groupOf[k]], rel);
                    dbig[groupOf[k]] = std::max(dbig[groupOf[k]], std::fabs(fd));
                    if (groupOf[k] == 4) colourMax = std::max(colourMax, std::fabs(an));
                }
            for (int g = 0; g < 4; ++g) {
                char m[220];
                std::snprintf(m, sizeof(m),
                              "DEPTH loss: %s gradients match finite differences "
                              "(worst relative error %.1e, largest %.2e)",
                              names[g], dworst[g], dbig[g]);
                Check(dworst[g] < 1e-4 && dbig[g] > 1e-6, m);
            }
            Check(colourMax == 0.0,
                  "DEPTH loss: colour gets no gradient from depth");
        }
    }

    // --- the GPU rasteriser agrees with the CPU one ------------------------
    //
    // The CPU rasteriser is the reference -- its gradients are checked
    // against finite differences above -- so the GPU one is checked against
    // IT: same scene, same camera, and every pixel of the render and every
    // gradient compared.
    //
    // Compared in AGGREGATE per parameter group, not element by element. The
    // GPU works in float and the CPU in double, so a Gaussian sitting on a
    // threshold -- alpha at exactly 1/255, transmittance at the stopping
    // point -- can fall on different sides of it, and one pixel's worth of
    // difference is not a bug. A wrong formula moves the aggregate by far
    // more than rounding does.
    {
        std::printf("\n--- splat rasteriser, GPU against CPU ---\n");
        ID3D12Device* dev = nullptr;
        if (FAILED(TestDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
                                     IID_PPV_ARGS(&dev)))) {
            std::printf("       no D3D12 device; skipped\n");
        } else {
            ComputeContext gpu;
            if (!gpu.Init(dev)) {
                std::printf("       compute init failed; skipped\n");
            } else {
                SplatCam cam;
                cam.R = Mat3::Identity();
                cam.fx = cam.fy = 90.0;
                cam.cx = 47.5; cam.cy = 39.5;
                cam.w = 96; cam.h = 80;   // six by five tiles

                RasterOptions opt;
                opt.background = Vec3{0.2, 0.1, 0.3};

                std::vector<SplatParam> sp;
                uint32_t seed = 99u;
                auto rnd = [&]() {
                    seed = seed * 1664525u + 1013904223u;
                    return double(seed >> 8) / double(1u << 24);
                };
                for (int i = 0; i < 400; ++i) {
                    Splat s;
                    s.mean = Vec3{rnd() * 2.2 - 1.1, rnd() * 1.8 - 0.9, 2.5 + rnd() * 2.0};
                    s.scale = Vec3{0.02 + 0.08 * rnd(), 0.02 + 0.08 * rnd(),
                                   0.01 + 0.03 * rnd()};
                    s.rot[0] = rnd() + 0.2; s.rot[1] = rnd() - 0.5;
                    s.rot[2] = rnd() - 0.5; s.rot[3] = rnd() - 0.5;
                    s.opacity = 0.2 + 0.7 * rnd();
                    s.color = Vec3{rnd(), rnd(), rnd()};
                    sp.push_back(ToParam(s));
                }
                std::vector<double> target(size_t(cam.w * cam.h * 3));
                for (double& t : target) t = rnd();

                // The loss has a DEPTH term as well as colour, so the device's
                // depth render and depth gradients are checked by the same
                // comparisons as everything else.
                std::vector<double> dTarget(size_t(cam.w * cam.h));
                for (double& t : dTarget) t = 2.5 + 2.0 * rnd();
                std::vector<double> depC, depG;
                auto run = [&](ComputeContext* device, std::vector<double>* img,
                               std::vector<SplatParam>* grad, bool* usedGpu,
                               std::vector<double>* dep) {
                    SplatRaster r;
                    r.SetGpu(device);
                    r.Forward(sp, cam, opt, img, dep);
                    *usedGpu = r.UsedGpu();
                    std::vector<double> d(img->size());
                    for (size_t i = 0; i < img->size(); ++i) d[i] = (*img)[i] - target[i];
                    std::vector<double> dd(dep->size());
                    for (size_t i = 0; i < dep->size(); ++i)
                        dd[i] = 0.1 * ((*dep)[i] - (1.0 - r.FinalT()[i]) * dTarget[i]);
                    std::vector<float> shiftF(dTarget.begin(), dTarget.end());
                    grad->assign(sp.size(), SplatParam::Zero());
                    r.Backward(sp, cam, opt, d, grad, &dd, &shiftF);
                    if (device && !r.UsedGpu())
                        std::printf("       GPU note: %s\n", r.GpuNote().c_str());
                };

                // Twice: without the colour gate, and with one that the
                // random depths straddle -- so some entries are gated and
                // some are not, on both paths.
                std::vector<SplatParam> ungated;
                for (double gate : {0.0, 0.15}) {
                opt.colourGate = gate;
                if (gate > 0.0) std::printf("    with colour_gate %.2f:\n", gate);
                std::vector<double> imgC, imgG;
                std::vector<SplatParam> gradC, gradG;
                bool usedC = false, usedG = false;
                run(nullptr, &imgC, &gradC, &usedC, &depC);
                run(&gpu, &imgG, &gradG, &usedG, &depG);
                if (gate == 0.0) {
                    ungated = gradC;
                } else {
                    // The gate removes colour gradient and touches nothing
                    // else: shape and opacity still learn from every pixel.
                    double colOff = 0.0, colOn = 0.0, otherDiff = 0.0;
                    for (size_t i = 0; i < sp.size(); ++i)
                        for (int q = 0; q < 14; ++q) {
                            const double a = ungated[i].Data()[q], b = gradC[i].Data()[q];
                            if (q >= 11) { colOff += std::fabs(a); colOn += std::fabs(b); }
                            else otherDiff += std::fabs(a - b);
                        }
                    Check(colOn < colOff * 0.95 && colOn > 0.0 && otherDiff == 0.0,
                          "the colour gate removes colour gradient (" +
                              std::to_string(int(100.0 * colOn / std::max(1e-30, colOff))) +
                              "% left) and nothing else");
                }
                Check(usedG, "the GPU path actually ran");
                if (usedG) {
                    double worstD = 0.0;
                    for (size_t i = 0; i < depC.size(); ++i)
                        worstD = std::max(worstD, std::fabs(depC[i] - depG[i]));
                    char m[200];
                    std::snprintf(m, sizeof(m),
                                  "the GPU depth render matches the CPU (worst "
                                  "pixel difference %.1e)", worstD);
                    Check(worstD < 1e-3, m);
                }

                if (usedG) {
                    double worst = 0.0;
                    for (size_t i = 0; i < imgC.size(); ++i)
                        worst = std::max(worst, std::fabs(imgC[i] - imgG[i]));
                    char m[200];
                    std::snprintf(m, sizeof(m),
                                  "the GPU render matches the CPU (worst pixel "
                                  "difference %.1e)", worst);
                    Check(worst < 1e-3, m);

                    const char* names[5] = {"mean", "log-scale", "quaternion",
                                            "opacity", "colour"};
                    const int groupOf[14] = {0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 4, 4, 4};
                    double diff[5] = {0, 0, 0, 0, 0}, mag[5] = {0, 0, 0, 0, 0};
                    for (size_t i = 0; i < sp.size(); ++i)
                        for (int q = 0; q < 14; ++q) {
                            const double a = gradC[i].Data()[q], b = gradG[i].Data()[q];
                            diff[groupOf[q]] += std::fabs(a - b);
                            mag[groupOf[q]] += std::fabs(a);
                        }
                    for (int gi = 0; gi < 5; ++gi) {
                        const double rel = diff[gi] / std::max(1e-12, mag[gi]);
                        char m2[200];
                        std::snprintf(m2, sizeof(m2),
                                      "%s gradients match the CPU (aggregate "
                                      "relative difference %.1e)",
                                      names[gi], rel);
                        Check(rel < 1e-3 && mag[gi] > 0.0, m2);
                    }
                }
                }   // gate
            }
            dev->Release();
        }
    }

    // --- densification on the device ------------------------------------------
    //
    // The GPU trainer rebuilds its own state from a plan made on the CPU. A
    // random state with colour and reflections goes up, a plan with every
    // kind of entry -- kept, copied, split, and sources left out -- is
    // applied, and what comes back must be the plan applied on the CPU:
    // moments kept only by kept ones, counters cleared, children moved by
    // R * (z * scale) and shrunk by 1.6. The summary densification decides
    // from, and the opacity reset, are checked on the same state.
    {
        std::printf("\n--- densification on the GPU ---\n");
        ID3D12Device* dev = nullptr;
        if (FAILED(TestDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&dev)))) {
            std::printf("       no D3D12 device; not checked\n");
        } else {
            ComputeContext gpu;
            if (gpu.Init(dev)) {
                std::mt19937 rng(11);
                std::uniform_real_distribution<double> u(-1.0, 1.0);
                const size_t n = 3000;   // more than one row of the 2048-wide textures
                std::vector<SplatParam> params(n);
                std::vector<double> m1(n * 14), m2(n * 14), acc(n), mx(n);
                std::vector<int> cnt(n);
                std::vector<double> sh(n * kShRest), sm1(n * kShRest), sm2(n * kShRest);
                std::vector<ReflParam> refl(n);
                std::vector<double> rm1(n * 4), rm2(n * 4);
                for (size_t i = 0; i < n; ++i) {
                    double* p = params[i].Data();
                    for (int q = 0; q < 14; ++q) p[q] = u(rng);
                    for (int q = 0; q < 14; ++q) {
                        m1[i * 14 + size_t(q)] = u(rng);
                        m2[i * 14 + size_t(q)] = u(rng) + 1.0;
                    }
                    acc[i] = u(rng) + 1.0;
                    cnt[i] = int(i % 5);
                    mx[i] = 10.0 * (u(rng) + 1.0);
                    for (int q = 0; q < kShRest; ++q) {
                        sh[i * kShRest + size_t(q)] = u(rng);
                        sm1[i * kShRest + size_t(q)] = u(rng);
                        sm2[i * kShRest + size_t(q)] = u(rng) + 1.0;
                    }
                    refl[i].reflLogit = u(rng);
                    for (double& c : refl[i].normal) c = u(rng);
                    for (int q = 0; q < 4; ++q) {
                        rm1[i * 4 + size_t(q)] = u(rng);
                        rm2[i * 4 + size_t(q)] = u(rng) + 1.0;
                    }
                }
                SplatTrainerGpu t(&gpu);
                std::string e;
                bool ok = t.Upload(params, m1, m2, acc, cnt, mx, &e) &&
                          t.UploadSh(sh, sm1, sm2, &e) && t.UploadRefl(refl, rm1, rm2, &e);

                // The summary.
                std::vector<DensifyIn> din;
                ok = ok && t.DensifyStats(&din, &e);
                double worst = 0.0;
                bool noneRight = true;
                if (ok) {
                    for (size_t i = 0; i < n; ++i) {
                        const SplatParam& p = params[i];
                        const double want[4] = {
                            p.opacity, std::max({p.logScale[0], p.logScale[1], p.logScale[2]}),
                            cnt[i] > 0 ? acc[i] / cnt[i] : -1.0, mx[i]};
                        const double got[4] = {din[i].opacityLogit, din[i].maxLogScale,
                                               din[i].gradAvg, din[i].maxScreen};
                        for (int k = 0; k < 4; ++k)
                            worst = std::max(worst, std::fabs(want[k] - got[k]) /
                                                        std::max(1.0, std::fabs(want[k])));
                        if (cnt[i] == 0 && din[i].gradAvg != -1.0) noneRight = false;
                    }
                }
                char m[200];
                std::snprintf(m, sizeof m, "the device's densification summary matches "
                                           "its state (worst %.1e)", worst);
                Check(ok && worst < 1e-6 && noneRight, ok ? std::string(m) : "summary: " + e);

                // The plan: every third source dropped, the rest kept, with
                // copies and split pairs mixed in.
                std::vector<DensifyPlan> plan;
                for (size_t i = 0; i < n; ++i) {
                    if (i % 3 == 1) continue;
                    if (i % 7 == 0) {
                        for (int c = 0; c < 2; ++c) {
                            DensifyPlan d{uint32_t(i), DensifyPlan::kSplit, {0, 0, 0}};
                            for (float& z : d.z) z = float(u(rng));
                            plan.push_back(d);
                        }
                        continue;
                    }
                    plan.push_back({uint32_t(i), DensifyPlan::kKeep, {0, 0, 0}});
                    if (i % 5 == 0) plan.push_back({uint32_t(i), DensifyPlan::kCopy, {0, 0, 0}});
                }
                ok = ok && t.ApplyPlan(plan, &e);
                std::vector<SplatParam> gp;
                std::vector<double> g1, g2, gacc, gmx, gsh, gs1, gs2, gr1, gr2;
                std::vector<int> gcnt;
                std::vector<ReflParam> grefl;
                ok = ok && t.Download(&gp, &g1, &g2, &gacc, &gcnt, &gmx, &e) &&
                     t.DownloadSh(&gsh, &gs1, &gs2, &e) && t.DownloadRefl(&grefl, &gr1, &gr2, &e);
                ok = ok && gp.size() == plan.size() && gsh.size() == plan.size() * kShRest &&
                     grefl.size() == plan.size();
                worst = 0.0;
                auto cmp = [&](double want, double got) {
                    worst = std::max(worst, std::fabs(want - got) / std::max(1.0, std::fabs(want)));
                };
                if (ok) {
                    for (size_t j = 0; j < plan.size(); ++j) {
                        const size_t i = plan[j].src;
                        const bool kept = plan[j].kind == DensifyPlan::kKeep;
                        SplatParam want = params[i];
                        if (plan[j].kind == DensifyPlan::kSplit) {
                            const Mat3 R = FromParam(params[i]).Rotation();
                            const double z[3] = {plan[j].z[0] * std::exp(want.logScale[0]),
                                                 plan[j].z[1] * std::exp(want.logScale[1]),
                                                 plan[j].z[2] * std::exp(want.logScale[2])};
                            for (int a = 0; a < 3; ++a)
                                want.mean[a] += R.m[a * 3] * z[0] + R.m[a * 3 + 1] * z[1] +
                                                R.m[a * 3 + 2] * z[2];
                            for (double& s : want.logScale) s -= std::log(1.6);
                        }
                        for (int q = 0; q < 14; ++q) {
                            cmp(want.Data()[q], gp[j].Data()[q]);
                            cmp(kept ? m1[i * 14 + size_t(q)] : 0.0, g1[j * 14 + size_t(q)]);
                            cmp(kept ? m2[i * 14 + size_t(q)] : 0.0, g2[j * 14 + size_t(q)]);
                        }
                        cmp(0.0, gacc[j]);
                        cmp(0.0, double(gcnt[j]));
                        cmp(0.0, gmx[j]);
                        for (int q = 0; q < kShRest; ++q) {
                            const size_t a = i * kShRest + size_t(q), b = j * kShRest + size_t(q);
                            cmp(sh[a], gsh[b]);
                            cmp(kept ? sm1[a] : 0.0, gs1[b]);
                            cmp(kept ? sm2[a] : 0.0, gs2[b]);
                        }
                        cmp(refl[i].reflLogit, grefl[j].reflLogit);
                        for (int k = 0; k < 3; ++k) cmp(refl[i].normal[k], grefl[j].normal[k]);
                        for (int q = 0; q < 4; ++q) {
                            cmp(kept ? rm1[i * 4 + size_t(q)] : 0.0, gr1[j * 4 + size_t(q)]);
                            cmp(kept ? rm2[i * 4 + size_t(q)] : 0.0, gr2[j * 4 + size_t(q)]);
                        }
                    }
                }
                std::snprintf(m, sizeof m,
                              "the device applies a densification plan as the CPU does "
                              "(%zu -> %zu, worst %.1e)", n, plan.size(), worst);
                Check(ok && worst < 1e-5, ok ? std::string(m) : "plan: " + e);

                // The opacity reset, on the rebuilt state.
                const double cap = std::log(0.01 / 0.99);
                std::vector<SplatParam> rp;
                std::vector<double> r1, r2, racc, rmx;
                std::vector<int> rcnt;
                ok = ok && t.ResetOpacity(cap, &e) &&
                     t.Download(&rp, &r1, &r2, &racc, &rcnt, &rmx, &e);
                worst = 0.0;
                if (ok) {
                    for (size_t j = 0; j < gp.size(); ++j)
                        for (int q = 0; q < 14; ++q) {
                            const bool op = q == 10;
                            cmp(op ? std::min(gp[j].opacity, cap) : gp[j].Data()[q],
                                 rp[j].Data()[q]);
                            cmp(op ? 0.0 : g1[j * 14 + size_t(q)], r1[j * 14 + size_t(q)]);
                            cmp(op ? 0.0 : g2[j * 14 + size_t(q)], r2[j * 14 + size_t(q)]);
                        }
                }
                std::snprintf(m, sizeof m, "...and resets opacity as the CPU does (worst %.1e)",
                              worst);
                Check(ok && worst < 1e-6, ok ? std::string(m) : "reset: " + e);
            }
            dev->Release();
        }
    }

    // --- deferred reflection shading on the GPU -----------------------------
    //
    // The GPU trainer shades and differentiates the reflection on the device;
    // ShadeDeferred and ShadeDeferredBackward (checked above against finite
    // differences) are the reference. Random maps -- some pixels with no
    // reflectivity or no normal, which take the other branch -- a turned
    // camera, so the ray's rotation is exercised, and a random environment.
    // The environment's gradient is summed in fixed point on the device.
    {
        std::printf("\n--- deferred shading on the GPU ---\n");
        ID3D12Device* dev = nullptr;
        if (FAILED(TestDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&dev)))) {
            std::printf("       no D3D12 device; not checked\n");
        } else {
            ComputeContext gpu;
            if (gpu.Init(dev)) {
                std::mt19937 rng(5);
                std::uniform_real_distribution<double> u(-1.0, 1.0);
                const int w = 40, h = 30;
                SplatCam cam;
                const double a = 0.4, b = 0.25;   // yaw, then pitch
                const double Ry[9] = {std::cos(a), 0, std::sin(a), 0, 1, 0, -std::sin(a), 0, std::cos(a)};
                const double Rx[9] = {1, 0, 0, 0, std::cos(b), -std::sin(b), 0, std::sin(b), std::cos(b)};
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j) {
                        double v = 0;
                        for (int k = 0; k < 3; ++k) v += Rx[i * 3 + k] * Ry[k * 3 + j];
                        cam.R.m[i * 3 + j] = v;
                    }
                cam.t = Vec3{0, 0, 0};
                cam.fx = cam.fy = 30.0; cam.cx = 19.5; cam.cy = 14.5; cam.w = w; cam.h = h;
                EnvMap env;
                env.Init(8, 0.0);
                for (double& t : env.texels) t = 0.5 + 0.5 * u(rng);
                const size_t np = size_t(w) * size_t(h) * 3;
                std::vector<double> Cd(np), Rm(np), Nm(np), dOut(np);
                for (size_t p = 0; p < np; p += 3) {
                    const size_t pix = p / 3;
                    const double r = (pix % 7 == 0) ? 0.0 : 0.5 + 0.5 * u(rng);
                    for (int ch = 0; ch < 3; ++ch) {
                        Cd[p + size_t(ch)] = 0.5 + 0.5 * u(rng);
                        Rm[p + size_t(ch)] = r;
                        Nm[p + size_t(ch)] = (pix % 11 == 0) ? 0.0 : u(rng);
                        dOut[p + size_t(ch)] = u(rng) / double(np);
                    }
                }
                const double sparsity = 0.3 / double(w * h);
                std::vector<double> sC, dCdC, dRmC, dNmC, egC(env.texels.size(), 0.0);
                ShadeDeferred(Cd, Rm, Nm, cam, env, &sC);
                ShadeDeferredBackward(Cd, Rm, Nm, cam, env, dOut, &dCdC, &dRmC, &dNmC, &egC);
                for (size_t p = 0; p < np; p += 3) dRmC[p] += sparsity;

                // The trainer builds its kernels on Upload: a token state.
                SplatTrainerGpu t(&gpu);
                std::string e;
                std::vector<SplatParam> one(1);
                std::vector<double> z14(14, 0.0), z1(1, 0.0);
                std::vector<int> zi(1, 0);
                std::vector<double> sG, dCdG, dRmG, dNmG, egG(env.texels.size(), 0.0);
                const bool ok = t.Upload(one, z14, z14, z1, zi, z1, &e) &&
                                t.ShadeCheck(cam, env, Cd, Rm, Nm, dOut, sparsity, &sG, &dCdG,
                                             &dRmG, &dNmG, &egG, &e);
                auto worstAbs = [&](const std::vector<double>& A, const std::vector<double>& B,
                                    int stride) {
                    double m = 0.0;
                    for (size_t i = 0; i < A.size(); i += size_t(stride))
                        m = std::max(m, std::fabs(A[i] - B[i]));
                    return m;
                };
                // Aggregate relative: summed difference over summed size. A
                // reflection landing on a cube seam can pick the other face
                // in float, which is a kink the CPU's double lands the other
                // side of -- real, rare, and swamped here by everything else.
                auto aggRel = [&](const std::vector<double>& A, const std::vector<double>& B) {
                    double d = 0.0, s = 0.0;
                    for (size_t i = 0; i < A.size(); ++i) { d += std::fabs(A[i] - B[i]); s += std::fabs(A[i]); }
                    return s > 0.0 ? d / s : 0.0;
                };
                char m[240];
                if (!ok) {
                    Check(false, "the GPU shading runs: " + e);
                } else {
                    const double ws = worstAbs(sC, sG, 1), wc = worstAbs(dCdC, dCdG, 1);
                    std::snprintf(m, sizeof m, "the GPU shades as the CPU does (worst %.1e), and "
                                  "passes the colour gradient back (worst %.1e)", ws, wc * double(np));
                    Check(ws < 1e-4 && wc * double(np) < 1e-4, m);
                    const double rR = aggRel(dRmC, dRmG), rN = aggRel(dNmC, dNmG),
                                 rE = aggRel(egC, egG);
                    std::snprintf(m, sizeof m, "...the reflectivity, normal and environment "
                                  "gradients (aggregate relative %.1e, %.1e, %.1e)", rR, rN, rE);
                    Check(rR < 1e-4 && rN < 1e-3 && rE < 1e-3, m);
                }
            }
            dev->Release();
        }
    }

    // --- train_splats recovers a known scene -------------------------------
    //
    // Gradients that match finite differences say the derivative is right;
    // they do not say the loop around it -- loss, Adam, learning rates, the
    // camera cycling -- actually fits anything. So: photograph a known splat
    // scene from three cameras (rendered by the rasteriser itself, so the
    // target is exactly representable), spoil a copy of the splats, and train
    // the copy back. The fit must improve by a wide margin.
    {
        std::printf("\n--- train_splats ---\n");

        const int W = 64, H = 48;
        PointCloud truth;
        for (int i = 0; i < 3; ++i) {
            Camera c;
            c.width = W; c.height = H;
            c.focal = 60.0; c.cx = W * 0.5 - 0.5; c.cy = H * 0.5 - 0.5;
            // Turned a little about Y and spaced along X.
            const double a = (double(i) - 1.0) * 0.12;
            c.R = Mat3::Identity();
            c.R.m[0] = std::cos(a); c.R.m[2] = -std::sin(a);
            c.R.m[6] = std::sin(a); c.R.m[8] = std::cos(a);
            const Vec3 centre{(double(i) - 1.0) * 0.4, 0.0, 0.0};
            c.t = c.R * centre * -1.0;
            c.solved = true;
            truth.cameras.push_back(c);
        }
        uint32_t seed = 4242u;
        auto rnd = [&]() {
            seed = seed * 1664525u + 1013904223u;
            return double(seed >> 8) / double(1u << 24);
        };
        for (int i = 0; i < 60; ++i) {
            Splat s;
            s.mean = Vec3{rnd() * 2.4 - 1.2, rnd() * 1.6 - 0.8, 3.0 + rnd() * 1.5};
            s.scale = Vec3{0.08 + 0.1 * rnd(), 0.08 + 0.1 * rnd(), 0.05};
            const double h = rnd() * 3.0;
            s.rot[0] = std::cos(h); s.rot[1] = 0.3 * std::sin(h);
            s.rot[2] = 0.0;         s.rot[3] = std::sin(h);
            const double qn = std::sqrt(s.rot[0] * s.rot[0] + s.rot[1] * s.rot[1] +
                                        s.rot[3] * s.rot[3]);
            for (double& q : s.rot) q /= qn;
            s.opacity = 0.5 + 0.4 * rnd();
            s.color = Vec3{rnd(), rnd(), rnd()};
            truth.splats.push_back(s);
        }

        // The photographs: the true splats, rendered.
        std::vector<Image> frames;
        {
            std::vector<SplatParam> tp;
            for (const Splat& s : truth.splats) tp.push_back(ToParam(s));
            RasterOptions opt;
            for (const Camera& c : truth.cameras) {
                SplatRaster r;
                std::vector<double> rgb;
                r.Forward(tp, SplatCamFrom(c, W, H), opt, &rgb);
                Image im;
                im.Alloc(ImageDesc{W, H, Format::RGBA32F});
                ImageView v = im.MapCpuWrite();
                for (int y = 0; y < H; ++y)
                    for (int x = 0; x < W; ++x) {
                        float* p = v.At<float>(x, y);
                        for (int ch = 0; ch < 3; ++ch)
                            p[ch] = float(rgb[(size_t(y) * W + size_t(x)) * 3 + ch]);
                        p[3] = 1.0f;
                    }
                frames.push_back(std::move(im));
            }
        }

        // The spoiled start: grey, half as opaque, and nudged.
        PointCloud start = truth;
        for (Splat& s : start.splats) {
            s.color = Vec3{0.5, 0.5, 0.5};
            s.opacity *= 0.5;
            s.mean = s.mean + Vec3{rnd() * 0.06 - 0.03, rnd() * 0.06 - 0.03, 0.0};
        }

        auto algo = Registry::Get().Create("train_splats");
        Check(algo != nullptr, "train_splats is registered");
        if (algo) {
            std::string e;
            if (ParamBase* p = algo->FindParam("iterations"))
                p->SetFromScript(Value(300.0), &e);
            if (ParamBase* p = algo->FindParam("downscale"))
                p->SetFromScript(Value(1.0), &e);
            // OFF HERE, because this tests the optimiser, and the scene is
            // exactly representable by the Gaussians it starts with -- the
            // case densification exists NOT to touch. Left on, it split 60
            // Gaussians into 470 and the fit came out 5 dB worse (40.8 against
            // 35.4): every gradient is large early on, while the colours are
            // still grey, so everything looked like it needed dividing.
            if (ParamBase* p = algo->FindParam("densify"))
                p->SetFromScript(Value(0.0), &e);
            // CANCELLATION: a run superseded part way must stop within an
            // iteration, not finish. 100000 iterations would take minutes
            // here; cancelled 100 ms in, it has to be back well inside a
            // second, saying so.
            {
                auto longRun = Registry::Get().Create("train_splats");
                std::string e2;
                longRun->FindParam("iterations")->SetFromScript(Value(100000.0), &e2);
                longRun->FindParam("densify")->SetFromScript(Value(0.0), &e2);
                CancelToken token;
                longRun->SetGroupCancel(&token);
                std::thread canceller([&] {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    token.Cancel();
                });
                PointCloud c = start;
                std::string cerr;
                const auto t0 = std::chrono::steady_clock::now();
                const bool ran = longRun->RunReconstruct(&frames, &c, &cerr);
                const double ms = std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - t0).count();
                canceller.join();
                char m[200];
                std::snprintf(m, sizeof m, "train_splats stops when its run is cancelled "
                              "(back in %.0f ms: \"%s\")", ms, cerr.c_str());
                Check(!ran && cerr == "cancelled" && ms < 1000.0, m);
            }

            PointCloud trained = start;
            std::string err;
            const bool ok = algo->RunReconstruct(&frames, &trained, &err);
            Check(ok, "train_splats runs" + (ok ? std::string() : ": " + err));
            if (ok) {
                const std::string note = algo->RunReport();
                std::printf("       %s\n", note.c_str());
                double before = 0, after = 0;
                const size_t at = note.find("PSNR ");
                if (at != std::string::npos)
                    std::sscanf(note.c_str() + at, "PSNR %lf -> %lf", &before, &after);
                char m[200];
                std::snprintf(m, sizeof(m),
                              "training fits the photographs far better "
                              "(PSNR %.2f -> %.2f dB)", before, after);
                Check(after > before + 8.0, m);
            }
        }

        // --- deferred reflection: the shading's gradients -------------------
        //
        // C = (1 - R) Cd + R Env(reflect(ray, N / |N|)), per pixel. Every
        // gradient the backward pass returns -- into Cd, R, the unnormalised
        // N and the environment's texels -- checked against a central
        // difference of the forward. The environment is SMOOTH (a low-order
        // function of direction), since the direction gradient is itself a
        // difference across half a texel and a sharp map would measure the
        // texel grid rather than the maths.
        {
            const int w = 6, h = 5;
            SplatCam cam;
            cam.R = Mat3::Identity();
            cam.t = Vec3{0, 0, 0};
            cam.fx = cam.fy = 5.0; cam.cx = 2.5; cam.cy = 2.0; cam.w = w; cam.h = h;
            EnvMap env;
            env.Init(32, 0.0);
            for (int f = 0; f < 6; ++f)
                for (int y = 0; y < 32; ++y)
                    for (int x = 0; x < 32; ++x)
                        for (int ch = 0; ch < 3; ++ch)
                            env.texels[((size_t(f) * 32 + size_t(y)) * 32 + size_t(x)) * 3 + size_t(ch)] =
                                0.5 + 0.3 * std::sin(0.1 * x + 0.7 * f + ch) * std::cos(0.08 * y - 0.3 * ch);
            const size_t np = size_t(w) * size_t(h) * 3;
            std::vector<double> Cd(np), Rm(np), Nm(np), dOut(np);
            for (size_t p = 0; p < np; p += 3) {
                const double r = 0.2 + 0.6 * rnd();
                Rm[p] = Rm[p + 1] = Rm[p + 2] = r;
                for (int ch = 0; ch < 3; ++ch) {
                    Cd[p + size_t(ch)] = rnd();
                    dOut[p + size_t(ch)] = rnd() - 0.5;
                }
                // Facing back toward the camera, as a real normal does.
                Nm[p] = rnd() - 0.5; Nm[p + 1] = rnd() - 0.5; Nm[p + 2] = -0.8 - 0.4 * rnd();
            }
            auto loss = [&]() {
                std::vector<double> out;
                ShadeDeferred(Cd, Rm, Nm, cam, env, &out);
                double s = 0;
                for (size_t i = 0; i < np; ++i) s += out[i] * dOut[i];
                return s;
            };
            std::vector<double> dCd, dRm, dNm, dEnv(env.texels.size(), 0.0);
            ShadeDeferredBackward(Cd, Rm, Nm, cam, env, dOut, &dCd, &dRm, &dNm, &dEnv);
            auto fd = [&](double* x, double hstep) {
                const double keep = *x;
                *x = keep + hstep; const double a = loss();
                *x = keep - hstep; const double b = loss();
                *x = keep;
                return (a - b) / (2.0 * hstep);
            };
            double worstC = 0, worstR = 0, worstN = 0, scaleN = 0, worstE = 0;
            double sumErrN = 0, sumN = 0;
            int envChecked = 0;
            for (size_t p = 0; p < np; p += 3) {
                for (int ch = 0; ch < 3; ++ch)
                    worstC = std::max(worstC, std::fabs(fd(&Cd[p + size_t(ch)], 1e-6) - dCd[p + size_t(ch)]));
                // R is read from channel 0; perturb all three together.
                const double keep = Rm[p];
                auto setR = [&](double v) { Rm[p] = Rm[p + 1] = Rm[p + 2] = v; };
                setR(keep + 1e-6); const double a = loss();
                setR(keep - 1e-6); const double b = loss();
                setR(keep);
                worstR = std::max(worstR, std::fabs((a - b) / 2e-6 - dRm[p]));
                for (int ch = 0; ch < 3; ++ch) {
                    const double num = fd(&Nm[p + size_t(ch)], 1e-3);
                    // Smooth here? A difference straddling a cube-face seam
                    // reads the jump, and halving the step changes it.
                    const double half = fd(&Nm[p + size_t(ch)], 5e-4);
                    if (std::fabs(num - half) > 0.05 * std::max(0.01, std::fabs(num))) continue;
                    worstN = std::max(worstN, std::fabs(num - dNm[p + size_t(ch)]));
                    scaleN = std::max(scaleN, std::fabs(num));
                    sumErrN += std::fabs(num - dNm[p + size_t(ch)]);
                    sumN += std::fabs(num);
                }
            }
            // The texels a reflection actually read -- most of the map is
            // never touched by 30 pixels, and checking those proves nothing.
            for (size_t t = 0; t < env.texels.size(); ++t) {
                if (dEnv[t] == 0.0) continue;
                worstE = std::max(worstE, std::fabs(fd(&env.texels[t], 1e-6) - dEnv[t]));
                ++envChecked;
            }
            char m[240];
            std::snprintf(m, sizeof m,
                          "deferred shading's gradients match finite differences "
                          "(colour %.1e, reflectivity %.1e, environment %.1e over %d texels)",
                          worstC, worstR, worstE, envChecked);
            Check(worstC < 1e-6 && worstR < 1e-6 && worstE < 1e-6 && envChecked > 50, m);
            // The normal's goes through the lookup's change with direction,
            // which is exact within a face but has kinks at texel and face
            // edges, so it is compared in AGGREGATE, over the pixels where
            // the finite difference is itself smooth (see above). A first
            // version took that change as a central difference across half a
            // texel, which straddled seams: 127% aggregate error, now 0.3%.
            std::snprintf(m, sizeof m,
                          "...and the normal's, through the environment lookup "
                          "(%.1f%% aggregate error over gradients up to %.2f)",
                          100.0 * sumErrN / std::max(1e-12, sumN), scaleN);
            Check(scaleN > 1e-2 && sumErrN < 0.02 * sumN, m);
            (void)worstN;
        }

        // --- deferred reflection: a mirror, learned -------------------------
        //
        // A flat disc of Gaussians, 70% mirror, reflecting a patterned
        // environment, seen from nine cameras on an arc with two held out.
        // The reflection slides across the disc as the camera moves -- what
        // view-dependent colour can only smear. Trained from the same start
        // twice: spherical harmonics alone (degree 3, the baseline), and with
        // reflections. The reflection model must fit the HELD-OUT views
        // clearly better: that is the difference between drawing a
        // reflection and memorising each photograph's.
        {
            const Vec3 centre{0.0, 0.0, 4.5};
            PointCloud mirror;
            for (int i = 0; i < 9; ++i) {
                const double a = (double(i) - 4.0) * 10.0 * 3.14159265358979 / 180.0;
                const Vec3 eye = centre + Vec3{3.5 * std::sin(a), -0.6, -3.5 * std::cos(a)};
                const Vec3 z = (centre - eye).Normalized();
                const Vec3 x = Vec3{0.0, 1.0, 0.0}.Cross(z).Normalized();
                const Vec3 y = z.Cross(x);
                Camera c;
                c.width = W; c.height = H;
                c.focal = 60.0; c.cx = W * 0.5 - 0.5; c.cy = H * 0.5 - 0.5;
                c.R.m[0] = x.x; c.R.m[1] = x.y; c.R.m[2] = x.z;
                c.R.m[3] = y.x; c.R.m[4] = y.y; c.R.m[5] = y.z;
                c.R.m[6] = z.x; c.R.m[7] = z.y; c.R.m[8] = z.z;
                c.t = c.R * eye * -1.0;
                c.solved = true;
                mirror.cameras.push_back(c);
            }
            for (int gy = 0; gy < 20; ++gy)
                for (int gx = 0; gx < 20; ++gx) {
                    Splat s;
                    s.mean = centre + Vec3{(gx - 9.5) * 0.09, (gy - 9.5) * 0.09, 0.0};
                    s.scale = Vec3{0.06, 0.06, 0.005};   // flat, facing -z
                    s.opacity = 0.95;
                    s.color = Vec3{0.25, 0.3, 0.35};
                    mirror.splats.push_back(s);
                }
            // The truth: 70% mirror, normal toward the cameras, and an
            // environment with a pattern the reflection can be seen to move.
            std::vector<ReflParam> tr(mirror.splats.size());
            for (ReflParam& r : tr) {
                r.reflLogit = std::log(0.7 / 0.3);
                r.normal[0] = 0; r.normal[1] = 0; r.normal[2] = -1;
            }
            EnvMap trueEnv;
            trueEnv.Init(16, 0.0);
            for (int f = 0; f < 6; ++f)
                for (int y = 0; y < 16; ++y)
                    for (int x = 0; x < 16; ++x) {
                        const size_t at = ((size_t(f) * 16 + size_t(y)) * 16 + size_t(x)) * 3;
                        const double band = 0.5 + 0.45 * std::sin(0.8 * x + 1.3 * f);
                        trueEnv.texels[at] = band;
                        trueEnv.texels[at + 1] = 0.5 + 0.4 * std::cos(0.6 * y + f);
                        trueEnv.texels[at + 2] = 1.0 - band;
                    }
            std::vector<SplatParam> tp;
            for (const Splat& s : mirror.splats) tp.push_back(ToParam(s));
            std::vector<Image> mirrorFrames;
            for (const Camera& c : mirror.cameras) {
                SplatRaster r;
                std::vector<double> rgb;
                RenderShaded(r, tp, std::vector<float>{}, 0, tr, trueEnv,
                             SplatCamFrom(c, W, H), RasterOptions{}, &rgb);
                Image im;
                im.Alloc(ImageDesc{W, H, Format::RGBA32F});
                ImageView v = im.MapCpuWrite();
                for (int y = 0; y < H; ++y)
                    for (int x = 0; x < W; ++x) {
                        float* p = v.At<float>(x, y);
                        for (int ch = 0; ch < 3; ++ch)
                            p[ch] = float(rgb[(size_t(y) * W + size_t(x)) * 3 + ch]);
                        p[3] = 1.0f;
                    }
                mirrorFrames.push_back(std::move(im));
            }

            auto trainMirror = [&](bool withReflect, double* held, double* train,
                                   PointCloud* outCloud, ComputeContext* device = nullptr,
                                   std::string* noteOut = nullptr) {
                auto a = Registry::Get().Create("train_splats");
                std::string e;
                a->FindParam("iterations")->SetFromScript(Value(600.0), &e);
                a->FindParam("downscale")->SetFromScript(Value(1.0), &e);
                a->FindParam("densify")->SetFromScript(Value(0.0), &e);
                a->FindParam("holdout")->SetFromScript(Value(4.0), &e);
                a->FindParam("sh_degree")->SetFromScript(Value(3.0), &e);
                a->FindParam("sh_every")->SetFromScript(Value(50.0), &e);
                // What the colour CAN fit in 400 iterations, not when the
                // loss stops falling fast: stop_below off.
                a->FindParam("stop_below")->SetFromScript(Value(0.0), &e);
                a->FindParam("reflect")->SetFromScript(Value(withReflect ? 1.0 : 0.0), &e);
                a->FindParam("reflect_from")->SetFromScript(Value(50.0), &e);
                a->FindParam("env_res")->SetFromScript(Value(16.0), &e);
                a->SetGroupGpu(device);
                PointCloud trained = mirror;
                for (Splat& s : trained.splats) s.color = Vec3{0.5, 0.5, 0.5};
                std::string err;
                if (!a->RunReconstruct(&mirrorFrames, &trained, &err)) {
                    std::printf("       %s\n", err.c_str());
                    return false;
                }
                const std::string note = a->RunReport();
                std::printf("       %s\n", note.c_str());
                if (noteOut) *noteOut = note;
                double b0 = 0;
                size_t at = note.find("PSNR ");
                if (at == std::string::npos) return false;
                std::sscanf(note.c_str() + at, "PSNR %lf -> %lf", &b0, train);
                at = note.find("HELD OUT");
                if (at == std::string::npos) return false;
                at = note.find("PSNR ", at);
                if (at == std::string::npos) return false;
                std::sscanf(note.c_str() + at, "PSNR %lf -> %lf", &b0, held);
                *outCloud = std::move(trained);
                return true;
            };
            double heldSh = 0, trainSh = 0, heldRefl = 0, trainRefl = 0;
            PointCloud cSh, cRefl;
            const bool ok = trainMirror(false, &heldSh, &trainSh, &cSh) &&
                            trainMirror(true, &heldRefl, &trainRefl, &cRefl);
            Check(ok, "train_splats runs on the mirror with and without reflections");
            char m[240];
            std::snprintf(m, sizeof m,
                          "reflections draw the mirror on views they never saw "
                          "(held out: %.2f dB with harmonics alone, %.2f with "
                          "reflections; trained on: %.2f, %.2f)",
                          heldSh, heldRefl, trainSh, trainRefl);
            Check(ok && heldRefl > heldSh + 2.0, m);
            Check(ok && cRefl.envRes == 16 && cRefl.splatRefl.size() == cRefl.splats.size() * 4,
                  "...and the trained cloud carries its reflectivity, normals and "
                  "environment");

            // THE SAME ON THE GPU: three composites per step through the
            // payload swap, three backwards gathered by kAccum, and the
            // reflection step on the device; the environment on the CPU.
            ID3D12Device* dev = nullptr;
            if (FAILED(TestDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
                                         IID_PPV_ARGS(&dev)))) {
                std::printf("       no D3D12 device; GPU reflections not checked\n");
            } else {
                ComputeContext gpu;
                if (gpu.Init(dev)) {
                    double heldG = 0, trainG = 0;
                    PointCloud cg;
                    std::string note;
                    const bool okG = trainMirror(true, &heldG, &trainG, &cg, &gpu, &note);
                    Check(okG && note.find("iterations on the GPU") != std::string::npos,
                          "reflection training runs on the GPU");
                    std::snprintf(m, sizeof m,
                                  "...and draws the mirror on unseen views as the CPU "
                                  "does (held out %.2f dB against %.2f)", heldG, heldRefl);
                    Check(okG && std::fabs(heldG - heldRefl) < 2.0, m);

                    // render_splats ON THE DEVICE, through training's own
                    // forward pass, against its CPU reference -- on this
                    // cloud, which has harmonics and reflections, so the
                    // whole forward path is compared, payload swap included.
                    if (okG) {
                        auto renderWith = [&](ComputeContext* device, ImageSet* outSet,
                                              std::string* rep) {
                            auto rs = Registry::Get().Create("render_splats");
                            rs->SetGroupGpu(device);
                            std::string rerr;
                            if (!rs->RunDense(nullptr, cg, outSet, &rerr)) { *rep = rerr; return false; }
                            *rep = rs->RunReport();
                            return true;
                        };
                        ImageSet onDev, onCpu;
                        std::string repDev, repCpu;
                        const bool okR = renderWith(&gpu, &onDev, &repDev) &&
                                         renderWith(nullptr, &onCpu, &repCpu);
                        int worst = 0;
                        long long off = 0, total = 0;
                        if (okR && onDev.images.size() == onCpu.images.size())
                            for (size_t k = 0; k < onDev.images.size(); ++k) {
                                ImageView a = onDev.images[k].MapCpuRead();
                                ImageView b = onCpu.images[k].MapCpuRead();
                                for (int y = 0; y < a.desc.height; ++y)
                                    for (int x = 0; x < a.desc.width; ++x)
                                        for (int ch = 0; ch < 3; ++ch) {
                                            const int d = std::abs(int(a.At<uint8_t>(x, y)[ch]) -
                                                                   int(b.At<uint8_t>(x, y)[ch]));
                                            worst = std::max(worst, d);
                                            off += d > 2 ? 1 : 0;
                                            ++total;
                                        }
                            }
                        Check(okR && repDev.find(", " + std::to_string(onDev.images.size()) +
                                                 " on the GPU") != std::string::npos,
                              "render_splats renders on the GPU (" + repDev + ")");
                        std::snprintf(m, sizeof m,
                                      "...matching its CPU reference (worst %d levels; %lld "
                                      "of %lld channel values more than 2 apart)",
                                      worst, off, total);
                        Check(okR && total > 0 && off * 1000 < total, m);
                    }

                    // ONE STEP, reflections on from the start, CPU against
                    // GPU: every parameter's movement, reflectivity and
                    // normal included. The fit above could come out close
                    // with a wrong term the optimiser works around; a single
                    // step cannot.
                    //
                    // ELLIPTICAL DISCS, not the round ones above: a round
                    // disc is unchanged by spinning about its own normal, so
                    // that quaternion component's gradient is exactly zero --
                    // and Adam's sign-like first step turned float rounding
                    // of zero into a full step on the GPU (1% of the whole
                    // movement, all in that one component) while double
                    // stayed at 3e-7. Not a kernel error; a symmetry.
                    auto oneStep = [&](ComputeContext* device, PointCloud* out) {
                        auto a = Registry::Get().Create("train_splats");
                        std::string e;
                        a->FindParam("iterations")->SetFromScript(Value(1.0), &e);
                        a->FindParam("downscale")->SetFromScript(Value(1.0), &e);
                        a->FindParam("densify")->SetFromScript(Value(0.0), &e);
                        a->FindParam("sh_degree")->SetFromScript(Value(0.0), &e);
                        a->FindParam("reflect")->SetFromScript(Value(1.0), &e);
                        a->FindParam("reflect_from")->SetFromScript(Value(0.0), &e);
                        a->FindParam("env_res")->SetFromScript(Value(16.0), &e);
                        a->SetGroupGpu(device);
                        *out = mirror;
                        for (Splat& s : out->splats) { s.color = Vec3{0.5, 0.5, 0.5}; s.scale.y *= 0.8; }
                        std::string err;
                        return a->RunReconstruct(&mirrorFrames, out, &err);
                    };
                    PointCloud s0 = mirror, sc, sg;
                    for (Splat& s : s0.splats) { s.color = Vec3{0.5, 0.5, 0.5}; s.scale.y *= 0.8; }
                    const bool okS = oneStep(nullptr, &sc) && oneStep(&gpu, &sg);
                    double diff = 0.0, moved = 0.0;
                    if (okS && sc.splatRefl.size() == sg.splatRefl.size()) {
                        for (size_t i = 0; i < s0.splats.size(); ++i) {
                            const SplatParam p0 = ToParam(s0.splats[i]);
                            const SplatParam pc = ToParam(sc.splats[i]);
                            const SplatParam pg = ToParam(sg.splats[i]);
                            for (int q = 0; q < SplatParam::kCount; ++q) {
                                const double dc = pc.Data()[q] - p0.Data()[q];
                                const double dg = pg.Data()[q] - p0.Data()[q];
                                diff += std::fabs(dc - dg);
                                moved += std::fabs(dc);
                            }
                        }
                        // Reflectivity starts at sigmoid(-4.6) and the normal
                        // at the disc's axis, identical on both paths, so the
                        // stored values' difference is the steps' difference.
                        const double r0 = 1.0 / (1.0 + std::exp(4.6));
                        for (size_t k = 0; k < sc.splatRefl.size(); ++k) {
                            const double d = std::fabs(double(sc.splatRefl[k]) - double(sg.splatRefl[k]));
                            diff += d;
                            if (k % 4 == 0) moved += std::fabs(double(sc.splatRefl[k]) - r0);
                        }
                    }
                    std::snprintf(m, sizeof m,
                                  "one GPU reflection step moves every parameter as the "
                                  "CPU step does (aggregate difference %.1e of the movement)",
                                  diff / std::max(1e-30, moved));
                    Check(okS && moved > 0.0 && diff / moved < 0.005, m);
                }
                dev->Release();
            }
        }

        // --- view-dependent colour: spherical harmonics ---------------------
        //
        // Gaussians whose colour CHANGES WITH THE VIEWPOINT -- a degree-1 term
        // along the horizontal, as a sheen brightening toward one side would --
        // photographed from seven cameras on an arc of +-60 degrees about the
        // scene. (Its own cameras: the three above sit within a few degrees of
        // each other, where every Gaussian looks nearly the same from all of
        // them and there is no view dependence to fit.) Plain colour can only
        // average the views; degree 1 can match each. The test is the gap
        // between the two, trained from the same grey start.
        {
            const Vec3 centre{0.0, 0.0, 3.75};
            PointCloud ring;
            for (int i = 0; i < 7; ++i) {
                const double a = (double(i) - 3.0) * (60.0 / 3.0) * 3.14159265358979 / 180.0;
                const Vec3 eye = centre + Vec3{3.5 * std::sin(a), 0.0, -3.5 * std::cos(a)};
                const Vec3 z = (centre - eye).Normalized();
                const Vec3 y{0.0, 1.0, 0.0};                 // down, as OpenCV's
                const Vec3 x = y.Cross(z).Normalized();      // x = y cross z
                Camera c;
                c.width = W; c.height = H;
                c.focal = 60.0; c.cx = W * 0.5 - 0.5; c.cy = H * 0.5 - 0.5;
                c.R.m[0] = x.x; c.R.m[1] = x.y; c.R.m[2] = x.z;
                c.R.m[3] = y.x; c.R.m[4] = y.y; c.R.m[5] = y.z;
                c.R.m[6] = z.x; c.R.m[7] = z.y; c.R.m[8] = z.z;
                c.t = c.R * eye * -1.0;
                c.solved = true;
                ring.cameras.push_back(c);
            }
            ring.splats = truth.splats;
            for (Splat& s : ring.splats) s.mean = centre + (s.mean - Vec3{0.0, 0.0, 3.75}) * 0.6;

            std::vector<float> trueSh(ring.splats.size() * size_t(kShRest), 0.0f);
            for (size_t i = 0; i < ring.splats.size(); ++i)
                for (int ch = 0; ch < 3; ++ch)   // k = 2 is the x term, -C1 x
                    trueSh[i * size_t(kShRest) + 2 * 3 + size_t(ch)] =
                        float((rnd() * 2.0 - 1.0) * 0.6);
            std::vector<SplatParam> tp;
            for (const Splat& s : ring.splats) tp.push_back(ToParam(s));
            std::vector<Image> viewFrames;
            RasterOptions opt;
            for (const Camera& c : ring.cameras) {
                const SplatCam sc = SplatCamFrom(c, W, H);
                std::vector<SplatParam> shaded;
                ShadeForView(tp, trueSh, 1, CentreOf(sc), &shaded);
                SplatRaster r;
                std::vector<double> rgb;
                r.Forward(shaded, sc, opt, &rgb);
                Image im;
                im.Alloc(ImageDesc{W, H, Format::RGBA32F});
                ImageView v = im.MapCpuWrite();
                for (int y = 0; y < H; ++y)
                    for (int x = 0; x < W; ++x) {
                        float* p = v.At<float>(x, y);
                        for (int ch = 0; ch < 3; ++ch)
                            p[ch] = float(rgb[(size_t(y) * W + size_t(x)) * 3 + ch]);
                        p[3] = 1.0f;
                    }
                viewFrames.push_back(std::move(im));
            }
            PointCloud ringStart = ring;
            for (Splat& s : ringStart.splats) s.color = Vec3{0.5, 0.5, 0.5};

            auto trainAt = [&](int degree, double* psnr, PointCloud* outCloud,
                                ComputeContext* device = nullptr, std::string* noteOut = nullptr) {
                auto a = Registry::Get().Create("train_splats");
                std::string e;
                a->FindParam("iterations")->SetFromScript(Value(400.0), &e);
                a->FindParam("downscale")->SetFromScript(Value(1.0), &e);
                a->FindParam("densify")->SetFromScript(Value(0.0), &e);
                a->FindParam("sh_degree")->SetFromScript(Value(double(degree)), &e);
                a->FindParam("sh_every")->SetFromScript(Value(50.0), &e);
                // What the colour CAN fit in 400 iterations, not when the
                // loss stops falling fast: stop_below off.
                a->FindParam("stop_below")->SetFromScript(Value(0.0), &e);
                a->SetGroupGpu(device);
                PointCloud trained = ringStart;
                std::string err;
                if (!a->RunReconstruct(&viewFrames, &trained, &err)) return false;
                const std::string note = a->RunReport();
                if (noteOut) *noteOut = note;
                double before = 0;
                const size_t at = note.find("PSNR ");
                if (at == std::string::npos) return false;
                std::sscanf(note.c_str() + at, "PSNR %lf -> %lf", &before, psnr);
                *outCloud = std::move(trained);
                return true;
            };
            double flat = 0, withSh = 0;
            PointCloud c0, c1;
            const bool ok = trainAt(0, &flat, &c0) && trainAt(1, &withSh, &c1);
            Check(ok, "train_splats runs with and without spherical harmonics");
            char m[200];
            std::snprintf(m, sizeof m,
                          "view-dependent colour fits what one colour cannot "
                          "(PSNR %.2f dB plain, %.2f with degree 1)", flat, withSh);
            Check(ok && withSh > flat + 3.0, m);
            Check(ok && c1.shDegree == 1 &&
                      c1.splatSh.size() == c1.splats.size() * size_t(kShRest),
                  "...and the trained cloud carries its coefficients");
            Check(ok && c0.shDegree == 0 && c0.splatSh.empty(),
                  "...while degree 0 leaves plain colour, with nothing extra");

            // THE SAME ON THE GPU, where the colour is shaded in the project
            // kernel and the coefficients stepped by their own. Same scene,
            // same degree: the fit must match the CPU reference's.
            ID3D12Device* dev = nullptr;
            if (FAILED(TestDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
                                         IID_PPV_ARGS(&dev)))) {
                std::printf("       no D3D12 device; GPU spherical harmonics not checked\n");
            } else {
                ComputeContext gpu;
                if (gpu.Init(dev)) {
                    double onGpu = 0;
                    PointCloud cg;
                    std::string note;
                    const bool okG = trainAt(1, &onGpu, &cg, &gpu, &note);
                    Check(okG && note.find("iterations on the GPU") != std::string::npos,
                          "degree-1 training runs on the GPU");
                    char mg[200];
                    std::snprintf(mg, sizeof mg,
                                  "...and fits the view-dependent scene as the CPU does "
                                  "(%.2f dB against %.2f)", onGpu, withSh);
                    Check(okG && std::fabs(onGpu - withSh) < 1.0, mg);
                    Check(okG && cg.shDegree == 1 &&
                              cg.splatSh.size() == cg.splats.size() * size_t(kShRest),
                          "...and brings its coefficients home");
                }
                dev->Release();
            }
        }

        // --- densification, where it is supposed to help ---------------------
        //
        // The case it exists for: too FEW Gaussians, too LARGE for the detail
        // in the photographs. Twelve big grey blobs where the scene has sixty
        // small coloured ones. Without densification training can only
        // reshape the twelve; with it, the set should grow and fit better.
        PointCloud coarse = truth;
        coarse.splats.clear();
        for (int i = 0; i < 12; ++i) {
            Splat s;
            s.mean = Vec3{rnd() * 2.4 - 1.2, rnd() * 1.6 - 0.8, 3.0 + rnd() * 1.5};
            s.scale = Vec3{0.35, 0.35, 0.35};
            s.opacity = 0.6;
            s.color = Vec3{0.5, 0.5, 0.5};
            coarse.splats.push_back(s);
        }
        auto run = [&](bool dens, PointCloud* outCloud, double* psnr) {
            auto a = Registry::Get().Create("train_splats");
            std::string e;
            a->FindParam("iterations")->SetFromScript(Value(600.0), &e);
            a->FindParam("downscale")->SetFromScript(Value(1.0), &e);
            a->FindParam("densify")->SetFromScript(Value(dens ? 1.0 : 0.0), &e);
            // The world-size limit is 10% of the CAMERA spread, which here is
            // 0.8 -- smaller than the true Gaussians. A known weakness of the
            // paper's heuristic for close-set cameras; see train_splats.cpp.
            a->FindParam("max_world_size")->SetFromScript(Value(0.0), &e);
            *outCloud = coarse;
            std::string err;
            if (!a->RunReconstruct(&frames, outCloud, &err)) return false;
            const std::string note = a->RunReport();
            std::printf("       densify %s: %s\n", dens ? "on " : "off", note.c_str());
            double before = 0;
            const size_t at = note.find("PSNR ");
            if (at == std::string::npos) return false;
            std::sscanf(note.c_str() + at, "PSNR %lf -> %lf", &before, psnr);
            return true;
        };
        PointCloud withD, withoutD;
        double pOn = 0, pOff = 0;
        const bool okOn = run(true, &withD, &pOn);
        const bool okOff = run(false, &withoutD, &pOff);
        Check(okOn && okOff, "train_splats runs with and without densification");
        if (okOn && okOff) {
            Check(withD.splats.size() > coarse.splats.size() * 2,
                  "densification grows a too-coarse set (12 -> " +
                      std::to_string(withD.splats.size()) + ")");
            char m[200];
            std::snprintf(m, sizeof(m),
                          "...and the grown set fits better (%.2f dB against "
                          "%.2f without)", pOn, pOff);
            Check(pOn > pOff + 2.0, m);
        }

        // --- max_gaussians CAPS THE START, not only growth ---------------------
        //
        // A 100-frame video fused to 2.6 million points, past what the GPU
        // trainer's state texture can hold, and training fell back to hours on
        // the CPU because the cap only limited densification. Twice the cap
        // in: exactly the cap out, and each kept Gaussian wider than before.
        {
            const size_t cap = start.splats.size() / 2;
            auto a = Registry::Get().Create("train_splats");
            std::string e;
            a->FindParam("iterations")->SetFromScript(Value(0.0), &e);
            a->FindParam("densify")->SetFromScript(Value(0.0), &e);
            a->FindParam("max_gaussians")->SetFromScript(Value(double(cap)), &e);
            PointCloud out = start;
            std::string err;
            const bool ok = a->RunReconstruct(&frames, &out, &err);
            Check(ok && out.splats.size() == cap,
                  "a start larger than max_gaussians is thinned to it (" +
                      std::to_string(start.splats.size()) + " -> " +
                      std::to_string(out.splats.size()) + ")" + (ok ? "" : ": " + err));
            if (ok && !out.splats.empty()) {
                const Splat& before = start.splats[0];
                const Splat& after = out.splats[0];
                const double wb = std::max({before.scale.x, before.scale.y, before.scale.z});
                const double wa = std::max({after.scale.x, after.scale.y, after.scale.z});
                Check(wa > wb * 1.3,
                      "...and the kept ones widened to cover for the rest");
            }
        }

        // --- training on the GPU matches training on the CPU -----------------
        //
        // The GPU trainer keeps the whole state on the device and runs its
        // own projection, chain rule and Adam step, in float. Checked two
        // ways against the CPU loop, which is itself checked against finite
        // differences:
        //
        //   * ONE step, comparing every parameter's update. Adam's first step
        //     moves each parameter by about the learning rate in the
        //     direction of its gradient, so this isolates the chain rule: a
        //     wrong term sends parameters the wrong way.
        //   * THREE HUNDRED steps, comparing the fit, which says the loop as
        //     a whole converges the same way.
        ID3D12Device* dev = nullptr;
        if (FAILED(TestDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
                                     IID_PPV_ARGS(&dev)))) {
            std::printf("       no D3D12 device; GPU training not checked\n");
        } else {
            ComputeContext gpu;
            if (!gpu.Init(dev)) {
                std::printf("       compute init failed; GPU training not checked\n");
            } else {
                auto train = [&](ComputeContext* device, int its, PointCloud* out,
                                 std::string* note) {
                    auto a = Registry::Get().Create("train_splats");
                    std::string e;
                    a->FindParam("iterations")->SetFromScript(Value(double(its)), &e);
                    a->FindParam("downscale")->SetFromScript(Value(1.0), &e);
                    a->FindParam("densify")->SetFromScript(Value(0.0), &e);
                    a->SetGroupGpu(device);
                    *out = start;
                    std::string err;
                    const bool ok = a->RunReconstruct(&frames, out, &err);
                    *note = ok ? a->RunReport() : err;
                    return ok;
                };

                PointCloud c1, g1;
                std::string n1c, n1g;
                const bool ok1 = train(nullptr, 1, &c1, &n1c) && train(&gpu, 1, &g1, &n1g);
                Check(ok1, "one training step runs on both paths");
                Check(ok1 && n1g.find("1 of 1 iterations on the GPU") != std::string::npos,
                      "...and the GPU one really ran there (" +
                          n1g.substr(n1g.find(';') == std::string::npos ? 0 : n1g.rfind(';')) + ")");
                if (ok1) {
                    // Aggregate: how far the two updates differ, against how
                    // far the parameters moved at all.
                    double diff = 0.0, moved = 0.0;
                    for (size_t i = 0; i < start.splats.size(); ++i) {
                        const SplatParam s0 = ToParam(start.splats[i]);
                        const SplatParam sc = ToParam(c1.splats[i]);
                        const SplatParam sg = ToParam(g1.splats[i]);
                        for (int q = 0; q < SplatParam::kCount; ++q) {
                            const double dc = sc.Data()[q] - s0.Data()[q];
                            const double dg = sg.Data()[q] - s0.Data()[q];
                            diff += std::fabs(dc - dg);
                            moved += std::fabs(dc);
                        }
                    }
                    char m1[200];
                    std::snprintf(m1, sizeof(m1),
                                  "one GPU step moves the parameters as the CPU "
                                  "step does (aggregate difference %.1e of the "
                                  "movement)", diff / std::max(1e-30, moved));
                    // Adam's first step is sign-like, so a gradient that is
                    // essentially zero can round to either sign in float and
                    // move a parameter the full step the other way. A few
                    // percent is that; a wrong chain rule is most of it.
                    Check(moved > 0.0 && diff / moved < 0.05, m1);
                }

                // THE SAME, WITH DEPTH SUPERVISION. Depth maps shaped as
                // plane_sweep emits them -- depth, confidence, colour per
                // camera -- passed as the third input. The values only need
                // to pull: each camera's is 10% beyond the Gaussians' mean
                // depth in it, so every depth gradient is non-zero.
                {
                    std::vector<Image> depthSet;
                    for (size_t c = 0; c < start.cameras.size(); ++c) {
                        const Camera& cam = start.cameras[c];
                        double zs = 0.0;
                        for (const Splat& s : start.splats) zs += (cam.R * s.mean + cam.t).z;
                        const float z = float(1.1 * zs / double(start.splats.size()));
                        const ImageDesc fd = frames[c].Desc();
                        ImageDesc dd{fd.width, fd.height, Format::R32F};
                        dd.isDepth = true;
                        dd.depthNear = z * 0.5f;
                        dd.depthFar = z * 2.0f;
                        Image dimg, cimg, rimg;
                        dimg.Alloc(dd);
                        cimg.Alloc(ImageDesc{fd.width, fd.height, Format::R32F});
                        rimg.Alloc(ImageDesc{fd.width, fd.height, Format::RGBA32F});
                        ImageView dv = dimg.MapCpuWrite(), cv = cimg.MapCpuWrite();
                        for (int y = 0; y < fd.height; ++y)
                            for (int x = 0; x < fd.width; ++x) {
                                *dv.At<float>(x, y) = z;
                                *cv.At<float>(x, y) = 1.0f;
                            }
                        dv = ImageView{};
                        cv = ImageView{};
                        depthSet.push_back(std::move(dimg));
                        depthSet.push_back(std::move(cimg));
                        depthSet.push_back(std::move(rimg));
                    }
                    auto trainD = [&](ComputeContext* device, PointCloud* out,
                                      std::string* note) {
                        auto a = Registry::Get().Create("train_splats");
                        std::string e;
                        a->FindParam("iterations")->SetFromScript(Value(1.0), &e);
                        a->FindParam("downscale")->SetFromScript(Value(1.0), &e);
                        a->FindParam("densify")->SetFromScript(Value(0.0), &e);
                        a->FindParam("depth_weight")->SetFromScript(Value(1.0), &e);
                        a->SetGroupGpu(device);
                        a->SetReconstructExtra(&depthSet);
                        *out = start;
                        std::string err;
                        const bool ok = a->RunReconstruct(&frames, out, &err);
                        *note = ok ? a->RunReport() : err;
                        return ok;
                    };
                    PointCloud cd, gd;
                    std::string ncd, ngd;
                    const bool okd = trainD(nullptr, &cd, &ncd) && trainD(&gpu, &gd, &ngd);
                    Check(okd && ncd.find("DEPTH") != std::string::npos,
                          "a depth-supervised step runs on both paths" +
                              (okd ? std::string() : ": " + ncd + " / " + ngd));
                    if (okd) {
                        double diff = 0.0, moved = 0.0, meanShift = 0.0;
                        for (size_t i = 0; i < start.splats.size(); ++i) {
                            const SplatParam s0 = ToParam(start.splats[i]);
                            const SplatParam sc = ToParam(cd.splats[i]);
                            const SplatParam sg = ToParam(gd.splats[i]);
                            const SplatParam sn = ToParam(c1.splats[i]);   // no depth
                            for (int q = 0; q < SplatParam::kCount; ++q) {
                                const double dc = sc.Data()[q] - s0.Data()[q];
                                const double dg = sg.Data()[q] - s0.Data()[q];
                                diff += std::fabs(dc - dg);
                                moved += std::fabs(dc);
                                if (q < 3) meanShift += std::fabs(sc.Data()[q] - sn.Data()[q]);
                            }
                        }
                        char m2[200];
                        std::snprintf(m2, sizeof(m2),
                                      "...and the GPU step matches the CPU one "
                                      "(aggregate difference %.1e of the movement)",
                                      diff / std::max(1e-30, moved));
                        Check(moved > 0.0 && diff / moved < 0.05, m2);
                        // Depth must actually change the step -- a term that
                        // was silently dropped would pass the comparison.
                        Check(meanShift > 0.0,
                              "...and depth changes where the Gaussians move");
                    }

                    // THE COLOUR GATE, with no depth loss: the same maps, 10%
                    // beyond the Gaussians, against a 5% gate -- so most
                    // pixels are gated. The device must still match the CPU
                    // (its target reaches the kernel by a path the depth
                    // loss does not take), and colour must move less than
                    // ungated.
                    auto trainG = [&](ComputeContext* device, PointCloud* out,
                                      std::string* note) {
                        auto a = Registry::Get().Create("train_splats");
                        std::string e;
                        a->FindParam("iterations")->SetFromScript(Value(1.0), &e);
                        a->FindParam("downscale")->SetFromScript(Value(1.0), &e);
                        a->FindParam("densify")->SetFromScript(Value(0.0), &e);
                        a->FindParam("depth_weight")->SetFromScript(Value(0.0), &e);
                        a->FindParam("colour_gate")->SetFromScript(Value(0.05), &e);
                        a->SetGroupGpu(device);
                        a->SetReconstructExtra(&depthSet);
                        *out = start;
                        std::string err;
                        const bool ok = a->RunReconstruct(&frames, out, &err);
                        *note = ok ? a->RunReport() : err;
                        return ok;
                    };
                    PointCloud cg, gg;
                    std::string ncg, ngg;
                    const bool okg = trainG(nullptr, &cg, &ncg) && trainG(&gpu, &gg, &ngg);
                    Check(okg, "a colour-gated step runs on both paths" +
                                   (okg ? std::string() : ": " + ncg + " / " + ngg));
                    if (okg) {
                        double diff = 0.0, moved = 0.0, colGated = 0.0, colFree = 0.0;
                        for (size_t i = 0; i < start.splats.size(); ++i) {
                            const SplatParam s0 = ToParam(start.splats[i]);
                            const SplatParam sc = ToParam(cg.splats[i]);
                            const SplatParam sg = ToParam(gg.splats[i]);
                            const SplatParam sn = ToParam(c1.splats[i]);   // ungated
                            for (int q = 0; q < SplatParam::kCount; ++q) {
                                const double dc = sc.Data()[q] - s0.Data()[q];
                                diff += std::fabs(dc - (sg.Data()[q] - s0.Data()[q]));
                                moved += std::fabs(dc);
                                if (q >= 11) {
                                    colGated += std::fabs(dc);
                                    colFree += std::fabs(sn.Data()[q] - s0.Data()[q]);
                                }
                            }
                        }
                        char m3[200];
                        std::snprintf(m3, sizeof(m3),
                                      "...and the GPU step matches the CPU one (aggregate "
                                      "difference %.1e of the movement)",
                                      diff / std::max(1e-30, moved));
                        Check(moved > 0.0 && diff / moved < 0.05, m3);
                        Check(colGated < colFree,
                              "...and colour moves less than ungated (" +
                                  std::to_string(colGated) + " against " +
                                  std::to_string(colFree) + ")");
                    }
                }

                PointCloud cN, gN;
                std::string nNc, nNg;
                const bool okN = train(nullptr, 300, &cN, &nNc) && train(&gpu, 300, &gN, &nNg);
                if (okN) {
                    double bc = 0, ac = 0, bg = 0, ag = 0;
                    std::sscanf(nNc.c_str() + nNc.find("PSNR "), "PSNR %lf -> %lf", &bc, &ac);
                    std::sscanf(nNg.c_str() + nNg.find("PSNR "), "PSNR %lf -> %lf", &bg, &ag);
                    std::printf("       CPU: %s\n       GPU: %s\n", nNc.c_str(), nNg.c_str());
                    char m2[200];
                    std::snprintf(m2, sizeof(m2),
                                  "300 GPU steps fit as well as 300 CPU steps "
                                  "(%.2f dB against %.2f)", ag, ac);
                    Check(ag > bg + 8.0 && std::fabs(ag - ac) < 1.0, m2);
                }
            }
            dev->Release();
        }
    }

    // --- the shipped script's viewer types ----------------------------------
    //
    // The app decides what KIND of panel a viewer wants from the pipeline's
    // declared port types, at build time, so the layout settles immediately
    // rather than rearranging when the first result lands. That decision is
    // what routes a reconstruction to a 3D viewport instead of an image panel
    // -- and before this, the worker silently dropped any viewer whose source
    // was not an Image, so a display() on a cloud showed "computing..."
    // forever.
    //
    // Checked on the shipped script rather than an inline one: the thing that
    // has to keep working is sfm.tgl.
    {
        std::printf("\n--- sfm.tgl viewer types ---\n");

        const std::string path = std::string(TGLAB_SOURCE_DIR) + "/scripts/sfm.tgl";
        std::ifstream in(path, std::ios::binary);
        std::ostringstream ss;
        ss << in.rdbuf();
        const std::string src = ss.str();
        Check(!src.empty(), "sfm.tgl is readable");

        if (!src.empty()) {
            Program prog;
            std::string err;
            Check(Parse(src, &prog, &err), "sfm.tgl parses" +
                                               (err.empty() ? "" : ": " + err));

            UiState ui;
            Pipeline p;
            std::vector<SourceImage> names{{"group", 0}};
            const InterpResult r = Interpret(prog, names, &ui, &p);
            Check(r.ok, "sfm.tgl interprets" + (r.ok ? "" : ": " + r.error));

            if (r.ok) {
                int images = 0, clouds = 0;
                for (const ViewerDecl& vd : p.Viewers()) {
                    const DataType t = p.PortType(vd.source);
                    if (t == DataType::PointCloud) ++clouds;
                    else ++images;
                }
                // TWO now: the sparse reconstruction and the dense cloud
                // fused from the depth maps. The check is that the routing
                // works at all -- a display() on a cloud used to show
                // "computing..." forever -- so what matters is that a
                // PointCloud viewer is recognised, not how many there are.
                Check(clouds >= 1,
                      "a viewer declares a reconstruction (" +
                          std::to_string(clouds) + ")");
                Check(images >= 1,
                      "...and at least one declares images (" +
                          std::to_string(images) + ")");

                // THE SPARSE CHAIN must still end at the bundle adjuster, or
                // the reconstruction everything downstream reads is not the
                // refined one.
                //
                // Stated as "the last stage that produces the sparse cloud"
                // rather than "the last stage", because the dense stages now
                // run after it. Finding bundle_adjust_sfm as the last stage
                // BEFORE plane_sweep is the same guarantee: the dense stages
                // consume its output, so a sweep running against unrefined
                // cameras would show up here.
                const auto& stages = p.Stages();
                std::string lastSparse = "none";
                for (const auto& s : stages) {
                    if (s.algoName == "plane_sweep" ||
                        s.algoName == "fuse_depth") break;
                    lastSparse = s.algoName;
                }
                // solve_cameras counts: it runs the chain internally and ends
                // with bundle adjustment -- see its header.
                Check(lastSparse == "bundle_adjust_sfm" || lastSparse == "solve_cameras",
                      "the sparse chain ends at bundle adjustment (" +
                          lastSparse + ")");

                // And the dense stages, if present, run in the only order
                // that makes sense: fusion consumes what the sweep produced.
                int sweepAt = -1, fuseAt = -1;
                for (int i = 0; i < int(stages.size()); ++i) {
                    if (stages[size_t(i)].algoName == "plane_sweep") sweepAt = i;
                    if (stages[size_t(i)].algoName == "fuse_depth")  fuseAt  = i;
                }
                Check((sweepAt < 0 && fuseAt < 0) ||
                          (sweepAt >= 0 && fuseAt > sweepAt),
                      "fusion follows the sweep that feeds it");
            }
        }
    }

    std::printf("\n%s\n", g_fail == 0 ? "all SfM checks passed" : "FAILURES PRESENT");
    return g_fail == 0 ? 0 : 1;
}
