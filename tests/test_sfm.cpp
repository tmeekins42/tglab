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
#include <vector>

#include "../src/algo_util/features.h"
#include "../src/algo_util/view_graph.h"
#include "../src/core/algorithm.h"
#include "../src/app/orbit_camera.h"
#include "../src/core/pipeline.h"
#include "../src/script/interp.h"
#include "../src/script/parser.h"
#include "../src/gpu/compute.h"
#include "../src/algorithms/sfm/gpu_sweep.h"

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
                    const Vec3 wantDir = (R_a * (C_b - C_a)).Normalized();
                    const double dot = std::fabs(e.direction.Dot(wantDir));
                    Check(dot > 0.99,
                          "the translation direction points from a to b (cos " +
                              std::to_string(dot) + ")");
                }
            }
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
        for (int i = 0; i < 64; ++i) {
            Track t;
            t.hasPoint = true;
            const double ux = (double(i % 8) - 3.5) * 0.35;
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
        if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
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
                Check(lastSparse == "bundle_adjust_sfm",
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
