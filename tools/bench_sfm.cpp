// Runs the whole Structure-from-Motion chain on real photographs, headlessly.
//
// The unit tests build their fixtures from ground truth, which is what lets
// them assert that an answer is RIGHT rather than merely plausible. That is the
// correct discipline and it has a blind spot: a synthetic fixture has exactly
// the noise, the overlap and the parallax the fixture author imagined. Real
// photographs have repeated windows, blown sky, a tree that moved, and baselines
// the photographer chose for aesthetic reasons.
//
// So this drives the same stages a script would -- detect, match, verify,
// relative pose, tracks, rotation averaging, positioning, triangulation, bundle
// adjustment -- over a directory of images and prints each stage's report. It
// is a measurement tool, not a test: it says what happened, and the operator
// decides whether that is right.
//
// WHAT TO LOOK AT, in rough order of what goes wrong first:
//
//   * Matches per pair. Below a hundred or so the view graph is too thin for
//     rotation averaging to have anything to average.
//   * Inlier RATE from relative_pose. A high count with a low rate means the
//     matcher is generous and RANSAC is doing all the work; a low rate on every
//     pair usually means the focal guess is badly wrong.
//   * Mean track length. Two means nothing is being chained and every point is
//     a bare pair; four or more is a healthy graph.
//   * Rotation residual, in degrees. This is the first number that says the
//     geometry is coherent rather than merely present.
//   * Reprojection RMS, before and after bundle adjustment. The only measure
//     that says the reconstruction is right.
//
//   bench_sfm <dir-or-image> [image...] [--detector NAME] [--window N]
//             [--max-dim N] [--rotation N] [--position N]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "../src/algo_util/features.h"
#include "../src/algo_util/view_graph.h"
#include "../src/core/algorithm.h"
#include "../src/core/image.h"
#include "../src/core/image_io.h"
#include "../src/core/pipeline.h"
#include "../src/core/shape.h"
#include "../src/script/interp.h"
#include "../src/script/parser.h"
#include "../src/script/value.h"

using namespace tglab;

namespace {

double Ms(std::chrono::steady_clock::time_point a,
          std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

void SetParam(AlgorithmBase* a, const char* name, double v) {
    if (!a) return;
    if (ParamBase* p = a->FindParam(name)) {
        std::string e;
        p->SetFromScript(Value(v), &e);
    }
}

// Shrinks an image so its longest side is at most `maxDim`.
//
// NOT an optimisation, or not only one. A detector's contrast thresholds and a
// matcher's ratio test were calibrated on images around a megapixel; on a 20 MP
// frame the same settings find tens of thousands of features, most of them on
// texture too fine to repeat between views. Downscaling first is what the
// reference pipelines do, and it makes the run tractable as a side effect.
bool Downscale(const Image& in, int maxDim, Image* out) {
    const ImageDesc& d = in.Desc();
    const int longest = std::max(d.width, d.height);
    if (longest <= maxDim) { *out = const_cast<Image&>(in).Clone(); return true; }

    auto algo = Registry::Get().Create("resize");
    if (!algo) return false;
    SetParam(algo.get(), "scale", double(maxDim) / double(longest));

    std::vector<Data> ins;
    ins.push_back(Data{const_cast<Image&>(in).Clone()});
    std::vector<const Data*> inPtrs{&ins[0]};
    std::vector<Data> outs(1);
    ImageDesc od = algo->OutputDesc(0, d);
    Image img;
    img.Alloc(od);
    outs[0] = Data{std::move(img)};
    RunCtx ctx(inPtrs, outs);
    algo->RunCPU(ctx);
    Image* o = std::get_if<Image>(&outs[0]);
    if (!o || !o->Valid()) return false;
    *out = o->Clone();
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);

    std::vector<std::string> files;
    std::string detector = "detect_akaze";
    int window = 3, maxDim = 1600, rotMethod = 1, posMethod = 0;
    double fov = 50.0;
    std::string matcher = "match_ann";
    double ratio = 0.8;
    int checks = 256;
    int features = 0;
    // -1 means "leave the matcher's own default alone", so the flag's absence
    // is distinguishable from an explicit --cross-check 1.
    int crossCheck = -1;
    double maxDistance = -1.0;   // negative: leave the stages' own defaults
    int poseMethod = -1;         // negative: leave relative_pose's own default
    int baIters = -1;            // negative: leave bundle_adjust_sfm's default
    std::string script;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--detector" && i + 1 < argc)      detector = argv[++i];
        else if (a == "--window" && i + 1 < argc)   window = std::atoi(argv[++i]);
        else if (a == "--max-dim" && i + 1 < argc)  maxDim = std::atoi(argv[++i]);
        else if (a == "--rotation" && i + 1 < argc) rotMethod = std::atoi(argv[++i]);
        else if (a == "--position" && i + 1 < argc) posMethod = std::atoi(argv[++i]);
        else if (a == "--fov" && i + 1 < argc)      fov = std::atof(argv[++i]);
        else if (a == "--matcher" && i + 1 < argc)  matcher = argv[++i];
        else if (a == "--ratio" && i + 1 < argc)    ratio = std::atof(argv[++i]);
        else if (a == "--checks" && i + 1 < argc)   checks = std::atoi(argv[++i]);
        else if (a == "--features" && i + 1 < argc) features = std::atoi(argv[++i]);
        else if (a == "--cross-check" && i + 1 < argc) crossCheck = std::atoi(argv[++i]);
        else if (a == "--max-distance" && i + 1 < argc) maxDistance = std::atof(argv[++i]);
        else if (a == "--pose-method" && i + 1 < argc) poseMethod = std::atoi(argv[++i]);
        else if (a == "--ba-iters" && i + 1 < argc) baIters = std::atoi(argv[++i]);
        else if (a == "--script" && i + 1 < argc)   script = argv[++i];
        else files.push_back(a);
    }

    // A directory expands to the images in it, sorted. Order matters: the
    // matchers chain against neighbours, so the filename order has to be the
    // capture order.
    if (files.size() == 1 && std::filesystem::is_directory(files[0])) {
        const std::string dir = files[0];
        files.clear();
        std::error_code ec;
        for (const auto& e : std::filesystem::directory_iterator(dir, ec)) {
            const std::string ext = e.path().extension().string();
            if (ext == ".jpg" || ext == ".JPG" || ext == ".jpeg" ||
                ext == ".png" || ext == ".PNG" || ext == ".tif" || ext == ".tiff")
                files.push_back(e.path().string());
        }
        std::sort(files.begin(), files.end());
    }

    if (files.size() < 3) {
        std::printf("usage: bench_sfm <dir-or-image> [image...] "
                    "[--detector NAME] [--window N] [--max-dim N] "
                    "[--rotation 0|1] [--position 0|1]\n");
        return 2;
    }

    std::printf("%d images, detector=%s, window=%d, max-dim=%d\n",
                int(files.size()), detector.c_str(), window, maxDim);
    std::printf("rotation=%s, position=%s\n\n",
                rotMethod == 0 ? "L2" : "L1-IRLS",
                posMethod == 0 ? "joint" : "translation averaging");

    // --- load ---------------------------------------------------------------
    const auto tLoad = std::chrono::steady_clock::now();
    ImageSet set;
    for (const std::string& f : files) {
        Image full;
        std::string err;
        if (!LoadImageFile(f, &full, &err)) {
            std::printf("  %s: %s\n", f.c_str(), err.c_str());
            return 1;
        }
        Image small;
        if (!Downscale(full, maxDim, &small)) {
            std::printf("  %s: could not downscale\n", f.c_str());
            return 1;
        }
        if (set.images.empty())
            std::printf("  %dx%d -> %dx%d\n", full.Desc().width, full.Desc().height,
                        small.Desc().width, small.Desc().height);
        set.images.push_back(std::move(small));
    }
    set.shape = Shape{{{"frame", int(set.images.size())}}};
    std::printf("  loaded %d in %.0f ms\n\n", int(set.images.size()),
                Ms(tLoad, std::chrono::steady_clock::now()));

    std::vector<Data> s;
    s.push_back(Data{std::move(set)});

    // --- script mode ---------------------------------------------------------
    //
    // Runs a shipped .tgl against these images instead of the chain below.
    //
    // Worth having because the two are NOT the same thing: the chain below is
    // what this tool builds, and a script is what a user actually runs. The
    // defaults differ -- params() exposes an algorithm's own defaults, which
    // for match_ann means `chain` OFF, so every frame matches against frame 0.
    // On a walk-around that rejects seventeen pairs of eighteen and the
    // reconstruction dies at triangulation, while this tool (which passes
    // chain=1) succeeds on the same images. Only running the script finds that.
    if (!script.empty()) {
        std::string source;
        {
            std::ifstream f(script, std::ios::binary);
            if (!f) { std::printf("could not read %s\n", script.c_str()); return 1; }
            std::ostringstream ss;
            ss << f.rdbuf();
            source = ss.str();
        }

        Program prog;
        std::string perr;
        if (!Parse(source, &prog, &perr)) {
            std::printf("parse: %s\n", perr.c_str());
            return 1;
        }

        UiState ui;
        Pipeline sp;
        std::vector<SourceImage> names{{"group", 0}};
        const InterpResult ir = Interpret(prog, names, &ui, &sp);
        if (!ir.ok) { std::printf("interpret: %s\n", ir.error.c_str()); return 1; }

        std::printf("running %s (%d stages, %d viewers)\n\n", script.c_str(),
                    int(sp.Stages().size()), int(sp.Viewers().size()));

        const auto ts = std::chrono::steady_clock::now();
        std::string serr;
        const bool sok = sp.Execute(&s, nullptr, &serr);
        const double sms = Ms(ts, std::chrono::steady_clock::now());

        for (const Stage& st : sp.Stages()) {
            if (!st.algo) continue;
            const std::string rep = st.Report();
            if (!rep.empty()) std::printf("  %s\n", rep.c_str());
            else if (!st.valid) std::printf("  %s: did not run\n", st.algoName.c_str());
        }
        std::printf("\n");

        if (!sok) { std::printf("FAILED: %s\n", serr.c_str()); return 1; }

        // What each viewer would show, and of what type -- the decision the app
        // makes when it chooses a panel kind.
        for (const ViewerDecl& vd : sp.Viewers()) {
            const Data* d = sp.Resolve(vd.source, &s);
            const char* kind = "nothing";
            if (d && std::holds_alternative<PointCloud>(*d)) kind = "PointCloud";
            else if (d && std::holds_alternative<Image>(*d)) kind = "Image";
            else if (d && std::holds_alternative<ImageSet>(*d)) kind = "ImageSet";
            std::printf("viewer \"%s\": %s", vd.name.c_str(), kind);
            if (d) if (const PointCloud* pc = std::get_if<PointCloud>(d))
                std::printf(" (%d cameras, %d points)", pc->SolvedCameras(),
                            pc->TriangulatedPoints());
            std::printf("\n");
        }
        std::printf("\ntotal %.0f ms\n", sms);
        return 0;
    }

    // --- the chain -----------------------------------------------------------
    //
    // Stage indices are explicit because each reads the previous one's port.
    // This is what a script would build; driving it directly keeps the tool
    // independent of the script layer.
    Pipeline p;
    int stage = 0;

    {
        auto det = Registry::Get().Create(detector);
        if (features > 0) SetParam(det.get(), "max_features", double(features));
        p.AddStage(std::move(det), detector.c_str(), {{-1, 0}}, 1, ++stage);
    }
    const int sDetect = stage - 1;

    {
        auto m = Registry::Get().Create(matcher);
        SetParam(m.get(), "chain", 1.0);
        SetParam(m.get(), "window", double(window));
        SetParam(m.get(), "ratio", ratio);
        SetParam(m.get(), "checks", double(checks));
        if (crossCheck >= 0) SetParam(m.get(), "cross_check", double(crossCheck));
        p.AddStage(std::move(m), matcher.c_str(), {{sDetect, 0}}, 1, ++stage);
    }
    const int sMatch = stage - 1;

    // NO align_features HERE, and its absence is deliberate.
    //
    // It was in this chain to put RANSAC's inlier flags on the matches. But it
    // fits a HOMOGRAPHY, which describes a pure camera rotation and nothing
    // else, so on a walk around a building it fits nothing: measured on
    // castle-P19 it kept 20% of matches and reported shifts of 28,000 px on a
    // 1600 px image. relative_pose now runs its own RANSAC against the
    // essential matrix -- the correct model -- and writes those inliers back
    // for build_tracks.
    //
    // Worse than useless, in fact: align_features treats a frame that lands
    // further than its own width from the prediction as a fatal ordering
    // error, which is right for a panorama and wrong here. At a looser ratio
    // test it aborted the entire reconstruction over a frame the essential
    // matrix handled without complaint.
    const int sVerify = sMatch;

    {
        auto rp = Registry::Get().Create("relative_pose");
        SetParam(rp.get(), "fov_deg", fov);
        if (poseMethod >= 0) SetParam(rp.get(), "method", double(poseMethod));
        p.AddStage(std::move(rp), "relative_pose", {{sVerify, 0}}, 1, ++stage);
    }
    const int sPose = stage - 1;

    // fov_deg must match relative_pose's: that stage solves in this geometry
    // and build_tracks fills in the camera slots everything downstream reads.
    auto bt = Registry::Get().Create("build_tracks");
    SetParam(bt.get(), "fov_deg", fov);
    p.AddStage(std::move(bt), "build_tracks", {{sPose, 0}}, 1, ++stage);
    const int sTracks = stage - 1;

    {
        auto r = Registry::Get().Create("rotation_average");
        SetParam(r.get(), "method", double(rotMethod));
        p.AddStage(std::move(r), "rotation_average", {{sTracks, 0}}, 1, ++stage);
    }
    const int sRot = stage - 1;

    {
        auto g = Registry::Get().Create("global_position");
        SetParam(g.get(), "method", double(posMethod));
        p.AddStage(std::move(g), "global_position", {{sRot, 0}}, 1, ++stage);
    }
    const int sPos = stage - 1;

    auto tri = Registry::Get().Create("triangulate");
    if (maxDistance > 0.0) SetParam(tri.get(), "max_distance", maxDistance);
    p.AddStage(std::move(tri), "triangulate", {{sPos, 0}}, 1, ++stage);
    const int sTri = stage - 1;

    auto ba = Registry::Get().Create("bundle_adjust_sfm");
    if (maxDistance > 0.0) SetParam(ba.get(), "max_distance", maxDistance);
    if (baIters > 0) SetParam(ba.get(), "iterations", double(baIters));
    p.AddStage(std::move(ba), "bundle_adjust_sfm", {{sTri, 0}}, 1, ++stage);
    const int sBa1 = stage - 1;

    // A SECOND ROUND, because the first triangulation judged every track
    // against cameras nothing had refined yet. See sfm.tgl: on castle-P19 this
    // takes the cloud from 2889 points to 7659 and the reprojection median
    // from 10.6 px to 1.1. The bench has to match the script or its numbers
    // describe a pipeline nobody runs.
    auto tri2 = Registry::Get().Create("triangulate");
    if (maxDistance > 0.0) SetParam(tri2.get(), "max_distance", maxDistance);
    p.AddStage(std::move(tri2), "triangulate", {{sBa1, 0}}, 1, ++stage);
    const int sTri2 = stage - 1;

    auto ba2 = Registry::Get().Create("bundle_adjust_sfm");
    if (maxDistance > 0.0) SetParam(ba2.get(), "max_distance", maxDistance);
    if (baIters > 0) SetParam(ba2.get(), "iterations", double(baIters));
    p.AddStage(std::move(ba2), "bundle_adjust_sfm", {{sTri2, 0}}, 1, ++stage);
    const int sBundle = stage - 1;

    const auto t0 = std::chrono::steady_clock::now();
    std::string err;
    const bool ok = p.Execute(&s, nullptr, &err);
    const double total = Ms(t0, std::chrono::steady_clock::now());

    // Every stage that ran gets its say, including on failure: the report from
    // the LAST stage that worked is usually what explains the first that did
    // not.
    for (const Stage& st : p.Stages()) {
        if (!st.algo) continue;
        const std::string r = st.Report();
        if (!r.empty()) std::printf("  %s\n", r.c_str());
        else if (!st.valid) std::printf("  %s: did not run\n", st.algoName.c_str());
    }

    // WHERE THE TIME WENT, in descending order.
    //
    // A total says a run took half a minute; it cannot say whether threading
    // the detector would help. Amdahl caps any stage's contribution at its own
    // share, so this listing is what orders the work: a stage at 5% of the
    // runtime is not worth parallelising however many cores it could use.
    //
    // `frames` is how many images the stage mapped across, which is the number
    // a parallel dispatch would divide by. A stage that ran over one frame --
    // an aligner, a reconstruction -- gets no benefit from that kind of
    // parallelism however slow it is, and needs its own inner loop threaded
    // instead. The two cases look identical in a total and completely
    // different here.
    {
        struct Row { std::string name; double ms; int frames; };
        std::vector<Row> rows;
        double sum = 0.0;
        for (const Stage& st : p.Stages()) {
            if (!st.algo || st.lastMs <= 0.0) continue;
            rows.push_back({st.algoName, st.lastMs, st.ranFrames});
            sum += st.lastMs;
        }
        std::sort(rows.begin(), rows.end(),
                  [](const Row& a, const Row& b) { return a.ms > b.ms; });

        if (!rows.empty()) {
            std::printf("\nstage timings (%.0f ms accounted of %.0f ms total)\n",
                        sum, total);
            for (const Row& r : rows) {
                // Built into a named string rather than a ternary on
                // `.c_str()`: the temporary from std::to_string would be dead
                // before printf read it.
                const std::string span = (r.frames > 1)
                                             ? std::to_string(r.frames) + " frames"
                                             : std::string("whole group");
                std::printf("  %-22s %8.0f ms  %5.1f%%  %s\n", r.name.c_str(),
                            r.ms, 100.0 * r.ms / std::max(1.0, sum),
                            span.c_str());
            }
        }
    }
    std::printf("\n");

    if (!ok) {
        std::printf("FAILED: %s\n", err.c_str());
        return 1;
    }

    // --- what came out --------------------------------------------------------
    // BEFORE AND AFTER BUNDLE ADJUSTMENT, because a filter in triangulate can
    // only speak for the cloud it produced. Measured on castle-P19, the
    // distance test rejected nothing while the final extent stayed at 65.9 --
    // the strays were not triangulated out there, they were CREATED by bundle
    // adjustment moving points after the filter had passed them.
    if (const Data* dt = p.Resolve({sTri, 0}, &s)) {
        if (const PointCloud* tri = std::get_if<PointCloud>(dt)) {
            Vec3 tlo{1e30, 1e30, 1e30}, thi{-1e30, -1e30, -1e30};
            for (const Track& t : tri->tracks) {
                if (!t.hasPoint) continue;
                tlo = Vec3{std::min(tlo.x, t.point.x), std::min(tlo.y, t.point.y),
                           std::min(tlo.z, t.point.z)};
                thi = Vec3{std::max(thi.x, t.point.x), std::max(thi.y, t.point.y),
                           std::max(thi.z, t.point.z)};
            }
            if (tri->TriangulatedPoints() > 0)
                std::printf("extent before bundle: %.2f x %.2f x %.2f\n",
                            thi.x - tlo.x, thi.y - tlo.y, thi.z - tlo.z);
        }
    }

    const Data* d = p.Resolve({sBundle, 0}, &s);
    const PointCloud* pc = d ? std::get_if<PointCloud>(d) : nullptr;
    if (!pc) {
        std::printf("no reconstruction\n");
        return 1;
    }

    // The extent says whether the geometry is plausible at a glance. A
    // reconstruction collapsed to a point, or one with a camera flung a
    // thousand times further than the rest, shows up here before anything
    // else.
    Vec3 lo{1e30, 1e30, 1e30}, hi{-1e30, -1e30, -1e30};
    for (const Track& t : pc->tracks) {
        if (!t.hasPoint) continue;
        lo = Vec3{std::min(lo.x, t.point.x), std::min(lo.y, t.point.y),
                  std::min(lo.z, t.point.z)};
        hi = Vec3{std::max(hi.x, t.point.x), std::max(hi.y, t.point.y),
                  std::max(hi.z, t.point.z)};
    }

    std::printf("reconstruction: %d of %d cameras, %d of %d points\n",
                pc->SolvedCameras(), int(pc->cameras.size()),
                pc->TriangulatedPoints(), int(pc->tracks.size()));
    if (pc->TriangulatedPoints() > 0) {
        std::printf("extent: %.2f x %.2f x %.2f\n",
                    hi.x - lo.x, hi.y - lo.y, hi.z - lo.z);

        // The same extent between the 2nd and 98th percentiles. A raw min-max
        // is decided by its two most extreme points, so a handful of strays
        // can report a reconstruction six times longer than it is -- and the
        // difference between the two numbers says which problem you have.
        auto spread = [&](int axis) {
            std::vector<double> v;
            for (const Track& t : pc->tracks) {
                if (!t.hasPoint) continue;
                v.push_back(axis == 0 ? t.point.x
                                      : (axis == 1 ? t.point.y : t.point.z));
            }
            if (v.size() < 10) return 0.0;
            const size_t a = size_t(0.02 * double(v.size() - 1));
            const size_t b = size_t(0.98 * double(v.size() - 1));
            std::nth_element(v.begin(), v.begin() + long(a), v.end());
            const double loV = v[a];
            std::nth_element(v.begin(), v.begin() + long(b), v.end());
            return v[b] - loV;
        };
        std::printf("extent (2-98%%): %.2f x %.2f x %.2f\n",
                    spread(0), spread(1), spread(2));
    }

    // WHICH FRAMES BUILT WHICH PART OF THE CLOUD.
    //
    // The viewport shows the reconstruction in two clumps, and a bounding box
    // cannot distinguish the two readings that matter: the scene genuinely has
    // two halves (a courtyard does), or one half is a DUPLICATE of the other,
    // placed wrongly because the chain broke. Those need opposite responses.
    //
    // Splitting the points at the median of the longest axis and reporting
    // which frames observe each side separates them. If the two sides are
    // observed by disjoint runs of frames, the scene really has two parts. If
    // the SAME frames observe both, part of the cloud has been duplicated.
    if (pc->TriangulatedPoints() > 0) {
        const Vec3 ext{hi.x - lo.x, hi.y - lo.y, hi.z - lo.z};
        const int axis = (ext.x >= ext.y && ext.x >= ext.z) ? 0
                         : (ext.y >= ext.z)                 ? 1
                                                            : 2;
        auto coord = [axis](const Vec3& v) {
            return axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
        };
        std::vector<double> vals;
        for (const Track& t : pc->tracks)
            if (t.hasPoint) vals.push_back(coord(t.point));
        std::nth_element(vals.begin(), vals.begin() + long(vals.size() / 2),
                         vals.end());
        const double mid = vals[vals.size() / 2];

        std::vector<int> loSeen(pc->cameras.size(), 0);
        std::vector<int> hiSeen(pc->cameras.size(), 0);
        for (const Track& t : pc->tracks) {
            if (!t.hasPoint) continue;
            const bool isHi = coord(t.point) > mid;
            for (const Observation& o : t.obs)
                if (o.frame >= 0 && o.frame < int(pc->cameras.size()))
                    (isHi ? hiSeen : loSeen)[size_t(o.frame)]++;
        }
        std::printf("clump A (below median of axis %d) frames:", axis);
        for (size_t i = 0; i < loSeen.size(); ++i)
            if (loSeen[i] > 0) std::printf(" %zu", i);
        std::printf("\nclump B (above) frames:");
        for (size_t i = 0; i < hiSeen.size(); ++i)
            if (hiSeen[i] > 0) std::printf(" %zu", i);
        std::printf("\n");
    }

    // TOTAL TURN AROUND THE SEQUENCE, summed over the consecutive relative
    // rotations.
    //
    // A capture that walks around a subject and returns near its start has
    // turned through about 360 degrees, and that is a ground truth available
    // WITHOUT any ground truth file: it comes from knowing how the photographs
    // were taken. It caught a real defect that every synthetic fixture missed
    // -- on castle-P19 the eight-point path totals 391 degrees, near enough one
    // circuit, where the five-point path totals 540, half a turn too much, so
    // its per-pair rotations are inflated on real data even though the two
    // agree on synthetic pairs.
    {
        double turn = 0.0;
        int n = 0;
        for (const PointCloud::ViewEdge& e : pc->edges) {
            if (e.j != e.i + 1) continue;   // consecutive pairs only
            turn += Mat3{}.AngleTo(e.R) * 180.0 / 3.14159265358979;
            ++n;
        }
        if (n > 0)
            std::printf("total turn over %d consecutive pairs: %.0f deg "
                        "(a closed walk-around is near 360)\n", n, turn);
    }

    // WHERE EACH CAMERA ACTUALLY ENDED UP, relative to the camera centroid and
    // in units of the camera cluster's radius.
    //
    // The clump analysis above splits the POINTS, which answers "is the
    // structure duplicated" but not "which cameras are misplaced". When the
    // viewport shows a detached group, this is the direct read: a camera far
    // from the centroid, or one whose distance jumps away from its neighbours'
    // is the one that broke loose.
    if (pc->SolvedCameras() > 0) {
        Vec3 cm{0, 0, 0};
        int n = 0;
        for (const Camera& c : pc->cameras) {
            if (!c.solved) continue;
            cm = cm + c.Center();
            ++n;
        }
        cm = cm * (1.0 / double(n));
        double rad = 0.0;
        for (const Camera& c : pc->cameras)
            if (c.solved) rad = std::max(rad, (c.Center() - cm).Norm());
        if (rad < 1e-9) rad = 1.0;

        std::printf("camera positions (centroid-relative, in cluster radii):\n");
        for (size_t i = 0; i < pc->cameras.size(); ++i) {
            if (!pc->cameras[i].solved) { std::printf("  %2zu: ---\n", i); continue; }
            const Vec3 d = (pc->cameras[i].Center() - cm) * (1.0 / rad);
            // The viewing direction too: a walk around a subject should show
            // the cameras all looking inward, and a group that broke loose
            // often shows as a run of frames aiming the wrong way.
            const Vec3 fwd = pc->cameras[i].R.Transpose() * Vec3{0, 0, 1};
            const double inward = (d.Norm() > 1e-9)
                                      ? -fwd.Dot(d * (1.0 / d.Norm()))
                                      : 0.0;
            std::printf("  %2zu: %6.2f %6.2f %6.2f   inward %+.2f\n",
                        i, d.x, d.y, d.z, inward);
        }
    }

    // Camera spacing: a walk-around should give a smooth progression, and a
    // wild outlier is a frame that registered against the wrong neighbours.
    double minGap = 1e30, maxGap = 0.0;
    int gaps = 0;
    for (size_t i = 1; i < pc->cameras.size(); ++i) {
        if (!pc->cameras[i].solved || !pc->cameras[i - 1].solved) continue;
        const double g = (pc->cameras[i].Center() -
                          pc->cameras[i - 1].Center()).Norm();
        minGap = std::min(minGap, g);
        maxGap = std::max(maxGap, g);
        ++gaps;
    }
    if (gaps > 0)
        std::printf("consecutive camera spacing: %.3f to %.3f (ratio %.1f)\n",
                    minGap, maxGap, maxGap / std::max(1e-9, minGap));

    // EVERY gap, not just the extremes, because a min and a max cannot tell a
    // reconstruction that drifts from one that SPLITS. A split shows as one
    // enormous step with ordinary steps either side of it -- the two halves
    // are each fine and simply placed wrong relative to each other -- and that
    // is a different defect from a chain whose spacing wanders throughout.
    if (gaps > 0) {
        std::printf("gaps:");
        for (size_t i = 1; i < pc->cameras.size(); ++i) {
            if (!pc->cameras[i].solved || !pc->cameras[i - 1].solved) {
                std::printf("  %2zu:----", i);
                continue;
            }
            const double g = (pc->cameras[i].Center() -
                              pc->cameras[i - 1].Center()).Norm();
            std::printf("  %2zu:%.2f", i, g);
        }
        std::printf("\n");
    }

    std::printf("\ntotal %.0f ms\n", total);
    return 0;
}
