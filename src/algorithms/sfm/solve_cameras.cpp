// solve_cameras — matched frames in, a reconstruction out, with the focal
// length found rather than asked for.
//
// THE CHAIN IT RUNS is the one sfm.tgl used to spell out stage by stage:
//
//   relative_pose -> build_tracks -> rotation_average -> global_position
//     -> triangulate -> bundle_adjust_sfm -> triangulate -> bundle_adjust_sfm
//
// Those stages still exist and can still be chained by hand, which is how to
// compare one of them against an alternative. This stage exists for the other
// use: a capture that should simply reconstruct, with nothing to tune.
//
// WHY IT IS A STAGE AND NOT A SCRIPT LINE: THE FOCAL IS A LOOP. Everything
// downstream of relative_pose inherits its focal length, and a wrong one is
// not a small error -- on a video walked around a toy cat, solving at 42
// degrees where the lens was about 34 made the rotations fail to close the
// circle by 31 degrees, and the solve broke into pieces. Bundle adjustment
// MEASURES the focal (it solves for it against reprojection error), but only
// after everything else has been solved with the old one. So: solve, read the
// focal bundle adjustment settled on, and solve again with it -- until it stops
// changing. Measured on that video, starting from 58 degrees:
//
//   round   solved at   bundle adjustment says   largest camera jump
//       1        58.0                   42.4                    1.21
//       2        42.4                   38.7                    0.88
//       3        38.7                   36.2                    1.08
//       4        36.2                   33.7                    0.27
//       5        33.7                   33.8                    --
//
// The first round starts from relative_pose's own estimate (fov_deg 0), which
// is usually within ten percent, so one or two rounds typically suffice.
//
// THE ROUND KEPT is the one with the most observations reprojecting within
// 2 px -- consistent AND complete -- which is not always the last: a round can
// land a focal that is right while its solve came out slightly worse.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "../../algo_util/view_graph.h"
#include "../../core/algorithm.h"
#include "../../core/pipeline.h"
#include "../../script/value.h"

namespace tglab {
namespace {

constexpr double kDeg = 180.0 / 3.14159265358979;

// The horizontal field of view of the cloud's first solved camera, in degrees.
double FovOf(const PointCloud& pc) {
    for (const Camera& c : pc.cameras)
        if (c.solved && c.focal > 0.0 && c.width > 0)
            return 2.0 * std::atan(0.5 * c.width / c.focal) * kDeg;
    return 0.0;
}

// Observations that reproject within `px` of where they were seen: how much
// of the data the reconstruction actually explains.
long long GoodObservations(const PointCloud& pc, double px) {
    long long good = 0;
    for (const Track& t : pc.tracks) {
        if (!t.hasPoint) continue;
        for (const Observation& o : t.obs) {
            if (o.frame < 0 || o.frame >= int(pc.cameras.size())) continue;
            const Camera& c = pc.cameras[size_t(o.frame)];
            if (!c.solved) continue;
            double u = 0.0, v = 0.0;
            if (!c.Project(t.point, &u, &v)) continue;
            if (std::hypot(u - o.x, v - o.y) <= px) ++good;
        }
    }
    return good;
}

// THE INNER CHAIN'S PROGRESS, mapped into this stage's: the inner pipeline
// says which of its eight steps is running (Set) and the step how far it has
// got (SetWithin), and this turns both into one fraction of the whole solve.
// Steps are weighted by their share of the time, measured on a 1576-frame room
// scan, so the bar moves at an even pace rather than racing through relative
// pose and then sitting on global positioning. Each round runs the bar from
// the start again (the round count is a maximum, usually not reached), with
// the round in the label.
struct InnerProgress : Progress {
    const AlgorithmBase* owner = nullptr;
    int round = 0, rounds = 1;
    int step = 0;
    std::string stepName, label;
    static constexpr double kWeight[8] = {2.1, 9.7, 0.1, 51.0, 0.9, 25.0, 0.9, 25.0};

    void Set(int done, int total, const char* what) override {
        if (total != 8) return;   // not the step count: ignore
        step = std::clamp(done, 0, 7);
        if (what) stepName = what;
        Report(0.0, stepName.c_str());
    }
    void SetWithin(double f, const char* what) override { Report(f, what); }

    void Report(double f, const char* what) {
        double before = 0.0, all = 0.0;
        for (int i = 0; i < 8; ++i) {
            if (i < step) before += kWeight[i];
            all += kWeight[i];
        }
        const double inRound = (before + kWeight[step] * std::clamp(f, 0.0, 1.0)) / all;
        label = std::string(what ? what : "") + " (round " + std::to_string(round + 1) + ")";
        owner->GroupProgress(inRound, label.c_str());
    }
};

class SolveCameras : public AlgorithmBase {
public:
    const char* Name()     const override { return "solve_cameras"; }
    const char* Category() const override { return "sfm"; }

    PortList Inputs() const override {
        return {{"src", DataType::ImageSet, FormatSpec::Any, ShapeSpec::Any}};
    }
    PortList Outputs() const override {
        return {{"out", DataType::PointCloud, FormatSpec::Any, ShapeSpec::Any}};
    }

    void RunCPU(RunCtx&) override {}
    bool IsReconstruct() const override { return true; }
    // Keypoints and focal are in image pixels; nothing rescales them.
    ProxyBehaviour Proxy() const override { return ProxyBehaviour::Never; }
    bool HasGPU() const override { return false; }

    bool RunReconstruct(const std::vector<Image>* images, PointCloud* cloud,
                        std::string* err) override {
        if (!images || images->size() < 2) {
            *err = "solve_cameras: needs a matched group -- solve_cameras(pairs)";
            return false;
        }
        const int n = int(images->size());
        const int rounds = std::max(1, int(m_rounds));
        double fov = double(m_fovDeg);   // 0: relative_pose estimates round 1's

        struct Round { double in = 0, out = 0; long long good = 0; int pts = 0; };
        m_time.clear();
        std::vector<Round> log;
        PointCloud best;
        long long bestGood = -1;
        std::string lastErr;

        for (int r = 0; r < rounds; ++r) {
            std::vector<Data> src;
            {
                ImageSet set;
                for (const Image& im : *images) set.images.push_back(im.Clone());
                set.shape = Shape{{{"frame", n}}};
                src.push_back(Data{std::move(set)});
            }
            Pipeline p;
            auto add = [&](const char* name, int input,
                           std::initializer_list<std::pair<const char*, double>> ps) {
                auto a = Registry::Get().Create(name);
                if (!a) return -1;
                for (const auto& [k, v] : ps)
                    if (ParamBase* pb = a->FindParam(k)) {
                        std::string e;
                        pb->SetFromScript(Value(v), &e);
                    }
                return p.AddStage(std::move(a), name, {{input, 0}}, 1, 0);
            };
            const double dist = double(m_maxDistance);
            int s = add("relative_pose", -1, {{"fov_deg", fov},
                                              {"min_inliers", double(int(m_minInliers))},
                                              {"method", double(int(m_method))}});
            s = add("build_tracks", s, {{"min_length", double(int(m_minLength))}});
            s = add("rotation_average", s, {});
            s = add("global_position", s, {});
            s = add("triangulate", s, {{"max_distance", dist}});
            s = add("bundle_adjust_sfm", s, {{"max_distance", dist}});
            s = add("triangulate", s, {{"max_distance", dist}});
            s = add("bundle_adjust_sfm", s, {{"max_distance", dist}});
            if (s < 0) { *err = "solve_cameras: a stage it runs is not registered"; return false; }

            // The inner chain is cancelled with this run (SetGroupCancel), and
            // reports its progress through this stage's (InnerProgress).
            InnerProgress inner;
            inner.owner = this;
            inner.round = r;
            inner.rounds = 1;   // the bar runs per round; the label names it
            std::string e;
            if (!p.Execute(&src, nullptr, &e, nullptr, ExecMode::Auto, nullptr,
                           GroupCancelToken(), GroupProgressSink() ? &inner : nullptr)) {
                if (GroupCancelled()) { *err = "cancelled"; return false; }
                lastErr = e;
                break;
            }
            const Data* d = p.Resolve({s, 0}, &src);
            const PointCloud* pc = d ? std::get_if<PointCloud>(d) : nullptr;
            if (!pc) { lastErr = "no reconstruction came out"; break; }

            // What round 1 actually solved at, when it was estimated: read
            // back off relative_pose's sidecar on its output frames.
            double in = fov;
            if (in <= 0.0)
                if (const Data* rd = p.Resolve({0, 0}, &src))
                    if (const ImageSet* rs = std::get_if<ImageSet>(rd))
                        for (const Image& im : rs->images)
                            if (const RelativePoseSidecar* rp = RelativePosesOf(im)) {
                                in = rp->fovDeg;
                                break;
                            }
            for (const Stage& st : p.Stages()) {
                auto it = std::find_if(m_time.begin(), m_time.end(),
                                       [&](const auto& t) { return t.first == st.algoName; });
                if (it == m_time.end()) m_time.push_back({st.algoName, st.lastMs});
                else it->second += st.lastMs;
            }
            Round rd;
            rd.in = in;
            rd.out = FovOf(*pc);
            rd.good = GoodObservations(*pc, 2.0);
            rd.pts = pc->TriangulatedPoints();
            log.push_back(rd);
            // A round that explains LESS than the best so far means the focal
            // is moving away rather than settling -- on a selfie video it crept
            // 41.9 -> 43.2 -> 44.7 -> 45.7 with every round worse than the
            // first. Another round would only walk further, so stop.
            if (rd.good <= bestGood) break;
            {
                bestGood = rd.good;
                best = *pc;
                m_stageNotes.clear();
                for (const Stage& st : p.Stages())
                    if (st.algo) m_stageNotes.push_back(st.Report());
            }

            // Converged: the focal bundle adjustment measured is the one this
            // round solved with.
            if (rd.out <= 0.0 || std::fabs(rd.out - in) < 0.02 * in) break;
            fov = rd.out;
        }

        if (bestGood < 0) {
            *err = "solve_cameras: " + (lastErr.empty() ? std::string("nothing solved") : lastErr);
            return false;
        }
        *cloud = std::move(best);

        char buf[160];
        std::snprintf(buf, sizeof buf, "solve_cameras: %d round%s;",
                      int(log.size()), log.size() == 1 ? "" : "s");
        m_note = buf;
        for (size_t i = 0; i < log.size(); ++i) {
            std::snprintf(buf, sizeof buf, " [%zu] fov %.1f -> %.1f, %d points, %lld obs within 2 px",
                          i + 1, log[i].in, log[i].out, log[i].pts, log[i].good);
            m_note += buf;
        }
        std::snprintf(buf, sizeof buf, "; kept the one explaining the most (fov %.1f deg)",
                      FovOf(*cloud));
        m_note += buf;
        m_note += "; time";
        for (const auto& [name, ms] : m_time) {
            std::snprintf(buf, sizeof buf, " %s %.1f s,", name.c_str(), ms * 1e-3);
            m_note += buf;
        }
        if (m_note.back() == ',') m_note.pop_back();
        return true;
    }

    // Its own summary, then the kept round's stage reports, one per line --
    // the detail the individual stages would have shown in a hand-built chain.
    std::string RunReport() const override {
        std::string s = m_note;
        for (const std::string& r : m_stageNotes)
            if (!r.empty()) s += "\n  " + r;
        return s;
    }

private:
    Param<float> m_fovDeg{this, "fov_deg", 0.0f, 0.0f, 150.0f,
        {.help = "Starting horizontal field of view in degrees. 0 (the "
                 "default) estimates it from the matches; either way it is "
                 "then refined by solving again with the focal bundle "
                 "adjustment measures."}};
    Param<int> m_rounds{this, "rounds", 5, 1, 10,
        {.help = "Most solves to run while the focal settles. Stops early "
                 "once bundle adjustment agrees with the focal it was given "
                 "to within 2%."}};
    Param<int> m_minInliers{this, "min_inliers", 20, 8, 500,
        {.help = "relative_pose's: fewest essential-matrix inliers for a pair "
                 "to count. Raise it when matching far-apart frames "
                 "(match_ann's revisit), where chance consensus is common."}};
    Param<int> m_method{this, "method", 0, 0, 1,
        {.help = "relative_pose's estimator: 0 eight-point, 1 five-point."}};
    Param<int> m_minLength{this, "min_length", 3, 2, 20,
        {.help = "build_tracks': fewest frames a track must appear in."}};
    Param<float> m_maxDistance{this, "max_distance", 3.0f, 1.0f, 1000.0f,
        {.help = "triangulate's and bundle adjustment's: how far out, in "
                 "camera-ring radii, a point may be. 3 suits a walk around a "
                 "subject; open scenes want more."}};

    std::string              m_note;
    std::vector<std::string> m_stageNotes;
    std::vector<std::pair<std::string, double>> m_time;   // ms per inner stage, all rounds
};

REGISTER_ALGORITHM(SolveCameras);

}  // namespace
}  // namespace tglab
