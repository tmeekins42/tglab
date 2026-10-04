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
#include "../src/algo_util/linalg.h"
#include "../src/algo_util/splat_raster.h"
#include "../src/algo_util/splat_reflect.h"
#include "../src/algo_util/view_graph.h"
#include "../src/core/algorithm.h"
#include "../src/core/image.h"
#include "../src/core/image_io.h"
#include "../src/core/pipeline.h"
#include "../src/core/shape.h"
#include "../src/core/video_io.h"
#include "../src/script/interp.h"
#include "../src/script/parser.h"
#include "../src/script/value.h"
#include "../src/gpu/compute.h"

#include <d3d12.h>
// windows.h, via d3d12.h, defines near, far and small as macros -- the
// last as `char`, which collides with this file's own variable of that name.
#ifdef small
#undef small
#endif
#ifdef near
#undef near
#endif
#ifdef far
#undef far
#endif

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

// How THICK the cloud is perpendicular to its dominant plane.
//
// WHY THIS IS THE RIGHT MEASUREMENT FOR A "DOUBLE WALL". A facade is flat, so
// a correct reconstruction of one is thin in exactly one direction -- the
// plane's normal. When triangulation places the same wall at two depths, the
// cloud gains a second sheet and that one direction thickens, while nothing
// else about the cloud changes. Point counts, reprojection error and rotation
// residuals are all blind to it: both sheets reproject perfectly, because each
// was placed to.
//
// Measured by PCA. The smallest eigenvalue's direction is the plane normal,
// and the spread along it is the thickness. Reported as a fraction of the
// in-plane extent so it is comparable across scenes and scales -- a flat wall
// is a few percent, and two sheets a plane-separation apart is much more.
//
// Percentiles rather than min/max, for the reason the viewer's framing uses
// them: a handful of strays decide an extremum and say nothing about the bulk.
void ReportPlanarity(const PointCloud& pc) {
    std::vector<Vec3> p;
    p.reserve(pc.tracks.size());
    for (const Track& t : pc.tracks) if (t.hasPoint) p.push_back(t.point);
    if (p.size() < 32) return;

    Vec3 c{0, 0, 0};
    for (const Vec3& v : p) { c.x += v.x; c.y += v.y; c.z += v.z; }
    c.x /= double(p.size()); c.y /= double(p.size()); c.z /= double(p.size());

    // Covariance, then Jacobi for its eigenvectors. Three iterations of cyclic
    // sweeps is ample for a symmetric 3x3.
    double m[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    for (const Vec3& v : p) {
        const double d[3] = {v.x - c.x, v.y - c.y, v.z - c.z};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) m[i][j] += d[i] * d[j];
    }
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) m[i][j] /= double(p.size());

    double ev[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    for (int sweep = 0; sweep < 12; ++sweep) {
        int q = 0, r = 1;
        double best = std::fabs(m[0][1]);
        if (std::fabs(m[0][2]) > best) { best = std::fabs(m[0][2]); q = 0; r = 2; }
        if (std::fabs(m[1][2]) > best) { best = std::fabs(m[1][2]); q = 1; r = 2; }
        if (best < 1e-18) break;

        const double theta = 0.5 * std::atan2(2.0 * m[q][r], m[q][q] - m[r][r]);
        const double cs = std::cos(theta), sn = std::sin(theta);
        for (int k = 0; k < 3; ++k) {
            const double a = m[k][q], b = m[k][r];
            m[k][q] = cs * a + sn * b;  m[k][r] = -sn * a + cs * b;
        }
        for (int k = 0; k < 3; ++k) {
            const double a = m[q][k], b = m[r][k];
            m[q][k] = cs * a + sn * b;  m[r][k] = -sn * a + cs * b;
            const double u = ev[k][q], w = ev[k][r];
            ev[k][q] = cs * u + sn * w; ev[k][r] = -sn * u + cs * w;
        }
    }

    int axis = 0;
    for (int i = 1; i < 3; ++i) if (m[i][i] < m[axis][axis]) axis = i;
    const Vec3 n{ev[0][axis], ev[1][axis], ev[2][axis]};

    // Spread along the normal, and along the widest in-plane direction.
    int wide = 0;
    for (int i = 1; i < 3; ++i) if (m[i][i] > m[wide][wide]) wide = i;
    const Vec3 w{ev[0][wide], ev[1][wide], ev[2][wide]};

    std::vector<double> dn, dw;
    dn.reserve(p.size()); dw.reserve(p.size());
    for (const Vec3& v : p) {
        const double d[3] = {v.x - c.x, v.y - c.y, v.z - c.z};
        dn.push_back(d[0] * n.x + d[1] * n.y + d[2] * n.z);
        dw.push_back(d[0] * w.x + d[1] * w.y + d[2] * w.z);
    }
    std::sort(dn.begin(), dn.end());
    std::sort(dw.begin(), dw.end());
    auto span = [](const std::vector<double>& v) {
        return v[size_t(0.98 * double(v.size() - 1))] -
               v[size_t(0.02 * double(v.size() - 1))];
    };
    const double thick = span(dn), width = span(dw);
    if (width <= 1e-12) return;

    std::printf("\n  planarity: thickness %.4f across a %.4f extent "
                "(%.1f%% -- a flat wall is a few percent)",
                thick, width, 100.0 * thick / width);

    // WHICH SIDE OF THE FACADE THE CAMERAS ARE ON.
    //
    // A reconstruction can be mirrored about the dominant plane and still
    // satisfy every reprojection test, because a mirrored scene photographed
    // by mirrored cameras produces identical images. The fix for that is
    // cheirality -- points must be IN FRONT of the cameras -- and the way to
    // see whether it held is to ask where the cameras sit relative to the
    // wall they photographed.
    //
    // They should all be on ONE side. Cameras straddling the plane means part
    // of the structure was reconstructed behind them, which is what a
    // mirrored surface looks like from the outside.
    {
        int front = 0, back = 0;
        for (const Camera& cam : pc.cameras) {
            if (!cam.solved) continue;
            const Vec3 e = cam.Center();
            const double d[3] = {e.x - c.x, e.y - c.y, e.z - c.z};
            (d[0] * n.x + d[1] * n.y + d[2] * n.z >= 0.0 ? front : back)++;
        }
        std::printf("\n  cameras: %d in front of the dominant plane, %d behind%s",
                    front, back,
                    (front > 0 && back > 0)
                        ? "  <-- STRADDLING: part of the scene is behind a camera"
                        : "");
    }

    // A SIDE WALL'S SLANT, and whether it runs the right way.
    //
    // The facade dominates the PCA, so a side wall returning at an angle is
    // invisible in the aggregate thickness -- it is a minority of the points
    // and it simply widens the distribution. Measuring it needs the edges of
    // the cloud looked at separately.
    //
    // Split along the widest in-plane direction and ask, in each outer fifth,
    // how the depth-from-the-facade trends as you move outward. A real side
    // wall recedes AWAY from the cameras. One reconstructed mirrored comes
    // TOWARD them, which is what "180 degrees off" looks like.
    {
        const double loW = dw[size_t(0.02 * double(dw.size() - 1))];
        const double hiW = dw[size_t(0.98 * double(dw.size() - 1))];
        const double q = (hiW - loW) / 5.0;

        // Camera side of the plane: the direction "toward the cameras".
        double camSide = 0.0;
        for (const Camera& cam : pc.cameras) {
            if (!cam.solved) continue;
            const Vec3 e = cam.Center();
            const double d[3] = {e.x - c.x, e.y - c.y, e.z - c.z};
            camSide += d[0] * n.x + d[1] * n.y + d[2] * n.z;
        }
        const double sgn = camSide >= 0.0 ? 1.0 : -1.0;

        auto edgeTrend = [&](double from, double to, const char* label) {
            std::vector<double> inner, outer;
            for (const Vec3& v : p) {
                const double d[3] = {v.x - c.x, v.y - c.y, v.z - c.z};
                const double along = d[0] * w.x + d[1] * w.y + d[2] * w.z;
                if (along < std::min(from, to) || along > std::max(from, to))
                    continue;
                const double depth =
                    sgn * (d[0] * n.x + d[1] * n.y + d[2] * n.z);
                const double t = std::fabs(along - from) / std::fabs(to - from);
                (t < 0.5 ? inner : outer).push_back(depth);
            }
            if (inner.size() < 30 || outer.size() < 30) return;
            std::sort(inner.begin(), inner.end());
            std::sort(outer.begin(), outer.end());
            const double mi = inner[inner.size() / 2];
            const double mo = outer[outer.size() / 2];
            // Positive depth is toward the cameras, so a receding wall has
            // the outer half FURTHER from them: mo < mi.
            std::printf("\n  %s edge: %s (%.3f -> %.3f toward the cameras)",
                        label,
                        mo < mi ? "recedes away, as a side wall should"
                                : "comes TOWARD the cameras  <-- suspect",
                        mi, mo);
        };

        // "A" and "B" rather than left and right: the PCA's widest axis has
        // an arbitrary SIGN, so which end is which has nothing to do with
        // what a viewer shows. Calling them left and right invited exactly
        // the mistake it caused -- the viewer was mirroring the scene, so the
        // two disagreed, and the probe's "left edge" was the screen's right.
        edgeTrend(loW + q, loW, "edge A");
        edgeTrend(hiW - q, hiW, "edge B");
    }

    // ONE THICK SHEET OR TWO THIN ONES? The thickness alone cannot tell them
    // apart, and they mean completely different things: a thick sheet is
    // noise, two sheets is the same surface triangulated at two depths.
    //
    // A histogram along the normal answers it. Two sheets show as two peaks
    // with a gap between; noise shows as one mode. Reported as the fraction
    // of points in the LARGEST bin -- with a genuine split, the bulk divides
    // between two well-separated bins and neither dominates.
    {
        const int kBins = 24;
        std::vector<int> hist(size_t(kBins), 0);
        const double lo = dn[size_t(0.02 * double(dn.size() - 1))];
        const double span2 = thick > 1e-12 ? thick : 1.0;
        for (double v : dn) {
            int b = int((v - lo) / span2 * double(kBins));
            b = std::max(0, std::min(kBins - 1, b));
            ++hist[size_t(b)];
        }

        // Peaks: a bin holding more than 6% of the points and at least as
        // many as both neighbours. Two of those, well apart, is a double wall.
        const int total = int(dn.size());
        int peaks = 0, firstPeak = -1, lastPeak = -1;
        for (int b = 0; b < kBins; ++b) {
            const int h = hist[size_t(b)];
            if (double(h) < 0.06 * double(total)) continue;
            const int l = (b > 0) ? hist[size_t(b - 1)] : 0;
            const int r = (b < kBins - 1) ? hist[size_t(b + 1)] : 0;
            if (h >= l && h >= r) {
                ++peaks;
                if (firstPeak < 0) firstPeak = b;
                lastPeak = b;
            }
        }
        std::printf("\n  depth profile: %d peak%s", peaks,
                    peaks == 1 ? "" : "s");
        if (peaks >= 2) {
            std::printf(" separated by %.1f%% of the thickness -- TWO SHEETS, "
                        "not one noisy surface",
                        100.0 * double(lastPeak - firstPeak) / double(kBins));

            // WHICH CAMERAS SEE WHICH SHEET, which distinguishes the two
            // possible causes. If each sheet is observed by its own subset of
            // frames, the cameras disagree about where the wall is and the
            // fault is in the POSES. If both sheets are seen by all of them,
            // the poses agree and the split is in the correspondences -- the
            // same wall matched to itself one brick course over.
            const double mid = lo + span2 * (double(firstPeak + lastPeak + 1) *
                                             0.5 / double(kBins));
            std::vector<int> nearCam, farCam;
            nearCam.assign(pc.cameras.size(), 0);
            farCam.assign(pc.cameras.size(), 0);

            for (const Track& t : pc.tracks) {
                if (!t.hasPoint) continue;
                const double d[3] = {t.point.x - c.x, t.point.y - c.y,
                                     t.point.z - c.z};
                const double v = d[0] * n.x + d[1] * n.y + d[2] * n.z;
                for (const Observation& o : t.obs) {
                    if (o.frame < 0 || o.frame >= int(pc.cameras.size()))
                        continue;
                    (v < mid ? nearCam : farCam)[size_t(o.frame)]++;
                }
            }

            int split = 0;
            for (size_t i = 0; i < pc.cameras.size(); ++i) {
                const int a = nearCam[i], b = farCam[i];
                if (a + b < 20) continue;
                const double frac = double(std::min(a, b)) / double(a + b);
                if (frac > 0.15) ++split;      // sees both sheets substantially
            }
            std::printf("\n  sheet membership: %d of %d cameras observe BOTH "
                        "sheets%s", split, int(pc.cameras.size()),
                        split > int(pc.cameras.size()) / 2
                            ? " -- the poses agree, so the split is in the "
                              "CORRESPONDENCES"
                            : " -- the cameras disagree, so the split is in "
                              "the POSES");
        }
    }
}

// --dump for a cloud with Gaussians: renders it from NOVEL viewpoints, the
// middle camera swung around the scene centre, and saves each as a PNG.
// Training views can hide floaters by construction -- the optimiser put them
// exactly where those views wanted them -- so seeing them needs somewhere
// else to look from, which is what the viewer's orbit does interactively.
// --cameras: the camera path, one line per camera -- centre relative to the
// centroid in units of the ring radius, heading turned since camera 0, and the
// step from the previous camera. A walk around a subject should read as one
// smooth ring with the heading advancing steadily; a solve that has broken
// into pieces shows as jumps in the step or the heading.
// WHERE THE CHAIN IS THIN: for each point in the clip, how many triangulated
// tracks were seen both before and after it. A sequence that came out as
// several copies of the subject has a cut that few tracks cross -- the two
// sides are then free to sit anywhere relative to each other -- and this
// says where, and how thin.
static void PrintWeakCuts(const PointCloud& pc) {
    const int n = int(pc.cameras.size());
    if (n < 3) return;
    // Counted twice: every track build_tracks made, and those still
    // triangulated. A split shows as the second thin where the first is not
    // -- tracks that crossed it existed, but did not fit the solved poses.
    for (int pass = 0; pass < 2; ++pass) {
        std::vector<int> d(size_t(n) + 1, 0);
        for (const Track& t : pc.tracks) {
            if ((pass == 1 && !t.hasPoint) || t.obs.empty()) continue;
            int lo = n, hi = -1;
            for (const Observation& o : t.obs) { lo = std::min(lo, o.frame); hi = std::max(hi, o.frame); }
            if (lo < hi && lo >= 0 && hi < n) { ++d[size_t(lo)]; --d[size_t(hi)]; }
        }
        std::vector<std::pair<int, int>> cuts;   // (tracks crossing, cut after frame k)
        int run = 0;
        for (int k = 0; k + 1 < n; ++k) { run += d[size_t(k)]; cuts.push_back({run, k}); }
        std::vector<std::pair<int, int>> sorted = cuts;
        std::sort(sorted.begin(), sorted.end());
        std::printf("  thinnest cuts, %s tracks (crossing@after frame):",
                    pass == 0 ? "all" : "triangulated");
        for (size_t i = 0; i < std::min<size_t>(6, sorted.size()); ++i)
            std::printf(" %d@%d", sorted[i].first, sorted[i].second);
        double mean = 0;
        for (const auto& c : cuts) mean += c.first;
        std::printf("; mean %.0f\n", mean / double(std::max<size_t>(1, cuts.size())));
    }
}

static void PrintCameras(const PointCloud& pc) {
    PrintWeakCuts(pc);
    Vec3 cm{0, 0, 0};
    int n = 0;
    for (const Camera& c : pc.cameras)
        if (c.solved) { cm = cm + c.Center(); ++n; }
    if (n == 0) return;
    cm = cm * (1.0 / double(n));
    double rad = 0.0;
    for (const Camera& c : pc.cameras)
        if (c.solved) rad += (c.Center() - cm).Norm();
    rad = rad > 1e-12 ? rad / double(n) : 1.0;
    // WHERE EACH CAMERA IS LOOKING: its centre plus its viewing axis times
    // the median depth of the points it observes. An orbit of one object
    // gives one cluster of these; a solve that split in two gives two, and
    // says which frames went where -- which the path alone does not, since
    // each half can be a perfectly smooth arc of its own.
    std::vector<std::vector<double>> depths(pc.cameras.size());
    for (const Track& t : pc.tracks) {
        if (!t.hasPoint) continue;
        for (const Observation& o : t.obs) {
            if (o.frame < 0 || size_t(o.frame) >= pc.cameras.size()) continue;
            const Camera& c = pc.cameras[size_t(o.frame)];
            if (!c.solved) continue;
            const Vec3 q = c.R * t.point + c.t;
            if (q.z > 0) depths[size_t(o.frame)].push_back(q.z);
        }
    }
    // TWO PASSES, ONE OBJECT? With TGLAB_SPLIT=k, the points seen only by
    // frames before k and only by frames from k on, and those both saw: if
    // the two passes placed the subject differently, their exclusive points
    // sit apart and few are shared.
    if (const char* sp = std::getenv("TGLAB_SPLIT")) {
        const int k = std::atoi(sp);
        Vec3 ca{0, 0, 0}, cb{0, 0, 0};
        long na = 0, nb = 0, both = 0;
        for (const Track& t : pc.tracks) {
            if (!t.hasPoint) continue;
            bool a = false, b = false;
            for (const Observation& o : t.obs) (o.frame < k ? a : b) = true;
            if (a && b) { ++both; continue; }
            if (a) { ca = ca + t.point; ++na; }
            if (b) { cb = cb + t.point; ++nb; }
        }
        if (na > 0 && nb > 0) {
            ca = (ca * (1.0 / double(na)) - cm) * (1.0 / rad);
            cb = (cb * (1.0 / double(nb)) - cm) * (1.0 / rad);
            std::printf("\nsplit at frame %d: %ld points only before (centre %.2f %.2f %.2f), "
                        "%ld only after (centre %.2f %.2f %.2f), %ld seen by both; "
                        "centres %.2f radii apart\n",
                        k, na, ca.x, ca.y, ca.z, nb, cb.x, cb.y, cb.z, both, (ca - cb).Norm());
        }
    }
    std::printf("\ncamera path (centroid-relative, in mean radii; look = where it looks):\n");
    const Camera* prev = nullptr;
    const Camera* first = nullptr;
    for (size_t i = 0; i < pc.cameras.size(); ++i) {
        const Camera& c = pc.cameras[i];
        if (!c.solved) { std::printf("  %3zu: ---\n", i); continue; }
        if (!first) first = &c;
        const Vec3 d = (c.Center() - cm) * (1.0 / rad);
        const Mat3 rel = c.R * first->R.Transpose();
        const double turned = Mat3::Identity().AngleTo(rel) * 57.2958;
        const double step = prev ? (c.Center() - prev->Center()).Norm() / rad : 0.0;
        Vec3 look{0, 0, 0};
        std::vector<double>& ds = depths[i];
        if (!ds.empty()) {
            std::nth_element(ds.begin(), ds.begin() + long(ds.size() / 2), ds.end());
            const Vec3 axis{c.R.m[6], c.R.m[7], c.R.m[8]};   // camera +z in world
            look = (c.Center() + axis * ds[ds.size() / 2] - cm) * (1.0 / rad);
        }
        std::printf("  %3zu: %6.2f %6.2f %6.2f   turned %6.1f   step %.3f   look %6.2f %6.2f %6.2f\n",
                    i, d.x, d.y, d.z, turned, step, look.x, look.y, look.z);
        prev = &c;
    }
}

static void DumpOrbit(const PointCloud& pc, const std::string& prefix) {
    const Camera& mid = pc.cameras[pc.cameras.size() / 2];
    if (!mid.solved) return;

    // Gaussians when there are any, otherwise the plain points -- drawn as
    // z-buffered dots, so a dense cloud's floaters show from the side.
    const bool asPoints = pc.splats.empty();
    std::vector<Vec3> pts, cols;
    // With TGLAB_SPLIT=k (see PrintCameras), points seen only before frame k
    // are drawn red, only from k on blue, by both white: a subject the two
    // passes placed differently shows as a red copy beside a blue one.
    const char* splitEnv = std::getenv("TGLAB_SPLIT");
    const int splitAt = splitEnv ? std::atoi(splitEnv) : -1;
    if (asPoints)
        for (const Track& t : pc.tracks) {
            if (!t.hasPoint) continue;
            pts.push_back(t.point);
            if (splitAt < 0) { cols.push_back(t.color); continue; }
            bool a = false, b = false;
            for (const Observation& o : t.obs) (o.frame < splitAt ? a : b) = true;
            cols.push_back(a && b ? Vec3{1, 1, 1} : a ? Vec3{0.9, 0.2, 0.2} : Vec3{0.2, 0.5, 1.0});
        }
    if (asPoints && pts.empty()) return;

    // Centre: the component-wise median, robust to strays.
    std::vector<double> xs, ys, zs;
    if (asPoints)
        for (const Vec3& p : pts) { xs.push_back(p.x); ys.push_back(p.y); zs.push_back(p.z); }
    else
        for (const Splat& s : pc.splats) {
            xs.push_back(s.mean.x);
            ys.push_back(s.mean.y);
            zs.push_back(s.mean.z);
        }
    auto median = [](std::vector<double>& v) {
        std::nth_element(v.begin(), v.begin() + long(v.size() / 2), v.end());
        return v[v.size() / 2];
    };
    const Vec3 centre{median(xs), median(ys), median(zs)};

    // Swing about the middle camera's own vertical axis (row 1 of R).
    const Vec3 up{mid.R.m[3], mid.R.m[4], mid.R.m[5]};
    const Vec3 p0 = mid.Center() - centre;

    std::vector<SplatParam> params;
    params.reserve(pc.splats.size());
    for (const Splat& s : pc.splats) params.push_back(ToParam(s));

    const int w = 768, h = int(std::lround(768.0 * mid.height / std::max(1, mid.width)));
    for (int deg : {-60, -30, 0, 30, 60}) {
        const double a = deg * 3.14159265358979 / 180.0;
        const Vec3 p = p0 * std::cos(a) + up.Cross(p0) * std::sin(a) +
                       up * (up.Dot(p0) * (1.0 - std::cos(a)));
        const Vec3 pos = centre + p;
        const Vec3 z = (centre - pos).Normalized();
        const Vec3 x = up.Cross(z).Normalized();
        const Vec3 y = z.Cross(x);

        Camera c = mid;
        c.R.m[0] = x.x; c.R.m[1] = x.y; c.R.m[2] = x.z;
        c.R.m[3] = y.x; c.R.m[4] = y.y; c.R.m[5] = y.z;
        c.R.m[6] = z.x; c.R.m[7] = z.y; c.R.m[8] = z.z;
        c.t = (c.R * pos) * -1.0;

        std::vector<double> rgb;
        if (asPoints) {
            // The orbit camera keeps the middle frame's focal, rescaled to w.
            const double f = c.focal * double(w) / std::max(1, c.width);
            rgb.assign(size_t(w) * size_t(h) * 3, 0.08);
            std::vector<float> zb(size_t(w) * size_t(h), 1e30f);
            for (size_t i = 0; i < pts.size(); ++i) {
                const Vec3 q = c.R * pts[i] + c.t;
                if (q.z <= 1e-6) continue;
                const int u = int(f * q.x / q.z + 0.5 * w), v = int(f * q.y / q.z + 0.5 * h);
                for (int dy = 0; dy < 2; ++dy)
                    for (int dx = 0; dx < 2; ++dx) {
                        const int uu = u + dx, vv = v + dy;
                        if (uu < 0 || vv < 0 || uu >= w || vv >= h) continue;
                        float& zz = zb[size_t(vv) * size_t(w) + size_t(uu)];
                        if (float(q.z) >= zz) continue;
                        zz = float(q.z);
                        double* s = &rgb[(size_t(vv) * size_t(w) + size_t(uu)) * 3];
                        s[0] = cols[i].x; s[1] = cols[i].y; s[2] = cols[i].z;
                    }
            }
        } else {
            SplatRaster r;
            RasterOptions opt;
            const SplatCam sc = SplatCamFrom(c, w, h);
            std::vector<ReflParam> refl;
            EnvMap env;
            ReflFromCloud(pc, &refl, &env);
            RenderShaded(r, params, pc.splatSh, pc.shDegree, refl, env, sc, opt, &rgb);
        }

        Image im;
        im.Alloc(ImageDesc{w, h, Format::RGBA8});
        ImageView v = im.MapCpuWrite();
        for (int yy = 0; yy < h; ++yy)
            for (int xx = 0; xx < w; ++xx) {
                uint8_t* q = v.At<uint8_t>(xx, yy);
                const double* s = &rgb[(size_t(yy) * size_t(w) + size_t(xx)) * 3];
                for (int ch = 0; ch < 3; ++ch)
                    q[ch] = uint8_t(std::clamp(s[ch], 0.0, 1.0) * 255.0 + 0.5);
                q[3] = 255;
            }
        v = ImageView{};
        char name[64];
        std::snprintf(name, sizeof(name), "_orbit%+03d.png", deg);
        std::string e;
        SavePng(prefix + name, im, &e);
    }
}

// From above: an orthographic view down the axis of the camera path (the
// least eigenvector of the camera centres' spread -- for a walk-around, the
// ring's normal). Points grey, the camera path as a trail coloured by frame,
// red -> green -> blue. Where a walk-around came out as several copies of
// the subject, this shows how many and which stretch of the clip made each.
static void DumpTop(const PointCloud& pc, const std::string& prefix) {
    std::vector<Vec3> cams;
    std::vector<int> camIdx;
    for (size_t i = 0; i < pc.cameras.size(); ++i)
        if (pc.cameras[i].solved) { cams.push_back(pc.cameras[i].Center()); camIdx.push_back(int(i)); }
    if (cams.size() < 3) return;
    Vec3 c0{0, 0, 0};
    for (const Vec3& p : cams) c0 = c0 + p;
    c0 = c0 * (1.0 / double(cams.size()));
    double A[9] = {};
    for (const Vec3& p : cams) {
        const Vec3 d = p - c0;
        const double v[3] = {d.x, d.y, d.z};
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) A[r * 3 + c] += v[r] * v[c];
    }
    double ev[3];
    Mat3 V;
    linalg::SymmetricEigen<3>(A, ev, V.m);
    const Vec3 e0 = V.Column(2), e1 = V.Column(1);   // the path's plane

    std::vector<Vec3> pts;
    for (const Track& t : pc.tracks)
        if (t.hasPoint) pts.push_back(t.point);
    // Extent: the cameras and the middle 96% of the points.
    std::vector<double> us, vs;
    for (const Vec3& p : pts) { us.push_back((p - c0).Dot(e0)); vs.push_back((p - c0).Dot(e1)); }
    double lo0 = 0, hi0 = 0, lo1 = 0, hi1 = 0;
    if (!us.empty()) {
        std::vector<double> a = us, b = vs;
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        lo0 = a[a.size() / 50]; hi0 = a[a.size() - 1 - a.size() / 50];
        lo1 = b[b.size() / 50]; hi1 = b[b.size() - 1 - b.size() / 50];
    }
    for (const Vec3& p : cams) {
        const double u = (p - c0).Dot(e0), v = (p - c0).Dot(e1);
        lo0 = std::min(lo0, u); hi0 = std::max(hi0, u);
        lo1 = std::min(lo1, v); hi1 = std::max(hi1, v);
    }
    const int W = 900;
    const double span = std::max(hi0 - lo0, hi1 - lo1) * 1.05 + 1e-9;
    const double s = double(W) / span;
    const double m0 = 0.5 * (lo0 + hi0), m1 = 0.5 * (lo1 + hi1);
    std::vector<double> rgb(size_t(W) * W * 3, 0.06);
    auto put = [&](double u, double v, double r, double g, double b, int rad) {
        const int x = int((u - m0) * s + 0.5 * W), y = int((v - m1) * s + 0.5 * W);
        for (int dy = -rad; dy <= rad; ++dy)
            for (int dx = -rad; dx <= rad; ++dx) {
                const int xx = x + dx, yy = y + dy;
                if (xx < 0 || yy < 0 || xx >= W || yy >= W) continue;
                double* q = &rgb[(size_t(yy) * W + size_t(xx)) * 3];
                q[0] = r; q[1] = g; q[2] = b;
            }
    };
    for (size_t i = 0; i < pts.size(); ++i) put(us[i], vs[i], 0.55, 0.55, 0.55, 0);
    const int nFrames = int(pc.cameras.size());
    for (size_t k = 0; k < cams.size(); ++k) {
        const double t = double(camIdx[k]) / std::max(1, nFrames - 1);
        const double r = std::max(0.0, 1.0 - 2.0 * t), b = std::max(0.0, 2.0 * t - 1.0);
        put((cams[k] - c0).Dot(e0), (cams[k] - c0).Dot(e1), r, 1.0 - r - b, b, 3);
    }
    Image im;
    im.Alloc(ImageDesc{W, W, Format::RGBA8});
    ImageView v = im.MapCpuWrite();
    for (int y = 0; y < W; ++y)
        for (int x = 0; x < W; ++x) {
            uint8_t* q = v.At<uint8_t>(x, y);
            const double* p = &rgb[(size_t(y) * W + size_t(x)) * 3];
            for (int ch = 0; ch < 3; ++ch) q[ch] = uint8_t(std::clamp(p[ch], 0.0, 1.0) * 255.0 + 0.5);
            q[3] = 255;
        }
    v = ImageView{};
    std::string e;
    SavePng(prefix + "_top.png", im, &e);
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);

    std::vector<std::string> files;
    std::string detector = "detect_akaze";
    int window = 3, maxDim = 1600, rotMethod = 1, posMethod = 0;
    int videoFrames = 0;     // --video-frames N: N time slots; default picks by motion
    double videoStep = 0.0;  // --video-step F: motion step, a fraction of the short side
    int    videoMax = -1;    // --video-max N: cap on frames kept by motion (0 = none)
    bool printCameras = false;   // --cameras: the camera path, per camera
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
    int minLen  = -1;            // negative: leave build_tracks's default
    double minAngle = -1.0;      // negative: leave triangulate's default
    double detThresh = -1.0;     // negative: leave the detector's default
    int detMaxDim = -1;          // negative: leave the detector's own default
    int detColour = -1;          // negative: leave the detector's default
    std::string script;
    std::string dumpDir;   // --dump: save every image-set viewer as PNGs
    bool useGpu = false;   // --gpu: give the pipeline a D3D12 device

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
        else if (a == "--min-length" && i + 1 < argc) minLen = std::atoi(argv[++i]);
        else if (a == "--min-angle" && i + 1 < argc) minAngle = std::atof(argv[++i]);
        else if (a == "--threshold" && i + 1 < argc) detThresh = std::atof(argv[++i]);
        else if (a == "--det-max-dim" && i + 1 < argc) detMaxDim = std::atoi(argv[++i]);
        else if (a == "--colour" && i + 1 < argc) detColour = std::atoi(argv[++i]);
        else if (a == "--script" && i + 1 < argc)   script = argv[++i];
        else if (a == "--dump" && i + 1 < argc)     dumpDir = argv[++i];
        else if (a == "--gpu")                         useGpu = true;
        else if (a == "--cameras")                     printCameras = true;
        else if (a == "--video-frames" && i + 1 < argc) videoFrames = std::atoi(argv[++i]);
        else if (a == "--video-step" && i + 1 < argc) videoStep = std::atof(argv[++i]);
        else if (a == "--video-max" && i + 1 < argc)  videoMax = std::atoi(argv[++i]);
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

    // A VIDEO: its sharpest frame per time slot, as the app's palette gets
    // them, standing in for a folder of photographs.
    std::vector<Image> videoSet;
    if (files.size() == 1 && IsVideoPath(files[0])) {
        VideoOptions vo;
        if (videoFrames > 0) {
            vo.pick = VideoPick::Time;
            vo.frames = videoFrames;
        }
        if (videoStep > 0.0) vo.step = videoStep;
        if (videoMax >= 0) vo.maxFrames = videoMax;
        VideoInfo vi;
        std::string verr;
        const auto tv = std::chrono::steady_clock::now();
        if (!LoadVideoFrames(files[0], vo, &videoSet, &vi, &verr)) {
            std::printf("%s\n", verr.c_str());
            return 1;
        }
        std::printf("video: %s in %.0f ms\n", VideoNote(vi, int(videoSet.size())).c_str(),
                    Ms(tv, std::chrono::steady_clock::now()));
        // The spacing, when picked by motion: how far apart kept frames are
        // in time and in view.
        if (!vi.motion.empty()) {
            double lo = 1e9, hi = 0.0;
            for (size_t i = 1; i < vi.times.size(); ++i) {
                lo = std::min(lo, vi.times[i] - vi.times[i - 1]);
                hi = std::max(hi, vi.times[i] - vi.times[i - 1]);
            }
            std::printf("video: %d by motion, %d on lost tracking, %d by time, halved %d "
                        "times; gaps %.2f..%.2f s\n",
                        vi.byMotion, vi.byTracking, vi.byTime, vi.thinned,
                        vi.times.size() > 1 ? lo : 0.0, hi);
        }
        // --video-frames-detail: each kept frame's time, sharpness and motion.
        if (std::getenv("TGLAB_VIDEO_DETAIL"))
            for (size_t i = 0; i < vi.times.size(); ++i)
                std::printf("  frame %3zu  t %6.2f s  sharpness %7.1f  motion %.3f\n", i,
                            vi.times[i], vi.sharpness[i],
                            i < vi.motion.size() ? vi.motion[i] : 0.0);
        files.assign(videoSet.size(), files[0]);
    }

    if (files.size() < 3) {
        std::printf("usage: bench_sfm <dir-or-image-or-video> [image...] "
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
    const bool fromVideo = !videoSet.empty();
    // A video's frames are already decoded, so they downscale in parallel:
    // one at a time was two minutes of a 1577-frame clip.
    if (!videoSet.empty()) {
        std::vector<Image> small(videoSet.size());
        std::vector<char> ok(videoSet.size(), 0);
        ParallelFor(videoSet.size(), [&](size_t fi) {
            ok[fi] = Downscale(videoSet[fi], maxDim, &small[fi]) ? 1 : 0;
        });
        for (size_t fi = 0; fi < small.size(); ++fi) {
            if (!ok[fi]) {
                std::printf("  frame %zu: could not downscale\n", fi);
                return 1;
            }
            if (fi == 0)
                std::printf("  %dx%d -> %dx%d\n", videoSet[0].Desc().width, videoSet[0].Desc().height,
                            small[0].Desc().width, small[0].Desc().height);
            set.images.push_back(std::move(small[fi]));
        }
        videoSet.clear();
    }
    for (size_t fi = 0; fi < files.size() && !fromVideo; ++fi) {
        const std::string& f = files[fi];
        Image full;
        std::string err;
        if (!videoSet.empty()) {
            full = std::move(videoSet[fi]);
        } else if (!LoadImageFile(f, &full, &err)) {
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
        SourceImage src{"group", 0};
        src.shape = std::get<ImageSet>(s[0]).shape;   // a group, so reductions see a set
        std::vector<SourceImage> names{src};
        const InterpResult ir = Interpret(prog, names, &ui, &sp);
        if (!ir.ok) { std::printf("interpret: %s\n", ir.error.c_str()); return 1; }

        std::printf("running %s (%d stages, %d viewers)\n\n", script.c_str(),
                    int(sp.Stages().size()), int(sp.Viewers().size()));

        // --gpu: a real device, so GPU paths run and can be TIMED here rather
        // than only in the app. Without it every GPU stage falls back to the
        // CPU and the bench measures that instead.
        ID3D12Device* dev = nullptr;
        ComputeContext gpu;
        ComputeContext* gpuPtr = nullptr;
        if (useGpu) {
            if (SUCCEEDED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
                                            IID_PPV_ARGS(&dev))) &&
                gpu.Init(dev)) {
                gpuPtr = &gpu;
                std::printf("GPU: on\n");
            } else {
                std::printf("GPU: requested but not available; running on the CPU\n");
            }
        }

        const auto ts = std::chrono::steady_clock::now();
        std::string serr;
        const bool sok = sp.Execute(&s, nullptr, &serr, gpuPtr);
        const double sms = Ms(ts, std::chrono::steady_clock::now());

        // The script's save() lines, which the app runs on request; here
        // every run, so a bench script can export what it made.
        if (sok && !sp.Saves().empty()) {
            const auto tw = std::chrono::steady_clock::now();
            std::vector<std::string> wrote;
            std::string werr;
            if (!sp.RunSaves(&s, &werr, &wrote))
                std::printf("save failed: %s\n", werr.c_str());
            for (const std::string& w : wrote)
                std::printf("saved %s (%.0f ms)\n", w.c_str(),
                            Ms(tw, std::chrono::steady_clock::now()));
        }

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

            // --dump: what a viewer would show, as files. The bench has no
            // window, and some things -- a render beside its photograph --
            // cannot be judged from numbers alone.
            if (!dumpDir.empty() && d) {
                if (const Image* one = std::get_if<Image>(d)) {
                    std::filesystem::create_directories(dumpDir);
                    Image copy = const_cast<Image&>(*one).Clone();
                    std::string e;
                    if (!SavePng(dumpDir + "/" + vd.name + ".png", copy, &e))
                        std::printf("dump %s: %s\n", vd.name.c_str(), e.c_str());
                }
                if (const ImageSet* set = std::get_if<ImageSet>(d)) {
                    std::filesystem::create_directories(dumpDir);
                    for (size_t i = 0; i < set->images.size(); ++i) {
                        Image copy = set->images[i].Clone();
                        // A depth or confidence map is R32F, which SavePng
                        // does not take: stretched to grey over its measured
                        // range, unmeasured pixels (0) left black.
                        if (copy.Desc().format == Format::R32F) {
                            const ImageDesc& cd = copy.Desc();
                            ImageView cv = copy.MapCpuRead();
                            float lo = 1e30f, hi = -1e30f;
                            for (int y = 0; y < cd.height; ++y)
                                for (int x = 0; x < cd.width; ++x) {
                                    const float v = *cv.At<float>(x, y);
                                    if (v > 0.0f && std::isfinite(v)) { lo = std::min(lo, v); hi = std::max(hi, v); }
                                }
                            Image g;
                            g.Alloc(ImageDesc{cd.width, cd.height, Format::RGBA8});
                            ImageView gv = g.MapCpuWrite();
                            for (int y = 0; y < cd.height; ++y)
                                for (int x = 0; x < cd.width; ++x) {
                                    const float v = *cv.At<float>(x, y);
                                    uint8_t b = 0;
                                    if (v > 0.0f && hi > lo) b = uint8_t(40 + 215.0f * (v - lo) / (hi - lo));
                                    uint8_t* px = gv.At<uint8_t>(x, y);
                                    px[0] = px[1] = px[2] = b;
                                    px[3] = 255;
                                }
                            copy = std::move(g);
                        }
                        char name[64];
                        std::snprintf(name, sizeof(name), "_%02d.png", int(i));
                        std::string e;
                        if (!SavePng(dumpDir + "/" + vd.name + name, copy, &e))
                            std::printf("dump %s: %s\n", name, e.c_str());
                    }
                }
                if (const PointCloud* pc = std::get_if<PointCloud>(d);
                    pc && !pc->cameras.empty()) {
                    std::filesystem::create_directories(dumpDir);
                    DumpOrbit(*pc, dumpDir + "/" + vd.name);
                    DumpTop(*pc, dumpDir + "/" + vd.name);
                }
            }
            const char* kind = "nothing";
            if (d && std::holds_alternative<PointCloud>(*d)) kind = "PointCloud";
            else if (d && std::holds_alternative<Image>(*d)) kind = "Image";
            else if (d && std::holds_alternative<ImageSet>(*d)) kind = "ImageSet";
            std::printf("viewer \"%s\": %s", vd.name.c_str(), kind);
            if (d) if (const PointCloud* pc = std::get_if<PointCloud>(d)) {
                std::printf(" (%d cameras, %d points)", pc->SolvedCameras(),
                            pc->TriangulatedPoints());
                ReportPlanarity(*pc);
                if (printCameras) PrintCameras(*pc);
            }
            std::printf("\n");
        }
        std::printf("\nstage timings\n");
        for (const Stage& st : sp.Stages())
            if (st.algo && st.lastMs > 0.0)
                std::printf("  %-22s %9.0f ms  %5.1f%%\n", st.algoName.c_str(), st.lastMs,
                            100.0 * st.lastMs / std::max(1.0, sms));
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
        if (detThresh > 0.0) SetParam(det.get(), "threshold", detThresh);
        if (detMaxDim > 0) SetParam(det.get(), "max_dim", double(detMaxDim));
        if (detColour >= 0) SetParam(det.get(), "colour", double(detColour));
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
    if (minLen > 0) SetParam(bt.get(), "min_length", double(minLen));
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
    if (minAngle > 0.0) SetParam(tri.get(), "min_angle", minAngle);
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
    if (minAngle > 0.0) SetParam(tri2.get(), "min_angle", minAngle);
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
