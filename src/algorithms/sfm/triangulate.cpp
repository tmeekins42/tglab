// triangulate — 3D points from known cameras and their observations.
//
// Given where the cameras are and where each saw a point, the point is where
// those rays meet. They never exactly meet -- measurement noise sees to that --
// so the answer is the position minimising distance to all of them.
//
// WHY THIS IS A SEPARATE STAGE, given that global_position's joint method
// already produces points as a by-product. Two reasons, and the second is the
// interesting one.
//
// The plain one: the translation-averaging arm does NOT produce points. It
// solves for camera positions from edge directions alone, so a reconstruction
// that went that way arrives at bundle adjustment with nothing to adjust.
//
// The real one: retriangulation. After bundle adjustment has moved the cameras,
// the points are no longer optimal for their new positions, and running this
// again recovers points that were poorly placed -- or were rejected outright --
// when the cameras were still wrong. COLMAP does exactly this, alternating
// bundle adjustment and retriangulation until neither improves. A stage that
// can only run before BA could not do that.
//
// THE ESTIMATOR is the linear (DLT) one: stack the two rows each view
// contributes to the projection equation and take the smallest singular
// vector. Its known weakness is that it minimises an ALGEBRAIC residual rather
// than reprojection error, so it is biased when the rays are nearly parallel.
// That is precisely the case the angle test below rejects, and what bundle
// adjustment afterwards corrects. The optimal L2 method (Hartley-Sturm) solves
// a sixth-degree polynomial for the exact two-view answer and does not
// generalise past two views; for N views with BA downstream, DLT is the
// standard choice and the one every pipeline makes.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "../../core/algorithm.h"

namespace tglab {
namespace {

// Smallest eigenvector of a symmetric 4x4, by cyclic Jacobi.
//
// The DLT system is 2N x 4; forming A^T A makes it 4x4 regardless of how many
// views saw the point, which matters because a long track can have dozens.
// Squaring costs condition number in principle, and in practice the normalised
// image coordinates used here keep the entries within an order of magnitude of
// each other, which is what makes it safe.
bool SmallestEigenvector4(const double A[16], double out[4]) {
    double M[16];
    std::copy(A, A + 16, M);
    double V[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

    for (int sweep = 0; sweep < 50; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < 4; ++p)
            for (int q = p + 1; q < 4; ++q) off += M[p * 4 + q] * M[p * 4 + q];
        if (off < 1e-26) break;

        for (int p = 0; p < 4; ++p) {
            for (int q = p + 1; q < 4; ++q) {
                const double apq = M[p * 4 + q];
                if (std::fabs(apq) < 1e-20) continue;
                const double theta = 0.5 * (M[q * 4 + q] - M[p * 4 + p]) / apq;
                const double t = (theta >= 0.0 ? 1.0 : -1.0) /
                                 (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;

                for (int k = 0; k < 4; ++k) {
                    const double mkp = M[k * 4 + p], mkq = M[k * 4 + q];
                    M[k * 4 + p] = c * mkp - s * mkq;
                    M[k * 4 + q] = s * mkp + c * mkq;
                }
                for (int k = 0; k < 4; ++k) {
                    const double mpk = M[p * 4 + k], mqk = M[q * 4 + k];
                    M[p * 4 + k] = c * mpk - s * mqk;
                    M[q * 4 + k] = s * mpk + c * mqk;
                }
                for (int k = 0; k < 4; ++k) {
                    const double vkp = V[k * 4 + p], vkq = V[k * 4 + q];
                    V[k * 4 + p] = c * vkp - s * vkq;
                    V[k * 4 + q] = s * vkp + c * vkq;
                }
            }
        }
    }

    int best = 0;
    for (int i = 1; i < 4; ++i) if (M[i * 4 + i] < M[best * 4 + best]) best = i;
    for (int i = 0; i < 4; ++i) out[i] = V[i * 4 + best];
    return true;
}

class Triangulate : public AlgorithmBase {
public:
    const char* Name()     const override { return "triangulate"; }
    const char* Category() const override { return "sfm"; }

    PortList Inputs() const override {
        return {{"src", DataType::PointCloud, FormatSpec::Any, ShapeSpec::Any}};
    }
    PortList Outputs() const override {
        return {{"out", DataType::PointCloud, FormatSpec::Any, ShapeSpec::Any}};
    }

    void RunCPU(RunCtx&) override {}
    bool IsReconstruct() const override { return true; }

    bool RunReconstruct(const std::vector<Image>*, PointCloud* cloud,
                        std::string* err) override {
        const int nCam = int(cloud->cameras.size());
        int solvedCams = 0;
        for (const Camera& c : cloud->cameras) if (c.solved) ++solvedCams;
        if (solvedCams < 2) {
            *err = "triangulate: fewer than two solved cameras -- run "
                   "rotation_average and global_position first";
            return false;
        }
        if (cloud->tracks.empty()) {
            *err = "triangulate: no tracks -- run build_tracks first";
            return false;
        }

        const double minAngleRad =
            double(m_minAngle) * 3.14159265358979 / 180.0;
        const double maxErr = double(m_maxError);

        // --- THE SCALE OF THE SCENE, from the cameras -----------------------
        //
        // A reconstruction has no metric scale -- that is a gauge freedom, and
        // the only length in the problem is how far apart the cameras ended
        // up. So "too far away" has to be expressed relative to them.
        //
        // WHY THIS TEST IS NEEDED at all, given the three that follow it. A
        // point far beyond the scene can pass every one of them: it reprojects
        // correctly (that is why it survived), it is in front of both cameras,
        // and two rays that are nearly-but-not-quite parallel meet at a huge
        // distance while still clearing the parallax threshold. The depth is
        // then decided by the last decimal of the correspondence.
        //
        // Measured on castle-P19: about 45 points of 1129 sat far enough out
        // to stretch the reported extent from 4.5 units -- the size of the
        // camera walk, and of the courtyard -- to 65.9, a 14x inflation from
        // under 4% of the cloud. On screen they framed the viewport around
        // empty space and made the real structure look like two small clumps.
        //
        // The radius of the camera cluster is the reference, not the mean
        // distance to the points: the points are what is being judged, and a
        // criterion computed from them would move with the very outliers it is
        // meant to reject.
        Vec3 camMean{0, 0, 0};
        int camCount = 0;
        for (const Camera& c : cloud->cameras) {
            if (!c.solved) continue;
            camMean = camMean + c.Center();
            ++camCount;
        }
        double camRadius = 0.0;
        if (camCount > 0) {
            camMean = camMean * (1.0 / double(camCount));
            for (const Camera& c : cloud->cameras) {
                if (!c.solved) continue;
                camRadius = std::max(camRadius, (c.Center() - camMean).Norm());
            }
        }
        // A degenerate cluster -- every camera in one place -- gives no scale
        // to work with, and the test switches itself off rather than rejecting
        // everything. The parallax test already covers that configuration.
        const double maxDist = (camRadius > 1e-9)
                                   ? camRadius * double(m_maxDistance)
                                   : 0.0;
        int tooDistant = 0;

        int kept = 0, tooNarrow = 0, behind = 0, tooFar = 0, short_ = 0;
        // Every reprojection error that got as far as the final check, so the
        // report can say whether the rejected majority sits just over the
        // threshold or is scattered far beyond it. Those call for opposite
        // responses: the first is a threshold too tight, the second is
        // geometry that is actually wrong.
        std::vector<double> errors;

        for (Track& tr : cloud->tracks) {
            // Gather the views that can see this point.
            struct View { const Camera* cam; double u, v; };
            std::vector<View> views;
            for (const Observation& o : tr.obs) {
                if (o.frame < 0 || o.frame >= nCam) continue;
                const Camera& c = cloud->cameras[size_t(o.frame)];
                if (!c.solved) continue;
                views.push_back(View{&c, double(o.x), double(o.y)});
            }
            if (views.size() < 2) { tr.hasPoint = false; ++short_; continue; }

            // --- DLT ---------------------------------------------------------
            //
            // Each view contributes two rows of x * (P3 row) - (P1 row) = 0, in
            // NORMALISED coordinates so the rows are comparably scaled. The
            // projection matrix is [R | t] once the pixel has been divided
            // through by the intrinsics.
            double A[16] = {};
            for (const View& vw : views) {
                const Camera& c = *vw.cam;
                const double xn = (vw.u - c.cx) / c.focal;
                const double yn = (vw.v - c.cy) / c.focal;

                // Rows of P = [R | t].
                const double p0[4] = {c.R.At(0, 0), c.R.At(0, 1), c.R.At(0, 2), c.t.x};
                const double p1[4] = {c.R.At(1, 0), c.R.At(1, 1), c.R.At(1, 2), c.t.y};
                const double p2[4] = {c.R.At(2, 0), c.R.At(2, 1), c.R.At(2, 2), c.t.z};

                double r0[4], r1[4];
                for (int k = 0; k < 4; ++k) {
                    r0[k] = xn * p2[k] - p0[k];
                    r1[k] = yn * p2[k] - p1[k];
                }
                for (int a = 0; a < 4; ++a)
                    for (int b = 0; b < 4; ++b)
                        A[a * 4 + b] += r0[a] * r0[b] + r1[a] * r1[b];
            }

            double X[4];
            if (!SmallestEigenvector4(A, X) || std::fabs(X[3]) < 1e-12) {
                tr.hasPoint = false;
                ++tooNarrow;
                continue;
            }
            const Vec3 P{X[0] / X[3], X[1] / X[3], X[2] / X[3]};

            // --- the three rejections ----------------------------------------
            //
            // Each catches a different way a triangulation can be worthless
            // while still being a number, and all three are standard.

            // 1. PARALLAX. Two rays that are nearly parallel meet at a distance
            //    the noise decides, so the depth is arbitrary. This is the most
            //    important of the three: it is what stops a reconstruction
            //    filling with points at implausible distance, and it is why a
            //    pure rotation produces no structure at all.
            double widest = 0.0;
            for (size_t i = 0; i < views.size(); ++i) {
                const Vec3 di = (P - views[i].cam->Center()).Normalized();
                for (size_t j = i + 1; j < views.size(); ++j) {
                    const Vec3 dj = (P - views[j].cam->Center()).Normalized();
                    double c = di.Dot(dj);
                    c = c < -1.0 ? -1.0 : (c > 1.0 ? 1.0 : c);
                    widest = std::max(widest, std::acos(c));
                }
            }
            if (widest < minAngleRad) { tr.hasPoint = false; ++tooNarrow; continue; }

            // 2. CHEIRALITY. A point behind a camera that supposedly saw it is
            //    not a hard case, it is a wrong answer -- the sign ambiguity in
            //    the geometry resolved the wrong way.
            bool allFront = true;
            for (const View& vw : views) {
                double px, py;
                if (!vw.cam->Project(P, &px, &py)) { allFront = false; break; }
            }
            if (!allFront) { tr.hasPoint = false; ++behind; continue; }

            // 3. DISTANCE. Is it a plausible distance from the cameras that
            //    saw it? See the note above: this catches the point that
            //    reprojects correctly but sits far outside the scene, which
            //    the other three tests cannot.
            if (maxDist > 0.0 && (P - camMean).Norm() > maxDist) {
                tr.hasPoint = false;
                ++tooDistant;
                continue;
            }

            // 4. REPROJECTION. The direct check: does it land where it was
            //    seen? A point can pass both tests above and still be wrong if
            //    the correspondence itself was.
            double worst = 0.0;
            for (const View& vw : views) {
                double px, py;
                vw.cam->Project(P, &px, &py);
                const double dx = px - vw.u, dy = py - vw.v;
                worst = std::max(worst, std::sqrt(dx * dx + dy * dy));
            }
            errors.push_back(worst);
            if (worst > maxErr) { tr.hasPoint = false; ++tooFar; continue; }

            tr.point = P;
            tr.hasPoint = true;
            ++kept;
        }

        auto pct = [&errors](double p) {
            if (errors.empty()) return 0.0;
            const size_t i = size_t(p * double(errors.size() - 1));
            std::nth_element(errors.begin(), errors.begin() + long(i),
                             errors.end());
            return errors[i];
        };
        const double p50 = pct(0.50), p90 = pct(0.90);

        char buf[420];
        std::snprintf(buf, sizeof buf,
                      "triangulate: %d of %d tracks (%d too narrow, %d behind a "
                      "camera, %d too distant, %d reprojected past %.1f px, "
                      "%d under two views); "
                      "reprojection median %.1f px, p90 %.1f px; "
                      "camera radius %.2f, distance limit %.2f",
                      kept, int(cloud->tracks.size()), tooNarrow, behind,
                      tooDistant, tooFar, maxErr, short_, p50, p90,
                      camRadius, maxDist);
        m_note = buf;

        if (kept == 0) {
            *err = "triangulate: nothing survived -- " + m_note;
            return false;
        }
        return true;
    }

    std::string RunReport() const override { return m_note; }
    ProxyBehaviour Proxy() const override { return ProxyBehaviour::Never; }
    bool HasGPU() const override { return false; }

private:
    Param<float> m_minAngle{this, "min_angle", 2.0f, 0.1f, 30.0f,
        {.help = "Smallest angle between two viewing rays, in degrees, for a "
                 "point to be accepted. Nearly parallel rays meet wherever the "
                 "noise puts them, so the depth is arbitrary -- this is what "
                 "stops a reconstruction filling with points at implausible "
                 "distance, and why a pure camera rotation yields no structure "
                 "at all."}};

    Param<float> m_maxDistance{this, "max_distance", 20.0f, 1.0f, 1000.0f,
        {.help = "How far a point may lie from the cameras before it is "
                 "rejected, as a MULTIPLE of the camera cluster's radius. A "
                 "reconstruction has no metric scale, so the cameras are the "
                 "only available reference. This catches the point that "
                 "reprojects correctly and is in front of both cameras but "
                 "sits far outside the scene -- two nearly parallel rays meet "
                 "at an enormous distance while still passing the parallax "
                 "test.\n\n"
                 "THE DEFAULT IS DELIBERATELY LOOSE, because the right value "
                 "depends on the capture and the two common ones disagree by "
                 "an order of magnitude. A walk AROUND a subject puts the "
                 "cameras on a ring wider than the subject is deep, so its "
                 "points sit within a few camera radii -- 4 was measured as "
                 "the knee on castle-P19. A scene shot from OUTSIDE, cameras "
                 "clustered and the subject in front, legitimately places "
                 "every point ten or more radii away, and 4 rejects the whole "
                 "reconstruction. This defaults to catching only the gross "
                 "outlier; lower it when the cameras surround the subject."}};

    Param<float> m_maxError{this, "max_error", 4.0f, 0.5f, 100.0f,
        {.help = "Largest reprojection error, in pixels, for a triangulated "
                 "point to be kept. The direct test: a point can have good "
                 "parallax and correct cheirality and still be wrong if the "
                 "correspondence was."}};

    std::string m_note;
};

REGISTER_ALGORITHM(Triangulate);

}  // namespace
}  // namespace tglab
