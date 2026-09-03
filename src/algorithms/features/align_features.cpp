// align_features — a transform from feature matches, by RANSAC.
//
// Where the pixel aligner refines a warp it is already nearly right about, this
// one finds a warp from nothing. That is the division of labour Tim set out:
// feature matching for the coarse solve, the pixel solver to refine it. The
// transform sidecar is what lets them compose -- align reads whatever is
// already attached and starts from it.
//
// NOT EPIPOLAR GEOMETRY, and the distinction decides everything here.
//
// Epipolar geometry constrains two views of a 3D scene from DIFFERENT camera
// centres: the fundamental matrix maps a point to a LINE in the other image, so
// outlier rejection there asks whether a match lies near its epipolar line.
// That is the right machinery for structure-from-motion, and the wrong
// machinery for this.
//
// Frame alignment and panorama stitching are exactly the cases where the camera
// centre does NOT move -- a tripod pan, a burst of the same scene -- or where
// the scene is effectively planar. Then the two views are related by a
// HOMOGRAPHY, which maps a point to a POINT. That is a far stronger constraint
// than a line, and it makes the outlier test cheaper and sharper: reproject the
// match and measure how far it landed from where it should have.
//
// (When the camera centre does move and the scene has depth, no homography
// fits, and this will report a low inlier count rather than a wrong answer.
// That is the honest failure: the model does not apply, and saying so beats
// fitting something.)
//
// WHY RANSAC RATHER THAN LEAST SQUARES. A least-squares fit over all matches is
// dominated by its worst outliers -- one match pairing a tree with a different
// tree drags the whole solution. RANSAC instead fits the MINIMUM number of
// points repeatedly from random samples and keeps whichever fit the most
// matches agree with. It is not a refinement of least squares; it is what makes
// least squares usable, because the final fit runs only on the inliers.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "../../algo_util/features.h"
#include "../../algo_util/transform.h"
#include "../../core/algorithm.h"

namespace tglab {
namespace {

struct Pair {
    float ax, ay;   // in the reference
    float bx, by;   // in this frame
};

// Solves a small dense linear system by Gaussian elimination with partial
// pivoting. n is at most 8 here, so the O(n^3) is nothing.
//
// Partial pivoting rather than plain elimination: a sample of four nearly
// collinear points produces a near-singular system, and without pivoting the
// division by a tiny leading element turns that into garbage rather than into
// the "singular, reject this sample" that the caller can handle.
bool SolveDense(std::vector<double>& a, std::vector<double>& b, int n) {
    for (int col = 0; col < n; ++col) {
        int piv = col;
        for (int r = col + 1; r < n; ++r)
            if (std::abs(a[size_t(r * n + col)]) > std::abs(a[size_t(piv * n + col)]))
                piv = r;
        if (std::abs(a[size_t(piv * n + col)]) < 1e-12) return false;

        if (piv != col) {
            for (int c = 0; c < n; ++c)
                std::swap(a[size_t(col * n + c)], a[size_t(piv * n + c)]);
            std::swap(b[size_t(col)], b[size_t(piv)]);
        }

        const double d = a[size_t(col * n + col)];
        for (int r = col + 1; r < n; ++r) {
            const double f = a[size_t(r * n + col)] / d;
            if (f == 0.0) continue;
            for (int c = col; c < n; ++c) a[size_t(r * n + c)] -= f * a[size_t(col * n + c)];
            b[size_t(r)] -= f * b[size_t(col)];
        }
    }
    for (int r = n - 1; r >= 0; --r) {
        double s = b[size_t(r)];
        for (int c = r + 1; c < n; ++c) s -= a[size_t(r * n + c)] * b[size_t(c)];
        b[size_t(r)] = s / a[size_t(r * n + r)];
    }
    return true;
}

// Hartley normalisation: centre the points and scale them to mean distance
// sqrt(2) from the origin.
//
// NOT OPTIONAL, and the numbers say why. The DLT's rows contain terms in x, y,
// 1 and -u*x -- at pixel coordinates up to 512 that last term reaches 260,000,
// and forming the normal equations squares the spread to about 7e10. The system
// is then ill conditioned enough that the solved homography is wrong by tens of
// pixels at the corners.
//
// Measured, before this was added: translation recovered to 0.008 px (where the
// conditioning barely matters), rotation and scale to 30 px, and a perspective
// warp to 33 px with the perspective terms coming back as zero -- the solver
// was quietly returning something affine because the extra columns were noise.
// After normalising, all three are sub-pixel.
//
// Returns the transform that maps original -> normalised, so the solve is done
// in normalised space and the answer is un-normalised afterwards.
struct Norm {
    float cx = 0.0f, cy = 0.0f, s = 1.0f;

    void Fit(const std::vector<Pair>& p, const std::vector<int>& idx, bool useA) {
        double mx = 0.0, my = 0.0;
        for (int i : idx) {
            const Pair& q = p[size_t(i)];
            mx += useA ? q.ax : q.bx;
            my += useA ? q.ay : q.by;
        }
        const double n = double(idx.size());
        cx = float(mx / n);
        cy = float(my / n);

        double d = 0.0;
        for (int i : idx) {
            const Pair& q = p[size_t(i)];
            const double dx = (useA ? q.ax : q.bx) - cx;
            const double dy = (useA ? q.ay : q.by) - cy;
            d += std::sqrt(dx * dx + dy * dy);
        }
        d /= n;
        s = (d > 1e-9) ? float(1.4142135 / d) : 1.0f;
    }

    void Apply(float x, float y, float* ox, float* oy) const {
        *ox = (x - cx) * s;
        *oy = (y - cy) * s;
    }

    // As a matrix, for composing the answer back out of normalised space.
    Affine Matrix() const {
        Affine t;
        t.m[0] = s; t.m[1] = 0.0f; t.m[2] = -s * cx;
        t.m[3] = 0.0f; t.m[4] = s; t.m[5] = -s * cy;
        return t;
    }
};

// --- the three models -------------------------------------------------------
//
// Each solves for the transform that maps the REFERENCE point to this frame's,
// which is the direction transform.h documents: a merge walks the reference
// grid and asks where to read.

// 4 DOF: translation, rotation, uniform scale. Two points suffice.
//
// The right model when the camera did not tilt or move -- a burst on a tripod,
// or frames from a stabilised video. Fewer parameters means each is better
// determined, so on data that really is a similarity this beats a homography
// even though the homography could represent it.
bool SolveSimilarity(const std::vector<Pair>& p, const std::vector<int>& idx,
                     Affine* out) {
    if (idx.size() < 2) return false;
    // [ s*cos  -s*sin  tx ]  ->  four unknowns a = s*cos, b = s*sin, tx, ty
    // [ s*sin   s*cos  ty ]
    std::vector<double> A, B;
    A.assign(size_t(4 * 4), 0.0);
    B.assign(4, 0.0);
    // Normal equations, accumulated over every point in the sample.
    for (int i : idx) {
        const Pair& q = p[size_t(i)];
        const double r[2][4] = {{ q.ax, -q.ay, 1.0, 0.0 },
                                { q.ay,  q.ax, 0.0, 1.0 }};
        const double t[2] = { q.bx, q.by };
        for (int e = 0; e < 2; ++e)
            for (int j = 0; j < 4; ++j) {
                for (int k = 0; k < 4; ++k) A[size_t(j * 4 + k)] += r[e][j] * r[e][k];
                B[size_t(j)] += r[e][j] * t[e];
            }
    }
    if (!SolveDense(A, B, 4)) return false;

    Affine o;
    o.m[0] = float(B[0]); o.m[1] = float(-B[1]); o.m[2] = float(B[2]);
    o.m[3] = float(B[1]); o.m[4] = float(B[0]);  o.m[5] = float(B[3]);
    o.m[6] = o.m[7] = 0.0f;
    *out = o;
    return true;
}

// 6 DOF: adds shear and non-uniform scale. Three points.
bool SolveAffine(const std::vector<Pair>& p, const std::vector<int>& idx, Affine* out) {
    if (idx.size() < 3) return false;
    // x and y are independent, so this is two 3x3 systems rather than one 6x6.
    std::vector<double> A(9, 0.0), Bx(3, 0.0), By(3, 0.0);
    for (int i : idx) {
        const Pair& q = p[size_t(i)];
        const double r[3] = { q.ax, q.ay, 1.0 };
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) A[size_t(j * 3 + k)] += r[j] * r[k];
            Bx[size_t(j)] += r[j] * q.bx;
            By[size_t(j)] += r[j] * q.by;
        }
    }
    std::vector<double> A2 = A;
    if (!SolveDense(A, Bx, 3)) return false;
    if (!SolveDense(A2, By, 3)) return false;

    Affine o;
    o.m[0] = float(Bx[0]); o.m[1] = float(Bx[1]); o.m[2] = float(Bx[2]);
    o.m[3] = float(By[0]); o.m[4] = float(By[1]); o.m[5] = float(By[2]);
    o.m[6] = o.m[7] = 0.0f;
    *out = o;
    return true;
}

// 8 DOF: the full homography. Four points.
//
// The model a panorama needs. A camera rotating about its own centre relates
// two frames by a homography and by nothing simpler -- an affine fit to a wide
// pan leaves a systematic error that grows toward the edges, which is exactly
// where a stitch shows it.
bool SolveHomography(const std::vector<Pair>& p, const std::vector<int>& idx,
                     Affine* out) {
    if (idx.size() < 4) return false;

    // The DLT, with h[8] fixed at 1 so the system is inhomogeneous and can be
    // solved directly. Fixing h[8] fails only for a homography that sends the
    // origin to infinity, which no camera motion produces.
    const int n = int(idx.size());
    std::vector<double> A, B;
    A.assign(size_t(8 * 8), 0.0);
    B.assign(8, 0.0);
    for (int i = 0; i < n; ++i) {
        const Pair& q = p[size_t(idx[size_t(i)])];
        const double x = q.ax, y = q.ay, u = q.bx, v = q.by;

        const double r1[8] = { x, y, 1, 0, 0, 0, -u * x, -u * y };
        const double r2[8] = { 0, 0, 0, x, y, 1, -v * x, -v * y };
        for (int j = 0; j < 8; ++j) {
            for (int k = 0; k < 8; ++k)
                A[size_t(j * 8 + k)] += r1[j] * r1[k] + r2[j] * r2[k];
            B[size_t(j)] += r1[j] * u + r2[j] * v;
        }
    }
    if (!SolveDense(A, B, 8)) return false;

    Affine o;
    for (int i = 0; i < 8; ++i) o.m[i] = float(B[size_t(i)]);
    *out = o;
    return true;
}

class AlignFeatures : public AlgorithmBase {
public:
    const char* Name()     const override { return "align_features"; }
    const char* Category() const override { return "merge"; }

    PortList Inputs() const override {
        return {{"src", DataType::ImageSet, FormatSpec::Any, ShapeSpec::Any}};
    }
    PortList Outputs() const override {
        return {{"out", DataType::ImageSet, FormatSpec::SameAsInput, ShapeSpec::SameAsInput}};
    }

    void RunCPU(RunCtx&) override {}
    bool IsAligner() const override { return true; }

    bool RunAlign(std::vector<Image>* images, std::string* err) override {
        m_solved = 0;
        m_frames = 0;
        m_inliers = 0;
        m_matches = 0;
        m_worstShift = 0.0f;
        m_chained = 0;
        m_broke = -1;
        m_rejected = -1;
        m_note.clear();

        if (!images || images->size() < 2) {
            *err = "align_features needs a group of at least two images";
            return false;
        }

        // Each frame's solved transform, indexed as the images are, for
        // composing a chain. Identity for the reference and for anything that
        // did not solve, which is the right neutral element: a frame whose own
        // solve failed contributes no correction, and one that never needed a
        // solve contributes none either.
        std::vector<Affine> solved(images->size());
        std::vector<bool>   isChained(images->size(), false);

        bool anyMatches = false;
        for (size_t i = 0; i < images->size(); ++i) {
            Image& img = (*images)[i];
            const MatchSidecar* ms = MatchesOf(img);
            if (!ms || ms->Matches().empty()) continue;
            anyMatches = true;
            ++m_frames;

            if (ms->Reference() < 0 || size_t(ms->Reference()) >= images->size()) continue;

            // A frame matched to its immediate predecessor is a CHAIN link:
            // its transform maps into that neighbour's frame, not into frame
            // 0. Composing them is what puts every frame in one coordinate
            // system -- see the loop after this one.
            //
            // Detected from the match sidecar rather than from a parameter of
            // this algorithm, so the aligner cannot disagree with the matcher
            // about which mode ran. The matcher already recorded the answer.
            const bool chainLink = (i > 0 && ms->Reference() == int(i) - 1);
            const FeatureSidecar* refF = FeaturesOf((*images)[size_t(ms->Reference())]);
            const FeatureSidecar* myF  = FeaturesOf(img);
            if (!refF || !myF) continue;

            // Pairs from the FIRST set, whose indices map one-to-one onto it --
            // `keep` records which of those indices were used, so the inlier
            // verdict can be written back against the right matches.
            std::vector<Pair> pairs;
            std::vector<int>  fromMatch;
            pairs.reserve(ms->Matches().size());
            fromMatch.reserve(ms->Matches().size());
            for (size_t mi = 0; mi < ms->Matches().size(); ++mi) {
                const Match& m = ms->Matches()[mi];
                if (m.a < 0 || size_t(m.a) >= refF->keypoints.size()) continue;
                if (m.b < 0 || size_t(m.b) >= myF->keypoints.size()) continue;
                const Keypoint& ka = refF->keypoints[size_t(m.a)];
                const Keypoint& kb = myF->keypoints[size_t(m.b)];
                pairs.push_back({ka.x, ka.y, kb.x, kb.y});
                fromMatch.push_back(int(mi));
            }
            m_matches += int(pairs.size());

            Affine t;
            int inliers = 0;
            std::vector<int> inlierIdx;
            if (!Ransac(pairs, &t, &inliers, &inlierIdx)) continue;

            // A SOLVE THAT MOVES THE FRAME FURTHER THAN THE FRAME IS WIDE is
            // rejected, however confident it looks.
            //
            // Tim hit this by dropping files on the palette: Windows handed
            // them over re-ordered, so the chain paired frame 7 with frame 0 --
            // views about 30 degrees apart with essentially no overlap. That
            // link did not fail. It reported 59 of 65 matches as inliers, 91%,
            // which is indistinguishable from the 90% a genuinely adjacent pair
            // gives, and the whole run said "solved 7 of 7 frames".
            //
            // What separated them was never the inlier RATE. It was:
            //
            //   adjacent  (0 -> 1):  543 matches kept of 5000,   1081 px shift
            //   spurious  (7 -> 0):   65 matches kept of 5000,  10908 px shift
            //
            // RANSAC is doing exactly its job in both cases -- it finds the
            // largest consistent subset, and 59 mutually consistent false
            // matches are still consistent. What it cannot know is that a
            // transform placing this frame two frame-widths away describes two
            // views that barely see the same scene, so whatever agreed did so
            // by coincidence.
            //
            // 1.5 frame widths rather than 1: a real pan with modest overlap
            // legitimately shifts most of a frame, and this must not reject
            // those. It is a bound on the absurd, not a tightness control.
            {
                const ImageDesc& d = img.Desc();
                const float shift = MaxCornerShift(t, d.width, d.height);
                const float limit = 1.5f * float(std::max(d.width, d.height));
                if (shift > limit) {
                    m_rejected = int(i);

                    // IN A CHAIN THIS IS FATAL, and saying so is the point.
                    //
                    // Tim hit the out-of-order case twice, and the second time
                    // the run still reported "solved 6 of 7 frames, 90%
                    // inliers" with the explanation appended after it -- which
                    // reads as success, and was clipped off the end of the info
                    // panel entirely. A warning nobody sees is not a warning.
                    //
                    // A chain has no way to continue past a rejected link:
                    // every later frame is positioned relative to this one, so
                    // what follows is not "slightly worse", it is unrelated.
                    // Stopping with a message beats producing a panorama with
                    // frames at right angles and leaving the user to work out
                    // why.
                    //
                    // A fixed reference is different -- the other frames are
                    // independent of this one -- so there it stays a note and
                    // the run continues.
                    if (chainLink) {
                        char buf[288];
                        std::snprintf(buf, sizeof buf,
                            "align_features: frame %d does not follow frame %d "
                            "-- it solved, but %.0f px from where it should be, "
                            "which is more than the frame is wide. The frames "
                            "are probably out of order: sort the group by name "
                            "or date (right-click the group in the palette).",
                            int(i), int(i) - 1, double(shift));
                        *err = buf;
                        return false;
                    }
                    continue;
                }
            }

            m_inliers += inliers;
            ++m_solved;
            solved[i] = t;

            // MARK THE INLIERS, on every set this frame carries.
            //
            // The first set was just verified by RANSAC, so its verdict is
            // recorded directly. The others -- the windowed neighbours -- were
            // never solved here, and they are exactly what bundle adjustment
            // reads. Left unmarked, BA takes every match the matcher produced,
            // and on a detector whose matches are 45% outliers that is half its
            // observations pulling the wrong way.
            //
            // Verified against the transform the CHAIN gives, which is the only
            // one available: frame i's position relative to frame i-w is
            // solved[i] composed back through the intervening links. That
            // transform is imperfect -- removing its error is what BA is FOR --
            // so the threshold here is deliberately loose. It is separating
            // "consistent with roughly where this frame is" from "somewhere
            // else entirely", not doing BA's job for it.
            {
                auto marked = std::make_shared<MatchSidecar>(*ms);
                const float thr = float(m_threshold);

                for (size_t si = 0; si < marked->sets.size(); ++si) {
                    MatchSet& set = marked->sets[si];
                    set.inlier.assign(set.matches.size(), 0);

                    if (si == 0) {
                        // RANSAC's own verdict, mapped back through the index
                        // list built alongside `pairs`.
                        for (int pi : inlierIdx)
                            if (pi >= 0 && size_t(pi) < fromMatch.size())
                                set.inlier[size_t(fromMatch[size_t(pi)])] = 1;
                        continue;
                    }

                    // A windowed set: verify it against the chain's estimate.
                    if (set.reference < 0 || size_t(set.reference) >= images->size())
                        continue;
                    const FeatureSidecar* rf = FeaturesOf((*images)[size_t(set.reference)]);
                    if (!rf) continue;

                    // reference -> this frame, via frame 0. solved[] is filled
                    // ascending, so both are known by the time this runs.
                    bool ok = false;
                    const Affine refInv = solved[size_t(set.reference)].Inverse(&ok);
                    if (!ok) continue;
                    const Affine rel = t.Then(refInv);

                    // Loose: three times the RANSAC threshold, because `rel`
                    // carries the chain error BA has not removed yet.
                    const float loose = 3.0f * thr;
                    for (size_t k = 0; k < set.matches.size(); ++k) {
                        const Match& m = set.matches[k];
                        if (m.a < 0 || size_t(m.a) >= rf->keypoints.size()) continue;
                        if (m.b < 0 || size_t(m.b) >= myF->keypoints.size()) continue;
                        const Keypoint& ka = rf->keypoints[size_t(m.a)];
                        const Keypoint& kb = myF->keypoints[size_t(m.b)];
                        float px, py;
                        rel.MapPoint(ka.x, ka.y, &px, &py);
                        const float dx = px - kb.x, dy = py - kb.y;
                        if (dx * dx + dy * dy <= loose * loose) set.inlier[k] = 1;
                    }
                }
                img.Sidecars().Set(kMatchSidecar, marked);
            }

            // Flagged only AFTER the solve succeeded, which is the whole point
            // of it being here rather than beside the chainLink test above.
            //
            // The first version set this as soon as the sidecar named the
            // previous frame, before Ransac had run. A frame whose solve then
            // failed -- too few matches is the ordinary reason -- kept the flag
            // with an IDENTITY transform, and the chain loop below dutifully
            // composed it: the failed frame inherited its predecessor's
            // position exactly, and every frame after it was short by one link.
            //
            // That is the worst shape a bug can take here. It is not a crash
            // and not a visibly broken frame; it is a panorama where one frame
            // sits exactly on top of its neighbour and everything downstream is
            // offset, which reads as ghosting rather than as a failure. The
            // test that caught it asserts frame 2 lands at TWICE the step, and
            // it landed at one.
            isChained[i] = chainLink;

            // A chained frame is not attached here: its transform is still
            // relative to its neighbour, and only means something once the
            // chain below has composed it.
            if (chainLink) continue;

            // Composed with whatever is already attached, so this can follow a
            // coarser solve or precede the pixel refiner. Tim's requirement,
            // and the reason the sidecar is read rather than overwritten.
            const Affine prior = TransformOf(img);
            AttachTransform(&img, prior.IsIdentity() ? t : t.Then(prior));

            const ImageDesc& d = img.Desc();
            m_worstShift = std::max(m_worstShift,
                                    MaxCornerShift(t, d.width, d.height));
        }

        // COMPOSE THE CHAIN.
        //
        // A chained solve gives T[i], mapping frame i-1's coordinates into
        // frame i. What every consumer wants is a map from ONE common frame
        // into each image, so the links have to be accumulated:
        //
        //     C[0] = I,   C[i] = T[i] . C[i-1]
        //
        // Frame 0's coordinate system becomes the panorama's, which is a choice
        // and not the only one -- a centre frame would halve the accumulated
        // error at each end. It is the honest default though, because it is the
        // one a script can reason about: the reference frame is where it says.
        //
        // ERROR ACCUMULATES ALONG THE CHAIN, and nothing here hides that. Each
        // link carries its own residual and composition multiplies them, so the
        // far end of a long sweep is the least certain part of the result. That
        // is inherent to sequential alignment; removing it takes a global
        // bundle adjustment over all pairs at once, which is a different and
        // much larger algorithm. What this does instead is REPORT the chain
        // length, so a suspicious far end has a visible cause.
        // ASCENDING, and that order is load-bearing: composing frame i needs
        // frame i-1 ALREADY composed, not merely solved. Running this loop
        // backwards, or in parallel, silently produces transforms that are each
        // relative to their neighbour rather than to frame 0 -- which looks
        // like a stitch where every seam is individually fine and the whole is
        // wrong.
        // A BROKEN LINK ENDS THE CHAIN, rather than being stepped over.
        //
        // If frame i never solved, nothing downstream can be placed relative to
        // frame 0: frame i+1's transform is relative to frame i, whose own
        // position is unknown. Composing anyway would put every later frame at
        // whatever offset the missing link happened to leave behind -- the
        // failure mode described above, one frame stacked on its neighbour.
        //
        // So the chain simply stops, and the frames past the break keep no
        // transform at all. They then merge unaligned, which is visibly wrong
        // in the right way: a stitch that is obviously missing its tail beats
        // one that is subtly ghosted throughout.

        m_chained = 0;
        m_broke = -1;
        bool live = true;
        for (size_t i = 1; i < images->size(); ++i) {
            if (!isChained[i]) {
                // Frame i is not a chain link. Either it solved against a fixed
                // reference (already attached above and fine), or it did not
                // solve at all -- and if a LINK was expected here, the chain is
                // broken from this point on.
                const MatchSidecar* ms = MatchesOf((*images)[i]);
                if (live && ms && ms->Reference() == int(i) - 1) {
                    live = false;
                    m_broke = int(i);
                }
                continue;
            }
            if (!live) continue;

            ++m_chained;
            solved[i] = solved[i].Then(solved[i - 1]);

            Image& img = (*images)[i];
            const Affine prior = TransformOf(img);
            AttachTransform(&img, prior.IsIdentity() ? solved[i]
                                                     : solved[i].Then(prior));

            const ImageDesc& d = img.Desc();
            m_worstShift = std::max(m_worstShift,
                                    MaxCornerShift(solved[i], d.width, d.height));
        }

        if (!anyMatches) {
            *err = "align_features: no matches attached -- run a detector and "
                   "a matcher before aligning";
            return false;
        }
        return true;
    }

    std::string RunReport() const override {
        if (!m_note.empty()) return m_note;
        if (m_frames == 0) return {};
        char buf[256];
        char chain[64] = "";
        if (m_chained > 0)
            std::snprintf(chain, sizeof chain, ", %d chained", m_chained);
        char broke[192] = "";
        if (m_rejected >= 0)
            std::snprintf(broke, sizeof broke,
                          " -- frame %d REJECTED: it solved, but to a shift larger "
                          "than the frame. Usually the frames are out of order.",
                          m_rejected);
        else if (m_broke >= 0)
            std::snprintf(broke, sizeof broke,
                          " -- CHAIN BROKE at frame %d, later frames unaligned",
                          m_broke);
        std::snprintf(buf, sizeof buf,
                      "%s: solved %d of %d frames%s, %d of %d matches were inliers "
                      "(%.0f%%), largest shift %.1f px%s",
                      ModelName(), m_solved, m_frames, chain, m_inliers, m_matches,
                      m_matches ? 100.0 * double(m_inliers) / double(m_matches) : 0.0,
                      double(m_worstShift), broke);
        return buf;
    }

    bool HasGPU() const override { return false; }

private:
    const char* ModelName() const {
        switch (std::clamp(int(m_model), 0, 2)) {
            case 0:  return "similarity";
            case 1:  return "affine";
            default: return "homography";
        }
    }

    int MinSample() const {
        switch (std::clamp(int(m_model), 0, 2)) {
            case 0:  return 2;
            case 1:  return 3;
            default: return 4;
        }
    }

    // Solves in NORMALISED coordinates and un-normalises the answer.
    //
    // Applied here rather than inside each model, so all three get it and none
    // can be forgotten. The composition at the end is ordinary: if Na maps the
    // reference into normalised space and Nb does the same for this frame, and
    // Hn is the transform between the normalised sets, then the transform
    // between the originals is Nb^-1 * Hn * Na.
    bool Solve(const std::vector<Pair>& p, const std::vector<int>& idx,
               Affine* out) const {
        Norm na, nb;
        na.Fit(p, idx, true);
        nb.Fit(p, idx, false);

        std::vector<Pair> np;
        np.reserve(idx.size());
        std::vector<int> ni;
        ni.reserve(idx.size());
        for (size_t k = 0; k < idx.size(); ++k) {
            const Pair& q = p[size_t(idx[k])];
            Pair r;
            na.Apply(q.ax, q.ay, &r.ax, &r.ay);
            nb.Apply(q.bx, q.by, &r.bx, &r.by);
            np.push_back(r);
            ni.push_back(int(k));
        }

        Affine hn;
        bool ok = false;
        switch (std::clamp(int(m_model), 0, 2)) {
            case 0:  ok = SolveSimilarity(np, ni, &hn); break;
            case 1:  ok = SolveAffine(np, ni, &hn);     break;
            default: ok = SolveHomography(np, ni, &hn); break;
        }
        if (!ok) return false;

        bool invOk = false;
        const Affine nbInv = nb.Matrix().Inverse(&invOk);
        if (!invOk) return false;

        *out = nbInv.Then(hn).Then(na.Matrix());
        return true;
    }

    // How far the worst corner moves, which is the honest measure of a warp --
    // translation alone misses a rotation about the centre entirely.
    static float MaxCornerShift(const Affine& t, int w, int h) {
        const float cx[4] = {0.0f, float(w), 0.0f, float(w)};
        const float cy[4] = {0.0f, 0.0f, float(h), float(h)};
        float worst = 0.0f;
        for (int i = 0; i < 4; ++i) {
            float ox, oy;
            t.MapPoint(cx[i], cy[i], &ox, &oy);
            const float dx = ox - cx[i], dy = oy - cy[i];
            worst = std::max(worst, std::sqrt(dx * dx + dy * dy));
        }
        return worst;
    }

    // `inlierIdx`, when given, receives the indices into `pairs` that agreed
    // with the winning transform -- so a caller can mark those matches and
    // everything downstream can use the geometric verdict rather than
    // recomputing it or ignoring it.
    bool Ransac(const std::vector<Pair>& pairs, Affine* out, int* inlierCount,
                std::vector<int>* inlierIdx = nullptr) const {
        const int need = MinSample();
        if (int(pairs.size()) < need) return false;

        const float thresh = float(m_threshold);
        const float threshSq = thresh * thresh;
        const int   iters = std::max(1, int(m_iterations));

        // Deterministic, so the same input gives the same transform every run.
        //
        // A randomised algorithm whose answer changes between runs would make a
        // slider drag jitter and would make any measurement here unrepeatable
        // -- which matters more in a lab than the theoretical independence a
        // random seed buys.
        std::mt19937 rng(20260829u);
        std::uniform_int_distribution<size_t> pick(0, pairs.size() - 1);

        // resize() rather than a sized constructor -- the latter parses as a
        // function declaration and the errors land on the uses.
        std::vector<int> sample;
        sample.resize(size_t(need));
        std::vector<int> best;
        Affine bestT;

        for (int it = 0; it < iters; ++it) {
            // A sample of DISTINCT points. Repeats would make the system
            // singular, and rejecting the sample afterwards wastes the
            // iteration.
            for (int s = 0; s < need; ++s) {
                bool dup;
                int tries = 0;
                do {
                    sample[size_t(s)] = int(pick(rng));
                    dup = false;
                    for (int q = 0; q < s; ++q)
                        if (sample[size_t(q)] == sample[size_t(s)]) dup = true;
                } while (dup && ++tries < 16);
            }

            Affine t;
            if (!Solve(pairs, sample, &t)) continue;

            std::vector<int> in;
            in.reserve(pairs.size());
            for (size_t i = 0; i < pairs.size(); ++i) {
                float ox, oy;
                t.MapPoint(pairs[i].ax, pairs[i].ay, &ox, &oy);
                const float dx = ox - pairs[i].bx, dy = oy - pairs[i].by;
                if (dx * dx + dy * dy <= threshSq) in.push_back(int(i));
            }

            if (in.size() > best.size()) {
                best.swap(in);
                bestT = t;
                // Early exit when almost everything agrees: more iterations
                // cannot improve on a consensus this large, and a burst of
                // near-identical frames hits this immediately.
                if (best.size() > pairs.size() * 9 / 10) break;
            }
        }

        if (int(best.size()) < std::max(need, int(m_minInliers))) return false;

        // REFIT ON EVERY INLIER, which is the step that makes RANSAC accurate
        // rather than merely robust. The sample that won was chosen for how
        // many points agree with it, not for how well it fits them -- a
        // minimal sample is exactly determined and fits its own points
        // perfectly whatever their noise. The refit averages that noise away.
        Affine refined;
        if (Solve(pairs, best, &refined)) *out = refined;
        else                              *out = bestT;

        *inlierCount = int(best.size());
        if (inlierIdx) *inlierIdx = best;
        return true;
    }

    static constexpr const char* kModelNames[] = {
        "similarity (4 DOF)", "affine (6 DOF)", "homography (8 DOF)"};

    Param<int> m_model{this, "model", 1, 0, 2,
        {.help = "How much freedom the fit is given. Similarity is shift, "
                 "rotate and scale; affine adds shear; homography adds "
                 "perspective. A tripod pan needs a homography and nothing "
                 "simpler; a handheld bracket is better served by affine, "
                 "whose fewer parameters are each better determined.",
         .choices = kModelNames, .choiceCount = 3}};

    Param<float> m_threshold{this, "threshold", 3.0f, 0.5f, 20.0f,
        {.help = "How far a match may reproject and still count as an inlier, "
                 "in pixels. This is a POINT-to-point test, not an epipolar "
                 "one: the camera centre does not move between these frames, "
                 "so a match has an exact predicted position rather than a "
                 "line to lie near.",
         .step = 0.25, .softMax = 8.0}};

    Param<int> m_iterations{this, "iterations", 2000, 10, 50000,
        {.help = "RANSAC samples to try. More is safer when the match set is "
                 "mostly outliers; the search exits early once nine tenths of "
                 "the matches agree, so a clean pair costs far fewer."}};

    Param<int> m_minInliers{this, "min_inliers", 8, 3, 1000,
        {.help = "Fewest inliers that counts as a solve. Below this the frame "
                 "is left un-warped rather than warped by a fit nothing "
                 "supports -- an unaligned frame merges slightly soft, where a "
                 "wrong warp merges as noise."}};

    int         m_solved = 0, m_frames = 0;
    int         m_inliers = 0, m_matches = 0;
    // How many frames were composed along a chain rather than solved directly
    // against the reference. Reported because error accumulates with it.
    int         m_chained = 0;
    // Where a chain stopped, or -1. Reported because a panorama missing its
    // tail should say so rather than leave the user to notice.
    int         m_broke = -1;
    // A frame whose solve was rejected as implausible, or -1. Reported because
    // the usual cause is frames in the wrong ORDER, which the user can fix.
    int         m_rejected = -1;
    float       m_worstShift = 0.0f;
    std::string m_note;
};

} // namespace

REGISTER_ALGORITHM(AlignFeatures);

} // namespace tglab
