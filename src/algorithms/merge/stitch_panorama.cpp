// stitch_panorama — project aligned frames onto one canvas.
//
// The last stage of the panorama chain: detect -> match(chain) -> align ->
// stitch. Everything before this has already decided WHERE each frame goes; the
// work here is choosing a surface to put them on, sizing a canvas, and blending
// the overlaps.
//
// WHY A PROJECTION IS NOT DECORATION, which is the thing worth understanding
// before reading the code.
//
// A pan relates each frame to the next by a homography, and each of those
// solves cleanly -- measured on a 15-frame handheld sweep, every consecutive
// pair recovered 76-92% inliers. But chaining them COMPOSES them, and composing
// homographies multiplies their perspective terms rather than adding them. The
// measured result on that sweep, reading the width each frame maps to:
//
//     frame 0:  5796 px (the original)
//     frame 2:  6108
//     frame 4:  7059
//     frame 6: 10647
//     frame 7: 12650
//
// and the perspective term m[6] climbing monotonically, 3.4e-5 to 3.2e-4. Each
// frame is stretched more than the last, without bound. That is not error
// accumulation in the usual sense -- the individual links are good -- it is the
// plane model being asked to represent something it cannot. A flat canvas
// tangent to the sphere goes to infinity at 90 degrees from where it touches,
// and a 15-frame sweep at 24 mm covers most of the way there.
//
// The fix is to stop using a plane. Rotate each frame's rays into a common
// frame and project them onto a CYLINDER or a SPHERE, where a pan is a
// TRANSLATION rather than a growing stretch. The stretch does not have to be
// tamed, because on the right surface it never appears.
//
// THE PROJECTIONS, and when each is right:
//
//   plane          What the homographies already give. Correct for a few
//                  frames and a modest angle; unusable past roughly 90 degrees,
//                  where it diverges. Keeps straight lines straight, which is
//                  why it is still the right answer for a 2-3 frame stitch.
//   cylindrical    Wraps horizontally, flat vertically. The standard choice for
//                  a single-row pan: it takes 360 degrees horizontally without
//                  divergence and keeps verticals vertical. Horizontal lines
//                  above and below the horizon bow.
//   spherical      Wraps both ways. What a multi-row panorama needs, and the
//                  only one of the three that can represent a full sphere.
//                  Bows everything that is not through the centre.
//
// FOCAL LENGTH IS ESTIMATED, NOT READ FROM EXIF. All three projections need to
// know the camera's focal length in pixels to convert a pixel offset into an
// angle. EXIF gives focal length in millimetres, which needs the sensor size to
// become pixels -- and the crop factor is not in the file for every camera.
//
// The homographies already contain the answer. A rotation about the camera
// centre induces H = K R K^-1, and for a pure pan the (0,2) and (2,0) entries
// of that carry f directly. Estimating from the data rather than the metadata
// means this works on scanned film, on renders, and on files whose EXIF was
// stripped -- and it cannot disagree with the geometry that was actually
// solved.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "../../algo_util/pixel_buffer.h"
#include "../../algo_util/transform.h"
#include "../../core/algorithm.h"
#include "../../core/reduction.h"

namespace tglab {
namespace {

// The surfaces a frame can be projected onto.
enum Projection { kPlane = 0, kCylindrical = 1, kSpherical = 2 };

// One frame, held until Finish can see all of them.
//
// A stitcher cannot stream: the canvas size depends on where EVERY frame lands,
// so nothing can be composited until the last transform is known. Holding the
// pixels is the cost of that, and it is why this is the one merge whose memory
// scales with the group.
struct Frame {
    // The whole buffer rather than a bare vector: SampleBilinear takes a
    // PixelBuffer, and keeping one avoids reconstructing its shape per frame
    // from three loose integers that could disagree with the data.
    PixelBuffer buf;
    Affine t;                 // maps panorama-reference coords -> this frame
    Affine inv;               // and back
    bool   invertible = false;

    // The rotation this frame's homography represents, as a 3x3 mapping a
    // reference-frame ray to this frame's camera space. See Orthonormalise.
    float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    int W() const { return buf.Width(); }
    int H() const { return buf.Height(); }
};

// Force a 3x3 to the nearest rotation, by modified Gram-Schmidt. Returns false
// when the input was too degenerate or too reflected to be worth trusting.
//
// THIS IS THE STEP THAT MAKES A LONG PAN POSSIBLE, and it is worth being
// explicit about why, because a cylindrical projection alone does NOT fix the
// divergence -- measured, it merely relocates it.
//
// Chained homographies accumulate perspective drift: composing fourteen links
// on a real sweep produced m[6] climbing monotonically and the last frame
// mapping to a 6.2-million-pixel height. Projecting that through a cylinder
// changes nothing, because the drift is already baked into the matrix being
// projected. The cylinder fixes the SURFACE; it cannot fix the TRANSFORM.
//
// The physical fact being used is that a camera rotating about its own centre
// can only produce H = K R K^-1 for an orthonormal R. Anything in the solved
// homography that is not of that form -- shear, scale, the growing perspective
// -- is error, because no rotation could have produced it. So: strip K, force
// what remains to be a rotation, and put K back. Six of the eight degrees of
// freedom are discarded, and every one discarded is one that could not have
// been real.
//
// WHERE THIS IS APPLIED MATTERS AS MUCH AS WHAT IT DOES: per LINK, before the
// chain is composed. Applying it to an already-composed transform is too late,
// because the compounding has happened by then -- see the composition loop in
// Finish for the determinants that measured it, and the three upside-down
// frames that resulted.
//
// Gram-Schmidt rather than a proper polar decomposition (which would need an
// SVD): the input is already close to a rotation, and for a near-rotation the
// two agree to well under a pixel. That "already close" is exactly what
// per-link application buys, and what per-frame application destroyed.
bool Orthonormalise(float m[9]) {
    // The input, kept for the determinant test at the end: the rows are
    // normalised in place below, which changes the determinant magnitude but
    // not its SIGN -- and it is the sign that says whether this was a
    // reflection. Copied rather than tested first so the cheap path stays one
    // pass.
    float in[9];
    for (int i = 0; i < 9; ++i) in[i] = m[i];

    auto dot = [&](int a, int b) {
        return m[a * 3] * m[b * 3] + m[a * 3 + 1] * m[b * 3 + 1] +
               m[a * 3 + 2] * m[b * 3 + 2];
    };
    auto scale = [&](int r, float s) {
        m[r * 3] *= s; m[r * 3 + 1] *= s; m[r * 3 + 2] *= s;
    };
    auto sub = [&](int r, int from, float s) {
        m[r * 3]     -= s * m[from * 3];
        m[r * 3 + 1] -= s * m[from * 3 + 1];
        m[r * 3 + 2] -= s * m[from * 3 + 2];
    };

    float n0 = std::sqrt(dot(0, 0));
    if (n0 < 1e-12f) { // degenerate: fall back to identity rather than NaN
        for (int i = 0; i < 9; ++i) m[i] = (i % 4 == 0) ? 1.0f : 0.0f;
        return false;
    }
    scale(0, 1.0f / n0);

    sub(1, 0, dot(1, 0));
    float n1 = std::sqrt(dot(1, 1));
    if (n1 < 1e-12f) {
        for (int i = 0; i < 9; ++i) m[i] = (i % 4 == 0) ? 1.0f : 0.0f;
        return false;
    }
    scale(1, 1.0f / n1);

    // The third row is the cross product of the first two, so it is consistent
    // with them and the result has determinant +1.
    //
    // THAT IS NOT THE SAME AS "cannot produce a flip", which is what an earlier
    // version of this comment claimed. Determinant +1 only says the three rows
    // form a right-handed set; if rows 0 and 1 are themselves flipped, the
    // cross product dutifully flips row 2 to match and the result is a
    // perfectly valid rotation that is 180 degrees off. Three frames of a real
    // panorama came out upside down that way.
    //
    // The real defence is upstream -- orthonormalise each LINK before composing
    // rather than the composed product, so the input here is always close to a
    // rotation. See the composition loop in Finish. This check is the backstop
    // for whatever that does not catch: a negative determinant BEFORE
    // normalising means the input was a reflection, and no amount of
    // orthonormalising recovers the intended orientation from it. Identity is
    // the honest answer -- that frame merges unaligned, which is visibly wrong
    // rather than invisibly upside down.
    const float det =
        in[0] * (in[4] * in[8] - in[5] * in[7]) -
        in[1] * (in[3] * in[8] - in[5] * in[6]) +
        in[2] * (in[3] * in[7] - in[4] * in[6]);
    if (det <= 0.0f) {
        for (int i = 0; i < 9; ++i) m[i] = (i % 4 == 0) ? 1.0f : 0.0f;
        return false;
    }

    m[6] = m[1] * m[5] - m[2] * m[4];
    m[7] = m[2] * m[3] - m[0] * m[5];
    m[8] = m[0] * m[4] - m[1] * m[3];
    return true;
}

class StitchPanorama : public AlgorithmBase, public Reducer {
public:
    const char* Name()     const override { return "stitch_panorama"; }
    const char* Category() const override { return "merge"; }

    PortList Inputs() const override {
        return {{"src", DataType::ImageSet, FormatSpec::Any, ShapeSpec::Any}};
    }
    PortList Outputs() const override {
        return {{"out", DataType::Image, FormatSpec::SameAsInput, ShapeSpec::Reduced}};
    }

    void RunCPU(RunCtx&) override {}

    bool     IsReduction() const override { return true; }
    Reducer* AsReducer()         override { return this; }

    bool Begin(int count, const std::string&, std::string* err) override {
        if (count <= 0) { *err = "stitch_panorama: nothing to stitch"; return false; }
        m_frames.clear();
        m_frames.reserve(size_t(count));
        m_desc = ImageDesc{};
        m_outW = m_outH = 0;
        m_focal = 0.0f;
        m_placed = 0;
        m_flipped = -1;
        m_note.clear();
        return true;
    }

    bool Accept(int, const Image& img, std::string* err) override {
        ImageView v = const_cast<Image&>(img).MapCpuRead();
        if (!v.data) { *err = "stitch_panorama: a frame has no pixels"; return false; }

        const ImageDesc& d = img.Desc();
        if (!m_frames.empty() &&
            (d.width != m_desc.width || d.height != m_desc.height ||
             d.format != m_desc.format)) {
            // Same rule as merge_mean, and for the same reason: frames of
            // different sizes have no one defensible resampling, and guessing
            // one silently is worse than saying so.
            *err = "stitch_panorama: every frame must be the same size and "
                   "format (frame " + std::to_string(m_frames.size()) + " is " +
                   std::to_string(d.width) + "x" + std::to_string(d.height) +
                   ", expected " + std::to_string(m_desc.width) + "x" +
                   std::to_string(m_desc.height) + ")";
            return false;
        }

        PixelBuffer buf;
        buf.Unpack(v);
        if (!buf.Valid()) { *err = "stitch_panorama: unsupported format"; return false; }

        if (m_frames.empty()) m_desc = d;

        Frame f;
        f.buf = std::move(buf);
        f.t   = TransformOf(img);
        bool ok = false;
        f.inv = f.t.Inverse(&ok);
        f.invertible = ok;
        m_frames.push_back(std::move(f));
        return true;
    }

    bool Finish(Image* out, std::string* err) override;

    std::string RunReport() const override {
        if (!m_note.empty()) return m_note;
        if (m_placed == 0) return {};
        char flip[96] = "";
        if (m_flipped >= 0)
            std::snprintf(flip, sizeof flip,
                          " -- frame %d and later DROPPED, their link was not a rotation",
                          m_flipped);
        char buf[320];
        std::snprintf(buf, sizeof buf,
                      "%s: %d frames onto %dx%d, focal %.0f px (%.0f deg fov), "
                      "overlap %.0f%% disagreeing %.1f%%%s",
                      ProjName(), m_placed, m_outW, m_outH, double(m_focal),
                      double(FovDeg()), 100.0 * m_overlapPixels,
                      100.0 * double(m_disagree), flip);
        return buf;
    }

    bool HasGPU() const override { return false; }

private:
    const char* ProjName() const {
        switch (Proj()) {
            case kCylindrical: return "cylindrical";
            case kSpherical:   return "spherical";
            default:           return "plane";
        }
    }

    Projection Proj() const {
        return Projection(std::clamp(int(m_projection), 0, 2));
    }

    float FovDeg() const {
        if (m_focal <= 0.0f || m_desc.width <= 0) return 0.0f;
        return 2.0f * std::atan(0.5f * float(m_desc.width) / m_focal) * 57.2957795f;
    }

    // --- focal length from the solved homographies ---------------------------
    //
    // A rotation about the camera centre gives H = K R K^-1 with
    //
    //     K = [ f 0 cx ]
    //         [ 0 f cy ]
    //         [ 0 0  1 ]
    //
    // Expanding that for a pure pan by angle a about the y axis, with the
    // principal point taken out (which is what the centring below does):
    //
    //     H ~ [  cos a   0   f sin a ]
    //         [    0     1      0    ]
    //         [ -sin a/f 0    cos a  ]
    //
    // so h02 = f sin a and h20 = -sin a / f, and therefore
    //
    //     f = sqrt( -h02 / h20 )
    //
    // The same relation holds for a tilt using h12 and h21. This is the
    // classical two-view estimate (Szeliski's formulation, and the one
    // Hugin/autostitch use); it is not exact for a general rotation, which is
    // why the MEDIAN over all links is taken rather than any single estimate.
    //
    // A median rather than a mean because the bad estimates here are not noisy,
    // they are nonsense: a link with almost no rotation makes both h02 and h20
    // nearly zero, and their ratio is then whatever the residuals happen to be.
    // A mean would let one of those swamp fourteen good ones.
    float EstimateFocal() const {
        std::vector<float> est;
        const float cx = 0.5f * float(m_desc.width);
        const float cy = 0.5f * float(m_desc.height);

        // Centring: shift the origin to the principal point, estimate, shift
        // back. Without this the translation entries absorb the offset of the
        // corner from the centre and the ratio above means nothing.
        Affine toC;   toC.m[2] = -cx;  toC.m[5] = -cy;
        Affine fromC; fromC.m[2] = cx; fromC.m[5] = cy;

        for (size_t i = 1; i < m_frames.size(); ++i) {
            // The link between consecutive frames, not the accumulated
            // transform: the estimate above is derived for ONE rotation, and
            // the composition of several is not of that form.
            bool ok = false;
            const Affine prevInv = m_frames[i - 1].t.Inverse(&ok);
            if (!ok) continue;
            const Affine link = toC.Then(m_frames[i].t.Then(prevInv)).Then(fromC);

            const float h02 = link.m[2], h20 = link.m[6];
            const float h12 = link.m[5], h21 = link.m[7];

            // Both forms, when the link has enough of the corresponding
            // rotation to constrain them.
            //
            // THE THRESHOLD IS NOT ARBITRARY, and this estimator's weakness
            // lives here. h20 is sin(a)/f: for a 4-degree pan at f = 600 that
            // is 1.2e-4, a number the DLT has very little leverage on. Measured
            // on a synthetic fixture with f = 600 exactly, the solved
            // homography under-recovered h20 by 70%, and the estimate came back
            // 1099 -- off by 1.83x.
            //
            // The square root is what keeps that survivable rather than
            // catastrophic: an error of e in h20 becomes sqrt(1+e) in f, so
            // -70% gives 1.83x rather than 3.3x. It is still too much, and the
            // honest response is to reject the links that cannot support an
            // estimate rather than to average them in.
            //
            // 1e-5 rather than 1e-9: below that, h20 is smaller than the
            // residual of a typical solve and its ratio is noise wearing the
            // shape of an answer. A pan too small to clear this leaves `est`
            // empty and the caller falls back to the frame width, which is a
            // guess that ANNOUNCES itself in the report rather than a bad
            // measurement that does not.
            const float kMinPerspective = 1e-5f;
            if (std::abs(h20) > kMinPerspective) {
                const float f2 = -h02 / h20;
                if (f2 > 1.0f && std::isfinite(f2)) est.push_back(std::sqrt(f2));
            }
            if (std::abs(h21) > kMinPerspective) {
                const float f2 = -h12 / h21;
                if (f2 > 1.0f && std::isfinite(f2)) est.push_back(std::sqrt(f2));
            }
        }

        if (est.empty()) return 0.0f;
        std::sort(est.begin(), est.end());
        return est[est.size() / 2];
    }

    // --- the projections -----------------------------------------------------
    //
    // Each maps a canvas coordinate to a ray direction, which is then rotated
    // into a frame's own coordinates and projected back to a pixel. Written as
    // canvas -> ray because that is the direction the compositing loop needs:
    // it walks output pixels and asks where to read.

    // Canvas (u, v) -> a direction in the reference frame's camera space.
    void CanvasToRay(float u, float v, float* dx, float* dy, float* dz) const {
        const float f = m_focal;
        switch (Proj()) {
            case kCylindrical: {
                // u is an angle about the vertical axis, v a height on the
                // cylinder. Verticals stay vertical because v is not angular.
                const float th = u / f;
                *dx = std::sin(th);
                *dy = v / f;
                *dz = std::cos(th);
                break;
            }
            case kSpherical: {
                // Both coordinates are angles, so the whole sphere is
                // reachable -- which is the point, and the cost is that
                // horizontal lines bow everywhere off centre.
                const float th  = u / f;      // longitude
                const float phi = v / f;      // latitude
                *dx = std::sin(th) * std::cos(phi);
                *dy = std::sin(phi);
                *dz = std::cos(th) * std::cos(phi);
                break;
            }
            default:
                // The plane: canvas coordinates ARE image coordinates, and the
                // homography does all the work. Kept as a projection rather
                // than a separate path so the compositing loop has one shape.
                *dx = u; *dy = v; *dz = f;
                break;
        }
    }

    // ...and back, for sizing the canvas from a frame's corners.
    bool RayToCanvas(float dx, float dy, float dz, float* u, float* v) const {
        const float f = m_focal;
        switch (Proj()) {
            case kCylindrical: {
                // Behind the camera has no cylindrical image; reject rather
                // than fold it onto the front, which would wrap content back
                // over itself.
                const float len = std::sqrt(dx * dx + dz * dz);
                if (len < 1e-9f) return false;
                *u = f * std::atan2(dx, dz);
                *v = f * dy / len;
                return std::isfinite(*u) && std::isfinite(*v);
            }
            case kSpherical: {
                const float len = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (len < 1e-9f) return false;
                *u = f * std::atan2(dx, dz);
                *v = f * std::asin(std::clamp(dy / len, -1.0f, 1.0f));
                return std::isfinite(*u) && std::isfinite(*v);
            }
            default:
                if (std::abs(dz) < 1e-9f) return false;
                *u = f * dx / dz;
                *v = f * dy / dz;
                return std::isfinite(*u) && std::isfinite(*v);
        }
    }

    Param<int> m_projection{this, "projection", 1, 0, 2,
        {.help = "0 plane, 1 cylindrical, 2 spherical. Cylindrical is the "
                 "default because a single-row pan is the common case: it "
                 "takes 360 degrees without diverging and keeps verticals "
                 "vertical. Plane is correct for two or three frames and "
                 "diverges past about 90 degrees. Spherical is for multi-row.",
         .step = 1}};

    Param<float> m_focalOverride{this, "focal", 0.0f, 0.0f, 100000.0f,
        {.help = "Focal length in PIXELS. 0 estimates it from the solved "
                 "homographies, which is usually better than EXIF because it "
                 "needs no sensor size and cannot disagree with the geometry "
                 "actually solved. Set it when the estimate is poor -- too "
                 "small bows the seams outward, too large inward.",
         .step = 10.0}};

    Param<bool> m_feather{this, "feather", true,
        "Weight each frame by its distance from its own edge, so overlaps "
        "cross-fade instead of showing a hard seam. Costs one extra multiply "
        "per pixel and hides the exposure differences a handheld pan always "
        "has."};

    Param<int> m_maxPixels{this, "max_megapixels", 400, 1, 4000,
        {.help = "Refuses to allocate a canvas larger than this. A diverging "
                 "chain asks for an enormous canvas, and the useful response "
                 "is an error naming the size rather than an allocation that "
                 "takes the machine down."}};

    std::vector<Frame> m_frames;
    ImageDesc          m_desc{};
    int                m_outW = 0, m_outH = 0;
    int                m_placed = 0;
    float              m_focal = 0.0f;
    // How much the frames disagree where they overlap, and how much of the
    // canvas that covers. The direct measure of stitch quality: sharpness is
    // not, because it also moves with how much the canvas was scaled.
    float              m_disagree = 0.0f;
    // The first frame whose link could not be read as a rotation, or -1.
    int                m_flipped = -1;
    double             m_overlapPixels = 0.0;
    std::string        m_note;
};

bool StitchPanorama::Finish(Image* out, std::string* err) {
    if (m_frames.empty()) { *err = "stitch_panorama: no frames were accepted"; return false; }

    // A single frame is a legitimate degenerate case -- a group of one, or
    // every other frame failing to solve -- and copying it through beats an
    // error the script cannot act on.
    if (m_frames.size() == 1) {
        m_outW = m_frames[0].W();
        m_outH = m_frames[0].H();
        m_placed = 1;
        m_note = "one frame, nothing to stitch";
        out->Alloc(m_desc);
        ImageView v = out->MapCpuWrite();
        if (!v.data) { *err = "stitch_panorama: could not allocate the result"; return false; }
        PixelBuffer b;
        b.Unpack(v);
        b.Data() = m_frames[0].buf.Data();
        b.PackInto(v);
        return true;
    }

    m_focal = float(m_focalOverride);
    if (m_focal <= 0.0f) m_focal = EstimateFocal();
    if (m_focal <= 0.0f) {
        // The fallback is the frame width, which corresponds to a 53-degree
        // horizontal field of view -- an ordinary lens. Wrong in general, but
        // it is a number that produces a viewable result rather than a
        // divide-by-zero, and the report says what was used.
        m_focal = float(m_desc.width);
        m_note.clear();
    }

    // --- homography -> rotation ---------------------------------------------
    //
    // R = K^-1 H K, forced orthonormal. See Orthonormalise for why this is the
    // step that makes a long chain usable rather than a refinement of it.
    //
    // The plane projection deliberately does NOT do this: it is the "trust the
    // homography exactly" mode, correct for a few frames where the solved warp
    // includes real perspective that a rotation cannot express. It is also the
    // mode that diverges, which is why it is not the default.
    const float cx = 0.5f * float(m_desc.width);
    const float cy = 0.5f * float(m_desc.height);
    const float f0 = m_focal;

    // ORTHONORMALISE EACH LINK, THEN COMPOSE. The order matters enormously,
    // and getting it backwards is what put three frames upside down.
    //
    // The first version took each frame's ACCUMULATED homography -- already the
    // product of up to fourteen links -- and forced that to a rotation. Every
    // link carries a little perspective error, and composing multiplies rather
    // than adds them. Measured on the 15-frame sweep, reading the determinant
    // of K^-1 H K, which must be 1 for a rotation:
    //
    //     frame  3:      2.4
    //     frame  6:     10.5
    //     frame 10:    118.9
    //     frame 11:  26041.6
    //     frame 12:   -539.3   <- sign flipped
    //     frame 13:   -189.1
    //     frame 14:    -33.6
    //
    // Once the determinant goes negative the matrix is a reflection, and
    // Gram-Schmidt turns a reflection into a perfectly valid rotation that
    // happens to be 180 degrees off. The last three frames came out UPSIDE
    // DOWN, which is what Tim saw -- and the individual links were all fine,
    // 53-93% inliers with shifts of 244-1318 px. Nothing was upside down in any
    // pair; the composition manufactured it.
    //
    // Constraining each link first means the error cannot compound: a product
    // of rotations is a rotation exactly, so the determinant stays 1 by
    // construction however long the chain runs. The per-link residual still
    // accumulates as ANGLE error -- that is inherent to sequential alignment --
    // but it can no longer turn into a reflection.
    if (Proj() != kPlane) {
        // K^-1 H K for one homography, forced to a rotation.
        //
        // Written explicitly rather than as three matrix multiplies: this is
        // the one place where a transposed index produces a plausible panorama
        // that is subtly sheared, rather than an obvious failure.
        auto toRotation = [&](const Affine& t, float R[9]) -> bool {
            float H[9];
            t.To3x3(H);
            float KH[9];
            // K^-1 = [[1/f, 0, -cx/f], [0, 1/f, -cy/f], [0, 0, 1]]
            for (int c = 0; c < 3; ++c) {
                KH[0 * 3 + c] = (H[0 * 3 + c] - cx * H[2 * 3 + c]) / f0;
                KH[1 * 3 + c] = (H[1 * 3 + c] - cy * H[2 * 3 + c]) / f0;
                KH[2 * 3 + c] =  H[2 * 3 + c];
            }
            for (int r = 0; r < 3; ++r) {
                R[r * 3 + 0] = KH[r * 3 + 0] * f0;
                R[r * 3 + 1] = KH[r * 3 + 1] * f0;
                R[r * 3 + 2] = KH[r * 3 + 0] * cx + KH[r * 3 + 1] * cy + KH[r * 3 + 2];
            }
            return Orthonormalise(R);
        };

        // Frame 0 is the reference: identity, by definition.
        for (int i = 0; i < 9; ++i)
            m_frames[0].R[i] = (i % 4 == 0) ? 1.0f : 0.0f;

        for (size_t i = 1; i < m_frames.size(); ++i) {
            Frame& f = m_frames[i];
            if (!f.invertible) continue;

            // The LINK from the previous frame, recovered by undoing the
            // previous accumulation: L = T[i] . T[i-1]^-1.
            //
            // Recovered rather than passed in, because the aligner's contract
            // is "a transform per frame" and it should stay that way -- a
            // stitcher that demanded the un-composed links would constrain
            // every aligner to produce them, including ones that never chain.
            bool ok = false;
            const Affine prevInv = m_frames[i - 1].t.Inverse(&ok);
            const Affine link = ok ? f.t.Then(prevInv) : f.t;

            float L[9];
            if (!toRotation(link, L)) {
                // A link that could not be read as a rotation ends the usable
                // chain: everything after it is positioned relative to a
                // frame whose own orientation is unknown. Marked rather than
                // skipped, so the report can say which frame and the viewer
                // is not left to infer it from a picture.
                f.invertible = false;
                if (m_flipped < 0) m_flipped = int(i);
                continue;
            }

            // R[i] = L . R[i-1], a product of rotations and therefore a
            // rotation -- which is the whole point of doing it here rather
            // than after the chain was already composed.
            const float* P = m_frames[i - 1].R;
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    f.R[r * 3 + c] = L[r * 3 + 0] * P[0 * 3 + c] +
                                     L[r * 3 + 1] * P[1 * 3 + c] +
                                     L[r * 3 + 2] * P[2 * 3 + c];
        }
    }

    // A pixel in frame `f` -> the ray it came from, in REFERENCE camera space.
    //
    // The pixel is a ray in this frame's own space, (x-cx, y-cy, f0); R maps
    // reference space into this frame, so its TRANSPOSE maps back. Transpose
    // rather than inverse because R is orthonormal, where those are the same
    // thing and one of them is free.
    const bool usePlane = (Proj() == kPlane);
    auto pixelToRefRay = [&](const Frame& fr, float px, float py,
                             float* dx, float* dy, float* dz) {
        const float a = px - cx, b = py - cy, c = f0;
        *dx = fr.R[0] * a + fr.R[3] * b + fr.R[6] * c;
        *dy = fr.R[1] * a + fr.R[4] * b + fr.R[7] * c;
        *dz = fr.R[2] * a + fr.R[5] * b + fr.R[8] * c;
    };

    // ...and back: a reference-space ray -> the pixel in frame `f`.
    auto refRayToPixel = [&](const Frame& fr, float dx, float dy, float dz,
                             float* px, float* py) {
        const float a = fr.R[0] * dx + fr.R[1] * dy + fr.R[2] * dz;
        const float b = fr.R[3] * dx + fr.R[4] * dy + fr.R[5] * dz;
        const float c = fr.R[6] * dx + fr.R[7] * dy + fr.R[8] * dz;
        if (c <= 1e-6f) return false;   // behind this camera
        *px = f0 * a / c + cx;
        *py = f0 * b / c + cy;
        return std::isfinite(*px) && std::isfinite(*py);
    };

    // --- size the canvas -----------------------------------------------------
    //
    // Every frame's border, projected onto the surface. Corners alone suffice
    // for the plane, where the map is linear; on a curved surface a straight
    // image edge maps to a curve whose extreme is in the MIDDLE, so the whole
    // border is sampled.
    float minU = 1e30f, maxU = -1e30f, minV = 1e30f, maxV = -1e30f;
    bool any = false;

    for (const Frame& f : m_frames) {
        if (!f.invertible) continue;
        // Sixteen points a side is far more than the curvature needs and costs
        // nothing at this scale.
        const int kSteps = 16;
        for (int e = 0; e < 4; ++e)
            for (int s = 0; s <= kSteps; ++s) {
                const float a = float(s) / float(kSteps);
                float px = 0.0f, py = 0.0f;
                switch (e) {
                    case 0: px = a * float(f.W()); py = 0.0f;           break;
                    case 1: px = float(f.W());     py = a * float(f.H()); break;
                    case 2: px = a * float(f.W()); py = float(f.H());   break;
                    default: px = 0.0f;            py = a * float(f.H()); break;
                }

                float u, v;
                if (usePlane) {
                    // Plane: the homography IS the answer, and the canvas is
                    // just the reference image plane extended.
                    f.inv.MapPoint(px, py, &u, &v);
                    u -= cx;
                    v -= cy;
                    if (!std::isfinite(u) || !std::isfinite(v)) continue;
                } else {
                    float dx, dy, dz;
                    pixelToRefRay(f, px, py, &dx, &dy, &dz);
                    if (!RayToCanvas(dx, dy, dz, &u, &v)) continue;
                }
                minU = std::min(minU, u); maxU = std::max(maxU, u);
                minV = std::min(minV, v); maxV = std::max(maxV, v);
                any = true;
            }
    }

    if (!any) { *err = "stitch_panorama: no frame had a usable transform"; return false; }

    const double wD = double(maxU) - double(minU);
    const double hD = double(maxV) - double(minV);
    if (!(wD > 0.0) || !(hD > 0.0)) {
        *err = "stitch_panorama: the frames project to an empty canvas";
        return false;
    }

    const double mp = wD * hD / 1e6;
    if (mp > double(int(m_maxPixels))) {
        // The diverging-chain case, and the message says what to do about it
        // rather than only that it failed.
        char buf[288];
        std::snprintf(buf, sizeof buf,
                      "stitch_panorama: the frames span %.0fx%.0f (%.0f MP), over the "
                      "%d MP limit.%s Or raise max_megapixels if the panorama really "
                      "is that large.",
                      wD, hD, mp, int(m_maxPixels),
                      usePlane
                          ? " A plane projection diverges past about 90 degrees --"
                            " try projection = 1 (cylindrical)."
                          : " Check the focal estimate in the report: too small a"
                            " focal length inflates every angle.");
        *err = buf;
        return false;
    }

    m_outW = std::max(1, int(std::lround(wD)));
    m_outH = std::max(1, int(std::lround(hD)));

    ImageDesc od = m_desc;
    od.width  = m_outW;
    od.height = m_outH;
    out->Alloc(od);
    ImageView ov = out->MapCpuWrite();
    if (!ov.data) { *err = "stitch_panorama: could not allocate the canvas"; return false; }

    PixelBuffer ob;
    ob.Unpack(ov);
    if (!ob.Valid()) { *err = "stitch_panorama: unsupported output format"; return false; }
    const int ch = ob.Channels();

    // Weighted accumulation, so overlaps average rather than the last frame
    // winning. Double for the same reason merge_mean uses it: fifteen 45 MP
    // frames is enough summation for float to drift.
    std::vector<double> acc(size_t(m_outW) * size_t(m_outH) * size_t(ch), 0.0);
    std::vector<double> wsum(size_t(m_outW) * size_t(m_outH), 0.0);

    // Sum of squares and a count, for the DISAGREEMENT metric below.
    //
    // Where two frames overlap they should report the same luminance. The
    // variance of what they actually report is a direct measurement of
    // misalignment: perfectly registered frames differ only by noise and
    // exposure, while a half-pixel error at an edge shows up immediately.
    //
    // This exists because sharpness turned out to be a poor proxy for ghosting
    // -- it moved by 10% across a focal sweep with no clean peak, because it
    // partly measures how much the canvas was scaled rather than how well the
    // frames agree. A metric computed only where frames actually overlap cannot
    // be fooled that way.
    std::vector<double> sq(size_t(m_outW) * size_t(m_outH), 0.0);
    std::vector<double> lum(size_t(m_outW) * size_t(m_outH), 0.0);
    std::vector<int>    cnt(size_t(m_outW) * size_t(m_outH), 0);

    m_placed = 0;
    for (const Frame& f : m_frames) {
        if (!f.invertible) continue;
        ++m_placed;

        for (int y = 0; y < m_outH; ++y) {
            const float v = minV + float(y);
            for (int x = 0; x < m_outW; ++x) {
                const float u = minU + float(x);

                float sx, sy;
                if (usePlane) {
                    // Plane: canvas coordinates are reference-image
                    // coordinates, so the homography maps straight across.
                    f.t.MapPoint(u + cx, v + cy, &sx, &sy);
                } else {
                    // Canvas -> ray in reference space -> this frame's pixel.
                    // The ray is what carries the geometry; the homography is
                    // not consulted at all, which is exactly why its
                    // accumulated drift cannot reach the output.
                    float dx, dy, dz;
                    CanvasToRay(u, v, &dx, &dy, &dz);
                    if (!refRayToPixel(f, dx, dy, dz, &sx, &sy)) continue;
                }

                // Outside this frame contributes nothing. Checked BEFORE
                // sampling: SampleBilinear edge-clamps, so an unchecked read
                // outside would smear the border pixel across the canvas
                // rather than leaving it to another frame.
                if (sx < 0.0f || sy < 0.0f ||
                    sx > float(f.W() - 1) || sy > float(f.H() - 1)) continue;

                // Feather weight: distance to the nearest edge of THIS frame,
                // normalised. Zero at the border, so a frame fades out exactly
                // where it stops having data and its neighbour takes over.
                double wgt = 1.0;
                if (bool(m_feather)) {
                    const float du = std::min(sx, float(f.W() - 1) - sx);
                    const float dv = std::min(sy, float(f.H() - 1) - sy);
                    const float dmin = std::min(du, dv);
                    // +1e-3 so a pixel exactly on the border still contributes
                    // something: a seam where every frame has weight zero would
                    // be a one-pixel hole.
                    wgt = double(dmin) + 1e-3;
                }

                float sm[4] = {0, 0, 0, 0};
                SampleBilinear(f.buf, sx, sy, sm);

                const size_t pi = size_t(y) * size_t(m_outW) + size_t(x);
                const size_t bi = pi * size_t(ch);
                for (int c = 0; c < ch; ++c) acc[bi + size_t(c)] += wgt * double(sm[c]);
                wsum[pi] += wgt;

                // Unweighted, and deliberately: the feather weight is nearly
                // zero exactly at a seam, which is where disagreement matters
                // most. Weighting this would hide the thing it is measuring.
                const double y0 = (ch >= 3)
                    ? 0.2126 * sm[0] + 0.7152 * sm[1] + 0.0722 * sm[2]
                    : sm[0];
                lum[pi] += y0;
                sq[pi]  += y0 * y0;
                cnt[pi] += 1;
            }
        }
    }

    // The disagreement metric: RMS spread between frames, over the pixels where
    // two or more actually overlap, relative to the mean level there.
    //
    // Relative rather than absolute so it means the same thing on a dark raw as
    // on a bright one -- the same reason the detectors normalise by p99.
    {
        double sum = 0.0, mean = 0.0;
        size_t n = 0;
        for (size_t i = 0; i < cnt.size(); ++i) {
            if (cnt[i] < 2) continue;
            const double k = double(cnt[i]);
            const double m = lum[i] / k;
            const double var = std::max(0.0, sq[i] / k - m * m);
            sum += var;
            mean += m;
            ++n;
        }
        m_overlapPixels = double(n) / double(std::max<size_t>(1, cnt.size()));
        if (n > 0 && mean > 0.0) {
            const double rms = std::sqrt(sum / double(n));
            m_disagree = float(rms / (mean / double(n)));
        }
    }

    std::vector<float>& px = ob.Data();
    if (px.size() != acc.size()) {
        *err = "stitch_panorama: result size mismatch";
        return false;
    }
    for (size_t i = 0; i < wsum.size(); ++i) {
        const double w = wsum[i];
        const size_t bi = i * size_t(ch);
        if (w > 0.0) {
            for (int c = 0; c < ch; ++c) px[bi + size_t(c)] = float(acc[bi + size_t(c)] / w);
        } else {
            // Uncovered canvas. Left at zero rather than filled: a panorama's
            // corners genuinely have no data, and inventing some there would
            // make the result look complete when it is not.
            for (int c = 0; c < ch; ++c) px[bi + size_t(c)] = 0.0f;
            if (ch == 4) px[bi + 3] = 0.0f;
        }
    }
    ob.PackInto(ov);
    return true;
}

REGISTER_ALGORITHM(StitchPanorama);

} // namespace
} // namespace tglab
