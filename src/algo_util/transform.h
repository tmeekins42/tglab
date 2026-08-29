// Transform: a 2D warp attached to an image, and the sampler that applies it.
//
// Tim's design call, and the right one: an aligner ATTACHES a transform rather
// than warping pixels, and whatever consumes the group samples through it. That
// avoids a resample per frame and a second copy of the group, and it means one
// interpolation -- in the merge -- rather than two.
//
// It also means the solver and the merge want the SAME sampler: the solver has
// to warp in order to evaluate its residual, and the merge has to warp in order
// to read aligned pixels. One piece of code, used by both, so they cannot
// disagree about what a transform means.
#pragma once

#include <algorithm>
#include <cmath>
#include <string>

#include "../core/image.h"
#include "../core/sidecar.h"
#include "pixel_buffer.h"

namespace tglab {

// The name a transform is filed under. A constant rather than a literal at
// every use, because a typo would silently mean "no alignment" -- which looks
// like a merge that simply did not align well.
inline constexpr const char* kTransformSidecar = "transform";

// A 2D affine warp, stored as the 2x3 matrix that maps a REFERENCE coordinate
// to the corresponding position in THIS image:
//
//     this_pos = T * reference_pos
//
// That direction is the useful one, and it is worth being explicit because
// getting it backwards is invisible: a consumer walks the reference grid and
// asks "where in this frame do I read?", which is exactly what T answers, with
// no inverse needed.
//
// The first version of merge_hdr documented it the other way round and applied
// T.Inverse(). Both halves were wrong, so the warp came out NEGATED: instead of
// removing a displacement of d it applied -d, leaving 2d of error. The symptom
// was an aligned merge blurrier than an unaligned one, which is the right clue
// -- a wrong-magnitude warp would blur, but only a wrong-DIRECTION one blurs
// worse than doing nothing.
//
//     [ a b c ]   x
//     [ d e f ]   y
//                 1
//
// Affine rather than a full homography. Six parameters cover translation,
// rotation, uniform and non-uniform scale, and shear -- which is what a camera
// on a tripod does between frames. A homography adds two more for perspective,
// and on frames that differ by about a pixel those two are not constrained by
// the data: the solver would fit noise with them.
//
// Identity means "no transform", so an image with one attached but unsolved
// behaves exactly like one with none.
struct Affine {
    // Eight parameters, not six. The first six are the affine matrix exactly as
    // before; m[6] and m[7] are the perspective row, zero for an affine warp.
    //
    // WHY THIS GREW, and why it is one type rather than two:
    //
    // The pixel aligner is 6 DOF for a stated reason -- on frames differing by
    // about a pixel, the perspective terms are not constrained by the data and
    // the solver would fit noise with them. Feature alignment changes that:
    // matches spread across the frame DO constrain perspective, and a camera
    // that rotated about its own centre (a pan, the panorama case) is related
    // to the next frame by a homography and by nothing simpler.
    //
    // Splitting into Affine and Homography types would mean two samplers, two
    // sidecars, and every merge choosing between them -- for a difference of
    // two numbers that are usually zero. One type, and an affine warp is the
    // special case where the perspective row is zero, which costs two multiplies
    // and a divide-by-one per pixel.
    float m[8] = {1.0f, 0.0f, 0.0f,
                  0.0f, 1.0f, 0.0f,
                  0.0f, 0.0f};

    void MapPoint(float x, float y, float* ox, float* oy) const {
        const float w = m[6] * x + m[7] * y + 1.0f;
        // Guarded: a point on the horizon line of a strong homography maps to
        // infinity, and letting that through produces coordinates the sampler
        // clamps to a corner -- a plausible-looking smear rather than an error.
        const float iw = (std::abs(w) > 1e-9f) ? 1.0f / w : 0.0f;
        *ox = (m[0] * x + m[1] * y + m[2]) * iw;
        *oy = (m[3] * x + m[4] * y + m[5]) * iw;
    }

    bool IsAffine() const { return m[6] == 0.0f && m[7] == 0.0f; }

    bool IsIdentity() const {
        return m[0] == 1.0f && m[1] == 0.0f && m[2] == 0.0f &&
               m[3] == 0.0f && m[4] == 1.0f && m[5] == 0.0f &&
               m[6] == 0.0f && m[7] == 0.0f;
    }

    // The full 3x3, with the bottom-right fixed at 1. Written out because the
    // composition and inverse below are ordinary matrix algebra once the
    // storage is unpacked, and doing it in terms of m[] indices is how a
    // transcription error hides.
    void To3x3(float h[9]) const {
        h[0] = m[0]; h[1] = m[1]; h[2] = m[2];
        h[3] = m[3]; h[4] = m[4]; h[5] = m[5];
        h[6] = m[6]; h[7] = m[7]; h[8] = 1.0f;
    }

    // Back from a 3x3, normalised so h[8] is 1.
    //
    // A homography is defined only up to scale, so two matrices differing by a
    // constant factor are the SAME transform -- normalising is what lets the
    // eight stored numbers mean one thing.
    static Affine From3x3(const float h[9]) {
        Affine o;
        const float s = (std::abs(h[8]) > 1e-12f) ? 1.0f / h[8] : 1.0f;
        o.m[0] = h[0] * s; o.m[1] = h[1] * s; o.m[2] = h[2] * s;
        o.m[3] = h[3] * s; o.m[4] = h[4] * s; o.m[5] = h[5] * s;
        o.m[6] = h[6] * s; o.m[7] = h[7] * s;
        return o;
    }

    // Composition: apply `inner` then `this`.
    Affine Then(const Affine& inner) const {
        float a[9], b[9], c[9] = {};
        To3x3(a);
        inner.To3x3(b);
        for (int r = 0; r < 3; ++r)
            for (int k = 0; k < 3; ++k)
                for (int col = 0; col < 3; ++col)
                    c[r * 3 + col] += a[r * 3 + k] * b[k * 3 + col];
        return From3x3(c);
    }

    // The inverse warp. Returns identity when the matrix is singular, which is
    // the safe answer: an un-warped frame merges slightly misaligned, where a
    // garbage warp merges as noise.
    Affine Inverse(bool* ok = nullptr) const {
        float h[9];
        To3x3(h);

        // Cofactor inverse of the full 3x3. The affine case falls out of this
        // with the same answer the 2x2 formula gave, so there is one path
        // rather than a branch that could disagree with itself.
        float c[9];
        c[0] = h[4] * h[8] - h[5] * h[7];
        c[1] = h[2] * h[7] - h[1] * h[8];
        c[2] = h[1] * h[5] - h[2] * h[4];
        c[3] = h[5] * h[6] - h[3] * h[8];
        c[4] = h[0] * h[8] - h[2] * h[6];
        c[5] = h[2] * h[3] - h[0] * h[5];
        c[6] = h[3] * h[7] - h[4] * h[6];
        c[7] = h[1] * h[6] - h[0] * h[7];
        c[8] = h[0] * h[4] - h[1] * h[3];

        const float det = h[0] * c[0] + h[1] * c[3] + h[2] * c[6];
        if (std::abs(det) < 1e-12f) { if (ok) *ok = false; return Affine{}; }
        if (ok) *ok = true;

        const float inv = 1.0f / det;
        for (int i = 0; i < 9; ++i) c[i] *= inv;
        return From3x3(c);
    }

    // Translation only, which is what a report wants to show.
    float Dx() const { return m[2]; }
    float Dy() const { return m[5]; }

    // Rotation in degrees, recovered from the linear part. Approximate for a
    // matrix with shear, which is fine for a status line.
    float RotationDeg() const {
        return std::atan2(m[3], m[0]) * 57.2957795f;
    }
};

// The sidecar wrapper. Derived from the pixels: an alignment solved against an
// image is wrong the moment that image changes, and silently keeping it would
// produce a subtly misaligned merge rather than an error.
class TransformSidecar : public SidecarBase {
public:
    explicit TransformSidecar(const Affine& t) : m_t(t) {}
    const Affine& Get() const { return m_t; }
    bool DerivedFromPixels() const override { return true; }

private:
    Affine m_t;
};

// Convenience: the transform attached to an image, or identity when there is
// none.
//
// Returning identity rather than a null is what lets a consumer be written
// without a branch -- and it is honest, since an unaligned frame IS the
// identity warp. This is the whole reason the mechanism beats a parameter: a
// merge reads a transform it did not ask for and may not find, and proceeds
// either way.
inline Affine TransformOf(const Image& img) {
    if (const auto* s = img.Sidecar<TransformSidecar>(kTransformSidecar))
        return s->Get();
    return Affine{};
}

inline void AttachTransform(Image* img, const Affine& t) {
    img->Sidecars().Set(kTransformSidecar,
                        std::make_shared<const TransformSidecar>(t));
}

// Bilinear sample of `src` at a fractional coordinate, edge-clamped.
//
// Bilinear rather than nearest because the whole point is SUB-pixel alignment:
// a nearest-neighbour sample quantises the correction back to whole pixels and
// throws away what the solver measured. Bilinear rather than bicubic because
// the merge averages several frames anyway, so the extra sharpness of a cubic
// kernel is mostly lost, and cubic can overshoot into negative values that the
// out-of-gamut overlay would then flag as real.
inline void SampleBilinear(const PixelBuffer& src, float x, float y, float* out) {
    const int   w = src.Width(), h = src.Height(), ch = src.Channels();
    const float cx = std::clamp(x, 0.0f, float(w - 1));
    const float cy = std::clamp(y, 0.0f, float(h - 1));

    const int   x0 = int(cx), y0 = int(cy);
    const int   x1 = std::min(x0 + 1, w - 1);
    const int   y1 = std::min(y0 + 1, h - 1);
    const float fx = cx - float(x0), fy = cy - float(y0);

    // An exact pixel centre returns that pixel VERBATIM, not a weighted sum
    // that happens to reduce to it.
    //
    // Tim asked for this explicitly, and it is worth being deliberate about:
    // with an identity transform every sample lands on an integer, and the
    // arithmetic below would return p00 * 1 + others * 0. That is bit-exact for
    // ordinary values, but it is exact by ACCIDENT rather than by construction
    // -- and it stops being exact for an infinity or a NaN, where 0 * inf is
    // NaN and one bad pixel would spread to its neighbours.
    //
    // Taking the early exit means an unaligned merge reads exactly the pixels
    // it would have read with no sampler at all, which is the property that
    // makes "attach a transform" strictly better than "warp the pixels": with
    // no transform, nothing is resampled.
    if (fx == 0.0f && fy == 0.0f) {
        const float* p = src.At(x0, y0);
        for (int c = 0; c < ch; ++c) out[c] = p[c];
        return;
    }

    const float* p00 = src.At(x0, y0);
    const float* p10 = src.At(x1, y0);
    const float* p01 = src.At(x0, y1);
    const float* p11 = src.At(x1, y1);

    for (int c = 0; c < ch; ++c) {
        const float top = p00[c] + (p10[c] - p00[c]) * fx;
        const float bot = p01[c] + (p11[c] - p01[c]) * fx;
        out[c] = top + (bot - top) * fy;
    }
}

// Luminance at a fractional coordinate, which is what the solver compares.
// Separate from the full sample because the solver reads one number per point
// and interpolating three channels to then collapse them would be three times
// the work for the same answer.
inline float SampleLuma(const PixelBuffer& src, float x, float y) {
    float px[4] = {0, 0, 0, 0};
    SampleBilinear(src, x, y, px);
    return (src.Channels() >= 3)
        ? 0.2126f * px[0] + 0.7152f * px[1] + 0.0722f * px[2]
        : px[0];
}

} // namespace tglab
