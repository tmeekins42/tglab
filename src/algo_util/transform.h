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
    float m[6] = {1.0f, 0.0f, 0.0f,
                  0.0f, 1.0f, 0.0f};

    void MapPoint(float x, float y, float* ox, float* oy) const {
        *ox = m[0] * x + m[1] * y + m[2];
        *oy = m[3] * x + m[4] * y + m[5];
    }

    bool IsIdentity() const {
        return m[0] == 1.0f && m[1] == 0.0f && m[2] == 0.0f &&
               m[3] == 0.0f && m[4] == 1.0f && m[5] == 0.0f;
    }

    // Composition: apply `inner` then `this`.
    Affine Then(const Affine& inner) const {
        Affine o;
        o.m[0] = m[0] * inner.m[0] + m[1] * inner.m[3];
        o.m[1] = m[0] * inner.m[1] + m[1] * inner.m[4];
        o.m[2] = m[0] * inner.m[2] + m[1] * inner.m[5] + m[2];
        o.m[3] = m[3] * inner.m[0] + m[4] * inner.m[3];
        o.m[4] = m[3] * inner.m[1] + m[4] * inner.m[4];
        o.m[5] = m[3] * inner.m[2] + m[4] * inner.m[5] + m[5];
        return o;
    }

    // The inverse warp. Returns identity when the matrix is singular, which is
    // the safe answer: an un-warped frame merges slightly misaligned, where a
    // garbage warp merges as noise.
    Affine Inverse(bool* ok = nullptr) const {
        const float det = m[0] * m[4] - m[1] * m[3];
        if (std::abs(det) < 1e-12f) { if (ok) *ok = false; return Affine{}; }
        if (ok) *ok = true;
        const float inv = 1.0f / det;
        Affine o;
        o.m[0] =  m[4] * inv;
        o.m[1] = -m[1] * inv;
        o.m[3] = -m[3] * inv;
        o.m[4] =  m[0] * inv;
        o.m[2] = -(o.m[0] * m[2] + o.m[1] * m[5]);
        o.m[5] = -(o.m[3] * m[2] + o.m[4] * m[5]);
        return o;
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
