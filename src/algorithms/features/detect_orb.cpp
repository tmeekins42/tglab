// detect_orb — Rublee, Rabaud, Konolige & Bradski (2011).
//
// FREE OF PATENTS, deliberately: ORB was published by the OpenCV group
// explicitly as an unencumbered alternative to SIFT and SURF, and that is part
// of why it is worth having alongside them.
//
// WHAT IT IS FOR, which is a different question from what it is good at.
//
// Tim's reasoning for adding it: a panorama does not need the descriptor
// quality that structure-from-motion needs, and speed is worth more there. That
// is right, and the measured gap is large. On a 45 MP frame the scale-space
// detectors in this directory cost 8-10 seconds EACH -- SIFT 8.2, SURF 9.2,
// AKAZE 9.9 -- which is what made a 15-frame stitch a three-minute job.
//
// ORB is fast because every stage of it is cheap by construction:
//
//   1. A FAST corner test: compare 16 pixels on a circle of radius 3 against
//      the centre, and accept when 9 contiguous ones are all brighter or all
//      darker. Integer comparisons, no arithmetic, and a two-test early
//      rejection throws out most pixels after four reads.
//   2. Harris response to RANK the corners, because FAST has no notion of
//      strength and would otherwise return every edge pixel in the image.
//   3. Scale from an image PYRAMID rather than a continuous scale space --
//      eight coarse levels instead of a blurred stack per octave.
//   4. Orientation from the intensity CENTROID: the vector from the corner to
//      the centre of mass of its patch. One pass, no histogram.
//   5. BRIEF descriptor: 256 pre-chosen pixel PAIRS, one bit each for which is
//      brighter. No gradients, no normalisation -- 256 comparisons and the
//      descriptor is done.
//
// WHERE IT IS WEAK, and it matters for which script uses it:
//
//   Scale invariance is coarse. A pyramid samples scale at discrete steps where
//   SIFT refines a sub-pixel scale, so a feature seen at 1.3x is found at a
//   slightly wrong size and its descriptor shifts. Fine for a pan, where every
//   frame is at the SAME scale; poor for matching across a zoom.
//
//   BRIEF is a raw intensity comparison, so it is invariant to a brightness
//   OFFSET and not to a change of contrast or a gamma. SIFT's normalise-clip-
//   normalise handles both.
//
//   The descriptor is 256 bits against AKAZE's 486, so it discriminates less.
//   That is the trade being made, not an oversight.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "../../algo_util/features.h"
#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"
#include "gpu_pyramid.h"

namespace tglab {
namespace {

// One pyramid level: a greyscale plane and the scale it stands for.
struct Level {
    std::vector<float> v;
    int   w = 0, h = 0;
    float scale = 1.0f;    // input pixels per pixel here

    float At(int x, int y) const {
        return v[size_t(std::clamp(y, 0, h - 1)) * size_t(w) +
                 size_t(std::clamp(x, 0, w - 1))];
    }
};

// The 16 pixels of the FAST circle, radius 3, in order around the ring.
//
// Order matters: the test asks for N CONTIGUOUS pixels, so these have to walk
// the circle rather than being any 16 points at that radius.
constexpr int kCircleX[16] = { 0,  1,  2,  3,  3,  3,  2,  1,
                               0, -1, -2, -3, -3, -3, -2, -1};
constexpr int kCircleY[16] = {-3, -3, -2, -1,  0,  1,  2,  3,
                              3,  3,  2,  1,  0, -1, -2, -3};

// FAST-9: is (x, y) a corner?
//
// The early rejection is the whole reason this is fast. Pixels 0, 4, 8 and 12
// are the compass points; for 9 contiguous of 16 to pass, at least three of
// those four must pass too. Testing them first rejects the great majority of
// pixels after four reads instead of sixteen.
bool FastCorner(const Level& L, int x, int y, float t) {
    const float c = L.At(x, y);
    const float hi = c + t, lo = c - t;

    int brightAxis = 0, darkAxis = 0;
    for (int i = 0; i < 16; i += 4) {
        const float p = L.At(x + kCircleX[i], y + kCircleY[i]);
        if (p > hi) ++brightAxis;
        else if (p < lo) ++darkAxis;
    }
    if (brightAxis < 3 && darkAxis < 3) return false;

    // The full ring, walked twice so a run can wrap around the end.
    int runBright = 0, runDark = 0;
    for (int i = 0; i < 32; ++i) {
        const int k = i & 15;
        const float p = L.At(x + kCircleX[k], y + kCircleY[k]);
        if (p > hi) { runDark = 0; if (++runBright >= 9) return true; }
        else if (p < lo) { runBright = 0; if (++runDark >= 9) return true; }
        else { runBright = runDark = 0; }
    }
    return false;
}

// Harris corner response, used to RANK the FAST corners rather than to find
// them.
//
// FAST answers "is this a corner" with a yes or no and nothing else, so a
// threshold low enough to find features in a flat sky also returns every pixel
// along every edge in the foreground. Harris gives the ordering that lets
// max_features mean "the strongest N" instead of "the first N found", which is
// the difference between a useful cap and an arbitrary one.
float HarrisResponse(const Level& L, int x, int y, int r) {
    float a = 0.0f, b = 0.0f, c = 0.0f;
    for (int dy = -r; dy <= r; ++dy)
        for (int dx = -r; dx <= r; ++dx) {
            const float gx = L.At(x + dx + 1, y + dy) - L.At(x + dx - 1, y + dy);
            const float gy = L.At(x + dx, y + dy + 1) - L.At(x + dx, y + dy - 1);
            a += gx * gx;
            b += gy * gy;
            c += gx * gy;
        }
    // det - k * trace^2, with Harris's k = 0.04.
    const float det = a * b - c * c;
    const float tr  = a + b;
    return det - 0.04f * tr * tr;
}

// Orientation by intensity centroid (Rosin's measure).
//
// The vector from the patch centre to its centre of mass. A corner is by
// definition asymmetric, so that vector is well defined and repeatable -- and
// it costs one pass over the patch, where SIFT builds and smooths a 36-bin
// histogram and may emit several orientations per keypoint.
float CentroidAngle(const Level& L, int x, int y, int r) {
    float m01 = 0.0f, m10 = 0.0f;
    for (int dy = -r; dy <= r; ++dy)
        for (int dx = -r; dx <= r; ++dx) {
            if (dx * dx + dy * dy > r * r) continue;   // circular patch
            const float p = L.At(x + dx, y + dy);
            m10 += float(dx) * p;
            m01 += float(dy) * p;
        }
    return std::atan2(m01, m10);
}

// The 256 BRIEF sampling pairs, generated once.
//
// Rublee's ORB learns its pairs from training data to maximise variance and
// minimise correlation. That table is not reproducible here without the
// training set, so these are drawn from an isotropic Gaussian instead -- which
// is Calonder's original BRIEF sampling, and the one the ORB paper measures its
// learned pairs AGAINST.
//
// What is lost is real but modest: the learned pairs are less correlated, so
// the 256 bits carry a little more information. What is kept is what matters
// here -- the pairs are FIXED, so two runs produce comparable descriptors, and
// a deterministic generator means that holds across machines too.
struct BriefPattern {
    int ax[256], ay[256], bx[256], by[256];

    BriefPattern(int patch) {
        // A fixed 32-bit LCG rather than <random>: the sequence has to be
        // identical on every platform or descriptors computed on one machine
        // would not match those from another.
        uint32_t s = 0x9E3779B9u;
        auto next = [&]() {
            s = s * 1664525u + 1013904223u;
            return s;
        };
        // Box-Muller from two uniforms, scaled to sigma = patch/5 as in BRIEF.
        auto gauss = [&]() {
            const float u1 = std::max(1e-7f, float(next() >> 8) / float(1 << 24));
            const float u2 = float(next() >> 8) / float(1 << 24);
            return std::sqrt(-2.0f * std::log(u1)) *
                   std::cos(6.2831853f * u2) * (float(patch) / 5.0f);
        };
        const int lim = patch / 2 - 1;
        for (int i = 0; i < 256; ++i) {
            ax[i] = std::clamp(int(std::lround(gauss())), -lim, lim);
            ay[i] = std::clamp(int(std::lround(gauss())), -lim, lim);
            bx[i] = std::clamp(int(std::lround(gauss())), -lim, lim);
            by[i] = std::clamp(int(std::lround(gauss())), -lim, lim);
        }
    }
};

Level Halve(const Level& src, float factor) {
    Level out;
    out.w = std::max(1, int(float(src.w) / factor));
    out.h = std::max(1, int(float(src.h) / factor));
    out.scale = src.scale * factor;
    out.v.assign(size_t(out.w) * size_t(out.h), 0.0f);
    // Box average over the source footprint, so downsampling band-limits rather
    // than aliasing. Point sampling here would make FAST fire on aliasing
    // artefacts that move between frames -- the opposite of repeatable.
    const float inv = 1.0f / (factor * factor);
    const int   n   = std::max(1, int(factor));
    for (int y = 0; y < out.h; ++y)
        for (int x = 0; x < out.w; ++x) {
            float a = 0.0f;
            int   c = 0;
            for (int dy = 0; dy < n; ++dy)
                for (int dx = 0; dx < n; ++dx) {
                    a += src.At(int(float(x) * factor) + dx,
                                int(float(y) * factor) + dy);
                    ++c;
                }
            out.v[size_t(y) * size_t(out.w) + size_t(x)] = c ? a / float(c) : a * inv;
        }
    return out;
}

class DetectOrb : public AlgorithmBase {
public:
    const char* Name()     const override { return "detect_orb"; }
    const char* Category() const override { return "features"; }

    PortList Inputs()  const override { return {{"src", DataType::Image, FormatSpec::Any}}; }
    PortList Outputs() const override {
        return {{"out", DataType::Image, FormatSpec::SameAsInput}};
    }

    void RunCPU(RunCtx& ctx) override {
        const ImageView src = ctx.In(0);
        ImageView       dst = ctx.Out(0);
        if (!src.Valid() || !dst.Valid()) return;

        PixelBuffer in;
        in.Unpack(src);
        if (!in.Valid()) return;

        const int w = in.Width(), h = in.Height(), ch = in.Channels();
        const float scale = in.ValueScale();

        PixelBuffer out;
        out.Unpack(dst);
        if (out.Valid()) {
            out.Data() = in.Data();
            out.PackInto(dst);
        }

        m_found = 0;
        if (w < 64 || h < 64) return;

        Level base;
        base.w = w; base.h = h;
        base.v.assign(size_t(w) * size_t(h), 0.0f);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const float* p = in.At(x, y);
                base.v[size_t(y) * size_t(w) + size_t(x)] = (ch >= 3)
                    ? (0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2]) / scale
                    : p[0] / scale;
            }

        // Normalised by the image's own level, for the reason given at
        // Percentile99 in features.h -- and ORB needs it more literally than
        // the others. Its threshold is an absolute intensity DIFFERENCE between
        // a centre pixel and its ring, so on a raw whose p99 is 0.12 a
        // threshold meant for display-referred data rejects every corner in the
        // image.
        m_level = Percentile99(base.v);
        if (m_level > 1e-6f) {
            const float inv = 1.0f / m_level;
            for (float& v : base.v) v *= inv;
        }

        auto sidecar = std::make_shared<FeatureSidecar>();
        sidecar->detector = "orb";
        sidecar->descriptors.kind = DescriptorKind::Binary;
        sidecar->descriptors.dim  = kDescBits;

        Detect(base, ctx, sidecar.get());
        m_found = int(sidecar->keypoints.size());

        if (Image* im = ctx.OutImage(0)) im->Sidecars().Set(kFeatureSidecar, sidecar);
    }

    std::string RunReport() const override {
        if (m_found <= 0) return {};
        char buf[80];
        std::snprintf(buf, sizeof buf, "%d ORB features (%d bits, level %.3f)",
                      m_found, kDescBits, m_level);
        return buf;
    }

    bool HasGPU() const override { return false; }

private:
    static constexpr int kDescBits  = 256;
    static constexpr int kPatch     = 31;   // BRIEF patch, Rublee's value
    static constexpr int kHarrisRad = 3;

    void Detect(const Level& base, RunCtx& ctx, FeatureSidecar* out) {
        const int   levels = std::clamp(int(m_levels), 1, 12);
        const float factor = std::max(1.05f, float(m_scaleFactor));
        const float thresh = std::max(0.0f, float(m_threshold));
        const int   border = kPatch / 2 + 4;   // room for the ring and the patch
        const BriefPattern pattern(kPatch);

        // Every candidate from every level, ranked together at the end.
        struct Cand {
            float x, y, scale, angle, response;
            int   octave, lx, ly;
        };
        std::vector<Cand> cands;

        std::vector<Level> pyramid;
        pyramid.push_back(base);
        for (int i = 1; i < levels; ++i) {
            Level next = Halve(pyramid.back(), factor);
            if (next.w < 2 * border || next.h < 2 * border) break;
            pyramid.push_back(std::move(next));
        }

        for (size_t o = 0; o < pyramid.size(); ++o) {
            if (ctx.Cancelled()) return;
            const Level& L = pyramid[o];

            // FAST and Harris are per-pixel tests over a small neighbourhood --
            // the shape a dispatch is good at, and 81 taps per pixel (32 for the
            // ring, 49 for the 7x7 response) across every level is where ORB's
            // time goes. The map comes back DENSE: the response where FAST
            // fired, 0 elsewhere. That keeps the scan below in its original
            // order, so the candidate list -- and therefore the ranking -- is
            // identical either way. See gpu_pyramid.h.
            std::vector<float> resp;
            bool haveMap = false;
            if (ComputeContext* dev = ctx.Gpu()) {
                GpuPlane src;
                src.v = L.v; src.w = L.w; src.h = L.h;
                GpuPlane map;
                std::string gerr;
                if (GpuFastHarris(dev, src, &map, thresh, kHarrisRad, border,
                                  &gerr)) {
                    resp = std::move(map.v);
                    haveMap = true;
                }
            }

            for (int y = border; y < L.h - border; ++y)
                for (int x = border; x < L.w - border; ++x) {
                    float r;
                    if (haveMap) {
                        r = resp[size_t(y) * size_t(L.w) + size_t(x)];
                        if (r <= 0.0f) continue;   // not a corner, or an edge
                    } else {
                        if (!FastCorner(L, x, y, thresh)) continue;
                        r = HarrisResponse(L, x, y, kHarrisRad);
                        if (r <= 0.0f) continue;   // an edge, not a corner
                    }

                    Cand c;
                    c.lx = x;
                    c.ly = y;
                    c.x  = float(x) * L.scale;
                    c.y  = float(y) * L.scale;
                    // The patch radius in INPUT pixels, which is what a
                    // visualiser draws and what a matcher compares against
                    // another detector's notion of scale.
                    c.scale    = float(kPatch) * 0.5f * L.scale;
                    c.angle    = 0.0f;         // filled in after ranking
                    c.response = r;
                    c.octave   = int(o);
                    cands.push_back(c);
                }
        }

        // RANK, then cap. FAST returns corners in raster order and a cap
        // applied during the scan would keep the top-left corner of the image
        // and discard the rest -- which looks like a detector that works only
        // on one part of the frame.
        const int cap = std::max(1, int(m_maxFeatures));
        if (int(cands.size()) > cap) {
            std::nth_element(cands.begin(), cands.begin() + cap, cands.end(),
                             [](const Cand& a, const Cand& b) {
                                 return a.response > b.response;
                             });
            cands.resize(size_t(cap));
        }

        out->keypoints.reserve(cands.size());
        out->descriptors.b.assign(cands.size() * size_t(kDescBits / 8), 0);

        const bool upright = bool(m_upright);
        for (size_t i = 0; i < cands.size(); ++i) {
            if (ctx.Cancelled()) return;
            const Cand& c = cands[i];
            const Level& L = pyramid[size_t(c.octave)];

            const float angle = upright ? 0.0f
                                        : CentroidAngle(L, c.lx, c.ly, kPatch / 2);

            Keypoint k;
            k.x        = c.x;
            k.y        = c.y;
            k.scale    = c.scale;
            k.angle    = angle;
            k.response = c.response;
            k.octave   = c.octave;
            out->keypoints.push_back(k);

            Describe(L, c.lx, c.ly, angle, pattern,
                     &out->descriptors.b[i * size_t(kDescBits / 8)]);
        }
    }

    // rBRIEF: the pattern, ROTATED by the keypoint's orientation.
    //
    // This is the "R" in ORB, and without it the descriptor is not rotation
    // invariant at all: plain BRIEF compares the same absolute pixel offsets
    // regardless of how the feature is turned, so rotating the camera by ten
    // degrees changes most of the 256 bits.
    static void Describe(const Level& L, int x, int y, float angle,
                         const BriefPattern& p, uint8_t* desc) {
        const float c = std::cos(angle), s = std::sin(angle);
        for (int i = 0; i < 256; ++i) {
            const int arx = int(std::lround(c * float(p.ax[i]) - s * float(p.ay[i])));
            const int ary = int(std::lround(s * float(p.ax[i]) + c * float(p.ay[i])));
            const int brx = int(std::lround(c * float(p.bx[i]) - s * float(p.by[i])));
            const int bry = int(std::lround(s * float(p.bx[i]) + c * float(p.by[i])));

            if (L.At(x + arx, y + ary) < L.At(x + brx, y + bry))
                desc[i >> 3] |= uint8_t(1u << (i & 7));
        }
    }

    Param<int> m_levels{this, "levels", 8, 1, 12,
        {.help = "Pyramid levels. ORB gets its scale invariance from a pyramid "
                 "rather than a continuous scale space, so this is the whole "
                 "range of sizes it can find -- coarser than SIFT's, which is "
                 "part of what makes it fast."}};

    Param<float> m_scaleFactor{this, "scale_factor", 1.2f, 1.05f, 2.0f,
        {.help = "Size ratio between pyramid levels. 1.2 is Rublee's: smaller "
                 "samples scale more finely and costs proportionally more "
                 "levels to cover the same range.",
         .step = 0.05}};

    // 0.06 rather than a value carried over from an 8-bit implementation.
    //
    // FAST's threshold is an absolute intensity difference, so it only means
    // anything once the input has been normalised -- which RunCPU does, by the
    // image's own 99th percentile. After that, 0.06 is "six percent of the
    // bright end", which is a real contrast step on any exposure.
    Param<float> m_threshold{this, "threshold", 0.06f, 0.001f, 0.5f,
        {.help = "How much brighter or darker the ring must be than the centre "
                 "for a FAST corner, relative to the image's own 99th "
                 "percentile. Raise it to keep only high-contrast corners.",
         .step = 0.005, .softMax = 0.2}};

    Param<bool> m_upright{this, "upright", false,
        "Skip the orientation estimate and describe every feature axis-aligned. "
        "Faster, and correct when the camera never rolls -- a tripod pan, or a "
        "scan. A rolled frame then matches nothing."};

    Param<int> m_maxFeatures{this, "max_features", 5000, 10, 50000,
        {.help = "Keep this many, strongest first by Harris response. Unlike "
                 "the scale-space detectors this really is a quality control: "
                 "FAST finds far more corners than are useful and the ranking "
                 "is what makes the cap meaningful."}};

    int   m_found = 0;
    float m_level = 0.0f;
};

REGISTER_ALGORITHM(DetectOrb);

} // namespace
} // namespace tglab
