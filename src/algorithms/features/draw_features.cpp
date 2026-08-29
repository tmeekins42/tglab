// draw_features — renders the features attached to an image, so they can be
// looked at.
//
// A detector's output is a sidecar, which is invisible. Numbers in the run
// report say how MANY were found and nothing about whether they are in sensible
// places -- and "2000 features" from a detector that is finding noise looks
// exactly like "2000 features" from one that is working.
//
// WHAT IT DRAWS, and why each part is not decoration:
//
//   circle   radius = the keypoint's scale. A scale-space detector finding a
//            large blob and a small one is doing its job; drawing both as
//            fixed-size dots would hide precisely what distinguishes it from a
//            corner detector.
//   radius   a line from centre to edge at the keypoint's angle. Orientation is
//            what the descriptor is measured in, so an orientation that jitters
//            between frames is a matching failure you can SEE here.
//   colour   by response, so the strong features are distinguishable from the
//            marginal ones that a higher threshold would remove.
//
// It does NOT need to know the descriptor type. Tim asked whether the
// visualiser has to understand the descriptor; for drawing, it does not --
// position, scale and angle are common to every detector, and those are what
// there is to draw. The descriptor kind is reported as text, since knowing
// whether you are looking at SIFT or AKAZE output matters when comparing them.
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "../../algo_util/features.h"
#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"

namespace tglab {
namespace {

class DrawFeatures : public AlgorithmBase {
public:
    const char* Name()     const override { return "draw_features"; }
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

        PixelBuffer out;
        out.Unpack(dst);
        if (!out.Valid()) return;

        // The image first, dimmed if asked. Features on a busy photograph are
        // hard to see; dimming the picture rather than brightening the marks
        // keeps the marks at full contrast whatever is underneath.
        const float dim = std::clamp(float(m_dim), 0.0f, 1.0f);
        out.Data() = in.Data();
        if (dim < 1.0f)
            for (float& v : out.Data()) v *= dim;

        m_drawn = 0;
        m_kind.clear();

        const Image* im = ctx.InImage(0);
        const FeatureSidecar* fs = im ? FeaturesOf(*im) : nullptr;
        if (!fs) {
            // Distinguishable from "found none": see FeaturesOf. Reported
            // rather than silent, because a script whose detector is missing
            // otherwise looks like a detector that failed.
            m_noSidecar = true;
            out.PackInto(dst);
            return;
        }
        m_noSidecar = false;
        m_detector  = fs->detector;

        switch (fs->descriptors.kind) {
            case DescriptorKind::Float:  m_kind = std::to_string(fs->descriptors.dim) + "f"; break;
            case DescriptorKind::Binary: m_kind = std::to_string(fs->descriptors.dim) + " bits"; break;
            case DescriptorKind::None:   m_kind = "no descriptor"; break;
        }

        // Strongest first, so the cap keeps the features worth seeing rather
        // than whichever the detector happened to emit first.
        std::vector<const Keypoint*> order;
        order.reserve(fs->keypoints.size());
        for (const Keypoint& k : fs->keypoints) order.push_back(&k);
        std::stable_sort(order.begin(), order.end(),
                         [](const Keypoint* a, const Keypoint* b) {
                             return a->response > b->response;
                         });

        float maxResp = 0.0f;
        for (const Keypoint* k : order) maxResp = std::max(maxResp, k->response);
        if (maxResp <= 0.0f) maxResp = 1.0f;

        const int cap = std::max(1, int(m_maxDraw));
        const float scale = in.ValueScale();

        for (const Keypoint* k : order) {
            if (m_drawn >= cap) break;
            DrawOne(out, *k, k->response / maxResp, scale);
            ++m_drawn;
        }

        out.PackInto(dst);
    }

    std::string RunReport() const override {
        if (m_noSidecar) return "no features attached -- run a detector first";
        if (m_drawn == 0) return {};
        std::string r = "drew " + std::to_string(m_drawn) + " features";
        if (!m_detector.empty()) r += " (" + m_detector;
        if (!m_kind.empty())     r += ", " + m_kind;
        if (!m_detector.empty()) r += ")";
        return r;
    }

    bool HasGPU() const override { return false; }

private:
    // Colour from strength: green for weak, yellow through red for strong.
    //
    // A ramp rather than one colour because the interesting question when
    // tuning a threshold is "what am I about to lose", and that is the weak
    // end -- so the weak end has to be the READABLE one.
    //
    // The first version ran blue -> red with green only in the middle, which
    // put the weakest features in pale blue: the hardest colour to see against
    // a photograph, and exactly the ones being judged. Green is the channel the
    // eye is most sensitive to (it is 0.72 of luma), so every feature here
    // carries a lot of it and the ramp varies red instead.
    static void Ramp(float t, float* rgb) {
        t = std::clamp(t, 0.0f, 1.0f);
        rgb[0] = t;                 // red rises with strength: green -> yellow -> red
        rgb[1] = 1.0f - 0.35f * t;  // green never drops far; it is what makes them visible
        rgb[2] = 0.0f;
    }

    void DrawOne(PixelBuffer& buf, const Keypoint& k, float strength, float scale) const {
        const int w = buf.Width(), h = buf.Height(), ch = buf.Channels();

        float rgb[3];
        Ramp(strength, rgb);
        for (int c = 0; c < 3; ++c) rgb[c] *= scale;   // into the buffer's units

        // Radius from the keypoint's own scale, with a floor so a fine feature
        // is still visible rather than a single pixel.
        const float r = std::max(2.0f, k.scale * float(m_scaleMul));

        // The circle, by walking the angle rather than the bounding box: at
        // large radii the box is mostly empty and the walk is proportional to
        // the circumference instead of its square.
        const int steps = std::max(12, int(r * 6.0f));
        for (int i = 0; i < steps; ++i) {
            const float a = float(i) / float(steps) * 6.2831853f;
            Plot(buf, int(std::lround(k.x + r * std::cos(a))),
                      int(std::lround(k.y + r * std::sin(a))), rgb, ch, w, h);
        }

        // The orientation radius.
        if (m_showAngle) {
            const int n = std::max(2, int(r));
            for (int i = 0; i <= n; ++i) {
                const float t = float(i) / float(n);
                Plot(buf, int(std::lround(k.x + r * t * std::cos(k.angle))),
                          int(std::lround(k.y + r * t * std::sin(k.angle))),
                          rgb, ch, w, h);
            }
        }
    }

    static void Plot(PixelBuffer& buf, int x, int y, const float* rgb,
                     int ch, int w, int h) {
        if (x < 0 || y < 0 || x >= w || y >= h) return;
        float* p = buf.At(x, y);
        for (int c = 0; c < std::min(ch, 3); ++c) p[c] = rgb[c];
    }

    Param<float> m_dim{this, "dim", 0.4f, 0.0f, 1.0f,
        {.help = "How much of the image to keep behind the marks. Lower makes "
                 "features easier to see on a busy photograph; 1 leaves the "
                 "image untouched.",
         .step = 0.05}};

    Param<float> m_scaleMul{this, "scale_mul", 1.5f, 0.2f, 8.0f,
        {.help = "Circle radius as a multiple of the keypoint's scale. The "
                 "scale is what the descriptor was measured over, so this "
                 "shows the region each feature actually describes.",
         .step = 0.1}};

    Param<bool> m_showAngle{this, "show_angle", true,
        "Draw a radius at each feature's orientation. That angle is the frame "
        "the descriptor is measured in, so one that jitters between frames is "
        "a matching failure you can see."};

    Param<int> m_maxDraw{this, "max_draw", 2000, 1, 100000,
        {.help = "Cap on how many to draw, strongest first. Ten thousand marks "
                 "on one frame is a solid mask rather than a picture."}};

    int         m_drawn = 0;
    bool        m_noSidecar = false;
    std::string m_detector;
    std::string m_kind;
};

} // namespace

REGISTER_ALGORITHM(DrawFeatures);

} // namespace tglab
