// draw_matches — shows where each matched feature moved to.
//
// Matches are a sidecar, so they are invisible, and a count says nothing about
// whether they are right. "196 matches" from a matcher pairing noise looks
// exactly like "196 matches" from one that works.
//
// WHAT IT DRAWS: a line from each feature's position in the REFERENCE frame to
// its position in this one. That is the useful picture because correct matches
// share a motion -- a pan gives a bundle of parallel lines, a rotation gives a
// fan -- while wrong ones point in random directions. A single glance
// separates them, which no number does.
//
// Colour is by the ratio that let each match through: green for confident,
// red for marginal. That makes the ratio parameter's effect visible rather
// than something to infer from a count.
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "../../algo_util/features.h"
#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"

namespace tglab {
namespace {

class DrawMatches : public AlgorithmBase {
public:
    const char* Name()     const override { return "draw_matches"; }
    const char* Category() const override { return "features"; }

    PortList Inputs() const override {
        return {{"src", DataType::ImageSet, FormatSpec::Any, ShapeSpec::Any}};
    }
    PortList Outputs() const override {
        return {{"out", DataType::ImageSet, FormatSpec::SameAsInput, ShapeSpec::SameAsInput}};
    }

    void RunCPU(RunCtx&) override {}

    // Needs the whole set: a match connects TWO frames, and the reference's
    // keypoints live in a different image from the matches.
    bool IsAligner() const override { return true; }

    bool RunAlign(std::vector<Image>* images, std::string* err) override {
        m_drawn = 0;
        m_noMatches = true;

        if (!images || images->size() < 2) {
            *err = "draw_matches needs a group of at least two images";
            return false;
        }

        const float dim = std::clamp(float(m_dim), 0.0f, 1.0f);

        for (Image& img : *images) {
            const MatchSidecar* ms = MatchesOf(img);
            if (!ms || ms->Matches().empty()) {
                // Dim it anyway, so every frame in the group looks alike and
                // the reference is not conspicuously brighter than the rest.
                if (dim < 1.0f) Dim(img, dim);
                continue;
            }
            m_noMatches = false;

            if (ms->Reference() < 0 || size_t(ms->Reference()) >= images->size()) continue;
            const FeatureSidecar* refF = FeaturesOf((*images)[size_t(ms->Reference())]);
            const FeatureSidecar* myF  = FeaturesOf(img);
            if (!refF || !myF) continue;

            DrawInto(img, *refF, *myF, *ms, dim);
        }
        return true;
    }

    std::string RunReport() const override {
        if (m_noMatches) return "no matches attached -- run a matcher first";
        if (m_drawn == 0) return {};
        return "drew " + std::to_string(m_drawn) + " match lines";
    }

    bool HasGPU() const override { return false; }

private:
    static void Dim(Image& img, float dim) {
        ImageView v = img.MapCpuWrite();
        if (!v.data) return;
        PixelBuffer pb;
        pb.Unpack(v);
        if (!pb.Valid()) return;
        for (float& x : pb.Data()) x *= dim;
        pb.PackInto(v);
    }

    void DrawInto(Image& img, const FeatureSidecar& refF, const FeatureSidecar& myF,
                  const MatchSidecar& ms, float dim) {
        ImageView v = img.MapCpuWrite();
        if (!v.data) return;
        PixelBuffer pb;
        pb.Unpack(v);
        if (!pb.Valid()) return;

        if (dim < 1.0f) for (float& x : pb.Data()) x *= dim;

        const int w = pb.Width(), h = pb.Height(), ch = pb.Channels();
        const float scale = pb.ValueScale();
        const int cap = std::max(1, int(m_maxDraw));

        // Most confident first, so the cap keeps the matches worth seeing.
        std::vector<const Match*> order;
        order.reserve(ms.Matches().size());
        for (const Match& m : ms.Matches()) order.push_back(&m);
        std::stable_sort(order.begin(), order.end(),
                         [](const Match* a, const Match* b) { return a->ratio < b->ratio; });

        for (const Match* m : order) {
            if (m_drawn >= cap) break;
            if (m->a < 0 || size_t(m->a) >= refF.keypoints.size()) continue;
            if (m->b < 0 || size_t(m->b) >= myF.keypoints.size()) continue;

            const Keypoint& ka = refF.keypoints[size_t(m->a)];
            const Keypoint& kb = myF.keypoints[size_t(m->b)];

            // Green for a confident match, red for a marginal one. The ratio
            // runs 0..1 with LOW being confident, so it is inverted here --
            // green is the readable colour and belongs on the good end.
            const float t = std::clamp(m->ratio, 0.0f, 1.0f);
            float rgb[3] = {t * scale, (1.0f - 0.35f * t) * scale, 0.0f};

            Line(pb, ka.x, ka.y, kb.x, kb.y, rgb, ch, w, h);

            // A mark at the destination, so a very short line is still visible.
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx)
                    Plot(pb, int(std::lround(kb.x)) + dx,
                             int(std::lround(kb.y)) + dy, rgb, ch, w, h);
            ++m_drawn;
        }

        pb.PackInto(v);
    }

    // Bresenham, so a line is drawn once per pixel rather than sampled at a
    // fixed count that either gaps on long lines or wastes work on short ones.
    static void Line(PixelBuffer& buf, float x0f, float y0f, float x1f, float y1f,
                     const float* rgb, int ch, int w, int h) {
        int x0 = int(std::lround(x0f)), y0 = int(std::lround(y0f));
        const int x1 = int(std::lround(x1f)), y1 = int(std::lround(y1f));
        const int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
        const int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
        int e = dx + dy;
        for (int guard = 0; guard < 4096; ++guard) {
            Plot(buf, x0, y0, rgb, ch, w, h);
            if (x0 == x1 && y0 == y1) break;
            const int e2 = 2 * e;
            if (e2 >= dy) { e += dy; x0 += sx; }
            if (e2 <= dx) { e += dx; y0 += sy; }
        }
    }

    static void Plot(PixelBuffer& buf, int x, int y, const float* rgb,
                     int ch, int w, int h) {
        if (x < 0 || y < 0 || x >= w || y >= h) return;
        float* p = buf.At(x, y);
        for (int c = 0; c < std::min(ch, 3); ++c) p[c] = rgb[c];
    }

    Param<float> m_dim{this, "dim", 0.4f, 0.0f, 1.0f,
        {.help = "How much of the image to keep behind the lines.",
         .step = 0.05}};

    Param<int> m_maxDraw{this, "max_draw", 500, 1, 20000,
        {.help = "Cap on lines drawn, most confident first. A few hundred is "
                 "enough to see whether the motion is coherent; thousands is a "
                 "solid mask."}};

    int  m_drawn = 0;
    bool m_noMatches = true;
};

} // namespace

REGISTER_ALGORITHM(DrawMatches);

} // namespace tglab
