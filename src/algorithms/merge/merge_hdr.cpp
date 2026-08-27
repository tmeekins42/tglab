// merge_hdr: combines an exposure bracket into one scene-linear image.
//
// The physically correct merge, not an average. Each frame records
//
//     value = radiance * exposure
//
// so dividing by that frame's own exposure recovers radiance, and every frame
// then agrees about the same scene up to noise and clipping. What differs is
// WHICH frames are trustworthy where: a dark frame resolves the highlights and
// buries the shadows in read noise, a bright one does the reverse. So each
// sample is weighted by how well exposed it is, and the estimate is
//
//     radiance = sum(w_i * v_i / e_i) / sum(w_i)
//
// The result carries real headroom -- a bracket spanning six stops produces
// values well above 1.0 -- which is the point of shooting one. Everything
// downstream is already linear float end to end, and the display tone curve
// maps it at the end.
//
// This is Debevec-style merging without the response-curve solve: a raw sensor
// is linear by construction, so there is no curve to recover. That step exists
// in the literature because the input there is usually JPEG.
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"

namespace tglab {
namespace {

// Weight for one sample, by how far it is from useless.
//
// Zero at both ends and smooth between: a clipped sample carries no
// information about how much brighter the scene really was, and a near-black
// one is mostly read noise. The classic choice is a hat function; this is its
// smooth cousin, which avoids a visible seam where one frame stops
// contributing to a gradient.
float Weight(float v) {
    if (v <= 0.0f || v >= 1.0f) return 0.0f;
    const float t = 2.0f * v - 1.0f;          // -1 .. 1
    const float w = 1.0f - t * t;             // parabola, zero at both ends
    return w * w;                             // squared: falls off faster near the ends
}

class MergeHdr : public AlgorithmBase, public Reducer {
public:
    const char* Name()     const override { return "merge_hdr"; }
    const char* Category() const override { return "merge"; }

    PortList Inputs() const override {
        return {{"src", DataType::ImageSet, FormatSpec::Any, ShapeSpec::Any}};
    }
    PortList Outputs() const override {
        // RGBA32F rather than the input format. The whole point is a result
        // with more range than any single frame, and half float would clip the
        // top of a wide bracket straight back off.
        return {{"out", DataType::Image, FormatSpec::RGBA32F, ShapeSpec::Reduced}};
    }

    void RunCPU(RunCtx&) override {}

    bool     IsReduction() const override { return true; }
    Reducer* AsReducer()         override { return this; }

    bool Begin(int count, const std::string&, std::string* err) override {
        if (count <= 0) { *err = "merge_hdr: nothing to merge"; return false; }
        m_seen = 0;
        m_num.clear();
        m_den.clear();
        m_best.clear();
        m_bestScore.clear();
        m_desc = ImageDesc{};
        m_noExposure = false;
        m_unexposed  = 0;
        m_minEv = 0.0f;
        m_maxEv = 0.0f;
        return true;
    }

    bool Accept(int, const Image& img, std::string* err) override {
        ImageView v = const_cast<Image&>(img).MapCpuRead();
        if (!v.data) { *err = "merge_hdr: a frame has no pixels"; return false; }

        const ImageDesc& d = img.Desc();
        if (m_seen > 0 && (d.width != m_desc.width || d.height != m_desc.height)) {
            *err = "merge_hdr: every frame must be the same size (frame " +
                   std::to_string(m_seen) + " is " + std::to_string(d.width) + "x" +
                   std::to_string(d.height) + ", expected " + std::to_string(m_desc.width) +
                   "x" + std::to_string(m_desc.height) + ")";
            return false;
        }

        // Relative exposure, from the EXIF the raw loader now keeps as numbers.
        //
        // A frame carrying no exposure information cannot be placed on the same
        // radiance scale as the others. Recorded rather than failed on, and
        // reported afterwards: silently treating a bracket as equal exposures
        // produces a confidently wrong result, which is the worst outcome.
        float e = d.RelativeExposure();
        if (e <= 0.0f) { e = 1.0f; m_noExposure = true; }

        // Track the spread, so the report can say how many stops were covered.
        const float ev = std::log2(e);
        if (m_seen == 0) { m_minEv = m_maxEv = ev; }
        else { m_minEv = std::min(m_minEv, ev); m_maxEv = std::max(m_maxEv, ev); }

        PixelBuffer buf;
        buf.Unpack(v);
        if (!buf.Valid()) { *err = "merge_hdr: unsupported format"; return false; }
        const std::vector<float>& px = buf.Data();

        if (m_seen == 0) {
            m_desc = d;
            m_ch   = buf.Channels();
            m_num.assign(px.size(), 0.0);
            m_den.assign(px.size(), 0.0);
            m_best.assign(px.size(), 0.0f);
            m_bestScore.assign(px.size(), -1.0f);
        }
        if (px.size() != m_num.size()) { *err = "merge_hdr: frame size changed"; return false; }

        const float inv = 1.0f / e;
        for (size_t i = 0; i < px.size(); ++i) {
            const float s = px[i];
            const float w = Weight(s);
            m_num[i] += double(w) * double(s) * double(inv);
            m_den[i] += double(w);

            // Fallback for a pixel that no frame exposes well: a light source
            // brighter than the shortest exposure, or a shadow below the
            // longest. Without it the weights all vanish and the estimate is
            // 0/0. Keeping the sample furthest from clipping is wrong but
            // bounded, and beats leaving a hole in the image.
            const float score = 1.0f - std::abs(2.0f * s - 1.0f);
            if (score > m_bestScore[i]) { m_bestScore[i] = score; m_best[i] = s * inv; }
        }

        ++m_seen;
        return true;
    }

    bool Finish(Image* out, std::string* err) override {
        if (m_seen == 0) { *err = "merge_hdr: no frames were accepted"; return false; }

        ImageDesc od = m_desc;
        od.format = Format::RGBA32F;
        od.linear = true;     // scene-linear radiance, not display encoded
        out->Alloc(od);
        ImageView ov = out->MapCpuWrite();
        if (!ov.data) { *err = "merge_hdr: could not allocate the result"; return false; }

        PixelBuffer ob;
        ob.Unpack(ov);
        std::vector<float>& op = ob.Data();

        // The accumulator holds the INPUT channel count while the output is
        // RGBA. Mapping channel by channel rather than assuming they match is
        // what lets a single-channel input merge without a separate path.
        const int    oc     = ob.Channels();
        const size_t pixels = m_num.size() / size_t(m_ch);

        for (size_t p = 0; p < pixels; ++p) {
            for (int c = 0; c < oc; ++c) {
                const size_t di = p * size_t(oc) + size_t(c);
                if (c == 3 && oc == 4) { op[di] = 1.0f; continue; }   // alpha
                const int    sc = (c < m_ch) ? c : (m_ch - 1);
                const size_t si = p * size_t(m_ch) + size_t(sc);

                if (m_den[si] > 1e-6) {
                    op[di] = float(m_num[si] / m_den[si]);
                } else {
                    op[di] = m_best[si];
                    if (c == 0) ++m_unexposed;
                }
            }
        }
        ob.PackInto(ov);

        m_num.clear();       m_num.shrink_to_fit();
        m_den.clear();       m_den.shrink_to_fit();
        m_best.clear();      m_best.shrink_to_fit();
        m_bestScore.clear(); m_bestScore.shrink_to_fit();
        return true;
    }

    std::string RunReport() const override {
        if (m_seen == 0) return {};
        std::string r = "merged " + std::to_string(m_seen) + " exposures";
        if (m_noExposure) {
            r += " (NO EXIF exposure: frames treated as equal)";
        } else if (m_maxEv > m_minEv) {
            char buf[32];
            std::snprintf(buf, sizeof buf, "%.1f", double(m_maxEv - m_minEv));
            r += std::string(" spanning ") + buf + " stops";
        }
        if (m_unexposed) {
            r += ", " + std::to_string(m_unexposed) +
                 " pixels outside every exposure";
        }
        return r;
    }

private:
    // Weighted sums in double. A bracket spans orders of magnitude, and adding
    // a bright frame's contribution to a dark frame's in float would lose the
    // dark one entirely.
    std::vector<double> m_num, m_den;

    // Best single sample per element, for pixels no frame exposes well.
    std::vector<float>  m_best, m_bestScore;

    ImageDesc m_desc{};
    int       m_ch    = 4;
    int       m_seen  = 0;
    bool      m_noExposure = false;
    int       m_unexposed  = 0;
    float     m_minEv = 0.0f, m_maxEv = 0.0f;
};

} // namespace

REGISTER_ALGORITHM(MergeHdr);

} // namespace tglab
