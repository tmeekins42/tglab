// merge_mean: averages every frame of a group into one image.
//
// The simplest honest reduction, and deliberately the first one: averaging is
// what a stack of identical exposures wants for noise reduction (N frames cut
// the standard deviation by sqrt(N)), and it exercises the whole streaming path
// without any of the questions a real HDR merge raises -- no exposure
// weighting, no alignment, no ghost detection.
//
// It accumulates in double rather than in the output format. Summing hundreds
// of values into a float loses the low bits well before the count gets
// interesting, and the accumulator is the one place the extra width costs
// nothing: it is one buffer regardless of how many frames stream through it.
#include <string>
#include <vector>

#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"

namespace tglab {
namespace {

class MergeMean : public AlgorithmBase, public Reducer {
public:
    const char* Name()     const override { return "merge_mean"; }
    const char* Category() const override { return "merge"; }

    PortList Inputs() const override {
        return {{"src", DataType::ImageSet, FormatSpec::Any, ShapeSpec::Any}};
    }
    PortList Outputs() const override {
        return {{"out", DataType::Image, FormatSpec::SameAsInput, ShapeSpec::Reduced}};
    }

    // Never called: a reduction goes through the accumulator instead. Present
    // because RunCPU is pure virtual on the base, which is right for every
    // other algorithm.
    void RunCPU(RunCtx&) override {}

    bool     IsReduction() const override { return true; }
    Reducer* AsReducer()         override { return this; }

    bool Begin(int count, const std::string&, std::string* err) override {
        if (count <= 0) { *err = "merge_mean: nothing to merge"; return false; }
        m_seen = 0;
        m_sum.clear();
        m_desc = ImageDesc{};
        return true;
    }

    bool Accept(int, const Image& img, std::string* err) override {
        ImageView v = const_cast<Image&>(img).MapCpuRead();
        if (!v.data) { *err = "merge_mean: a frame has no pixels"; return false; }

        const ImageDesc& d = img.Desc();
        if (m_seen > 0 && (d.width != m_desc.width || d.height != m_desc.height ||
                           d.format != m_desc.format)) {
            // Averaging frames of different sizes has no defensible answer --
            // cropping, padding and resampling are different operations the
            // script should ask for explicitly.
            *err = "merge_mean: every frame must be the same size and format (frame " +
                   std::to_string(m_seen) + " is " + std::to_string(d.width) + "x" +
                   std::to_string(d.height) + ", expected " + std::to_string(m_desc.width) +
                   "x" + std::to_string(m_desc.height) + ")";
            return false;
        }

        PixelBuffer buf;
        buf.Unpack(v);
        if (!buf.Valid()) { *err = "merge_mean: unsupported format"; return false; }

        const std::vector<float>& px = buf.Data();
        if (m_seen == 0) {
            m_desc = d;
            m_sum.assign(px.size(), 0.0);
        }
        for (size_t i = 0; i < px.size(); ++i) m_sum[i] += double(px[i]);

        ++m_seen;
        return true;
    }

    bool Finish(Image* out, std::string* err) override {
        if (m_seen == 0) { *err = "merge_mean: no frames were accepted"; return false; }

        out->Alloc(m_desc);
        ImageView v = out->MapCpuWrite();
        if (!v.data) { *err = "merge_mean: could not allocate the result"; return false; }

        // Unpack the (freshly allocated) destination to get a correctly shaped
        // buffer, fill it with the mean, and pack it back. Cheaper than it
        // reads: the unpack is of zeros and the alternative is duplicating
        // PixelBuffer's format knowledge here.
        PixelBuffer buf;
        buf.Unpack(v);
        std::vector<float>& px = buf.Data();
        if (px.size() != m_sum.size()) { *err = "merge_mean: result size mismatch"; return false; }

        const double inv = 1.0 / double(m_seen);
        for (size_t i = 0; i < px.size(); ++i) px[i] = float(m_sum[i] * inv);
        buf.PackInto(v);

        m_sum.clear();
        m_sum.shrink_to_fit();
        return true;
    }

    std::string RunReport() const override {
        if (m_seen == 0) return {};
        return "averaged " + std::to_string(m_seen) + " frames";
    }

private:
    std::vector<double> m_sum;    // one accumulator, regardless of frame count
    ImageDesc           m_desc{};
    int                 m_seen = 0;
};

} // namespace

REGISTER_ALGORITHM(MergeMean);

} // namespace tglab
