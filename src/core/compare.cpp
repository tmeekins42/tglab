#include "compare.h"

#include <algorithm>
#include <chrono>
#include <cmath>

#include "../gpu/compute.h"
#include "../script/interp.h"

namespace tglab {

namespace {

using clockt = std::chrono::steady_clock;

// Reads one channel as a double, whatever the underlying format.
double ReadChannel(const ImageView& v, int x, int y, int c) {
    switch (v.desc.format) {
        case Format::RGBA8:   return double(v.At<uint8_t>(x, y)[c]);
        case Format::R32F:    return c == 0 ? double(*v.At<float>(x, y)) : 0.0;
        case Format::RGBA32F: return double(v.At<float>(x, y)[c]);
        case Format::RGBA16F: return double(HalfToFloat(v.At<uint16_t>(x, y)[c]));
        default:              return 0.0;
    }
}

int ChannelCount(Format f) {
    switch (f) {
        case Format::RGBA8:   return 4;
        case Format::R32F:    return 1;
        case Format::RGBA32F: return 4;
        case Format::RGBA16F: return 4;
        default:              return 0;
    }
}

} // namespace

CompareStats CompareImages(Image& a, Image& b, double tolerance) {
    CompareStats s;
    if (!a.Valid() || !b.Valid()) return s;
    if (a.Desc() != b.Desc()) return s;

    // MapCpuRead() pulls back whichever side is GPU-resident. That transfer is
    // the price of comparing, and only happens because we asked.
    ImageView va = a.MapCpuRead();
    ImageView vb = b.MapCpuRead();
    if (!va.Valid() || !vb.Valid()) return s;

    const int w  = va.desc.width;
    const int h  = va.desc.height;
    const int ch = ChannelCount(va.desc.format);
    // Alpha is left untouched by most algorithms, so including it would
    // flatter the comparison. Compare colour channels only when there are 4.
    const int cmp = (ch == 4) ? 3 : ch;

    double sum = 0, sumSq = 0;
    long long n = 0;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            bool pixelDiffers = false;
            for (int c = 0; c < cmp; ++c) {
                const double d = std::fabs(ReadChannel(va, x, y, c) - ReadChannel(vb, x, y, c));
                if (d > s.maxAbsDiff) s.maxAbsDiff = d;
                if (d > tolerance) pixelDiffers = true;
                sum   += d;
                sumSq += d * d;
                ++n;
            }
            if (pixelDiffers) ++s.diffPixels;
        }
    }

    s.totalPixels = w * h;
    if (n > 0) {
        s.meanAbsDiff = sum / double(n);
        s.rmse        = std::sqrt(sumSq / double(n));
    }
    return s;
}

bool MakeDiffImage(Image& a, Image& b, float amplify, Image* out) {
    if (!a.Valid() || !b.Valid() || a.Desc() != b.Desc()) return false;

    ImageView va = a.MapCpuRead();
    ImageView vb = b.MapCpuRead();
    if (!va.Valid() || !vb.Valid()) return false;

    const int w = va.desc.width;
    const int h = va.desc.height;
    const int ch = ChannelCount(va.desc.format);
    const int cmp = (ch == 4) ? 3 : ch;

    // Always RGBA8: this is for looking at, not for feeding onward.
    out->Alloc({w, h, Format::RGBA8});
    ImageView vo = out->MapCpuWrite();

    // Normalise float formats by their own range so a difference of 0.001 in
    // an R32F gradient is still visible.
    double scale = 1.0;
    if (va.desc.format != Format::RGBA8) {
        double maxV = 0;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                for (int c = 0; c < cmp; ++c)
                    maxV = std::max(maxV, std::fabs(ReadChannel(va, x, y, c)));
        scale = maxV > 1e-9 ? 255.0 / maxV : 255.0;
    }

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            double worst = 0;
            for (int c = 0; c < cmp; ++c)
                worst = std::max(worst,
                                 std::fabs(ReadChannel(va, x, y, c) - ReadChannel(vb, x, y, c)));
            const double v = std::clamp(worst * scale * double(amplify), 0.0, 255.0);
            uint8_t* p = vo.At<uint8_t>(x, y);
            // Red-tinted: differences read as "wrong" at a glance, and stay
            // distinguishable from a legitimately grey result.
            p[0] = uint8_t(v);
            p[1] = uint8_t(v * 0.25);
            p[2] = uint8_t(v * 0.25);
            p[3] = 255;
        }
    }
    return true;
}

CompareResult CompareCpuGpu(Pipeline& pipe, std::vector<Data>* sources,
                            ComputeContext* gpu, int stageIndex) {
    CompareResult r;

    if (pipe.Stages().empty()) { r.error = "no stages to compare"; return r; }

    // stageIndex < 0 means "pick one that can actually be compared". A
    // pipeline commonly ends in a CPU-only stage, so defaulting to the last
    // one would usually just report that it has no GPU kernel.
    int idx = stageIndex;
    if (idx < 0) {
        for (size_t i = 0; i < pipe.Stages().size(); ++i) {
            if (pipe.Stages()[i].algo->HasGPU()) { idx = int(i); break; }
        }
        if (idx < 0) {
            r.error = "no stage in this pipeline has a GPU implementation";
            return r;
        }
    }
    if (size_t(idx) >= pipe.Stages().size()) { r.error = "stage out of range"; return r; }

    r.algorithm = pipe.Stages()[size_t(idx)].algoName;

    if (!gpu) { r.error = "no GPU context"; return r; }
    if (!pipe.Stages()[size_t(idx)].algo->HasGPU()) {
        r.error = "'" + r.algorithm + "' has no GPU implementation";
        return r;
    }

    // Run the whole pipeline twice. Running just the one stage would be
    // cheaper but would not prove the two paths agree *in context*, which is
    // what actually matters when a kernel is wrong only for certain inputs.
    std::string err;

    const auto t0 = clockt::now();
    if (!pipe.Execute(sources, nullptr, &err, gpu, ExecMode::ForceCPU)) {
        r.error = "CPU run failed: " + err;
        return r;
    }
    r.cpuMs = std::chrono::duration<double, std::milli>(clockt::now() - t0).count();
    r.cpuImage = std::get<Image>(pipe.Stages()[size_t(idx)].outputs[0]).Clone();

    const auto t1 = clockt::now();
    if (!pipe.Execute(sources, nullptr, &err, gpu, ExecMode::ForceGPU)) {
        r.error = "GPU run failed: " + err;
        return r;
    }
    // Include the readback: a GPU result you cannot look at is not a result.
    r.gpuImage = std::get<Image>(pipe.Stages()[size_t(idx)].outputs[0]).Clone();
    r.gpuMs = std::chrono::duration<double, std::milli>(clockt::now() - t1).count();

    // Tolerance of 1 unit for 8-bit; float formats compare much tighter.
    const double tol = r.cpuImage.Desc().format == Format::RGBA8 ? 1.0 : 1e-3;
    r.stats = CompareImages(r.cpuImage, r.gpuImage, tol);
    MakeDiffImage(r.cpuImage, r.gpuImage, 16.0f, &r.diffImage);

    r.ok = true;
    return r;
}

} // namespace tglab
