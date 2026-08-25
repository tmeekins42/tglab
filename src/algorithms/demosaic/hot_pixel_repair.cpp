// hot_pixel_repair — replaces stuck sensels in a Bayer mosaic.
//
// A stuck (or hot) sensel reads far above its neighbours regardless of the
// light falling on it. It is a property of the sensor, not the scene: on Tim's
// 5D the same two sites, (1671,2108) and (3457,1530), appear in every frame,
// at the same CFA parity, with the excess scaling with ISO -- +4600 levels at
// ISO 400 and +10000 at ISO 1000.
//
// This runs on the MOSAIC, before demosaic. After demosaicing, one bad sensel
// has been smeared across a neighbourhood by interpolation: it is no longer a
// single-pixel outlier, and repairing it means repairing a blob whose extent
// depends on which demosaic ran. Fixing one sample before that happens is both
// simpler and more accurate.
//
// Comparison is against the eight same-colour neighbours two steps away, since
// the Bayer period is 2. Comparing a red sensel against adjacent greens would
// call every saturated red flower a defect.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "../../core/algorithm.h"

namespace tglab {
namespace {

class HotPixelRepair : public AlgorithmBase {
public:
    const char* Name()     const override { return "hot_pixel_repair"; }
    const char* Category() const override { return "demosaic"; }

    // R32F in, R32F out: a Bayer mosaic, with the CFA metadata carried through
    // so a demosaic downstream still sees a mosaic.
    //
    // Declared R32F rather than Any deliberately. The whole method rests on
    // "the sensel two steps away is the same colour", which is a property of a
    // CFA mosaic and means nothing on a finished RGB image -- there, two steps
    // away is just another pixel. Accepting Any let the CPU path read an RGBA8
    // image as though it were floats while the GPU read it as UNORM, and the
    // two disagreed by the full range. The pipeline now inserts a conversion
    // instead, and a script that points this at an ordinary photo gets a
    // sensible answer rather than a silently wrong one.
    PortList Inputs()  const override { return {{"src", DataType::Image, FormatSpec::R32F}}; }
    PortList Outputs() const override { return {{"out", DataType::Image, FormatSpec::SameAsInput}}; }

    void RunCPU(RunCtx& ctx) override {
        const ImageView src = ctx.In(0);
        ImageView       dst = ctx.Out(0);
        if (!src.Valid() || !dst.Valid()) return;

        const int w = src.desc.width;
        const int h = src.desc.height;

        // Start from a copy: most sensels are untouched, and this keeps the
        // detection reading ORIGINAL values rather than ones an earlier repair
        // in the same pass already changed.
        std::memcpy(dst.data, src.data, size_t(dst.Pitch()) * size_t(h));

        // The threshold is in the sensor's own units, which is what both the
        // raw loader and an R32F SRV hand over unchanged -- so it means the
        // same thing on the CPU and the GPU with no scaling on either side.
        //
        // An earlier version rescaled by the white level to be "portable
        // across bit depths". That could not be expressed in GpuConstants(),
        // which is const and runs before RunCPU, so the two paths would have
        // disagreed. A slider the user sets against what they see beats a
        // normalisation neither path can honour.
        const double minExcess = double(float(m_threshold));
        const bool   repairDark = m_repairDark;

        auto at = [&](int x, int y) {
            return double(*src.At<float>(std::clamp(x, 0, w - 1),
                                         std::clamp(y, 0, h - 1)));
        };

        int repaired = 0;
        for (int y = 0; y < h; ++y) {
            float* row = dst.At<float>(0, y);
            for (int x = 0; x < w; ++x) {
                const double c = at(x, y);

                double n[8] = {at(x - 2, y - 2), at(x, y - 2), at(x + 2, y - 2),
                               at(x - 2, y),                   at(x + 2, y),
                               at(x - 2, y + 2), at(x, y + 2), at(x + 2, y + 2)};
                std::sort(n, n + 8);
                const double med = 0.5 * (n[3] + n[4]);
                const double hiN = n[7];
                const double loN = n[0];

                // Three conditions. The absolute excess alone is not enough,
                // and neither is "brighter than all its neighbours".
                //
                // Measured on _MG_9673: those two tests together flagged 20
                // sensels, of which only 2 were real defects. The other 18 sat
                // inside bright textured highlights, where one sensel
                // legitimately wins and the excess over the median is large
                // simply because the whole neighbourhood is varying.
                //
                // What separates them is how QUIET the neighbourhood is. The
                // real defect at (1671,2108) reads 10989 among neighbours
                // spanning 1073..1306 -- a spread of ~230. The false positive
                // at (2659,1839) sits among neighbours spanning ~5000..15000.
                // A stuck sensel stands out from a calm neighbourhood; a
                // highlight peak stands in a neighbourhood already full of
                // contrast.
                //
                // So the excess must also beat the neighbourhood's own spread
                // by a margin. That keeps a genuine one-pixel star -- which
                // sits on quiet sky -- detectable, which is why the threshold
                // and the whole algorithm remain switchable for astro work.
                const double spread = n[6] - n[1];   // robust: ignores the ends
                const double margin = double(float(m_spreadFactor)) * spread;

                const bool hot  = (c - med > minExcess) && (c > hiN) &&
                                  (c - med > margin);
                const bool cold = repairDark && (med - c > minExcess) &&
                                  (c < loN) && (med - c > margin);
                if (!hot && !cold) continue;

                // The median of the same-colour neighbours: the value the
                // sensel would plausibly have read. Not the mean, which one
                // more defect in the neighbourhood would drag along.
                row[x] = float(med);
                ++repaired;
            }
        }
        m_lastRepaired = repaired;
    }

    // How many sensels the last run replaced, for the info panel. A count that
    // suddenly jumps is the signal that the threshold has been set too low and
    // the filter has started eating real detail.
    int LastRepaired() const { return m_lastRepaired; }

    // --- GPU implementation -------------------------------------------------
    //
    // Worth having even though almost nothing changes: this runs on every raw
    // load, and the CPU version costs 2.1 s at 21 MP. That is not a cost to add
    // silently to opening a photograph.
    //
    // One dispatch, no scratch: each sensel reads its eight same-colour
    // neighbours and decides independently.
    bool HasGPU() const override { return true; }

    const char* GpuSource() const override {
        return R"(
Texture2D<float4>   Src : register(t0);
RWTexture2D<float4> Dst : register(u0);

cbuffer Params : register(b0) {
    uint  Width;
    uint  Height;
    uint  MinExcessBits;    // in sensor levels
    uint  SpreadFactorBits;
    uint  RepairDark;
};

// A mosaic is single-channel, carried in .r.
float S(int x, int y) {
    int2 p = clamp(int2(x, y), int2(0, 0), int2(Width - 1, Height - 1));
    return Src[p].r;
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;

    int x = int(tid.x), y = int(tid.y);
    float c = S(x, y);

    // The eight same-colour neighbours: the Bayer period is 2, so stepping by
    // two keeps red compared with red.
    float n[8];
    n[0] = S(x-2, y-2); n[1] = S(x, y-2); n[2] = S(x+2, y-2);
    n[3] = S(x-2, y  );                   n[4] = S(x+2, y  );
    n[5] = S(x-2, y+2); n[6] = S(x, y+2); n[7] = S(x+2, y+2);

    // Insertion sort: eight elements, no branching worth avoiding, and it keeps
    // the kernel free of a sorting network nobody would be able to read.
    for (int i = 1; i < 8; ++i) {
        float v = n[i];
        int j = i - 1;
        while (j >= 0 && n[j] > v) { n[j+1] = n[j]; --j; }
        n[j+1] = v;
    }

    float med    = 0.5 * (n[3] + n[4]);
    float spread = n[6] - n[1];
    float margin = asfloat(SpreadFactorBits) * spread;
    float minEx  = asfloat(MinExcessBits);

    bool hot  = (c - med > minEx) && (c > n[7]) && (c - med > margin);
    bool cold = (RepairDark != 0) && (med - c > minEx) && (c < n[0]) &&
                (med - c > margin);

    Dst[tid.xy] = float4((hot || cold) ? med : c, 0, 0, 1);
}
)";
    }

    std::vector<uint32_t> GpuConstants(int) const override {
        auto bits = [](float f) {
            uint32_t u;
            std::memcpy(&u, &f, sizeof(u));
            return u;
        };
        // No 0..255 scaling here, unlike the RGBA8 kernels: a mosaic is R32F,
        // so an SRV hands the shader the sensor's own units rather than a
        // normalised 0..1. The threshold therefore means the same thing on both
        // paths, which is what the CPU/GPU agreement test checks.
        return {bits(float(m_threshold)),
                bits(float(m_spreadFactor)),
                uint32_t(bool(m_repairDark) ? 1 : 0)};
    }

private:
    // Default 500 levels.
    //
    // This was 2000, calibrated on a 5D at ISO 1000 whose defects sit +4100 to
    // +10500 above their neighbours. That number does not travel: a hot pixel's
    // excess scales with ISO, so on an R5 at ISO 400 the same kind of defect
    // measures only +644 to +2012 and a 2000-level floor missed three of four.
    // At ISO 100 the same sites read +2 to +352, which is why they are
    // invisible there.
    //
    // 500 is where the counts stop being defects and start being texture.
    // Repaired sensels per frame at each floor, spread factor 6:
    //
    //             2000   1000    500    250    100
    //   R5          1      3     11     114    669
    //   5D          7      8     15      49   1349
    //   RX100       42    232    642    1039  2076
    //
    // The jump below 500 is an order of magnitude on every body, and it is fine
    // detail -- grass, sand, foliage -- not defects.
    Param<float> m_threshold{this, "threshold", 500.0f, 100.0f, 8000.0f,
        {.help = "How far above its same-colour neighbours a sensel must read "
                 "to count as stuck, in sensor levels (0..16383 for a 14-bit "
                 "sensor). Lower catches fainter defects but risks eating real "
                 "single-pixel highlights. Raise it -- or switch the algorithm "
                 "off -- for astrophotography, where a star IS a genuine "
                 "one-pixel bright point.",
         .step = 50.0, .softMin = 500.0, .softMax = 6000.0}};

    // How far above the neighbourhood's own variation the excess must sit.
    //
    // 6.0, raised from 3.0 when the absolute floor came down. This is the test
    // that actually travels between cameras -- it scales with the image, where
    // the floor above does not -- so it carries more of the work now.
    //
    // Measured on a frame of coastal grass and sand, where fine detail
    // legitimately produces isolated bright sensels: at floor 500 the count
    // falls from 8334 to 642 going from 3.0 to 6.0, while all four real R5
    // defects survive with margins of 2.5x to 6x. Set to 0 to disable the test
    // and fall back to the absolute threshold alone.
    Param<float> m_spreadFactor{this, "spread_factor", 6.0f, 0.0f, 20.0f,
        {.help = "How many times the local variation the excess must exceed. "
                 "Stops bright textured areas -- specular highlights, glints -- "
                 "from being mistaken for defects, since a real stuck sensel "
                 "sits in a quiet neighbourhood. 0 disables the test.",
         .step = 0.1, .softMin = 0.0, .softMax = 12.0}};

    // Off by default: a dead sensel is far less visible than a bright one on a
    // dark background, and the test is more likely to catch real dark detail
    // (a pupil, a deep shadow) than the hot test is to catch a highlight.
    Param<bool> m_repairDark{this, "repair_dark", false,
        "Also replace sensels reading far BELOW their neighbours (dead pixels)."};

    int m_lastRepaired = 0;
};

REGISTER_ALGORITHM(HotPixelRepair);

} // namespace
} // namespace tglab
