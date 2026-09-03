// Measures lens falloff from a sweep of frames, without a flat field.
//
// THE TRICK, and the reason this works on ordinary photographs:
//
// Lens vignetting is fixed to the SENSOR -- it darkens the same pixels of every
// frame by the same fraction, no matter where the camera was pointed. Scene
// content is fixed to the WORLD, so a pan puts something different at any given
// sensor position in every frame.
//
// So average many frames together in SENSOR coordinates. The scene, being
// different each time, averages toward a constant; the falloff, being identical
// each time, survives untouched. With fifteen frames of a pan that separation is
// already good, and it needs no flat field, no lens profile, and no EXIF.
//
// WHERE THIS DOES NOT WORK, which turned out to be the interesting case:
//
// It needs the scene to actually differ at a given sensor position from frame to
// frame. A purely HORIZONTAL pan does not provide that vertically -- the top of
// every frame is sky, the middle is the lake, the bottom is snow, in all fifteen
// frames. Averaging cannot remove a structure that is identical in every frame,
// so what comes out is the scene's own vertical layout, cleanly measured and
// completely useless as a lens profile. Run on a real 15-frame Crater Lake pan
// this reported a 4x swing from row to row: sky bright, water dark, foreground
// snow brightest. Lens falloff is a few percent and was buried under it.
//
// That is why the radial check exists and why it is worth reading FIRST.
// Vignetting is a function of distance from the optical centre alone, so cells
// at equal radius must agree. On that sweep the spread at every radius was
// 0.31-0.52 -- ten to twenty times what a lens would produce -- which says
// plainly that the grid is showing scene, not lens, and must not be divided out.
//
// For this to measure a lens it needs either a genuine flat field (an evenly lit
// wall, or defocused sky), or a MULTI-ROW pan whose vertical motion decorrelates
// the vertical structure the same way the horizontal motion decorrelates the
// horizontal.
//
//   vignette_profile <raw> [raw...]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "../src/algo_util/pixel_buffer.h"
#include "../src/core/algorithm.h"
#include "../src/core/image.h"
#include "../src/core/raw_io.h"

using namespace tglab;

namespace {

bool Demosaic(const char* name, const Image& in, Image* out) {
    auto algo = Registry::Get().Create(name);
    if (!algo) return false;
    std::vector<Data> ins;
    ins.push_back(Data{const_cast<Image&>(in).Clone()});
    std::vector<const Data*> inPtrs{&ins[0]};
    std::vector<Data> outs(1);
    ImageDesc d = in.Desc();
    d.format = Format::RGBA16F;
    d.cfa    = CfaPattern::None;
    d.linear = true;
    Image img;
    img.Alloc(d);
    outs[0] = Data{std::move(img)};
    RunCtx ctx(inPtrs, outs);
    algo->PrepareGpu({in.Desc()});
    algo->RunCPU(ctx);
    Image* o = std::get_if<Image>(&outs[0]);
    if (!o || !o->Valid()) return false;
    *out = o->Clone();
    return true;
}

constexpr int kGrid = 9;   // odd, so one cell sits on the centre

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("usage: vignette_profile <raw> <raw> [raw...]\n"
                    "  needs several frames of the same sweep\n");
        return 2;
    }

    std::vector<double> sum(kGrid * kGrid, 0.0);
    std::vector<long long> cnt(kGrid * kGrid, 0);
    int frames = 0;

    for (int i = 1; i < argc; ++i) {
        Image mosaic;
        std::string err;
        if (!LoadRawMosaic(argv[i], &mosaic, &err)) {
            std::printf("  %s: %s\n", argv[i], err.c_str());
            continue;
        }
        Image rgb;
        if (!Demosaic("demosaic_ahd", mosaic, &rgb)) continue;

        ImageView v = rgb.MapCpuRead();
        PixelBuffer pb;
        pb.Unpack(v);
        if (!pb.Valid()) continue;

        // Per-frame NORMALISED by its own mean before accumulating, so a frame
        // that is simply brighter overall does not tilt the profile. What is
        // wanted is the SHAPE, not the level.
        std::vector<double> cell(kGrid * kGrid, 0.0);
        std::vector<long long> ccnt(kGrid * kGrid, 0);
        for (int y = 0; y < pb.Height(); y += 4)
            for (int x = 0; x < pb.Width(); x += 4) {
                const float* q = pb.At(x, y);
                const double y0 = (pb.Channels() >= 3)
                    ? 0.2126 * q[0] + 0.7152 * q[1] + 0.0722 * q[2]
                    : q[0];
                if (y0 <= 1e-6) continue;
                const int gx = std::min(kGrid - 1, x * kGrid / pb.Width());
                const int gy = std::min(kGrid - 1, y * kGrid / pb.Height());
                cell[size_t(gy * kGrid + gx)] += y0;
                ++ccnt[size_t(gy * kGrid + gx)];
            }

        double mean = 0.0;
        int nz = 0;
        for (int k = 0; k < kGrid * kGrid; ++k)
            if (ccnt[size_t(k)]) {
                cell[size_t(k)] /= double(ccnt[size_t(k)]);
                mean += cell[size_t(k)];
                ++nz;
            }
        if (!nz) continue;
        mean /= double(nz);
        if (mean <= 0.0) continue;

        for (int k = 0; k < kGrid * kGrid; ++k)
            if (ccnt[size_t(k)]) {
                sum[size_t(k)] += cell[size_t(k)] / mean;
                ++cnt[size_t(k)];
            }
        ++frames;
        std::printf("  averaged %s\n", argv[i]);
    }

    if (frames < 2) {
        std::printf("\nneed at least two frames that loaded\n");
        return 1;
    }

    std::printf("\n%d frames averaged in SENSOR coordinates.\n"
                "Scene content differs per frame and averages out; lens falloff\n"
                "is identical in every frame and survives.\n\n", frames);

    std::vector<double> prof(kGrid * kGrid, 0.0);
    for (int k = 0; k < kGrid * kGrid; ++k)
        prof[size_t(k)] = cnt[size_t(k)] ? sum[size_t(k)] / double(cnt[size_t(k)])
                                         : 0.0;

    const double centre = prof[size_t((kGrid / 2) * kGrid + kGrid / 2)];
    std::printf("relative to centre (%.4f):\n", centre);
    for (int gy = 0; gy < kGrid; ++gy) {
        std::printf("  ");
        for (int gx = 0; gx < kGrid; ++gx) {
            const double p = prof[size_t(gy * kGrid + gx)];
            std::printf(" %6.3f", centre > 0.0 ? p / centre : 0.0);
        }
        std::printf("\n");
    }

    // Radial summary. Vignetting is a function of distance from the optical
    // centre alone, so if this is the lens the values at equal radius agree and
    // fall monotonically outward.
    double worstSpread = 0.0;
    std::printf("\nby radius (0 = centre, 1 = corner):\n");
    for (int band = 0; band < 5; ++band) {
        const double r0 = double(band) / 5.0, r1 = double(band + 1) / 5.0;
        double s = 0.0, s2 = 0.0;
        int n = 0;
        for (int gy = 0; gy < kGrid; ++gy)
            for (int gx = 0; gx < kGrid; ++gx) {
                const double dx = (double(gx) - (kGrid - 1) / 2.0) / ((kGrid - 1) / 2.0);
                const double dy = (double(gy) - (kGrid - 1) / 2.0) / ((kGrid - 1) / 2.0);
                const double r = std::sqrt(dx * dx + dy * dy) / std::sqrt(2.0);
                if (r < r0 || r >= r1) continue;
                const double p = prof[size_t(gy * kGrid + gx)];
                if (p <= 0.0) continue;
                const double rel = centre > 0.0 ? p / centre : 0.0;
                s += rel;
                s2 += rel * rel;
                ++n;
            }
        if (!n) continue;
        const double m = s / double(n);
        const double sd = std::sqrt(std::max(s2 / double(n) - m * m, 0.0));
        // The SPREAD at a given radius is the test. Small means the variation
        // really is radial, i.e. the lens. Large means whatever this is depends
        // on direction as well, so it is not vignetting and dividing it out
        // would damage the picture.
        std::printf("  r %.1f-%.1f:  %.4f  (spread %.4f, %d cells)%s\n",
                    r0, r1, m, sd, n,
                    sd > 0.02 ? "   <- not radial" : "");
        worstSpread = std::max(worstSpread, sd);
    }

    // The verdict, stated rather than left to be inferred from the numbers.
    //
    // A grid that is not radial is the SCENE, and dividing it out would burn the
    // scene's own lighting into every frame. Saying so is the whole value of the
    // radial check -- without it this table looks exactly like a lens profile.
    std::printf("\n");
    if (worstSpread > 0.02) {
        std::printf("NOT A LENS PROFILE. Cells at equal radius disagree by up to\n"
                    "%.3f, where vignetting would agree to a few thousandths.\n"
                    "What this measured is the SCENE, which a single-row pan\n"
                    "cannot average away vertically -- every frame has sky at the\n"
                    "top and ground at the bottom. Do not divide this out.\n\n"
                    "To measure the lens: shoot a flat field (evenly lit wall or\n"
                    "defocused sky), or use a multi-row pan.\n", worstSpread);
    } else {
        std::printf("Consistent with lens falloff: radially symmetric to within\n"
                    "%.4f. Safe to use as a correction profile.\n", worstSpread);
    }
    return 0;
}
