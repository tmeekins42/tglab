// Reports what actually differs in exposure across a sweep of frames.
//
// Before writing a correction it is worth knowing which of three things the
// seams are, because they want different fixes:
//
//   1. The camera changed its settings between frames (aperture priority, auto
//      ISO). EXIF says so directly, and the fix is a per-frame GAIN derived
//      from the settings -- exact, no fitting needed.
//
//   2. The settings were locked but the scene brightness differs anyway, e.g.
//      the sun moved or a cloud passed. A per-frame gain still fixes it, but it
//      has to be SOLVED from the overlaps rather than read off EXIF.
//
//   3. Neither -- the frames match globally and only their EDGES differ. That
//      is lens vignetting, and a single gain per frame cannot fix it: the
//      correction has to vary across the frame.
//
// The centre-vs-corner figures separate case 3 from the first two.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "../src/core/exif.h"
#include "../src/core/image.h"
#include "../src/core/raw_io.h"

using namespace tglab;

namespace {

// Mean of the raw mosaic inside, and outside, a centred box.
//
// The MOSAIC rather than a demosaiced image: this is about how much light hit
// the sensor, and demosaicing would fold in white balance and the colour matrix
// -- both per-frame decisions that would confuse the measurement.
//
// Both halves in one pass so the frame is only walked once; at 30 MP a second
// pass is a needless second cache sweep.
// Mean over four corner boxes and the centre box, as a RADIAL profile.
//
// This separates the two things a centre-vs-surround ratio cannot. Lens
// vignetting is radially symmetric and fixed to the lens, so every frame darkens
// the same way toward all four corners. A scene gradient -- panning from snow
// toward trees -- is not symmetric: it darkens one side and not the other.
//
// So: if left and right corners track each other while both fall below centre,
// it is the lens. If they diverge, it is the scene, and no per-frame correction
// should touch it because the difference is real.
void CornerProfile(const Image& img, double* centre, double* left, double* right) {
    Image& m = const_cast<Image&>(img);
    const ImageView v = m.MapCpuRead();
    if (!v.Valid()) { *centre = *left = *right = 0.0; return; }

    const int w = v.desc.width, h = v.desc.height;
    const int bw = std::max(1, w / 6), bh = std::max(1, h / 6);

    auto box = [&](int x0, int y0) {
        double s = 0.0;
        long long n = 0;
        for (int y = y0; y < std::min(y0 + bh, h); ++y)
            for (int x = x0; x < std::min(x0 + bw, w); ++x) {
                s += double(*v.At<float>(x, y));
                ++n;
            }
        return n ? s / double(n) : 0.0;
    };

    *centre = box((w - bw) / 2, (h - bh) / 2);
    // Corners averaged top and bottom, so a vertical scene gradient (sky over
    // ground) cancels and only the horizontal asymmetry survives.
    *left  = 0.5 * (box(0, 0)          + box(0, h - bh));
    *right = 0.5 * (box(w - bw, 0)     + box(w - bw, h - bh));
}

void MeanInOut(const Image& img, double frac, double* inside, double* outside) {
    Image& m = const_cast<Image&>(img);
    const ImageView v = m.MapCpuRead();
    if (!v.Valid()) { *inside = *outside = 0.0; return; }

    const int w = v.desc.width, h = v.desc.height;
    const int bw = std::max(1, int(double(w) * frac));
    const int bh = std::max(1, int(double(h) * frac));
    const int x0 = (w - bw) / 2, y0 = (h - bh) / 2;

    double si = 0.0, so = 0.0;
    long long ni = 0, no = 0;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const double s = double(*v.At<float>(x, y));
            if (x >= x0 && x < x0 + bw && y >= y0 && y < y0 + bh) {
                si += s; ++ni;
            } else {
                so += s; ++no;
            }
        }
    *inside  = ni ? si / double(ni) : 0.0;
    *outside = no ? so / double(no) : 0.0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: frame_exposure <raw> [raw...]\n");
        return 2;
    }

    std::printf("%-16s %-10s %-8s %-8s %9s %9s %6s %8s %8s\n",
                "frame", "shutter", "f", "ISO", "centre", "surround", "c/s",
                "L/centre", "R/centre");

    std::vector<double> centres;
    for (int i = 1; i < argc; ++i) {
        Image raw;
        std::string err;
        if (!LoadRawMosaic(argv[i], &raw, &err)) {
            std::printf("%-16s FAILED: %s\n", argv[i], err.c_str());
            continue;
        }
        const ExifData ex = ReadExif(argv[i]);

        double c = 0.0, s = 0.0;
        MeanInOut(raw, 0.4, &c, &s);

        double cc = 0.0, lf = 0.0, rt = 0.0;
        CornerProfile(raw, &cc, &lf, &rt);

        // Basename only, so the table stays readable.
        std::string name = argv[i];
        const size_t slash = name.find_last_of("/\\");
        if (slash != std::string::npos) name = name.substr(slash + 1);

        std::printf("%-16s %-10s %-8s %-8s %9.1f %9.1f %6.3f %8.3f %8.3f\n",
                    name.c_str(),
                    ex.exposureTime.empty() ? "-" : ex.exposureTime.c_str(),
                    ex.aperture.empty()     ? "-" : ex.aperture.c_str(),
                    ex.iso.empty()          ? "-" : ex.iso.c_str(),
                    c, s, s > 0.0 ? c / s : 0.0,
                    cc > 0.0 ? lf / cc : 0.0,
                    cc > 0.0 ? rt / cc : 0.0);
        centres.push_back(c);
    }

    if (centres.size() > 1) {
        const auto mm = std::minmax_element(centres.begin(), centres.end());
        const double lo = *mm.first, hi = *mm.second;
        std::printf("\ncentre brightness spread: %.1f to %.1f", lo, hi);
        if (lo > 0.0)
            std::printf("  (%.2f stops)", std::log2(hi / lo));
        std::printf("\n");
    }
    return 0;
}
