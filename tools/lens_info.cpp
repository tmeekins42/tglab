// lens_info: which lensfun profile each raw would get, and what it says.
//
//   lens_info [--render <prefix>] <raw> [raw...]
//
// --render also develops each raw with lens_correct's profile off and on and
// saves <prefix><n>_off.png and <prefix><n>_on.png, for looking at.
//
// Prints the lens and body as the file names them, the database entry they
// matched, and the profile in lens_correct's units, so a mismatch -- the
// wrong lens, the wrong calibration crop -- is visible without the app.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "../src/core/image.h"
#include "../src/core/image_io.h"
#include "../src/core/lensfun.h"
#include "../src/core/pipeline.h"
#include "../src/core/raw_io.h"
#include "../src/script/interp.h"
#include "../src/script/parser.h"

using namespace tglab;

// Demosaic, lens_correct with the profile on or off, a tone map, saved.
static void Render(const Image& mosaic, bool profile, const std::string& path) {
    std::vector<Data> src;
    src.push_back(Data{const_cast<Image&>(mosaic).Clone()});
    SourceImage si;
    si.name = "test";
    si.isMosaic = true;
    std::vector<SourceImage> names{si};
    UiState ui;
    Pipeline pipe;
    Program prog;
    std::string err;
    const std::string script =
        std::string("s = image(\"test\")\no = lens_correct(s, auto_ca = 0, profile = ") +
        (profile ? "1" : "0") + ")\nt = tonemap(o)\ndisplay(t)\n";
    if (!Parse(script, &prog, &err) || !Interpret(prog, names, &ui, &pipe).ok) {
        std::printf("  script: %s\n", err.c_str());
        return;
    }
    const std::vector<uint64_t> versions{1};
    if (!pipe.Execute(&src, nullptr, &err, nullptr, ExecMode::ForceCPU, &versions)) {
        std::printf("  run: %s\n", err.c_str());
        return;
    }
    for (const auto& st : pipe.Stages())
        if (!st.Report().empty()) std::printf("  [%s] %s\n", profile ? "on" : "off", st.Report().c_str());
    const Data* d = pipe.Resolve(pipe.Viewers()[0].source, &src);
    if (!d || !std::holds_alternative<Image>(*d)) return;
    if (!SaveImage(path, const_cast<Image&>(std::get<Image>(*d)), &err))
        std::printf("  save: %s\n", err.c_str());
}

int main(int argc, char** argv) {
    std::string render;
    int first = 1;
    if (argc > 2 && std::string(argv[1]) == "--render") {
        render = argv[2];
        first = 3;
    }
    if (argc <= first) {
        std::printf("usage: lens_info [--render <prefix>] <raw> [raw...]\n");
        return 1;
    }
    std::printf("database: %s (%d lenses, %d cameras)\n", LensfunDirectory().c_str(),
                LensfunLensCount(), LensfunCameraCount());
    for (int i = first; i < argc; ++i) {
        Image img;
        std::string err;
        if (!LoadRawMosaic(argv[i], &img, &err)) {
            std::printf("%s: %s\n", argv[i], err.c_str());
            continue;
        }
        const ImageDesc& d = img.Desc();
        const LensTag& t = LensTagOf(d.lensTag);
        std::printf("\n%s\n  file: %s %s, \"%s\" at %.1f mm f/%.1f, crop35 %.3f, %dx%d\n", argv[i],
                    t.make.c_str(), t.model.c_str(), t.lens.c_str(), d.focalMm, d.aperture, d.crop35,
                    d.width, d.height);
        const LensProfile p = FindLensProfile(t.make.c_str(), t.model.c_str(), t.lens.c_str(), d.focalMm,
                                              d.aperture, d.crop35);
        if (!p.valid) {
            std::printf("  no profile\n");
        } else {
            std::printf("  lens: %s; camera: %s; crop %.3f, calibrated at %.3f\n", p.lens.c_str(),
                        p.camera.empty() ? "(not in database)" : p.camera.c_str(), p.cropFactor,
                        p.calibCrop);
            if (p.hasDistortion) {
                double f1 = 0;
                for (int k = 4; k >= 0; --k) f1 = f1 + p.dist[k];
                std::printf("  distortion: %.5f %+.5f r %+.5f r^2 %+.5f r^3 %+.5f r^4  (corner %.4f)\n",
                            p.dist[0], p.dist[1], p.dist[2], p.dist[3], p.dist[4], f1);
            }
            if (p.hasTca)
                std::printf("  tca: red %.6f %+.6f r %+.6f r^2, blue %.6f %+.6f r %+.6f r^2\n",
                            p.tca[0][0], p.tca[0][1], p.tca[0][2], p.tca[1][0], p.tca[1][1], p.tca[1][2]);
            if (p.hasVignetting) {
                const double kept = 1 + p.vig[0] + p.vig[1] + p.vig[2];
                std::printf("  vignetting: k %.4f %.4f %.4f  (corner keeps %.3f, %.2f stops)\n",
                            p.vig[0], p.vig[1], p.vig[2], kept, -std::log2(kept));
            }
        }
        if (!render.empty()) {
            const std::string base = render + std::to_string(i - first);
            Render(img, false, base + "_off.png");
            Render(img, true, base + "_on.png");
        }
    }
    return 0;
}
