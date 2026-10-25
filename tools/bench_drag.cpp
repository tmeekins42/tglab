// bench_drag: what a slider drag costs, tick by tick.
//
//   bench_drag <image> <script> <value> [value...] [--proxy S] [--cpu]
//              [--group a b c ...] [--region x,y,w,h]
//
// --group loads the listed files as one group, named "group" in the script, as
// the app's palette does (the <image> argument is then ignored). --region asks
// for the visible rectangle only, in full-resolution pixels -- what the app does
// when zoomed in.
//
// The script is a template: every "%V" in it is replaced by the next value,
// the result interpreted into a fresh pipeline, and run against the previous
// one exactly as the app does on each slider tick -- so cached stages are
// reused, dirty ones re-run, and the time printed is what the user waits
// for. The first value is the cold run.
//
// Written because nothing else measured the repeated case. A stage whose
// kernels or scratch planes were rebuilt on every tick looks fine in every
// one-shot benchmark and every test, and costs its whole compile time again
// each time the slider moves.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <d3d12.h>

#include "../src/core/image.h"
#include "../src/algo_util/white_balance.h"
#include "../src/core/image_io.h"
#include "../src/core/pipeline.h"
#include "../src/gpu/compute.h"
#include "../src/script/interp.h"
#include "../src/script/parser.h"

using namespace tglab;

int main(int argc, char** argv) {
    std::vector<std::string> pos, group;
    float proxy = 1.0f;
    bool cpu = false;
    Pipeline::Rect region{};
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--proxy" && i + 1 < argc) proxy = float(std::atof(argv[++i]));
        else if (a == "--cpu") cpu = true;
        else if (a == "--region" && i + 1 < argc)
            std::sscanf(argv[++i], "%d,%d,%d,%d", &region.x, &region.y, &region.w, &region.h);
        else if (a == "--group") {
            while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) group.push_back(argv[++i]);
        }
        else pos.push_back(a);
    }
    if (pos.size() < 3) {
        std::printf("usage: bench_drag <image> <script with %%V> <value> [value...] "
                    "[--proxy S] [--cpu]\n");
        return 1;
    }
    Image img;
    std::string err;
    ImageSet set;
    if (!group.empty()) {
        for (const std::string& g : group) {
            Image fr;
            if (!LoadImageFile(g, &fr, &err)) { std::printf("%s: %s\n", g.c_str(), err.c_str()); return 1; }
            set.images.push_back(std::move(fr));
        }
        set.shape = Shape{{{"frame", int(set.images.size())}}};
    } else if (!LoadImageFile(pos[0], &img, &err)) { std::printf("%s\n", err.c_str()); return 1; }
    std::ifstream f(pos[1]);
    if (!f) { std::printf("cannot read %s\n", pos[1].c_str()); return 1; }
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string tmpl = ss.str();

    SourceImage src{group.empty() ? "test" : "group", 0};
    const ImageDesc& fd = group.empty() ? img.Desc() : set.images.front().Desc();
    src.isMosaic = fd.IsMosaic();
    AsShotWhiteBalance(fd, &src.asShotKelvin, &src.asShotTint);
    if (!group.empty()) src.shape = set.shape;
    const std::vector<SourceImage> names{src};
    std::vector<Data> sources;
    if (group.empty()) sources.push_back(Data{std::move(img)});
    else sources.push_back(Data{std::move(set)});
    const std::vector<uint64_t> versions{1};

    ID3D12Device* dev = nullptr;
    ComputeContext gpu;
    ComputeContext* gpuPtr = nullptr;
    if (!cpu && SUCCEEDED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&dev))) &&
        gpu.Init(dev))
        gpuPtr = &gpu;
    std::printf("%s, proxy %.2f\n", gpuPtr ? "GPU" : "CPU", double(proxy));

    std::unique_ptr<Pipeline> prev;
    double total = 0.0;
    for (size_t k = 2; k < pos.size(); ++k) {
        std::string s = tmpl;
        for (size_t at; (at = s.find("%V")) != std::string::npos;) s.replace(at, 2, pos[k]);
        Program prog;
        if (!Parse(s, &prog, &err)) { std::printf("parse: %s\n", err.c_str()); return 1; }
        UiState ui;
        auto p = std::make_unique<Pipeline>();
        const InterpResult ir = Interpret(prog, names, &ui, p.get());
        if (!ir.ok) { std::printf("interpret: %s\n", ir.error.c_str()); return 1; }
        p->SetProxyScale(proxy);
        if (region.Valid()) p->SetRegion(region);
        const auto t0 = std::chrono::steady_clock::now();
        if (!p->Execute(&sources, prev.get(), &err, gpuPtr, ExecMode::Auto, &versions)) {
            std::printf("execute: %s\n", err.c_str());
            return 1;
        }
        const double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (k > 2) total += ms;
        const size_t fdi = p->FirstDirtyStage();
        std::printf("  %-10s %8.1f ms   gpu %d  cpu %d  cached %d  first re-run: %s%s\n", pos[k].c_str(), ms,
                    p->GpuStageCount(), p->CpuStageCount(), p->CachedStageCount(),
                    fdi < p->Stages().size() ? p->Stages()[fdi].algoName.c_str() : "(none)",
                    k == 2 ? "   (cold)" : "");
        for (const std::string& fb : p->GpuFallbacks()) std::printf("    fallback: %s\n", fb.c_str());
        prev = std::move(p);
    }
    if (pos.size() > 3)
        std::printf("  warm ticks: mean %.1f ms\n", total / double(pos.size() - 3));
    return 0;
}
