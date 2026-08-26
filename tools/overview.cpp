// Render a whole frame small, with a coordinate grid, to pick crop regions.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include "../src/algo_util/tone_curve.h"
#include "../src/core/algorithm.h"
#include "../src/core/image.h"
#include "../src/core/image_io.h"
#include "../src/core/raw_io.h"
using namespace tglab;
int main(int argc, char** argv) {
    Image raw; std::string err;
    if (!LoadRawMosaic(argv[1], &raw, &err)) { std::printf("%s\n", err.c_str()); return 1; }
    auto algo = Registry::Get().Create("demosaic_consistent");
    std::vector<Data> ins; ins.push_back(Data{raw.Clone()});
    std::vector<const Data*> ip{&ins[0]};
    std::vector<Data> outs(1);
    ImageDesc d = raw.Desc(); d.format=Format::RGBA16F; d.cfa=CfaPattern::None; d.linear=true;
    Image img; img.Alloc(d); outs[0]=Data{std::move(img)};
    RunCtx ctx(ip,outs); algo->PrepareGpu({raw.Desc()}); algo->RunCPU(ctx);
    Image* o = std::get_if<Image>(&outs[0]);
    ImageView sv = o->MapCpuRead();
    const int F = 8;
    const int ow = d.width/F, oh = d.height/F;
    Image png; ImageDesc pd{ow,oh,Format::RGBA8}; png.Alloc(pd);
    ImageView dv = png.MapCpuWrite();
    const float gain = std::exp2(float(argc>3?atof(argv[3]):2.0));
    for (int y=0;y<oh;++y) for (int x=0;x<ow;++x) {
        double acc[3]={0,0,0};
        for (int sy=0;sy<F;++sy) for (int sx=0;sx<F;++sx) {
            const uint16_t* s = sv.At<uint16_t>(x*F+sx, y*F+sy);
            for (int c=0;c<3;++c) acc[c]+=HalfToFloat(s[c]);
        }
        uint8_t* p = dv.At<uint8_t>(x,y);
        for (int c=0;c<3;++c)
            p[c]=uint8_t(ToneCurve(float(acc[c]/(F*F))*gain)*255.0f+0.5f);
        // Grid every 500 source pixels, to read coordinates off the image.
        const int srcx=x*F, srcy=y*F;
        if (srcx%500 < F || srcy%500 < F) { p[0]=255;p[1]=0;p[2]=0; }
        p[3]=255;
    }
    std::string e; SavePng(argv[2], png, &e);
    std::printf("wrote %s  (%dx%d, grid every 500 source px)\n", argv[2], ow, oh);
    return 0;
}
