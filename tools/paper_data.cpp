// paper_data — generates the tables and figures for papers/.
//
// Kept in the tree rather than run once and discarded, so every number in the
// paper can be regenerated and checked. A figure whose provenance is "I ran
// something in August" is not a result.
//
//   build/Release/paper_data.exe --table  <raw>...        LaTeX table rows
//   build/Release/paper_data.exe --crops  <raw> x y size  PNG crops per method
//
// The measurements are deliberately split into DETAIL and NOISE rather than
// combined into one error figure, because the two pull in opposite directions
// and a single number hides the trade. See the paper's methodology section.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../src/algo_util/tone_curve.h"
#include "../src/core/algorithm.h"
#include "../src/core/image.h"
#include "../src/core/image_io.h"
#include "../src/core/raw_io.h"
#include "../src/script/value.h"

using namespace tglab;

namespace {

struct Method {
    const char* algo;
    std::vector<std::pair<std::string, double>> params;
    const char* label;   // for the paper
    const char* slug;    // for filenames
};

const Method kMethods[] = {
    {"demosaic_bilinear",   {}, "Bilinear", "bilinear"},
    {"demosaic_malvar",     {}, "Malvar",   "malvar"},
    {"demosaic_ahd",        {}, "AHD",      "ahd"},
    {"demosaic_consistent", {}, "Proposed", "consistent"},
};

bool Run(const Method& m, const Image& in, Image* out) {
    auto algo = Registry::Get().Create(m.algo);
    if (!algo) return false;
    for (const auto& [pn, pv] : m.params) {
        ParamBase* p = algo->FindParam(pn);
        std::string e;
        if (p) p->SetFromScript(Value{pv}, &e);
    }
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

struct Metrics {
    double detail = 0, luma = 0, chroma = 0;
};

// Detail in textured regions; noise in flat ones.
//
// Flat and textured are separated by the local luminance range relative to its
// own level, so the same thresholds mean the same thing in shadow and highlight.
// In a flat region any high-frequency content is noise by definition, which is
// the only place the question can be asked without ambiguity.
Metrics Measure(Image& img) {
    Metrics r;
    ImageView v = img.MapCpuRead();
    const ImageDesc& d = img.Desc();

    auto L = [&](int x, int y) {
        const uint16_t* p = v.At<uint16_t>(std::clamp(x, 0, d.width - 1),
                                           std::clamp(y, 0, d.height - 1));
        return 0.2126 * HalfToFloat(p[0]) + 0.7152 * HalfToFloat(p[1]) +
               0.0722 * HalfToFloat(p[2]);
    };
    auto C = [&](int x, int y, int ch) {
        const uint16_t* p = v.At<uint16_t>(std::clamp(x, 0, d.width - 1),
                                           std::clamp(y, 0, d.height - 1));
        return double(HalfToFloat(p[ch]));
    };

    std::vector<double> detail, lumaSig, chromaSig;
    for (int y = 8; y < d.height - 8; y += 7) {
        for (int x = 8; x < d.width - 8; x += 7) {
            double lo = 1e30, hi = -1e30, mean = 0;
            for (int dy = -2; dy <= 2; ++dy)
                for (int dx = -2; dx <= 2; ++dx) {
                    const double l = L(x + dx, y + dy);
                    lo = std::min(lo, l);
                    hi = std::max(hi, l);
                    mean += l;
                }
            mean /= 25.0;
            if (mean < 0.02) continue;

            if ((hi - lo) <= 0.25 * mean) {
                double var = 0;
                for (int dy = -2; dy <= 2; ++dy)
                    for (int dx = -2; dx <= 2; ++dx) {
                        const double e = L(x + dx, y + dy) - mean;
                        var += e * e;
                    }
                lumaSig.push_back(std::sqrt(var / 25.0) / mean);

                double mrg = 0, mbg = 0;
                for (int dy = -2; dy <= 2; ++dy)
                    for (int dx = -2; dx <= 2; ++dx) {
                        mrg += C(x+dx, y+dy, 0) - C(x+dx, y+dy, 1);
                        mbg += C(x+dx, y+dy, 2) - C(x+dx, y+dy, 1);
                    }
                mrg /= 25.0; mbg /= 25.0;
                double cvar = 0;
                for (int dy = -2; dy <= 2; ++dy)
                    for (int dx = -2; dx <= 2; ++dx) {
                        const double a = (C(x+dx,y+dy,0) - C(x+dx,y+dy,1)) - mrg;
                        const double b = (C(x+dx,y+dy,2) - C(x+dx,y+dy,1)) - mbg;
                        cvar += 0.5 * (a*a + b*b);
                    }
                chromaSig.push_back(std::sqrt(cvar / 25.0) / mean);
            } else if ((hi - lo) >= 0.60 * mean) {
                const double lap = 4*L(x,y) - L(x-1,y) - L(x+1,y) - L(x,y-1) - L(x,y+1);
                detail.push_back(std::fabs(lap) / mean);
            }
        }
    }
    auto med = [](std::vector<double>& t) {
        if (t.empty()) return 0.0;
        std::sort(t.begin(), t.end());
        return t[t.size() / 2];
    };
    r.detail = med(detail);
    r.luma   = med(lumaSig);
    r.chroma = med(chromaSig);
    return r;
}

const char* Basename(const char* p) {
    const char* a = std::strrchr(p, '\\');
    const char* b = std::strrchr(p, '/');
    const char* c = (a > b) ? a : b;
    return c ? c + 1 : p;
}

// A LaTeX-safe version of a filename: underscores would otherwise be subscripts.
std::string TexEscape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '_') o += "\\_";
        else          o += c;
    }
    return o;
}

int DoTable(int argc, char** argv) {
    std::printf("%% Generated by tools/paper_data.cpp --table\n");
    std::printf("%% Each figure is relative to bilinear on the same frame.\n");
    for (int i = 2; i < argc; ++i) {
        Image raw;
        std::string err;
        if (!LoadRawMosaic(argv[i], &raw, &err)) {
            std::fprintf(stderr, "%s: %s\n", argv[i], err.c_str());
            continue;
        }
        ExifData ex;
        ReadRawMetadata(argv[i], &ex);

        Metrics base;
        bool haveBase = false;
        std::string row = TexEscape(Basename(argv[i]));
        // ISO as a bare number, for sorting in the table.
        std::string iso = ex.iso;
        const size_t sp = iso.find(' ');
        if (sp != std::string::npos) iso = iso.substr(sp + 1);
        row += " & " + (iso.empty() ? std::string("--") : iso);

        // Collect every method's figures first, so the best in each column can
        // be emphasised. Rounded to whole percent before comparing, so what is
        // bolded matches what is printed -- comparing at full precision would
        // occasionally bold one of two entries that both display as the same
        // number, which reads as an error.
        struct Row { double det, lum, chr; bool ok; };
        std::vector<Row> rows;
        for (const Method& m : kMethods) {
            Image out;
            if (!Run(m, raw, &out)) { rows.push_back({0, 0, 0, false}); continue; }
            const Metrics t = Measure(out);
            if (!haveBase) { base = t; haveBase = true; }
            rows.push_back({std::round(100.0 * t.detail / std::max(base.detail, 1e-12)),
                            std::round(100.0 * t.luma   / std::max(base.luma,   1e-12)),
                            std::round(100.0 * t.chroma / std::max(base.chroma, 1e-12)),
                            true});
        }

        // Bilinear is excluded from the comparison. It is 100 in every column
        // by construction, being the baseline every other figure is expressed
        // against -- bolding it as "lowest noise" would present the reference
        // as a winner, which it is not.
        double bestDet = -1e30, bestLum = 1e30, bestChr = 1e30;
        for (size_t k = 1; k < rows.size(); ++k) {
            if (!rows[k].ok) continue;
            bestDet = std::max(bestDet, rows[k].det);
            bestLum = std::min(bestLum, rows[k].lum);
            bestChr = std::min(bestChr, rows[k].chr);
        }

        auto cell = [](double v, bool best) {
            char b[64];
            if (best) std::snprintf(b, sizeof b, " & \\textbf{%.0f}", v);
            else      std::snprintf(b, sizeof b, " & %.0f", v);
            return std::string(b);
        };

        for (size_t k = 0; k < rows.size(); ++k) {
            if (!rows[k].ok) { row += " & -- & -- & --"; continue; }
            const bool isBase = (k == 0);
            row += cell(rows[k].det, !isBase && rows[k].det == bestDet);
            row += cell(rows[k].lum, !isBase && rows[k].lum == bestLum);
            row += cell(rows[k].chr, !isBase && rows[k].chr == bestChr);
        }
        std::printf("%s \\\\\n", row.c_str());
        std::fflush(stdout);
    }
    return 0;
}

int DoCrops(int argc, char** argv) {
    if (argc < 6) {
        std::fprintf(stderr, "usage: paper_data --crops <raw> <x> <y> <size> [outdir] [stops]\n");
        return 2;
    }
    const char* path = argv[2];
    const int cx = std::atoi(argv[3]);
    const int cy = std::atoi(argv[4]);
    const int cs = std::atoi(argv[5]);
    const std::string outdir = (argc > 6) ? argv[6] : ".";
    const float stops = (argc > 7) ? float(std::atof(argv[7])) : 2.0f;

    Image raw;
    std::string err;
    if (!LoadRawMosaic(path, &raw, &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    const ImageDesc& fd = raw.Desc();

    // The crop must keep the CFA phase: an odd offset rotates the pattern and
    // swaps red with blue.
    const int x0 = std::clamp(cx & ~1, 0, fd.width  - cs - 2);
    const int y0 = std::clamp(cy & ~1, 0, fd.height - cs - 2);

    Image crop;
    ImageDesc cd = fd;
    cd.width = cs; cd.height = cs;
    crop.Alloc(cd);
    {
        ImageView sv = raw.MapCpuRead();
        ImageView dv = crop.MapCpuWrite();
        for (int y = 0; y < cs; ++y)
            for (int x = 0; x < cs; ++x)
                *dv.At<float>(x, y) = *sv.At<float>(x0 + x, y0 + y);
    }

    const float gain = std::exp2(stops);
    for (const Method& m : kMethods) {
        Image out;
        if (!Run(m, crop, &out)) continue;

        Image png;
        ImageDesc pd{cs, cs, Format::RGBA8};
        png.Alloc(pd);
        ImageView sv = out.MapCpuRead();
        ImageView dv = png.MapCpuWrite();
        for (int y = 0; y < cs; ++y)
            for (int x = 0; x < cs; ++x) {
                const uint16_t* s = sv.At<uint16_t>(x, y);
                uint8_t* d = dv.At<uint8_t>(x, y);
                // Developed through the same tone curve the viewer uses, so a
                // reader sees what the application shows.
                for (int c = 0; c < 3; ++c)
                    d[c] = uint8_t(ToneCurve(HalfToFloat(s[c]) * gain) * 255.0f + 0.5f);
                d[3] = 255;
            }
        const std::string file = outdir + "/" + m.slug + ".png";
        std::string e;
        if (SavePng(file, png, &e)) std::printf("wrote %s\n", file.c_str());
        else                        std::fprintf(stderr, "%s: %s\n", file.c_str(), e.c_str());
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage:\n"
                     "  paper_data --table <raw>...\n"
                     "  paper_data --crops <raw> <x> <y> <size> [outdir] [stops]\n");
        return 2;
    }
    if (std::strcmp(argv[1], "--table") == 0) return DoTable(argc, argv);
    if (std::strcmp(argv[1], "--crops") == 0) return DoCrops(argc, argv);
    std::fprintf(stderr, "unknown mode %s\n", argv[1]);
    return 2;
}
