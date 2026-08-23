// Verifies wavelet_denoise: identity at zero strength, real noise reduction
// with strength, and edge preservation.
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <random>
#include <span>
#include <string>
#include <vector>
#include "../src/core/image.h"
#include "../src/core/data.h"
#include "../src/core/algorithm.h"
#include "../src/script/value.h"
using namespace tglab;

static int g_fail = 0;
static void Check(bool ok, const std::string& what) {
    std::printf("%s  %s\n", ok ? "[ ok ]" : "[FAIL]", what.c_str());
    if (!ok) ++g_fail;
}

// A step edge plus gaussian noise: the two things a denoiser must tell apart.
static Image MakeNoisy(int dim, double sigma, unsigned seed) {
    Image img; img.Alloc({dim, dim, Format::RGBA32F});
    ImageView v = img.MapCpuWrite();
    std::mt19937 rng(seed);
    std::normal_distribution<double> n(0.0, sigma);
    for (int y = 0; y < dim; ++y)
        for (int x = 0; x < dim; ++x) {
            const double base = (x < dim/2) ? 0.25 : 0.75;
            float* p = v.At<float>(x, y);
            for (int c = 0; c < 3; ++c) p[c] = float(base + n(rng));
            p[3] = 1.0f;
        }
    return img;
}

static Image Run(const Image& src, double luma, double chroma, int levels) {
    auto algo = Registry::Get().Create("wavelet_denoise");
    std::string e;
    algo->FindParam("luma")->SetFromScript(Value(luma), &e);
    algo->FindParam("chroma")->SetFromScript(Value(chroma), &e);
    algo->FindParam("levels")->SetFromScript(Value(double(levels)), &e);
    Data dIn{const_cast<Image&>(src).Clone()};
    Data dOut{Image{}};
    std::get<Image>(dOut).Alloc(src.Desc());
    const Data* p = &dIn;
    RunCtx ctx(std::span<const Data* const>(&p,1), std::span<Data>(&dOut,1));
    algo->RunCPU(ctx);
    return std::get<Image>(dOut).Clone();
}

// Noise measured away from the edge, so the step itself does not count as
// deviation.
static double FlatSigma(Image& img) {
    ImageView v = img.MapCpuRead();
    const int dim = v.desc.width;
    std::vector<double> s;
    for (int y = 8; y < dim-8; ++y)
        for (int x = 8; x < dim/2 - 8; ++x)
            s.push_back(double(*v.At<float>(x,y)));
    double m = 0; for (double d : s) m += d; m /= double(s.size());
    double var = 0; for (double d : s) var += (d-m)*(d-m);
    return std::sqrt(var / double(s.size()));
}

int main() {
    const int dim = 192;
    Image noisy = MakeNoisy(dim, 0.05, 1234);
    const double sigma0 = FlatSigma(noisy);
    std::printf("input noise sigma %.4f\n\n", sigma0);

    {
        // Zero strength must be an exact identity. A denoiser that alters the
        // image with its strength at zero has a bug in the transform itself,
        // and every other measurement here would be built on it.
        Image out = Run(noisy, 0.0, 0.0, 4);
        ImageView a = noisy.MapCpuRead(), b = out.MapCpuRead();
        double maxD = 0;
        for (int y=0;y<dim;++y) for (int x=0;x<dim;++x)
            for (int c=0;c<3;++c)
                maxD = std::max(maxD, std::fabs(double(a.At<float>(x,y)[c]) -
                                                double(b.At<float>(x,y)[c])));
        Check(maxD < 1e-4, "zero strength round-trips the a-trous transform "
                           "(max delta " + std::to_string(maxD) + ")");
    }
    {
        Image out = Run(noisy, 0.04, 0.10, 4);
        const double s = FlatSigma(out);
        std::printf("       denoised sigma %.4f (%.0f%% of input)\n", s, 100.0*s/sigma0);
        Check(s < sigma0 * 0.6, "denoising measurably reduces noise");
    }
    {
        // The edge must survive. A filter that removes noise by blurring
        // everything would pass the test above and be useless.
        Image out = Run(noisy, 0.04, 0.10, 4);
        ImageView v = out.MapCpuRead();
        double left = 0, right = 0; int n = 0;
        for (int y = 8; y < dim-8; ++y) {
            left  += double(*v.At<float>(dim/2 - 6, y));
            right += double(*v.At<float>(dim/2 + 6, y));
            ++n;
        }
        left /= n; right /= n;
        std::printf("       edge %.3f -> %.3f (input 0.25 -> 0.75)\n", left, right);
        Check(right - left > 0.40, "the step edge survives denoising");
    }
    std::printf("\n%s\n", g_fail ? "FAILURES PRESENT" : "all wavelet checks passed");
    return g_fail ? 1 : 0;
}
