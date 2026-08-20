// anisotropic_diffusion — Perona-Malik.
//
// Blurring is heat diffusion: run the heat equation on an image and it becomes
// a Gaussian blur, with the variance set by how long it runs. Perona and Malik's
// insight (1990) was to make the conductance depend on the local gradient, so
// heat flows freely inside smooth regions and barely at all across edges:
//
//   dI/dt = div( c(|grad I|) * grad I )
//
// Discretised on the 4-neighbourhood, each iteration is:
//
//   I += lambda * sum_over_neighbours( c(|dI|) * dI )
//
// Two conductance functions from the original paper, selectable because they
// behave differently and the choice is exactly the kind of thing this lab is
// for. Both are 1 at zero gradient and fall towards 0 as the gradient grows:
//
//   exponential  c = exp(-(|dI| / k)^2)   favours high-contrast edges
//   quadratic    c = 1 / (1 + (|dI| / k)^2)   favours wide regions
//
// Unlike a Gaussian, this is iterative and has no closed form -- the result
// depends on iterations *and* lambda together, which is why both are exposed.
//
// lambda must stay <= 0.25 for the 4-neighbour scheme to remain stable; above
// that the explicit update diverges and the image blows up. The range is
// clamped accordingly rather than letting a slider produce garbage.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"

namespace tglab {

class AnisotropicDiffusion : public AlgorithmBase {
public:
    const char* Name()     const override { return "anisotropic_diffusion"; }
    const char* Category() const override { return "filter"; }

    PortList Inputs()  const override { return {{"src", DataType::Image, FormatSpec::Any}}; }
    PortList Outputs() const override { return {{"out", DataType::Image, FormatSpec::SameAsInput}}; }

    void RunCPU(RunCtx& ctx) override {
        const ImageView src = ctx.In(0);
        ImageView       dst = ctx.Out(0);
        if (!src.Valid() || !dst.Valid()) return;

        m_a.Unpack(src);
        if (!m_a.Valid()) return;
        m_b.AllocLike(m_a);

        const int   iterations = std::max(1, int(m_iterations));
        const float lambda     = std::clamp(float(m_lambda), 0.0f, 0.25f);
        // k is declared as a fraction of the intensity range so it means the
        // same thing for 8-bit and float sources.
        const float k          = std::max(1e-4f, float(m_k)) * m_a.ValueScale();
        const bool  exponential = m_exponential;

        const int w = m_a.Width(), h = m_a.Height(), ch = m_a.Channels();
        const int filtered = (ch == 1) ? 1 : 3;
        const float invK2 = 1.0f / (k * k);

        PixelBuffer* cur  = &m_a;
        PixelBuffer* next = &m_b;

        for (int it = 0; it < iterations; ++it) {
            // Per iteration rather than per row: one iteration is a cheap pass,
            // and it is the count that makes this slow.
            if (ctx.Cancelled()) return;
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    const float* c0 = cur->At(x, y);

                    // The 4-neighbourhood, edge-clamped: a border pixel simply
                    // sees itself across the boundary, so nothing flows in.
                    const float* nN = cur->AtClamped(x, y - 1);
                    const float* nS = cur->AtClamped(x, y + 1);
                    const float* nE = cur->AtClamped(x + 1, y);
                    const float* nW = cur->AtClamped(x - 1, y);

                    for (int c = 0; c < filtered; ++c) {
                        const float dN = nN[c] - c0[c];
                        const float dS = nS[c] - c0[c];
                        const float dE = nE[c] - c0[c];
                        const float dW = nW[c] - c0[c];

                        const float flow =
                            Conduct(dN, invK2, exponential) * dN +
                            Conduct(dS, invK2, exponential) * dS +
                            Conduct(dE, invK2, exponential) * dE +
                            Conduct(dW, invK2, exponential) * dW;

                        next->Set(x, y, c, c0[c] + lambda * flow);
                    }
                    if (ch == 4) next->Set(x, y, 3, cur->Get(x, y, 3));
                }
            }
            std::swap(cur, next);
        }

        cur->PackInto(dst);
    }

    // --- GPU implementation -------------------------------------------------
    // The natural fit for a compute shader: each pass is a pure local stencil,
    // and the framework ping-pongs between two images so a pass always reads
    // the completed previous one. Iterations that cost a second on the CPU run
    // in a few milliseconds here, which is what makes the sliders usable.
    bool HasGPU() const override { return true; }
    int  GpuIterations() const override { return std::max(1, int(m_iterations)); }

    const char* GpuSource() const override {
        return R"(
Texture2D<float4>   Src : register(t0);
RWTexture2D<float4> Dst : register(u0);

cbuffer Params : register(b0) {
    uint Width;
    uint Height;
    uint LambdaBits;
    uint InvK2Bits;
    uint Exponential;
};

float Conduct(float d, float invK2, bool expo) {
    float r2 = d * d * invK2;
    return expo ? exp(-r2) : 1.0 / (1.0 + r2);
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;

    float lambda = asfloat(LambdaBits);
    float invK2  = asfloat(InvK2Bits);
    bool  expo   = (Exponential != 0);

    int2 hi = int2(Width - 1, Height - 1);
    int2 p  = int2(tid.xy);

    float4 c = Src[p];
    // Edge-clamped 4-neighbourhood: a border pixel sees itself across the
    // boundary, so nothing flows in or out there.
    float4 nN = Src[clamp(p + int2( 0, -1), int2(0, 0), hi)];
    float4 nS = Src[clamp(p + int2( 0,  1), int2(0, 0), hi)];
    float4 nE = Src[clamp(p + int2( 1,  0), int2(0, 0), hi)];
    float4 nW = Src[clamp(p + int2(-1,  0), int2(0, 0), hi)];

    float4 result = c;
    // RGB only; alpha is carried through untouched, matching the CPU path.
    for (int ch = 0; ch < 3; ++ch) {
        float dN = nN[ch] - c[ch];
        float dS = nS[ch] - c[ch];
        float dE = nE[ch] - c[ch];
        float dW = nW[ch] - c[ch];

        float flow = Conduct(dN, invK2, expo) * dN +
                     Conduct(dS, invK2, expo) * dS +
                     Conduct(dE, invK2, expo) * dE +
                     Conduct(dW, invK2, expo) * dW;

        result[ch] = c[ch] + lambda * flow;
    }
    Dst[tid.xy] = result;
}
)";
    }

    std::vector<uint32_t> GpuConstants(int) const override {
        const float lambda = std::clamp(float(m_lambda), 0.0f, 0.25f);
        // GPU textures are normalised 0..1, so k -- declared as a fraction of
        // the intensity range -- needs no rescaling here, unlike the CPU path
        // where the buffer carries 0..255 values.
        const float k     = std::max(1e-4f, float(m_k));
        const float invK2 = 1.0f / (k * k);

        uint32_t lb, ib;
        std::memcpy(&lb, &lambda, sizeof(lb));
        std::memcpy(&ib, &invK2,  sizeof(ib));
        return {lb, ib, uint32_t(m_exponential ? 1 : 0)};
    }

private:
    static float Conduct(float d, float invK2, bool exponential) {
        const float r2 = d * d * invK2;
        return exponential ? std::exp(-r2) : 1.0f / (1.0f + r2);
    }

    Param<int> m_iterations{
        this, "iterations", 10, 1, 200,
        {.help = "How many diffusion steps to run. Higher smooths more; "
                 "cost is linear in this. Together with lambda it sets the "
                 "total amount of blurring.",
         .softMin = 1, .softMax = 50}};
    // Above 0.25 the explicit 4-neighbour scheme is unstable.
    Param<float> m_lambda{
        this, "lambda", 0.20f, 0.01f, 0.25f,
        {.help = "Step size per iteration. Higher smooths faster per step. "
                 "Capped at 0.25: above that the scheme is numerically "
                 "unstable and the image diverges.",
         .step = 0.01}};
    // Gradient scale, as a fraction of the intensity range: edges stronger than
    // this survive, weaker ones diffuse away.
    Param<float> m_k{
        this, "k", 0.05f, 0.001f, 1.0f,
        {.help = "Edge threshold, as a fraction of the intensity range. "
                 "Gradients above k are kept, below it are smoothed away. "
                 "Note this runs OPPOSITE to a blur radius: LOWER k preserves "
                 "more, so small k keeps even impulse noise as 'edges'.",
         .step = 0.005, .softMin = 0.005, .softMax = 0.3}};
    Param<bool> m_exponential{
        this, "exponential", true,
        "Conductance function. On: exp(-(g/k)^2), which favours keeping "
        "high-contrast edges. Off: 1/(1+(g/k)^2), which favours keeping "
        "wide regions."};

    PixelBuffer m_a, m_b;
};

REGISTER_ALGORITHM(AnisotropicDiffusion);

} // namespace tglab
