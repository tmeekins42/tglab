#include "gpu_pyramid.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>

#include "../../core/image.h"
#include "../../gpu/compute.h"

namespace tglab {
namespace {

// The blur, one axis per dispatch.
//
// Both directions are the same kernel with an Axis flag rather than two shaders,
// because a separable blur IS one pass run twice -- writing it twice would let
// them drift, and the drift would be invisible (a slightly wrong blur still
// looks like a blur).
//
// The weights arrive as a texture, not root constants. b0 holds 32 uints total,
// and sigma is user-driven: a sigma of 4 already needs radius 12, so 25 weights,
// and the whole budget is gone by sigma 5. A texture has no such ceiling, so the
// shader works at any sigma the UI can produce.
//
// T0 is the source plane, T1 the weights (2r+1 texels wide, one row).
constexpr const char* kBlurHlsl = R"(
Texture2D<float4>   T0 : register(t0);
Texture2D<float4>   T1 : register(t1);
RWTexture2D<float4> U0 : register(u0);

// Width and Height are filled by Dispatch from the OUTPUT descriptor; only the
// fields after them come from the caller's constants list.
cbuffer Params : register(b0) {
    uint Width;
    uint Height;
    uint Radius;
    uint Axis;      // 0 = horizontal, 1 = vertical
};

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;

    int r = int(Radius);
    int2 step = (Axis == 0) ? int2(1, 0) : int2(0, 1);
    int2 hi   = int2(int(Width) - 1, int(Height) - 1);

    float acc = 0.0;
    for (int i = -r; i <= r; ++i) {
        // Clamp-to-edge on BOTH axes, matching Plane::At. Clamping only the
        // axis being walked would read out of bounds on the other one for a
        // dispatch whose tile overhangs the image.
        int2 p = clamp(int2(tid.xy) + step * i, int2(0, 0), hi);
        acc += T1[int2(i + r, 0)].x * T0[p].x;
    }
    U0[tid.xy] = float4(acc, acc, acc, 1.0);
}
)";

// 2x decimation by point sampling, matching Halve().
constexpr const char* kHalveHlsl = R"(
Texture2D<float4>   T0 : register(t0);
RWTexture2D<float4> U0 : register(u0);

cbuffer Params : register(b0) {
    uint Width;     // DESTINATION extent
    uint Height;
    uint SrcW;
    uint SrcH;
};

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    int2 p = clamp(int2(tid.xy) * 2, int2(0, 0), int2(int(SrcW) - 1, int(SrcH) - 1));
    float s = T0[p].x;
    U0[tid.xy] = float4(s, s, s, 1.0);
}
)";

// FAST-9 plus Harris, one dispatch, dense output.
//
// The early rejection is kept even though a GPU has no branch to save: the
// point here is not speed but that the two paths agree. A pixel where the
// compass test fails must produce 0 on both sides, and the only way to be sure
// of that is to run the same test in the same order.
//
// Harris is computed only where FAST fired, matching the CPU: the response of a
// non-corner is never looked at, so computing it anyway would be wasted work AND
// would let a difference hide in pixels the CPU never evaluates.
// Shared by both corner kernels: the bindings, the ring, and the FAST-9 test.
//
// One copy of FastCorner rather than one per kernel. ORB and BRISK use the
// SAME corner test -- BRISK only asks it repeatedly at different thresholds --
// so two transcriptions could drift apart, and a drift here would change which
// pixels each detector considers a corner at all.
constexpr const char* kFastCommonHlsl = R"(
Texture2D<float4>   T0 : register(t0);
RWTexture2D<float4> U0 : register(u0);

cbuffer Params : register(b0) {
    uint Width;
    uint Height;
    uint ThreshBits;    // asfloat -- FAST's intensity margin
    uint HarrisRadius;  // unused by the score kernel
    uint Border;
};

static const int2 kCircle[16] = {
    int2( 0, -3), int2( 1, -3), int2( 2, -2), int2( 3, -1),
    int2( 3,  0), int2( 3,  1), int2( 2,  2), int2( 1,  3),
    int2( 0,  3), int2(-1,  3), int2(-2,  2), int2(-3,  1),
    int2(-3,  0), int2(-3, -1), int2(-2, -2), int2(-1, -3),
};

float S(int x, int y) {
    int2 p = clamp(int2(x, y), int2(0, 0), int2(int(Width) - 1, int(Height) - 1));
    return T0[p].x;
}

bool FastCorner(int x, int y, float t) {
    float c  = S(x, y);
    float hi = c + t, lo = c - t;

    int brightAxis = 0, darkAxis = 0;
    for (int i = 0; i < 16; i += 4) {
        float p = S(x + kCircle[i].x, y + kCircle[i].y);
        if (p > hi) ++brightAxis;
        else if (p < lo) ++darkAxis;
    }
    if (brightAxis < 3 && darkAxis < 3) return false;

    int runBright = 0, runDark = 0;
    for (int j = 0; j < 32; ++j) {
        int k = j & 15;
        float p = S(x + kCircle[k].x, y + kCircle[k].y);
        if (p > hi) { runDark = 0; if (++runBright >= 9) return true; }
        else if (p < lo) { runBright = 0; if (++runDark >= 9) return true; }
        else { runBright = 0; runDark = 0; }
    }
    return false;
}
)";

// ORB: Harris response where FAST fired.
constexpr const char* kHarrisMainHlsl = R"(
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    int x = int(tid.x), y = int(tid.y);

    int b = int(Border);
    if (x < b || y < b || x >= int(Width) - b || y >= int(Height) - b) {
        U0[tid.xy] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    float t = asfloat(ThreshBits);
    if (!FastCorner(x, y, t)) {
        U0[tid.xy] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    int r = int(HarrisRadius);
    float a = 0.0, bb = 0.0, c = 0.0;
    for (int dy = -r; dy <= r; ++dy)
        for (int dx = -r; dx <= r; ++dx) {
            float gx = S(x + dx + 1, y + dy) - S(x + dx - 1, y + dy);
            float gy = S(x + dx, y + dy + 1) - S(x + dx, y + dy - 1);
            a  += gx * gx;
            bb += gy * gy;
            c  += gx * gy;
        }
    float det = a * bb - c * c;
    float tr  = a + bb;
    float resp = det - 0.04 * tr * tr;

    // The CPU keeps only responses > 0, so anything at or below it is not a
    // candidate. Writing 0 rather than the negative value keeps "0 means no
    // candidate" true for the reader without it needing the rule too.
    U0[tid.xy] = float4(max(resp, 0.0), 0.0, 0.0, 1.0);
}
)";

// FAST-9 with the bisected score. BRISK's inner loop.
//
// Shares kFastHarrisHlsl's ring walk by textual concatenation rather than by
// being a second entry point in one shader: the two kernels want different
// constants, and a shared cbuffer would make each carry the other's fields.
constexpr const char* kFastScoreHlsl = R"(
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    int x = int(tid.x), y = int(tid.y);

    int b = int(Border);
    if (x < b || y < b || x >= int(Width) - b || y >= int(Height) - b) {
        U0[tid.xy] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    float t = asfloat(ThreshBits);
    if (!FastCorner(x, y, t)) {
        U0[tid.xy] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // Bisection, exactly eight steps like the CPU. The count is part of the
    // result -- the score is wherever the search happens to have narrowed to --
    // so a "more accurate" extra step would produce a different number.
    float lo = t, hi = 1.0;
    for (int i = 0; i < 8; ++i) {
        float mid = 0.5 * (lo + hi);
        if (FastCorner(x, y, mid)) lo = mid;
        else                       hi = mid;
    }
    U0[tid.xy] = float4(lo, 0.0, 0.0, 1.0);
}
)";

// One explicit diffusion step. AKAZE's scale space.
//
// T0 is the plane being evolved, T1 the conductivity map. The border ring is
// COPIED rather than evolved, matching the CPU's `for y in 1..h-1` -- evolving
// it would need a boundary condition the CPU does not have, and the difference
// would show up as a bright edge that grows with every step.
constexpr const char* kDiffuseHlsl = R"(
Texture2D<float4>   T0 : register(t0);
Texture2D<float4>   T1 : register(t1);
RWTexture2D<float4> U0 : register(u0);

cbuffer Params : register(b0) {
    uint Width;
    uint Height;
    uint DtBits;
};

float P(int x, int y) {
    int2 p = clamp(int2(x, y), int2(0, 0), int2(int(Width) - 1, int(Height) - 1));
    return T0[p].x;
}
float C(int x, int y) {
    int2 p = clamp(int2(x, y), int2(0, 0), int2(int(Width) - 1, int(Height) - 1));
    return T1[p].x;
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    int x = int(tid.x), y = int(tid.y);

    float here = P(x, y);
    if (x < 1 || y < 1 || x >= int(Width) - 1 || y >= int(Height) - 1) {
        U0[tid.xy] = float4(here, here, here, 1.0);
        return;
    }

    float dt = asfloat(DtBits);
    float c  = C(x, y);

    // Conductivity between two pixels as the average of theirs, keeping the
    // scheme symmetric -- an asymmetric weight would let brightness leak in one
    // direction and drift the whole image.
    float fl = 0.5 * (c + C(x - 1, y)) * (P(x - 1, y) - here);
    float fr = 0.5 * (c + C(x + 1, y)) * (P(x + 1, y) - here);
    float fu = 0.5 * (c + C(x, y - 1)) * (P(x, y - 1) - here);
    float fd = 0.5 * (c + C(x, y + 1)) * (P(x, y + 1) - here);

    float v = here + dt * (fl + fr + fu + fd);
    U0[tid.xy] = float4(v, v, v, 1.0);
}
)";

ImageDesc PlaneDesc(int w, int h) {
    ImageDesc d;
    d.width  = w;
    d.height = h;
    d.format = Format::R32F;
    return d;
}

// A plane's pixels as an ImageView, for Upload/Readback.
//
// R32F is one tightly packed float per pixel, so the vector's storage IS the
// texture's, and no staging copy is needed.
ImageView ViewOf(GpuPlane& p) {
    ImageView v;
    v.data = reinterpret_cast<uint8_t*>(p.v.data());
    v.desc = PlaneDesc(p.w, p.h);
    return v;
}

// Compiled once and reused. Building a kernel means invoking DXC, which is
// milliseconds -- per octave per level that would dwarf the blur it enables.
struct Kernels {
    ComputeKernel blur, halve;
    bool ready = false;

    bool Ensure(ComputeContext* gpu, std::string* err) {
        if (ready) return true;
        if (!gpu->CreateKernel(kBlurHlsl, "main", "sift_blur", &blur, err))
            return false;
        if (!gpu->CreateKernel(kHalveHlsl, "main", "sift_halve", &halve, err))
            return false;
        ready = true;
        return true;
    }
};

// Separate from Kernels so a detector that never calls it does not pay DXC for
// it -- compiling a shader is milliseconds, and SIFT has no use for this one.
struct CornerKernel {
    ComputeKernel k;
    bool ready = false;

    bool Ensure(ComputeContext* gpu, const char* body, const char* name,
                std::string* err) {
        if (ready) return true;
        const std::string src = std::string(kFastCommonHlsl) + body;
        if (!gpu->CreateKernel(src, "main", name, &k, err)) return false;
        ready = true;
        return true;
    }
};

struct DiffuseKernel {
    ComputeKernel k;
    bool ready = false;

    bool Ensure(ComputeContext* gpu, std::string* err) {
        if (ready) return true;
        if (!gpu->CreateKernel(kDiffuseHlsl, "main", "akaze_diffuse", &k, err))
            return false;
        ready = true;
        return true;
    }
};

DiffuseKernel& TheDiffuseKernel() {
    static DiffuseKernel k;
    return k;
}

CornerKernel& TheHarrisKernel() {
    static CornerKernel k;
    return k;
}

CornerKernel& TheScoreKernel() {
    static CornerKernel k;
    return k;
}

// Uploads a plane, runs one corner kernel over it, and reads the map back.
//
// Both corner entry points do exactly this and differ only in which kernel and
// which constants, so the plumbing lives here -- a second copy would be a
// second place for the upload/dispatch/readback order to go wrong.
bool RunCornerKernel(ComputeContext* gpu, CornerKernel& ck, const GpuPlane& src,
                     GpuPlane* dst, const std::vector<uint32_t>& constants,
                     std::string* err) {
    const int w = src.w, h = src.h;
    const ImageDesc desc = PlaneDesc(w, h);

    GpuImage in{}, out{};
    if (!gpu->CreateImage(desc, &in) || !gpu->CreateImage(desc, &out)) {
        *err = "could not allocate the corner map";
        return false;
    }

    GpuPlane seed = src;
    if (!gpu->Upload(ViewOf(seed), &in)) {
        *err = "could not upload the source plane";
        return false;
    }

    if (!gpu->Dispatch(ck.k, {&in}, {&out}, constants, err)) return false;

    dst->w = w;
    dst->h = h;
    dst->v.assign(size_t(w) * size_t(h), 0.0f);
    ImageView dv = ViewOf(*dst);
    if (!gpu->Readback(out, &dv)) {
        *err = "could not read back the corner map";
        return false;
    }
    return true;
}

// Shared preconditions for the corner entry points.
bool CornerReady(ComputeContext* gpu, const GpuPlane& src, std::string* err) {
    if (!gpu || !gpu->Ready()) { *err = "no compute device"; return false; }
    if (src.w <= 0 || src.h <= 0 ||
        src.v.size() != size_t(src.w) * size_t(src.h)) {
        *err = "malformed source plane";
        return false;
    }
    return true;
}

uint32_t Bits(float f) {
    uint32_t b = 0;
    std::memcpy(&b, &f, sizeof b);
    return b;
}

Kernels& TheKernels() {
    static Kernels k;
    return k;
}

// Uploads the 1-D kernel as a 1-row texture.
bool UploadWeights(ComputeContext* gpu, const std::vector<float>& k,
                   GpuImage* out, std::string* err) {
    if (!gpu->CreateImage(PlaneDesc(int(k.size()), 1), out)) {
        *err = "could not allocate the blur weights";
        return false;
    }
    std::vector<float> row = k;
    ImageView v;
    v.data = reinterpret_cast<uint8_t*>(row.data());
    v.desc = PlaneDesc(int(row.size()), 1);
    if (!gpu->Upload(v, out)) {
        *err = "could not upload the blur weights";
        return false;
    }
    return true;
}

// One separable blur, source and destination already on the device.
// `scratch` holds the horizontal pass and must match src's extent.
bool BlurOnDevice(ComputeContext* gpu, Kernels& kern, const GpuImage& src,
                  GpuImage* scratch, GpuImage* dst, const GpuImage& weights,
                  int w, int h, int r, std::string* err) {
    (void)w; (void)h;
    // Only the EXTRA constants. Dispatch fills b0's first two slots with the
    // output's width and height itself, so passing them again would shift
    // everything after them by two -- and the shader would read the image width
    // as its radius, which is a plausible-looking blur rather than a crash.
    const uint32_t ur = uint32_t(r);
    if (!gpu->Dispatch(kern.blur, {&src, &weights}, {scratch}, {ur, 0u}, err))
        return false;
    return gpu->Dispatch(kern.blur, {scratch, &weights}, {dst}, {ur, 1u}, err);
}

}  // namespace

std::vector<float> GaussianKernel(float sigma, int* radiusOut) {
    const int r = std::max(1, int(std::ceil(sigma * 3.0f)));
    if (radiusOut) *radiusOut = r;

    std::vector<float> k(size_t(r) * 2 + 1);
    float sum = 0.0f;
    for (int i = -r; i <= r; ++i) {
        const float e = std::exp(-float(i * i) / (2.0f * sigma * sigma));
        k[size_t(i + r)] = e;
        sum += e;
    }
    for (float& x : k) x /= sum;
    return k;
}

bool GpuBlur(ComputeContext* gpu, const GpuPlane& src, GpuPlane* dst,
             float sigma, std::string* err) {
    std::vector<float> sig{sigma};
    std::vector<GpuPlane> out;
    if (!GpuBlurStack(gpu, src, sig, &out, err)) return false;
    if (out.size() != 2) { *err = "blur stack returned the wrong depth"; return false; }
    *dst = std::move(out[1]);
    return true;
}

bool GpuFastHarris(ComputeContext* gpu, const GpuPlane& src, GpuPlane* dst,
                   float threshold, int harrisRadius, int border,
                   std::string* err) {
    if (!CornerReady(gpu, src, err)) return false;

    CornerKernel& ck = TheHarrisKernel();
    if (!ck.Ensure(gpu, kHarrisMainHlsl, "fast_harris", err)) return false;

    return RunCornerKernel(
        gpu, ck, src, dst,
        {Bits(threshold), uint32_t(harrisRadius), uint32_t(border)}, err);
}

bool GpuFastScore(ComputeContext* gpu, const GpuPlane& src, GpuPlane* dst,
                  float threshold, int border, std::string* err) {
    if (!CornerReady(gpu, src, err)) return false;

    CornerKernel& ck = TheScoreKernel();
    if (!ck.Ensure(gpu, kFastScoreHlsl, "fast_score", err)) return false;

    // HarrisRadius is unused here but still occupies its slot: the two kernels
    // share one cbuffer layout, so Border must land where the shader expects it.
    return RunCornerKernel(gpu, ck, src, dst,
                           {Bits(threshold), 0u, uint32_t(border)}, err);
}

bool GpuDiffuse(ComputeContext* gpu, const GpuPlane& src, const GpuPlane& cond,
                int steps, float dt, GpuPlane* dst, std::string* err) {
    if (!CornerReady(gpu, src, err)) return false;
    if (cond.w != src.w || cond.h != src.h ||
        cond.v.size() != src.v.size()) {
        *err = "conductivity map does not match the plane";
        return false;
    }
    if (steps <= 0) { *dst = src; return true; }

    DiffuseKernel& dk = TheDiffuseKernel();
    if (!dk.Ensure(gpu, err)) return false;

    const int w = src.w, h = src.h;
    const ImageDesc desc = PlaneDesc(w, h);

    // Ping-pong between two planes so the chain never leaves the device. The
    // conductivity map is uploaded once: it is constant across the whole chain.
    GpuImage a{}, b{}, cmap{};
    if (!gpu->CreateImage(desc, &a) || !gpu->CreateImage(desc, &b) ||
        !gpu->CreateImage(desc, &cmap)) {
        *err = "could not allocate the diffusion planes";
        return false;
    }

    GpuPlane seed = src, cseed = cond;
    if (!gpu->Upload(ViewOf(seed), &a) || !gpu->Upload(ViewOf(cseed), &cmap)) {
        *err = "could not upload the diffusion inputs";
        return false;
    }

    GpuImage* cur = &a;
    GpuImage* nxt = &b;
    for (int i = 0; i < steps; ++i) {
        if (!gpu->Dispatch(dk.k, {cur, &cmap}, {nxt}, {Bits(dt)}, err))
            return false;
        std::swap(cur, nxt);
    }

    dst->w = w;
    dst->h = h;
    dst->v.assign(size_t(w) * size_t(h), 0.0f);
    ImageView dv = ViewOf(*dst);
    if (!gpu->Readback(*cur, &dv)) {
        *err = "could not read back the diffused plane";
        return false;
    }
    return true;
}

bool GpuBlurStack(ComputeContext* gpu, const GpuPlane& base,
                  const std::vector<float>& sigmas,
                  std::vector<GpuPlane>* out, std::string* err) {
    if (!gpu || !gpu->Ready()) { *err = "no compute device"; return false; }
    if (base.w <= 0 || base.h <= 0 ||
        base.v.size() != size_t(base.w) * size_t(base.h)) {
        *err = "malformed base plane";
        return false;
    }

    Kernels& kern = TheKernels();
    if (!kern.Ensure(gpu, err)) return false;

    const int w = base.w, h = base.h;
    const ImageDesc desc = PlaneDesc(w, h);

    // Three device planes: the level being read, the level being written, and
    // the horizontal pass's intermediate. Ping-ponging between two means the
    // chain never leaves the GPU, which is the whole point of doing the stack
    // in one call.
    GpuImage a{}, b{}, scratch{}, weights{};
    if (!gpu->CreateImage(desc, &a) || !gpu->CreateImage(desc, &b) ||
        !gpu->CreateImage(desc, &scratch)) {
        *err = "could not allocate the blur pyramid";
        return false;
    }

    GpuPlane seed = base;   // Upload needs a non-const view
    if (!gpu->Upload(ViewOf(seed), &a)) {
        *err = "could not upload the base plane";
        return false;
    }

    std::vector<GpuPlane> result;
    result.reserve(sigmas.size() + 1);
    result.push_back(base);

    GpuImage* cur = &a;
    GpuImage* nxt = &b;
    for (float sigma : sigmas) {
        int r = 0;
        const std::vector<float> k = GaussianKernel(sigma, &r);

        // Fresh weights per level: the radius changes with sigma, so the
        // texture's width does too.
        GpuImage wtex{};
        if (!UploadWeights(gpu, k, &wtex, err)) return false;

        if (!BlurOnDevice(gpu, kern, *cur, &scratch, nxt, wtex, w, h, r, err))
            return false;

        GpuPlane p;
        p.w = w;
        p.h = h;
        p.v.assign(size_t(w) * size_t(h), 0.0f);
        ImageView pv = ViewOf(p);
        if (!gpu->Readback(*nxt, &pv)) {
            *err = "could not read back a blurred plane";
            return false;
        }
        result.push_back(std::move(p));

        std::swap(cur, nxt);
    }

    *out = std::move(result);
    return true;
}

}  // namespace tglab
