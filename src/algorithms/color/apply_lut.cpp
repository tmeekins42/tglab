// apply_lut — run the image through a 3D colour lookup table.
//
// A .cube file is a cube of RGB triples on a regular grid: 35,937 of them at
// the usual 33^3. Each says "input colour here comes out as THIS colour", and
// everything between grid points is interpolated. That is the whole mechanism,
// and it is why a LUT is the standard way to ship a film emulation or a
// creative grade -- it can express any colour mapping at all, including ones no
// set of sliders could describe.
//
// It is also why a LUT cannot be turned back into sliders. Contrast is not a
// number inside the file; it is a pattern distributed across all 35,937
// entries, as are saturation, colour casts, and split-toning. Fitting a
// ten-parameter model to a 35,937-entry table finds the nearest approximation
// and silently discards whatever it could not express -- which on a film LUT is
// most of what makes it that film.
//
// A LUT IS DISPLAY-REFERRED, AND THIS PIPELINE IS NOT, which is the one thing
// worth understanding before using this stage. The table's domain is almost
// always 0..1, so it says nothing about a scene-linear highlight sitting at
// 6.0. Sampling clamps -- see Lut3D::Sample, which clamps rather than
// extrapolating, because extrapolating from the edge cells produces wild
// colours on exactly those highlights.
//
// So: TONE MAP FIRST. `orton`, `bloom` and the rest are happy in scene-linear,
// but a LUT belongs after `tonemap`, once the data is in the range the table
// was authored for. Rather than silently clipping, this stage reports how much
// of the image was out of domain in its status line, so a pipeline in the wrong
// order says so instead of merely looking wrong.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"
#include "../../core/lut.h"

namespace tglab {

class ApplyLut : public AlgorithmBase {
public:
    const char* Name()     const override { return "apply_lut"; }
    const char* Category() const override { return "color"; }

    PortList Inputs()  const override { return {{"src", DataType::Image, FormatSpec::Any}}; }
    PortList Outputs() const override { return {{"out", DataType::Image, FormatSpec::SameAsInput}}; }

    void RunCPU(RunCtx& ctx) override {
        const ImageView src = ctx.In(0);
        ImageView       dst = ctx.Out(0);
        if (!src.Valid() || !dst.Valid()) return;

        m_in.Unpack(src);
        if (!m_in.Valid()) return;
        m_out.AllocLike(m_in);

        const int w = m_in.Width(), h = m_in.Height(), ch = m_in.Channels();
        if (w <= 0 || h <= 0) return;

        EnsureLoaded();

        if (!m_lut.Valid()) {
            // No table, or it failed to load: pass through rather than
            // returning a black frame, and say why in the status line.
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x)
                    for (int c = 0; c < ch; ++c)
                        m_out.Set(x, y, c, m_in.Get(x, y, c));
            m_out.PackInto(dst);
            return;
        }

        const float white    = m_in.ValueScale();
        const float strength = float(m_strength);
        const int colours    = (ch == 4) ? 3 : ch;

        long long outOfDomain = 0;
        const float domHi = m_lut.DomainMax()[0];

        for (int y = 0; y < h; ++y) {
            if (ctx.Cancelled()) return;
            for (int x = 0; x < w; ++x) {
                // The table is indexed in its own domain, so the image's units
                // have to be normalised into it -- 0..255 on an RGBA8 image is
                // 0..1 to the LUT.
                float in[3];
                for (int c = 0; c < 3; ++c)
                    in[c] = (c < colours ? m_in.Get(x, y, c) : m_in.Get(x, y, 0)) / white;

                if (in[0] > domHi || in[1] > domHi || in[2] > domHi) ++outOfDomain;

                float outv[3];
                m_lut.Sample(in[0], in[1], in[2], outv);

                for (int c = 0; c < colours; ++c) {
                    const float graded = outv[c] * white;
                    const float orig   = m_in.Get(x, y, c);
                    m_out.Set(x, y, c, orig + (graded - orig) * strength);
                }
                if (ch == 4) m_out.Set(x, y, 3, m_in.Get(x, y, 3));
            }
        }

        m_outOfDomain = double(outOfDomain) / (double(w) * double(h));
        m_out.PackInto(dst);
    }

    bool IsNoOp() const override { return float(m_strength) == 0.0f; }

    // --- GPU implementation -------------------------------------------------
    //
    // The table rides in as a stage-owned texture rather than a constant: a
    // 33^3 LUT is 431 KB against the constant buffer's 64 KB limit, and
    // shrinking it to the 17^3 that would fit throws away most of what makes a
    // film LUT that film.
    //
    // PACKED AS 2D, NOT 3D. The GPU layer binds every SRV as a Texture2D, so
    // rather than widen it the cube is laid out as `size` wide by `size*size`
    // tall: slice b occupies rows b*size .. b*size+size-1. Addressing is two
    // multiplies, and it keeps this change to one hook.
    //
    // A 3D texture would also have brought hardware TRILINEAR filtering, which
    // sounds like a bonus and is not: the CPU path is TETRAHEDRAL, so the two
    // would visibly disagree on any LUT with a sharp transition and the audit
    // would rightly flag it. Doing tetrahedral by hand here is both the same
    // maths as the CPU and the maths Resolve uses, so a LUT looks the same in
    // all three.
    bool HasGPU() const override { return true; }

    std::vector<const Image*> GpuExtraInputs() const override {
        if (!m_table.Valid()) return {};
        return {&m_table};
    }

    const char* GpuSource() const override {
        return R"(
Texture2D<float4>   Src   : register(t0);
Texture2D<float4>   Table : register(t1);
RWTexture2D<float4> Dst   : register(u0);

cbuffer Params : register(b0) {
    uint Width;
    uint Height;
    uint SizeBits;        // LUT edge length, as a float
    uint StrengthBits;
    uint WhiteBits;       // the image's own white, so 0..255 data normalises
    uint DomLoBits;
    uint DomHiBits;
};

float3 Fetch(int ri, int gi, int bi, int n) {
    // Blue slowest, matching .cube's own ordering and the CPU path's At().
    return Table.Load(int3(ri, bi * n + gi, 0)).rgb;
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;

    int   n        = int(asfloat(SizeBits));
    float strength = asfloat(StrengthBits);
    float white    = asfloat(WhiteBits);
    float domLo    = asfloat(DomLoBits);
    float domHi    = asfloat(DomHiBits);

    float4 c = Src[int2(tid.xy)];
    float3 v = c.rgb / white;

    // Clamped, not extrapolated: a table says nothing about colours outside
    // its domain, and extrapolating from the edge cells goes wild on exactly
    // the scene-linear highlights this is most likely to meet.
    float3 t = saturate((v - domLo) / max(domHi - domLo, 1e-6)) * float(n - 1);

    int3   i0 = min(int3(t), int3(n - 2, n - 2, n - 2));
    float3 d  = t - float3(i0);

    // Tetrahedral: only the four corners of the containing tetrahedron
    // contribute, so a sharp transition in the table stays sharp. Which four,
    // and their weights, follow from the ordering of d.r/d.g/d.b -- kept in
    // step with Lut3D::Sample.
    float3 c000 = Fetch(i0.x,     i0.y,     i0.z,     n);
    float3 c111 = Fetch(i0.x + 1, i0.y + 1, i0.z + 1, n);
    float3 outv;

    if (d.r > d.g) {
        if (d.g > d.b) {
            float3 c100 = Fetch(i0.x + 1, i0.y,     i0.z,     n);
            float3 c110 = Fetch(i0.x + 1, i0.y + 1, i0.z,     n);
            outv = c000 + (c100 - c000) * d.r + (c110 - c100) * d.g + (c111 - c110) * d.b;
        } else if (d.r > d.b) {
            float3 c100 = Fetch(i0.x + 1, i0.y,     i0.z,     n);
            float3 c101 = Fetch(i0.x + 1, i0.y,     i0.z + 1, n);
            outv = c000 + (c100 - c000) * d.r + (c101 - c100) * d.b + (c111 - c101) * d.g;
        } else {
            float3 c001 = Fetch(i0.x,     i0.y,     i0.z + 1, n);
            float3 c101 = Fetch(i0.x + 1, i0.y,     i0.z + 1, n);
            outv = c000 + (c001 - c000) * d.b + (c101 - c001) * d.r + (c111 - c101) * d.g;
        }
    } else {
        if (d.b > d.g) {
            float3 c001 = Fetch(i0.x,     i0.y,     i0.z + 1, n);
            float3 c011 = Fetch(i0.x,     i0.y + 1, i0.z + 1, n);
            outv = c000 + (c001 - c000) * d.b + (c011 - c001) * d.g + (c111 - c011) * d.r;
        } else if (d.b > d.r) {
            float3 c010 = Fetch(i0.x,     i0.y + 1, i0.z,     n);
            float3 c011 = Fetch(i0.x,     i0.y + 1, i0.z + 1, n);
            outv = c000 + (c010 - c000) * d.g + (c011 - c010) * d.b + (c111 - c011) * d.r;
        } else {
            float3 c010 = Fetch(i0.x,     i0.y + 1, i0.z,     n);
            float3 c110 = Fetch(i0.x + 1, i0.y + 1, i0.z,     n);
            outv = c000 + (c010 - c000) * d.g + (c110 - c010) * d.r + (c111 - c110) * d.b;
        }
    }

    float3 graded = outv * white;
    Dst[tid.xy] = float4(lerp(c.rgb, graded, strength), c.a);
}
)";
    }

    // The GPU path never calls RunCPU, so the table has to be loaded and packed
    // here or the shader would bind a stale texture.
    void PrepareGpu(const std::vector<ImageDesc>& inputs) override {
        EnsureLoaded();
        m_gpuWhite = inputs.empty() ? 1.0f
                   : (inputs[0].format == Format::RGBA8 ? 255.0f : 1.0f);
    }

    std::vector<uint32_t> GpuConstants(int) const override {
        auto bits = [](float f) { uint32_t u; std::memcpy(&u, &f, sizeof u); return u; };
        // A UNORM SRV hands the shader 0..1 whatever the storage format, so the
        // shader's own white is always 1.0 -- unlike the CPU path, which works
        // in the image's own units. Matching brightness.cpp's note.
        return {bits(float(m_lut.Size())),
                bits(float(m_strength)),
                bits(1.0f),
                bits(m_lut.Valid() ? m_lut.DomainMin()[0] : 0.0f),
                bits(m_lut.Valid() ? m_lut.DomainMax()[0] : 1.0f)};
    }

    // The status line is where a mis-ordered pipeline announces itself: a LUT
    // applied to scene-linear data clamps most of the frame, and a percentage
    // says so far more clearly than the image does.
    std::string RunReport() const override {
        // Strength 0 makes this a no-op, so the stage is skipped entirely and
        // the table is never loaded. Saying "not loaded" there would read as a
        // failure; say what actually happened.
        if (float(m_strength) == 0.0f) return "strength 0 -- stage skipped";
        if (!m_loadError.empty()) return "LUT: " + m_loadError;
        if (m_path.get().empty()) return "no LUT set -- passing through";
        if (!m_lut.Valid()) return "LUT not loaded -- passing through";

        char buf[256];
        const std::string& title = m_lut.Title();
        std::snprintf(buf, sizeof buf, "%s%d^3%s%s",
                      m_lut.Is1D() ? "1D LUT, " : "",
                      m_lut.Size(),
                      title.empty() ? "" : " \"",
                      title.empty() ? "" : title.c_str());
        std::string s = buf;
        if (!title.empty()) s += "\"";
        if (m_outOfDomain > 0.005) {
            std::snprintf(buf, sizeof buf,
                          " -- %.0f%% of pixels are above the table's domain and "
                          "were clamped; tonemap before this stage",
                          m_outOfDomain * 100.0);
            s += buf;
        }
        return s;
    }

private:
    // Load only when the path changes: parsing 35,937 entries per slider drag
    // would dominate the run, and on the GPU path it would also mean
    // re-uploading the table every frame.
    void EnsureLoaded() {
        const std::string& path = m_path.get();
        if (path == m_loadedPath) return;

        m_loadError.clear();
        m_lut = Lut3D{};
        if (!path.empty() && !m_lut.Load(path, &m_loadError)) m_lut = Lut3D{};
        m_loadedPath = path;
        PackTable();
    }

    // The cube as a 2D texture: `size` wide by `size*size` tall, slice b on
    // rows b*size .. b*size+size-1. See the note above GpuSource for why this
    // is 2D rather than a 3D texture.
    void PackTable() {
        m_table = Image{};
        if (!m_lut.Valid() || m_lut.Is1D()) return;

        const int n = m_lut.Size();
        ImageDesc d{n, n * n, Format::RGBA32F};
        d.linear = true;
        m_table.Alloc(d);

        ImageView v = m_table.MapCpuWrite();
        float rgb[3];
        for (int b = 0; b < n; ++b)
            for (int g = 0; g < n; ++g)
                for (int r = 0; r < n; ++r) {
                    // Sampled at the grid point, which returns the stored entry
                    // exactly -- rather than reaching into Lut3D's storage and
                    // duplicating its layout here.
                    const float s = float(n - 1);
                    const float lo0 = m_lut.DomainMin()[0], hi0 = m_lut.DomainMax()[0];
                    const float lo1 = m_lut.DomainMin()[1], hi1 = m_lut.DomainMax()[1];
                    const float lo2 = m_lut.DomainMin()[2], hi2 = m_lut.DomainMax()[2];
                    m_lut.Sample(lo0 + (hi0 - lo0) * float(r) / s,
                                 lo1 + (hi1 - lo1) * float(g) / s,
                                 lo2 + (hi2 - lo2) * float(b) / s, rgb);
                    float* p = v.At<float>(r, b * n + g);
                    p[0] = rgb[0]; p[1] = rgb[1]; p[2] = rgb[2]; p[3] = 1.0f;
                }
    }

    Param<std::string> m_path{
        this, "file", {},
        "Path to a .cube LUT, relative to the script. A 3D table cannot be "
        "decomposed into sliders, so this is load-and-apply; stack the ordinary "
        "adjustments after it to taste."};

    Param<float> m_strength{
        this, "strength", 1.0f, 0.0f, 1.0f,
        {.help = "Blend between the original and the graded result. 1 is the "
                 "LUT as authored, 0.5 half-strength, 0 off.",
         .step = 0.01}};

    Lut3D       m_lut;
    Image       m_table;      // the cube packed as a 2D texture, for the GPU
    float       m_gpuWhite = 1.0f;
    std::string m_loadedPath;
    std::string m_loadError;
    double      m_outOfDomain = 0.0;

    PixelBuffer m_in, m_out;
};

REGISTER_ALGORITHM(ApplyLut);

}  // namespace tglab
