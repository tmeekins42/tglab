// vignette — darken or lighten the corners.
//
// Lightroom's sign convention, because it is the one people already have in
// their fingers: NEGATIVE amount darkens the corners, which is what a lens
// actually does and what almost everyone wants; POSITIVE lightens them, which
// is the "white vignette" used to lift a subject off a pale background.
//
// WHAT THE SHAPE IS. The falloff runs on the distance from the centre,
// normalised so that 1.0 sits at the corner rather than at the edge. That
// matters: normalising to the shorter axis would put the effect fully on
// before the corners were reached on anything but a square, so a 3:2 frame
// would clip flat along its long edges.
//
//   d = |p - centre| / |corner - centre|,  with p in ASPECT-CORRECTED units
//
// Aspect correction is what makes the darkened region an ellipse matching the
// frame rather than a circle inscribed in it. Without it a 16:9 frame gets a
// circular vignette whose top and bottom are already black while its sides are
// untouched.
//
// `midpoint` is where the falloff begins, as a fraction of that distance, and
// `roundness` blends between an ellipse following the frame and a true circle.
// `feather` is the softness of the transition, applied with smoothstep so the
// derivative is continuous at both ends -- a linear ramp leaves a visible
// crease where it meets the untouched middle, which on a smooth sky is exactly
// where the eye looks.
//
// WHY IT MULTIPLIES RATHER THAN OFFSETS. A vignette is a transmission loss:
// less light reaches the corner of the frame. Scaling is what that means, and
// it has the property an offset lacks -- it cannot push a pixel below zero, it
// preserves black, and it acts proportionally, so a bright corner darkens more
// in absolute terms than an already-dim one. Subtracting a constant instead
// would crush shadow detail to flat black at the corners and lift true black
// to grey at positive amounts.
//
// The lightening direction is not the same operation with the sign flipped.
// Multiplying by more than 1 blows highlights out at the corners while leaving
// the shadows nearly untouched, which looks like a lighting error rather than
// a vignette. Lifting toward white instead -- a lerp of the pixel toward the
// maximum -- is what a white vignette actually looks like, so that is what the
// positive direction does. See ApplyVignette.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "../../algo_util/pixel_buffer.h"
#include "../../core/algorithm.h"

namespace tglab {

class Vignette : public AlgorithmBase {
public:
    const char* Name()     const override { return "vignette"; }
    const char* Category() const override { return "adjust"; }

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

        // WHERE THIS RASTER SITS IN THE PICTURE. Both are identities for a
        // whole, full-resolution image -- the common case -- and both matter
        // the moment the pipeline hands over a crop or a proxy, because the
        // geometry below is anchored to the frame rather than to the pixels.
        const ImageDesc& d = ctx.InDesc(0);
        const int fullW = std::max(1, d.FullWidth());
        const int fullH = std::max(1, d.FullHeight());
        const int offX  = d.originX;
        const int offY  = d.originY;

        // White is 255 on an RGBA8 image and 1.0 on a float one, and the
        // lightening direction lerps toward it -- so unlike a pure multiply,
        // this direction genuinely needs the image's own units.
        const float white = m_in.ValueScale();

        const float amount    = float(m_amount);
        const float midpoint  = float(m_midpoint);
        const float feather   = float(m_feather);
        const float roundness = float(m_roundness);

        for (int y = 0; y < h; ++y) {
            if (ctx.Cancelled()) return;
            for (int x = 0; x < w; ++x) {
                const float dist = Distance(x + offX, y + offY, fullW, fullH, roundness);
                const float k = Falloff(dist, midpoint, feather);

                const int colours = (ch == 4) ? 3 : ch;
                for (int c = 0; c < colours; ++c)
                    m_out.Set(x, y, c,
                              ApplyVignette(m_in.Get(x, y, c), k, amount, white));
                // Alpha untouched: a vignette changes what light reaches the
                // corner, not what is transparent.
                if (ch == 4) m_out.Set(x, y, 3, m_in.Get(x, y, 3));
            }
        }

        m_out.PackInto(dst);
    }

    // Amount 0 is exactly the identity whatever the other controls say, so the
    // stage is skipped outright -- no allocation, no dispatch, no copy.
    bool IsNoOp() const override { return float(m_amount) == 0.0f; }

    // --- GPU implementation -------------------------------------------------
    // One fetch, a little arithmetic, one store. The falloff is duplicated in
    // HLSL rather than shared, which is the one place the two paths could
    // drift; the CPU/GPU agreement test is what holds them together.
    bool HasGPU() const override { return true; }

    const char* GpuSource() const override {
        return R"(
Texture2D<float4>   Src : register(t0);
RWTexture2D<float4> Dst : register(u0);

cbuffer Params : register(b0) {
    uint Width;
    uint Height;
    uint AmountBits;
    uint MidpointBits;
    uint FeatherBits;
    uint RoundnessBits;

    // The FULL frame and where this raster sits in it. Equal to Width/Height
    // and 0,0 for a whole image; different when the pipeline hands over a crop
    // of the visible area, which must not move the vignette.
    uint FullWidth;
    uint FullHeight;
    uint OffsetX;
    uint OffsetY;
};

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;

    float amount    = asfloat(AmountBits);
    float midpoint  = asfloat(MidpointBits);
    float feather   = asfloat(FeatherBits);
    float roundness = asfloat(RoundnessBits);

    // Pixel centres, so the distance is symmetric about the middle: sampling
    // at the top-left corner of each texel biases everything half a pixel up
    // and left, which shows as an off-centre vignette on a small image.
    // Against the FULL frame, offset by where this raster starts in it.
    float2 fullSize = float2(max(FullWidth, 1u), max(FullHeight, 1u));
    float2 pos      = float2(tid.xy) + float2(OffsetX, OffsetY);
    float2 p = (pos + 0.5) / fullSize * 2.0 - 1.0;

    // Aspect correction, blended by roundness. At 0 the vignette is an ellipse
    // following the frame; at 1 it is a true circle, so the corners of a wide
    // frame stay lighter than its long edges.
    float aspect = fullSize.x / fullSize.y;
    float2 q = p;
    if (aspect > 1.0) q.x = p.x * lerp(1.0, aspect, roundness);
    else              q.y = p.y * lerp(1.0, 1.0 / aspect, roundness);

    // Normalised so 1.0 is the corner rather than the edge.
    float2 corner = float2(1.0, 1.0);
    if (aspect > 1.0) corner.x = lerp(1.0, aspect, roundness);
    else              corner.y = lerp(1.0, 1.0 / aspect, roundness);
    float d = length(q) / max(length(corner), 1e-6);

    // Where the falloff starts, and how soft it is. smoothstep rather than a
    // linear ramp: a continuous derivative at both ends, so there is no crease
    // where the effect meets the untouched middle.
    float inner = midpoint * (1.0 - feather);
    float outer = midpoint + (1.0 - midpoint) * feather;
    float k = smoothstep(inner, max(outer, inner + 1e-4), d);

    float4 c = Src[int2(tid.xy)];
    float3 rgb;
    if (amount < 0.0) {
        // Darken: a transmission loss, so it scales. Cannot go below zero,
        // preserves black, acts proportionally.
        rgb = c.rgb * (1.0 + amount * k);
    } else {
        // Lighten: a lerp toward white. Multiplying up instead would blow the
        // corner highlights while barely moving its shadows -- a lighting
        // error rather than a vignette. White is 1.0 here because a UNORM SRV
        // hands the shader 0..1 whatever the storage format.
        rgb = lerp(c.rgb, float3(1.0, 1.0, 1.0), amount * k);
    }
    Dst[tid.xy] = float4(rgb, c.a);
}
)";
    }

    // The frame this raster is a window onto. See Distance(): the geometry is
    // anchored to the picture, so a crop or a proxy must not change it.
    void PrepareGpu(const std::vector<ImageDesc>& inputs) override {
        if (inputs.empty()) return;
        m_fullW = std::max(1, inputs[0].FullWidth());
        m_fullH = std::max(1, inputs[0].FullHeight());
        m_offX  = inputs[0].originX;
        m_offY  = inputs[0].originY;
    }

    std::vector<uint32_t> GpuConstants(int) const override {
        auto bits = [](float f) {
            uint32_t u;
            std::memcpy(&u, &f, sizeof(u));
            return u;
        };
        return {bits(float(m_amount)), bits(float(m_midpoint)),
                bits(float(m_feather)), bits(float(m_roundness)),
                uint32_t(m_fullW), uint32_t(m_fullH),
                uint32_t(m_offX),  uint32_t(m_offY)};
    }

private:
    // Distance from the centre, 1.0 at the corner. Kept in step with the HLSL
    // above -- see the comments there for why each step is as it is.
    //
    // x/y are the pixel's position in the FULL FRAME and w/h the full frame's
    // size, NOT this image's. A vignette is a property of the lens, so it is
    // anchored to the picture rather than to whatever raster happens to be in
    // hand. Passing the local extent instead re-centred the darkening on the
    // visible rectangle during a zoomed-in drag -- the effect appeared to
    // follow the viewport, which is exactly what a vignette must not do.
    static float Distance(int x, int y, int w, int h, float roundness) {
        const float px = (float(x) + 0.5f) / float(w) * 2.0f - 1.0f;
        const float py = (float(y) + 0.5f) / float(h) * 2.0f - 1.0f;
        const float aspect = float(w) / float(h);

        float qx = px, qy = py;
        float cx = 1.0f, cy = 1.0f;
        if (aspect > 1.0f) {
            const float s = 1.0f + (aspect - 1.0f) * roundness;
            qx = px * s;
            cx = s;
        } else {
            const float inv = 1.0f / aspect;
            const float s = 1.0f + (inv - 1.0f) * roundness;
            qy = py * s;
            cy = s;
        }
        const float len    = std::sqrt(qx * qx + qy * qy);
        const float cornerLen = std::sqrt(cx * cx + cy * cy);
        return len / std::max(cornerLen, 1e-6f);
    }

    static float Falloff(float d, float midpoint, float feather) {
        const float inner = midpoint * (1.0f - feather);
        const float outer = midpoint + (1.0f - midpoint) * feather;
        return SmoothStep(inner, std::max(outer, inner + 1e-4f), d);
    }

    static float SmoothStep(float a, float b, float x) {
        const float t = std::clamp((x - a) / std::max(b - a, 1e-6f), 0.0f, 1.0f);
        return t * t * (3.0f - 2.0f * t);
    }

    // The asymmetry between the two directions is deliberate -- see the note at
    // the top of the file.
    static float ApplyVignette(float v, float k, float amount, float white) {
        if (amount < 0.0f) return v * (1.0f + amount * k);
        return v + (white - v) * (amount * k);
    }

    Param<float> m_amount{
        this, "amount", 0.0f, -1.0f, 1.0f,
        {.help = "Negative darkens the corners, positive lightens them, "
                 "0 is off. Darkening scales the pixel, as a real lens does; "
                 "lightening lifts it toward white, which is what a white "
                 "vignette looks like.",
         .step = 0.01}};

    Param<float> m_midpoint{
        this, "midpoint", 0.5f, 0.0f, 1.0f,
        {.help = "How far out the falloff begins, as a fraction of the "
                 "distance from centre to corner. Lower reaches further in.",
         .step = 0.01}};

    Param<float> m_feather{
        this, "feather", 0.5f, 0.0f, 1.0f,
        {.help = "Softness of the transition. 0 is a hard edge at the "
                 "midpoint; 1 spreads it across the whole frame.",
         .step = 0.01}};

    Param<float> m_roundness{
        this, "roundness", 0.0f, 0.0f, 1.0f,
        {.help = "0 follows the frame, so the shape is an ellipse matching its "
                 "aspect. 1 is a true circle, leaving the corners of a wide "
                 "frame lighter than its long edges.",
         .step = 0.01}};

    PixelBuffer m_in, m_out;

    // Set by PrepareGpu from the input descriptor; the identity for a whole
    // full-resolution image.
    int m_fullW = 1, m_fullH = 1, m_offX = 0, m_offY = 0;
};

REGISTER_ALGORITHM(Vignette);

}  // namespace tglab
