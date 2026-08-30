// crop — trim the edges and straighten, with a preview that shows what will go.
//
// THE PREVIEW TOGGLE IS THE POINT, and it is Tim's design. A crop that only
// ever outputs the cropped raster is almost unusable interactively: the moment
// a slider moves, the part being cropped away is off screen, so there is no way
// to see what is about to be lost or to judge where the edge should sit. The
// input is still there -- nothing was destroyed -- but the viewer is not
// showing it.
//
// So `preview` splits the algorithm in two:
//
//   preview = 1  the output is the FULL frame with the crop rectangle drawn on
//                it, dimmed outside. Move the sliders and watch the rectangle.
//   preview = 0  the output is the cropped raster, at its new size.
//
// Same parameters, same rectangle, two ways of showing it. Set the rectangle in
// preview, then turn preview off to get the result.
//
// ROTATION BELONGS HERE rather than in its own algorithm, and not only because
// Tim asked for it in the same breath. Straightening and cropping are one
// operation: rotating a frame by two degrees leaves wedges of empty corner that
// have to be cropped away, and cropping first then rotating re-introduces them.
// Doing both at once means the rectangle is measured in the ROTATED frame, so
// what the preview draws is exactly what comes out -- corners included.
//
// The rectangle is in FRACTIONS of the frame, not pixels. A crop set on a
// preview-sized image would otherwise mean something different on the full-size
// one, and a script that crops a group would mean different things for frames
// of different sizes. Fractions make "trim 10% off the top" survive both.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../../algo_util/pixel_buffer.h"
#include "../../algo_util/transform.h"
#include "../../core/algorithm.h"

namespace tglab {
namespace {

class Crop : public AlgorithmBase {
public:
    const char* Name()     const override { return "crop"; }
    const char* Category() const override { return "adjust"; }

    PortList Inputs()  const override { return {{"src", DataType::Image, FormatSpec::Any}}; }
    PortList Outputs() const override {
        return {{"out", DataType::Image, FormatSpec::SameAsInput}};
    }

    // In preview the output is the input's size, because the whole frame is
    // being shown with a rectangle on it. Applied, it is the rectangle.
    ImageDesc OutputDesc(int, const ImageDesc& in) const override {
        if (bool(m_preview)) return in;

        ImageDesc d = in;
        int x0, y0, w, h;
        Rect(in.width, in.height, &x0, &y0, &w, &h);
        d.width  = w;
        d.height = h;
        return d;
    }

    // A crop that trims nothing and rotates nothing is a pass-through, and
    // saying so lets the pipeline alias it away entirely. Worth having on
    // exactly this algorithm: a script that keeps a crop stage permanently in
    // the chain pays nothing for it until it is actually used.
    //
    // Not in preview, though -- there the output genuinely differs from the
    // input, because it has a rectangle drawn on it.
    bool IsNoOp() const override {
        return !bool(m_preview) &&
               float(m_left) == 0.0f && float(m_right) == 0.0f &&
               float(m_top) == 0.0f && float(m_bottom) == 0.0f &&
               float(m_angle) == 0.0f;
    }

    void RunCPU(RunCtx& ctx) override {
        const ImageView src = ctx.In(0);
        ImageView       dst = ctx.Out(0);
        if (!src.Valid() || !dst.Valid()) return;

        PixelBuffer in;
        in.Unpack(src);
        if (!in.Valid()) return;

        PixelBuffer out;
        out.Unpack(dst);
        if (!out.Valid()) return;

        const int w = in.Width(), h = in.Height(), ch = in.Channels();
        int x0, y0, cw, chh;
        Rect(w, h, &x0, &y0, &cw, &chh);

        m_w = cw;
        m_h = chh;

        // The rotation, about the CENTRE OF THE CROP RECTANGLE rather than the
        // centre of the frame.
        //
        // That choice is what makes the two modes agree. Rotating about the
        // frame centre would swing the rectangle around as the angle changed,
        // so the preview's rectangle and the applied crop would show different
        // parts of the image for the same settings. About the rectangle's own
        // centre, the angle turns the CONTENT inside a rectangle that stays
        // put -- which is what "straighten this" means.
        const float a  = float(m_angle) * 3.14159265f / 180.0f;
        const float ca = std::cos(a), sa = std::sin(a);
        const float rcx = float(x0) + 0.5f * float(cw);
        const float rcy = float(y0) + 0.5f * float(chh);

        if (bool(m_preview)) {
            DrawPreview(in, out, x0, y0, cw, chh, ca, sa, rcx, rcy, ch);
        } else {
            float sm[4] = {0, 0, 0, 0};
            for (int y = 0; y < chh; ++y) {
                if (ctx.Cancelled()) return;
                for (int x = 0; x < cw; ++x) {
                    // Output pixel -> its position in the rotated rectangle ->
                    // the source pixel to read.
                    const float ox = float(x) - 0.5f * float(cw);
                    const float oy = float(y) - 0.5f * float(chh);
                    const float sx = rcx + ca * ox - sa * oy;
                    const float sy = rcy + sa * ox + ca * oy;
                    SampleBilinear(in, sx, sy, sm);
                    float* p = out.At(x, y);
                    for (int c = 0; c < ch; ++c) p[c] = sm[c];
                }
            }
        }
        out.PackInto(dst);
    }

    std::string RunReport() const override {
        if (bool(m_preview)) {
            char buf[96];
            std::snprintf(buf, sizeof buf,
                          "preview: the crop would be %dx%d -- turn preview off to apply",
                          m_w, m_h);
            return buf;
        }
        if (m_w <= 0) return {};
        char buf[64];
        std::snprintf(buf, sizeof buf, "cropped to %dx%d", m_w, m_h);
        return buf;
    }

    bool HasGPU() const override { return true; }

    // One kernel for both modes, branching on Preview.
    //
    // Two kernels would be tidier to read and worse in the one way that
    // matters: the modes must place the rectangle IDENTICALLY, and two shaders
    // are two places for that arithmetic to drift apart. The branch is uniform
    // across the whole dispatch, so it costs nothing -- every thread takes the
    // same side of it.
    //
    // BILINEAR IS DONE BY HAND rather than through a sampler. The root
    // signature here binds SRVs and UAVs and no sampler at all, and adding one
    // for this would change it for all fourteen GPU algorithms. Four integer
    // loads and three lerps is the same arithmetic a sampler would do, and it
    // matches SampleBilinear on the CPU exactly -- which is what lets the two
    // paths be compared rather than merely both look right.
    const char* GpuSource() const override {
        return R"(
Texture2D<float4>   Src : register(t0);
RWTexture2D<float4> Dst : register(u0);

cbuffer Params : register(b0) {
    uint Width;        // the OUTPUT's width: the crop in applied mode
    uint Height;
    uint PreviewBits;  // 0 applied, 1 preview
    uint SrcWBits;
    uint SrcHBits;
    uint CosBits;
    uint SinBits;
    uint RcxBits;
    uint RcyBits;
    uint CwBits;
    uint ChBits;
    uint DimBits;
    uint LineWBits;
    uint LineValBits;
};

// Edge-clamped bilinear, matching algo_util/transform.h's SampleBilinear --
// including its exact-pixel early exit, so an unrotated crop copies pixels
// verbatim rather than reconstructing them from a weighted sum that happens to
// reduce to the same value.
float4 SampleBilin(float x, float y, float w, float h) {
    float cx = clamp(x, 0.0, w - 1.0);
    float cy = clamp(y, 0.0, h - 1.0);
    int   x0 = (int)cx, y0 = (int)cy;
    int   x1 = min(x0 + 1, (int)w - 1);
    int   y1 = min(y0 + 1, (int)h - 1);
    float fx = cx - (float)x0, fy = cy - (float)y0;

    if (fx == 0.0 && fy == 0.0) return Src[int2(x0, y0)];

    float4 p00 = Src[int2(x0, y0)];
    float4 p10 = Src[int2(x1, y0)];
    float4 p01 = Src[int2(x0, y1)];
    float4 p11 = Src[int2(x1, y1)];
    float4 top = lerp(p00, p10, fx);
    float4 bot = lerp(p01, p11, fx);
    return lerp(top, bot, fy);
}

// Signed distance from `pt` to the edge a->b, positive on the inside for a
// clockwise winding in screen coordinates. Matches the CPU's inside test.
float EdgeDist(float2 a, float2 b, float2 pt) {
    float2 e = b - a;
    float len = sqrt(e.x * e.x + e.y * e.y);
    if (len < 1e-6) return 1e30;
    return (e.x * (pt.y - a.y) - e.y * (pt.x - a.x)) / len;
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;

    float ca = asfloat(CosBits), sa = asfloat(SinBits);
    float rcx = asfloat(RcxBits), rcy = asfloat(RcyBits);
    float cw = asfloat(CwBits), chh = asfloat(ChBits);
    float sw = asfloat(SrcWBits), sh = asfloat(SrcHBits);

    if (PreviewBits == 0) {
        // Applied: this output pixel's offset from the rectangle's centre,
        // rotated into the source frame.
        float ox = (float)tid.x - 0.5 * cw;
        float oy = (float)tid.y - 0.5 * chh;
        Dst[tid.xy] = SampleBilin(rcx + ca * ox - sa * oy,
                                  rcy + sa * ox + ca * oy, sw, sh);
        return;
    }

    // Preview: the full frame, with the rectangle drawn and the outside dimmed.
    float4 s = Src[int2(tid.xy)];

    // The rectangle's corners, in the same order as the CPU path: the winding
    // is clockwise ON SCREEN because y increases downward, and the signed
    // distance below depends on that.
    //
    // Written out rather than looped over a local array. An indexed float[4]
    // inside an unrolled loop is exactly the shape DXC is entitled to place in
    // indexable temp registers, and the first version of this returned
    // uninitialised garbage on three of four test cases -- values around 1e36,
    // with the one case that never entered this branch (an unrotated applied
    // crop) matching the CPU bit for bit. Four corners do not need a loop.
    float hx = 0.5 * cw, hy = 0.5 * chh;

    float2 c0 = float2(rcx + ca * -hx - sa * -hy, rcy + sa * -hx + ca * -hy);
    float2 c1 = float2(rcx + ca *  hx - sa * -hy, rcy + sa *  hx + ca * -hy);
    float2 c2 = float2(rcx + ca *  hx - sa *  hy, rcy + sa *  hx + ca *  hy);
    float2 c3 = float2(rcx + ca * -hx - sa *  hy, rcy + sa * -hx + ca *  hy);

    float2 pt = float2((float)tid.x, (float)tid.y);
    float inside = min(min(EdgeDist(c0, c1, pt), EdgeDist(c1, c2, pt)),
                       min(EdgeDist(c2, c3, pt), EdgeDist(c3, c0, pt)));

    float lineW   = asfloat(LineWBits);
    float lineVal = asfloat(LineValBits);
    float dim     = asfloat(DimBits);

    if (inside >= -lineW * 0.5 && inside <= lineW * 0.5)
        Dst[tid.xy] = float4(0.0, lineVal, 0.0, s.a);
    else if (inside > 0.0)
        Dst[tid.xy] = s;
    else
        Dst[tid.xy] = float4(s.rgb * dim, s.a);
}
)";
    }

    // The input's size and level, which the shader cannot see for itself.
    //
    // The rectangle is a FRACTION of the input, and the constants are in
    // pixels -- so they cannot be computed without knowing the input's
    // dimensions. GpuConstants() runs after the descriptors are out of reach,
    // which is exactly what PrepareGpu exists for.
    void PrepareGpu(const std::vector<ImageDesc>& inputs) override {
        if (inputs.empty()) return;
        m_srcW = inputs[0].width;
        m_srcH = inputs[0].height;
        Rect(m_srcW, m_srcH, &m_x0, &m_y0, &m_w, &m_h);
    }

    // The preview's line colour has to be MEASURED, and only the CPU can
    // measure it -- so the pixels are pulled back for exactly that, and only
    // when a preview is actually being drawn.
    //
    // Gated on m_preview because the applied path needs nothing measured, and
    // this readback is not free: it is the one sync point in an otherwise
    // fully-GPU stage. Paying it while cropping a 45 MP panorama, to compute a
    // colour that is never drawn, would be a real cost for nothing.
    bool GpuNeedsInputPixels() const override { return bool(m_preview); }

    void MeasureForGpu(const std::vector<const Image*>& inputs) override {
        if (inputs.empty() || !inputs[0]) return;
        ImageView v = const_cast<Image*>(inputs[0])->MapCpuRead();
        if (!v.data) return;
        PixelBuffer b;
        b.Unpack(v);
        if (!b.Valid()) return;
        // Divided by ValueScale for the same reason brightness.cpp does not
        // multiply by 255 on the GPU: a UNORM SRV hands the shader 0..1
        // whatever the storage format, so the value has to be expressed in
        // those units rather than in the source's.
        const float s = b.ValueScale();
        m_lineVal = (s > 1e-6f) ? LineValue(b) / s : LineValue(b);
    }

    std::vector<uint32_t> GpuConstants(int) const override {
        auto bits = [](float f) {
            uint32_t u;
            std::memcpy(&u, &f, sizeof(u));
            return u;
        };
        const float a  = float(m_angle) * 3.14159265f / 180.0f;
        const float rcx = float(m_x0) + 0.5f * float(m_w);
        const float rcy = float(m_y0) + 0.5f * float(m_h);

        // The line value comes from MeasureForGpu, so both paths draw the
        // same colour. See LineValue for why that measurement is shared.
        return {uint32_t(bool(m_preview) ? 1 : 0),
                bits(float(m_srcW)), bits(float(m_srcH)),
                bits(std::cos(a)),   bits(std::sin(a)),
                bits(rcx),           bits(rcy),
                bits(float(m_w)),    bits(float(m_h)),
                bits(std::clamp(float(m_dim), 0.0f, 1.0f)),
                bits(std::max(1.0f, float(m_lineWidth))),
                bits(m_lineVal)};
    }

private:
    // How bright to draw the preview rectangle, in the image's own units.
    //
    // A fixed 1.0 would be invisible on a scene-referred raw whose bright end
    // is 0.16, and would clip on an 8-bit image; so it is measured, on the same
    // reasoning as the detectors' Percentile99.
    //
    // ONE FUNCTION FOR BOTH PATHS, and that is the point of it being a function
    // at all. The first version measured this inside the CPU draw and passed a
    // hard-coded 1.0 to the shader, on the theory that a UNORM SRV always hands
    // the shader 0..1. That is true for RGBA8 and false for a float image,
    // where the two paths then drew different-coloured lines -- the agreement
    // test measured 0.38 apart. The measurement now happens once, on the CPU,
    // and the shader is told the answer.
    static float LineValue(const PixelBuffer& in) {
        float m = 0.0f;
        // Every 17th pixel: a stride rather than every pixel because this is a
        // colour, not a threshold -- and prime, so it does not fall into step
        // with a regular pattern in the image.
        for (int y = 0; y < in.Height(); y += 17)
            for (int x = 0; x < in.Width(); x += 17)
                m = std::max(m, in.At(x, y)[0]);
        return m > 1e-6f ? m : in.ValueScale();
    }

    // The crop rectangle in pixels, from the fractional parameters.
    //
    // Clamped so the rectangle is always at least one pixel and always inside
    // the frame: sliders that cross over (left past right) would otherwise
    // produce a negative width, and an image cannot be allocated from that. A
    // crossed pair collapses to a sliver instead, which is visible and
    // recoverable rather than an error the user has to read.
    void Rect(int w, int h, int* x0, int* y0, int* cw, int* ch) const {
        const float l = std::clamp(float(m_left),   0.0f, 0.95f);
        const float r = std::clamp(float(m_right),  0.0f, 0.95f);
        const float t = std::clamp(float(m_top),    0.0f, 0.95f);
        const float b = std::clamp(float(m_bottom), 0.0f, 0.95f);

        int a = int(std::lround(double(l) * w));
        int z = w - int(std::lround(double(r) * w));
        int c = int(std::lround(double(t) * h));
        int d = h - int(std::lround(double(b) * h));

        if (z <= a) z = std::min(w, a + 1);
        if (d <= c) d = std::min(h, c + 1);

        *x0 = std::clamp(a, 0, std::max(0, w - 1));
        *y0 = std::clamp(c, 0, std::max(0, h - 1));
        *cw = std::max(1, std::min(z, w) - *x0);
        *ch = std::max(1, std::min(d, h) - *y0);
    }

    // Preview: the whole frame, dimmed outside the rectangle, with the
    // rectangle's edges drawn.
    //
    // Dimmed rather than hidden, because the point of the preview is to judge
    // what is being given up -- a black surround would answer the question
    // "where is the edge" and not "should the edge be there".
    void DrawPreview(const PixelBuffer& in, PixelBuffer& out,
                     int x0, int y0, int cw, int chh,
                     float ca, float sa, float rcx, float rcy, int ch) const {
        const int w = in.Width(), h = in.Height();
        const float dim = std::clamp(float(m_dim), 0.0f, 1.0f);

        // The rectangle's four corners in FRAME coordinates, rotated. Computed
        // once: the inside test below is four half-plane tests against these.
        float px[4], py[4];
        const float hx = 0.5f * float(cw), hy = 0.5f * float(chh);
        const float lx[4] = {-hx,  hx, hx, -hx};
        const float ly[4] = {-hy, -hy, hy,  hy};
        for (int i = 0; i < 4; ++i) {
            px[i] = rcx + ca * lx[i] - sa * ly[i];
            py[i] = rcy + sa * lx[i] + ca * ly[i];
        }

        const float lineVal = LineValue(in);

        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const float* s = in.At(x, y);
                float*       d = out.At(x, y);

                // Inside the rotated rectangle? The signed distance to each
                // edge, taken as the minimum, doubles as the line test -- so
                // one computation answers both "dim this" and "draw this".
                // The sign convention is worth stating rather than assuming.
                // The corners are listed (-x,-y), (+x,-y), (+x,+y), (-x,+y),
                // which walks CLOCKWISE on screen because y increases downward
                // -- the opposite of the same list in a y-up frame. For that
                // winding the inward normal of edge i->j is (-ey, ex), so the
                // signed distance is the cross product taken in this order.
                //
                // The first version had it the other way round, which put every
                // pixel outside: the preview came back uniformly dimmed with no
                // rectangle drawn at all, which is exactly what a consistently
                // wrong sign looks like -- no partial failure to hint at it.
                float inside = 1e30f;
                for (int i = 0; i < 4; ++i) {
                    const int j = (i + 1) & 3;
                    const float ex = px[j] - px[i], ey = py[j] - py[i];
                    const float len = std::sqrt(ex * ex + ey * ey);
                    if (len < 1e-6f) continue;
                    const float dist = (ex * (float(y) - py[i]) -
                                        ey * (float(x) - px[i])) / len;
                    inside = std::min(inside, dist);
                }

                const float lineW = std::max(1.0f, float(m_lineWidth));
                if (inside >= -lineW * 0.5f && inside <= lineW * 0.5f) {
                    // On the edge: draw the line.
                    for (int c = 0; c < ch; ++c)
                        d[c] = (c == 1 || ch < 3) ? lineVal : 0.0f;   // green
                    if (ch == 4) d[3] = s[3];
                } else if (inside > 0.0f) {
                    for (int c = 0; c < ch; ++c) d[c] = s[c];
                } else {
                    for (int c = 0; c < ch; ++c) d[c] = s[c] * dim;
                    if (ch == 4) d[3] = s[3];
                }
            }
    }

    Param<bool> m_preview{this, "preview", true,
        "Show the whole frame with the crop rectangle drawn on it, instead of "
        "the cropped result. Set the rectangle here -- where what is being cut "
        "away is still visible -- then turn this off to apply it."};

    Param<float> m_left{this, "left", 0.0f, 0.0f, 0.95f,
        {.help = "Trim from the left, as a fraction of the width. Fractions "
                 "rather than pixels so the same crop means the same thing on "
                 "a preview-sized image and a full-sized one.",
         .step = 0.005, .softMax = 0.4}};

    Param<float> m_right{this, "right", 0.0f, 0.0f, 0.95f,
        {.help = "Trim from the right, as a fraction of the width.",
         .step = 0.005, .softMax = 0.4}};

    Param<float> m_top{this, "top", 0.0f, 0.0f, 0.95f,
        {.help = "Trim from the top, as a fraction of the height.",
         .step = 0.005, .softMax = 0.4}};

    Param<float> m_bottom{this, "bottom", 0.0f, 0.0f, 0.95f,
        {.help = "Trim from the bottom, as a fraction of the height.",
         .step = 0.005, .softMax = 0.4}};

    Param<float> m_angle{this, "angle", 0.0f, -45.0f, 45.0f,
        {.help = "Straighten, in degrees, about the centre of the crop "
                 "rectangle. About the RECTANGLE rather than the frame, so "
                 "turning the angle rotates the content inside a rectangle "
                 "that stays where it was put.",
         .step = 0.1, .softMin = -10.0, .softMax = 10.0}};

    Param<float> m_dim{this, "preview_dim", 0.35f, 0.0f, 1.0f,
        {.help = "How much to darken the area outside the rectangle in "
                 "preview. Dimmed rather than hidden, so what is being given "
                 "up stays visible -- which is the question the preview is "
                 "there to answer.",
         .step = 0.05}};

    Param<float> m_lineWidth{this, "preview_line", 3.0f, 1.0f, 20.0f,
        {.help = "Thickness of the preview rectangle, in pixels. Wider is "
                 "easier to see on a large image shown zoomed out.",
         .step = 1.0}};

    int m_w = 0, m_h = 0;
    // The input size and rectangle origin, captured in PrepareGpu: the shader
    // needs them in pixels and cannot see the descriptors.
    int m_srcW = 0, m_srcH = 0;
    // The preview line colour, measured in MeasureForGpu.
    float m_lineVal = 1.0f;
    int m_x0 = 0, m_y0 = 0;
};

REGISTER_ALGORITHM(Crop);

} // namespace
} // namespace tglab
