#include "viewport3d.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>

#include "../gpu/device.h"
#include "../gpu/shader.h"
#include "imgui.h"

namespace tglab {
namespace {

// One point: world position, colour, and a size.
//
// Sixteen-byte aligned on purpose -- a structured buffer's stride should be a
// multiple of 16 or the shader reads padded garbage between elements.
struct GpuPoint {
    float x, y, z, size;
    float r, g, b, a;
};

// The vertex shader expands each point into a screen-facing quad.
//
// SV_VertexID rather than a vertex buffer: six vertices per point, index / 6
// selects the point and index % 6 selects the corner. That means the only
// upload is the point data itself -- no index buffer, no per-corner
// duplication, and changing the point size costs a constant rather than a
// re-upload.
constexpr const char* kVertexShader = R"(
struct Point { float4 posSize; float4 colour; };
StructuredBuffer<Point> Points : register(t0);

cbuffer Constants : register(b0) {
    // ROW-MAJOR, stated explicitly because HLSL's default is the opposite.
    //
    // A float4x4 read from a constant buffer is packed COLUMN-major unless
    // told otherwise, while OrbitCamera::ViewProj writes rows -- so the shader
    // silently reads the transpose. The symptom is not a crash or a blank
    // screen: a transposed view-projection still projects, it just projects a
    // three-dimensional cloud onto a LINE. The extent was 12.7 x 2.8 x 10.8
    // and the screen showed a diagonal streak.
    //
    // The orbit camera's own tests pass either way, because they multiply the
    // matrix themselves and never ask what HLSL would do with it.
    row_major float4x4 ViewProj;
    float2   InvViewport;   // 1/width, 1/height
    float    SizeScale;
    float    Pad;
};

struct VSOut {
    float4 pos    : SV_Position;
    float4 colour : COLOR;
    float2 uv     : TEXCOORD0;
};

VSOut main(uint vid : SV_VertexID) {
    Point p = Points[vid / 6];

    // Two triangles: 0,1,2 and 2,1,3 in a unit square.
    uint corner = vid % 6;
    const uint2 lut[6] = { uint2(0,0), uint2(1,0), uint2(0,1),
                           uint2(0,1), uint2(1,0), uint2(1,1) };
    float2 c = float2(lut[corner]) * 2.0 - 1.0;

    float4 clip = mul(float4(p.posSize.xyz, 1.0), ViewProj);

    // Offset in CLIP space, scaled by w so the quad is a constant size in
    // pixels rather than shrinking with distance. A dot that shrank would be
    // invisible at the back of a cloud, which is where the interesting
    // structure usually is.
    clip.xy += c * p.posSize.w * SizeScale * InvViewport * clip.w;

    VSOut o;
    o.pos    = clip;
    o.colour = p.colour;
    o.uv     = c;
    return o;
}
)";

// A round dot with a soft edge, rather than the square the quad actually is.
//
// Discarding outside the unit circle costs nothing and makes a dense cloud far
// easier to read: square dots tile into a solid mass where round ones stay
// distinguishable.
constexpr const char* kPixelShader = R"(
struct VSOut {
    float4 pos    : SV_Position;
    float4 colour : COLOR;
    float2 uv     : TEXCOORD0;
};

float4 main(VSOut i) : SV_Target {
    float r2 = dot(i.uv, i.uv);
    if (r2 > 1.0) discard;
    // A little falloff at the rim, so dots read as round rather than as
    // aliased discs.
    float a = saturate((1.0 - r2) * 3.0);
    return float4(i.colour.rgb, i.colour.a * a);
}
)";

struct Constants {
    float viewProj[16];
    float invViewport[2];
    float sizeScale;
    float pad;
};

// --- Gaussian splats ----------------------------------------------------------
//
// One Gaussian as the shader reads it: the mean and opacity, the six unique
// entries of the 3D covariance, and the colour. Sixty-four bytes, a multiple
// of sixteen for the reason GpuPoint gives.
//
// The covariance is built on the CPU from scale and rotation (Splat::
// Covariance) rather than in the shader: it changes only when the splats do,
// and the vertex shader runs it six times per splat per frame.
struct GpuSplat {
    float mx, my, mz, opacity;
    float cxx, cxy, cxz, cyy;
    float cyz, czz, pad0, pad1;
    float r, g, b, pad2;
};

// The splat vertex shader: EWA splatting, as in Zwicker et al. 2002 and the
// Gaussian-splatting paper after it.
//
// A 3D Gaussian seen through a pinhole is, to first order, a 2D Gaussian on
// the screen. The 2D covariance is J W Sigma W^T J^T, where W rotates world
// into the camera's frame and J is the Jacobian of the perspective projection
// at the Gaussian's centre -- the linearisation that makes the projected shape
// an ellipse rather than something the rasteriser cannot describe.
//
// The quad is then the ellipse's bounding box along its own axes, three
// standard deviations out, and the pixel shader evaluates the Gaussian inside
// it. Everything is in the same screen-pixel frame, x right and y up, so the
// offset the vertex shader writes and the conic the pixel shader evaluates
// agree without any sign bookkeeping.
constexpr const char* kSplatVertexShader = R"(
struct GSplat { float4 meanOp; float4 covA; float4 covB; float4 colour; };
StructuredBuffer<GSplat> Splats : register(t0);
StructuredBuffer<uint>   Order  : register(t1);

cbuffer Constants : register(b0) {
    row_major float4x4 ViewProj;   // see the point shader on row_major
    float4 Right;                  // camera basis in world space
    float4 Up;
    float4 Fwd;
    float4 EyeNear;                // eye position, near plane distance
    float2 Viewport;               // width, height in pixels
    float2 Focal;                  // projection's x and y scale, NDC per unit
    float  Scale;                  // user size multiplier
    float3 Pad;
};

struct VSOut {
    float4 pos    : SV_Position;
    float4 colour : COLOR;
    float3 conic  : TEXCOORD0;     // inverse 2D covariance: a, b, c
    float2 d      : TEXCOORD1;     // offset from the centre, pixels
};

VSOut main(uint vid : SV_VertexID) {
    GSplat s = Splats[Order[vid / 6]];

    uint corner = vid % 6;
    const uint2 lut[6] = { uint2(0,0), uint2(1,0), uint2(0,1),
                           uint2(0,1), uint2(1,0), uint2(1,1) };
    float2 c = float2(lut[corner]) * 2.0 - 1.0;

    VSOut o;
    o.colour = float4(s.colour.rgb, s.meanOp.w);
    o.conic  = float3(0, 0, 0);
    o.d      = float2(0, 0);
    // A position with z < 0 is clipped: used to drop a splat entirely.
    o.pos    = float4(0, 0, -1, 1);

    // Into the camera frame.
    float3 dw = s.meanOp.xyz - EyeNear.xyz;
    float3 t  = float3(dot(dw, Right.xyz), dot(dw, Up.xyz), dot(dw, Fwd.xyz));
    if (t.z <= EyeNear.w) return o;   // behind or at the near plane

    float3x3 S = { s.covA.x, s.covA.y, s.covA.z,
                   s.covA.y, s.covA.w, s.covB.x,
                   s.covA.z, s.covB.x, s.covB.y };
    float3x3 W = { Right.xyz, Up.xyz, Fwd.xyz };
    float3x3 Sc = mul(W, mul(S, transpose(W)));

    // The Jacobian of (x/z, y/z), in PIXELS. Focal is the projection's NDC
    // scale, so half the viewport turns it into pixels -- and with the
    // aspect already folded into Focal.x, both axes come out square.
    float fx = Focal.x * Viewport.x * 0.5;
    float fy = Focal.y * Viewport.y * 0.5;
    float iz = 1.0 / t.z;
    float2x3 J = { fx * iz, 0.0,     -fx * t.x * iz * iz,
                   0.0,     fy * iz, -fy * t.y * iz * iz };
    float2x2 cov = mul(J, mul(Sc, transpose(J))) * (Scale * Scale);

    // A LOW-PASS OF A THIRD OF A PIXEL, as the paper does. A Gaussian far
    // away or seen edge-on projects smaller than a pixel, and evaluating it
    // at pixel centres then aliases into sparkle. Adding a small isotropic
    // term guarantees every splat covers at least about one pixel.
    cov[0][0] += 0.3;
    cov[1][1] += 0.3;

    float a = cov[0][0], b = cov[0][1], cc = cov[1][1];
    float det = a * cc - b * b;
    if (det <= 1e-12) return o;

    // The ellipse's axes: eigenvectors of the 2x2 covariance.
    float mid  = 0.5 * (a + cc);
    float disc = sqrt(max(0.0, mid * mid - det));
    float l1 = mid + disc;
    float l2 = max(1e-4, mid - disc);
    float2 v1 = abs(b) > 1e-12 ? normalize(float2(b, l1 - a))
                               : (a >= cc ? float2(1, 0) : float2(0, 1));
    float2 v2 = float2(-v1.y, v1.x);

    // Three standard deviations, capped so a Gaussian that projects huge --
    // one just in front of the camera -- cannot become a quad covering the
    // whole screen hundreds of times over.
    float r1 = min(3.0 * sqrt(l1), 1024.0);
    float r2 = min(3.0 * sqrt(l2), 1024.0);
    float2 off = c.x * r1 * v1 + c.y * r2 * v2;

    float4 clip = mul(float4(s.meanOp.xyz, 1.0), ViewProj);
    clip.xy += off / (Viewport * 0.5) * clip.w;

    o.pos   = clip;
    o.conic = float3(cc, -b, a) / det;
    o.d     = off;
    return o;
}
)";

// The Gaussian itself, evaluated per pixel against the conic.
constexpr const char* kSplatPixelShader = R"(
struct VSOut {
    float4 pos    : SV_Position;
    float4 colour : COLOR;
    float3 conic  : TEXCOORD0;
    float2 d      : TEXCOORD1;
};

float4 main(VSOut i) : SV_Target {
    float power = -0.5 * (i.conic.x * i.d.x * i.d.x + i.conic.z * i.d.y * i.d.y)
                  - i.conic.y * i.d.x * i.d.y;
    if (power > 0.0) discard;
    // Capped just below one, as the paper does, so no single splat can make
    // everything behind it vanish completely.
    float alpha = min(0.99, i.colour.a * exp(power));
    if (alpha < 1.0 / 255.0) discard;
    return float4(i.colour.rgb, alpha);
}
)";

struct SplatConstants {
    float viewProj[16];
    float right[4], up[4], fwd[4], eyeNear[4];
    float viewport[2];
    float focal[2];
    float scale;
    float pad[3];
};

// A buffer in the UPLOAD heap, filled from `data`. Released and replaced
// rather than rewritten, because the GPU may still be reading the previous
// frame's copy -- see DeferRelease.
ID3D12Resource* MakeUploadBuffer(Device& dev, const void* data, size_t bytes) {
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_UPLOAD;

    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = UINT64(std::max<size_t>(bytes, 16));
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = DXGI_FORMAT_UNKNOWN;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    ID3D12Resource* res = nullptr;
    if (FAILED(dev.Get()->CreateCommittedResource(
            &hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr, IID_PPV_ARGS(&res))))
        return nullptr;

    void* dst = nullptr;
    D3D12_RANGE none = {0, 0};
    if (FAILED(res->Map(0, &none, &dst))) { res->Release(); return nullptr; }
    std::memcpy(dst, data, bytes);
    res->Unmap(0, nullptr);
    return res;
}

// A float mapped to an unsigned key that sorts in the same order. Flip the
// sign bit of positives and every bit of negatives, the standard trick that
// makes IEEE floats radix-sortable.
inline uint32_t SortableFloat(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, 4);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

}  // namespace

Viewport3D::~Viewport3D() {
    if (m_dev) ReleaseGpu(*m_dev);
}

void Viewport3D::ReleaseGpu(Device& dev) {
    if (m_points) { dev.DeferRelease(m_points); m_points = nullptr; }
    if (m_pso)  { m_pso->Release();  m_pso = nullptr; }
    if (m_root) { m_root->Release(); m_root = nullptr; }
    m_pointCount = 0;

    if (m_splatBuf) { dev.DeferRelease(m_splatBuf); m_splatBuf = nullptr; }
    if (m_orderBuf) { dev.DeferRelease(m_orderBuf); m_orderBuf = nullptr; }
    if (m_splatPso)  { m_splatPso->Release();  m_splatPso = nullptr; }
    if (m_splatRoot) { m_splatRoot->Release(); m_splatRoot = nullptr; }
    m_splatCount = 0;
}

void Viewport3D::SetPointCloud(std::shared_ptr<const PointCloud> c) {
    m_cloud = std::move(c);
}

bool Viewport3D::EnsurePipeline(Device& dev) {
    if (m_pso) return true;
    m_dev = &dev;

    ShaderCompiler sc;
    if (!sc.Init()) return false;

    ShaderBlob vs, ps;
    std::string err;
    if (!sc.CompileCompute(kVertexShader, "main", "viewport3d.vs", &vs, &err,
                           "vs_6_0")) {
        std::fprintf(stderr, "[3d] vertex shader: %s\n", err.c_str());
        sc.Shutdown();
        return false;
    }
    if (!sc.CompileCompute(kPixelShader, "main", "viewport3d.ps", &ps, &err,
                           "ps_6_0")) {
        std::fprintf(stderr, "[3d] pixel shader: %s\n", err.c_str());
        sc.Shutdown();
        return false;
    }
    sc.Shutdown();

    // Root signature: the constants inline, and the point buffer as a plain
    // SRV descriptor rather than a table.
    //
    // A ROOT SRV rather than a descriptor table, because a root SRV takes a
    // GPU virtual address directly -- no descriptor, no heap slot, and
    // therefore no interaction with the one shader-visible heap ImGui has
    // bound. That is the same constraint the display path works around, and
    // sidestepping it entirely is simpler than sharing.
    D3D12_ROOT_PARAMETER params[2] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.Num32BitValues = sizeof(Constants) / 4;
    params[0].Constants.ShaderRegister = 0;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;

    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[1].Descriptor.ShaderRegister = 0;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;

    D3D12_ROOT_SIGNATURE_DESC rs = {};
    rs.NumParameters = 2;
    rs.pParameters = params;
    rs.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ID3DBlob* sig = nullptr;
    ID3DBlob* sigErr = nullptr;
    if (FAILED(D3D12SerializeRootSignature(&rs, D3D_ROOT_SIGNATURE_VERSION_1,
                                           &sig, &sigErr))) {
        if (sigErr) {
            std::fprintf(stderr, "[3d] root signature: %.*s\n",
                         int(sigErr->GetBufferSize()),
                         static_cast<const char*>(sigErr->GetBufferPointer()));
            sigErr->Release();
        }
        if (sig) sig->Release();
        return false;
    }
    const HRESULT hr = dev.Get()->CreateRootSignature(
        0, sig->GetBufferPointer(), sig->GetBufferSize(), IID_PPV_ARGS(&m_root));
    sig->Release();
    if (sigErr) sigErr->Release();
    if (FAILED(hr)) return false;

    D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = {};
    pd.pRootSignature = m_root;
    pd.VS = {vs.dxil.data(), vs.dxil.size()};
    pd.PS = {ps.dxil.data(), ps.dxil.size()};
    pd.SampleMask = UINT_MAX;
    pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pd.NumRenderTargets = 1;
    pd.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    pd.DSVFormat = DXGI_FORMAT_D32_FLOAT;
    pd.SampleDesc.Count = 1;

    pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    // No culling: the quads are built facing the viewer, and a winding mistake
    // would make points vanish from one side of the orbit only -- a confusing
    // symptom for no benefit.
    pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pd.RasterizerState.DepthClipEnable = TRUE;

    // DEPTH TESTING ON, which is the whole reason for a private pipeline:
    // ImGui's disables it, and a cloud drawn in buffer order shows its far
    // side through its near side.
    pd.DepthStencilState.DepthEnable = TRUE;
    pd.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    pd.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;

    D3D12_RENDER_TARGET_BLEND_DESC& rt = pd.BlendState.RenderTarget[0];
    rt.BlendEnable = TRUE;
    rt.SrcBlend = D3D12_BLEND_SRC_ALPHA;
    rt.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
    rt.BlendOp = D3D12_BLEND_OP_ADD;
    rt.SrcBlendAlpha = D3D12_BLEND_ONE;
    rt.DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
    rt.BlendOpAlpha = D3D12_BLEND_OP_ADD;
    rt.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;

    if (FAILED(dev.Get()->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&m_pso)))) {
        std::fprintf(stderr, "[3d] could not create the point pipeline state\n");
        m_root->Release();
        m_root = nullptr;
        return false;
    }
    return true;
}

bool Viewport3D::EnsureSplatPipeline(Device& dev) {
    if (m_splatPso) return true;
    m_dev = &dev;

    ShaderCompiler sc;
    if (!sc.Init()) return false;
    ShaderBlob vs, ps;
    std::string err;
    if (!sc.CompileCompute(kSplatVertexShader, "main", "splat.vs", &vs, &err,
                           "vs_6_0")) {
        std::fprintf(stderr, "[3d] splat vertex shader: %s\n", err.c_str());
        sc.Shutdown();
        return false;
    }
    if (!sc.CompileCompute(kSplatPixelShader, "main", "splat.ps", &ps, &err,
                           "ps_6_0")) {
        std::fprintf(stderr, "[3d] splat pixel shader: %s\n", err.c_str());
        sc.Shutdown();
        return false;
    }
    sc.Shutdown();

    // Constants, then the splats and the draw order as root SRVs -- the same
    // reason as the point pipeline: no descriptor heap, so nothing to share
    // with ImGui's.
    D3D12_ROOT_PARAMETER params[3] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.Num32BitValues = sizeof(SplatConstants) / 4;
    params[0].Constants.ShaderRegister = 0;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[1].Descriptor.ShaderRegister = 0;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[2].Descriptor.ShaderRegister = 1;
    params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;

    D3D12_ROOT_SIGNATURE_DESC rs = {};
    rs.NumParameters = 3;
    rs.pParameters = params;

    ID3DBlob* sig = nullptr;
    ID3DBlob* sigErr = nullptr;
    if (FAILED(D3D12SerializeRootSignature(&rs, D3D_ROOT_SIGNATURE_VERSION_1,
                                           &sig, &sigErr))) {
        if (sigErr) {
            std::fprintf(stderr, "[3d] splat root signature: %.*s\n",
                         int(sigErr->GetBufferSize()),
                         static_cast<const char*>(sigErr->GetBufferPointer()));
            sigErr->Release();
        }
        if (sig) sig->Release();
        return false;
    }
    const HRESULT hr = dev.Get()->CreateRootSignature(
        0, sig->GetBufferPointer(), sig->GetBufferSize(),
        IID_PPV_ARGS(&m_splatRoot));
    sig->Release();
    if (sigErr) sigErr->Release();
    if (FAILED(hr)) return false;

    D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = {};
    pd.pRootSignature = m_splatRoot;
    pd.VS = {vs.dxil.data(), vs.dxil.size()};
    pd.PS = {ps.dxil.data(), ps.dxil.size()};
    pd.SampleMask = UINT_MAX;
    pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pd.NumRenderTargets = 1;
    pd.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    pd.DSVFormat = DXGI_FORMAT_D32_FLOAT;
    pd.SampleDesc.Count = 1;
    pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pd.RasterizerState.DepthClipEnable = TRUE;

    // DEPTH TESTING OFF. Translucent things composite correctly only in
    // order, and the order is what the sort provides; a depth test would
    // discard the soft far edge of every Gaussian behind a nearer one, which
    // is exactly the blending that makes splats look continuous.
    pd.DepthStencilState.DepthEnable = FALSE;
    pd.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;

    // "Over", back to front: each splat covers what is behind it by its alpha.
    D3D12_RENDER_TARGET_BLEND_DESC& rt = pd.BlendState.RenderTarget[0];
    rt.BlendEnable = TRUE;
    rt.SrcBlend = D3D12_BLEND_SRC_ALPHA;
    rt.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
    rt.BlendOp = D3D12_BLEND_OP_ADD;
    rt.SrcBlendAlpha = D3D12_BLEND_ONE;
    rt.DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
    rt.BlendOpAlpha = D3D12_BLEND_OP_ADD;
    rt.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;

    if (FAILED(dev.Get()->CreateGraphicsPipelineState(
            &pd, IID_PPV_ARGS(&m_splatPso)))) {
        std::fprintf(stderr, "[3d] could not create the splat pipeline state\n");
        m_splatRoot->Release();
        m_splatRoot = nullptr;
        return false;
    }
    return true;
}

bool Viewport3D::UploadSplats(Device& dev) {
    if (!m_cloud || m_cloud->splats.empty()) return false;
    if (m_splatBuf && m_splatVersion == m_version) return m_splatCount > 0;

    const std::vector<Splat>& src = m_cloud->splats;
    std::vector<GpuSplat> g(src.size());
    m_splatMeans.resize(src.size() * 3);
    for (size_t i = 0; i < src.size(); ++i) {
        const Splat& s = src[i];
        double c[6];
        s.Covariance(c);
        GpuSplat& o = g[i];
        o.mx = float(s.mean.x); o.my = float(s.mean.y); o.mz = float(s.mean.z);
        o.opacity = float(s.opacity);
        o.cxx = float(c[0]); o.cxy = float(c[1]); o.cxz = float(c[2]);
        o.cyy = float(c[3]); o.cyz = float(c[4]); o.czz = float(c[5]);
        o.pad0 = o.pad1 = o.pad2 = 0.0f;
        o.r = float(s.color.x); o.g = float(s.color.y); o.b = float(s.color.z);
        m_splatMeans[i * 3 + 0] = o.mx;
        m_splatMeans[i * 3 + 1] = o.my;
        m_splatMeans[i * 3 + 2] = o.mz;
    }

    if (m_splatBuf) { dev.DeferRelease(m_splatBuf); m_splatBuf = nullptr; }
    m_splatBuf = MakeUploadBuffer(dev, g.data(), g.size() * sizeof(GpuSplat));
    if (!m_splatBuf) { m_splatCount = 0; return false; }

    m_splatCount = int(src.size());
    m_splatVersion = m_version;
    // New splats need a new order even if the camera has not moved.
    m_sortedEye = Vec3{1e30, 1e30, 1e30};
    m_dev = &dev;
    return true;
}

// Back-to-front draw order for the current camera.
//
// ON THE CPU, and only when the camera has moved. A radix sort on 32-bit depth
// keys is linear in the splat count -- a few tens of milliseconds for a
// million -- which keeps orbiting interactive without a GPU sort. The GPU
// sort is the obvious next step if it stops being enough, and the reason it
// is not the first step is this machine's history with multi-dispatch GPU
// work.
bool Viewport3D::SortSplats(Device& dev, const OrbitCamera& cam) {
    const Vec3 eye = cam.Eye();
    if (m_orderBuf && (eye - m_sortedEye).Norm() < 1e-9 &&
        (cam.target - m_sortedTarget).Norm() < 1e-9)
        return true;

    const auto t0 = std::chrono::steady_clock::now();
    Vec3 right, up, fwd;
    cam.Basis(&right, &up, &fwd);

    const size_t n = size_t(m_splatCount);
    m_keys.resize(n);
    m_order.resize(n);
    m_keysTmp.resize(n);
    m_orderTmp.resize(n);
    const float ex = float(eye.x), ey = float(eye.y), ez = float(eye.z);
    const float fx = float(fwd.x), fy = float(fwd.y), fz = float(fwd.z);
    for (size_t i = 0; i < n; ++i) {
        const float d = (m_splatMeans[i * 3 + 0] - ex) * fx +
                        (m_splatMeans[i * 3 + 1] - ey) * fy +
                        (m_splatMeans[i * 3 + 2] - ez) * fz;
        // Inverted, so ASCENDING keys are DESCENDING depth: farthest first.
        m_keys[i] = ~SortableFloat(d);
        m_order[i] = uint32_t(i);
    }

    // LSD radix sort, four passes of eight bits.
    for (int shift = 0; shift < 32; shift += 8) {
        size_t count[257] = {};
        for (size_t i = 0; i < n; ++i) ++count[((m_keys[i] >> shift) & 0xFF) + 1];
        for (int b = 0; b < 256; ++b) count[b + 1] += count[b];
        for (size_t i = 0; i < n; ++i) {
            const size_t dst = count[(m_keys[i] >> shift) & 0xFF]++;
            m_keysTmp[dst] = m_keys[i];
            m_orderTmp[dst] = m_order[i];
        }
        m_keys.swap(m_keysTmp);
        m_order.swap(m_orderTmp);
    }

    if (m_orderBuf) { dev.DeferRelease(m_orderBuf); m_orderBuf = nullptr; }
    m_orderBuf = MakeUploadBuffer(dev, m_order.data(), n * sizeof(uint32_t));
    if (!m_orderBuf) return false;

    m_sortedEye = eye;
    m_sortedTarget = cam.target;
    m_sortMs = std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t0).count();
    return true;
}

bool Viewport3D::UploadGeometry(Device& dev) {
    if (!m_cloud) return false;
    // THE CACHE KEY IS EVERYTHING THE BUFFER DEPENDS ON, not just the content
    // version. `m_showCameras` changes which points are written, so a rebuild
    // keyed on the version alone left the cameras baked in and the checkbox
    // did nothing -- it flipped a bool that no longer reached the geometry.
    const bool splatMode = DrawingSplats();
    if (m_points && m_uploadedVersion == m_version &&
        m_uploadedCameras == m_showCameras && m_uploadedSplatMode == splatMode) {
        return m_pointCount > 0;
    }

    std::vector<GpuPoint> pts;
    pts.reserve(m_cloud->tracks.size() + m_cloud->cameras.size() * 2);

    // When the splats are drawn they ARE the structure; the dots would only
    // be drawn underneath them. The cameras still come through this buffer.
    for (const Track& t : m_cloud->tracks) {
        if (splatMode) break;
        if (!t.hasPoint) continue;
        GpuPoint g;
        g.x = float(t.point.x); g.y = float(t.point.y); g.z = float(t.point.z);
        g.size = 1.0f;
        // Track colour when there is one, otherwise a neutral grey. A cloud
        // with no colour is the normal case until the sampler that reads it
        // off the source frames exists.
        // Exactly zero means unset; see Track::color. A dark point sampled
        // from real pixels is never all three channels at exactly 0.0.
        const bool haveColour =
            t.color.x != 0.0 || t.color.y != 0.0 || t.color.z != 0.0;
        g.r = haveColour ? float(t.color.x) : 0.78f;
        g.g = haveColour ? float(t.color.y) : 0.80f;
        g.b = haveColour ? float(t.color.z) : 0.84f;
        g.a = 1.0f;
        pts.push_back(g);
    }

    // The cameras, drawn larger and in a colour nothing else uses.
    //
    // Worth showing even though they are not structure: the commonest way a
    // reconstruction is wrong is a camera in the wrong place, and a trajectory
    // that doubles back or flings one frame away from the rest says so
    // instantly where a point cloud alone would not.
    if (m_showCameras) {
        for (const Camera& c : m_cloud->cameras) {
            if (!c.solved) continue;
            const Vec3 e = c.Center();
            GpuPoint g;
            g.x = float(e.x); g.y = float(e.y); g.z = float(e.z);
            g.size = 3.0f;
            g.r = 1.0f; g.g = 0.55f; g.b = 0.15f; g.a = 1.0f;
            pts.push_back(g);
        }
    }

    if (pts.empty()) { m_pointCount = 0; return false; }

    const size_t bytes = pts.size() * sizeof(GpuPoint);
    if (m_points) { dev.DeferRelease(m_points); m_points = nullptr; }

    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_UPLOAD;

    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = UINT64(bytes);
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = DXGI_FORMAT_UNKNOWN;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    if (FAILED(dev.Get()->CreateCommittedResource(
            &hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr, IID_PPV_ARGS(&m_points)))) {
        m_pointCount = 0;
        return false;
    }

    void* dst = nullptr;
    D3D12_RANGE none = {0, 0};
    if (FAILED(m_points->Map(0, &none, &dst))) { m_pointCount = 0; return false; }
    std::memcpy(dst, pts.data(), bytes);
    m_points->Unmap(0, nullptr);

    m_pointCount = int(pts.size());
    m_uploadedVersion = m_version;
    m_uploadedCameras = m_showCameras;
    m_uploadedSplatMode = splatMode;
    m_dev = &dev;
    return true;
}

void Viewport3D::Draw(Device& dev, Image*) {
    if (!ImGui::Begin(m_name.c_str())) {
        m_visible = false;
        ImGui::End();
        return;
    }
    m_visible = true;

    OrbitCamera& cam = m_shared ? *m_shared : m_own;

    // Nothing yet. A cloud with Gaussians but no points -- one imported from
    // a .ply -- is ready, not computing.
    if (!m_cloud || (m_cloud->tracks.empty() && m_cloud->splats.empty())) {
        ImGui::TextDisabled("computing...");
        ImGui::End();
        return;
    }

    // --- toolbar ------------------------------------------------------------
    const bool hasSplats = !m_cloud->splats.empty();
    if (hasSplats && m_showSplats)
        ImGui::Text("%d splats, %d cameras  (sort %.0f ms)",
                    int(m_cloud->splats.size()), m_cloud->SolvedCameras(),
                    m_sortMs);
    else
        ImGui::Text("%d points, %d cameras", m_cloud->TriangulatedPoints(),
                    m_cloud->SolvedCameras());
    ImGui::SameLine();
    if (ImGui::SmallButton("Frame")) m_framed = false;
    ImGui::SameLine();
    ImGui::Checkbox("cameras", &m_showCameras);
    if (hasSplats) {
        ImGui::SameLine();
        ImGui::Checkbox("splats", &m_showSplats);
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(90.0f);
    if (hasSplats && m_showSplats)
        ImGui::SliderFloat("scale", &m_splatScale, 0.1f, 3.0f, "%.2f");
    else
        ImGui::SliderFloat("size", &m_pointSize, 1.0f, 12.0f, "%.1f");

    // FRAMED ON THE POINTS, not on the cameras. A reconstruction's scale is a
    // gauge freedom, so there is no fixed distance that works -- and a stray
    // camera flung far from the rest would otherwise set the extent and shrink
    // the actual structure to nothing.
    // PERCENTILES RATHER THAN MIN AND MAX, because a reconstruction always has
    // outliers and a bounding box is decided entirely by them.
    //
    // A single point triangulated a hundred times too far away sets the extent,
    // so the camera backs off to fit something nobody wants to see and aims at
    // the midpoint of a box that is mostly empty. The structure then sits in a
    // corner of the view and orbiting swings it around a pivot that is nowhere
    // near it -- which is exactly how this presented.
    //
    // The 2nd and 98th percentiles per axis instead: the framing follows where
    // the points ARE. An outlier is still drawn, it just stops dictating the
    // view.
    if (!m_framed) {
        std::vector<double> xs, ys, zs;
        xs.reserve(m_cloud->tracks.size());
        for (const Track& t : m_cloud->tracks) {
            if (!t.hasPoint) continue;
            xs.push_back(t.point.x);
            ys.push_back(t.point.y);
            zs.push_back(t.point.z);
        }
        // No points -- an imported .ply of Gaussians -- so frame on those.
        if (xs.empty())
            for (const Splat& s : m_cloud->splats) {
                xs.push_back(s.mean.x);
                ys.push_back(s.mean.y);
                zs.push_back(s.mean.z);
            }
        if (!xs.empty()) {
            auto pct = [](std::vector<double>& v, double p) {
                const size_t i = size_t(p * double(v.size() - 1));
                std::nth_element(v.begin(), v.begin() + long(i), v.end());
                return v[i];
            };
            // Order matters: nth_element partially sorts, so the low
            // percentile is taken first and the high one searched after.
            const Vec3 lo{pct(xs, 0.02), pct(ys, 0.02), pct(zs, 0.02)};
            const Vec3 hi{pct(xs, 0.98), pct(ys, 0.98), pct(zs, 0.98)};
            cam.Frame(lo, hi);
            m_framed = true;
        }
    }

    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const int w = int(std::max(16.0f, avail.x));
    const int h = int(std::max(16.0f, avail.y));

    if (!EnsurePipeline(dev) || !m_target.Ensure(dev, w, h)) {
        ImGui::TextDisabled("could not create the 3D pipeline");
        ImGui::End();
        return;
    }
    // Either may legitimately be empty -- splats with the cameras hidden
    // leave no dots at all -- but not both.
    const bool havePoints = UploadGeometry(dev);
    const bool haveSplats = DrawingSplats() && EnsureSplatPipeline(dev) &&
                            UploadSplats(dev) && SortSplats(dev, cam);
    if (!havePoints && !haveSplats) {
        ImGui::TextDisabled(DrawingSplats() ? "could not create the splat pipeline"
                                            : "nothing to draw");
        ImGui::End();
        return;
    }

    // --- render --------------------------------------------------------------
    ID3D12GraphicsCommandList* cl = dev.CurrentCommandList();
    if (cl) {
        const float clear[4] = {0.08f, 0.09f, 0.11f, 1.0f};
        if (m_target.Begin(cl, clear)) {
            Constants k = {};
            cam.ViewProj(k.viewProj);
            // The projection above assumes a square aspect; correcting x here
            // keeps the camera's own maths aspect-free and testable.
            const float aspect = float(w) / float(std::max(1, h));
            for (int i = 0; i < 4; ++i) k.viewProj[i * 4 + 0] /= aspect;
            k.invViewport[0] = 1.0f / float(w);
            k.invViewport[1] = 1.0f / float(h);
            k.sizeScale = m_pointSize;

            cl->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

            // Splats first: with depth testing off they would paint over
            // anything drawn before them, and the camera markers should sit
            // on top.
            if (haveSplats) {
                SplatConstants sk = {};
                std::memcpy(sk.viewProj, k.viewProj, sizeof(sk.viewProj));
                Vec3 right, up, fwd;
                cam.Basis(&right, &up, &fwd);
                const Vec3 eye = cam.Eye();
                sk.right[0] = float(right.x); sk.right[1] = float(right.y); sk.right[2] = float(right.z);
                sk.up[0]    = float(up.x);    sk.up[1]    = float(up.y);    sk.up[2]    = float(up.z);
                sk.fwd[0]   = float(fwd.x);   sk.fwd[1]   = float(fwd.y);   sk.fwd[2]   = float(fwd.z);
                sk.eyeNear[0] = float(eye.x); sk.eyeNear[1] = float(eye.y); sk.eyeNear[2] = float(eye.z);
                sk.eyeNear[3] = float(cam.nearZ);
                sk.viewport[0] = float(w);
                sk.viewport[1] = float(h);
                // The same scale factors ViewProj applies to x and y, the
                // aspect correction included, so the ellipses and the centres
                // they are drawn around come from one projection.
                const float f = float(1.0 / std::tan(cam.fovY * 0.5));
                sk.focal[0] = f / aspect;
                sk.focal[1] = f;
                sk.scale = m_splatScale;

                cl->SetPipelineState(m_splatPso);
                cl->SetGraphicsRootSignature(m_splatRoot);
                cl->SetGraphicsRoot32BitConstants(
                    0, sizeof(SplatConstants) / 4, &sk, 0);
                cl->SetGraphicsRootShaderResourceView(
                    1, m_splatBuf->GetGPUVirtualAddress());
                cl->SetGraphicsRootShaderResourceView(
                    2, m_orderBuf->GetGPUVirtualAddress());
                cl->DrawInstanced(UINT(m_splatCount) * 6, 1, 0, 0);
            }

            if (havePoints && m_pointCount > 0) {
                cl->SetPipelineState(m_pso);
                cl->SetGraphicsRootSignature(m_root);
                cl->SetGraphicsRoot32BitConstants(0, sizeof(Constants) / 4, &k, 0);
                cl->SetGraphicsRootShaderResourceView(
                    1, m_points->GetGPUVirtualAddress());
                cl->DrawInstanced(UINT(m_pointCount) * 6, 1, 0, 0);
            }

            m_target.End(cl);

            // BACK TO THE SCREEN. Not optional, and not just the heap.
            //
            // Begin() above called OMSetRenderTargets to bind this panel's
            // offscreen texture, and a render target stays bound until
            // something changes it. BeginFrame() binds the back buffer ONCE
            // per frame, so without this every draw that follows -- including
            // all of ImGui, which draws last -- lands in this panel's texture
            // instead of on screen.
            //
            // The symptom is the whole window going black the moment a 3D
            // panel first renders: no error, no validation message, just an
            // application drawing itself into a texture nobody displays. The
            // first version restored only the descriptor heap, which fixed
            // nothing because the heap was never the problem.
            dev.RestoreBackBuffer();
        }
    }

    // The rendered target, drawn as an ordinary image. From here the panel is
    // indistinguishable from an image viewer, which is what makes it dock and
    // resize like one.
    if (m_target.Valid()) {
        ImGui::GetWindowDrawList()->AddImage(
            ImTextureRef(static_cast<ImTextureID>(m_target.Srv().ptr)), origin,
            ImVec2(origin.x + float(w), origin.y + float(h)));
        ImGui::InvisibleButton("##canvas", ImVec2(float(w), float(h)));

        // --- camera control, mirroring the 2D panels ------------------------
        if (ImGui::IsItemHovered()) {
            const ImGuiIO& io = ImGui::GetIO();
            if (io.MouseWheel != 0.0f) cam.Dolly(double(io.MouseWheel));
        }
        if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
            const ImVec2 d = ImGui::GetIO().MouseDelta;
            if (ImGui::GetIO().KeyShift) cam.Pan(double(d.x), double(d.y));
            else cam.Rotate(double(d.x) * 0.008, double(-d.y) * 0.008);
        }
    }

    ImGui::End();
}

}  // namespace tglab
