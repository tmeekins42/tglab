#include "viewport3d.h"

#include <algorithm>
#include <cstdio>

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

}  // namespace

Viewport3D::~Viewport3D() {
    if (m_dev) ReleaseGpu(*m_dev);
}

void Viewport3D::ReleaseGpu(Device& dev) {
    if (m_points) { dev.DeferRelease(m_points); m_points = nullptr; }
    if (m_pso)  { m_pso->Release();  m_pso = nullptr; }
    if (m_root) { m_root->Release(); m_root = nullptr; }
    m_pointCount = 0;
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

bool Viewport3D::UploadGeometry(Device& dev) {
    if (!m_cloud) return false;
    if (m_points && m_uploadedVersion == m_version) return m_pointCount > 0;

    std::vector<GpuPoint> pts;
    pts.reserve(m_cloud->tracks.size() + m_cloud->cameras.size() * 2);

    for (const Track& t : m_cloud->tracks) {
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

    if (!m_cloud || m_cloud->tracks.empty()) {
        ImGui::TextDisabled("computing...");
        ImGui::End();
        return;
    }

    // --- toolbar ------------------------------------------------------------
    ImGui::Text("%d points, %d cameras", m_cloud->TriangulatedPoints(),
                m_cloud->SolvedCameras());
    ImGui::SameLine();
    if (ImGui::SmallButton("Frame")) m_framed = false;
    ImGui::SameLine();
    ImGui::Checkbox("cameras", &m_showCameras);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(90.0f);
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

    if (!EnsurePipeline(dev) || !m_target.Ensure(dev, w, h) ||
        !UploadGeometry(dev)) {
        ImGui::TextDisabled("could not create the 3D pipeline");
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

            cl->SetPipelineState(m_pso);
            cl->SetGraphicsRootSignature(m_root);
            cl->SetGraphicsRoot32BitConstants(0, sizeof(Constants) / 4, &k, 0);
            cl->SetGraphicsRootShaderResourceView(
                1, m_points->GetGPUVirtualAddress());
            cl->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            cl->DrawInstanced(UINT(m_pointCount) * 6, 1, 0, 0);

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
