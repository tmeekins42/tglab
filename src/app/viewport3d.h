// Viewport3D — a dockable panel showing a reconstruction.
//
// The counterpart to ImageViewPanel, and it docks beside one: a script can
// display the source frames and the point cloud they produced side by side,
// which is what makes a wrong reconstruction obvious rather than merely
// suspected.
//
// HOW IT COMPOSITES WITH ImGui. tglab's own GPU code is compute-only, but
// ImGui's DX12 backend is a full graphics pipeline and the image viewers draw
// through it. What that pipeline cannot do is what a point cloud needs: it is
// triangle-list, depth testing disabled, with a fixed 2D shader. So this owns
// its own pipeline state, renders into an offscreen RenderTarget, and hands
// the result's SRV to ImGui::Image(). As far as ImGui is concerned the panel
// contains a picture, so docking, resizing and z-order all work unchanged.
//
// POINTS AS QUADS rather than D3D's point primitive. A point-list primitive
// rasterises exactly one pixel regardless of distance, which makes a sparse
// cloud nearly invisible and gives no sense of depth. Two triangles per point,
// sized in screen space, draw a square dot that can be made bigger or smaller
// and that shades with distance. The vertex shader expands them from a
// structured buffer, so there is no per-point vertex data to upload beyond the
// positions themselves.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../core/data.h"
#include "../gpu/render_target.h"
#include "orbit_camera.h"
#include "view.h"

struct ID3D12PipelineState;
struct ID3D12RootSignature;
struct ID3D12Resource;

namespace tglab {

class Viewport3D : public View {
public:
    explicit Viewport3D(std::string name) : m_name(std::move(name)) {}
    ~Viewport3D() override;

    const std::string& Name() const override { return m_name; }

    void Draw(Device& dev, Image* image) override;
    void SetContentVersion(uint64_t v) override { m_version = v; }
    void SetPointCloud(std::shared_ptr<const PointCloud> c) override;

    // Shared between 3D panels, the way ViewCamera is shared between image
    // panels: comparing two reconstructions means turning them together.
    void ShareCamera(OrbitCamera* cam) { m_shared = cam; }

    bool Visible() const { return m_visible; }

private:
    bool EnsurePipeline(Device& dev);
    bool UploadGeometry(Device& dev);
    void ReleaseGpu(Device& dev);

    std::string m_name;
    uint64_t    m_version = 0;

    std::shared_ptr<const PointCloud> m_cloud;
    uint64_t m_uploadedVersion = 0;   // which version the buffers hold

    RenderTarget m_target;
    OrbitCamera  m_own;
    OrbitCamera* m_shared = nullptr;

    // Whether the camera has been framed on this cloud. A reconstruction's
    // scale is a gauge freedom, so opening at a fixed distance would show
    // nothing for a scene measured in millimetres and nothing for one measured
    // in kilometres.
    bool m_framed = false;

    ID3D12RootSignature* m_root = nullptr;
    ID3D12PipelineState* m_pso  = nullptr;

    // Point positions and colours, as a structured buffer the vertex shader
    // indexes. UPLOAD heap rather than DEFAULT: a cloud changes only when the
    // pipeline re-runs, the sizes are modest, and a staging copy would need a
    // command list and a fence for no gain.
    ID3D12Resource* m_points = nullptr;
    int             m_pointCount = 0;
    Device*         m_dev = nullptr;   // for deferred release

    bool m_visible = false;
    bool m_showCameras = true;
    float m_pointSize = 3.0f;
};

}  // namespace tglab
