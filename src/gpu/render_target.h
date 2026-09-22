// RenderTarget — an offscreen surface a 3D view draws into.
//
// WHY THIS EXISTS, given that ImGui already rasterises.
//
// tglab's own GPU code is compute-only, but ImGui's DX12 backend is a full
// graphics pipeline and the image viewers already draw through it
// (`dl->AddImage()`). What is missing is not a rasteriser -- it is a PIPELINE
// STATE WE CONTROL. ImGui's is triangle-list, depth testing DISABLED, with a
// fixed 2D shader that transforms by an orthographic matrix. All three are
// wrong for a point cloud: points are not triangles, a cloud without depth
// testing draws back-to-front in arbitrary order, and a camera orbit needs a
// real view-projection matrix.
//
// So a 3D view renders into one of these with its own pipeline state, and then
// hands the result's SRV to `ImGui::Image()` like any other texture. The panel
// docks, resizes and z-orders exactly as an image viewer does, because as far
// as ImGui is concerned that is what it is.
//
// THE DEPTH BUFFER IS THE POINT. Without it a point cloud is drawn in whatever
// order the vertex buffer happens to hold, so the far side of an object shows
// through the near side and the shape is unreadable. It is not an optimisation
// here; it is what makes the image mean anything.
#pragma once

#include <d3d12.h>

#include <cstdint>

namespace tglab {

class Device;

class RenderTarget {
public:
    ~RenderTarget() { Release(); }

    RenderTarget() = default;
    RenderTarget(const RenderTarget&) = delete;
    RenderTarget& operator=(const RenderTarget&) = delete;

    // Creates or resizes to `w` x `h`. Cheap and idempotent when the size is
    // unchanged, so a view can call it every frame with its panel size -- which
    // is what makes a docked 3D panel follow its splitter.
    bool Ensure(Device& dev, int w, int h);

    // Transitions to RENDER_TARGET, binds, and clears colour and depth.
    // Returns false when there is nothing to draw into.
    bool Begin(ID3D12GraphicsCommandList* cl, const float clear[4]);

    // Transitions back to PIXEL_SHADER_RESOURCE so ImGui can sample it.
    void End(ID3D12GraphicsCommandList* cl);

    // The handle to pass to ImGui::Image(). Zero until Ensure() succeeds.
    D3D12_GPU_DESCRIPTOR_HANDLE Srv() const { return m_srv; }

    int Width()  const { return m_w; }
    int Height() const { return m_h; }
    bool Valid() const { return m_colour != nullptr; }

    void Release();

private:
    Device* m_dev = nullptr;

    ID3D12Resource* m_colour = nullptr;
    ID3D12Resource* m_depth  = nullptr;

    // Its OWN one-entry RTV and DSV heaps rather than slots in a shared pool.
    //
    // Neither is shader-visible, so they cost nothing to have and nothing to
    // bind, and a per-target heap means a view can be created and destroyed
    // without a free-list. The SRV is different -- it must live in the ONE
    // shader-visible heap ImGui has bound, so it comes from Device::Srv().
    ID3D12DescriptorHeap* m_rtvHeap = nullptr;
    ID3D12DescriptorHeap* m_dsvHeap = nullptr;

    D3D12_CPU_DESCRIPTOR_HANDLE m_rtv{};
    D3D12_CPU_DESCRIPTOR_HANDLE m_dsv{};
    D3D12_CPU_DESCRIPTOR_HANDLE m_srvCpu{};
    D3D12_GPU_DESCRIPTOR_HANDLE m_srv{};

    int  m_w = 0, m_h = 0;
    bool m_inTarget = false;   // tracks the resource state between Begin/End
};

}  // namespace tglab
