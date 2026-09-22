#include "render_target.h"

#include <cstdio>

#include "device.h"

namespace tglab {

void RenderTarget::Release() {
    // Deferred, not immediate. The GPU may still be reading this target from a
    // frame already submitted -- a panel resized while an earlier frame still
    // samples it -- and releasing underneath that faults the device. A fault
    // here is particularly unpleasant: the device is removed, its fence never
    // signals, and the next BeginFrame() blocks forever with no error.
    if (m_dev) {
        if (m_colour) m_dev->DeferRelease(m_colour);
        if (m_depth)  m_dev->DeferRelease(m_depth);
        if (m_srv.ptr) m_dev->Srv().Free(m_srvCpu, m_srv);
    }
    m_colour = nullptr;
    m_depth = nullptr;
    m_srv = {};
    m_srvCpu = {};

    if (m_rtvHeap) { m_rtvHeap->Release(); m_rtvHeap = nullptr; }
    if (m_dsvHeap) { m_dsvHeap->Release(); m_dsvHeap = nullptr; }
    m_w = m_h = 0;
    m_inTarget = false;
}

bool RenderTarget::Ensure(Device& dev, int w, int h) {
    if (w <= 0 || h <= 0) return false;
    if (m_colour && w == m_w && h == m_h) return true;

    // A panel dragged across the screen resizes every frame, so this path runs
    // often; the deferred release above is what keeps that safe.
    Release();
    m_dev = &dev;

    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;

    // --- colour ------------------------------------------------------------
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width            = UINT64(w);
    rd.Height           = UINT(h);
    rd.DepthOrArraySize = 1;
    rd.MipLevels        = 1;
    // Matching the swap chain's format, so the result composites with ImGui's
    // own drawing without a conversion.
    rd.Format           = DXGI_FORMAT_R8G8B8A8_UNORM;
    rd.SampleDesc.Count = 1;
    rd.Flags            = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;

    // A clear value MUST be supplied for a render target, and must match what
    // Begin() actually clears to. A mismatch is not an error -- it silently
    // costs the fast clear path, which is exactly the kind of performance bug
    // that never gets found.
    D3D12_CLEAR_VALUE cv = {};
    cv.Format = rd.Format;
    cv.Color[0] = cv.Color[1] = cv.Color[2] = 0.0f;
    cv.Color[3] = 1.0f;

    if (FAILED(dev.Get()->CreateCommittedResource(
            &hp, D3D12_HEAP_FLAG_NONE, &rd,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, &cv,
            IID_PPV_ARGS(&m_colour)))) {
        std::fprintf(stderr, "[rt] could not create a %dx%d colour target\n", w, h);
        return false;
    }

    // --- depth --------------------------------------------------------------
    D3D12_RESOURCE_DESC dd = rd;
    dd.Format = DXGI_FORMAT_D32_FLOAT;
    dd.Flags  = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;

    D3D12_CLEAR_VALUE dcv = {};
    dcv.Format = dd.Format;
    dcv.DepthStencil.Depth = 1.0f;

    if (FAILED(dev.Get()->CreateCommittedResource(
            &hp, D3D12_HEAP_FLAG_NONE, &dd, D3D12_RESOURCE_STATE_DEPTH_WRITE,
            &dcv, IID_PPV_ARGS(&m_depth)))) {
        std::fprintf(stderr, "[rt] could not create a %dx%d depth buffer\n", w, h);
        Release();
        return false;
    }

    // --- views ---------------------------------------------------------------
    D3D12_DESCRIPTOR_HEAP_DESC hd = {};
    hd.NumDescriptors = 1;
    hd.Type  = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    if (FAILED(dev.Get()->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&m_rtvHeap)))) {
        Release();
        return false;
    }
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    if (FAILED(dev.Get()->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&m_dsvHeap)))) {
        Release();
        return false;
    }

    m_rtv = m_rtvHeap->GetCPUDescriptorHandleForHeapStart();
    m_dsv = m_dsvHeap->GetCPUDescriptorHandleForHeapStart();
    dev.Get()->CreateRenderTargetView(m_colour, nullptr, m_rtv);

    D3D12_DEPTH_STENCIL_VIEW_DESC dsv = {};
    dsv.Format        = DXGI_FORMAT_D32_FLOAT;
    dsv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    dev.Get()->CreateDepthStencilView(m_depth, &dsv, m_dsv);

    // The SRV goes in the SHADER-VISIBLE heap, because that is the one ImGui
    // binds and only one can be bound at a time.
    D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Format                  = rd.Format;
    srv.ViewDimension           = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Texture2D.MipLevels     = 1;
    dev.Srv().Alloc(&m_srvCpu, &m_srv);
    dev.Get()->CreateShaderResourceView(m_colour, &srv, m_srvCpu);

    m_w = w;
    m_h = h;
    return true;
}

bool RenderTarget::Begin(ID3D12GraphicsCommandList* cl, const float clear[4]) {
    if (!cl || !m_colour) return false;

    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource   = m_colour;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    b.Transition.StateAfter  = D3D12_RESOURCE_STATE_RENDER_TARGET;
    cl->ResourceBarrier(1, &b);
    m_inTarget = true;

    cl->OMSetRenderTargets(1, &m_rtv, FALSE, &m_dsv);
    cl->ClearRenderTargetView(m_rtv, clear, 0, nullptr);
    cl->ClearDepthStencilView(m_dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);

    D3D12_VIEWPORT vp = {};
    vp.Width = float(m_w);
    vp.Height = float(m_h);
    vp.MaxDepth = 1.0f;
    cl->RSSetViewports(1, &vp);

    D3D12_RECT sc = {0, 0, LONG(m_w), LONG(m_h)};
    cl->RSSetScissorRects(1, &sc);
    return true;
}

void RenderTarget::End(ID3D12GraphicsCommandList* cl) {
    if (!cl || !m_colour || !m_inTarget) return;

    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource   = m_colour;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    b.Transition.StateAfter  = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    cl->ResourceBarrier(1, &b);
    m_inTarget = false;
}

}  // namespace tglab
