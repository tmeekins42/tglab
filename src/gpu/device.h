// Minimal D3D12 device + swap chain, sized for an ImGui application.
#pragma once

#include <d3d12.h>
#include <dxgi1_4.h>
#include <windows.h>

#include "descriptor_heap.h"

namespace tglab {

inline constexpr int kNumFramesInFlight = 3;
inline constexpr int kNumBackBuffers    = 3;

class Device {
public:
    bool Init(HWND hwnd, bool enableDebugLayer);
    void Shutdown();

    // Begins a frame: waits for its fence, resets the allocator, transitions
    // the back buffer to RENDER_TARGET, and returns the open command list.
    ID3D12GraphicsCommandList* BeginFrame();
    void EndFrame();                 // transitions to PRESENT, executes, presents
    void WaitForLastSubmittedFrame();

    // Prints any pending debug-layer messages to stderr.
    void DrainValidationMessages();

    void OnResize(UINT width, UINT height);

    ID3D12Device*       Get()   const { return m_device; }
    ID3D12CommandQueue* Queue() const { return m_queue; }
    SrvHeap&            Srv()         { return m_srv; }

    // Valid only between BeginFrame() and EndFrame(). Texture uploads record
    // into this list so they complete before ImGui samples them.
    ID3D12GraphicsCommandList* CurrentCommandList() const { return m_cmdList; }

    // Which frame-in-flight slot is being recorded. Per-frame staging
    // resources index by this; BeginFrame() has already waited on its fence.
    UINT FrameSlot() const { return m_frameIndex; }

    bool Ready() const { return m_device != nullptr && m_swapChain != nullptr; }

private:
    struct FrameCtx {
        ID3D12CommandAllocator* allocator = nullptr;
        UINT64                  fenceValue = 0;
    };

    bool CreateRenderTargets();
    void ReleaseRenderTargets();

    ID3D12Device*              m_device     = nullptr;
    ID3D12CommandQueue*        m_queue      = nullptr;
    ID3D12GraphicsCommandList* m_cmdList    = nullptr;
    IDXGISwapChain3*           m_swapChain  = nullptr;
    HANDLE                     m_swapWaitable = nullptr;

    ID3D12DescriptorHeap*       m_rtvHeap = nullptr;
    D3D12_CPU_DESCRIPTOR_HANDLE m_rtvHandles[kNumBackBuffers]{};
    ID3D12Resource*             m_backBuffers[kNumBackBuffers]{};

    ID3D12Fence* m_fence      = nullptr;
    HANDLE       m_fenceEvent = nullptr;
    UINT64       m_fenceLast  = 0;

    FrameCtx m_frames[kNumFramesInFlight]{};
    UINT     m_frameIndex = 0;
    UINT     m_backBufferIndex = 0;

    SrvHeap m_srv;
};

} // namespace tglab
