#include "device.h"

#include <cassert>
#include <cstdio>
#include <vector>

namespace tglab {

// SRV heap must hold the ImGui font atlas plus one descriptor per live view
// texture. 256 is far more than M1 needs and costs almost nothing.
static constexpr uint32_t kSrvCapacity = 256;

// Set TGLAB_VERBOSE=1 to trace startup when the window never appears.
static void Trace(const char* stage, HRESULT hr = S_OK) {
    static const bool on = GetEnvironmentVariableA("TGLAB_VERBOSE", nullptr, 0) > 0;
    if (!on) return;
    char buf[256];
    wsprintfA(buf, "[tglab] %s hr=0x%08X\n", stage, unsigned(hr));
    OutputDebugStringA(buf);
    fputs(buf, stderr);
    fflush(stderr);
}

bool Device::Init(HWND hwnd, bool enableDebugLayer) {
    Trace("Init begin");
    // The debug layer must be enabled before device creation. It is what
    // catches the barrier/queue mistakes that are otherwise intermittent.
    if (enableDebugLayer) {
        ID3D12Debug* dbg = nullptr;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dbg)))) {
            dbg->EnableDebugLayer();
            dbg->Release();
        }
    }

    D3D_FEATURE_LEVEL level = D3D_FEATURE_LEVEL_11_0;
    HRESULT hr = D3D12CreateDevice(nullptr, level, IID_PPV_ARGS(&m_device));
    Trace("CreateDevice", hr);
    if (FAILED(hr)) return false;

    // Print validation messages instead of breaking into the debugger, so a
    // console run reports the actual complaint rather than dying at 0x087D.
    if (enableDebugLayer) {
        ID3D12InfoQueue* iq = nullptr;
        if (SUCCEEDED(m_device->QueryInterface(IID_PPV_ARGS(&iq)))) {
            iq->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, FALSE);
            iq->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, FALSE);
            iq->Release();
        }
    }

    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type  = D3D12_COMMAND_LIST_TYPE_DIRECT;
    qd.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    if (FAILED(m_device->CreateCommandQueue(&qd, IID_PPV_ARGS(&m_queue)))) return false;

    D3D12_DESCRIPTOR_HEAP_DESC rtvDesc = {};
    rtvDesc.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtvDesc.NumDescriptors = kNumBackBuffers;
    rtvDesc.Flags          = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    if (FAILED(m_device->CreateDescriptorHeap(&rtvDesc, IID_PPV_ARGS(&m_rtvHeap)))) return false;

    const UINT rtvStride = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    D3D12_CPU_DESCRIPTOR_HANDLE rtv = m_rtvHeap->GetCPUDescriptorHandleForHeapStart();
    for (int i = 0; i < kNumBackBuffers; ++i) {
        m_rtvHandles[i] = rtv;
        rtv.ptr += rtvStride;
    }

    Trace("rtv heap ok");
    if (!m_srv.Init(m_device, kSrvCapacity)) { Trace("srv heap FAILED"); return false; }
    Trace("srv heap ok");

    for (int i = 0; i < kNumFramesInFlight; ++i) {
        if (FAILED(m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                    IID_PPV_ARGS(&m_frames[i].allocator))))
            return false;
    }

    if (FAILED(m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                           m_frames[0].allocator, nullptr,
                                           IID_PPV_ARGS(&m_cmdList))))
        return false;
    m_cmdList->Close();

    if (FAILED(m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)))) return false;
    m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (!m_fenceEvent) return false;

    DXGI_SWAP_CHAIN_DESC1 sd = {};
    sd.BufferCount = kNumBackBuffers;
    sd.Width       = 0;
    sd.Height      = 0;
    sd.Format      = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.SwapEffect  = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    sd.SampleDesc.Count = 1;
    sd.AlphaMode   = DXGI_ALPHA_MODE_UNSPECIFIED;
    sd.Flags       = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;

    Trace("fence ok");

    IDXGIFactory4* factory = nullptr;
    hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    Trace("dxgi factory", hr);
    if (FAILED(hr)) return false;

    IDXGISwapChain1* sc1 = nullptr;
    hr = factory->CreateSwapChainForHwnd(m_queue, hwnd, &sd, nullptr, nullptr, &sc1);
    factory->Release();
    Trace("swapchain", hr);
    if (FAILED(hr)) return false;

    hr = sc1->QueryInterface(IID_PPV_ARGS(&m_swapChain));
    sc1->Release();
    if (FAILED(hr)) return false;
    Trace("swapchain3 ok");

    m_swapChain->SetMaximumFrameLatency(kNumBackBuffers);
    m_swapWaitable = m_swapChain->GetFrameLatencyWaitableObject();

    if (!CreateRenderTargets()) { Trace("render targets FAILED"); return false; }
    m_backBufferIndex = m_swapChain->GetCurrentBackBufferIndex();
    Trace("Init complete");
    return true;
}

bool Device::CreateRenderTargets() {
    for (int i = 0; i < kNumBackBuffers; ++i) {
        ID3D12Resource* buf = nullptr;
        if (FAILED(m_swapChain->GetBuffer(UINT(i), IID_PPV_ARGS(&buf)))) return false;
        m_device->CreateRenderTargetView(buf, nullptr, m_rtvHandles[i]);
        m_backBuffers[i] = buf;
    }
    return true;
}

void Device::ReleaseRenderTargets() {
    for (int i = 0; i < kNumBackBuffers; ++i) {
        if (m_backBuffers[i]) { m_backBuffers[i]->Release(); m_backBuffers[i] = nullptr; }
    }
}

void Device::OnResize(UINT width, UINT height) {
    if (!m_swapChain || width == 0 || height == 0) return;
    WaitForLastSubmittedFrame();
    ReleaseRenderTargets();

    DXGI_SWAP_CHAIN_DESC1 sd = {};
    m_swapChain->GetDesc1(&sd);
    m_swapChain->ResizeBuffers(kNumBackBuffers, width, height, sd.Format, sd.Flags);

    CreateRenderTargets();
    m_backBufferIndex = m_swapChain->GetCurrentBackBufferIndex();
}

void Device::DeferRelease(ID3D12Resource* res) {
    if (!res) return;
    // m_fenceLast is the newest value signalled, so waiting for it covers every
    // frame already submitted -- including the one still recording, which is
    // signalled with a higher value in EndFrame().
    m_pendingReleases.push_back({res, m_fenceLast + 1});
}

// force: release regardless of the fence, for shutdown after the queue is idle.
void Device::CollectPendingReleases(bool force) {
    const UINT64 done = m_fence ? m_fence->GetCompletedValue() : 0;
    for (auto& p : m_pendingReleases) {
        if (!force && p.fence > done) continue;
        p.res->Release();
        p.res = nullptr;
    }
    std::erase_if(m_pendingReleases, [](const PendingRelease& p) { return p.res == nullptr; });
}

ID3D12GraphicsCommandList* Device::BeginFrame() {
    if (m_swapWaitable) WaitForSingleObject(m_swapWaitable, 1000);

    FrameCtx& f = m_frames[m_frameIndex];

    // Wait until this frame slot's previous work has completed.
    if (f.fenceValue != 0 && m_fence->GetCompletedValue() < f.fenceValue) {
        m_fence->SetEventOnCompletion(f.fenceValue, m_fenceEvent);
        WaitForSingleObject(m_fenceEvent, INFINITE);
    }

    m_backBufferIndex = m_swapChain->GetCurrentBackBufferIndex();

    // Now that this slot's fence has passed, anything it was protecting is free.
    CollectPendingReleases(false);

    f.allocator->Reset();
    m_cmdList->Reset(f.allocator, nullptr);

    D3D12_RESOURCE_BARRIER b = {};
    b.Type  = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    b.Transition.pResource   = m_backBuffers[m_backBufferIndex];
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    b.Transition.StateAfter  = D3D12_RESOURCE_STATE_RENDER_TARGET;
    m_cmdList->ResourceBarrier(1, &b);

    const float clear[4] = {0.10f, 0.11f, 0.12f, 1.0f};
    m_cmdList->ClearRenderTargetView(m_rtvHandles[m_backBufferIndex], clear, 0, nullptr);
    m_cmdList->OMSetRenderTargets(1, &m_rtvHandles[m_backBufferIndex], FALSE, nullptr);

    ID3D12DescriptorHeap* heaps[] = {m_srv.Heap()};
    m_cmdList->SetDescriptorHeaps(1, heaps);

    return m_cmdList;
}

void Device::DrainValidationMessages() {
    if (!m_device) return;
    ID3D12InfoQueue* iq = nullptr;
    if (FAILED(m_device->QueryInterface(IID_PPV_ARGS(&iq)))) return;

    const UINT64 n = iq->GetNumStoredMessages();
    for (UINT64 i = 0; i < n; ++i) {
        SIZE_T len = 0;
        iq->GetMessage(i, nullptr, &len);
        std::vector<char> buf(len);
        auto* msg = reinterpret_cast<D3D12_MESSAGE*>(buf.data());
        if (SUCCEEDED(iq->GetMessage(i, msg, &len)) && msg->pDescription) {
            std::fprintf(stderr, "[d3d12] %s\n", msg->pDescription);
            std::fflush(stderr);
        }
    }
    iq->ClearStoredMessages();
    iq->Release();
}

void Device::EndFrame() {
    D3D12_RESOURCE_BARRIER b = {};
    b.Type  = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource   = m_backBuffers[m_backBufferIndex];
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    b.Transition.StateAfter  = D3D12_RESOURCE_STATE_PRESENT;
    m_cmdList->ResourceBarrier(1, &b);
    m_cmdList->Close();

    ID3D12CommandList* lists[] = {m_cmdList};
    m_queue->ExecuteCommandLists(1, lists);

    m_swapChain->Present(1, 0);

    const UINT64 v = ++m_fenceLast;
    m_queue->Signal(m_fence, v);
    m_frames[m_frameIndex].fenceValue = v;
    m_frameIndex = (m_frameIndex + 1) % kNumFramesInFlight;

    DrainValidationMessages();
}

void Device::WaitForLastSubmittedFrame() {
    if (!m_fence || m_fenceLast == 0) return;
    if (m_fence->GetCompletedValue() < m_fenceLast) {
        m_fence->SetEventOnCompletion(m_fenceLast, m_fenceEvent);
        WaitForSingleObject(m_fenceEvent, INFINITE);
    }
}

void Device::Shutdown() {
    WaitForLastSubmittedFrame();
    // The queue is idle, so deferred resources can go now regardless of fence.
    CollectPendingReleases(true);
    ReleaseRenderTargets();

    if (m_swapWaitable) { CloseHandle(m_swapWaitable); m_swapWaitable = nullptr; }
    if (m_swapChain)    { m_swapChain->Release();      m_swapChain = nullptr; }
    if (m_cmdList)      { m_cmdList->Release();        m_cmdList = nullptr; }
    for (auto& f : m_frames) {
        if (f.allocator) { f.allocator->Release(); f.allocator = nullptr; }
    }
    if (m_fenceEvent) { CloseHandle(m_fenceEvent); m_fenceEvent = nullptr; }
    if (m_fence)      { m_fence->Release();        m_fence = nullptr; }
    m_srv.Shutdown();
    if (m_rtvHeap) { m_rtvHeap->Release(); m_rtvHeap = nullptr; }
    if (m_queue)   { m_queue->Release();   m_queue = nullptr; }
    if (m_device)  { m_device->Release();  m_device = nullptr; }
}

} // namespace tglab
