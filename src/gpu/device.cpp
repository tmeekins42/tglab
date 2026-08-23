#include "device.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
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

// Breadcrumb op -> short name. Generated from the SDK enum; a switch means
// an unknown value prints its number instead of indexing off a table.
static const char* BreadcrumbOpName(D3D12_AUTO_BREADCRUMB_OP op) {
    switch (op) {
        case D3D12_AUTO_BREADCRUMB_OP_SETMARKER: return "SETMARKER";
        case D3D12_AUTO_BREADCRUMB_OP_BEGINEVENT: return "BEGINEVENT";
        case D3D12_AUTO_BREADCRUMB_OP_ENDEVENT: return "ENDEVENT";
        case D3D12_AUTO_BREADCRUMB_OP_DRAWINSTANCED: return "DRAWINSTANCED";
        case D3D12_AUTO_BREADCRUMB_OP_DRAWINDEXEDINSTANCED: return "DRAWINDEXEDINSTANCED";
        case D3D12_AUTO_BREADCRUMB_OP_EXECUTEINDIRECT: return "EXECUTEINDIRECT";
        case D3D12_AUTO_BREADCRUMB_OP_DISPATCH: return "DISPATCH";
        case D3D12_AUTO_BREADCRUMB_OP_COPYBUFFERREGION: return "COPYBUFFERREGION";
        case D3D12_AUTO_BREADCRUMB_OP_COPYTEXTUREREGION: return "COPYTEXTUREREGION";
        case D3D12_AUTO_BREADCRUMB_OP_COPYRESOURCE: return "COPYRESOURCE";
        case D3D12_AUTO_BREADCRUMB_OP_COPYTILES: return "COPYTILES";
        case D3D12_AUTO_BREADCRUMB_OP_RESOLVESUBRESOURCE: return "RESOLVESUBRESOURCE";
        case D3D12_AUTO_BREADCRUMB_OP_CLEARRENDERTARGETVIEW: return "CLEARRENDERTARGETVIEW";
        case D3D12_AUTO_BREADCRUMB_OP_CLEARUNORDEREDACCESSVIEW: return "CLEARUNORDEREDACCESSVIEW";
        case D3D12_AUTO_BREADCRUMB_OP_CLEARDEPTHSTENCILVIEW: return "CLEARDEPTHSTENCILVIEW";
        case D3D12_AUTO_BREADCRUMB_OP_RESOURCEBARRIER: return "RESOURCEBARRIER";
        case D3D12_AUTO_BREADCRUMB_OP_EXECUTEBUNDLE: return "EXECUTEBUNDLE";
        case D3D12_AUTO_BREADCRUMB_OP_PRESENT: return "PRESENT";
        case D3D12_AUTO_BREADCRUMB_OP_RESOLVEQUERYDATA: return "RESOLVEQUERYDATA";
        case D3D12_AUTO_BREADCRUMB_OP_BEGINSUBMISSION: return "BEGINSUBMISSION";
        case D3D12_AUTO_BREADCRUMB_OP_ENDSUBMISSION: return "ENDSUBMISSION";
        case D3D12_AUTO_BREADCRUMB_OP_DECODEFRAME: return "DECODEFRAME";
        case D3D12_AUTO_BREADCRUMB_OP_PROCESSFRAMES: return "PROCESSFRAMES";
        case D3D12_AUTO_BREADCRUMB_OP_ATOMICCOPYBUFFERUINT: return "ATOMICCOPYBUFFERUINT";
        case D3D12_AUTO_BREADCRUMB_OP_ATOMICCOPYBUFFERUINT64: return "ATOMICCOPYBUFFERUINT64";
        case D3D12_AUTO_BREADCRUMB_OP_RESOLVESUBRESOURCEREGION: return "RESOLVESUBRESOURCEREGION";
        case D3D12_AUTO_BREADCRUMB_OP_WRITEBUFFERIMMEDIATE: return "WRITEBUFFERIMMEDIATE";
        case D3D12_AUTO_BREADCRUMB_OP_DECODEFRAME1: return "DECODEFRAME1";
        case D3D12_AUTO_BREADCRUMB_OP_SETPROTECTEDRESOURCESESSION: return "SETPROTECTEDRESOURCESESSION";
        case D3D12_AUTO_BREADCRUMB_OP_DECODEFRAME2: return "DECODEFRAME2";
        case D3D12_AUTO_BREADCRUMB_OP_PROCESSFRAMES1: return "PROCESSFRAMES1";
        case D3D12_AUTO_BREADCRUMB_OP_BUILDRAYTRACINGACCELERATIONSTRUCTURE: return "BUILDRAYTRACINGACCELERATIONSTRUCTURE";
        case D3D12_AUTO_BREADCRUMB_OP_EMITRAYTRACINGACCELERATIONSTRUCTUREPOSTBUILDINFO: return "EMITRAYTRACINGACCELERATIONSTRUCTUREPOSTBUILDINFO";
        case D3D12_AUTO_BREADCRUMB_OP_COPYRAYTRACINGACCELERATIONSTRUCTURE: return "COPYRAYTRACINGACCELERATIONSTRUCTURE";
        case D3D12_AUTO_BREADCRUMB_OP_DISPATCHRAYS: return "DISPATCHRAYS";
        case D3D12_AUTO_BREADCRUMB_OP_INITIALIZEMETACOMMAND: return "INITIALIZEMETACOMMAND";
        case D3D12_AUTO_BREADCRUMB_OP_EXECUTEMETACOMMAND: return "EXECUTEMETACOMMAND";
        case D3D12_AUTO_BREADCRUMB_OP_ESTIMATEMOTION: return "ESTIMATEMOTION";
        case D3D12_AUTO_BREADCRUMB_OP_RESOLVEMOTIONVECTORHEAP: return "RESOLVEMOTIONVECTORHEAP";
        case D3D12_AUTO_BREADCRUMB_OP_SETPIPELINESTATE1: return "SETPIPELINESTATE1";
        case D3D12_AUTO_BREADCRUMB_OP_INITIALIZEEXTENSIONCOMMAND: return "INITIALIZEEXTENSIONCOMMAND";
        case D3D12_AUTO_BREADCRUMB_OP_EXECUTEEXTENSIONCOMMAND: return "EXECUTEEXTENSIONCOMMAND";
        case D3D12_AUTO_BREADCRUMB_OP_DISPATCHMESH: return "DISPATCHMESH";
        case D3D12_AUTO_BREADCRUMB_OP_ENCODEFRAME: return "ENCODEFRAME";
        case D3D12_AUTO_BREADCRUMB_OP_RESOLVEENCODEROUTPUTMETADATA: return "RESOLVEENCODEROUTPUTMETADATA";
        case D3D12_AUTO_BREADCRUMB_OP_BARRIER: return "BARRIER";
        case D3D12_AUTO_BREADCRUMB_OP_BEGIN_COMMAND_LIST: return "BEGIN_COMMAND_LIST";
        case D3D12_AUTO_BREADCRUMB_OP_DISPATCHGRAPH: return "DISPATCHGRAPH";
        case D3D12_AUTO_BREADCRUMB_OP_SETPROGRAM: return "SETPROGRAM";
        case D3D12_AUTO_BREADCRUMB_OP_PROCESSFRAMES2: return "PROCESSFRAMES2";
    }
    return "<unknown op>";
}

// Dumps DRED breadcrumbs and page-fault data after a device removal.
//
// The point of this is the ordering: for each command list DRED reports how
// many ops COMPLETED. Op N completed and op N+1 did not, so N+1 is where the
// GPU stopped -- which is the single fact DXGI_ERROR_DEVICE_HUNG withholds.
// Ops after the stall are printed too, marked, because the hang is usually
// caused by the op that never finished rather than the last one that did.
//
// Safe to call on a dead device: DRED data is retained by the runtime
// precisely so it can be read after removal.
void ReportDeviceRemoval(ID3D12Device* device, const char* where) {
    if (!device) return;

    const HRESULT reason = device->GetDeviceRemovedReason();
    std::fprintf(stderr, "\n=== DEVICE REMOVED (%s) reason=0x%08lX ===\n",
                 where, static_cast<unsigned long>(reason));

    ID3D12DeviceRemovedExtendedData1* dred = nullptr;
    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&dred))) || !dred) {
        std::fprintf(stderr, "  (no DRED interface; run a Debug build)\n");
        std::fflush(stderr);
        return;
    }

    D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT1 crumbs = {};
    if (SUCCEEDED(dred->GetAutoBreadcrumbsOutput1(&crumbs))) {
        int listIndex = 0;
        for (const D3D12_AUTO_BREADCRUMB_NODE1* n = crumbs.pHeadAutoBreadcrumbNode;
             n; n = n->pNext, ++listIndex) {
            const UINT done = n->pLastBreadcrumbValue ? *n->pLastBreadcrumbValue : 0;
            // A list that finished everything did not stall; skip the noise.
            if (n->BreadcrumbCount == 0 || done == n->BreadcrumbCount) continue;

            // Prefer the wide name: SetName() sets W, so the A fields are null
            // for anything this app named. %ls handles the wide string.
            std::fprintf(stderr, "  [list %d] %ls (%p) on queue %ls (%p) -- completed %u of %u\n",
                         listIndex,
                         n->pCommandListDebugNameW ? n->pCommandListDebugNameW : L"<unnamed>",
                         static_cast<void*>(n->pCommandList),
                         n->pCommandQueueDebugNameW ? n->pCommandQueueDebugNameW : L"<unnamed>",
                         static_cast<void*>(n->pCommandQueue),
                         done, n->BreadcrumbCount);

            // A window around the stall: enough for context, not the whole list.
            const UINT lo = done > 4 ? done - 4 : 0;
            const UINT hi = (done + 4 < n->BreadcrumbCount) ? done + 4 : n->BreadcrumbCount;
            for (UINT i = lo; i < hi; ++i) {
                std::fprintf(stderr, "      %s op[%u] = %s\n",
                             i == done ? "-->" : "   ", i,
                             BreadcrumbOpName(n->pCommandHistory[i]));
            }
        }
    }

    D3D12_DRED_PAGE_FAULT_OUTPUT1 pf = {};
    if (SUCCEEDED(dred->GetPageFaultAllocationOutput1(&pf)) && pf.PageFaultVA) {
        std::fprintf(stderr, "  page fault VA = 0x%llx\n",
                     static_cast<unsigned long long>(pf.PageFaultVA));
        for (const D3D12_DRED_ALLOCATION_NODE1* a = pf.pHeadExistingAllocationNode;
             a; a = a->pNext) {
            std::fprintf(stderr, "    live alloc: %s\n",
                         a->ObjectNameA ? a->ObjectNameA : "<unnamed>");
        }
        // Freed-but-faulting is the signature of a use-after-free.
        for (const D3D12_DRED_ALLOCATION_NODE1* a = pf.pHeadRecentFreedAllocationNode;
             a; a = a->pNext) {
            std::fprintf(stderr, "    RECENTLY FREED: %s\n",
                         a->ObjectNameA ? a->ObjectNameA : "<unnamed>");
        }
    }

    dred->Release();
    std::fprintf(stderr, "=== end DRED ===\n\n");
    std::fflush(stderr);
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

    // Device Removed Extended Data. Must be turned on BEFORE device creation,
    // like the debug layer, because it changes how the runtime records work.
    //
    // This is what a GPU hang leaves behind. DXGI_ERROR_DEVICE_HUNG on its own
    // names nothing: the device is gone and the debug layer has no faulting
    // call to report. Auto-breadcrumbs record the ops each command list
    // actually completed, so the last one before the gap is where the GPU
    // stopped. Page-fault data adds the offending VA and the resource that
    // owned it, which also catches use-after-free.
    //
    // Costs per-op overhead, so it rides with the debug layer rather than
    // shipping in release builds.
    if (enableDebugLayer) {
        ID3D12DeviceRemovedExtendedDataSettings1* dred = nullptr;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dred)))) {
            dred->SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            dred->SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            dred->Release();
            Trace("DRED enabled");
        }
    }

    // GPU-based validation, opt-in via TGLAB_GBV=1.
    //
    // Off by default because it instruments every shader and costs 10-100x --
    // far too slow for interactive use. But it validates resource STATE at
    // dispatch time on the GPU, which is precisely the class of mistake the
    // ordinary debug layer cannot see: it checks descriptors as executed
    // rather than as recorded.
    //
    // SynchronizedCommandQueueValidation serialises queue execution so that
    // cross-queue hazards are reported against the call that caused them.
    // That is the check that matters for a texture the worker writes while
    // the UI samples it.
    if (enableDebugLayer) {
        size_t gbvLen = 0;
        const bool wantGbv =
            getenv_s(&gbvLen, nullptr, 0, "TGLAB_GBV") == 0 && gbvLen > 0;
        if (wantGbv) {
            ID3D12Debug1* dbg1 = nullptr;
            if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dbg1)))) {
                dbg1->SetEnableGPUBasedValidation(TRUE);
                dbg1->SetEnableSynchronizedCommandQueueValidation(TRUE);
                dbg1->Release();
                Trace("GPU-based validation ON (slow)");
            }
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
            // WARNING too: an object destroyed while still referenced raises
            // 0x087D at teardown, and breaking there loses the message that
            // says WHICH object -- the one useful part.
            iq->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_WARNING, FALSE);
            iq->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_INFO, FALSE);
            iq->Release();
        }
    }

    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type  = D3D12_COMMAND_LIST_TYPE_DIRECT;
    qd.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    if (FAILED(m_device->CreateCommandQueue(&qd, IID_PPV_ARGS(&m_queue)))) return false;
    // Named so DRED breadcrumbs say WHICH queue stalled rather than <unnamed>.
    m_queue->SetName(L"ui.direct.queue");

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
    m_cmdList->SetName(L"ui.direct.list");
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

    // Keep the adapter for QueryVideoMemoryInfo(). Intermediates at 45 MP are
    // hundreds of megabytes each, so knowing how close the pipeline is to the
    // card's budget is the difference between "it is slow" and "it is paging".
    {
        IDXGIAdapter1* adapter1 = nullptr;
        if (SUCCEEDED(factory->EnumAdapters1(0, &adapter1))) {
            // IDXGIAdapter3 is where QueryVideoMemoryInfo lives; a machine
            // without it simply reports no usage rather than failing to start.
            adapter1->QueryInterface(IID_PPV_ARGS(&m_adapter));
            adapter1->Release();
        }
    }

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

void Device::VideoMemory(uint64_t* used, uint64_t* budget) const {
    *used = 0;
    *budget = 0;
    if (!m_adapter) return;

    DXGI_QUERY_VIDEO_MEMORY_INFO info = {};
    if (SUCCEEDED(m_adapter->QueryVideoMemoryInfo(
            0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) {
        *used   = info.CurrentUsage;
        *budget = info.Budget;
    }
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
    //
    // Bounded, not INFINITE. A faulted GPU is removed by the driver and its
    // fence never signals again, so an infinite wait here is the difference
    // between an app that reports a problem and one that simply freezes with no
    // clue why -- which is exactly how a descriptor-aliasing bug presented.
    // Five seconds is far beyond any real frame and well past the ~2s TDR limit.
    if (f.fenceValue != 0 && m_fence->GetCompletedValue() < f.fenceValue) {
        m_fence->SetEventOnCompletion(f.fenceValue, m_fenceEvent);
        if (WaitForSingleObject(m_fenceEvent, 5000) != WAIT_OBJECT_0) {
            std::fprintf(stderr, "[gpu] frame fence did not signal within 5s\n");
            std::fflush(stderr);
            ReportDeviceRemoval(m_device, "frame fence timeout");
        }
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
    if (m_adapter)      { m_adapter->Release();        m_adapter = nullptr; }
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
