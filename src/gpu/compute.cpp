#include "compute.h"

#include <cstring>

#include "device.h"

namespace tglab {

// Defined in gpu_image.cpp; declared here rather than included because
// gpu_image.h includes this header.
void InstallGpuResidencyHooks();

namespace {

// Descriptors a single dispatch can bind. Kept small and fixed so the root
// signature is a constant, which keeps kernels interchangeable.
constexpr UINT kMaxSrv       = 4;
constexpr UINT kMaxUav       = 4;
constexpr UINT kNumConstants = 32;   // b0: width, height, then parameters
constexpr UINT kHeapSize     = 64;   // (kMaxSrv + kMaxUav) * a few dispatches

DXGI_FORMAT ToDxgi(Format f) {
    switch (f) {
        case Format::RGBA8:   return DXGI_FORMAT_R8G8B8A8_UNORM;
        case Format::R32F:    return DXGI_FORMAT_R32_FLOAT;
        case Format::RGBA32F: return DXGI_FORMAT_R32G32B32A32_FLOAT;
        case Format::RGBA16F: return DXGI_FORMAT_R16G16B16A16_FLOAT;
        default:              return DXGI_FORMAT_UNKNOWN;
    }
}

} // namespace

bool ComputeContext::Init(ID3D12Device* device) {
    m_device = device;

    // Own queue: this context runs on the worker thread while ImGui submits on
    // the direct queue from the UI thread.
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    if (FAILED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&m_queue)))) return false;

    if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                              IID_PPV_ARGS(&m_alloc))))
        return false;
    if (FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                         m_alloc, nullptr, IID_PPV_ARGS(&m_list))))
        return false;
    m_list->Close();

    if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)))) return false;
    m_event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (!m_event) return false;

    D3D12_DESCRIPTOR_HEAP_DESC hd = {};
    hd.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.NumDescriptors = kHeapSize;
    hd.Flags          = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&m_srvHeap)))) return false;
    m_srvStride = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    // Install here rather than at the call site: any code path that has a
    // ComputeContext can produce GPU-resident images, and without the hook
    // MapCpuRead() would silently hand back stale pixels instead of failing.
    InstallGpuResidencyHooks();

    return m_compiler.Init();
}

void ComputeContext::Shutdown() {
    if (m_queue && m_fence) {
        const UINT64 v = ++m_fenceVal;
        m_queue->Signal(m_fence, v);
        if (m_fence->GetCompletedValue() < v) {
            m_fence->SetEventOnCompletion(v, m_event);
            WaitForSingleObject(m_event, INFINITE);
        }
    }
    for (ID3D12Resource* r : m_staging) if (r) r->Release();
    m_staging.clear();

    m_compiler.Shutdown();
    if (m_srvHeap) { m_srvHeap->Release(); m_srvHeap = nullptr; }
    if (m_event)   { CloseHandle(m_event); m_event = nullptr; }
    if (m_fence)   { m_fence->Release();   m_fence = nullptr; }
    if (m_list)    { m_list->Release();    m_list = nullptr; }
    if (m_alloc)   { m_alloc->Release();   m_alloc = nullptr; }
    if (m_queue)   { m_queue->Release();   m_queue = nullptr; }
    m_device = nullptr;
}

bool ComputeContext::CreateKernel(const std::string& hlsl, const std::string& entry,
                                  const std::string& debugName,
                                  ComputeKernel* out, std::string* errors) {
    ShaderBlob blob;
    if (!m_compiler.CompileCompute(hlsl, entry, debugName, &blob, errors)) return false;

    // Fixed layout so every kernel binds the same way:
    //   b0 = 32 root constants, t0..t3 = inputs, u0..u3 = outputs.
    D3D12_DESCRIPTOR_RANGE ranges[2] = {};
    ranges[0].RangeType          = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors     = kMaxSrv;
    ranges[0].BaseShaderRegister = 0;
    ranges[1].RangeType          = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors     = kMaxUav;
    ranges[1].BaseShaderRegister = 0;
    ranges[1].OffsetInDescriptorsFromTableStart = kMaxSrv;

    D3D12_ROOT_PARAMETER params[2] = {};
    params[0].ParameterType            = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.ShaderRegister = 0;
    params[0].Constants.Num32BitValues = kNumConstants;

    params[1].ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 2;
    params[1].DescriptorTable.pDescriptorRanges   = ranges;

    D3D12_ROOT_SIGNATURE_DESC rs = {};
    rs.NumParameters = 2;
    rs.pParameters   = params;

    ID3DBlob* sig = nullptr;
    ID3DBlob* sigErr = nullptr;
    if (FAILED(D3D12SerializeRootSignature(&rs, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &sigErr))) {
        *errors = sigErr ? std::string(static_cast<const char*>(sigErr->GetBufferPointer()))
                         : "root signature serialisation failed";
        if (sig) sig->Release();
        if (sigErr) sigErr->Release();
        return false;
    }
    if (sigErr) sigErr->Release();

    ID3D12RootSignature* root = nullptr;
    const HRESULT hr = m_device->CreateRootSignature(0, sig->GetBufferPointer(),
                                                     sig->GetBufferSize(), IID_PPV_ARGS(&root));
    sig->Release();
    if (FAILED(hr)) { *errors = "could not create root signature"; return false; }

    D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
    pd.pRootSignature = root;
    pd.CS.pShaderBytecode = blob.dxil.data();
    pd.CS.BytecodeLength  = blob.dxil.size();

    ID3D12PipelineState* pso = nullptr;
    if (FAILED(m_device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso)))) {
        root->Release();
        *errors = "could not create compute pipeline state (unsigned DXIL?)";
        return false;
    }

    out->root = root;
    out->pso  = pso;
    return true;
}

bool ComputeContext::CreateImage(const ImageDesc& d, GpuImage* out) {
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width            = UINT64(d.width);
    rd.Height           = UINT(d.height);
    rd.DepthOrArraySize = 1;
    rd.MipLevels        = 1;
    rd.Format           = ToDxgi(d.format);
    rd.SampleDesc.Count = 1;
    // UNORDERED_ACCESS so compute can write it; SIMULTANEOUS_ACCESS so the
    // UI's direct queue can read it for display without a cross-queue
    // transition (which would need the writing queue to transition it back).
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS |
               D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;

    if (rd.Format == DXGI_FORMAT_UNKNOWN) return false;

    ID3D12Resource* res = nullptr;
    if (FAILED(m_device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                                                 D3D12_RESOURCE_STATE_COMMON,
                                                 nullptr, IID_PPV_ARGS(&res))))
        return false;

    out->res  = res;
    out->desc = d;
    return true;
}

// Opens the command list for recording, reusing the batch already in progress.
//
// This is what makes lazy flushing safe. Every recording site used to begin
// with m_alloc->Reset() + m_list->Reset(), which DISCARDS anything recorded but
// not yet submitted. Once Dispatch() stopped flushing, a second dispatch would
// silently throw the first one away -- the pipeline would report success and
// display an image missing a stage.
//
// Resetting only when nothing is pending means consecutive dispatches
// accumulate into one command list, and a UAV barrier (recorded by Dispatch)
// orders them against each other.
bool ComputeContext::BeginRecording() {
    if (m_pendingWork) return true;   // keep appending to the open list

    m_heapCursor = 0;   // a fresh list starts at the top of the heap
    if (FAILED(m_alloc->Reset())) return false;
    if (FAILED(m_list->Reset(m_alloc, nullptr))) return false;
    return true;
}

bool ComputeContext::Upload(const ImageView& src, GpuImage* dst) {
    if (!src.Valid() || !dst->Valid()) return false;

    const UINT rowPitch = UINT(src.Pitch());
    const UINT aligned  = (rowPitch + D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1) &
                          ~(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1);
    const UINT64 total  = UINT64(aligned) * UINT64(src.desc.height);

    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_UPLOAD;

    D3D12_RESOURCE_DESC bd = {};
    bd.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    bd.Width            = total;
    bd.Height           = 1;
    bd.DepthOrArraySize = 1;
    bd.MipLevels        = 1;
    bd.Format           = DXGI_FORMAT_UNKNOWN;
    bd.SampleDesc.Count = 1;
    bd.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    ID3D12Resource* staging = nullptr;
    if (FAILED(m_device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd,
                                                 D3D12_RESOURCE_STATE_GENERIC_READ,
                                                 nullptr, IID_PPV_ARGS(&staging))))
        return false;

    void* mapped = nullptr;
    D3D12_RANGE noRead = {0, 0};
    if (FAILED(staging->Map(0, &noRead, &mapped))) { staging->Release(); return false; }
    for (int y = 0; y < src.desc.height; ++y)
        std::memcpy(static_cast<uint8_t*>(mapped) + size_t(y) * aligned,
                    src.data + size_t(y) * rowPitch, rowPitch);
    staging->Unmap(0, nullptr);

    if (!BeginRecording()) return false;

    D3D12_TEXTURE_COPY_LOCATION d = {};
    d.pResource = dst->res;
    d.Type      = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;

    D3D12_TEXTURE_COPY_LOCATION s = {};
    s.pResource                          = staging;
    s.Type                               = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    s.PlacedFootprint.Footprint.Format   = ToDxgi(src.desc.format);
    s.PlacedFootprint.Footprint.Width    = UINT(src.desc.width);
    s.PlacedFootprint.Footprint.Height   = UINT(src.desc.height);
    s.PlacedFootprint.Footprint.Depth    = 1;
    s.PlacedFootprint.Footprint.RowPitch = aligned;

    m_list->CopyTextureRegion(&d, 0, 0, 0, &s, nullptr);

    m_staging.push_back(staging);
    m_pendingWork = true;   // the copy above still has to be submitted
    std::string err;
    return Flush(&err);
}

bool ComputeContext::Readback(const GpuImage& src, ImageView* dst) {
    if (!src.Valid() || !dst->Valid()) return false;

    const UINT rowPitch = UINT(dst->Pitch());
    const UINT aligned  = (rowPitch + D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1) &
                          ~(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1);
    const UINT64 total  = UINT64(aligned) * UINT64(src.desc.height);

    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_READBACK;

    D3D12_RESOURCE_DESC bd = {};
    bd.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    bd.Width            = total;
    bd.Height           = 1;
    bd.DepthOrArraySize = 1;
    bd.MipLevels        = 1;
    bd.Format           = DXGI_FORMAT_UNKNOWN;
    bd.SampleDesc.Count = 1;
    bd.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    ID3D12Resource* staging = nullptr;
    if (FAILED(m_device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd,
                                                 D3D12_RESOURCE_STATE_COPY_DEST,
                                                 nullptr, IID_PPV_ARGS(&staging))))
        return false;

    if (!BeginRecording()) return false;

    D3D12_TEXTURE_COPY_LOCATION d = {};
    d.pResource                          = staging;
    d.Type                               = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    d.PlacedFootprint.Footprint.Format   = ToDxgi(src.desc.format);
    d.PlacedFootprint.Footprint.Width    = UINT(src.desc.width);
    d.PlacedFootprint.Footprint.Height   = UINT(src.desc.height);
    d.PlacedFootprint.Footprint.Depth    = 1;
    d.PlacedFootprint.Footprint.RowPitch = aligned;

    D3D12_TEXTURE_COPY_LOCATION s = {};
    s.pResource = src.res;
    s.Type      = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;

    m_list->CopyTextureRegion(&d, 0, 0, 0, &s, nullptr);

    m_pendingWork = true;   // the copy above still has to be submitted
    std::string err;
    if (!Flush(&err)) { staging->Release(); return false; }

    void* mapped = nullptr;
    D3D12_RANGE range = {0, SIZE_T(total)};
    if (FAILED(staging->Map(0, &range, &mapped))) { staging->Release(); return false; }
    for (int y = 0; y < dst->desc.height; ++y)
        std::memcpy(dst->data + size_t(y) * rowPitch,
                    static_cast<const uint8_t*>(mapped) + size_t(y) * aligned, rowPitch);
    staging->Unmap(0, nullptr);
    staging->Release();
    return true;
}

bool ComputeContext::Dispatch(const ComputeKernel& k,
                              const std::vector<const GpuImage*>& inputs,
                              const std::vector<GpuImage*>& outputs,
                              const std::vector<uint32_t>& constants,
                              std::string* err) {
    if (!k.Valid())          { *err = "kernel is not valid"; return false; }
    if (outputs.empty())     { *err = "dispatch needs at least one output"; return false; }
    if (inputs.size() > kMaxSrv || outputs.size() > kMaxUav) {
        *err = "too many bound images for this root signature";
        return false;
    }

    if (!BeginRecording()) return false;

    // Build the descriptor table: SRVs first, then UAVs at a fixed offset.
    //
    // Each dispatch takes its OWN slice of the heap. The GPU reads these
    // descriptors when the list executes, which -- with batching -- is long
    // after every dispatch in the batch has been recorded. Writing them all to
    // slot 0 would leave every dispatch seeing the last one's bindings.
    constexpr UINT kSlotsPerDispatch = kMaxSrv + kMaxUav;
    if (m_heapCursor + kSlotsPerDispatch > kHeapSize) {
        // Out of heap for this batch: submit what is recorded so the slots are
        // free again. Correctness over batching -- the alternative is silently
        // aliasing another dispatch's descriptors.
        if (!Flush(err)) return false;
        if (!BeginRecording()) return false;
    }
    const UINT base = m_heapCursor;
    m_heapCursor += kSlotsPerDispatch;

    D3D12_CPU_DESCRIPTOR_HANDLE heapCpu = m_srvHeap->GetCPUDescriptorHandleForHeapStart();
    D3D12_GPU_DESCRIPTOR_HANDLE heapGpu = m_srvHeap->GetGPUDescriptorHandleForHeapStart();
    D3D12_CPU_DESCRIPTOR_HANDLE cpu = {heapCpu.ptr + SIZE_T(base) * m_srvStride};
    D3D12_GPU_DESCRIPTOR_HANDLE gpu = {heapGpu.ptr + SIZE_T(base) * m_srvStride};

    for (UINT i = 0; i < kMaxSrv; ++i) {
        D3D12_CPU_DESCRIPTOR_HANDLE h = {cpu.ptr + SIZE_T(i) * m_srvStride};
        if (i < inputs.size() && inputs[i] && inputs[i]->Valid()) {
            D3D12_SHADER_RESOURCE_VIEW_DESC sd = {};
            sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            sd.Format                  = ToDxgi(inputs[i]->desc.format);
            sd.ViewDimension           = D3D12_SRV_DIMENSION_TEXTURE2D;
            sd.Texture2D.MipLevels     = 1;
            m_device->CreateShaderResourceView(inputs[i]->res, &sd, h);
        } else {
            // A null descriptor keeps unused slots legal to bind.
            D3D12_SHADER_RESOURCE_VIEW_DESC sd = {};
            sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            sd.Format                  = DXGI_FORMAT_R8G8B8A8_UNORM;
            sd.ViewDimension           = D3D12_SRV_DIMENSION_TEXTURE2D;
            sd.Texture2D.MipLevels     = 1;
            m_device->CreateShaderResourceView(nullptr, &sd, h);
        }
    }
    for (UINT i = 0; i < kMaxUav; ++i) {
        D3D12_CPU_DESCRIPTOR_HANDLE h = {cpu.ptr + SIZE_T(kMaxSrv + i) * m_srvStride};
        if (i < outputs.size() && outputs[i] && outputs[i]->Valid()) {
            D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
            ud.Format        = ToDxgi(outputs[i]->desc.format);
            ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
            m_device->CreateUnorderedAccessView(outputs[i]->res, nullptr, &ud, h);
        } else {
            D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
            ud.Format        = DXGI_FORMAT_R8G8B8A8_UNORM;
            ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
            m_device->CreateUnorderedAccessView(nullptr, nullptr, &ud, h);
        }
    }

    ID3D12DescriptorHeap* heaps[] = {m_srvHeap};
    m_list->SetDescriptorHeaps(1, heaps);
    m_list->SetComputeRootSignature(k.root);
    m_list->SetPipelineState(k.pso);

    // b0: width, height, then the algorithm's own constants.
    const ImageDesc& od = outputs[0]->desc;
    uint32_t roots[kNumConstants] = {};
    roots[0] = uint32_t(od.width);
    roots[1] = uint32_t(od.height);
    for (size_t i = 0; i < constants.size() && (i + 2) < kNumConstants; ++i)
        roots[i + 2] = constants[i];
    m_list->SetComputeRoot32BitConstants(0, kNumConstants, roots, 0);
    m_list->SetComputeRootDescriptorTable(1, gpu);

    // 8x8 threads per group, matching the [numthreads(8,8,1)] convention.
    const UINT gx = UINT((od.width  + 7) / 8);
    const UINT gy = UINT((od.height + 7) / 8);
    m_list->Dispatch(gx, gy, 1);

    // Chained dispatches on the same resource need a UAV barrier, not a fence.
    for (GpuImage* o : outputs) {
        if (!o || !o->Valid()) continue;
        D3D12_RESOURCE_BARRIER b = {};
        b.Type          = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        b.UAV.pResource = o->res;
        m_list->ResourceBarrier(1, &b);
    }

    // Recorded, not submitted. The barrier above already orders this dispatch
    // against the next one on the same resource, so a fence wait here would buy
    // nothing and cost a full submit-and-block per stage. Whoever needs the
    // pixels -- Upload, Readback, or an explicit Flush -- submits.
    m_pendingWork = true;
    (void)err;
    return true;
}

bool ComputeContext::Flush(std::string* err) {
    // Nothing recorded: succeed rather than trying to close an already-closed
    // command list, which fails. Flush() is called both by the pipeline's
    // success path and by its cleanup guard, so the second call routinely
    // arrives with nothing to do and must not report an error for it.
    if (!m_pendingWork) return true;

    // Submitting clears the batch either way: on success the work is done, and
    // on failure the list is closed and must be reset before anything else can
    // record into it.
    m_pendingWork = false;

    if (FAILED(m_list->Close())) { *err = "could not close command list"; return false; }

    ID3D12CommandList* lists[] = {m_list};
    m_queue->ExecuteCommandLists(1, lists);

    const UINT64 v = ++m_fenceVal;
    m_queue->Signal(m_fence, v);
    if (m_fence->GetCompletedValue() < v) {
        m_fence->SetEventOnCompletion(v, m_event);
        // Bounded, not INFINITE. A kernel that overruns the GPU watchdog gets
        // its device removed, the fence never signals, and an INFINITE wait
        // parks the worker thread forever -- the app then looks frozen with no
        // clue why. Five seconds is far longer than any sane dispatch and well
        // past the ~2s TDR limit.
        if (WaitForSingleObject(m_event, 5000) != WAIT_OBJECT_0) {
            m_deviceLost = true;
            *err = "GPU did not complete in time (device hung or was removed); "
                   "falling back to the CPU";
            return false;
        }
    }

    // A hang shows up here even when the wait succeeds.
    if (m_device->GetDeviceRemovedReason() != S_OK) {
        m_deviceLost = true;
        *err = "GPU device was removed (a kernel most likely ran too long); "
               "falling back to the CPU";
        return false;
    }

    // Staging buffers are only safe to free once the GPU is done with them.
    for (ID3D12Resource* r : m_staging) if (r) r->Release();
    m_staging.clear();
    return true;
}

} // namespace tglab
