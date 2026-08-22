#include "texture.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include "device.h"
#include "gpu_image.h"
#include "shader.h"

namespace tglab {

void GpuTexture::Release() {
    // Frames already submitted may still be sampling this texture or copying
    // out of its staging buffers -- resizing one mid-flight is the common case,
    // since a dropped image rarely matches the slot's current dimensions.
    // Releasing now faults the GPU, and a removed device never signals its
    // fence, so the next BeginFrame() waits forever. Hand them to the device
    // instead; it frees them once the fence says the GPU is done.
    const bool deferrable = m_dev && m_dev->Ready();

    if (m_res) {
        // Srv().Free() tolerates being called after the heap is gone, but the
        // descriptor is only meaningful while the device lives.
        if (deferrable) {
            m_dev->Srv().Free(m_cpu, m_gpu);
            m_dev->DeferRelease(m_res);
        } else {
            m_res->Release();
        }
        m_res = nullptr;
    }
    for (int i = 0; i < kMaxFramesInFlight; ++i) {
        if (m_upload[i]) {
            if (deferrable) m_dev->DeferRelease(m_upload[i]);
            else            m_upload[i]->Release();
            m_upload[i] = nullptr;
        }
        m_uploadSize[i] = 0;
    }
    m_desc    = {};
    m_version = UINT64_MAX;
}

bool GpuTexture::Create(Device& dev, const ImageDesc& d) {
    Release();
    m_dev = &dev;

    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width            = UINT64(d.width);
    rd.Height           = UINT(d.height);
    rd.DepthOrArraySize = 1;
    rd.MipLevels        = 1;
    rd.Format           = DXGI_FORMAT_R8G8B8A8_UNORM;   // display format
    rd.SampleDesc.Count = 1;
    rd.Layout           = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    // UNORDERED_ACCESS so the display conversion can write it from a compute
    // shader. Without this the only way to fill it is a CPU upload, which means
    // reading the result back first.
    rd.Flags            = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    if (FAILED(dev.Get()->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                                                  D3D12_RESOURCE_STATE_COPY_DEST,
                                                  nullptr, IID_PPV_ARGS(&m_res))))
        return false;

    D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Format                  = rd.Format;
    srv.ViewDimension           = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Texture2D.MipLevels     = 1;

    dev.Srv().Alloc(&m_cpu, &m_gpu);
    dev.Get()->CreateShaderResourceView(m_res, &srv, m_cpu);

    m_desc = d;
    m_freshlyCreated = true;
    return true;
}


namespace {

// Half-float to display byte, by table.
//
// A half has only 65536 possible values, so the entire conversion -- decode,
// clamp to 0..1, scale to 0..255 -- collapses into one indexed load. That
// matters more than it sounds: at 21 MP a viewer converts 85 million
// components per frame, and HalfToFloat is an out-of-line call in another
// translation unit, so none of it inlines.
//
// Measured at 5634x3752: 260 ms per viewer becomes 28 ms. It was the single
// largest cost between moving a slider and seeing the result -- larger than
// running the pipeline itself -- and invisible to the status bar, which times
// only the pipeline.
//
// 64 KB, built once on first use.
const uint8_t* HalfDisplayTable() {
    static const std::vector<uint8_t> table = [] {
        std::vector<uint8_t> t(65536);
        for (int i = 0; i < 65536; ++i)
            t[size_t(i)] = uint8_t(
                std::clamp(HalfToFloat(uint16_t(i)), 0.0f, 1.0f) * 255.0f + 0.5f);
        return t;
    }();
    return table.data();
}

} // namespace
bool GpuTexture::Update(Device& dev, Image& img, uint64_t contentVersion) {
    if (!img.Valid()) return false;
    const ImageDesc& d = img.Desc();

    const bool recreated = (!m_res || m_desc.width != d.width || m_desc.height != d.height);
    if (recreated) {
        if (!Create(dev, d)) return false;
    }

    // Skip the upload entirely when nothing changed — the common case once
    // the pipeline is idle, and what keeps a static frame free.
    if (!recreated && contentVersion == m_version) return true;
    m_version = contentVersion;

    // Convert to RGBA8 for display. Float formats are normalized so that
    // signed data (e.g. gradients) is visible rather than clipped to black.
    ImageView v = img.MapCpuRead();
    const int w = d.width, h = d.height;

    std::vector<uint8_t> rgba;
    const uint8_t* srcBytes = nullptr;

    if (d.format == Format::RGBA8) {
        srcBytes = v.data;
    } else if (d.format == Format::R32F) {
        float lo = 0.0f, hi = 1.0f;
        const float* p = reinterpret_cast<const float*>(v.data);
        lo = hi = p[0];
        for (int i = 1; i < w * h; ++i) { lo = std::min(lo, p[i]); hi = std::max(hi, p[i]); }
        const float range = (hi - lo) > 1e-8f ? (hi - lo) : 1.0f;

        rgba.resize(size_t(w) * size_t(h) * 4);
        for (int i = 0; i < w * h; ++i) {
            const uint8_t g = uint8_t(std::clamp((p[i] - lo) / range, 0.0f, 1.0f) * 255.0f);
            rgba[size_t(i) * 4 + 0] = g;
            rgba[size_t(i) * 4 + 1] = g;
            rgba[size_t(i) * 4 + 2] = g;
            rgba[size_t(i) * 4 + 3] = 255;
        }
        srcBytes = rgba.data();
    } else if (d.format == Format::RGBA32F) {
        const float* p = reinterpret_cast<const float*>(v.data);
        rgba.resize(size_t(w) * size_t(h) * 4);
        for (int i = 0; i < w * h * 4; ++i)
            rgba[size_t(i)] = uint8_t(std::clamp(p[i], 0.0f, 1.0f) * 255.0f);
        srcBytes = rgba.data();
    } else if (d.format == Format::RGBA16F) {
        const uint16_t* p = reinterpret_cast<const uint16_t*>(v.data);
        const uint8_t* lut = HalfDisplayTable();
        rgba.resize(size_t(w) * size_t(h) * 4);
        const size_t count = size_t(w) * size_t(h) * 4;
        for (size_t i = 0; i < count; ++i) rgba[i] = lut[p[i]];
        srcBytes = rgba.data();
    } else {
        return false;
    }

    // Upload through a staging buffer, respecting the 256-byte row alignment.
    const UINT rowPitch = UINT(w) * 4;
    const UINT aligned  = (rowPitch + D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1) &
                          ~(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1);
    const UINT64 uploadSize = UINT64(aligned) * UINT64(h);

    // Use this frame's slot. BeginFrame() already waited on its fence, so the
    // GPU is finished with whatever was here last time round.
    const UINT slot = dev.FrameSlot() % kMaxFramesInFlight;
    ID3D12Resource*& upload = m_upload[slot];
    if (upload && m_uploadSize[slot] < uploadSize) {
        upload->Release();
        upload = nullptr;
    }

    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_UPLOAD;

    D3D12_RESOURCE_DESC bd = {};
    bd.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    bd.Width            = uploadSize;
    bd.Height           = 1;
    bd.DepthOrArraySize = 1;
    bd.MipLevels        = 1;
    bd.Format           = DXGI_FORMAT_UNKNOWN;
    bd.SampleDesc.Count = 1;
    bd.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    if (!upload) {
        if (FAILED(dev.Get()->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd,
                                                      D3D12_RESOURCE_STATE_GENERIC_READ,
                                                      nullptr, IID_PPV_ARGS(&upload))))
            return false;
        m_uploadSize[slot] = uploadSize;
    }

    void* mapped = nullptr;
    D3D12_RANGE noRead = {0, 0};
    if (FAILED(upload->Map(0, &noRead, &mapped))) return false;
    for (int y = 0; y < h; ++y) {
        std::memcpy(static_cast<uint8_t*>(mapped) + size_t(y) * aligned,
                    srcBytes + size_t(y) * rowPitch, rowPitch);
    }
    upload->Unmap(0, nullptr);

    // Record the copy on the frame's command list. It runs before ImGui's
    // draw commands sample the texture, so no extra sync is needed.
    ID3D12GraphicsCommandList* cl = dev.CurrentCommandList();
    if (!cl) return false;

    D3D12_TEXTURE_COPY_LOCATION dst = {};
    dst.pResource        = m_res;
    dst.Type             = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.SubresourceIndex = 0;

    D3D12_TEXTURE_COPY_LOCATION src = {};
    src.pResource                          = upload;
    src.Type                               = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint.Footprint.Format   = DXGI_FORMAT_R8G8B8A8_UNORM;
    src.PlacedFootprint.Footprint.Width    = UINT(w);
    src.PlacedFootprint.Footprint.Height   = UINT(h);
    src.PlacedFootprint.Footprint.Depth    = 1;
    src.PlacedFootprint.Footprint.RowPitch = aligned;

    D3D12_RESOURCE_BARRIER toCopy = {};
    toCopy.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    toCopy.Transition.pResource   = m_res;
    toCopy.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    toCopy.Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    toCopy.Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_DEST;

    // A freshly created resource is already in COPY_DEST.
    if (!m_freshlyCreated) cl->ResourceBarrier(1, &toCopy);
    m_freshlyCreated = false;

    cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

    D3D12_RESOURCE_BARRIER toSrv = toCopy;
    toSrv.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    toSrv.Transition.StateAfter  = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    cl->ResourceBarrier(1, &toSrv);

    return true;
}


namespace {

// The format a compute SRV reads the source as. Mirrors ComputeContext's
// mapping, but declared here so gpu/texture.cpp does not depend on the compute
// context just for a switch.
DXGI_FORMAT DisplaySrvFormat(Format f) {
    switch (f) {
        case Format::RGBA8:   return DXGI_FORMAT_R8G8B8A8_UNORM;
        case Format::R32F:    return DXGI_FORMAT_R32_FLOAT;
        case Format::RGBA32F: return DXGI_FORMAT_R32G32B32A32_FLOAT;
        case Format::RGBA16F: return DXGI_FORMAT_R16G16B16A16_FLOAT;
        default:              return DXGI_FORMAT_UNKNOWN;
    }
}

// The display conversion, run on the graphics queue.
//
// Whatever a stage produced -- half-float scene-linear, single-channel float,
// or already-RGBA8 -- becomes the RGBA8 texture ImGui samples. Doing it here
// rather than on the CPU is the whole point: the pixels never leave the device.
//
// The R32F branch normalises over a range supplied by the caller so that signed
// data (a sobel gradient) is visible rather than clipped to black, matching
// what the CPU path did.
const char* kDisplayHlsl = R"(
Texture2D<float4>   Src : register(t0);
RWTexture2D<float4> Dst : register(u0);

cbuffer Params : register(b0) {
    uint Width;
    uint Height;
    uint Mode;      // 0 = pass through, 1 = single channel normalised
    uint LoBits;
    uint SpanBits;
};

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    float4 c = Src[int2(tid.xy)];
    if (Mode == 1) {
        float lo   = asfloat(LoBits);
        float span = asfloat(SpanBits);
        float g = saturate((c.x - lo) / span);
        Dst[tid.xy] = float4(g, g, g, 1.0);
    } else {
        Dst[tid.xy] = float4(saturate(c.rgb), 1.0);
    }
}
)";

// One pipeline shared by every view: they all run the same shader.
struct DisplayPipeline {
    ID3D12RootSignature* root = nullptr;
    ID3D12PipelineState* pso  = nullptr;

    // Non-shader-visible descriptors, so the table can be built per call
    // without disturbing ImGui's own heap allocations.
    ID3D12DescriptorHeap* heap = nullptr;
    UINT stride = 0;
    UINT cursor = 0;

    bool valid = false;
};

DisplayPipeline g_display;

constexpr UINT kDisplaySlots = 64;   // (SRV + UAV) per view, several views

bool EnsureDisplayPipeline(Device& dev) {
    if (g_display.valid) return true;

    // ShaderCompiler is a handle onto PROCESS-GLOBAL DXC objects, not an
    // independent instance -- so Shutdown() tears them down for everyone.
    // Calling it here reset the globals out from under the worker thread, whose
    // next CreateKernel() then dereferenced a null compiler. It crashed on
    // dropping a file, because that is when a new stage first needs a kernel
    // compiled after the display pipeline has been built.
    //
    // So: Init() is idempotent and safe to call, but this must never Shutdown().
    // The app owns that, once, at exit.
    ShaderCompiler compiler;
    if (!compiler.Init()) return false;

    ShaderBlob blob;
    std::string errors;
    if (!compiler.CompileCompute(kDisplayHlsl, "main", "display_convert", &blob, &errors)) {
        std::fprintf(stderr, "[display] shader: %s\n", errors.c_str());
        return false;
    }

    D3D12_DESCRIPTOR_RANGE ranges[2] = {};
    ranges[0].RangeType          = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors     = 1;
    ranges[0].BaseShaderRegister = 0;
    ranges[1].RangeType          = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors     = 1;
    ranges[1].BaseShaderRegister = 0;
    ranges[1].OffsetInDescriptorsFromTableStart = 1;

    D3D12_ROOT_PARAMETER params[2] = {};
    params[0].ParameterType            = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.Num32BitValues = 8;
    params[1].ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 2;
    params[1].DescriptorTable.pDescriptorRanges   = ranges;

    D3D12_ROOT_SIGNATURE_DESC rsd = {};
    rsd.NumParameters = 2;
    rsd.pParameters   = params;

    ID3DBlob* sig = nullptr;
    ID3DBlob* sigErr = nullptr;
    if (FAILED(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1,
                                           &sig, &sigErr))) {
        if (sigErr) sigErr->Release();
        return false;
    }
    HRESULT hr = dev.Get()->CreateRootSignature(0, sig->GetBufferPointer(),
                                                sig->GetBufferSize(),
                                                IID_PPV_ARGS(&g_display.root));
    sig->Release();
    if (sigErr) sigErr->Release();
    if (FAILED(hr)) { return false; }

    D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
    pd.pRootSignature   = g_display.root;
    pd.CS.pShaderBytecode = blob.dxil.data();
    pd.CS.BytecodeLength  = blob.dxil.size();
    if (FAILED(dev.Get()->CreateComputePipelineState(&pd, IID_PPV_ARGS(&g_display.pso)))) {
        return false;
    }

    // Shader-visible: the table is bound for the dispatch.
    D3D12_DESCRIPTOR_HEAP_DESC hd = {};
    hd.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.NumDescriptors = kDisplaySlots;
    hd.Flags          = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(dev.Get()->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&g_display.heap))))
        return false;

    g_display.stride = dev.Get()->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    g_display.valid = true;
    return true;
}

} // namespace

void ReleaseDisplayPipeline() {
    if (g_display.pso)  { g_display.pso->Release();  g_display.pso  = nullptr; }
    if (g_display.root) { g_display.root->Release(); g_display.root = nullptr; }
    if (g_display.heap) { g_display.heap->Release(); g_display.heap = nullptr; }
    g_display.valid = false;
}


bool GpuTexture::ReadRegion(Device& dev, const SharedGpuTexture& src,
                            int x, int y, int w, int h,
                            std::vector<uint8_t>* outPixels) {
    if (!src.res || w <= 0 || h <= 0) return false;

    // Clamp to the image; the caller centres a window on the cursor, which
    // runs off the edge whenever the cursor is near one.
    x = std::clamp(x, 0, src.desc.width  - 1);
    y = std::clamp(y, 0, src.desc.height - 1);
    w = std::min(w, src.desc.width  - x);
    h = std::min(h, src.desc.height - y);
    if (w <= 0 || h <= 0) return false;

    const UINT bpp      = UINT(BytesPerPixel(src.desc.format));
    const UINT rowPitch = UINT(w) * bpp;
    const UINT aligned  = (rowPitch + D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1) &
                          ~(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1);
    const UINT64 total  = UINT64(aligned) * UINT64(h);

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
    if (FAILED(dev.Get()->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd,
                                                  D3D12_RESOURCE_STATE_COPY_DEST,
                                                  nullptr, IID_PPV_ARGS(&staging))))
        return false;

    // Its own command list and fence rather than the frame's: this has to
    // complete before the pixels can be read, and the frame list is not
    // submitted until EndFrame().
    ID3D12CommandAllocator* alloc = nullptr;
    ID3D12GraphicsCommandList* list = nullptr;
    ID3D12Fence* fence = nullptr;
    HANDLE evt = nullptr;
    bool ok = false;

    if (SUCCEEDED(dev.Get()->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                    IID_PPV_ARGS(&alloc))) &&
        SUCCEEDED(dev.Get()->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc,
                                               nullptr, IID_PPV_ARGS(&list))) &&
        SUCCEEDED(dev.Get()->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)))) {
        evt = CreateEventA(nullptr, FALSE, FALSE, nullptr);

        D3D12_TEXTURE_COPY_LOCATION d = {};
        d.pResource                          = staging;
        d.Type                               = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        d.PlacedFootprint.Footprint.Format   = DisplaySrvFormat(src.desc.format);
        d.PlacedFootprint.Footprint.Width    = UINT(w);
        d.PlacedFootprint.Footprint.Height   = UINT(h);
        d.PlacedFootprint.Footprint.Depth    = 1;
        d.PlacedFootprint.Footprint.RowPitch = aligned;

        D3D12_TEXTURE_COPY_LOCATION s = {};
        s.pResource = src.res;
        s.Type      = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;

        // The source carries ALLOW_SIMULTANEOUS_ACCESS, so it can be copied
        // from on this queue without transitioning it away from the compute
        // queue that owns it.
        const D3D12_BOX box = {UINT(x), UINT(y), 0, UINT(x + w), UINT(y + h), 1};
        list->CopyTextureRegion(&d, 0, 0, 0, &s, &box);

        if (SUCCEEDED(list->Close())) {
            ID3D12CommandList* lists[] = {list};
            dev.Queue()->ExecuteCommandLists(1, lists);
            dev.Queue()->Signal(fence, 1);
            if (fence->GetCompletedValue() < 1 && evt) {
                fence->SetEventOnCompletion(1, evt);
                // Bounded: a hung copy must not park the UI thread forever.
                WaitForSingleObject(evt, 2000);
            }

            void* mapped = nullptr;
            D3D12_RANGE readAll = {0, SIZE_T(total)};
            if (SUCCEEDED(staging->Map(0, &readAll, &mapped))) {
                outPixels->resize(size_t(rowPitch) * size_t(h));
                for (int row = 0; row < h; ++row)
                    std::memcpy(outPixels->data() + size_t(row) * rowPitch,
                                static_cast<const uint8_t*>(mapped) + size_t(row) * aligned,
                                rowPitch);
                staging->Unmap(0, nullptr);
                ok = true;
            }
        }
    }

    if (evt)     CloseHandle(evt);
    if (fence)   fence->Release();
    if (list)    list->Release();
    if (alloc)   alloc->Release();
    staging->Release();
    return ok;
}
bool GpuTexture::UpdateFromGpu(Device& dev, const SharedGpuTexture& src,
                               uint64_t contentVersion) {
    if (!src.res || !src.desc.Valid()) return false;

    const bool recreated = (!m_res || m_desc.width != src.desc.width ||
                            m_desc.height != src.desc.height);
    if (recreated && !Create(dev, src.desc)) return false;

    // Nothing changed since the last upload: the common case once the pipeline
    // is idle, and what keeps a static frame free.
    if (!recreated && contentVersion == m_version) return true;
    m_version = contentVersion;

    if (!EnsureDisplayPipeline(dev)) return false;

    ID3D12GraphicsCommandList* cl = dev.CurrentCommandList();
    if (!cl) return false;

    // A slice per call, wrapping. Descriptors are consumed when the list
    // executes, so two views sharing slot 0 within a frame would both sample
    // whichever was written last -- the same trap as batched compute
    // dispatches.
    if (g_display.cursor + 2 > kDisplaySlots) g_display.cursor = 0;
    const UINT slot = g_display.cursor;
    g_display.cursor += 2;

    D3D12_CPU_DESCRIPTOR_HANDLE cpu0 =
        g_display.heap->GetCPUDescriptorHandleForHeapStart();
    D3D12_GPU_DESCRIPTOR_HANDLE gpu0 =
        g_display.heap->GetGPUDescriptorHandleForHeapStart();
    D3D12_CPU_DESCRIPTOR_HANDLE srvCpu{cpu0.ptr + SIZE_T(slot) * g_display.stride};
    D3D12_CPU_DESCRIPTOR_HANDLE uavCpu{cpu0.ptr + SIZE_T(slot + 1) * g_display.stride};
    D3D12_GPU_DESCRIPTOR_HANDLE tableGpu{gpu0.ptr + SIZE_T(slot) * g_display.stride};

    D3D12_SHADER_RESOURCE_VIEW_DESC sd = {};
    sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    sd.Format                  = DisplaySrvFormat(src.desc.format);
    sd.ViewDimension           = D3D12_SRV_DIMENSION_TEXTURE2D;
    sd.Texture2D.MipLevels     = 1;
    if (sd.Format == DXGI_FORMAT_UNKNOWN) return false;
    dev.Get()->CreateShaderResourceView(src.res, &sd, srvCpu);

    D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
    ud.Format        = DXGI_FORMAT_R8G8B8A8_UNORM;
    ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    dev.Get()->CreateUnorderedAccessView(m_res, nullptr, &ud, uavCpu);

    // The display texture sits in PIXEL_SHADER_RESOURCE for ImGui between
    // frames (or COPY_DEST when freshly created), and has to become a UAV to be
    // written.
    D3D12_RESOURCE_BARRIER b = {};
    b.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource   = m_res;
    b.Transition.StateBefore = m_freshlyCreated
                                   ? D3D12_RESOURCE_STATE_COPY_DEST
                                   : D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    b.Transition.StateAfter  = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cl->ResourceBarrier(1, &b);
    m_freshlyCreated = false;

    ID3D12DescriptorHeap* heaps[] = {g_display.heap};
    cl->SetDescriptorHeaps(1, heaps);
    cl->SetComputeRootSignature(g_display.root);
    cl->SetPipelineState(g_display.pso);

    auto bits = [](float f) { uint32_t u; std::memcpy(&u, &f, sizeof(u)); return u; };
    uint32_t roots[8] = {};
    roots[0] = uint32_t(src.desc.width);
    roots[1] = uint32_t(src.desc.height);
    roots[2] = (src.desc.format == Format::R32F) ? 1u : 0u;
    // A single-channel result is normalised over 0..1 here rather than over its
    // own min/max: finding the true range would need a reduction pass, and the
    // CPU path only had one because it already held every pixel.
    roots[3] = bits(0.0f);
    roots[4] = bits(1.0f);
    cl->SetComputeRoot32BitConstants(0, 8, roots, 0);
    cl->SetComputeRootDescriptorTable(1, tableGpu);

    cl->Dispatch(UINT((src.desc.width + 7) / 8), UINT((src.desc.height + 7) / 8), 1);

    // Back to a shader resource for ImGui to sample this frame.
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    b.Transition.StateAfter  = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    cl->ResourceBarrier(1, &b);

    // Restore ImGui's heap. Binding is command-list state, not per-dispatch, so
    // leaving mine set makes every later ImGui draw reference handles from a
    // heap that is no longer bound -- which the debug layer reports as
    // "descriptor heap ... is different from currently set descriptor heap",
    // and which crashes the app outright.
    ID3D12DescriptorHeap* uiHeap = dev.Srv().Heap();
    cl->SetDescriptorHeaps(1, &uiHeap);
    return true;
}
} // namespace tglab
