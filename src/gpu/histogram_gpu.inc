// Included by compute.cpp. Kept separate purely for size: BuildHistogram is
// long because it records two dispatches by hand rather than going through
// Dispatch(), which binds float textures and cannot express the integer UAVs
// that HLSL atomics require.

bool ComputeContext::BuildHistogram(const GpuImage& src, HistogramResult* out,
                                    std::string* err) {
    if (!Ready())     { *err = "no compute context"; return false; }
    if (!src.Valid()) { *err = "image is not GPU-resident"; return false; }
    if (!CreateHistogramResources(err)) return false;

    // Compiled once and kept: recompiling per slider tick would cost more than
    // the readback this replaces.
    if (!m_histRangeKernel.Valid() &&
        !CreateKernel(kHistRangeHlsl, "main", "hist_range", &m_histRangeKernel, err))
        return false;
    if (!m_histBinKernel.Valid() &&
        !CreateKernel(kHistBinHlsl, "main", "hist_bin", &m_histBinKernel, err))
        return false;

    const int w = src.desc.width, h = src.desc.height;
    if (w <= 0 || h <= 0) { *err = "empty image"; return false; }

    // Square sampling grid, so a wide image is not sampled more densely in one
    // direction than the other -- the rule the CPU path follows.
    int step = 1;
    while ((int64_t(w / (step + 1)) * int64_t(h / (step + 1))) > kHistMaxSamples) ++step;

    const bool isSingle = (src.desc.format == Format::R32F);

    // An 8-bit image is binned over its natural 0..255 range rather than over
    // what happens to be present, matching the CPU path. A UNORM SRV hands the
    // shader 0..1, so the flag says "use 0..1" and the range reported below is
    // scaled back to 0..255 for the caller.
    const bool fixedRange = (src.desc.format == Format::RGBA8);

    // Anything recorded but not yet submitted goes first: this reads what the
    // pipeline just wrote, and Dispatch() only records.
    if (!Flush(err)) return false;
    if (!BeginRecording()) { *err = "could not begin recording"; return false; }

    ID3D12DescriptorHeap* heaps[] = {m_srvHeap};
    m_list->SetDescriptorHeaps(1, heaps);

    D3D12_CPU_DESCRIPTOR_HANDLE heapCpu = m_srvHeap->GetCPUDescriptorHandleForHeapStart();
    D3D12_GPU_DESCRIPTOR_HANDLE heapGpu = m_srvHeap->GetGPUDescriptorHandleForHeapStart();
    auto cpuAt = [&](UINT s) {
        D3D12_CPU_DESCRIPTOR_HANDLE x{heapCpu.ptr + SIZE_T(s) * m_srvStride};
        return x;
    };
    auto gpuAt = [&](UINT s) {
        D3D12_GPU_DESCRIPTOR_HANDLE x{heapGpu.ptr + SIZE_T(s) * m_srvStride};
        return x;
    };

    D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
    ud.Format        = DXGI_FORMAT_R32_UINT;
    ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;

    // These live in COMMON between calls (that is where they are created, and
    // where the copies below leave them). Both are written as UAVs first, so
    // transition them explicitly rather than relying on a promotion that does
    // not happen for a texture with UAV access.
    auto transition = [&](ID3D12Resource* r, D3D12_RESOURCE_STATES from,
                          D3D12_RESOURCE_STATES to) {
        D3D12_RESOURCE_BARRIER b = {};
        b.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource   = r;
        b.Transition.StateBefore = from;
        b.Transition.StateAfter  = to;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        m_list->ResourceBarrier(1, &b);
    };
    transition(m_histBins,  D3D12_RESOURCE_STATE_COMMON,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    transition(m_histRange, D3D12_RESOURCE_STATE_COMMON,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    // Clearing needs the same UAV described in two heaps: a shader-visible
    // handle to bind, and a CPU-readable one for the driver to read. The main
    // heap is shader-visible and therefore CPU write-only, so the readable half
    // comes from m_histClearHeap.
    const UINT kClearBins = 40, kClearRange = 41;
    D3D12_CPU_DESCRIPTOR_HANDLE clearHeapCpu =
        m_histClearHeap->GetCPUDescriptorHandleForHeapStart();
    auto clearCpuAt = [&](UINT i) {
        D3D12_CPU_DESCRIPTOR_HANDLE x{clearHeapCpu.ptr + SIZE_T(i) * m_srvStride};
        return x;
    };

    m_device->CreateUnorderedAccessView(m_histBins,  nullptr, &ud, cpuAt(kClearBins));
    m_device->CreateUnorderedAccessView(m_histRange, nullptr, &ud, cpuAt(kClearRange));
    m_device->CreateUnorderedAccessView(m_histBins,  nullptr, &ud, clearCpuAt(0));
    m_device->CreateUnorderedAccessView(m_histRange, nullptr, &ud, clearCpuAt(1));

    const UINT zero[4] = {0, 0, 0, 0};
    const UINT ones[4] = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
    m_list->ClearUnorderedAccessViewUint(gpuAt(kClearBins), clearCpuAt(0),
                                         m_histBins, zero, 0, nullptr);

    // The two range texels need OPPOSITE starting values, and a clear writes
    // one value to the whole resource -- so clear to all-ones (correct for the
    // min at texel 0) and then zero texel 1 with a one-texel clear rect.
    //
    // Clearing both to all-ones leaves the max as the largest possible key, so
    // InterlockedMax never replaces it and the decoded maximum comes back NaN.
    m_list->ClearUnorderedAccessViewUint(gpuAt(kClearRange), clearCpuAt(1),
                                         m_histRange, ones, 0, nullptr);
    const D3D12_RECT maxTexel = {1, 0, 2, 1};
    m_list->ClearUnorderedAccessViewUint(gpuAt(kClearRange), clearCpuAt(1),
                                         m_histRange, zero, 1, &maxTexel);

    D3D12_RESOURCE_BARRIER uav = {};
    uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;

    const UINT gx = UINT((w / step + 7) / 8);
    const UINT gy = UINT((h / step + 7) / 8);

    D3D12_SHADER_RESOURCE_VIEW_DESC srcSrv = {};
    srcSrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srcSrv.Format                  = ToDxgi(src.desc.format);
    srcSrv.ViewDimension           = D3D12_SRV_DIMENSION_TEXTURE2D;
    srcSrv.Texture2D.MipLevels     = 1;

    // Each pass takes its own heap slice, for the same reason ordinary
    // dispatches do: descriptors are read when the list executes, not when it
    // is recorded, so two passes sharing slot 0 would both see the second one.
    auto runPass = [&](const ComputeKernel& k, UINT base, ID3D12Resource* target,
                       bool bindRangeSrv) {
        m_device->CreateShaderResourceView(src.res, &srcSrv, cpuAt(base));
        if (bindRangeSrv) {
            D3D12_SHADER_RESOURCE_VIEW_DESC rs = {};
            rs.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            rs.Format                  = DXGI_FORMAT_R32_UINT;
            rs.ViewDimension           = D3D12_SRV_DIMENSION_TEXTURE2D;
            rs.Texture2D.MipLevels     = 1;
            m_device->CreateShaderResourceView(m_histRange, &rs, cpuAt(base + 1));
        } else {
            m_device->CreateShaderResourceView(nullptr, &srcSrv, cpuAt(base + 1));
        }
        // Unused slots still need legal descriptors to bind the table.
        for (UINT i = 2; i < kMaxSrv; ++i)
            m_device->CreateShaderResourceView(nullptr, &srcSrv, cpuAt(base + i));

        m_device->CreateUnorderedAccessView(target, nullptr, &ud, cpuAt(base + kMaxSrv));
        for (UINT i = 1; i < kMaxUav; ++i)
            m_device->CreateUnorderedAccessView(nullptr, nullptr, &ud,
                                                cpuAt(base + kMaxSrv + i));

        m_list->SetComputeRootSignature(k.root);
        m_list->SetPipelineState(k.pso);
        uint32_t roots[kNumConstants] = {};
        roots[0] = uint32_t(w);
        roots[1] = uint32_t(h);
        roots[2] = uint32_t(step);
        roots[3] = isSingle ? 1u : 0u;
        roots[4] = fixedRange ? 1u : 0u;
        m_list->SetComputeRoot32BitConstants(0, kNumConstants, roots, 0);
        m_list->SetComputeRootDescriptorTable(1, gpuAt(base));
        m_list->Dispatch(gx, gy, 1);
    };

    runPass(m_histRangeKernel, 0, m_histRange, false);

    // Pass 2 READS what pass 1 wrote -- as an SRV, not a UAV -- so this is a
    // state change, not just an ordering one.
    transition(m_histRange, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    runPass(m_histBinKernel, 16, m_histBins, true);

    // Both are copied out next, and both end back in COMMON for the next call.
    uav.UAV.pResource = m_histBins;
    m_list->ResourceBarrier(1, &uav);
    transition(m_histBins, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_COPY_SOURCE);
    transition(m_histRange, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               D3D12_RESOURCE_STATE_COPY_SOURCE);

    // 4 KB of bins and 8 bytes of range, against the 84 MB the CPU path had to
    // fetch to compute the same picture.
    auto copyOut = [&](ID3D12Resource* tex, UINT tw, UINT th, UINT64 offset) {
        D3D12_TEXTURE_COPY_LOCATION d = {};
        d.pResource                          = m_histRead;
        d.Type                               = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        d.PlacedFootprint.Offset             = offset;
        d.PlacedFootprint.Footprint.Format   = DXGI_FORMAT_R32_UINT;
        d.PlacedFootprint.Footprint.Width    = tw;
        d.PlacedFootprint.Footprint.Height   = th;
        d.PlacedFootprint.Footprint.Depth    = 1;
        d.PlacedFootprint.Footprint.RowPitch = kHistRowPitch;
        D3D12_TEXTURE_COPY_LOCATION s = {};
        s.pResource = tex;
        s.Type      = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        m_list->CopyTextureRegion(&d, 0, 0, 0, &s, nullptr);
    };
    copyOut(m_histBins, kHistBins, kHistRows, 0);
    copyOut(m_histRange, 2, 1, UINT64(kHistRowPitch) * kHistRows);

    // Back to COMMON, which is where the next call expects to find them.
    transition(m_histBins,  D3D12_RESOURCE_STATE_COPY_SOURCE,
               D3D12_RESOURCE_STATE_COMMON);
    transition(m_histRange, D3D12_RESOURCE_STATE_COPY_SOURCE,
               D3D12_RESOURCE_STATE_COMMON);

    m_pendingWork = true;
    if (!Flush(err)) return false;

    void* mapped = nullptr;
    D3D12_RANGE all = {0, SIZE_T(kHistRowPitch) * (kHistRows + 1)};
    if (FAILED(m_histRead->Map(0, &all, &mapped))) {
        *err = "could not map the histogram readback";
        return false;
    }
    const uint8_t* bytes = static_cast<const uint8_t*>(mapped);
    auto row = [&](int r) {
        return reinterpret_cast<const uint32_t*>(
            bytes + size_t(r) * kHistRowPitch);
    };
    auto take = [&](int r, std::vector<uint32_t>* dst) {
        dst->assign(row(r), row(r) + kHistBins);
    };

    if (isSingle) { out->r.clear(); out->g.clear(); out->b.clear(); }
    else          { take(0, &out->r); take(1, &out->g); take(2, &out->b); }
    take(3, &out->luma);

    const uint32_t* rng = row(kHistRows);
    const float lo = KeyToFloat(rng[0]);
    const float hi = KeyToFloat(rng[1]);
    m_histRead->Unmap(0, nullptr);

    // An RGBA8 image has a natural range; a float one is binned over what is
    // actually present, which is what keeps a gradient in [-2, 2] from
    // collapsing into a single bin.
    if (src.desc.format == Format::RGBA8) {
        out->rangeMin = 0.0;
        out->rangeMax = 255.0;
    } else {
        out->rangeMin = double(lo);
        out->rangeMax = (hi - lo < 1e-12f) ? double(lo) + 1.0 : double(hi);
    }

    out->count = 0;
    for (uint32_t c : out->luma) out->count += c;
    return true;
}
