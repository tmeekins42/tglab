// dxc_smoke — proves the dxcompiler.dll beside this exe actually compiles.
//
// The packaging guard used to check only that the DLLs were PRESENT. That is
// not the same as working: the first published release shipped a dxcompiler.dll
// from an older SDK whose DxcCreateInstance fails, so the app started, silently
// fell back to the CPU for every algorithm, and looked merely slow. The archive
// passed every check we had.
//
// This compiles a real kernel through the same ShaderCompiler the app uses, so
// a DLL that cannot do the job fails the build instead of shipping.
#include <cstdio>

#include <string>

#include "../src/algo_util/tone_curve.h"
#include "../src/gpu/shader.h"

using namespace tglab;

int main() {
    // Not a trivial passthrough: a kernel with a constant buffer, a texture
    // read and a UAV write, so the compile exercises the same features the real
    // kernels use rather than the smallest thing DXC will accept.
    const char* kSource = R"(
Texture2D<float4>   Src : register(t0);
RWTexture2D<float4> Dst : register(u0);
cbuffer Params : register(b0) { uint Width; uint Height; uint Mode; };

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= Width || tid.y >= Height) return;
    float4 c = Src[tid.xy];
    Dst[tid.xy] = (Mode == 0) ? c : float4(1.0 - c.rgb, c.a);
}
)";

    ShaderCompiler sc;
    if (!sc.Init()) {
        std::fprintf(stderr,
                     "FAIL: DxcCreateInstance failed.\n"
                     "  dxcompiler.dll is missing, or too old to be loaded.\n");
        return 1;
    }

    ShaderBlob  blob;
    std::string err;
    if (!sc.CompileCompute(kSource, "main", "smoke", &blob, &err)) {
        std::fprintf(stderr, "FAIL: compile failed: %s\n", err.c_str());
        return 1;
    }

    // A blob that compiled but was not signed is rejected by D3D12 at PSO
    // creation, with an error that never mentions dxil.dll. Since signing is
    // what dxil.dll does, an empty or tiny blob means it was absent.
    if (blob.dxil.size() < 64) {
        std::fprintf(stderr,
                     "FAIL: DXIL blob is %zu bytes -- too small to be signed.\n"
                     "  dxil.dll is probably missing beside dxcompiler.dll.\n",
                     blob.dxil.size());
        return 1;
    }

    std::printf("ok: DXC compiled and signed a %zu-byte kernel\n", blob.dxil.size());

    // The tone curve's HLSL, which the display shader concatenates in.
    //
    // Checked here for the same reason this file exists at all: a display
    // shader that fails to compile does not report an error to the user, it
    // falls back to CPU readback -- so the viewer still shows a picture and
    // nothing looks wrong, it merely gets slower. That precise failure mode
    // already cost a day once, when Reserve() demanded descriptors from index 0
    // and every conversion quietly took the slow path.
    //
    // The curve lives in a C++ header and is duplicated as an HLSL string, so
    // this is also what catches the two drifting apart in a way that only
    // breaks at runtime.
    {
        const std::string display = std::string(kToneCurveHlsl) + R"(
RWTexture2D<float4> Dst : register(u0);
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    Dst[tid.xy] = float4(ToneCurve3(float3(0.18, 1.0, 4.0)), 1.0);
}
)";
        ShaderBlob  db;
        std::string derr;
        if (!sc.CompileCompute(display, "main", "tone_curve", &db, &derr)) {
            std::fprintf(stderr, "FAIL: the tone-curve HLSL does not compile: %s\n",
                         derr.c_str());
            return 1;
        }
        std::printf("ok: the tone-curve HLSL compiles (%zu bytes)\n", db.dxil.size());
    }

    return 0;
}
