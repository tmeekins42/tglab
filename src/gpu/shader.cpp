#include "shader.h"

#include <windows.h>
#include <dxcapi.h>
#include <wrl/client.h>

#include <fstream>
#include <sstream>

using Microsoft::WRL::ComPtr;

namespace tglab {

namespace {

// DXC is loaded through its COM-ish API. Keeping the instances alive across
// compiles avoids re-initialising for every shader edit.
ComPtr<IDxcUtils>          g_utils;
ComPtr<IDxcCompiler3>      g_compiler;
ComPtr<IDxcIncludeHandler> g_includes;

std::string Narrow(const wchar_t* w) {
    if (!w) return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    std::string s(size_t(n > 0 ? n - 1 : 0), '\0');
    if (n > 1) WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), n, nullptr, nullptr);
    return s;
}

std::wstring Widen(const std::string& s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), int(s.size()), nullptr, 0);
    std::wstring w(size_t(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), int(s.size()), w.data(), n);
    return w;
}

} // namespace

// How many live ShaderCompilers hold the globals above.
//
// The DXC objects are process-global, but ShaderCompiler looks like an ordinary
// per-instance object -- so one instance's Shutdown() used to tear them down
// for every other. That is not theoretical: the display-conversion pipeline
// compiles its shader on the UI thread and shut its compiler down afterwards,
// which reset g_compiler out from under the worker thread. The worker's next
// CreateKernel() dereferenced null and crashed the app, reproducibly, on
// dropping a file -- because a drop is when a new stage first needs a kernel
// compiled after the display pipeline exists.
//
// Refcounted, so the globals live until the last user is done with them and an
// ownership mistake cannot break an unrelated thread.
int g_refCount = 0;

bool ShaderCompiler::Init() {
    if (m_ready) return true;

    if (g_refCount == 0) {
        if (FAILED(DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&g_utils))))       return false;
        if (FAILED(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&g_compiler)))) return false;
        if (FAILED(g_utils->CreateDefaultIncludeHandler(&g_includes)))               return false;
    }
    ++g_refCount;
    m_ready = true;
    return true;
}

void ShaderCompiler::Shutdown() {
    if (!m_ready) return;
    m_ready = false;
    if (--g_refCount > 0) return;   // someone else is still compiling

    g_includes.Reset();
    g_compiler.Reset();
    g_utils.Reset();
}

bool ShaderCompiler::CompileCompute(const std::string& source,
                                    const std::string& entryPoint,
                                    const std::string& debugName,
                                    ShaderBlob* out,
                                    std::string* errors) {
    errors->clear();
    if (!m_ready) { *errors = "shader compiler not initialised"; return false; }

    DxcBuffer buf = {};
    buf.Ptr      = source.data();
    buf.Size     = source.size();
    buf.Encoding = DXC_CP_UTF8;

    const std::wstring wEntry = Widen(entryPoint);
    const std::wstring wName  = Widen(debugName.empty() ? "shader" : debugName);

    std::vector<LPCWSTR> args = {
        wName.c_str(),              // shows up in diagnostics
        L"-E", wEntry.c_str(),
        L"-T", L"cs_6_0",
        L"-HV", L"2021",
    };
#ifdef _DEBUG
    args.push_back(L"-Zi");          // debug info for PIX
    args.push_back(L"-Qembed_debug");
    args.push_back(L"-Od");
#else
    args.push_back(L"-O3");
#endif

    ComPtr<IDxcResult> result;
    HRESULT hr = g_compiler->Compile(&buf, args.data(), UINT32(args.size()),
                                     g_includes.Get(), IID_PPV_ARGS(&result));
    if (FAILED(hr)) { *errors = "DXC invocation failed"; return false; }

    // Diagnostics come back even on success (warnings), so always collect them.
    ComPtr<IDxcBlobUtf8> diag;
    if (SUCCEEDED(result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&diag), nullptr)) &&
        diag && diag->GetStringLength() > 0) {
        *errors = std::string(diag->GetStringPointer(), diag->GetStringLength());
    }

    HRESULT status = S_OK;
    result->GetStatus(&status);
    if (FAILED(status)) {
        if (errors->empty()) *errors = "shader compilation failed";
        return false;
    }

    ComPtr<IDxcBlob> object;
    if (FAILED(result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&object), nullptr)) || !object) {
        *errors = "shader produced no output";
        return false;
    }

    const auto* bytes = static_cast<const uint8_t*>(object->GetBufferPointer());
    out->dxil.assign(bytes, bytes + object->GetBufferSize());
    return true;
}

bool ShaderCompiler::CompileFile(const std::string& path,
                                 const std::string& entryPoint,
                                 ShaderBlob* out,
                                 std::string* errors) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { *errors = "could not open '" + path + "'"; return false; }
    std::ostringstream ss;
    ss << f.rdbuf();

    std::string name = path;
    if (auto slash = name.find_last_of("/\\"); slash != std::string::npos)
        name = name.substr(slash + 1);

    return CompileCompute(ss.str(), entryPoint, name, out, errors);
}

} // namespace tglab
