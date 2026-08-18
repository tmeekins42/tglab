// HLSL compute shader compilation via DXC (Shader Model 6.x).
//
// Compiles at runtime so a shader can be edited and reloaded like a script.
// Errors are returned as text for the UI rather than crashing — a compute
// shader you are actively writing will fail to compile often.
#pragma once

#include <d3d12.h>

#include <string>
#include <vector>

namespace tglab {

struct ShaderBlob {
    std::vector<uint8_t> dxil;
    bool Valid() const { return !dxil.empty(); }
};

class ShaderCompiler {
public:
    bool Init();
    void Shutdown();

    // Compiles `source` as a compute shader. On failure returns false and puts
    // DXC's diagnostics (with line numbers) in `errors`.
    bool CompileCompute(const std::string& source,
                        const std::string& entryPoint,
                        const std::string& debugName,
                        ShaderBlob* out,
                        std::string* errors);

    // Reads and compiles a .hlsl file.
    bool CompileFile(const std::string& path,
                     const std::string& entryPoint,
                     ShaderBlob* out,
                     std::string* errors);

    bool Ready() const { return m_ready; }

private:
    bool m_ready = false;
};

} // namespace tglab
