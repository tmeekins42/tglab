// load_ply — a point cloud or Gaussian splats from a .ply file.
//
// A SOURCE STAGE: it takes no inputs, so a script calls it with none and
// names the file --
//
//     imported = load_ply( file = "D:/splats/garden.ply" )
//     display( imported, "imported" )
//
// -- and what it produces is an ordinary reconstruction. Everything
// downstream treats it as it would one made here: the 3D viewer draws it,
// render_splats renders it (from its cameras, which a file does not carry),
// and save() writes it back out.
//
// The format is the 3DGS one when the file has Gaussians, plain coloured
// points otherwise; see ply_io.h, including what is dropped on the way in.
#include <string>

#include "../../core/algorithm.h"
#include "../../core/ply_io.h"

namespace tglab {
namespace {

class LoadPlyStage : public AlgorithmBase {
public:
    const char* Name()     const override { return "load_ply"; }
    const char* Category() const override { return "sfm"; }

    PortList Inputs() const override { return {}; }
    PortList Outputs() const override {
        return {{"out", DataType::PointCloud, FormatSpec::Any, ShapeSpec::Any}};
    }

    void RunCPU(RunCtx&) override {}
    bool IsReconstruct() const override { return true; }
    ProxyBehaviour Proxy() const override { return ProxyBehaviour::Never; }
    bool HasGPU() const override { return false; }

    bool RunReconstruct(const std::vector<Image>*, PointCloud* cloud,
                        std::string* err) override {
        const std::string& path = m_file.get();
        if (path.empty()) {
            *err = "load_ply: no file -- load_ply( file = \"scene.ply\" )";
            return false;
        }
        std::string note, lerr;
        if (!tglab::LoadPly(path, cloud, &note, &lerr)) {
            *err = "load_ply: " + lerr;
            return false;
        }
        m_note = "load_ply: " + note + " from " + path;
        return true;
    }

    std::string RunReport() const override { return m_note; }

private:
    Param<std::string> m_file{
        this, "file", {},
        "The .ply to read: Gaussian splats in the 3DGS layout, or a plain "
        "point cloud. A relative path is taken from the working directory."};

    std::string m_note;
};

REGISTER_ALGORITHM(LoadPlyStage);

}  // namespace
}  // namespace tglab
