// reshape: reinterprets a flat group as a multi-axis one.
//
// A drop gives a flat list -- 180 files is [frame=180]. What those 180 files
// MEAN is 12 camera positions x 5 focus steps x 3 exposures, and only the
// photographer knows that. shape() in a script says so:
//
//     frames = image("shoot")
//     frames = shape(frames, position=12, focus=5, exposure=3)
//
// It moves no pixels. The images are already in the right order -- row-major,
// last axis fastest, which is the order a shoot is taken in and the order a
// name sort produces -- so this only rewrites the shape that describes them.
//
// The interpreter needs the new shape at BUILD time, to check later lines and
// to resolve over=, so it computes the shape itself and hands it here through
// the axis spec. This stage exists so the run-time Data agrees with what the
// interpreter already decided.
#include <string>
#include <vector>

#include "../../core/algorithm.h"

namespace tglab {
namespace {

class Reshape : public AlgorithmBase {
public:
    const char* Name()     const override { return "reshape"; }
    const char* Category() const override { return "merge"; }

    PortList Inputs() const override {
        return {{"src", DataType::ImageSet, FormatSpec::Any, ShapeSpec::Any}};
    }
    PortList Outputs() const override {
        return {{"out", DataType::ImageSet, FormatSpec::Any, ShapeSpec::Any}};
    }

    // Handled by the pipeline, which needs the whole Data rather than the
    // per-image view RunCPU is given.
    void RunCPU(RunCtx&) override {}

    bool IsReshape() const override { return true; }
};

} // namespace

REGISTER_ALGORITHM(Reshape);

} // namespace tglab
