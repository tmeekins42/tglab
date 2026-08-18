// Data: what flows between algorithm ports.
//
// M1 has only Image. The variant exists now so that FeatureSet, Matrix,
// PointCloud etc. become new alternatives later without changing PortRef,
// the stage cache, the interpreter's value type, or any existing algorithm.
#pragma once

#include <variant>

#include "image.h"

namespace tglab {

// Port data type tags, used for declaration and type-checking.
enum class DataType : uint8_t {
    None = 0,
    Image,
    // Future: Matrix, FeatureSet, Palette, PointCloud, Splats
};

const char* DataTypeName(DataType t);

using Data = std::variant<std::monostate, Image>;

inline DataType TypeOf(const Data& d) {
    return std::holds_alternative<Image>(d) ? DataType::Image : DataType::None;
}

} // namespace tglab
