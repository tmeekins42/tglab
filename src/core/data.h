// Data: what flows between algorithm ports.
//
// M1 had only Image. The variant exists so that FeatureSet, Matrix, PointCloud
// etc. become new alternatives later without changing PortRef, the stage cache,
// the interpreter's value type, or any existing algorithm.
#pragma once

#include <variant>
#include <vector>

#include "image.h"
#include "shape.h"

namespace tglab {

// Port data type tags, used for declaration and type-checking.
enum class DataType : uint8_t {
    None = 0,
    Image,
    ImageSet,   // several images on one or more named axes -- see shape.h
    // Future: Matrix, FeatureSet, Palette, PointCloud, Splats
};

const char* DataTypeName(DataType t);

// Several images with a shape describing how they are arranged.
//
// Deliberately a separate alternative rather than making Image itself hold a
// shape. An Image is one image, and every algorithm written so far relies on
// that; widening Image would put a shape check inside code that cannot
// meaningfully act on it. A distinct alternative means existing algorithms keep
// working on Image unchanged and the type system stops the mistake at the port
// rather than at a crash.
//
// The shape's Count() equals images.size(); NOT enforced by the type, so the
// pipeline checks it when a stage produces one.
struct ImageSet {
    std::vector<Image> images;
    Shape              shape;
};

using Data = std::variant<std::monostate, Image, ImageSet>;

inline DataType TypeOf(const Data& d) {
    if (std::holds_alternative<Image>(d))    return DataType::Image;
    if (std::holds_alternative<ImageSet>(d)) return DataType::ImageSet;
    return DataType::None;
}

// The shape of a value. A plain Image is scalar -- which is the whole point of
// scalar meaning "one image" rather than "no images".
inline Shape ShapeOf(const Data& d) {
    if (const auto* s = std::get_if<ImageSet>(&d)) return s->shape;
    return Shape::Scalar();
}

} // namespace tglab
