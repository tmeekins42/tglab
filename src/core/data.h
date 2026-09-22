// Data: what flows between algorithm ports.
//
// M1 had only Image. The variant exists so that FeatureSet, Matrix, PointCloud
// etc. become new alternatives later without changing PortRef, the stage cache,
// the interpreter's value type, or any existing algorithm.
#pragma once

#include <variant>
#include <vector>

#include "geometry.h"
#include "image.h"
#include "shape.h"

namespace tglab {

// Port data type tags, used for declaration and type-checking.
enum class DataType : uint8_t {
    None = 0,
    Image,
    ImageSet,     // several images on one or more named axes -- see shape.h
    PointCloud,   // a 3D reconstruction: cameras, tracks, points
    // Future: Matrix, FeatureSet, Palette, Splats
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

// A 3D reconstruction.
//
// WHY THIS IS A DATA TYPE AND NOT A SIDECAR. Per-image results attach to their
// image: a feature set, a transform, a camera pose. A reconstruction does not
// -- the tracks and the points belong to the GROUP, seen by many frames and
// owned by none. Putting it on one image would make that image's identity
// load-bearing for data about all of them.
//
// Cameras are indexed in step with the ImageSet the reconstruction came from,
// so `cameras[i]` describes `images[i]`. Deliberately a parallel array rather
// than a pointer back to the images: a reconstruction outlives the pipeline run
// that produced it, and a stage that re-develops its inputs must not
// invalidate the geometry.
struct PointCloud {
    std::vector<Camera> cameras;
    std::vector<Track>  tracks;

    // The view graph, flattened out of the source frames' sidecars by whichever
    // reconstruction stage ran first.
    //
    // CARRIED HERE RATHER THAN RE-READ, because after the first stage the
    // pipeline is passing a PointCloud and the frames are no longer an input.
    // Every later stage -- rotation averaging, positioning, bundle adjustment
    // -- needs the same relative poses, and re-deriving them would mean either
    // a second input port on every stage or solving the essential matrices
    // again.
    //
    // A copy rather than a reference for the reason the cameras are: a
    // reconstruction outlives the run that produced it, and a stage that
    // re-develops its inputs must not invalidate the geometry.
    struct ViewEdge {
        int  i = -1, j = -1;   // R takes frame i's coordinates into frame j's
        Mat3 R;
        Vec3 direction;        // unit, in frame i's coordinates; scale unknown
        int  inliers = 0;
    };
    std::vector<ViewEdge> edges;

    // The group this was reconstructed from, for reporting and for viewers
    // that want to show a camera's image.
    Shape shape;

    int SolvedCameras() const {
        int n = 0;
        for (const Camera& c : cameras) if (c.solved) ++n;
        return n;
    }
    int TriangulatedPoints() const {
        int n = 0;
        for (const Track& t : tracks) if (t.hasPoint) ++n;
        return n;
    }
};

using Data = std::variant<std::monostate, Image, ImageSet, PointCloud>;

inline DataType TypeOf(const Data& d) {
    if (std::holds_alternative<Image>(d))      return DataType::Image;
    if (std::holds_alternative<ImageSet>(d))   return DataType::ImageSet;
    if (std::holds_alternative<PointCloud>(d)) return DataType::PointCloud;
    return DataType::None;
}

// The shape of a value. A plain Image is scalar -- which is the whole point of
// scalar meaning "one image" rather than "no images".
inline Shape ShapeOf(const Data& d) {
    if (const auto* s = std::get_if<ImageSet>(&d)) return s->shape;
    return Shape::Scalar();
}

} // namespace tglab
