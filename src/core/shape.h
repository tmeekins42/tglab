// Shape: how many images a port carries, and along what axes.
//
// The design decision this file encodes is that there is ONE image type with
// named axes, rather than separate Image and ImageGroup types. A bracket of 5
// exposures is shape [5] on an axis named "exposure"; those brackets taken at 12
// focus positions are shape [5,12] on axes "exposure" and "focus"; a video is
// shape [n] on "time". Merging is then reduction along a NAMED axis
// (over="exposure"), which is the same operation regardless of what the axis
// means -- and an algorithm that reduces exposures does not need to know it is
// sitting inside a panorama.
//
// The alternative -- a distinct group type -- forces every algorithm to handle
// both, and forces a new type for every combination anyone wants to nest.
//
// SCALAR IS THE DEFAULT AND MEANS "one image", which is what every algorithm
// written so far takes. Shape is introduced here with nothing yet producing a
// non-scalar value: the point is to prove the concept fits the existing Data,
// ports and cache before any multi-image algorithm depends on it.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tglab {

// One axis: a name and an extent. The name is what a script reduces over.
struct Axis {
    std::string name;
    int         extent = 0;

    bool operator==(const Axis&) const = default;
};

// An ordered list of axes. Empty means scalar -- a single image.
class Shape {
public:
    Shape() = default;
    explicit Shape(std::vector<Axis> axes) : m_axes(std::move(axes)) {}

    static Shape Scalar() { return Shape{}; }
    static Shape Of(std::string name, int extent) {
        return Shape{{Axis{std::move(name), extent}}};
    }

    bool IsScalar() const { return m_axes.empty(); }
    int  Rank()     const { return int(m_axes.size()); }

    const std::vector<Axis>& Axes() const { return m_axes; }

    // Total images this shape describes. 1 for scalar, since a scalar is one
    // image rather than none -- an empty product is 1 and that is the useful
    // answer here.
    int64_t Count() const {
        int64_t n = 1;
        for (const auto& a : m_axes) n *= a.extent;
        return n;
    }

    // Index of a named axis, or -1. This is what over="exposure" resolves
    // through, and returning -1 rather than throwing lets the caller produce
    // the error message with the script line in it.
    int Find(const std::string& name) const {
        for (int i = 0; i < int(m_axes.size()); ++i)
            if (m_axes[size_t(i)].name == name) return i;
        return -1;
    }

    // This shape with one axis removed -- the result of reducing over it.
    Shape Without(int axis) const {
        std::vector<Axis> out;
        out.reserve(m_axes.size());
        for (int i = 0; i < int(m_axes.size()); ++i)
            if (i != axis) out.push_back(m_axes[size_t(i)]);
        return Shape{std::move(out)};
    }

    bool operator==(const Shape&) const = default;

    // "[]" for scalar, otherwise "[exposure=5, focus=12]". For error messages,
    // where naming the axes is the difference between a usable diagnostic and
    // "expected [5], got [12]".
    std::string ToString() const;

private:
    std::vector<Axis> m_axes;
};

// What a PORT accepts or produces, as opposed to what a value IS.
//
// Separate from Shape because a declaration is a constraint, not a value: most
// algorithms accept exactly one image, a reduction accepts any rank and removes
// one axis, and a few (a video codec, a loader) produce a rank the framework
// cannot know until the stage runs.
enum class ShapeSpec : uint8_t {
    Scalar = 0,   // exactly one image -- the default, and what every existing
                  // algorithm declares by saying nothing
    Any,          // any rank, including scalar; the algorithm handles it
    SameAsInput,  // an output that matches the input's shape
    Reduced,      // an output that is the input's shape minus the reduced axis
};

const char* ShapeSpecName(ShapeSpec s);

} // namespace tglab
