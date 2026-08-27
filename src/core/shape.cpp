#include "shape.h"

namespace tglab {

std::string Shape::ToString() const {
    if (m_axes.empty()) return "[]";
    std::string s = "[";
    for (size_t i = 0; i < m_axes.size(); ++i) {
        if (i) s += ", ";
        s += m_axes[i].name;
        s += "=";
        s += std::to_string(m_axes[i].extent);
    }
    s += "]";
    return s;
}

const char* ShapeSpecName(ShapeSpec s) {
    switch (s) {
        case ShapeSpec::Scalar:      return "a single image";
        case ShapeSpec::Any:         return "any shape";
        case ShapeSpec::SameAsInput: return "same as input";
        case ShapeSpec::Reduced:     return "input shape minus one axis";
    }
    return "?";
}

} // namespace tglab
