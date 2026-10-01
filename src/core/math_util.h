// math_util — small scalar functions shared across the code.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

namespace tglab {

// A float as the uint its bits make, for a GPU root constant: the shader
// reads it back with asfloat(). Every kernel's constants need this.
inline uint32_t FloatBits(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof u);
    return u;
}

// The logistic function and its inverse. Splats keep opacity and
// reflectivity as LOGITS so an optimiser can move them freely while the
// value stays in (0, 1). Logit is unbounded at 0 and 1: callers clamp first.
inline double Sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
inline double Logit(double p) { return std::log(p / (1.0 - p)); }

}  // namespace tglab
