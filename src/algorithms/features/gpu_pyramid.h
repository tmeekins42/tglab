#pragma once

// GPU scale-space pyramid: separable Gaussian blur and 2x decimation, for the
// feature detectors.
//
// WHY THIS IS NOT A RunGPU PATH
//
// The pipeline's GPU path is image-in/image-out: RunStageGpu binds the input
// textures, dispatches, and leaves the result on the device for the next stage.
// A detector does not fit that shape. Its product is a SIDECAR -- a list of
// keypoints -- and its image output is the input passed through untouched. So
// there is nothing for RunGPU to write, and the interesting work (extrema
// detection, orientation, descriptors) is branchy, serial, and wants to be on
// the CPU anyway.
//
// What IS worth offloading is the front of the detector: 65% of SIFT's runtime
// at 22 MP is the blur pyramid (5,626 ms of 8,646 ms), and a separable blur is
// the ideal shape for a compute dispatch -- every output pixel independent.
//
// So this is a helper a CPU algorithm calls, not an execution mode. It takes
// the device from RunCtx::Gpu(), runs the blurs there, reads the planes back,
// and hands them to the same CPU code that would have built them. The detector
// stays a CPU algorithm; only its inner loops move.
//
// MATCHING THE CPU EXACTLY
//
// The kernels reproduce detect_sift.cpp's Blur() and Halve() decision for
// decision, because a pyramid that differs even slightly produces different
// keypoints, and a detector that finds different features depending on whether
// a GPU was present would be useless for the comparisons this lab exists for:
//
//   * radius = max(1, ceil(sigma * 3))      -- same 3-sigma rule
//   * kernel normalised by its own sum      -- so it is unit-gain at any radius
//   * clamp-to-edge addressing              -- Plane::At clamps both axes
//   * horizontal pass, then vertical        -- the intermediate is not rounded,
//                                              so pass order is observable
//   * Halve() point-samples (x*2, y*2)      -- no averaging; the source is
//                                              already band-limited
//
// The kernel weights are computed on the CPU and uploaded, rather than
// evaluated in the shader. exp() is not required to agree bit-for-bit between
// the CPU and the GPU, and a kernel that differs in the last ulp shifts the
// DoG extrema near threshold. Computing them once on the CPU removes the
// question entirely -- and it is 2r+1 values, so it costs nothing.

#include <string>
#include <vector>

namespace tglab {

class ComputeContext;

// One level of the scale space, matching detect_sift.cpp's Plane.
struct GpuPlane {
    std::vector<float> v;
    int w = 0, h = 0;
};

// Builds the Gaussian blur kernel for a sigma, using the CPU rule.
// Returned so the caller can share it with its own CPU path.
std::vector<float> GaussianKernel(float sigma, int* radiusOut);

// Blurs `src` into `dst` by `sigma` on the device.
//
// Returns false and leaves `dst` untouched when the device is unavailable or a
// dispatch fails -- the caller falls back to its CPU blur. A GPU failure must
// degrade to a slow correct result, never to a wrong one.
bool GpuBlur(ComputeContext* gpu, const GpuPlane& src, GpuPlane* dst,
             float sigma, std::string* err);

// FAST-9 corner test combined with a Harris response, as one map.
//
// This is ORB's inner loop rather than a pyramid stage, but it belongs here for
// the same reason: it is a per-pixel test over a local neighbourhood, which is
// the shape a dispatch is good at, and the detector around it stays on the CPU.
//
// The result is DENSE -- the Harris response where FAST fired, and 0 where it
// did not -- rather than a candidate list. A list would need an atomic append
// and would come back in a nondeterministic order, so the same image could rank
// its candidates differently from run to run. A dense map costs one float per
// pixel and keeps the CPU's scan order, which is what makes the two paths
// produce byte-identical keypoints.
//
// `threshold` is FAST's intensity margin; `harrisRadius` the response window;
// `border` how far from the edge to skip, matching the caller's own margin.
// Pixels inside the border, and pixels FAST rejects, come back as 0.
bool GpuFastHarris(ComputeContext* gpu, const GpuPlane& src, GpuPlane* dst,
                   float threshold, int harrisRadius, int border,
                   std::string* err);

// FAST-9 with a bisected score, as one map. BRISK's inner loop.
//
// BRISK does not just ask whether a pixel is a corner; it asks how strong a
// threshold the corner survives, by running FAST eight more times in a binary
// search. That is up to nine ring walks per pixel, which is where BRISK's time
// goes and why offloading only its blur would not have helped much.
//
// Dense like GpuFastHarris, and for the same reason: the caller's scan order
// has to survive so the candidate list is identical. 0 means FAST rejected the
// pixel outright; anything above is the bisected score.
//
// The scale-space non-maximum suppression that follows reads NEIGHBOURING
// layers, so it stays on the CPU -- it is not a per-pixel test over one plane
// and does not fit this shape.
bool GpuFastScore(ComputeContext* gpu, const GpuPlane& src, GpuPlane* dst,
                  float threshold, int border, std::string* err);

// N steps of nonlinear diffusion, chained on the device. AKAZE's scale space.
//
// Each step is a 5-point stencil -- every output pixel depends only on its four
// neighbours and the conductivity map -- so a step parallelises perfectly even
// though the SEQUENCE of steps does not. Running the whole chain in one call is
// what makes it pay: the intermediate never leaves the device, so N steps cost
// one upload and one readback rather than N of each.
//
// `cond` is the conductivity map, constant across the chain. `dt` is the time
// per step, which the caller has already subdivided to stay under the explicit
// scheme's stability limit -- this does not re-derive it, because the caller's
// subdivision is part of the result.
//
// The interior only, matching the CPU: the border ring is left as it was.
bool GpuDiffuse(ComputeContext* gpu, const GpuPlane& src, const GpuPlane& cond,
                int steps, float dt, GpuPlane* dst, std::string* err);

// A whole octave's blur stack, kept on the device between levels.
//
// This is the call that actually pays. Each level is the previous level blurred
// by the DIFFERENCE in sigma, so the levels form a chain -- and doing the chain
// in one call keeps the intermediate planes on the GPU instead of reading each
// one back and uploading it again. At 22 MP a round trip is ~8 ms per level, so
// a 9-level octave saves ~130 ms of pure transfer over calling GpuBlur nine
// times.
//
// `sigmas[i]` is the blur to ADD at level i; level 0 is `base` unmodified.
// On success `out` has sigmas.size() + 1 planes. On failure `out` is untouched.
bool GpuBlurStack(ComputeContext* gpu, const GpuPlane& base,
                  const std::vector<float>& sigmas,
                  std::vector<GpuPlane>* out, std::string* err);

}  // namespace tglab
