// gpu_sweep — the plane sweep's inner loop on the GPU.
//
// WHY THIS STAGE AND NOT ANOTHER. plane_sweep is the only algorithm here
// where the CPU version, already seven times faster than its first form, is
// still the dominant cost of the whole pipeline: 33 s at full resolution
// against 13 s for every other stage combined. It is also an unusually clean
// fit for a compute shader -- every plane is independent, and there is no
// reduction until the very end.
//
// ONE DISPATCH PER PLANE, and that number is the design.
//
// The first version mirrored the CPU's structure: warp into a buffer, sum
// horizontally into another, sum vertically and correlate into a third, fold
// the neighbours into the running state -- thirteen dispatches per plane,
// six intermediate textures rewritten each time, 624 dispatches per frame.
// The kernels were correct (0.000% against the CPU) and the device hung
// anyway, taking the desktop with it.
//
// ComputeContext's notes record the same failure from the develop chains: a
// command list holding several large dispatches hangs, one per submission
// never does, and nobody has found out why. Every GPU path that works here
// records a handful of dispatches per frame. The sweep recorded hundreds,
// and reusing intermediates across them meant write-after-read on the same
// texture hundreds of times per frame -- a pattern nothing else exercises.
//
// So the whole plane is now ONE kernel: each thread warps its own window
// through each neighbour's homography, sums, correlates, takes the best half
// and folds into the running state. No intermediates at all. 48 dispatches
// per frame instead of 624, each submitted on its own, which is the one
// pattern this machine has never hung on.
//
// It is also the better GPU design. The three-pass form existed to make the
// CPU's window sums O(1); a GPU thread re-reading a 7x7 window from a cached
// texture costs less than writing it out and reading it back through memory.
//
// MEASURED in the app on fountain-P11, 11 frames, 48 planes, 4 neighbours:
//
//                     CPU (warp once)   GPU (fused)
//   0.4 scale                  15 s           2 s
//   full resolution            33 s          18 s
//
// No hang at either size, and the desktop stayed responsive. Why the gain
// shrinks at full resolution is not measured. The likely reasons: each
// thread re-reads its window per neighbour, so the kernel is probably
// bandwidth-bound where the CPU's sliding sums are not, and one submission
// per plane leaves the GPU idle between them.
//
// WHAT RUNS HERE AND WHAT DOES NOT. Scoring and the running winner state.
// Choosing the final depth, the subpixel fit and the confidence margin stay
// on the CPU, from one readback per frame.
#pragma once

#include <memory>
#include <string>
#include <vector>

namespace tglab {

class ComputeContext;

// One image plane, matching plane_sweep's own Plane.
struct SweepPlane {
    const float* v = nullptr;
    int w = 0, h = 0;
};

// Holds one reference frame and its neighbours on the device for the whole
// sweep. Create once per frame, call Plane() once per plane, Finish() once.
class GpuSweepSession {
public:
    GpuSweepSession();
    ~GpuSweepSession();

    GpuSweepSession(const GpuSweepSession&)            = delete;
    GpuSweepSession& operator=(const GpuSweepSession&) = delete;

    // Uploads the reference, its window sums, every neighbour, and the whole
    // homography table -- nPlanes x neighbours.size() row-major 3x3s, each
    // taking REFERENCE pixels to that neighbour's pixels as PlaneHomography
    // produces them.
    //
    // Returns false on any device failure, and the caller falls back to the
    // CPU path rather than producing a half-swept frame.
    //
    // The caller holds a GpuLock across the whole session: ComputeContext has
    // one command queue with no locking of its own.
    bool Begin(ComputeContext* gpu, const SweepPlane& ref,
               const std::vector<SweepPlane>& neighbours, int radius,
               int nPlanes, int skip, const double* H, std::string* err);

    // Scores one plane against every neighbour and folds it into the running
    // state. One dispatch, submitted on its own.
    bool Plane(int planeIndex, std::string* err);

    // Reads the accumulated per-pixel state back: the winning score, the
    // scores either side of it (for the parabola fit), the best separated
    // rival (for the margin) and which plane won. Each sized ref.w * ref.h.
    bool Finish(std::vector<float>* best, std::vector<float>* prev,
                std::vector<float>* next, std::vector<float>* rival,
                std::vector<int>* bestPlane, std::string* err);

private:
    struct Impl;
    std::unique_ptr<Impl> m;
};

// True when the device is present and the kernel compiles. Checked once so a
// GPU-less build or a failed shader compile costs one attempt rather than one
// per frame.
bool GpuSweepReady(ComputeContext* gpu);

}  // namespace tglab
