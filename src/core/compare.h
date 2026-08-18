// Compare mode: run the same algorithm on CPU and GPU, diff the results, and
// time both.
//
// This is the feature the lab exists for — a GPU kernel is only useful if it
// agrees with the reference C++, and "it looks right" is not a measurement.
#pragma once

#include <string>
#include <vector>

#include "image.h"
#include "pipeline.h"

namespace tglab {

class ComputeContext;

struct CompareStats {
    double maxAbsDiff  = 0;   // largest single-channel difference
    double meanAbsDiff = 0;   // average across all channels
    double rmse        = 0;
    int    diffPixels  = 0;   // pixels differing by more than `tolerance`
    int    totalPixels = 0;

    // Fraction of pixels that differ beyond tolerance.
    double DiffFraction() const {
        return totalPixels > 0 ? double(diffPixels) / double(totalPixels) : 0.0;
    }
};

struct CompareResult {
    bool         ok = false;
    std::string  error;

    std::string  algorithm;
    double       cpuMs = 0;
    double       gpuMs = 0;
    CompareStats stats;

    Image cpuImage;
    Image gpuImage;
    Image diffImage;   // amplified absolute difference, for display

    double Speedup() const { return gpuMs > 0 ? cpuMs / gpuMs : 0.0; }
};

// Compares two images channel by channel. `tolerance` is in the image's own
// units (0-255 for RGBA8, absolute for float formats).
CompareStats CompareImages(Image& a, Image& b, double tolerance);

// Builds a visualisation of |a - b|, scaled by `amplify` so that differences
// of a few LSBs are actually visible.
bool MakeDiffImage(Image& a, Image& b, float amplify, Image* out);

// Runs one stage of an already-interpreted pipeline both ways and compares.
// `stageIndex` selects which stage; -1 means the last one.
CompareResult CompareCpuGpu(Pipeline& pipe, std::vector<Data>* sources,
                            ComputeContext* gpu, int stageIndex = -1);

} // namespace tglab
