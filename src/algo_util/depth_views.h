// depth_views — plane_sweep's per-camera maps, read back for the stages that
// consume them, and the free-space test those stages share.
//
// plane_sweep emits three images per camera: depth, confidence and colour.
// fuse_depth reads them to find where cameras AGREE; carve_splats reads them
// to find where a camera looked straight THROUGH something. Both need the
// same unpacking and the same see-through test, so both live here.
#pragma once

#include <string>
#include <vector>

#include "../core/data.h"
#include "../core/image.h"

namespace tglab {

// One camera's maps as raw planes. Raw rather than through Image accessors
// for the reason the sweep writes them that way: the consumers' inner loops
// run per pixel per camera, and a format switch there would dominate.
struct DepthView {
    const float* depth = nullptr;
    const float* conf  = nullptr;
    // The reference frame's own pixels, RGBA32F, carried through by the sweep
    // so a dense point takes the colour of the pixel it was measured from.
    const float* rgb   = nullptr;
    int w = 0, h = 0;
    bool ok = false;

    Vec3 Colour(int x, int y) const {
        if (!rgb || x < 0 || y < 0 || x >= w || y >= h)
            return Vec3{0.72, 0.72, 0.74};
        const float* p = &rgb[(size_t(y) * size_t(w) + size_t(x)) * 4];
        return Vec3{double(p[0]), double(p[1]), double(p[2])};
    }

    // Depth at an integer pixel, or 0 where nothing was measured. The sweep
    // writes exactly 0.0 for an unmeasured pixel, which is distinguishable
    // from a real depth because a real one is strictly positive.
    float At(int x, int y) const {
        if (x < 0 || y < 0 || x >= w || y >= h) return 0.0f;
        return depth[size_t(y) * size_t(w) + size_t(x)];
    }
    float Conf(int x, int y) const {
        if (!conf || x < 0 || y < 0 || x >= w || y >= h) return 0.0f;
        return conf[size_t(y) * size_t(w) + size_t(x)];
    }
};

// Maps `images` -- plane_sweep's output, three per camera -- into one
// DepthView per camera. `holds` keeps the CPU mappings alive and must outlive
// `views`. `stage` names the caller in error messages. Checked rather than
// assumed: handing a stage the colour frames by mistake is an easy error and
// would otherwise produce nonsense silently.
bool ReadDepthViews(const std::vector<Image>& images, int nCam, const char* stage,
                    std::vector<DepthView>* views, std::vector<ImageView>* holds,
                    std::string* err);

// How many cameras other than `skip` looked THROUGH `world`: it projects into
// their image, they measured a surface there with at least `minConf`
// confidence, and that surface is more than `tol` (a fraction of the
// distance) BEHIND the point along their line of sight. Each such camera saw
// empty space where the point claims to be. Pass skip = -1 to ask them all.
int CountSeenThrough(const PointCloud& cloud, const std::vector<DepthView>& views,
                     const Vec3& world, int skip, double minConf, double tol);

}  // namespace tglab
