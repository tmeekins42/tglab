#include "depth_views.h"

namespace tglab {

bool ReadDepthViews(const std::vector<Image>& images, int nCam, const char* stage,
                    std::vector<DepthView>* views, std::vector<ImageView>* holds,
                    std::string* err) {
    if (int(images.size()) != nCam * 3) {
        *err = std::string(stage) + ": expected " + std::to_string(nCam * 3) +
               " images (a depth, confidence and colour map per camera), got " +
               std::to_string(images.size()) +
               " -- the depth input should be plane_sweep's output";
        return false;
    }

    views->assign(size_t(nCam), DepthView{});
    holds->clear();
    holds->reserve(size_t(nCam) * 3);

    for (int c = 0; c < nCam; ++c) {
        ImageView dv = const_cast<Image&>(images[size_t(c) * 3 + 0]).MapCpuRead();
        ImageView cv = const_cast<Image&>(images[size_t(c) * 3 + 1]).MapCpuRead();
        ImageView rv = const_cast<Image&>(images[size_t(c) * 3 + 2]).MapCpuRead();
        holds->push_back(dv);
        holds->push_back(cv);
        holds->push_back(rv);

        if (!dv.Valid() || dv.desc.format != Format::R32F) continue;
        if (!dv.desc.isDepth) {
            *err = std::string(stage) + ": depth input image " +
                   std::to_string(c * 3) + " is not a depth map -- the depth "
                   "input should be plane_sweep's output";
            return false;
        }

        DepthView& v = (*views)[size_t(c)];
        v.depth = dv.At<float>(0, 0);
        v.w = dv.desc.width;
        v.h = dv.desc.height;
        v.conf = (cv.Valid() && cv.desc.format == Format::R32F) ? cv.At<float>(0, 0)
                                                                : nullptr;
        v.rgb = (rv.Valid() && rv.desc.format == Format::RGBA32F) ? rv.At<float>(0, 0)
                                                                  : nullptr;
        v.ok = true;
    }
    return true;
}

int CountSeenThrough(const PointCloud& cloud, const std::vector<DepthView>& views,
                     const Vec3& world, int skip, double minConf, double tol,
                     int* measured) {
    const int nCam = int(cloud.cameras.size());
    int seen = 0, meas = 0;
    for (int cj = 0; cj < nCam; ++cj) {
        if (cj == skip) continue;
        const Camera& oc = cloud.cameras[size_t(cj)];
        const DepthView& ov = views[size_t(cj)];
        if (!oc.solved || !ov.ok) continue;

        const Vec3 local = oc.R * world + oc.t;
        if (local.z <= 1e-9) continue;               // behind it
        double px = 0.0, py = 0.0;
        if (!oc.Project(world, &px, &py)) continue;
        const int ix = int(px + 0.5), iy = int(py + 0.5);
        if (ix < 0 || iy < 0 || ix >= ov.w || iy >= ov.h) continue;

        const float od = ov.At(ix, iy);
        if (od <= 0.0f) continue;                    // it measured nothing
        if (ov.Conf(ix, iy) < float(minConf)) continue;
        ++meas;
        if ((double(od) - local.z) / local.z > tol) ++seen;
    }
    if (measured) *measured = meas;
    return seen;
}

}  // namespace tglab
