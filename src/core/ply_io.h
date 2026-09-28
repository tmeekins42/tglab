// ply_io — point clouds and Gaussian splats to and from .ply files.
//
// WRITTEN IN THE 3DGS LAYOUT when the cloud has Gaussians, because that is
// the de-facto interchange format: the reference implementation (Kerbl et al.
// 2023) writes it, and every splat viewer and editor reads it -- SuperSplat,
// the INRIA viewer, antimatter15's web viewer. Per vertex, as floats:
//
//   x y z            the mean
//   nx ny nz         zeros; the reference writer emits them and some readers
//                    expect them
//   f_dc_0..2        colour as the DC term of spherical harmonics:
//                    colour = 0.5 + C0 * f_dc, C0 = 0.28209479177387814
//   opacity          as a LOGIT: the raw parameter, sigmoid applied on read
//   scale_0..2       as LOGS: the raw parameter, exp applied on read
//   rot_0..3         quaternion w x y z, not necessarily unit
//
// Higher spherical-harmonic bands (f_rest_*) are not written: a splat here has
// one colour, and a file with none is valid -- readers take the band count
// from how many f_rest properties there are, zero included. On READ they are
// ignored for the same reason, which drops a downloaded splat's
// view-dependent shading (reflections, sheen) and keeps its base colour.
//
// A cloud WITHOUT Gaussians is written as plain points -- x y z and 8-bit
// red green blue -- which every point-cloud tool opens (MeshLab,
// CloudCompare, Blender).
//
// COORDINATES are written as they are held: the same camera convention as
// COLMAP and the 3DGS reference (x right, y down, z forward), so files made
// by those tools line up with scenes made here. What any given file calls
// "up" is its own business.
#pragma once

#include <string>

#include "data.h"

namespace tglab {

// Writes `cloud` to `path`: its Gaussians in the 3DGS layout when it has any,
// otherwise its triangulated points. Binary little-endian.
bool SavePly(const std::string& path, const PointCloud& cloud, std::string* err);

// Reads a .ply into `cloud`, replacing its contents. A file with scale_0,
// rot_0 and opacity properties becomes Gaussians; anything else with x y z
// becomes points (tracks with no observations), coloured when the file has
// red/green/blue. Binary (either endianness) or ASCII, properties in any
// order, extra ones ignored. `note` receives a one-line summary.
bool LoadPly(const std::string& path, PointCloud* cloud, std::string* note,
             std::string* err);

}  // namespace tglab
