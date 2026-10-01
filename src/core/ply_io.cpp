#include "ply_io.h"

#include "math_util.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

namespace tglab {
namespace {

// The zeroth spherical-harmonic basis function, a constant: 1 / (2 sqrt(pi)).
constexpr double kC0 = 0.28209479177387814;

enum class PType { I8, U8, I16, U16, I32, U32, F32, F64, Bad };

PType ParseType(const std::string& t) {
    if (t == "char" || t == "int8")    return PType::I8;
    if (t == "uchar" || t == "uint8")  return PType::U8;
    if (t == "short" || t == "int16")  return PType::I16;
    if (t == "ushort" || t == "uint16") return PType::U16;
    if (t == "int" || t == "int32")    return PType::I32;
    if (t == "uint" || t == "uint32")  return PType::U32;
    if (t == "float" || t == "float32") return PType::F32;
    if (t == "double" || t == "float64") return PType::F64;
    return PType::Bad;
}

int SizeOf(PType t) {
    switch (t) {
        case PType::I8: case PType::U8:   return 1;
        case PType::I16: case PType::U16: return 2;
        case PType::I32: case PType::U32: case PType::F32: return 4;
        case PType::F64: return 8;
        default: return 0;
    }
}

// One scalar from raw bytes, byte-swapped first when the file's endianness
// differs from this machine's (little-endian: x86 and ARM Windows both).
double Decode(const uint8_t* p, PType t, bool swap) {
    uint8_t b[8];
    const int n = SizeOf(t);
    for (int i = 0; i < n; ++i) b[i] = swap ? p[n - 1 - i] : p[i];
    switch (t) {
        case PType::I8:  { int8_t v;   std::memcpy(&v, b, 1); return v; }
        case PType::U8:  { uint8_t v;  std::memcpy(&v, b, 1); return v; }
        case PType::I16: { int16_t v;  std::memcpy(&v, b, 2); return v; }
        case PType::U16: { uint16_t v; std::memcpy(&v, b, 2); return v; }
        case PType::I32: { int32_t v;  std::memcpy(&v, b, 4); return v; }
        case PType::U32: { uint32_t v; std::memcpy(&v, b, 4); return v; }
        case PType::F32: { float v;    std::memcpy(&v, b, 4); return v; }
        case PType::F64: { double v;   std::memcpy(&v, b, 8); return v; }
        default: return 0.0;
    }
}

struct Prop {
    std::string name;
    PType       type = PType::Bad;
    int         offset = 0;   // within one vertex record, binary only
};

}  // namespace

// --- writing ---------------------------------------------------------------------

bool SavePly(const std::string& path, const PointCloud& cloud, std::string* err) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        *err = "could not open '" + path + "' for writing";
        return false;
    }

    if (!cloud.splats.empty()) {
        const size_t n = cloud.splats.size();
        // The view-dependent colour, in the reference layout: f_rest after
        // f_dc, one channel's coefficients after another.
        const int deg = (cloud.shDegree > 0 && cloud.splatSh.size() >= n * 45)
                            ? std::min(cloud.shDegree, 3) : 0;
        const int perChannel = (deg + 1) * (deg + 1) - 1;
        const int nRest = 3 * perChannel;
        f << "ply\nformat binary_little_endian 1.0\n"
          << "comment written by tglab: 3D Gaussian splats, degree-" << deg << " colour\n"
          << "element vertex " << n << "\n";
        const char* head[] = {"x", "y", "z", "nx", "ny", "nz",
                              "f_dc_0", "f_dc_1", "f_dc_2"};
        const char* tail[] = {"opacity", "scale_0", "scale_1", "scale_2",
                              "rot_0", "rot_1", "rot_2", "rot_3"};
        for (const char* nm : head) f << "property float " << nm << "\n";
        for (int q = 0; q < nRest; ++q) f << "property float f_rest_" << q << "\n";
        for (const char* nm : tail) f << "property float " << nm << "\n";
        f << "end_header\n";

        std::vector<float> rec(size_t(17 + nRest));
        for (size_t si = 0; si < n; ++si) {
            const Splat& s = cloud.splats[si];
            const double o = std::clamp(s.opacity, 1e-6, 1.0 - 1e-6);
            const double c[3] = {s.color.x, s.color.y, s.color.z};
            const double sc[3] = {s.scale.x, s.scale.y, s.scale.z};
            rec[0] = float(s.mean.x);
            rec[1] = float(s.mean.y);
            rec[2] = float(s.mean.z);
            rec[3] = rec[4] = rec[5] = 0.0f;
            for (int k = 0; k < 3; ++k) rec[size_t(6 + k)] = float((c[k] - 0.5) / kC0);
            for (int ch = 0; ch < 3; ++ch)
                for (int k = 0; k < perChannel; ++k)
                    rec[size_t(9 + ch * perChannel + k)] =
                        cloud.splatSh[si * 45 + size_t(k * 3 + ch)];
            const size_t t0 = size_t(9 + nRest);
            rec[t0] = float(Logit(o));
            for (int k = 0; k < 3; ++k)
                rec[t0 + 1 + size_t(k)] = float(std::log(std::max(sc[k], 1e-12)));
            for (int k = 0; k < 4; ++k) rec[t0 + 4 + size_t(k)] = float(s.rot[k]);
            f.write(reinterpret_cast<const char*>(rec.data()),
                    std::streamsize(rec.size() * sizeof(float)));
        }
    } else {
        size_t n = 0;
        for (const Track& t : cloud.tracks) n += t.hasPoint ? 1 : 0;
        if (n == 0) {
            *err = "nothing to write: the cloud has no Gaussians and no "
                   "triangulated points";
            return false;
        }
        f << "ply\nformat binary_little_endian 1.0\n"
          << "comment written by tglab: point cloud\n"
          << "element vertex " << n << "\n"
          << "property float x\nproperty float y\nproperty float z\n"
          << "property uchar red\nproperty uchar green\nproperty uchar blue\n"
          << "end_header\n";
        for (const Track& t : cloud.tracks) {
            if (!t.hasPoint) continue;
            const float p[3] = {float(t.point.x), float(t.point.y), float(t.point.z)};
            // An unset colour is exactly zero (see Track::color); write the
            // neutral grey a viewer would draw rather than black.
            const bool has = t.color.x != 0.0 || t.color.y != 0.0 || t.color.z != 0.0;
            const Vec3 c = has ? t.color : Vec3{0.72, 0.72, 0.74};
            const uint8_t rgb[3] = {
                uint8_t(std::clamp(c.x, 0.0, 1.0) * 255.0 + 0.5),
                uint8_t(std::clamp(c.y, 0.0, 1.0) * 255.0 + 0.5),
                uint8_t(std::clamp(c.z, 0.0, 1.0) * 255.0 + 0.5)};
            f.write(reinterpret_cast<const char*>(p), sizeof(p));
            f.write(reinterpret_cast<const char*>(rgb), sizeof(rgb));
        }
    }

    if (!f) {
        *err = "write to '" + path + "' failed";
        return false;
    }
    return true;
}

// --- reading ---------------------------------------------------------------------

bool LoadPly(const std::string& path, PointCloud* cloud, std::string* note,
             std::string* err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        *err = "could not open '" + path + "'";
        return false;
    }

    // --- header ---
    std::string line;
    if (!std::getline(f, line) || line.rfind("ply", 0) != 0) {
        *err = "'" + path + "' is not a PLY file";
        return false;
    }
    enum class Fmt { Ascii, LE, BE } fmt = Fmt::Ascii;
    bool gotFormat = false;
    // Only the FIRST element is read, and it must be the vertices: every
    // splat file and nearly every point cloud puts them first. Faces or
    // other elements after it are simply not read.
    std::string firstElement;
    size_t count = 0;
    std::vector<Prop> props;
    bool inFirst = false, sawEnd = false;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream ls(line);
        std::string kw;
        ls >> kw;
        if (kw == "format") {
            std::string v;
            ls >> v;
            if (v == "ascii") fmt = Fmt::Ascii;
            else if (v == "binary_little_endian") fmt = Fmt::LE;
            else if (v == "binary_big_endian") fmt = Fmt::BE;
            else { *err = "unknown PLY format '" + v + "'"; return false; }
            gotFormat = true;
        } else if (kw == "element") {
            std::string name;
            size_t c = 0;
            ls >> name >> c;
            if (firstElement.empty()) {
                firstElement = name;
                count = c;
                inFirst = true;
            } else {
                inFirst = false;
            }
        } else if (kw == "property" && inFirst) {
            std::string t, name;
            ls >> t;
            if (t == "list") {
                *err = "the first element of '" + path + "' has a list "
                       "property; expected plain vertices";
                return false;
            }
            ls >> name;
            Prop p;
            p.name = name;
            p.type = ParseType(t);
            if (p.type == PType::Bad) {
                *err = "unknown PLY property type '" + t + "'";
                return false;
            }
            props.push_back(p);
        } else if (kw == "end_header") {
            sawEnd = true;
            break;
        }
    }
    if (!sawEnd || !gotFormat) {
        *err = "'" + path + "' has an incomplete PLY header";
        return false;
    }
    if (firstElement != "vertex") {
        *err = "'" + path + "' does not start with vertices (first element is '" +
               firstElement + "')";
        return false;
    }

    int stride = 0;
    for (Prop& p : props) {
        p.offset = stride;
        stride += SizeOf(p.type);
    }
    auto index = [&](const char* name) {
        for (size_t i = 0; i < props.size(); ++i)
            if (props[i].name == name) return int(i);
        return -1;
    };
    const int ix = index("x"), iy = index("y"), iz = index("z");
    if (ix < 0 || iy < 0 || iz < 0) {
        *err = "'" + path + "' has no x/y/z vertex properties";
        return false;
    }
    const int iOp = index("opacity"), iS0 = index("scale_0"), iR0 = index("rot_0");
    const bool isSplat = iOp >= 0 && iS0 >= 0 && iR0 >= 0;
    int iS[3] = {iS0, index("scale_1"), index("scale_2")};
    int iR[4] = {iR0, index("rot_1"), index("rot_2"), index("rot_3")};
    int iDc[3] = {index("f_dc_0"), index("f_dc_1"), index("f_dc_2")};
    int iC[3] = {index("red"), index("green"), index("blue")};
    if (isSplat) {
        for (int k : iS) if (k < 0) { *err = "'" + path + "' lacks scale_1/2"; return false; }
        for (int k : iR) if (k < 0) { *err = "'" + path + "' lacks rot_1..3"; return false; }
    }
    int rest = 0;
    for (const Prop& p : props) rest += p.name.rfind("f_rest_", 0) == 0 ? 1 : 0;

    // VIEW-DEPENDENT COLOUR, when the file has it: the reference layout is
    // f_rest_{channel * perChannel + k}, 9 / 24 / 45 values for degree 1 / 2
    // / 3. Kept in tglab's order, rest[k * 3 + channel] (algo_util/
    // splat_sh.h), with any degree the file stops short of left zero.
    int shDeg = 0, perChannel = 0;
    std::vector<int> iRest;
    if (isSplat && rest >= 9) {
        perChannel = rest / 3;
        for (int d = 3; d >= 1; --d)
            if (perChannel >= (d + 1) * (d + 1) - 1) { shDeg = d; break; }
        perChannel = (shDeg + 1) * (shDeg + 1) - 1;
        for (int q = 0; q < 3 * perChannel; ++q) {
            const int at = index(("f_rest_" + std::to_string(q)).c_str());
            if (at < 0) { shDeg = 0; iRest.clear(); break; }
            iRest.push_back(at);
        }
    }

    // --- body ---
    std::vector<double> v(props.size());
    const bool swap = (fmt == Fmt::BE);
    // Binary records are read 64k at a time: a downloaded splat is millions
    // of Gaussians at ~250 bytes each, and one stream read per record is
    // most of the load time at that size.
    const size_t kChunk = 65536;
    std::vector<uint8_t> chunk;
    size_t have = 0, used = 0, left = count;
    auto next = [&]() -> bool {
        if (fmt == Fmt::Ascii) {
            for (double& x : v)
                if (!(f >> x)) return false;
            return true;
        }
        if (used == have) {
            const size_t n = std::min(kChunk, left);
            chunk.resize(n * size_t(stride));
            if (n == 0 || !f.read(reinterpret_cast<char*>(chunk.data()),
                                  std::streamsize(chunk.size())))
                return false;
            have = n;
            used = 0;
            left -= n;
        }
        const uint8_t* rec = chunk.data() + used * size_t(stride);
        ++used;
        for (size_t i = 0; i < props.size(); ++i)
            v[i] = Decode(rec + props[i].offset, props[i].type, swap);
        return true;
    };

    PointCloud out;
    if (isSplat) out.splats.reserve(count);
    else out.tracks.reserve(count);
    for (size_t n = 0; n < count; ++n) {
        if (!next()) {
            *err = "'" + path + "' ends after " + std::to_string(n) + " of " +
                   std::to_string(count) + " vertices";
            return false;
        }
        const Vec3 pos{v[size_t(ix)], v[size_t(iy)], v[size_t(iz)]};
        if (isSplat) {
            Splat s;
            s.mean = pos;
            s.scale = Vec3{std::exp(v[size_t(iS[0])]), std::exp(v[size_t(iS[1])]),
                           std::exp(v[size_t(iS[2])])};
            double q[4], len = 0.0;
            for (int k = 0; k < 4; ++k) { q[k] = v[size_t(iR[k])]; len += q[k] * q[k]; }
            len = std::sqrt(len);
            if (len < 1e-12) { q[0] = 1.0; q[1] = q[2] = q[3] = 0.0; len = 1.0; }
            for (int k = 0; k < 4; ++k) s.rot[k] = q[k] / len;
            s.opacity = Sigmoid(v[size_t(iOp)]);
            double c[3] = {0.5, 0.5, 0.5};
            for (int k = 0; k < 3; ++k)
                if (iDc[k] >= 0) c[k] = 0.5 + kC0 * v[size_t(iDc[k])];
                else if (iC[k] >= 0) c[k] = v[size_t(iC[k])] / 255.0;
            // Clamped as training keeps its own: see train_splats' clamp_colour.
            s.color = Vec3{std::clamp(c[0], 0.0, 1.0), std::clamp(c[1], 0.0, 1.0),
                           std::clamp(c[2], 0.0, 1.0)};
            out.splats.push_back(s);
            if (shDeg > 0) {
                const size_t base = out.splatSh.size();
                out.splatSh.resize(base + 45, 0.0f);
                for (int ch = 0; ch < 3; ++ch)
                    for (int k = 0; k < perChannel; ++k)
                        out.splatSh[base + size_t(k * 3 + ch)] =
                            float(v[size_t(iRest[size_t(ch * perChannel + k)])]);
            }
        } else {
            Track t;
            t.point = pos;
            t.hasPoint = true;
            if (iC[0] >= 0 && iC[1] >= 0 && iC[2] >= 0) {
                // 8-bit channels are 0..255; float ones are usually 0..1.
                const bool bytes = props[size_t(iC[0])].type == PType::U8 ||
                                   v[size_t(iC[0])] > 1.0 || v[size_t(iC[1])] > 1.0 ||
                                   v[size_t(iC[2])] > 1.0;
                const double k = bytes ? 1.0 / 255.0 : 1.0;
                t.color = Vec3{v[size_t(iC[0])] * k, v[size_t(iC[1])] * k,
                               v[size_t(iC[2])] * k};
            }
            out.tracks.push_back(std::move(t));
        }
    }

    out.shDegree = shDeg;
    *cloud = std::move(out);
    char buf[240];
    if (isSplat && shDeg > 0)
        std::snprintf(buf, sizeof(buf), "%zu Gaussians, view-dependent colour to degree %d",
                      count, shDeg);
    else if (isSplat)
        std::snprintf(buf, sizeof(buf), "%zu Gaussians%s", count,
                      rest > 0 ? ", view-dependent colour dropped (an unrecognised layout)"
                               : "");
    else
        std::snprintf(buf, sizeof(buf), "%zu points%s", count,
                      iC[0] >= 0 ? ", coloured" : "");
    *note = buf;
    return true;
}

}  // namespace tglab
