#include "lut.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace tglab {
namespace {

// .cube allows blank lines and # comments anywhere, including between data
// rows, so every read has to skip them rather than assuming the data is
// contiguous after the header.
bool IsSkippable(const std::string& line) {
    for (char c : line) {
        if (c == '#') return true;
        if (!std::isspace(static_cast<unsigned char>(c))) return false;
    }
    return true;
}

}  // namespace

bool Lut3D::Load(const std::string& path, std::string* err) {
    auto fail = [&](const std::string& what) {
        if (err) *err = path + ": " + what;
        m_size = 0;
        m_data.clear();
        return false;
    };

    std::ifstream f(path);
    if (!f) return fail("cannot open");

    m_path = path;
    m_title.clear();
    m_size = 0;
    m_is1D = false;
    m_data.clear();
    m_domainMin = {0.0f, 0.0f, 0.0f};
    m_domainMax = {1.0f, 1.0f, 1.0f};

    std::vector<float> data;
    std::string line;

    while (std::getline(f, line)) {
        if (IsSkippable(line)) continue;

        std::istringstream ls(line);
        std::string tok;
        ls >> tok;

        if (tok == "TITLE") {
            // Quoted, and may contain spaces.
            std::string rest;
            std::getline(ls, rest);
            const size_t a = rest.find('"'), b = rest.rfind('"');
            m_title = (a != std::string::npos && b != std::string::npos && b > a)
                          ? rest.substr(a + 1, b - a - 1)
                          : rest;
            continue;
        }
        if (tok == "LUT_3D_SIZE" || tok == "LUT_1D_SIZE") {
            int n = 0;
            if (!(ls >> n)) return fail("malformed " + tok);
            // 2 is the smallest table that can interpolate at all; 256 is far
            // beyond anything real (a 256^3 table is 50 million entries) and
            // exists only to stop a corrupt header allocating unboundedly.
            if (n < 2 || n > 256)
                return fail(tok + " of " + std::to_string(n) + " is out of range");
            m_size = n;
            m_is1D = (tok == "LUT_1D_SIZE");
            const size_t need = m_is1D ? size_t(n) * 3
                                       : size_t(n) * size_t(n) * size_t(n) * 3;
            data.reserve(need);
            continue;
        }
        if (tok == "DOMAIN_MIN" || tok == "DOMAIN_MAX") {
            float a = 0, b = 0, c = 0;
            if (!(ls >> a >> b >> c)) return fail("malformed " + tok);
            if (tok == "DOMAIN_MIN") m_domainMin = {a, b, c};
            else                     m_domainMax = {a, b, c};
            continue;
        }
        if (tok == "LUT_3D_INPUT_RANGE" || tok == "LUT_1D_INPUT_RANGE") {
            // An older spelling of the same thing, one range for all channels.
            float lo = 0, hi = 1;
            if (!(ls >> lo >> hi)) return fail("malformed " + tok);
            m_domainMin = {lo, lo, lo};
            m_domainMax = {hi, hi, hi};
            continue;
        }

        // Anything else must be a data row: three floats, the first already in
        // `tok`.
        float r = 0, g = 0, b = 0;
        try {
            r = std::stof(tok);
        } catch (...) {
            return fail("unexpected line: " + line.substr(0, 40));
        }
        if (!(ls >> g >> b)) return fail("data row with fewer than 3 values");
        data.push_back(r);
        data.push_back(g);
        data.push_back(b);
    }

    if (m_size == 0) return fail("no LUT_3D_SIZE or LUT_1D_SIZE");

    const size_t want = m_is1D
                            ? size_t(m_size) * 3
                            : size_t(m_size) * size_t(m_size) * size_t(m_size) * 3;
    if (data.size() != want)
        return fail("expected " + std::to_string(want / 3) + " entries, found " +
                    std::to_string(data.size() / 3));

    for (int i = 0; i < 3; ++i)
        if (!(m_domainMax[size_t(i)] > m_domainMin[size_t(i)]))
            return fail("DOMAIN_MAX is not above DOMAIN_MIN");

    m_data = std::move(data);
    return true;
}

void Lut3D::Sample(float r, float g, float b, float* out) const {
    if (!Valid()) {
        out[0] = r; out[1] = g; out[2] = b;
        return;
    }

    const float in[3] = {r, g, b};
    float t[3];
    for (int i = 0; i < 3; ++i) {
        const float lo = m_domainMin[size_t(i)], hi = m_domainMax[size_t(i)];
        // Out-of-domain input is CLAMPED rather than extrapolated. A LUT says
        // nothing about colours outside its table, and extrapolating from the
        // edge cells produces wild values on exactly the scene-linear
        // highlights this is most likely to meet.
        t[i] = std::clamp((in[i] - lo) / (hi - lo), 0.0f, 1.0f) *
               float(m_size - 1);
    }

    if (m_is1D) {
        // Three independent curves; no cube to walk.
        for (int i = 0; i < 3; ++i) {
            const int i0 = std::min(int(t[i]), m_size - 1);
            const int i1 = std::min(i0 + 1, m_size - 1);
            const float f = t[i] - float(i0);
            out[i] = m_data[size_t(i0) * 3 + size_t(i)] * (1.0f - f) +
                     m_data[size_t(i1) * 3 + size_t(i)] * f;
        }
        return;
    }

    const int i0[3] = {std::min(int(t[0]), m_size - 2),
                       std::min(int(t[1]), m_size - 2),
                       std::min(int(t[2]), m_size - 2)};
    const float d[3] = {t[0] - float(i0[0]), t[1] - float(i0[1]),
                        t[2] - float(i0[2])};

    // Tetrahedral interpolation. The unit cell splits into six tetrahedra by
    // the ordering of the three fractional coordinates, and only the four
    // corners of the containing one contribute -- so a sharp transition in the
    // table stays sharp instead of being rounded off by corners on the far
    // side of it. Which six, and their weights, follow from sorting dr/dg/db.
    const float dr = d[0], dg = d[1], db = d[2];

    const float* c000 = At(i0[0],     i0[1],     i0[2]);
    const float* c111 = At(i0[0] + 1, i0[1] + 1, i0[2] + 1);

    for (int k = 0; k < 3; ++k) {
        float v;
        if (dr > dg) {
            if (dg > db) {          // dr > dg > db
                const float* c100 = At(i0[0] + 1, i0[1],     i0[2]);
                const float* c110 = At(i0[0] + 1, i0[1] + 1, i0[2]);
                v = c000[k] + (c100[k] - c000[k]) * dr +
                              (c110[k] - c100[k]) * dg +
                              (c111[k] - c110[k]) * db;
            } else if (dr > db) {   // dr > db > dg
                const float* c100 = At(i0[0] + 1, i0[1],     i0[2]);
                const float* c101 = At(i0[0] + 1, i0[1],     i0[2] + 1);
                v = c000[k] + (c100[k] - c000[k]) * dr +
                              (c101[k] - c100[k]) * db +
                              (c111[k] - c101[k]) * dg;
            } else {                // db > dr > dg
                const float* c001 = At(i0[0],     i0[1],     i0[2] + 1);
                const float* c101 = At(i0[0] + 1, i0[1],     i0[2] + 1);
                v = c000[k] + (c001[k] - c000[k]) * db +
                              (c101[k] - c001[k]) * dr +
                              (c111[k] - c101[k]) * dg;
            }
        } else {
            if (db > dg) {          // db > dg > dr
                const float* c001 = At(i0[0],     i0[1],     i0[2] + 1);
                const float* c011 = At(i0[0],     i0[1] + 1, i0[2] + 1);
                v = c000[k] + (c001[k] - c000[k]) * db +
                              (c011[k] - c001[k]) * dg +
                              (c111[k] - c011[k]) * dr;
            } else if (db > dr) {   // dg > db > dr
                const float* c010 = At(i0[0],     i0[1] + 1, i0[2]);
                const float* c011 = At(i0[0],     i0[1] + 1, i0[2] + 1);
                v = c000[k] + (c010[k] - c000[k]) * dg +
                              (c011[k] - c010[k]) * db +
                              (c111[k] - c011[k]) * dr;
            } else {                // dg > dr > db
                const float* c010 = At(i0[0],     i0[1] + 1, i0[2]);
                const float* c110 = At(i0[0] + 1, i0[1] + 1, i0[2]);
                v = c000[k] + (c010[k] - c000[k]) * dg +
                              (c110[k] - c010[k]) * dr +
                              (c111[k] - c110[k]) * db;
            }
        }
        out[k] = v;
    }
}

}  // namespace tglab
