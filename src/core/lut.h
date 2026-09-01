// A 3D colour lookup table, loaded from a .cube file.
//
// WHAT A LUT IS, because the name undersells how blunt it is: a cube of RGB
// triples on a regular grid. A 33^3 table is 35,937 entries, each saying "input
// colour at this grid point comes out as THIS colour". Everything in between is
// interpolated. There is no "contrast" number inside a LUT and no "saturation"
// number -- those are patterns distributed across all 35,937 entries at once,
// which is why a LUT cannot be decomposed back into sliders and why applying
// one is the only sensible way to use it.
//
// .cube is Adobe's format and what almost every downloadable film LUT ships as.
// It is plain text: some header keywords, then size^3 lines of "r g b" in
// blue-slowest order. 1D LUTs also exist in the same format and are handled,
// since a few tone-curve LUTs use them.
#pragma once

#include <array>
#include <string>
#include <vector>

namespace tglab {

class Lut3D {
public:
    // Loads a .cube file. Returns false and fills `err` on any problem --
    // missing file, bad size, wrong entry count, unparseable line.
    bool Load(const std::string& path, std::string* err);

    bool Valid() const { return m_size >= 2 && !m_data.empty(); }
    int  Size()  const { return m_size; }
    bool Is1D()  const { return m_is1D; }

    // Where this came from, for the UI and for error messages.
    const std::string& Path() const { return m_path; }
    const std::string& Title() const { return m_title; }

    // The domain the table covers. Almost always 0..1, but the format allows
    // otherwise and a LUT authored for log footage may not be 0..1 -- applying
    // such a table as though it were would silently mangle the image.
    const std::array<float, 3>& DomainMin() const { return m_domainMin; }
    const std::array<float, 3>& DomainMax() const { return m_domainMax; }

    // TETRAHEDRAL interpolation, not trilinear.
    //
    // Trilinear blends all eight corners of the cell, which on a LUT with a
    // sharp transition -- a hue shift that starts abruptly, say -- pulls in
    // corners on the far side of that transition and rounds it off. Tetrahedral
    // splits the cell into six tetrahedra and uses only the four corners of the
    // one the sample falls in, so a sharp edge in the table stays sharp. It is
    // also what Resolve and Photoshop use, so a LUT looks the same here as
    // where it was authored, which matters more than the small accuracy
    // difference.
    void Sample(float r, float g, float b, float* out) const;

private:
    const float* At(int ri, int gi, int bi) const {
        // Blue slowest, matching .cube's own ordering, so the file loads with
        // no transposition.
        return &m_data[(size_t(bi) * size_t(m_size) * size_t(m_size) +
                        size_t(gi) * size_t(m_size) + size_t(ri)) * 3];
    }

    int                  m_size = 0;
    bool                 m_is1D = false;
    std::vector<float>   m_data;      // size^3 * 3, or size * 3 when 1D
    std::string          m_path;
    std::string          m_title;
    std::array<float, 3> m_domainMin{0.0f, 0.0f, 0.0f};
    std::array<float, 3> m_domainMax{1.0f, 1.0f, 1.0f};
};

}  // namespace tglab
