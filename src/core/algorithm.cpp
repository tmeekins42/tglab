#include "algorithm.h"

#include <algorithm>
#include <cassert>

namespace tglab {

const char* DataTypeName(DataType t) {
    switch (t) {
        case DataType::ImageSet: return "ImageSet";
        case DataType::Image: return "Image";
        case DataType::None:  return "None";
    }
    return "?";
}

// --- RunCtx -----------------------------------------------------------------

ImageView RunCtx::In(size_t i) const {
    assert(i < m_in.size());
    const Data* d = m_in[i];
    if (!d || !std::holds_alternative<Image>(*d)) return {};
    // The input is const to the algorithm, but MapCpuRead is non-const on
    // Image (it may await a fence in M3), so cast away only the constness
    // we imposed on the container, not on the pixels.
    return const_cast<Image&>(std::get<Image>(*d)).MapCpuRead();
}

ImageView RunCtx::Out(size_t i) const {
    assert(i < m_out.size());
    Data& d = m_out[i];
    if (!std::holds_alternative<Image>(d)) return {};
    return std::get<Image>(d).MapCpuWrite();
}

const ImageDesc& RunCtx::InDesc(size_t i) const {
    static const ImageDesc kEmpty{};
    assert(i < m_in.size());
    const Data* d = m_in[i];
    if (!d || !std::holds_alternative<Image>(*d)) return kEmpty;
    return std::get<Image>(*d).Desc();
}

// --- AlgorithmBase ----------------------------------------------------------

ParamBase* AlgorithmBase::FindParam(std::string_view name) const {
    for (ParamBase* p : m_params)
        if (name == p->Name()) return p;
    return nullptr;
}

uint64_t AlgorithmBase::ParamHash() const {
    // FNV-1a over each parameter's value hash.
    uint64_t h = 1469598103934665603ull;
    for (const ParamBase* p : m_params) {
        uint64_t v = p->HashValue();
        for (int i = 0; i < 8; ++i) {
            h ^= (v >> (i * 8)) & 0xff;
            h *= 1099511628211ull;
        }
    }
    return h;
}

// --- Registry ---------------------------------------------------------------

Registry& Registry::Get() {
    static Registry s_instance;
    return s_instance;
}

void Registry::Add(const char* name, Factory f) {
    // Later registration of the same name would be a silent shadowing bug.
    assert(!Contains(name) && "duplicate algorithm name");
    m_entries.push_back({name, f});
}

std::unique_ptr<AlgorithmBase> Registry::Create(std::string_view name) const {
    for (const Entry& e : m_entries)
        if (e.name == name) return e.factory();
    return nullptr;
}

bool Registry::Contains(std::string_view name) const {
    for (const Entry& e : m_entries)
        if (e.name == name) return true;
    return false;
}

std::vector<std::string> Registry::Names() const {
    std::vector<std::string> out;
    out.reserve(m_entries.size());
    for (const Entry& e : m_entries) out.push_back(e.name);
    std::sort(out.begin(), out.end());
    return out;
}

std::vector<std::string> Registry::NamesInCategory(std::string_view category) const {
    std::vector<std::string> out;
    for (const Entry& e : m_entries) {
        auto a = e.factory();
        if (a && category == a->Category()) out.push_back(e.name);
    }
    std::sort(out.begin(), out.end());
    return out;
}

} // namespace tglab
