#include "image_group.h"

#include <cstdio>

#include <cctype>

namespace tglab {

namespace {
// The one invariant: the extent equals the image count. Every mutation goes
// through this rather than setting the shape by hand.
void Restate(ImageSet* s, const std::string& axis) {
    s->shape = Shape::Of(axis, int(s->images.size()));
}
} // namespace

void MakeGroup(Data* d, const std::string& axis) {
    if (std::holds_alternative<ImageSet>(*d)) { SetGroupAxis(d, axis); return; }
    ImageSet s;
    if (auto* had = std::get_if<Image>(d); had && had->Valid())
        s.images.push_back(std::move(*had));
    Restate(&s, axis);
    *d = Data{std::move(s)};
}

void AppendToGroup(Data* d, Image&& img, const std::string& axis) {
    if (!std::holds_alternative<ImageSet>(*d)) MakeGroup(d, axis);
    ImageSet& s = std::get<ImageSet>(*d);
    s.images.push_back(std::move(img));
    Restate(&s, axis);
}

void RemoveLastFromGroup(Data* d, const std::string& axis) {
    auto* s = std::get_if<ImageSet>(d);
    if (!s || s->images.empty()) return;
    s->images.pop_back();
    Restate(s, axis);
}

void SetGroupAxis(Data* d, const std::string& axis) {
    if (auto* s = std::get_if<ImageSet>(d)) Restate(s, axis);
}

bool FilenameLess(const std::string& a, const std::string& b) {
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        const bool da = std::isdigit(static_cast<unsigned char>(a[i]));
        const bool db = std::isdigit(static_cast<unsigned char>(b[j]));
        if (da && db) {
            // Compare the whole digit RUN as a number, so 9 < 10. Leading zeros
            // fall out of this for free: 007 and 7 compare equal here, and the
            // characters after them decide.
            size_t ia = i, jb = j;
            while (ia < a.size() && std::isdigit(static_cast<unsigned char>(a[ia]))) ++ia;
            while (jb < b.size() && std::isdigit(static_cast<unsigned char>(b[jb]))) ++jb;
            unsigned long long va = 0, vb = 0;
            for (size_t k = i; k < ia; ++k) va = va * 10 + unsigned(a[k] - '0');
            for (size_t k = j; k < jb; ++k) vb = vb * 10 + unsigned(b[k] - '0');
            if (va != vb) return va < vb;
            i = ia; j = jb;
            continue;
        }
        const char ca = char(std::tolower(static_cast<unsigned char>(a[i])));
        const char cb = char(std::tolower(static_cast<unsigned char>(b[j])));
        if (ca != cb) return ca < cb;
        ++i; ++j;
    }
    // One is a prefix of the other: the shorter remainder sorts first.
    return (a.size() - i) < (b.size() - j);
}

void Ungroup(Data* d) {
    Image first;
    if (auto* s = std::get_if<ImageSet>(d); s && !s->images.empty())
        first = std::move(s->images.front());
    *d = Data{std::move(first)};
}


// See the header for why the fallback sorts first rather than interleaving.
std::string DateSortKey(const std::string& exifDate, long long fileTime) {
    if (!exifDate.empty()) return exifDate;

    // Zero padded to a fixed width, because these keys are COMPARED AS STRINGS
    // and an unpadded number is not ordered by one: "99" sorts after "100".
    // Windows file times happen to be uniformly 18 digits, so this would not
    // bite today -- which is exactly why it would be a nasty surprise later, on
    // a platform whose epoch counts vary in length.
    //
    // 20 digits covers the full range of a signed 64-bit count. Negative times
    // (before the epoch, which Windows does not produce but POSIX can) sort as
    // text and would land after positives; not worth handling until something
    // can generate one.
    char buf[32];
    std::snprintf(buf, sizeof buf, " %020lld", fileTime);
    return buf;
}

} // namespace tglab
