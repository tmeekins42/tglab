#include "image_group.h"

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

void Ungroup(Data* d) {
    Image first;
    if (auto* s = std::get_if<ImageSet>(d); s && !s->images.empty())
        first = std::move(s->images.front());
    *d = Data{std::move(first)};
}

} // namespace tglab
