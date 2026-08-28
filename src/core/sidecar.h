// Sidecar: non-pixel data an algorithm attaches to an image.
//
// The general case of "where do features live". An algorithm computes something
// ABOUT an image -- a feature set, an alignment transform, a camera pose, a set
// of object boxes -- and a later algorithm wants it. Passing that through ports
// would mean every merge declaring an input it usually does not need; passing it
// out of band would mean nothing keeps it with the image it describes.
//
// So the image carries a table, keyed by name:
//
//     transform   from an aligner, used by any merge that wants to warp
//     features    from a detector, used by matching and alignment
//     camera      pose and intrinsics, produced by SfM
//     boxes       from an object detector, used by masking
//
// The consumer READS A SIDECAR IT DID NOT ASK FOR AND MAY NOT FIND. merge_hdr
// samples through a transform if one is attached and merges unaligned if not.
// It does not require alignment, declare a dependency on it, or fail without
// it. That is what makes the mechanism worth having rather than a parameter: an
// alignment stage inserted anywhere upstream improves every downstream merge,
// with no merge algorithm changing.
//
// NOT a void pointer, which was the first suggestion. A sidecar has to survive
// being copied between stages and dropped when an image is evicted, and a raw
// pointer supports neither -- nor any type checking at the point of use.
#pragma once

#include <memory>
#include <string>
#include <unordered_map>

namespace tglab {

// Base for anything attachable. A concrete sidecar derives from this and is
// retrieved with a checked cast, so asking for the wrong type returns null
// rather than reinterpreting bytes.
class SidecarBase {
public:
    virtual ~SidecarBase() = default;

    // Whether this describes the PIXELS or the CAPTURE.
    //
    // Derived (the default) means it is invalidated when the pixels change: a
    // feature set computed from an image is wrong the moment that image is
    // re-developed, and silently keeping it would produce an alignment solved
    // against an image that no longer exists -- a failure that shows up as a
    // subtly wrong merge rather than an error.
    //
    // Intrinsic means it describes the capture and survives: a camera pose from
    // SfM does not move because the exposure changed.
    //
    // Defaulting to derived is the safe direction. Wrongly dropping a sidecar
    // costs a recompute; wrongly keeping one produces a wrong answer that looks
    // plausible.
    virtual bool DerivedFromPixels() const { return true; }
};

// Shared and const, because sidecars are naturally shared: a reduction over
// twelve frames wants all twelve transforms alive at once, cheaply, and nothing
// should be mutating one while another stage reads it.
using SidecarPtr = std::shared_ptr<const SidecarBase>;

// The table an Image carries. Copying it shares the entries rather than
// duplicating them, which is what makes Image::Clone() cheap for an image with
// a feature set attached.
class SidecarTable {
public:
    void Set(std::string name, SidecarPtr p) {
        if (p) m_map[std::move(name)] = std::move(p);
        else   m_map.erase(name);
    }

    // Checked retrieval. Returns null when absent OR when the stored entry is
    // not the requested type, so a consumer that guesses wrong gets nothing
    // rather than a bad cast.
    template <class T>
    const T* Get(const std::string& name) const {
        const auto it = m_map.find(name);
        if (it == m_map.end()) return nullptr;
        return dynamic_cast<const T*>(it->second.get());
    }

    bool   Has(const std::string& name) const { return m_map.count(name) != 0; }
    void   Remove(const std::string& name)    { m_map.erase(name); }
    bool   Empty() const                      { return m_map.empty(); }
    size_t Size()  const                      { return m_map.size(); }
    void   Clear()                            { m_map.clear(); }

    // Drops everything that describes the pixels, keeping what describes the
    // capture. Called when an image's pixels change.
    void DropDerived() {
        for (auto it = m_map.begin(); it != m_map.end(); ) {
            if (it->second && it->second->DerivedFromPixels()) it = m_map.erase(it);
            else ++it;
        }
    }

private:
    std::unordered_map<std::string, SidecarPtr> m_map;
};

} // namespace tglab
