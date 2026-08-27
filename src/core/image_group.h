// Group operations on a palette entry's Data.
//
// Free functions on Data rather than methods on the app's PaletteEntry, so the
// state machine can be tested. It is small but easy to get subtly wrong -- the
// shape's extent has to stay in step with the image count through every
// transition, and a stale extent is the kind of error that surfaces much later
// as a reduction over the wrong number of frames.
#pragma once

#include <string>

#include "data.h"

namespace tglab {

// Converts to a set, keeping whatever single image was there. Idempotent.
void MakeGroup(Data* d, const std::string& axis);

// Adds one image, converting first if needed. The extent follows the count.
void AppendToGroup(Data* d, Image&& img, const std::string& axis);

// Drops the last image, if any. The extent follows the count.
void RemoveLastFromGroup(Data* d, const std::string& axis);

// Renames the axis, restating the shape so a script's over="..." matches.
void SetGroupAxis(Data* d, const std::string& axis);

// Back to a single image, keeping the FIRST. Deliberately keeps one rather than
// emptying the slot: a script referring to the entry by name should still
// resolve, and losing every dropped file to a mis-click would be worse than
// losing the tail.
void Ungroup(Data* d);

} // namespace tglab
