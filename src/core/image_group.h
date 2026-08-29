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

// Compares two filenames with digit runs read as NUMBERS, so IMG_9 precedes
// IMG_10 rather than following it.
//
// This matters more than a tidiness argument suggests: Windows hands a
// multi-file drop over in selection order, and the file clicked last routinely
// arrives out of place -- so a bracket dropped in visual order can still reach
// the palette shuffled. A reduction then consumes it in the wrong order and
// produces a silently wrong result rather than an error.
//
// Deliberately not a full natural sort: it handles the numbered-burst case that
// actually occurs, and nothing else.
bool FilenameLess(const std::string& a, const std::string& b);

// A sort key for "when was this taken", from an EXIF date and a file time.
//
// `exifDate` is EXIF's "2026:08:19 14:32:07" -- fixed width and zero padded, so
// comparing the strings IS comparing the instants. No parsing, and no timezone
// to get wrong.
//
// A frame with no EXIF date falls back to its file timestamp, which is a
// different clock and not comparable with the first. Rather than interleave two
// incompatible orderings, a fallback key is prefixed with a space so every one
// of them sorts BEFORE every EXIF key: the unknowns group together and keep
// whatever order they arrived in. Arbitrary, but predictable, which a mixed
// ordering would not be.
//
// Here rather than in the UI so it can be tested: the sort itself is a
// permutation over ImGui state, and this is the part that can be wrong.
std::string DateSortKey(const std::string& exifDate, long long fileTime);

// Back to a single image, keeping the FIRST. Deliberately keeps one rather than
// emptying the slot: a script referring to the entry by name should still
// resolve, and losing every dropped file to a mis-click would be worse than
// losing the tail.
void Ungroup(Data* d);

} // namespace tglab
