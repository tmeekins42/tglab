#pragma once

#include <functional>
#include <string>

#include "ast.h"

namespace tglab {

// Resolves an `include "name"` to source text.
//
// A CALLBACK rather than the parser opening files itself, so the script layer
// keeps knowing nothing about the filesystem -- it has no idea where scripts
// live, and the app's rule (beside the including file) is the app's policy.
// It is also what lets the tests include from an in-memory map with no
// temporary files, and what lets the app collect the include list for its
// hot-reload watcher.
//
// `from` is the file doing the including, empty for the top-level script, so a
// resolver can make relative paths work. Returns false when the file cannot be
// read; the parser turns that into a normal script error naming the line.
using IncludeResolver = std::function<bool(const std::string& name,
                                           const std::string& from,
                                           std::string* source,
                                           std::string* resolvedPath)>;

// Parses source into `out`. On failure returns false and fills `err` with a
// message carrying a line number.
//
// Without a resolver `include` is a parse error, which is what a caller that
// has no filesystem wants: the alternative is silently ignoring an include and
// producing a program missing half its stages.
bool Parse(std::string_view src, Program* out, std::string* err,
           const IncludeResolver& resolve = nullptr,
           const std::string& selfPath = "");

} // namespace tglab
