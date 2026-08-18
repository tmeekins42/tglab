#pragma once

#include <string>

#include "ast.h"

namespace tglab {

// Parses source into `out`. On failure returns false and fills `err` with a
// message carrying a line number.
bool Parse(std::string_view src, Program* out, std::string* err);

} // namespace tglab
