#pragma once

#include <optional>

#include "osr/routing/route.h"

namespace icc {

std::optional<osr::location> parse_location(std::string_view);

}  // namespace icc