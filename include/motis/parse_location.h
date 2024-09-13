#pragma once

#include <optional>
#include <string>
#include <string_view>

#include "osr/routing/route.h"

#include "nigiri/routing/query.h"
#include "nigiri/types.h"

namespace motis {

std::optional<osr::location> parse_location(std::string_view);

nigiri::unixtime_t get_date_time(std::optional<std::string> const& date,
                                 std::optional<std::string> const& time);

nigiri::routing::query parse_cursor(std::string_view s);

}  // namespace motis