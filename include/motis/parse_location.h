#pragma once

#include <optional>
#include <string>
#include <string_view>

#include "osr/routing/route.h"

#include "nigiri/routing/query.h"
#include "nigiri/types.h"

namespace motis {

std::optional<osr::location> parse_location(std::string_view,
                                            char separator = ',');

date::sys_days parse_iso_date(std::string_view);

nigiri::routing::query cursor_to_query(std::string_view);

std::pair<nigiri::direction, nigiri::unixtime_t> parse_cursor(std::string_view);

}  // namespace motis