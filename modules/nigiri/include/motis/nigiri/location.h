#pragma once

#include <string>
#include <vector>

#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace motis::nigiri {

std::pair<std::string_view, std::string_view> split_tag_and_location_id(
    std::string_view station_id);

::nigiri::source_idx_t get_src(
    ::nigiri::hash_map<std::string, ::nigiri::source_idx_t> const& tags,
    std::string_view tag);

::nigiri::location_id motis_station_to_nigiri_id(
    std::vector<std::string> const& tags, std::string const& station_id);

::nigiri::location_idx_t get_location_idx(std::vector<std::string> const& tags,
                                          ::nigiri::timetable const& tt,
                                          std::string const& station_id);

}  // namespace motis::nigiri