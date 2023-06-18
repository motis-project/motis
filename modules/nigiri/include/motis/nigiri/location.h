#pragma once

#include <string>
#include <vector>

#include "nigiri/types.h"

#include "motis/nigiri/tag_map.h"

namespace nigiri {
struct timetable;
}

namespace motis::nigiri {

std::pair<std::string_view, std::string_view> split_tag_and_location_id(
    std::string_view station_id);

::nigiri::source_idx_t get_src(tag_map_t const&, std::string_view tag);

::nigiri::location_id motis_station_to_nigiri_id(tag_map_t const&,
                                                 std::string_view station_id);

::nigiri::location_idx_t get_location_idx(tag_map_t const&,
                                          ::nigiri::timetable const&,
                                          std::string_view station_id);

}  // namespace motis::nigiri