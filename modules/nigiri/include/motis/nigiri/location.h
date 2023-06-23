#pragma once

#include <string>

#include "nigiri/types.h"

#include "motis/nigiri/tag_lookup.h"
#include "motis/string.h"

namespace nigiri {
struct timetable;
}

namespace motis::nigiri {

mcd::string get_station_id(tag_lookup const&, ::nigiri::timetable const&,
                           ::nigiri::location_idx_t);

std::pair<std::string_view, std::string_view> split_tag_and_location_id(
    std::string_view station_id);

::nigiri::location_id motis_station_to_nigiri_id(tag_lookup const&,
                                                 std::string_view station_id);

::nigiri::location_idx_t get_location_idx(tag_lookup const&,
                                          ::nigiri::timetable const&,
                                          std::string_view station_id);

}  // namespace motis::nigiri