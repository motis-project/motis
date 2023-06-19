#pragma once

#include <string>
#include <vector>

#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace motis::nigiri {

::nigiri::location_id motis_station_to_nigiri_id(
    std::vector<std::string> const& tags, std::string_view station_id);

::nigiri::location_idx_t get_location_idx(std::vector<std::string> const& tags,
                                          ::nigiri::timetable const& tt,
                                          std::string_view station_id);

}  // namespace motis::nigiri