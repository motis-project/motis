#pragma once

#include <string_view>

#include "motis/core/schedule/schedule.h"
#include "motis/core/journey/extern_trip.h"

namespace motis {

trip const* get_trip(schedule const&, std::string_view eva_nr,
                     uint32_t train_nr, std::time_t timestamp,
                     std::string_view target_eva_nr,
                     std::time_t target_timestamp, std::string_view line_id,
                     bool fuzzy = false);

trip const* get_trip(schedule const&, extern_trip const&);

trip const* find_trip(schedule const&, primary_trip_id);

trip const* find_trip(schedule const&, full_trip_id);

trip const* get_trip(schedule const&, std::string const& trip_id, std::time_t);

}  // namespace motis
