#pragma once

#include <string_view>

#include "motis/core/schedule/schedule.h"
#include "motis/core/journey/extern_trip.h"

namespace motis {

trip const* get_gtfs_trip(schedule const&, gtfs_trip_id const&);

concrete_trip get_trip(schedule const&, std::string_view eva_nr,
                     uint32_t train_nr, unixtime timestamp,
                     std::string_view target_eva_nr, unixtime target_timestamp,
                     std::string_view line_id, bool fuzzy = false);

concrete_trip get_trip(schedule const&, extern_trip const&);

std::optional<concrete_trip> find_trip(schedule const&, primary_trip_id const&,
                                       day_idx_t);

std::optional<concrete_trip> find_trip(schedule const&, full_trip_id const&,
                                       day_idx_t);

unsigned stop_seq_to_stop_idx(trip const&, unsigned stop_seq);

}  // namespace motis
