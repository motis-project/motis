#pragma once

#include "cista/memory_holder.h"

#include "osr/types.h"

#include "motis/fwd.h"
#include "motis/types.h"

namespace motis {

using elevator_footpath_map_t = hash_map<
    osr::node_idx_t,
    hash_set<std::pair<nigiri::location_idx_t, nigiri::location_idx_t>>>;

elevator_footpath_map_t compute_footpaths(osr::ways const&,
                                          osr::lookup const&,
                                          osr::platforms const&,
                                          nigiri::timetable&,
                                          osr::elevation_storage const*,
                                          bool update_coordinates,
                                          bool extend_missing,
                                          std::chrono::seconds max_duration,
                                          double max_matching_distance);

}  // namespace motis