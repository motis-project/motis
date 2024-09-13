#pragma once

#include "nigiri/timetable.h"

#include "osr/types.h"

#include "motis/fwd.h"
#include "motis/types.h"

namespace motis {

using elevator_footpath_map_t = hash_map<
    osr::node_idx_t,
    hash_set<std::pair<nigiri::location_idx_t, nigiri::location_idx_t>>>;

elevator_footpath_map_t compute_footpaths(nigiri::timetable&,
                                          osr::ways const&,
                                          osr::lookup const&,
                                          osr::platforms const&,
                                          bool update_coordinates);

void write(std::filesystem::path const&, elevator_footpath_map_t const&);

cista::wrapped<elevator_footpath_map_t> read_elevator_footpath_map(
    std::filesystem::path const&);

}  // namespace motis