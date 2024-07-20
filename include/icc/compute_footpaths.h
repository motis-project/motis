#pragma once

#include "nigiri/timetable.h"

#include "osr/lookup.h"
#include "osr/platforms.h"
#include "osr/ways.h"

#include "icc/types.h"

namespace icc {

using elevator_footpath_map_t = hash_map<
    osr::node_idx_t,
    hash_set<std::pair<nigiri::location_idx_t, nigiri::location_idx_t>>>;

elevator_footpath_map_t compute_footpaths(nigiri::timetable&,
                                          osr::ways const&,
                                          osr::lookup const&,
                                          osr::platforms const&,
                                          bool update_coordinates);

}  // namespace icc