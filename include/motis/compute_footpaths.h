#pragma once

#include "cista/memory_holder.h"

#include "osr/types.h"

#include "motis/fwd.h"
#include "motis/types.h"

#include "motis/tag_lookup.h"

namespace motis {

using elevator_footpath_map_t = hash_map<
    osr::node_idx_t,
    hash_set<std::pair<nigiri::location_idx_t, nigiri::location_idx_t>>>;

elevator_footpath_map_t compute_footpaths(
    osr::ways const&,
    osr::lookup const&,
    osr::platforms const&,
    nigiri::timetable&,
    tag_lookup const&,
    bool const update_coordinates,
    bool const extend_missing,
    std::chrono::seconds const max_duration);

}  // namespace motis