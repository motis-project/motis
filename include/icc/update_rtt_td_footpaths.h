#pragma once

#include <vector>

#include "osr/lookup.h"
#include "osr/routing/route.h"
#include "osr/ways.h"

#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

#include "icc/compute_footpaths.h"
#include "icc/elevators/elevators.h"
#include "icc/match_platforms.h"

namespace icc {

std::vector<nigiri::td_footpath> get_td_footpaths(
    osr::ways const&,
    osr::lookup const&,
    osr::platforms const&,
    nigiri::timetable const&,
    point_rtree<nigiri::location_idx_t> const&,
    elevators const&,
    elevator_footpath_map_t const&,
    platform_matches_t const&,
    osr::location start,
    osr::direction,
    osr::search_profile,
    osr::bitvec<osr::node_idx_t>& blocked);

void update_rtt_td_footpaths(osr::ways const&,
                             osr::lookup const&,
                             osr::platforms const&,
                             nigiri::timetable const&,
                             point_rtree<nigiri::location_idx_t> const&,
                             elevators const&,
                             elevator_footpath_map_t const&,
                             platform_matches_t const&,
                             nigiri::rt_timetable&);

}  // namespace icc