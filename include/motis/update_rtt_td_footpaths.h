#pragma once

#include <vector>

#include "osr/lookup.h"
#include "osr/routing/route.h"

#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

#include "motis/compute_footpaths.h"
#include "motis/data.h"
#include "motis/elevators/elevators.h"
#include "motis/fwd.h"
#include "motis/match_platforms.h"
#include "motis/osr/parameters.h"

namespace motis {

using nodes_t = std::vector<osr::node_idx_t>;
using states_t = std::vector<bool>;

osr::bitvec<osr::node_idx_t>& set_blocked(nodes_t const&,
                                          states_t const&,
                                          osr::bitvec<osr::node_idx_t>&);

std::vector<nigiri::td_footpath> get_td_footpaths(
    osr::ways const&,
    osr::lookup const&,
    osr::platforms const&,
    nigiri::timetable const&,
    nigiri::rt_timetable const*,
    point_rtree<nigiri::location_idx_t> const&,
    elevators const&,
    platform_matches_t const&,
    nigiri::location_idx_t start_l,
    osr::location start,
    osr::direction,
    osr::search_profile,
    std::chrono::seconds max,
    double max_matching_distance,
    osr_parameters const&,
    osr::bitvec<osr::node_idx_t>& blocked_mem);

std::optional<std::pair<nodes_t, states_t>> get_states_at(osr::ways const&,
                                                          osr::lookup const&,
                                                          elevators const&,
                                                          nigiri::unixtime_t,
                                                          geo::latlng const&);

void update_rtt_td_footpaths(
    osr::ways const&,
    osr::lookup const&,
    osr::platforms const&,
    nigiri::timetable const&,
    point_rtree<nigiri::location_idx_t> const&,
    elevators const&,
    platform_matches_t const&,
    hash_set<std::pair<nigiri::location_idx_t, osr::direction>> const& tasks,
    nigiri::rt_timetable const* old_rtt,
    nigiri::rt_timetable&,
    std::chrono::seconds max);

void update_rtt_td_footpaths(osr::ways const&,
                             osr::lookup const&,
                             osr::platforms const&,
                             nigiri::timetable const&,
                             point_rtree<nigiri::location_idx_t> const&,
                             elevators const&,
                             elevator_footpath_map_t const&,
                             platform_matches_t const&,
                             nigiri::rt_timetable&,
                             std::chrono::seconds max);

}  // namespace motis
