#pragma once

#include "osr/location.h"
#include "osr/routing/profile.h"
#include "osr/types.h"

#include "nigiri/routing/query.h"

#include "motis/flex/mode_id.h"
#include "motis/fwd.h"
#include "motis/match_platforms.h"
#include "motis/osr/parameters.h"

namespace motis::flex {

using flex_routings_t =
    hash_map<std::pair<nigiri::flex_stop_seq_idx_t, nigiri::stop_idx_t>,
             std::vector<mode_id>>;

osr::sharing_data prepare_sharing_data(nigiri::timetable const&,
                                       osr::ways const&,
                                       osr::lookup const&,
                                       osr::platforms const*,
                                       flex_areas const&,
                                       platform_matches_t const*,
                                       mode_id,
                                       osr::direction,
                                       flex_routing_data&);

bool is_in_flex_stop(nigiri::timetable const&,
                     osr::ways const&,
                     flex_areas const&,
                     flex_routing_data const&,
                     nigiri::flex_stop_t const&,
                     osr::node_idx_t);

flex_routings_t get_flex_routings(nigiri::timetable const&,
                                  point_rtree<nigiri::location_idx_t> const&,
                                  nigiri::routing::start_time_t,
                                  geo::latlng const&,
                                  osr::direction,
                                  std::chrono::seconds max);

void add_flex_td_offsets(osr::ways const&,
                         osr::lookup const&,
                         osr::platforms const*,
                         platform_matches_t const*,
                         way_matches_storage const*,
                         nigiri::timetable const&,
                         flex_areas const&,
                         point_rtree<nigiri::location_idx_t> const&,
                         nigiri::routing::start_time_t,
                         osr::location const&,
                         osr::direction,
                         std::chrono::seconds max,
                         double const max_matching_distance,
                         osr_parameters const&,
                         flex_routing_data&,
                         nigiri::routing::td_offsets_t&,
                         std::map<std::string, std::uint64_t>& stats);

}  // namespace motis::flex
