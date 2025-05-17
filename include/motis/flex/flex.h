#pragma once

#include <functional>

#include "osr/location.h"
#include "osr/routing/profile.h"
#include "osr/types.h"

#include "nigiri/routing/query.h"

#include "motis/flex/mode_id.h"
#include "motis/fwd.h"
#include "motis/match_platforms.h"

namespace motis::flex {

osr::sharing_data prepare_sharing_data(nigiri::timetable const&,
                                       osr::ways const&,
                                       osr::lookup const&,
                                       osr::platforms const*,
                                       platform_matches_t const*,
                                       mode_id,
                                       osr::direction,
                                       flex_routing_data&);

bool is_in_flex_stop(nigiri::timetable const&,
                     osr::ways const&,
                     flex_routing_data const&,
                     nigiri::flex_stop_t const&,
                     osr::node_idx_t);

void for_each_flex_transport(nigiri::timetable const&,
                             point_rtree<nigiri::location_idx_t> const&,
                             nigiri::routing::start_time_t,
                             geo::latlng const&,
                             osr::direction,
                             std::chrono::seconds,
                             std::function<void(mode_id)> const&);

void add_flex_td_offsets(osr::ways const&,
                         osr::lookup const&,
                         osr::platforms const*,
                         platform_matches_t const*,
                         nigiri::timetable const&,
                         point_rtree<nigiri::location_idx_t> const&,
                         nigiri::routing::start_time_t,
                         osr::location const&,
                         osr::direction,
                         std::chrono::seconds max,
                         double const max_matching_distance,
                         flex_routing_data&,
                         nigiri::routing::td_offsets_t&);

}  // namespace motis::flex
