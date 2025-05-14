#pragma once

#include "osr/location.h"
#include "osr/routing/profile.h"
#include "osr/types.h"

#include "nigiri/routing/query.h"

#include "motis-api/motis-api.h"
#include "motis/flex/mode_id.h"
#include "motis/fwd.h"
#include "motis/match_platforms.h"

namespace motis {

namespace ep {
struct routing;
}

namespace flex {

osr::sharing_data prepare_sharing_data(nigiri::timetable const&,
                                       osr::ways const&,
                                       osr::lookup const&,
                                       osr::platforms const*,
                                       platform_matches_t const*,
                                       mode_id,
                                       osr::direction,
                                       flex_routing_data&);

void add_flex_td_offsets(ep::routing const&,
                         osr::location const&,
                         osr::direction,
                         double const max_matching_distance,
                         std::chrono::seconds const max,
                         nigiri::routing::start_time_t const&,
                         nigiri::routing::td_offsets_t& ret);

}  // namespace flex

}  // namespace motis
