#pragma once

#include "osr/location.h"
#include "osr/routing/profile.h"
#include "osr/types.h"

#include "motis/osr/one_to_many_searches.h"
#include "motis/osr/street_routing.h"

#include "nigiri/routing/query.h"

#include "motis/flex/mode_payload.h"
#include "motis/fwd.h"
#include "motis/match_platforms.h"
#include "motis/osr/parameters.h"

namespace motis::flex {

// Key: stop sequence, boarding and alighting stop index (travel order).
// All transports of one key share the same street routing.
using flex_routings_t = hash_map<
    std::pair<nigiri::flex_stop_seq_idx_t,
              std::pair<nigiri::stop_idx_t, nigiri::stop_idx_t>>,
    std::vector<mode_payload>>;

// Logs how much of the mode payload's capacity (transports, stop rows per
// flex transport) the timetable uses; throws if it does not fit.
void verify_flex_limits(nigiri::timetable const&);

osr::sharing_data prepare_sharing_data(nigiri::timetable const&,
                                       osr::ways const&,
                                       osr::lookup const&,
                                       osr::platforms const*,
                                       flex_areas const&,
                                       platform_matches_t const*,
                                       mode_payload,
                                       flex_routing_data&);

flex_routings_t get_flex_routings(nigiri::timetable const&,
                                  point_rtree<nigiri::location_idx_t> const&,
                                  nigiri::routing::start_time_t,
                                  geo::latlng const&,
                                  osr::direction,
                                  std::chrono::seconds max,
                                  osr_parameters const&);

// When the ride starts (pickup) and ends (drop-off), relative to the start
// of the whole access / egress / direct path in travel direction.
struct flex_ride {
  nigiri::duration_t pickup_;
  nigiri::duration_t drop_off_;
};

// Departure times (start of the whole path) at which transport `id`
// operating on service `day` (midnight UTC of its traffic day) can be used:
// the ride starts inside the boarding stop's window and ends inside the
// alighting stop's window,
//   W = [a_from - pickup, b_from - pickup]
//       ∩ [a_to - drop_off, b_to - drop_off].
// Both window ends are inclusive. A zero-length window [T, T] is a fixed
// departure at minute T (the Austrian feeds encode call-taxis with a fixed
// departure this way; GTFS-Flex has no valid encoding for it). Returned
// half-open at minute resolution: [start, end + 1 min).
nigiri::interval<nigiri::unixtime_t> get_departure_window(
    nigiri::timetable const&,
    mode_payload id,
    nigiri::unixtime_t day,
    flex_ride);

// Sets the pickup / drop-off windows of a FLEX leg for service day `day`.
void set_flex_windows(nigiri::timetable const&,
                      mode_payload id,
                      date::sys_days day,
                      api::Leg&);

// Moves a direct FLEX itinerary (routed for `ids.front()`, starting or - for
// arrive_by - ending at `time`) to the first departure (latest for arrive_by)
// that one of the transports `ids` offers on a traffic day: the ride has to
// start inside the window of the boarding stop and end inside the window of
// the alighting stop. Returns false if there is none, or if the itinerary
// contains no ride (the car_sharing profile may walk all the way).
bool fit_direct_to_windows(nigiri::timetable const&,
                           std::vector<mode_payload> const& ids,
                           nigiri::unixtime_t time,
                           bool arrive_by,
                           api::Itinerary&);

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
                         std::map<std::string, std::uint64_t>& stats,
                         one_to_many_side* states = nullptr);

}  // namespace motis::flex
