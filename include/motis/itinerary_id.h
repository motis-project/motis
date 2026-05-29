#pragma once

#include <cstddef>
#include <string>

#include "nigiri/routing/clasz_mask.h"
#include "nigiri/types.h"

#include "motis-api/motis-api.h"
#include "motis/adr_extend_tt.h"
#include "motis/fwd.h"
#include "motis/place.h"

namespace nigiri::routing {
struct journey;
}  // namespace nigiri::routing

namespace motis {

namespace ep {
struct routing;
struct stop_times;
}  // namespace ep

std::string generate_itinerary_id(nigiri::routing::journey const&,
                                  tag_lookup const&,
                                  nigiri::timetable const&,
                                  nigiri::rt_timetable const*,
                                  osr::ways const*,
                                  osr::platforms const*,
                                  platform_matches_t const*,
                                  adr_ext const*,
                                  tz_map_t const*,
                                  place_t const& start,
                                  place_t const& dest);

api::Itinerary reconstruct_itinerary(
    ep::routing const&,
    ep::stop_times const&,
    rt const&,
    std::string const& id,
    bool const require_display_name_match = true,
    bool const join_interlined_legs = true,
    bool const detailed_transfers = false,
    bool const detailed_legs = true,
    bool const with_fares = false,
    bool const with_scheduled_skipped_stops = false,
    nigiri::lang_t const& lang = nigiri::lang_t{},
    std::size_t num_leg_alternatives = 0U,
    nigiri::routing::clasz_mask_t allowed_claszes =
        nigiri::routing::all_clasz_allowed(),
    bool require_bike_transport = false,
    bool require_car_transport = false,
    nigiri::profile_idx_t prf_idx = 0U);

}  // namespace motis
