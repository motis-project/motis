#pragma once

#include <chrono>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "nigiri/routing/clasz_mask.h"
#include "nigiri/types.h"

#include "motis-api/motis-api.h"
#include "motis/adr_extend_tt.h"
#include "motis/fwd.h"
#include "motis/osr/parameters.h"
#include "motis/place.h"
#include "motis/rental_options.h"

namespace nigiri::routing {
struct journey;
}  // namespace nigiri::routing

namespace motis {

class ItineraryId;  // protobuf

namespace ep {
struct routing;
struct stop_times;
}  // namespace ep

ItineraryId decode_itinerary_id(std::string const& base64_id);

struct first_last_mile_options {
  api::PedestrianProfileEnum pedestrian_profile_;
  api::ElevationCostsEnum elevation_costs_;
  osr_parameters osr_params_;
  double max_matching_distance_;
  std::chrono::seconds max_pre_transit_;
  std::chrono::seconds max_post_transit_;
  rental_options pre_transit_;
  rental_options post_transit_;
  std::vector<api::ModeEnum> pre_transit_modes_;
  std::vector<api::ModeEnum> post_transit_modes_;
};

first_last_mile_options make_first_last_mile_options(
    api::refreshItinerary_params const&);

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
    bool const with_scheduled_skipped_stops = false,
    nigiri::lang_t const& lang = nigiri::lang_t{},
    std::size_t num_leg_alternatives = 0U,
    nigiri::routing::clasz_mask_t allowed_claszes =
        nigiri::routing::all_clasz_allowed(),
    bool require_bike_transport = false,
    bool require_car_transport = false,
    nigiri::profile_idx_t prf_idx = 0U,
    first_last_mile_options const& flm =
        make_first_last_mile_options(api::refreshItinerary_params{}));

}  // namespace motis
