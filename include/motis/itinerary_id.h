#pragma once

#include <cstddef>
#include <string>

#include "nigiri/routing/clasz_mask.h"
#include "nigiri/types.h"

#include "motis-api/motis-api.h"
#include "motis/fwd.h"

namespace motis {

namespace ep {
struct routing;
struct stop_times;
}  // namespace ep

std::string get_single_leg_id(api::Leg const&,
                              std::string const& leg_display_name);

std::string generate_itinerary_id(
    api::Itinerary const& itin,
    std::vector<std::string> const& default_display_names,
    std::vector<std::size_t> const& default_display_names_indices);

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
