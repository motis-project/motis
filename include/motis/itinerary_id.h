#pragma once

#include <string>

#include "motis-api/motis-api.h"
#include "motis/endpoints/stop_times.h"

namespace motis {

std::string get_single_leg_id(api::Leg const&,
                              std::string const& leg_display_name);

api::Itinerary reconstruct_itinerary(
    ep::stop_times const&,
    nigiri::shapes_storage const*,
    rt const&,
    std::string const& id,
    bool const require_display_name_match = true,
    bool const join_interlined_legs = true,
    bool const detailed_transfers = false,
    bool const detailed_legs = true,
    bool const with_fares = false,
    bool const with_scheduled_skipped_stops = false,
    nigiri::lang_t const& lang = nigiri::lang_t{});

}  // namespace motis
