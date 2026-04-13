#pragma once

#include <string>

#include "motis-api/motis-api.h"
#include "motis/endpoints/stop_times.h"

namespace motis {

std::string get_single_leg_id(api::Leg const&, nigiri::lang_t const&);

api::Itinerary reconstruct_itinerary(ep::stop_times const&,
                                     nigiri::shapes_storage const*,
                                     rt const&,
                                     std::string const& id,
                                     bool const require_display_name = true);

}  // namespace motis
