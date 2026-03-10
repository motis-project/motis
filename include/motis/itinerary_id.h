#pragma once

#include <string>

#include "motis-api/motis-api.h"
#include "motis/endpoints/stop_times.h"

namespace motis {

std::string get_leg_id(api::Leg const&);

api::Itinerary reconstruct_itinerary(ep::stop_times const&,
                                     nigiri::shapes_storage const*,
                                     rt const&,
                                     std::string const& id);

}  // namespace motis
