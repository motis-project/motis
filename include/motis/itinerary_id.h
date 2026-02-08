#pragma once

#include <string>
#include "motis-api/motis-api.h"
#include "motis/endpoints/map/stops.h"
#include "motis/endpoints/stop_times.h"

namespace motis {

std::string generate_itinerary_id(api::Itinerary const&);
api::Itinerary reconstruct_itinerary(ep::stops const&,
                                     ep::stop_times const&,
                                     std::string const&);

}  // namespace motis
