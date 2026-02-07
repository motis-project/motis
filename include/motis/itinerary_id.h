#pragma once

#include "motis/fwd.h"

namespace motis {

std::string generate_itinerary_id(api::Itinerary const&) { return ""; }

api::Itinerary reconstruct_itinerary(nigiri::timetable const&,
                                     std::string const&) {
  return {};
}

}  // namespace motis