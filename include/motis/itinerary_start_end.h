#pragma once

#include "journey_to_response.h"

namespace motis {
void resolve_itinerary_start_end(
    api::Itinerary& itinerary, adr::reverse const& r_,
    adr::typeahead const& ta_, adr::formatter const& f_, adr_ext const* ae_,
    nigiri::timetable const* tt_, tag_lookup const* tags_, osr::ways const* w_,
    osr::platforms const* pl_, platform_matches_t const* matches_);
}  // namespace motis