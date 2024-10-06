#pragma once

#include <string_view>
#include <variant>

#include "nigiri/routing/journey.h"
#include "nigiri/types.h"

#include "osr/location.h"
#include "osr/routing/route.h"
#include "osr/types.h"

#include "motis-api/motis-api.h"
#include "motis/elevators/elevators.h"
#include "motis/fwd.h"
#include "motis/types.h"

namespace motis {

using place_t = std::variant<osr::location, nigiri::location_idx_t>;

inline std::ostream& operator<<(std::ostream& out, place_t const p) {
  return std::visit([&](auto const l) -> std::ostream& { return out << l; }, p);
}

using street_routing_cache_t = hash_map<std::tuple<osr::location,
                                                   osr::location,
                                                   osr::search_profile,
                                                   std::vector<bool>>,
                                        std::optional<osr::path>>;

api::Place to_place(nigiri::timetable const&, place_t, std::string_view);

api::Itinerary journey_to_response(
    osr::ways const&,
    osr::lookup const&,
    nigiri::timetable const&,
    osr::platforms const&,
    elevators const* e,
    nigiri::rt_timetable const*,
    vector_map<nigiri::location_idx_t, osr::platform_idx_t> const& matches,
    bool const wheelchair,
    nigiri::routing::journey const&,
    place_t const& start,
    place_t const& dest,
    street_routing_cache_t&,
    osr::bitvec<osr::node_idx_t>& blocked_mem);

}  // namespace motis