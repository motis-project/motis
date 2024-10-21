#pragma once

#include <string_view>
#include <variant>

#include "nigiri/routing/journey.h"
#include "nigiri/rt/frun.h"
#include "nigiri/types.h"

#include "osr/location.h"
#include "osr/routing/route.h"
#include "osr/types.h"

#include "motis-api/motis-api.h"
#include "motis/elevators/elevators.h"
#include "motis/fwd.h"
#include "motis/match_platforms.h"
#include "motis/types.h"

namespace motis {

struct tt_location {
  explicit tt_location(nigiri::rt::run_stop const& stop);
  explicit tt_location(
      nigiri::location_idx_t l,
      nigiri::location_idx_t scheduled = nigiri::location_idx_t::invalid());

  nigiri::location_idx_t l_;
  nigiri::location_idx_t scheduled_;
};

using place_t = std::variant<osr::location, tt_location>;

inline std::ostream& operator<<(std::ostream& out, place_t const p) {
  return std::visit([&](auto const l) -> std::ostream& { return out << l; }, p);
}

using street_routing_cache_t = hash_map<std::tuple<osr::location,
                                                   osr::location,
                                                   osr::search_profile,
                                                   std::vector<bool>>,
                                        std::optional<osr::path>>;

api::Place to_place(osr::location, std::string_view name);

api::Place to_place(nigiri::timetable const&,
                    tag_lookup const&,
                    osr::ways const* w,
                    osr::platforms const* pl,
                    platform_matches_t const* matches,
                    place_t l,
                    place_t start = osr::location{},
                    place_t dest = osr::location{},
                    std::string_view name = "");

double get_level(osr::ways const*,
                 osr::platforms const*,
                 platform_matches_t const*,
                 nigiri::location_idx_t);

api::Itinerary journey_to_response(osr::ways const*,
                                   osr::lookup const*,
                                   osr::platforms const*,
                                   nigiri::timetable const&,
                                   tag_lookup const&,
                                   elevators const* e,
                                   nigiri::rt_timetable const*,
                                   platform_matches_t const* matches,
                                   nigiri::shapes_storage const*,
                                   bool const wheelchair,
                                   nigiri::routing::journey const&,
                                   place_t const& start,
                                   place_t const& dest,
                                   street_routing_cache_t&,
                                   osr::bitvec<osr::node_idx_t>& blocked_mem);

}  // namespace motis