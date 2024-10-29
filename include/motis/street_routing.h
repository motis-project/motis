#pragma once

#include "osr/location.h"
#include "osr/routing/route.h"

#include "motis-api/motis-api.h"
#include "motis/fwd.h"
#include "motis/types.h"

namespace motis {

using transport_mode_t = std::uint32_t;

using street_routing_cache_key_t = std::
    tuple<osr::location, osr::location, transport_mode_t, nigiri::unixtime_t>;

using street_routing_cache_t =
    hash_map<street_routing_cache_key_t, std::optional<osr::path>>;

std::optional<osr::path> get_path(osr::ways const&,
                                  osr::lookup const&,
                                  elevators const*,
                                  osr::sharing_data const*,
                                  osr::location const& from,
                                  osr::location const& to,
                                  transport_mode_t const,
                                  osr::search_profile const,
                                  std::chrono::sys_seconds const start_time,
                                  street_routing_cache_t&,
                                  osr::bitvec<osr::node_idx_t>& blocked_mem);

template <std::int64_t Precision = 5>
api::EncodedPolyline to_polyline(geo::polyline const& polyline);

std::vector<api::StepInstruction> get_step_instructions(
    osr::ways const&, std::span<osr::path::segment const>);

api::Itinerary street_routing(osr::location const& from,
                              osr::location const& to,
                              api::ModeEnum,
                              street_routing_cache_t&);

}  // namespace motis