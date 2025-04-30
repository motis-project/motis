#pragma once

#include <optional>

#include "osr/location.h"
#include "osr/routing/profile.h"
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

api::Itinerary dummy_itinerary(api::Place const& from,
                               api::Place const& to,
                               api::ModeEnum,
                               nigiri::unixtime_t const start_time,
                               nigiri::unixtime_t const end_time);

api::Itinerary route(osr::ways const&,
                     osr::lookup const&,
                     gbfs::gbfs_routing_data&,
                     elevators const*,
                     osr::elevation_storage const*,
                     api::Place const& from,
                     api::Place const& to,
                     api::ModeEnum,
                     osr::search_profile,
                     nigiri::unixtime_t start_time,
                     std::optional<nigiri::unixtime_t> end_time,
                     double max_matching_distance,
                     gbfs::gbfs_products_ref,
                     street_routing_cache_t&,
                     osr::bitvec<osr::node_idx_t>& blocked_mem,
                     unsigned api_version,
                     std::chrono::seconds max = std::chrono::seconds{3600},
                     bool dummy = false);

}  // namespace motis
