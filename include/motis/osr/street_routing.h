#pragma once

#include <array>
#include <cassert>
#include <memory>
#include <optional>
#include <vector>

#include "osr/location.h"
#include "osr/routing/profile.h"
#include "osr/routing/route.h"

#include "nigiri/special_stations.h"
#include "motis-api/motis-api.h"

#include "motis/flex/flex_routing_data.h"
#include "motis/fwd.h"
#include "motis/match_platforms.h"
#include "motis/osr/one_to_many_searches.h"
#include "motis/osr/parameters.h"
#include "motis/transport_mode.h"
#include "motis/types.h"

namespace motis {

struct output {
  output() = default;
  virtual ~output() = default;
  output(output const&) = default;
  output(output&&) = default;
  output& operator=(output const&) = default;
  output& operator=(output&&) = default;

  virtual api::ModeEnum get_mode() const = 0;
  virtual osr::search_profile get_profile() const = 0;
  virtual bool is_time_dependent() const = 0;
  virtual transport_mode_t get_cache_key() const = 0;
  virtual osr::sharing_data const* get_sharing_data() const = 0;
  virtual bool allows_free_floating_return_at(osr::location const&) const {
    return false;
  }
  virtual void annotate_leg(nigiri::lang_t const&,
                            osr::node_idx_t from_node,
                            osr::node_idx_t to_node,
                            api::Leg&) const = 0;
  virtual api::Place get_place(nigiri::lang_t const&,
                               osr::node_idx_t,
                               std::optional<std::string> const& tz) const = 0;
};

using street_routing_cache_key_t = std::tuple<osr::location,
                                              osr::location,
                                              transport_mode_t,
                                              nigiri::unixtime_t,
                                              osr::direction,
                                              bool>;

using street_routing_cache_t =
    hash_map<street_routing_cache_key_t, std::optional<osr::path>>;

api::Itinerary dummy_itinerary(api::Place const& from,
                               api::Place const& to,
                               api::ModeEnum,
                               nigiri::unixtime_t const start_time,
                               nigiri::unixtime_t const end_time,
                               unsigned const api_version,
                               bool cancelled = false);

api::Itinerary street_routing(
    osr::ways const&,
    osr::lookup const&,
    elevators const*,
    osr::elevation_storage const*,
    nigiri::lang_t const& lang,
    api::Place const& from,
    api::Place const& to,
    output const&,
    std::optional<nigiri::unixtime_t> start_time,
    std::optional<nigiri::unixtime_t> end_time,
    double max_matching_distance,
    osr_parameters const&,
    street_routing_cache_t&,
    osr::bitvec<osr::node_idx_t>& blocked_mem,
    unsigned api_version,
    bool detailed_leg = true,
    std::chrono::seconds max = std::chrono::seconds{3600},
    precomputed_route const& precomputed = {});

}  // namespace motis
