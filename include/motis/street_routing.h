#pragma once

#include <optional>

#include "osr/location.h"
#include "osr/routing/profile.h"
#include "osr/routing/route.h"

#include "motis-api/motis-api.h"
#include "motis/fwd.h"
#include "motis/match_platforms.h"
#include "motis/types.h"

namespace motis {

using transport_mode_t = std::uint32_t;

struct output {
  output() = default;
  virtual ~output() = default;
  output(output const&) = default;
  output(output&&) = default;
  output& operator=(output const&) = default;
  output& operator=(output&&) = default;

  virtual transport_mode_t get_cache_key(osr::search_profile) const = 0;
  virtual api::VertexTypeEnum get_vertex_type() const = 0;
  virtual std::string get_node_name(osr::node_idx_t const n) const = 0;
  virtual osr::sharing_data const* get_sharing_data() const = 0;
  virtual geo::latlng get_node_pos(osr::node_idx_t) const = 0;
  virtual void annotate_leg(osr::node_idx_t from_node,
                            osr::node_idx_t to_node,
                            api::Leg&) const = 0;
  virtual void annotate_place(api::Place&) const = 0;
};

struct default_output final : public output {
  ~default_output() override;
  transport_mode_t get_cache_key(osr::search_profile) const override;
  api::VertexTypeEnum get_vertex_type() const override;
  std::string get_node_name(osr::node_idx_t) const override;
  osr::sharing_data const* get_sharing_data() const override;
  geo::latlng get_node_pos(osr::node_idx_t) const override;
  void annotate_leg(osr::node_idx_t, osr::node_idx_t, api::Leg&) const override;
  void annotate_place(api::Place&) const override;
};

extern default_output g_default_output;

using street_routing_cache_key_t = std::
    tuple<osr::location, osr::location, transport_mode_t, nigiri::unixtime_t>;

using street_routing_cache_t =
    hash_map<street_routing_cache_key_t, std::optional<osr::path>>;

api::Itinerary dummy_itinerary(api::Place const& from,
                               api::Place const& to,
                               api::ModeEnum,
                               nigiri::unixtime_t const start_time,
                               nigiri::unixtime_t const end_time);

api::Itinerary street_routing(
    osr::ways const&,
    osr::lookup const&,
    osr::platforms const*,
    platform_matches_t const*,
    nigiri::timetable const*,
    elevators const*,
    osr::elevation_storage const*,
    api::Place const& from,
    api::Place const& to,
    api::ModeEnum,
    nigiri::unixtime_t start_time,
    std::optional<nigiri::unixtime_t> end_time,
    double max_matching_distance,
    street_routing_cache_t&,
    osr::bitvec<osr::node_idx_t>& blocked_mem,
    unsigned api_version,
    std::chrono::seconds max = std::chrono::seconds{3600},
    bool dummy = false);

}  // namespace motis
