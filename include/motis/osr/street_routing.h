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
  virtual void annotate_leg(nigiri::lang_t const&,
                            osr::node_idx_t from_node,
                            osr::node_idx_t to_node,
                            api::Leg&) const = 0;
  virtual api::Place get_place(nigiri::lang_t const&,
                               osr::node_idx_t,
                               std::optional<std::string> const& tz) const = 0;
};

struct default_output final : public output {
  default_output(osr::ways const&, osr::search_profile);
  default_output(osr::ways const&, transport_mode_t);
  ~default_output() override;

  bool is_time_dependent() const override;
  api::ModeEnum get_mode() const override;
  osr::search_profile get_profile() const override;
  transport_mode_t get_cache_key() const override;
  osr::sharing_data const* get_sharing_data() const override;
  void annotate_leg(nigiri::lang_t const&,
                    osr::node_idx_t,
                    osr::node_idx_t,
                    api::Leg&) const override;
  api::Place get_place(nigiri::lang_t const&,
                       osr::node_idx_t,
                       std::optional<std::string> const& tz) const override;

  osr::ways const& w_;
  osr::search_profile profile_;
  transport_mode_t id_;
};

using street_routing_cache_key_t = std::tuple<osr::location,
                                              osr::location,
                                              transport_mode_t,
                                              nigiri::unixtime_t,
                                              osr::direction>;

using street_routing_cache_t =
    hash_map<street_routing_cache_key_t, std::optional<osr::path>>;

// The place a set of offsets was computed from: the journey's start or its
// destination, which nigiri addresses as these two special stations.
constexpr nigiri::special_station flip(nigiri::special_station const s) {
  return s == nigiri::special_station::kStart ? nigiri::special_station::kEnd
                                              : nigiri::special_station::kStart;
}

// One one-to-many street search of a request, kept so that the offset legs of
// journeys and their alternatives can be reconstructed from it instead of
// being routed again.
struct one_to_many_search {
  std::unique_ptr<osr::one_to_many_state> state_;
  hash_map<nigiri::location_idx_t, std::size_t> dest_idx_;  // stop -> dest
  // Flex only: the additional nodes the search ran with. The routing rebuilds
  // its own for the next flex group, so the legs reconstructed from this
  // search are rendered with these.
  flex::flex_additional_nodes flex_additional_nodes_;
};

// The retained searches of one place. Several transport modes can share a
// search: a flex routing group runs one search for every transport of a stop
// sequence that boards at the same stop.
struct one_to_many_side {
  std::vector<one_to_many_search> searches_;
  hash_map<transport_mode_t, std::size_t> by_mode_;
};

struct one_to_many_searches {
  // Only `kStart` and `kEnd` address a side; no offsets are computed from a
  // via station.
  static std::size_t idx(nigiri::special_station const s) {
    assert(s == nigiri::special_station::kStart ||
           s == nigiri::special_station::kEnd);
    return static_cast<std::size_t>(s);
  }

  one_to_many_side const& operator[](nigiri::special_station const s) const {
    return sides_[idx(s)];
  }
  one_to_many_side& operator[](nigiri::special_station const s) {
    return sides_[idx(s)];
  }

  std::array<one_to_many_side, 2> sides_;
};

// Reconstruct destination `dest_idx_` of `state_` instead of routing.
struct precomputed_route {
  osr::one_to_many_state* state_{nullptr};
  std::size_t dest_idx_{0U};
  // Borrowed from the `one_to_many_search` that owns them.
  flex::flex_additional_nodes const* flex_additional_nodes_{nullptr};
};

// A request's retained searches as seen from one journey. `flipped_` maps the
// journey's sides onto the sides the offsets were computed for: alternatives
// of an arriveBy query are rendered from a direction-flipped query, where the
// special start station refers to the journey's destination place.
struct one_to_many_view {
  // Returns an empty route unless the leg touches the journey's start or end
  // place and that side has a search covering `target`.
  precomputed_route find(nigiri::location_idx_t leg_from,
                         nigiri::location_idx_t leg_to,
                         transport_mode_t,
                         nigiri::location_idx_t target) const;

  one_to_many_searches const* searches_{nullptr};
  bool flipped_{false};
};

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
