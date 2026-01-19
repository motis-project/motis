#pragma once

#include <optional>

#include "osr/location.h"
#include "osr/routing/profile.h"
#include "osr/routing/route.h"

#include "motis-api/motis-api.h"
#include "motis/fwd.h"
#include "motis/match_platforms.h"
#include "motis/osr/parameters.h"
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
  default_output(osr::ways const&, nigiri::transport_mode_id_t);
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
  nigiri::transport_mode_id_t id_;
};

using street_routing_cache_key_t = std::
    tuple<osr::location, osr::location, transport_mode_t, nigiri::unixtime_t>;

using street_routing_cache_t =
    hash_map<street_routing_cache_key_t, std::optional<osr::path>>;

api::Itinerary dummy_itinerary(api::Place const& from,
                               api::Place const& to,
                               api::ModeEnum,
                               nigiri::unixtime_t const start_time,
                               nigiri::unixtime_t const end_time);

api::Itinerary street_routing(osr::ways const&,
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
                              std::chrono::seconds max = std::chrono::seconds{
                                  3600});

}  // namespace motis
