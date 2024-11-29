#pragma once

#include <optional>
#include <utility>
#include <vector>

#include "osr/location.h"
#include "osr/types.h"

#include "nigiri/routing/clasz_mask.h"

#include "motis-api/motis-api.h"
#include "motis/elevators/elevators.h"
#include "motis/fwd.h"
#include "motis/match_platforms.h"

namespace motis::ep {

struct routing {
  api::plan_response operator()(boost::urls::url_view const&) const;

  std::vector<nigiri::routing::offset> get_offsets(
      osr::location const&,
      osr::direction,
      std::vector<api::ModeEnum> const&,
      std::optional<std::vector<api::RentalFormFactorEnum>> const&,
      std::optional<std::vector<api::RentalPropulsionTypeEnum>> const&,
      std::optional<std::vector<std::string>> const& rental_providers,
      bool wheelchair,
      std::chrono::seconds max,
      unsigned max_matching_distance,
      gbfs::gbfs_routing_data&) const;

  nigiri::hash_map<nigiri::location_idx_t,
                   std::vector<nigiri::routing::td_offset>>
  get_td_offsets(elevators const&,
                 osr::location const&,
                 osr::direction,
                 std::vector<api::ModeEnum> const&,
                 bool wheelchair,
                 std::chrono::seconds max) const;

  std::pair<std::vector<api::Itinerary>, nigiri::duration_t> route_direct(
      elevators const*,
      gbfs::gbfs_routing_data&,
      api::Place const& from,
      api::Place const& to,
      std::vector<api::ModeEnum> const&,
      std::optional<std::vector<api::RentalFormFactorEnum>> const&,
      std::optional<std::vector<api::RentalPropulsionTypeEnum>> const&,
      std::optional<std::vector<std::string>> const& rental_providers,
      nigiri::unixtime_t start_time,
      bool wheelchair,
      std::chrono::seconds max) const;

  osr::ways const* w_;
  osr::lookup const* l_;
  osr::platforms const* pl_;
  nigiri::timetable const* tt_;
  tag_lookup const* tags_;
  point_rtree<nigiri::location_idx_t> const* loc_tree_;
  platform_matches_t const* matches_;
  std::shared_ptr<rt> const& rt_;
  nigiri::shapes_storage const* shapes_;
  std::shared_ptr<gbfs::gbfs_data> const& gbfs_;
};

}  // namespace motis::ep
