#pragma once

#include <chrono>
#include <optional>
#include <utility>
#include <vector>

#include "boost/thread/tss.hpp"

#include "osr/types.h"

#include "nigiri/routing/clasz_mask.h"
#include "nigiri/routing/raptor/raptor_state.h"
#include "nigiri/routing/raptor_search.h"

#include "motis-api/motis-api.h"
#include "motis/elevators/elevators.h"
#include "motis/fwd.h"
#include "motis/match_platforms.h"
#include "motis/place.h"

namespace motis::ep {

constexpr auto const kInfinityDuration =
    nigiri::duration_t{std::numeric_limits<nigiri::duration_t::rep>::max()};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
extern boost::thread_specific_ptr<nigiri::routing::search_state> search_state;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
extern boost::thread_specific_ptr<nigiri::routing::raptor_state> raptor_state;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
extern boost::thread_specific_ptr<osr::bitvec<osr::node_idx_t>> blocked;

using stats_map_t = std::map<std::string, std::uint64_t>;

place_t get_place(nigiri::timetable const*,
                  tag_lookup const*,
                  std::string_view user_input);

bool is_intermodal(place_t const&);

nigiri::routing::location_match_mode get_match_mode(place_t const&);

std::vector<nigiri::routing::offset> station_start(nigiri::location_idx_t);

std::vector<nigiri::routing::via_stop> get_via_stops(
    nigiri::timetable const&,
    tag_lookup const&,
    std::optional<std::vector<std::string>> const& vias,
    std::vector<std::int64_t> const& times);

std::vector<api::ModeEnum> deduplicate(std::vector<api::ModeEnum>);

void remove_slower_than_fastest_direct(nigiri::routing::query&);

struct routing {
  api::plan_response operator()(boost::urls::url_view const&) const;

  std::vector<nigiri::routing::offset> get_offsets(
      nigiri::rt_timetable const*,
      place_t const&,
      osr::direction,
      std::vector<api::ModeEnum> const&,
      std::optional<std::vector<api::RentalFormFactorEnum>> const&,
      std::optional<std::vector<api::RentalPropulsionTypeEnum>> const&,
      std::optional<std::vector<std::string>> const& rental_providers,
      bool ignore_rental_return_constraints,
      api::PedestrianProfileEnum,
      api::ElevationCostsEnum,
      std::chrono::seconds max,
      double max_matching_distance,
      gbfs::gbfs_routing_data&) const;

  nigiri::hash_map<nigiri::location_idx_t,
                   std::vector<nigiri::routing::td_offset>>
  get_td_offsets(nigiri::rt_timetable const* rtt,
                 elevators const*,
                 place_t const&,
                 osr::direction,
                 std::vector<api::ModeEnum> const&,
                 api::PedestrianProfileEnum,
                 api::ElevationCostsEnum,
                 double max_matching_distance,
                 std::chrono::seconds max,
                 nigiri::routing::start_time_t const&) const;

  std::pair<std::vector<api::Itinerary>, nigiri::duration_t> route_direct(
      elevators const*,
      gbfs::gbfs_routing_data&,
      api::Place const& from,
      api::Place const& to,
      std::vector<api::ModeEnum> const&,
      std::optional<std::vector<api::RentalFormFactorEnum>> const&,
      std::optional<std::vector<api::RentalPropulsionTypeEnum>> const&,
      std::optional<std::vector<std::string>> const& rental_providers,
      bool ignore_rental_return_constraints,
      nigiri::unixtime_t start_time,
      api::PedestrianProfileEnum,
      api::ElevationCostsEnum,
      std::chrono::seconds max,
      double max_matching_distance,
      double fastest_direct_factor,
      unsigned api_version) const;

  config const& config_;
  osr::ways const* w_;
  osr::lookup const* l_;
  osr::platforms const* pl_;
  osr::elevation_storage const* elevations_;
  nigiri::timetable const* tt_;
  tag_lookup const* tags_;
  point_rtree<nigiri::location_idx_t> const* loc_tree_;
  flex::flex_areas const* fa_;
  platform_matches_t const* matches_;
  std::shared_ptr<rt> const& rt_;
  nigiri::shapes_storage const* shapes_;
  std::shared_ptr<gbfs::gbfs_data> const& gbfs_;
  odm::bounds const* odm_bounds_;
  metrics_registry* metrics_;
};

}  // namespace motis::ep
