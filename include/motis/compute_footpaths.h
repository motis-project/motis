#pragma once

#include "cista/memory_holder.h"

#include "osr/routing/profile.h"
#include "osr/types.h"

#include "motis/fwd.h"
#include "motis/match_platforms.h"
#include "motis/types.h"

namespace motis {

using elevator_footpath_map_t = hash_map<
    osr::node_idx_t,
    hash_set<std::pair<nigiri::location_idx_t, nigiri::location_idx_t>>>;

struct routed_transfers_settings {
  osr::search_profile profile_;
  nigiri::profile_idx_t profile_idx_;
  double max_matching_distance_;
  bool extend_missing_{false};
  std::chrono::seconds max_duration_;
  std::function<bool(nigiri::location_idx_t)> is_candidate_{};

  // Additionally rebuild the default profile from this profile's routed
  // layer: the beeline fill-in the loader wrote is only there because nigiri
  // has no street routing. The transfers.txt rules stay authoritative over
  // the routed durations, and the default profile's hubs are rebuilt around
  // the new walks.
  bool rebuild_default_profile_{false};
};

elevator_footpath_map_t compute_footpaths(
    osr::ways const&,
    osr::lookup const&,
    osr::platforms const&,
    nigiri::timetable&,
    platform_matches_t const&,
    way_matches_storage const*,
    osr::elevation_storage const*,
    std::vector<routed_transfers_settings> const& settings);

}  // namespace motis