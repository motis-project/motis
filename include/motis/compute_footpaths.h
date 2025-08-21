#pragma once

#include "cista/memory_holder.h"

#include "osr/routing/profile.h"
#include "osr/types.h"

#include "motis/fwd.h"
#include "motis/types.h"

namespace motis {

using elevator_footpath_map_t = hash_map<
    osr::node_idx_t,
    hash_set<std::pair<nigiri::location_idx_t, nigiri::location_idx_t>>>;

struct transfer_routing_options {
  osr::search_profile profile_;
  double max_matching_distance_;
  bool extend_missing_;
  std::chrono::seconds max_duration_;
};

nigiri::profile_idx_t get_profile_idx(osr::search_profile);

using transfer_routing_profiles_t =
    std::array<std::optional<transfer_routing_options>, nigiri::kNProfiles>;

elevator_footpath_map_t compute_footpaths(
    osr::ways const&,
    osr::lookup const&,
    osr::platforms const&,
    nigiri::timetable&,
    osr::elevation_storage const*,
    bool update_coordinates,
    transfer_routing_profiles_t const& settings);

}  // namespace motis