#pragma once

#include <utility>
#include <vector>

#include "geo/latlng.h"

#include "motis/footpaths/platform/platform.h"
#include "motis/footpaths/platform/platform_index.h"

#include "nigiri/location.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace motis::footpaths {

struct matching_data {
  ::nigiri::timetable::locations const& locations_to_match_;

  hash_map<nlocation_key_t, platform> const& already_matched_nloc_keys_;

  platform_index const& old_state_pf_idx_;
  platform_index const& update_state_pf_idx_;

  bool has_update_state_pf_idx_;
};

struct matching_options {
  double max_matching_dist_;
  double max_bus_stop_matching_dist_;
};

struct matching_result {
  platform pf_;
  geo::latlng nloc_pos_;
};
using matching_results = std::vector<matching_result>;

matching_results match_locations_and_platforms(matching_data const&,
                                               matching_options const&);

// -- match functions --
std::pair<bool, matching_result> match_by_distance(
    ::nigiri::location const& /*nloc*/, matching_data const&,
    matching_options const&);

}  // namespace motis::footpaths
