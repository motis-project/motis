#pragma once

#include <utility>
#include <vector>

#include "geo/latlng.h"

#include "motis/footpaths/platforms.h"
#include "motis/footpaths/state.h"

#include "nigiri/location.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace motis::footpaths {

struct matching_data {
  nigiri::timetable::locations const& locations_;

  state const& old_state_;
  state const& update_state_;
};

struct matching_options {
  double max_matching_dist_;
  double max_bus_stop_matching_dist_;
};

struct matching_result {
  platform pf_;
  nigiri::location_idx_t nloc_idx_;
  geo::latlng nloc_pos_;
};
using matching_results = std::vector<matching_result>;

matching_results match_locations_and_platforms(matching_data const&,
                                               matching_options const&);

// -- match functions --
std::pair<bool, matching_result> match_by_distance(
    nigiri::location const& /*nloc*/, state const& /* old_state */,
    state const& /* update_state */, matching_options const&);

}  // namespace motis::footpaths
