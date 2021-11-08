#pragma once

#include "utl/concat.h"
#include "utl/to_vec.h"

#include "motis/raptor/raptor_timetable.h"
#include "motis/raptor/raptor_util.h"

namespace motis::raptor {

struct additional_start {
  additional_start() = delete;
  additional_start(stop_id const s_id, time const offset)
      : s_id_(s_id), offset_(offset) {}
  stop_id s_id_{invalid<stop_id>};
  time offset_{invalid<time>};
};

auto inline get_add_starts(raptor_schedule const& raptor_sched,
                           stop_id const source, bool const use_start_footpaths,
                           bool const use_start_metas) {
  std::vector<additional_start> add_starts;

  if (use_start_footpaths) {
    auto const& init_footpaths = raptor_sched.initialization_footpaths_[source];
    utl::concat(add_starts, utl::to_vec(init_footpaths, [](auto const f) {
                  return additional_start(f.to_, f.duration_);
                }));
  }

  if (use_start_metas) {
    auto const& equis = raptor_sched.equivalent_stations_[source];
    utl::concat(add_starts, utl::to_vec(equis, [](auto const s_id) {
                  return additional_start(s_id, 0);
                }));

    // Footpaths from meta stations
    if (use_start_footpaths) {
      for (auto const equi : equis) {
        auto const& init_footpaths =
            raptor_sched.initialization_footpaths_[equi];
        utl::concat(add_starts, utl::to_vec(init_footpaths, [](auto const f) {
                      return additional_start(f.to_, f.duration_);
                    }));
      }
    }
  }

  return add_starts;
}

// returns the maximum amount of additional starts for a raptor query
// which is from a query using use_source_metas and use_start_footpaths
inline size_t get_max_add_starts(raptor_schedule const& sched) {
  size_t max_add_starts = 0;

  for (auto s_id = 0; s_id < sched.departure_events_.size(); ++s_id) {
    size_t add_starts = 0;
    auto const& equis = sched.equivalent_stations_[s_id];

    add_starts += equis.size();

    for (auto const& equi : equis) {
      auto const& init_footpaths = sched.initialization_footpaths_[equi];
      add_starts += init_footpaths.size();
    }

    max_add_starts = std::max(max_add_starts, add_starts);
  }

  return max_add_starts;
}

}  // namespace motis::raptor
