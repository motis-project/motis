#include "motis/raptor/additional_start.h"

#include "utl/concat.h"
#include "utl/to_vec.h"

namespace motis::raptor {

std::vector<additional_start> get_add_starts(raptor_meta_info const& meta_info,
                                             stop_id const source,
                                             bool const use_start_footpaths,
                                             bool const use_start_metas) {
  std::vector<additional_start> add_starts;

  if (use_start_footpaths) {
    auto const& init_footpaths = meta_info.initialization_footpaths_[source];
    utl::concat(add_starts, utl::to_vec(init_footpaths, [](auto const f) {
                  return additional_start{f.to_, f.duration_};
                }));
  }

  if (use_start_metas) {
    auto const& equis = meta_info.equivalent_stations_[source];
    utl::concat(add_starts, utl::to_vec(equis, [](auto const s_id) {
                  return additional_start{s_id, 0};
                }));

    // Footpaths from meta stations
    if (use_start_footpaths) {
      for (auto const equi : equis) {
        auto const& init_footpaths = meta_info.initialization_footpaths_[equi];
        utl::concat(add_starts, utl::to_vec(init_footpaths, [](auto const f) {
                      return additional_start{f.to_, f.duration_};
                    }));
      }
    }
  }

  return add_starts;
}

size_t get_max_add_starts(raptor_meta_info const& meta_info) {
  size_t max_add_starts = 0;

  for (auto s_id = 0; s_id < meta_info.departure_events_.size(); ++s_id) {
    size_t add_starts = 0;
    auto const& equis = meta_info.equivalent_stations_[s_id];

    add_starts += equis.size();

    for (auto const& equi : equis) {
      auto const& init_footpaths = meta_info.initialization_footpaths_[equi];
      add_starts += init_footpaths.size();
    }

    max_add_starts = std::max(max_add_starts, add_starts);
  }

  std::cout << "Max Additional Starts: " << +max_add_starts << std::endl;

  return max_add_starts;
}

}  // namespace motis::raptor