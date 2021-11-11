#pragma once

#include "motis/raptor/raptor_timetable.h"

namespace motis::raptor {

struct additional_start {
  additional_start() = delete;

  stop_id s_id_;
  time offset_;
};

std::vector<additional_start> get_add_starts(raptor_meta_info const& meta_info,
                                             stop_id source,
                                             bool use_start_footpaths,
                                             bool use_start_metas);

// returns the maximum amount of additional starts for a raptor query
// which is from a query using use_source_metas and use_start_footpaths
size_t get_max_add_starts(raptor_meta_info const& meta_info);

}  // namespace motis::raptor
