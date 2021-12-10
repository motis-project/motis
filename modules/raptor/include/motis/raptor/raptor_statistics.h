#pragma once

#include <cinttypes>

#include "motis/core/statistics/statistics.h"

namespace motis::raptor {

struct raptor_statistics {
  uint64_t raptor_time_{0};

  uint64_t total_route_update_time_{0};
  uint64_t total_footpath_update_time_{0};
  uint64_t total_criteria_calc_time_{0};

  uint64_t cpu_routes_scanned_{0};
  uint64_t rec_time_{0};
  uint64_t arrival_allocation_time_{0};
  uint64_t total_calculation_time_{0};
  uint64_t raptor_queries_{0};
};

inline stats_category to_stats_category(char const* name,
                                        raptor_statistics const& s) {
  return {name,
          {{"raptor_time (ms)", s.raptor_time_},

           {"route update time (us)", s.total_route_update_time_},
           {"footpath update time (us)", s.total_footpath_update_time_},
           {"criteria calc time (us)", s.total_criteria_calc_time_},

           {"cpu_routes_scanned", s.cpu_routes_scanned_},
           {"rec_time (us)", s.rec_time_},
           {"arrival_allocation_time", s.arrival_allocation_time_},
           {"total_calculation_time (ms)", s.total_calculation_time_},
           {"raptor_queries", s.raptor_queries_}}};
}

}  // namespace motis::raptor
