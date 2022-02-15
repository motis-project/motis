#pragma once

#include <cinttypes>

#include "motis/core/statistics/statistics.h"

namespace motis::raptor {

struct raptor_statistics {
  uint64_t raptor_time_{0};

  uint64_t cpu_time_routes_{0};
  uint64_t cpu_time_footpath_{0};
  uint64_t cpu_time_clear_arrivals_{0};

  uint64_t rec_time_{0};
  uint64_t arrival_allocation_time_{0};
  uint64_t total_calculation_time_{0};
  uint64_t raptor_queries_{0};
  uint64_t raptor_connections_{0};

  uint32_t scanned_trips_1_{0};
  uint32_t scanned_trips_2_{0};
  uint32_t scanned_trips_3_{0};
  uint32_t scanned_trips_4_{0};
  uint32_t scanned_trips_5_{0};
  uint32_t scanned_trips_6_{0};
  uint32_t scanned_trips_7_{0};

  uint64_t number_of_rounds_{0};
};

inline stats_category to_stats_category(char const* name,
                                        raptor_statistics const& s) {
  return {name,
          {{"raptor_time_ms", s.raptor_time_},

           {"cpu_route_time_us", s.cpu_time_routes_},
           {"cpu_footpath_time_us", s.cpu_time_footpath_},
           {"cpu_sweep_arrivals_time_us", s.cpu_time_clear_arrivals_},

           {"rec_time_us", s.rec_time_},
           {"arrival_allocation_time", s.arrival_allocation_time_},
           {"total_calculation_time_ms", s.total_calculation_time_},
           {"raptor_queries", s.raptor_queries_},

           {"raptor_connections", s.raptor_connections_},
           {"number_of_rounds", s.number_of_rounds_},

           {"scanned_trips_r1", s.scanned_trips_1_},
           {"scanned_trips_r2", s.scanned_trips_2_},
           {"scanned_trips_r3", s.scanned_trips_3_},
           {"scanned_trips_r4", s.scanned_trips_4_},
           {"scanned_trips_r5", s.scanned_trips_5_},
           {"scanned_trips_r6", s.scanned_trips_6_},
           {"scanned_trips_r7", s.scanned_trips_7_},
          }};
}

}  // namespace motis::raptor
