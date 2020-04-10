#pragma once

#include <cstdint>

#include "motis/core/statistics/statistics.h"

namespace motis::csa {

struct csa_statistics {
  uint64_t start_count_{};
  uint64_t destination_count_{};
  uint64_t connections_scanned_{};
  uint64_t footpaths_expanded_{};
  uint64_t reconstruction_count_{};
  uint64_t reachable_via_station_{};
  uint64_t reachable_via_trip_{};
  uint64_t trip_reachable_updates_{};
  uint64_t labels_created_{};
  uint64_t existing_labels_dominated_{};
  uint64_t new_labels_dominated_{};
  uint64_t max_labels_per_station_{};
  uint64_t trip_price_init_{};
  uint64_t price_bounds_updated_{};
  uint64_t price_bounds_filtered_{};
  uint64_t search_duration_{};
  uint64_t reconstruction_duration_{};
  uint64_t total_duration_{};
};

inline stats_category to_stats_category(char const* name,
                                        csa_statistics const& s) {
  return {name,
          {{"start_count", s.start_count_},
           {"destination_count", s.destination_count_},
           {"connections_scanned", s.connections_scanned_},
           {"footpaths_expanded", s.footpaths_expanded_},
           {"reconstruction_count", s.reconstruction_count_},
           {"reachable_via_station", s.reachable_via_station_},
           {"reachable_via_trip", s.reachable_via_trip_},
           {"trip_reachable_updates", s.trip_reachable_updates_},
           {"labels_created", s.labels_created_},
           {"existing_labels_dominated", s.existing_labels_dominated_},
           {"new_labels_dominated", s.new_labels_dominated_},
           {"max_labels_per_station", s.max_labels_per_station_},
           {"trip_price_init", s.trip_price_init_},
           {"price_bounds_updated", s.price_bounds_updated_},
           {"price_bounds_filtered", s.price_bounds_filtered_},
           {"search_duration", s.search_duration_},
           {"reconstruction_duration", s.reconstruction_duration_},
           {"total_duration", s.total_duration_}}};
}

}  // namespace motis::csa
