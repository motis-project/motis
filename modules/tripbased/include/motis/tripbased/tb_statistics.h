#pragma once

#include <cstdint>
#include <array>

#include "motis/core/statistics/statistics.h"

namespace motis::tripbased {

struct tb_statistics {
  uint64_t start_count_{};
  uint64_t destination_count_{};
  uint64_t trip_segments_scanned_{};
  uint64_t lines_reaching_destination_{};
  uint64_t destination_arrivals_scanned_{};
  uint64_t destination_reached_{};
  uint64_t results_added_{};
  uint64_t queue_count_{};
  uint64_t transfers_scanned_{};
  uint64_t reconstruction_count_{};
  uint64_t search_duration_{};
  std::array<uint64_t, 8> queue_size_{};
  uint64_t max_queue_size_{};
  uint64_t search_iterations_{};
  uint64_t max_travel_time_reached_{};
  uint64_t pruned_by_earliest_arrival_{};
  uint64_t all_destinations_reached_{};
  uint64_t total_earliest_arrival_updates_{};
  uint64_t lower_bounds_duration_;
};

inline stats_category to_stats_category(char const* name,
                                        tb_statistics const& s) {
  return {
      name,
      {{"start_count", s.start_count_},
       {"destination_count", s.destination_count_},
       {"trip_segments_scanned", s.trip_segments_scanned_},
       {"lines_reaching_destination", s.lines_reaching_destination_},
       {"destination_arrivals_scanned", s.destination_arrivals_scanned_},
       {"destination_reached", s.destination_reached_},
       {"results_added", s.results_added_},
       {"queue_count", s.queue_count_},
       {"transfers_scanned", s.transfers_scanned_},
       {"reconstruction_count", s.reconstruction_count_},
       {"search_duration", s.search_duration_},
       {"max_queue_size", s.max_queue_size_},
       {"queue_0_size", s.queue_size_[0]},
       {"queue_1_size", s.queue_size_[1]},
       {"queue_2_size", s.queue_size_[2]},
       {"queue_3_size", s.queue_size_[3]},
       {"queue_4_size", s.queue_size_[4]},
       {"queue_5_size", s.queue_size_[5]},
       {"queue_6_size", s.queue_size_[6]},
       {"queue_7_size", s.queue_size_[7]},
       {"search_iterations", s.search_iterations_},
       {"max_travel_time_reached", s.max_travel_time_reached_},
       {"pruned_by_earliest_arrival", s.pruned_by_earliest_arrival_},
       {"all_destinations_reached", s.all_destinations_reached_},
       {"total_earliest_arrival_updates", s.total_earliest_arrival_updates_},
       {"lower_bounds_duration", s.lower_bounds_duration_}}};
}

}  // namespace motis::tripbased
