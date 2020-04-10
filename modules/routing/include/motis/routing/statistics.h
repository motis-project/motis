#pragma once

#include <ostream>

#include "utl/to_vec.h"

#include "motis/core/statistics/statistics.h"

#include "motis/protocol/RoutingResponse_generated.h"

namespace motis::routing {

struct statistics {
  statistics() = default;
  explicit statistics(uint64_t travel_time_lb)
      : travel_time_lb_{travel_time_lb} {}

  bool max_label_quit_{};
  std::size_t labels_created_{};
  uint64_t labels_popped_{};
  uint64_t labels_dominated_by_results_{};
  uint64_t labels_filtered_{};
  uint64_t labels_dominated_by_former_labels_{};
  uint64_t labels_dominated_by_later_labels_{};
  uint64_t labels_popped_until_first_result_{};
  uint64_t labels_popped_after_last_result_{};
  uint64_t priority_queue_max_size_{};
  uint64_t start_label_count_{};
  uint64_t labels_equals_popped_{};
  uint64_t travel_time_lb_{};
  uint64_t transfers_lb_{};
  uint64_t price_l_b_{};
  uint64_t total_calculation_time_{};
  uint64_t pareto_dijkstra_{};
  uint64_t num_bytes_in_use_{};
  uint64_t labels_to_journey_{};
  uint64_t interval_extensions_{};

  friend flatbuffers::Offset<Statistics> to_fbs(
      flatbuffers::FlatBufferBuilder& fbb, char const* category,
      statistics const& s) {
    std::vector<flatbuffers::Offset<StatisticsEntry>> stats{};

    auto const add_entry = [&](char const* key, auto const val) {
      if (val != 0U) {
        stats.emplace_back(
            CreateStatisticsEntry(fbb, fbb.CreateString(key), val));
      }
    };

    add_entry("labels_created", s.labels_created_);
    add_entry("labels_dominated_by_former_labels",
              s.labels_dominated_by_former_labels_);
    add_entry("labels_dominated_by_later_labels",
              s.labels_dominated_by_later_labels_);
    add_entry("labels_dominated_by_results", s.labels_dominated_by_results_);
    add_entry("labels_equals_popped", s.labels_equals_popped_);
    add_entry("labels_filtered", s.labels_filtered_);
    add_entry("labels_popped_after_last_result",
              s.labels_popped_after_last_result_);
    add_entry("labels_popped", s.labels_popped_);
    add_entry("labels_popped_until_first_result",
              s.labels_popped_until_first_result_);
    add_entry("labels_to_journey", s.labels_to_journey_);
    add_entry("max_label_quit", s.max_label_quit_ ? 1 : 0);
    add_entry("num_bytes_in_use", s.num_bytes_in_use_);
    add_entry("pareto_dijkstra", s.pareto_dijkstra_);
    add_entry("priority_queue_max_size", s.priority_queue_max_size_);
    add_entry("start_label_count", s.start_label_count_);
    add_entry("total_calculation_time", s.total_calculation_time_);
    add_entry("transfers_lb", s.transfers_lb_);
    add_entry("travel_time_lb", s.travel_time_lb_);
    add_entry("interval_extensions", s.interval_extensions_);

    return CreateStatistics(fbb, fbb.CreateString(category),
                            fbb.CreateVectorOfSortedTables(&stats));
  }

  friend stats_category to_stats_category(char const* name,
                                          motis::routing::statistics const& s) {
    return stats_category{
        name,
        {{"labels_created", s.labels_created_},
         {"labels_dominated_by_former_labels",
          s.labels_dominated_by_former_labels_},
         {"labels_dominated_by_later_labels",
          s.labels_dominated_by_later_labels_},
         {"labels_dominated_by_results", s.labels_dominated_by_results_},
         {"labels_equals_popped", s.labels_equals_popped_},
         {"labels_filtered", s.labels_filtered_},
         {"labels_popped_after_last_result",
          s.labels_popped_after_last_result_},
         {"labels_popped", s.labels_popped_},
         {"labels_popped_until_first_result",
          s.labels_popped_until_first_result_},
         {"labels_to_journey", s.labels_to_journey_},
         {"max_label_quit", s.max_label_quit_ ? 1U : 0U},
         {"num_bytes_in_use", s.num_bytes_in_use_},
         {"pareto_dijkstra", s.pareto_dijkstra_},
         {"priority_queue_max_size", s.priority_queue_max_size_},
         {"start_label_count", s.start_label_count_},
         {"total_calculation_time", s.total_calculation_time_},
         {"transfers_lb", s.transfers_lb_},
         {"travel_time_lb", s.travel_time_lb_},
         {"interval_extensions", s.interval_extensions_}}};
  }
};

}  // namespace motis::routing
