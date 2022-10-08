#pragma once

#include "motis/core/statistics/statistics.h"

namespace motis::intermodal {

struct statistics {
  uint64_t start_edges_{};
  uint64_t destination_edges_{};
  uint64_t linear_distance_{};
  uint64_t dominated_by_direct_connection_{};
  uint64_t mumo_edge_duration_{};
  uint64_t routing_duration_{};
  uint64_t direct_connection_duration_{};
  uint64_t revise_duration_{};
  uint64_t ondemand_server_area_inquery_{};
  uint64_t ondemand_server_product_inquery_{};
  uint64_t ondemand_remove_duration_{};
  uint64_t ondemand_removed_journeys_{};
  uint64_t ondemand_check_availability_{};
  uint64_t ondemand_journey_count_{};
  uint64_t journey_count_begin_{};
  uint64_t journey_count_end_{};
};

inline stats_category to_stats_category(char const* name, statistics const& s) {
  return {
      name,
      {{"start_edges", s.start_edges_},
       {"destination_edges", s.destination_edges_},
       {"linear_distance", s.linear_distance_},
       {"dominated_by_direct_connection", s.dominated_by_direct_connection_},
       {"mumo_edge_duration", s.mumo_edge_duration_},
       {"routing_duration", s.routing_duration_},
       {"direct_connection_duration", s.direct_connection_duration_},
       {"revise_duration", s.revise_duration_},
       {"ondemand_server_area_inquery", s.ondemand_server_area_inquery_},
       {"ondemand_server_product_inquery", s.ondemand_server_product_inquery_},
       {"ondemand_remove_duration", s.ondemand_remove_duration_},
       {"ondemand_removed_journeys", s.ondemand_removed_journeys_},
       {"ondemand_check_availability", s.ondemand_check_availability_},
       {"ondemand_jounrey_count", s.ondemand_journey_count_},
       {"journey_count_begin", s.journey_count_begin_},
       {"journey_count_end", s.journey_count_end_}}};
}

}  // namespace motis::intermodal
