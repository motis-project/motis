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
  //uint64_t ondemand_duration_{};
  uint64_t ondemand_server_first_inquery_{};
  uint64_t ondemand_server_second_inquery_{};
  uint64_t ondemand_remove_not_available_{};
  uint64_t ondemand_removed_journeys_{};
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
       //{"ondemand_duration", s.ondemand_duration_},
       {"ondemand_server_first_inquery", s.ondemand_server_first_inquery_},
       {"ondemand_server_second_inquery", s.ondemand_server_second_inquery_},
       {"ondemand_remove_not_available", s.ondemand_remove_not_available_},
       {"ondemand_removed_journeys", s.ondemand_removed_journeys_}}};
}

}  // namespace motis::intermodal
