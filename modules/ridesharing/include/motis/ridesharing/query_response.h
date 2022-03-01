#pragma once

#include "motis/module/module.h"

#include "motis/ridesharing/leg_idx.h"

#include <string>

#include "geo/latlng.h"

namespace motis::ridesharing {

struct ridesharing_edge {
  std::string from_station_id_;
  geo::latlng from_pos_;
  leg_idx from_leg_{0U};
  int parking_id_{0};
  geo::latlng parking_pos_;
  std::string to_station_id_;
  geo::latlng to_pos_;
  leg_idx to_leg_{0U};
  uint16_t rs_price_{0U};
  long rs_t_;
  uint16_t rs_duration_{0U};
  uint16_t ppr_duration_{0U};
  uint16_t ppr_accessibility_{0U};
  std::string lift_key_;
};

struct query_response {
  query_response(uint64_t rt) : routing_time_(rt) {}
  uint64_t routing_time_;
  uint64_t edges_time_;
  uint64_t close_station_time_;
  std::vector<ridesharing_edge> arrs_;
  std::vector<ridesharing_edge> deps_;
  std::vector<ridesharing_edge> direct_connections_;
};

}  // namespace motis::ridesharing
