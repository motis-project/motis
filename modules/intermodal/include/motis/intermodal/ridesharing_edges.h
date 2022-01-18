#pragma once

#include <vector>

#include "motis/module/module.h"

namespace motis::intermodal {

struct mumo_edge;
struct direct_connection;

struct ridesharing_edges {
  ~ridesharing_edges();
  std::vector<mumo_edge> arrs_;
  std::vector<mumo_edge> deps_;
  std::vector<direct_connection> direct_connections_;
};

}  // namespace motis::intermodal
