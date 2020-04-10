#pragma once

#include <vector>

#include "motis/vector.h"

#include "motis/loader/graph_builder.h"

namespace motis::loader {

struct rule_route {
  std::map<Service const*, bitfield> traffic_days_;
  std::vector<Rule const*> rules_;
  unsigned first_day_{}, last_day_{};
};

struct rule_service_graph_builder {
  explicit rule_service_graph_builder(graph_builder&);

  void add_rule_services(mcd::vector<rule_route> const& rule_services);

  graph_builder& gb_;
};

}  // namespace motis::loader
