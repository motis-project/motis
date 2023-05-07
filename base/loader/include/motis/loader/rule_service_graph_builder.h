#pragma once

#include <vector>

#include "motis/vector.h"

#include "motis/loader/graph_builder.h"

namespace motis::loader {

void print_bitfield(std::ostream&, date::sys_days const first_day,
                    bitfield const&);

struct rule_route {
  void print(std::ostream&, date::sys_days) const;

  std::map<Service const*, bitfield> traffic_days_;
  std::vector<Rule const*> rules_;
};

struct rule_service_graph_builder {
  explicit rule_service_graph_builder(graph_builder&);

  void add_rule_services(mcd::vector<rule_route> const& rule_services);

  graph_builder& gb_;
};

}  // namespace motis::loader
