#pragma once

#include <vector>

#include "motis/vector.h"

#include "motis/loader/graph_builder.h"

namespace motis::loader {

struct rule_service_graph_builder {
  explicit rule_service_graph_builder(graph_builder&);

  void add_rule_services(flatbuffers64::Vector<flatbuffers64::Offset<
                             motis::loader::RuleService>> const* rule_services);

  graph_builder& gb_;
};

}  // namespace motis::loader