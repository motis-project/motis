#include "graph_builder_test.h"

#include <iostream>

#include "motis/loader/loader.h"

#include "../hrd/paths.h"

namespace motis::loader {

loader_graph_builder_test::loader_graph_builder_test(std::string schedule_name,
                                                     std::string schedule_begin,
                                                     int num_days)
    : schedule_name_(std::move(schedule_name)),
      schedule_begin_(std::move(schedule_begin)),
      num_days_(num_days) {}

void loader_graph_builder_test::SetUp() {
  sched_ =
      load_schedule(loader_options{{(hrd::SCHEDULES / schedule_name_).string()},
                                   schedule_begin_,
                                   num_days_});
}

edge const* loader_graph_builder_test::get_route_edge(node const* route_node) {
  auto it =
      std::find_if(begin(route_node->edges_), end(route_node->edges_),
                   [](edge const& e) { return e.type() == edge::ROUTE_EDGE; });
  if (it == end(route_node->edges_)) {
    return nullptr;
  } else {
    return &(*it);
  }
}

std::vector<std::tuple<light_connection const*, node const*, node const*>>
loader_graph_builder_test::get_connections(node const* first_route_node,
                                           time departure_time) {
  decltype(get_connections(first_route_node, departure_time)) cons;
  edge const* route_edge = nullptr;
  node const* route_node = first_route_node;
  while ((route_edge = get_route_edge(route_node)) != nullptr) {
    auto const* con = route_edge->get_connection(departure_time);
    if (con != nullptr) {
      cons.emplace_back(con, route_node, route_edge->to_);
      route_node = route_edge->to_;
      departure_time = std::get<0>(cons.back())->a_time_;
    } else {
      break;
    }
  }
  return cons;
}

}  // namespace motis::loader
