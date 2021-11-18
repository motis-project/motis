#include "graph_builder_test.h"

#include <iostream>

#include "motis/core/access/trip_iterator.h"
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

std::vector<
    std::tuple<light_connection const*, day_idx_t, node const*, node const*>>
loader_graph_builder_test::get_connections(node const* first_route_node,
                                           time departure_time) {
  decltype(get_connections(first_route_node, departure_time)) cons;
  edge const* route_edge = nullptr;
  node const* route_node = first_route_node;
  while ((route_edge = get_route_edge(route_node)) != nullptr) {
    auto const [con, day_idx] = route_edge->get_connection(departure_time);
    if (con != nullptr) {
      cons.emplace_back(con, day_idx, route_node, route_edge->to_);
      route_node = route_edge->to_;
      departure_time = std::get<light_connection const*>(cons.back())
                           ->event_time(event_type::ARR, day_idx);
    } else {
      break;
    }
  }
  return cons;
}

void loader_graph_builder_test::print_trip(concrete_trip const trp) {
  auto const& id = trp.trp_->id_;
  std::clog << "trip: ((" << id.primary_.station_id_ << ", "
            << id.primary_.train_nr_ << ", "
            << format_time(trp.get_first_dep_time()) << "), ("
            << id.secondary_.target_station_id_ << ", "
            << format_time(trp.get_last_arr_time()) << ", "
            << id.secondary_.line_id_ << "))" << std::endl;
  std::clog << "  " << trp.trp_->edges_->size()
            << " edges, lcon_idx=" << trp.trp_->lcon_idx_ << std::endl;
  std::clog << "  stops: ";
  for (auto const& stop : motis::access::stops(trp)) {
    std::clog << stop.get_station(*sched_).name_ << " ";
  }
  std::clog << std::endl;
  for (auto const& sec : motis::access::sections(trp)) {
    auto con_info = sec.lcon().full_con_->con_info_;
    std::clog << "  section " << sec.index() << ": "
              << sec.from_station(*sched_).name_ << " "
              << format_time(sec.dep_time()) << " -> "
              << sec.to_station(*sched_).name_ << " "
              << format_time(sec.arr_time())
              << " train_nr=" << con_info->train_nr_;
    con_info = con_info->merged_with_;
    while (con_info != nullptr) {
      std::clog << " merged_with=" << con_info->train_nr_;
      con_info = con_info->merged_with_;
    }
    std::clog << std::endl;
  }
  std::clog << "\n\n";
}

}  // namespace motis::loader
