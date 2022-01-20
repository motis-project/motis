#pragma once

#include <vector>

#include "motis/core/schedule/schedule.h"
#include "motis/routing/lower_bounds.h"
#include "motis/routing/mem_manager.h"
#include "motis/routing/start_label_generators/base_gen.h"
#include "motis/routing/start_label_generators/gen_util.h"

namespace motis::routing {

template <search_dir Dir, typename Label>
struct ontrip_gen {
  static std::vector<Label*> generate(schedule const& sched, mem_manager& mem,
                                      lower_bounds& lbs, edge const* start_edge,
                                      std::vector<edge> const&,
                                      std::vector<edge> const& query_edges,
                                      time interval_begin,
                                      time /* interval_end */,
                                      light_connection const* lcon,
                                      bool starting_footpaths) {
    std::vector<Label*> labels;
    auto const start = sched.station_nodes_.at(0).get();
    if (start_edge->get_destination<Dir>() == start) {
      generate_intermodal_starts(sched, mem, lbs, start_edge, query_edges,
                                 interval_begin, starting_footpaths, labels);
    } else if (start_edge->get_destination<Dir>()->is_route_node()) {
      utl::verify(lcon != nullptr, "ontrip train start missing lcon");
      generate_train_start(sched, mem, lbs, start_edge, interval_begin, lcon,
                           labels);
    } else {
      generate_station_starts(sched, mem, lbs, start_edge, interval_begin,
                              starting_footpaths, lcon, labels);
    }
    return labels;
  }

  static void generate_intermodal_starts(
      schedule const& sched, mem_manager& mem, lower_bounds& lbs,
      edge const* start_edge, std::vector<edge> const& query_edges,
      time start_time, bool starting_footpaths, std::vector<Label*>& labels) {
    auto const start = sched.station_nodes_.at(0).get();
    for (auto const& qe : query_edges) {
      if ((Dir == search_dir::FWD && qe.from_ != start) ||
          (Dir == search_dir::BWD && qe.to_ != start)) {
        continue;
      } else if ((Dir == search_dir::FWD && !qe.to_->is_station_node()) ||
                 (Dir == search_dir::BWD && !qe.from_->is_station_node()) ||
                 (qe.type() != edge::TIME_DEPENDENT_MUMO_EDGE &&
                  qe.type() != edge::MUMO_EDGE)) {
        throw std::runtime_error("unsupported edge type");
      }

      std::vector<std::pair<edge const*, int>> path{{start_edge, 0}, {&qe, 0}};

      generate_labels_at_route_nodes(sched, mem, lbs, path, start_time,
                                     starting_footpaths, true, nullptr, labels);
    }
  }

  static void generate_station_starts(schedule const& sched, mem_manager& mem,
                                      lower_bounds& lbs, edge const* start_edge,
                                      time start_time, bool starting_footpaths,
                                      light_connection const* lcon,
                                      std::vector<Label*>& labels) {
    generate_labels_at_route_nodes(sched, mem, lbs, {{start_edge, 0}},
                                   start_time, starting_footpaths, false, lcon,
                                   labels);
  }

  static void generate_train_start(schedule const&, mem_manager& mem,
                                   lower_bounds& lbs, edge const* start_edge,
                                   time start_time,
                                   light_connection const* lcon,
                                   std::vector<Label*>& labels) {
    generate_start_label(mem, lbs, {{start_edge, 0}}, start_time, lcon, labels);
  }

  static void generate_labels_at_route_nodes(
      schedule const& sched, mem_manager& mem, lower_bounds& lbs,
      std::vector<std::pair<edge const*, int>> const& initial_path,
      time start_time, bool starting_footpaths, bool add_first_interchange_time,
      light_connection const* lcon, std::vector<Label*>& labels) {
    base_gen<Dir, Label>::generate_labels_at_route_nodes(
        sched, initial_path, starting_footpaths, add_first_interchange_time,
        [&](std::vector<std::pair<edge const*, int>> const& path, edge const&,
            duration) {
          return generate_start_label(mem, lbs, path, start_time, lcon, labels);
        });
  }

  static void generate_start_label(
      mem_manager& mem, lower_bounds& lbs,
      std::vector<std::pair<edge const*, int>> const& path, time start_time,
      light_connection const* lcon, std::vector<Label*>& labels) {
    Label* l = nullptr;
    for (auto const& [e, additional_time_cost] : path) {
      if (l == nullptr) {
        assert(additional_time_cost == 0);
        l = mem.create<Label>(e, nullptr, start_time, lbs, lcon);
      } else {
        auto new_label = mem.create<Label>();
        if (l->create_label(*new_label, *e, lbs, false, additional_time_cost) !=
            create_label_result::CREATED) {
          return;
        }
        l = new_label;
      }
    }
    labels.push_back(l);
  }
};

}  // namespace motis::routing
