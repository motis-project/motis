#pragma once

#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

#include "utl/to_vec.h"

#include "motis/core/schedule/schedule.h"
#include "motis/routing/lower_bounds.h"
#include "motis/routing/mem_manager.h"
#include "motis/routing/start_label_generators/base_gen.h"
#include "motis/routing/start_label_generators/gen_util.h"

namespace motis::routing {

template <search_dir Dir, typename Label>
struct pretrip_gen {
  static std::vector<Label*> generate(schedule const& sched, mem_manager& mem,
                                      lower_bounds& lbs, edge const* start_edge,
                                      std::vector<edge> const& meta_edges,
                                      std::vector<edge> const& query_edges,
                                      time interval_begin, time interval_end,
                                      light_connection const*,
                                      bool starting_footpaths) {
    std::vector<Label*> labels;
    auto const start = sched.station_nodes_.at(0).get();
    if ((start_edge->to_ == start && Dir == search_dir::FWD) ||
        (start_edge->from_ == start && Dir == search_dir::BWD)) {
      generate_intermodal_starts(sched, mem, lbs, start_edge, query_edges,
                                 interval_begin, interval_end,
                                 starting_footpaths, labels);
    } else {
      generate_meta_starts(sched, mem, lbs, meta_edges, interval_begin,
                           interval_end, starting_footpaths, labels);
    }
    return labels;
  }

  static void generate_intermodal_starts(schedule const& sched,
                                         mem_manager& mem, lower_bounds& lbs,
                                         edge const* start_edge,
                                         std::vector<edge> const& query_edges,
                                         time interval_begin, time interval_end,
                                         bool starting_footpaths,
                                         std::vector<Label*>& labels) {
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

      auto const td = qe.type() == edge::TIME_DEPENDENT_MUMO_EDGE;
      auto const edge_interval_begin =
          td ? std::max(qe.m_.foot_edge_.interval_begin_, interval_begin)
             : interval_begin;
      auto const edge_interval_end =
          td ? std::min(qe.m_.foot_edge_.interval_end_, interval_end)
             : interval_end;

      generate_labels_at_route_nodes(sched, mem, lbs, path, edge_interval_begin,
                                     edge_interval_end, starting_footpaths,
                                     true, labels);
    }
  }

  static void generate_meta_starts(schedule const& sched, mem_manager& mem,
                                   lower_bounds& lbs,
                                   std::vector<edge> const& meta_edges,
                                   time interval_begin, time interval_end,
                                   bool starting_footpaths,
                                   std::vector<Label*>& labels) {
    for (auto const& me : meta_edges) {
      generate_labels_at_route_nodes(sched, mem, lbs, {{&me, 0}},
                                     interval_begin, interval_end,
                                     starting_footpaths, false, labels);
    }
  }

  static void generate_labels_at_route_nodes(
      schedule const& sched, mem_manager& mem, lower_bounds& lbs,
      const std::vector<std::pair<edge const*, int>>& initial_path,
      time interval_begin, time interval_end, bool starting_footpaths,
      bool add_first_interchange_time, std::vector<Label*>& labels) {
    base_gen<Dir, Label>::generate_labels_at_route_nodes(
        sched, initial_path, starting_footpaths, add_first_interchange_time,
        [&](std::vector<std::pair<edge const*, int>> const& path,
            edge const& re, duration initial_walk) {
          return generate_start_labels(path, re, mem, lbs, interval_begin,
                                       interval_end, initial_walk, labels);
        });
  }

  static void generate_start_labels(
      std::vector<std::pair<edge const*, int>> const& path, edge const& re,
      mem_manager& mem, lower_bounds& lbs, time interval_begin,
      time interval_end, duration initial_walk, std::vector<Label*>& labels) {
    assert(!path.empty());

    auto const departure_begin = static_cast<time>(
        Dir == search_dir::FWD ? interval_begin + initial_walk
                               : interval_begin - initial_walk);
    auto const departure_end =
        static_cast<time>(Dir == search_dir::FWD ? interval_end + initial_walk
                                                 : interval_end - initial_walk);

    create_labels<Dir>(departure_begin, departure_end, re, [&](time t) {
      auto const start = static_cast<time>(
          Dir == search_dir::FWD ? t - initial_walk : t + initial_walk);
      Label* l = nullptr;
      for (auto const& [e, additional_time_cost] : path) {
        if (l == nullptr) {
          assert(additional_time_cost == 0);
          l = mem.create<Label>(e, nullptr, start, lbs);
        } else {
          auto new_label = mem.create<Label>();
          if (l->create_label(*new_label, *e, lbs, false,
                              additional_time_cost) !=
              create_label_result::CREATED) {
            return;
          }
          l = new_label;
        }
      }
      labels.push_back(l);
    });

    generate_ontrip_label(mem, lbs, path, interval_begin, interval_end, labels);
  }

  static void generate_ontrip_label(
      mem_manager& mem, lower_bounds& lbs,
      std::vector<std::pair<edge const*, int>> const& path, time interval_begin,
      time interval_end, std::vector<Label*>& labels) {
    auto const start =
        Dir == search_dir::FWD ? interval_end + 1 : interval_begin - 1;
    Label* l = nullptr;
    for (auto const& [e, additional_time_cost] : path) {
      if (l == nullptr) {
        assert(additional_time_cost == 0);
        l = mem.create<Label>(e, nullptr, start, lbs);
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
