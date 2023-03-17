#pragma once

#include <functional>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/core/schedule/schedule.h"
#include "motis/routing/lower_bounds.h"
#include "motis/routing/mem_manager.h"
#include "motis/routing/start_label_generators/gen_util.h"

namespace motis::routing {

template <search_dir Dir, typename Label>
struct base_gen {
  static inline bool edge_can_be_used(edge const& e) {
    if (Dir == search_dir::FWD) {
      return e.type() != edge::INVALID_EDGE && e.type() != edge::BWD_EDGE &&
             e.type() != edge::AFTER_TRAIN_BWD_EDGE;
    } else {
      return e.type() != edge::INVALID_EDGE && e.type() != edge::FWD_EDGE &&
             e.type() != edge::AFTER_TRAIN_FWD_EDGE;
    }
  }

  static inline bool has_no_time_cost(edge const& e) {
    if (Dir == search_dir::FWD) {
      return e.type() == edge::ENTER_EDGE;
    } else {
      return e.type() == edge::EXIT_EDGE;
    }
  }

  static inline duration get_edge_duration(edge const& e) {
    utl::verify(e.type() != edge::ROUTE_EDGE && e.type() != edge::HOTEL_EDGE,
                "start label generator: invalid edge type");
    return has_no_time_cost(e) ? static_cast<duration>(0U)
                               : e.m_.foot_edge_.time_cost_;
  }

  static duration get_duration(
      std::vector<std::pair<edge const*, int>> const& edges) {
    return static_cast<duration>(std::accumulate(
        begin(edges), end(edges), 0,
        [](auto const sum, std::pair<edge const*, int> const& e) {
          return sum + get_edge_duration(*e.first) + e.second;
        }));
  }

  struct path {
    explicit path(std::vector<std::pair<edge const*, int>> edges,
                  bool add_interchange_time)
        : edges_{std::move(edges)},
          add_interchange_time_(add_interchange_time) {
      assert(!edges_.empty());
      dist_ = get_duration(edges_);
      to_ = edges_.back().first->template get_destination<Dir>();
    }

    path(path const& prev, edge const* new_fe, edge const* new_e,
         duration const dist, node const* to)
        : dist_{prev.dist_},
          edges_{prev.edges_},
          to_{to},
          add_interchange_time_(false) {
      edges_.emplace_back(new_e, 0);
      edges_.emplace_back(new_fe, 0);
      dist_ += dist;
    }

    duration dist_;
    std::vector<std::pair<edge const*, int>> edges_;
    node const* to_;
    bool add_interchange_time_;
  };

  struct get_bucket {
    std::size_t operator()(path const& p) const { return p.dist_; }
  };

  static void generate_labels_at_route_nodes(
      schedule const& sched,
      std::vector<std::pair<edge const*, int>> initial_path,
      bool starting_footpaths, bool add_first_interchange_time,
      std::function<void(std::vector<std::pair<edge const*, int>> const&,
                         edge const&, duration)> const& generate_start_labels) {
    constexpr auto const MAX_FOOT_PATH_LENGTH = MINUTES_A_DAY;

    dial<path, MAX_FOOT_PATH_LENGTH, get_bucket> pq;
    pq.push(path(initial_path, add_first_interchange_time));
    std::set<uint32_t> visited;

    auto const expand_foot_edges = [&](node const& foot_node, path const& prev,
                                       edge const& last) {
      for_each_edge<Dir>(&foot_node, [&](auto&& fe) {
        auto const& dest = fe.template get_destination<Dir>();
        if (dest->is_station_node() &&
            fe.type() ==
                (Dir == search_dir::FWD ? edge::FWD_EDGE : edge::BWD_EDGE)) {
          auto const dist = fe.m_.foot_edge_.time_cost_;
          utl::verify(prev.dist_ + dist < MAX_FOOT_PATH_LENGTH,
                      "max foot path length exceeded");
          pq.push(path(prev, &fe, &last, dist, dest));
        }
      });
    };

    while (!pq.empty()) {
      auto p = pq.top();
      pq.pop();

      const auto station = p.to_->as_station_node();
      utl::verify(station != nullptr,
                  "start label generator: not a station node");

      if (visited.find(station->id_) != end(visited)) {
        continue;
      };
      visited.emplace(station->id_);

      for_each_edge<Dir>(station, [&](auto&& e) {
        if (!edge_can_be_used(e)) {
          return;
        }

        auto const node = e.get_destination(Dir);

        if (node->is_foot_node() && starting_footpaths) {
          expand_foot_edges(*node, p, e);
        } else if (node->is_route_node()) {
          auto const additional_cost =
              p.add_interchange_time_
                  ? sched.stations_[p.to_->id_]->transfer_time_
                  : 0;
          auto new_path = p.edges_;
          new_path.emplace_back(&e, additional_cost);
          for_each_edge<Dir>(node, [&](auto&& re) {
            if (re.type() != edge::ROUTE_EDGE) {
              return;
            }
            generate_start_labels(new_path, re, get_duration(new_path));
          });
        }
      });
    }
  }
};

}  // namespace motis::routing
