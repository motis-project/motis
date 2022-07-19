#pragma once

#include <array>
#include <iostream>
#include <set>

#include "motis/hash_map.h"
#include "motis/hash_set.h"

#include "motis/core/schedule/schedule.h"
#include "motis/core/access/bfs.h"

namespace motis::debug {

struct route_graph {
  route_graph(schedule const& sched, ev_key const k,
              bool const colored_trips = true)
      : sched_{sched}, k_{k}, colored_trips_{colored_trips} {}

  route_graph(schedule const& sched, motis::trip const* trp,
              bool const colored_trips = true)
      : sched_{sched},
        k_{trp->edges_->at(0).get_edge(), trp->lcon_idx_, event_type::DEP},
        colored_trips_{colored_trips} {}

  friend std::ostream& operator<<(std::ostream& out, route_graph const& rg) {
    auto const join = [&](auto const& collection, auto const& fn,
                          auto const& sep) {
      auto first = true;
      for (auto const& elm : collection) {
        if (first) {
          first = false;
        } else {
          out << sep;
        }
        fn(elm);
      }
    };

    auto const& sched = rg.sched_;
    auto const colored_trips = rg.colored_trips_;
    auto const edges = route_bfs(rg.k_, bfs_direction::BOTH, true);
    if (edges.empty()) {
      return out << "// empty route\n";
    }
    auto const route_id = edges.begin()->route_node_->route_;

    mcd::hash_set<node const*> nodes;
    for (auto const& e : edges) {
      nodes.insert(e->get_source());
      nodes.insert(e->get_destination());
    }

    out << "digraph \"route_" << route_id << "\" {\n"
        << "  rankdir=\"LR\";\n"
        << "  node [shape=box];\n";

    for (auto const* n : nodes) {
      out << "  " << n->id_ << " [label=\"" << n->id_ << "\\n"
          << sched.stations_.at(n->get_station()->id_)->name_ << "\"]"
          << "; // route=" << n->route_ << ", in=" << n->incoming_edges_.size()
          << ", out=" << n->edges_.size() << "\n";
    }
    out << "\n";

    auto const trip_color_map = rg.get_trip_colors(edges);

    for (auto const& e : edges) {
      auto const is_route_edge = e->type() == edge::ROUTE_EDGE;
      out << "  " << e->get_source()->id_ << " -> " << e->get_destination()->id_
          << " [minlen=3, label=<";
      if (is_route_edge) {
        auto const& lcons = e->m_.route_edge_.conns_;
        join(
            lcons,
            [&](auto const& lc) {
              join(
                  *sched.merged_trips_.at(lc.trips_),
                  [&](auto const& t) {
                    if (colored_trips) {
                      out << "<font color=\"" << trip_color_map.at(t) << "\">"
                          << t->trip_idx_ << "</font>";
                    } else {
                      out << t->trip_idx_;
                    }
                  },
                  " ");
            },
            "<br/>");

        out << ">, taillabel=\"";
        join(
            lcons, [&](auto const& lc) { out << format_time(lc.d_time_); },
            "\\n");

        out << "\", headlabel=\"";
        join(
            lcons, [&](auto const& lc) { out << format_time(lc.a_time_); },
            "\\n");
        out << "\"]";
      } else {
        out << "through\"]";
      }

      out << ";\n";
    }

    out << "}\n";
    return out;
  }

private:
  mcd::hash_map<motis::trip const*, std::string> get_trip_colors(
      std::set<motis::trip::route_edge> const& edges) const {
    static auto const colors =
        std::array{"#b91c1c", "#1d4ed8", "#15803d", "#7e22ce", "#b45309",
                   "#be185d", "#0e7490", "#0f766e", "#171717"};

    auto color_map = mcd::hash_map<motis::trip const*, std::string>{};
    if (!colored_trips_) {
      return color_map;
    }

    mcd::hash_set<motis::trip const*> trips;
    for (auto const& e : edges) {
      if (e->type() == edge::ROUTE_EDGE) {
        for (auto const& lcon : e->m_.route_edge_.conns_) {
          auto const& merged_trips = *sched_.merged_trips_.at(lcon.trips_);
          trips.insert(begin(merged_trips), end(merged_trips));
        }
      }
    }

    auto next_color = 0U;
    for (auto const& trp : trips) {
      color_map[trp] = colors.at(next_color);
      if (next_color + 1 < colors.size()) {
        ++next_color;
      }
    }
    return color_map;
  }

  schedule const& sched_;
  ev_key k_;
  bool colored_trips_;
};

}  // namespace motis::debug
