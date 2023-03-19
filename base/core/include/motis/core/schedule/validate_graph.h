#pragma once

#include <iostream>

#include "utl/verify.h"

#include "motis/core/schedule/schedule.h"
#include "motis/core/access/service_access.h"
#include "motis/core/access/trip_iterator.h"
#include "motis/core/debug/trip.h"

namespace motis {

inline bool incoming_edges_correct_2(schedule const& sched) {
  auto const check_edges = [](node const* n) {
    return std::all_of(
        begin(n->incoming_edges_), end(n->incoming_edges_), [n](edge* const e) {
          auto const& out = e->from_->edges_;
          return e->to_ == n && e >= out.begin() && e < out.end();
        });
  };

  return std::all_of(
      begin(sched.station_nodes_), end(sched.station_nodes_), [&](auto&& sn) {
        return check_edges(sn.get()) &&
               std::all_of(
                   begin(sn->incoming_edges_), end(sn->incoming_edges_),
                   [&](auto const& e) { return check_edges(e->from_); });
      });
}

inline void validate_graph(schedule const& sched) {
  utl::verify(
      [&] {
        for (auto const& sn : sched.station_nodes_) {
          for (auto const& se : sn->edges_) {
            if (se.to_->type() != node_type::ROUTE_NODE) {
              continue;
            }

            for (auto const& re : se.to_->edges_) {
              if (re.empty()) {
                continue;
              }

              auto const& lcons = re.m_.route_edge_.conns_;
              auto const is_sorted_dep = std::is_sorted(
                  begin(lcons), end(lcons),
                  [](light_connection const& a, light_connection const& b) {
                    return a.d_time_ < b.d_time_;
                  });
              auto const is_sorted_arr = std::is_sorted(
                  begin(lcons), end(lcons),
                  [](light_connection const& a, light_connection const& b) {
                    return a.a_time_ < b.a_time_;
                  });
              if (!is_sorted_dep || !is_sorted_arr) {
                return false;
              }
            }
          }
        }
        return true;
      }(),
      "all light connections sorted after rt update");

  utl::verify(
      [&] {
        auto const check_edges = [](node const* n) {
          return std::all_of(
              begin(n->edges_), end(n->edges_), [n](edge const& e) {
                auto const in = e.to_->incoming_edges_;
                return e.from_ == n &&
                       std::find(begin(in), end(in), &e) != end(in);
              });
        };

        return std::all_of(
            begin(sched.station_nodes_), end(sched.station_nodes_),
            [&](mcd::unique_ptr<station_node> const& sn) {
              return check_edges(sn.get()) &&
                     std::all_of(
                         begin(sn->edges_), end(sn->edges_),
                         [&](auto const& e) { return check_edges(e.to_); });
            });
      }(),
      "incoming edges correct 1");

  utl::verify(incoming_edges_correct_2(sched), "incoming edges correct 2");

  auto const check_edges = [](node const* n) {
    return std::all_of(begin(n->edges_), end(n->edges_),
                       [&](edge const& e) { return e.from_ == n; }) &&
           std::all_of(begin(n->incoming_edges_), end(n->incoming_edges_),
                       [&](edge const* e) { return e->to_ == n; });
  };

  utl::verify(
      [&]() {
        return std::all_of(
            begin(sched.station_nodes_), end(sched.station_nodes_),
            [&](auto&& sn) {
              return check_edges(sn.get()) &&
                     std::all_of(begin(sn->child_nodes_), end(sn->child_nodes_),
                                 [&](auto&& n) {
                                   return n->get_station() == sn.get() &&
                                          check_edges(n.get());
                                 }) &&
                     (sn->foot_node_.get() == nullptr ||
                      (sn->foot_node_->get_station() == sn.get() &&
                       check_edges(sn->foot_node_.get())));
            });
      }(),
      "edge from pointer correct");

  utl::verify(
      [&] {
        for (auto const route : sched.expanded_trips_) {
          auto const route_idx = route.index();
          auto route_trip_idx = 0U;
          for (auto const& tp : route) {
            time last_time = 0;
            for (auto const sec : access::sections{tp}) {
              auto const& lc = sec.lcon();
              auto const section_times_ok = lc.d_time_ <= lc.a_time_;
              auto const stop_times_ok = last_time <= lc.d_time_;
              last_time = lc.a_time_;
              if (!section_times_ok || !stop_times_ok) {
                std::cout << "\nvalidate_graph: expanded trip times "
                             "inconsistent (expanded route "
                          << route_idx << ", expanded trip "
                          << route.data_index(route_trip_idx) << ", "
                          << route_trip_idx << "/" << route.size()
                          << " in route):\n"
                          << debug::trip_with_sections{sched, tp} << "\n";
                return false;
              }
            }
            ++route_trip_idx;
          }
        }
        return true;
      }(),
      "expanded trip times consistent");
}

inline void print_graph(schedule const& sched) {
  auto const indent_line = [](size_t const indent) {
    for (auto i = 0U; i != indent; ++i) {
      std::cerr << "  ";
    }
  };

  auto const station_name = [&](node const* n) {
    return sched.stations_.at(n->get_station()->id_)->name_;
  };

  std::cerr << "\n\nGraph:\n";
  auto const print_edge = [&](edge const* e, size_t const indent) {
    indent_line(indent);
    std::cerr << e->type_str() << ": " << station_name(e->from_) << " -> "
              << station_name(e->to_) << "[" << e->to_->id_ << "]\n";
    if (!e->empty()) {
      for (auto const& lcon : e->m_.route_edge_.conns_) {
        indent_line(indent + 1);

        auto con_info = lcon.full_con_->con_info_;
        while (con_info != nullptr) {
          std::cerr << get_service_name(sched, con_info);
          con_info = con_info->merged_with_;
          if (con_info != nullptr) {
            std::cerr << "|";
          }
        }

        std::cerr << ", dep=" << format_time(lcon.d_time_)
                  << ", arr=" << format_time(lcon.a_time_);
        std::cerr << "}\n";
      }
    }
  };

  auto const print_node = [&](node const* n, size_t const indent) {
    indent_line(indent);
    std::cerr << "id=" << n->id_ << ", " << n->type_str() << " at "
              << station_name(n) << ":" << std::endl;

    indent_line(indent + 1);
    std::cerr << n->edges_.size() << " outgoing edges:" << std::endl;
    for (auto const& e : n->edges_) {
      print_edge(&e, indent + 2);
    }

    indent_line(indent + 1);
    std::cerr << n->incoming_edges_.size() << " incoming edges:" << std::endl;
    for (auto const& e : n->incoming_edges_) {
      print_edge(e, indent + 2);
    }
  };

  for (auto const& sn : sched.station_nodes_) {
    print_node(sn.get(), 0);
    for (auto const& e : sn->edges_) {
      print_node(e.to_, 1);
    }
  }
  std::cerr << "\n\n";
}

}  // namespace motis
