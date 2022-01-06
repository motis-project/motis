#pragma once

#include <iostream>

#include "utl/verify.h"

#include "motis/core/schedule/schedule.h"
#include "motis/core/access/service_access.h"
#include "motis/core/access/trip_iterator.h"

namespace motis {

inline void validate_graph(schedule const& sched) {
  utl::verify(
      [&] {
        for (auto const& sn : sched.station_nodes_) {
          for (auto const& se : sn->edges_) {
            if (se.to()->type() != node_type::ROUTE_NODE) {
              continue;
            }

            for (auto const& re : se.to()->edges_) {
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
        for (auto const& sn : sched.station_nodes_) {
          for (auto const& se : sn->edges_) {
            if (se.to()->type() != node_type::ROUTE_NODE) {
              continue;
            }

            for (auto const& re : se.to()->edges_) {
              if (re.empty()) {
                continue;
              }

              auto const& lcons = re.m_.route_edge_.conns_;
              if (!std::all_of(begin(lcons), end(lcons),
                               [](light_connection const& lcon) {
                                 return lcon.traffic_days_ != nullptr &&
                                        lcon.traffic_days_->any();
                               })) {
                return false;
              }
            }
          }
        }
        return true;
      }(),
      "lcons with traffic days = nullptr or zero traffic days");

  utl::verify(
      [&] {
        auto const check_edges = [](node const* n) {
          return std::all_of(
              begin(n->edges_), end(n->edges_), [n](edge const& e) {
                auto const in = e.to()->incoming_edges_;
                return e.from() == n &&
                       std::find(begin(in), end(in), &e) != end(in);
              });
        };

        return std::all_of(
            begin(sched.station_nodes_), end(sched.station_nodes_),
            [&](auto&& sn) {
              return check_edges(sn.get()) &&
                     std::all_of(
                         begin(sn->edges_), end(sn->edges_),
                         [&](auto const& e) { return check_edges(e.to()); });
            });
      }(),
      "incoming edges correct 1");

  utl::verify(
      [&] {
        auto const check_edges = [](node const* n) {
          return std::all_of(begin(n->incoming_edges_), end(n->incoming_edges_),
                             [n](edge* const e) {
                               auto const& out = e->from_->edges_;
                               return e->to_ == n && e >= out.begin() &&
                                      e < out.end();
                             });
        };

        return std::all_of(
            begin(sched.station_nodes_), end(sched.station_nodes_),
            [&](auto&& sn) {
              return check_edges(sn.get()) &&
                     std::all_of(
                         begin(sn->incoming_edges_), end(sn->incoming_edges_),
                         [&](auto const& e) { return check_edges(e->from_); });
            });
      }(),
      "incoming edges correct 2");

  auto const check_edges = [](node const* n) {
    return std::all_of(begin(n->edges_), end(n->edges_),
                       [&](edge const& e) { return e.from() == n; }) &&
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
      std::all_of(begin(sched.trips_), end(sched.trips_),
                  [](auto const& t) { return t.second->edges_ != nullptr; }),
      "missing trip edges");
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

  auto const traffic_days = [&](bitfield_idx_or_ptr const& b) {
    if (b.bitfield_idx_ < 1000) {
      return sched.bitfields_.at(b.bitfield_idx_);
    } else {
      return *b.traffic_days_;
    }
  };

  std::cerr << "\n\nGraph:\n";
  auto const print_edge = [&](edge const* e, size_t const indent) {
    indent_line(indent);
    std::cerr << e->type_str() << ": " << station_name(e->from_) << " -> "
              << station_name(e->to_) << "[" << e->to_->id_ << "]\n";
    if (e->is_route_edge()) {
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

        std::cerr << ", dep=" << format_time(time{0, lcon.d_time_})
                  << ", arr=" << format_time(time{0, lcon.a_time_})
                  << ", traffic_days={";
        auto first = true;
        for (auto i = day_idx_t{0}; i != MAX_DAYS; ++i) {
          if (traffic_days(lcon.traffic_days_).test(i)) {
            if (!first) {
              std::cerr << ", ";
            } else {
              first = false;
            }
            std::cerr << i;
          }
        }
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
      print_node(e.to(), 1);
    }
  }
  std::cerr << "\n\n";
}

}  // namespace motis
