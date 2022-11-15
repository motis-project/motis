#include "motis/paxmon/checks.h"

#include <algorithm>
#include <vector>

#include "fmt/core.h"

#include "utl/pairwise.h"
#include "utl/verify.h"

#include "motis/core/access/realtime_access.h"
#include "motis/core/access/trip_access.h"
#include "motis/core/debug/trip.h"

#include "motis/paxmon/debug.h"
#include "motis/paxmon/reachability.h"
#include "motis/paxmon/service_info.h"

namespace motis::paxmon {

bool check_edge_in_incoming(universe const& uv, edge const& e) {
  return std::any_of(begin(uv.graph_.incoming_edges(e.to_)),
                     end(uv.graph_.incoming_edges(e.to_)),
                     [&](auto const& ie) { return &ie == &e; });
}

bool check_edge_in_outgoing(universe const& uv, edge const& e) {
  return std::any_of(begin(uv.graph_.outgoing_edges(e.from_)),
                     end(uv.graph_.outgoing_edges(e.from_)),
                     [&](auto const& oe) { return &oe == &e; });
}

bool check_graph_integrity(universe const& uv, schedule const& sched) {
  auto ok = true;

  for (auto const& n : uv.graph_.nodes_) {
    for (auto const& e : n.outgoing_edges(uv)) {
      if (!check_edge_in_incoming(uv, e)) {
        std::cout << "!! outdoing edge missing in target incoming edges\n";
        ok = false;
      }
      for (auto const& pgwr : uv.pax_connection_info_.group_routes_[e.pci_]) {
        auto const& gr = uv.passenger_groups_.route(pgwr);
        if (!e.is_trip()) {
          continue;
        }
        auto const edges = uv.passenger_groups_.route_edges(gr.edges_index_);
        if (std::find_if(begin(edges), end(edges), [&](auto const& ei) {
              return ei.get(uv) == &e;
            }) == end(edges)) {
          std::cout << "!! edge missing in route_edges @" << e.type() << "\n";
          ok = false;
        }
        auto trip_leg_found = false;
        auto const& trips = e.get_trips(sched);
        auto const cj = uv.passenger_groups_.journey(gr.compact_journey_index_);
        for (auto const& trp : trips) {
          if (std::find_if(begin(cj.legs()), end(cj.legs()),
                           [&](journey_leg const& leg) {
                             return leg.trip_idx_ == trp->trip_idx_;
                           }) != end(cj.legs())) {
            trip_leg_found = true;
            break;
          }
        }
        if (!trip_leg_found) {
          std::cout << "!! trip leg missing in planned journey\n";
          ok = false;
        }
      }
      if (e.is_trip() && e.is_valid(uv)) {
        auto grp_route_count =
            uv.pax_connection_info_.group_routes_[e.pci_].size();
        auto const& trips = e.get_trips(sched);
        for (auto const& trp : trips) {
          auto const td_edges = uv.trip_data_.edges(trp);
          if (std::find_if(begin(td_edges), end(td_edges), [&](auto const& ei) {
                return ei.get(uv) == &e;
              }) == end(td_edges)) {
            std::cout << "!! edge missing in trip_data.edges @" << e.type()
                      << ", grp_route_count=" << grp_route_count << "\n";
            ok = false;
          }
        }
      }
    }

    for (auto const& e : n.incoming_edges(uv)) {
      if (!check_edge_in_outgoing(uv, e)) {
        std::cout << "!! incoming edge missing in source outgoing edges\n";
        ok = false;
      }
    }

    if (!n.is_enter_exit_node() && n.is_valid()) {
      auto const in_trip_edges =
          std::count_if(begin(n.incoming_edges(uv)), end(n.incoming_edges(uv)),
                        [&](edge const& e) {
                          return e.is_trip() && e.is_valid(uv) &&
                                 !e.from(uv)->is_enter_exit_node();
                        });
      auto const out_trip_edges =
          std::count_if(begin(n.outgoing_edges(uv)), end(n.outgoing_edges(uv)),
                        [&](edge const& e) {
                          return e.is_trip() && e.is_valid(uv) &&
                                 !e.to(uv)->is_enter_exit_node();
                        });
      if (in_trip_edges > 1 || out_trip_edges > 1) {
        auto const& st = sched.stations_.at(n.station_idx());
        std::cout << "!! " << in_trip_edges << " incoming + " << out_trip_edges
                  << " outgoing trip edges at station " << st->eva_nr_ << " ("
                  << st->name_ << ")\n";
        ok = false;

        std::cout << "  incoming edges:\n";
        for (auto const& in_edge : n.incoming_edges(uv)) {
          auto const& from_station =
              sched.stations_.at(in_edge.from(uv)->station_idx());
          std::cout << "    " << in_edge.type()
                    << " (valid=" << in_edge.is_valid(uv)
                    << ", canceled=" << in_edge.is_canceled(uv)
                    << "), merged_trips_idx=" << in_edge.get_merged_trips_idx()
                    << " => " << in_edge.get_trips(sched).size()
                    << " trips, from=" << from_station->eva_nr_ << " "
                    << from_station->name_
                    << ", capacity=" << in_edge.capacity() << "\n";
          for (auto const& trp : in_edge.get_trips(sched)) {
            auto const service_infos = get_service_infos(sched, trp);
            std::cout << "      " << debug::trip{sched, trp}
                      << "\n        services:";
            for (auto const& [si, count] : service_infos) {
              std::cout << " " << count << "x " << si.category_ << " "
                        << si.train_nr_ << ", line=" << si.line_
                        << ", name=" << si.name_;
            }
            std::cout << "\n";
          }
        }
        std::cout << "  outgoing edges:\n";
        for (auto const& out_edge : n.outgoing_edges(uv)) {
          auto const& to_station =
              sched.stations_.at(out_edge.to(uv)->station_idx());
          std::cout << "    " << out_edge.type()
                    << " (valid=" << out_edge.is_valid(uv)
                    << ", canceled=" << out_edge.is_canceled(uv)
                    << "), merged_trips_idx=" << out_edge.get_merged_trips_idx()
                    << " => " << out_edge.get_trips(sched).size()
                    << " trips, to=" << to_station->eva_nr_ << " "
                    << to_station->name_ << ", capacity=" << out_edge.capacity()
                    << "\n";
          for (auto const& trp : out_edge.get_trips(sched)) {
            auto const service_infos = get_service_infos(sched, trp);
            std::cout << "      " << debug::trip{sched, trp}
                      << "\n        services:";
            for (auto const& [si, count] : service_infos) {
              std::cout << " " << count << "x " << si.category_ << " "
                        << si.train_nr_ << ", line=" << si.line_
                        << ", name=" << si.name_;
            }
            std::cout << "\n";
          }
        }
      }
    }
  }

  for (auto const& [trp_idx, tdi] : uv.trip_data_.mapping_) {
    auto const* trp = get_trip(sched, trp_idx);
    for (auto const& ei : uv.trip_data_.edges(tdi)) {
      auto const* e = ei.get(uv);
      auto const& trips = e->get_trips(sched);
      if (std::find(begin(trips), end(trips), trp) == end(trips)) {
        std::cout << "!! trip missing in edge.trips @" << e->type() << "\n";
        ok = false;
      }
    }
  }

  for (auto const* pg : uv.passenger_groups_) {
    if (pg == nullptr) {
      continue;
    }
    for (auto const& gr : uv.passenger_groups_.routes(pg->id_)) {
      auto const pgwr =
          passenger_group_with_route{pg->id_, gr.local_group_route_index_};
      auto const edges = uv.passenger_groups_.route_edges(gr.edges_index_);
      for (auto const& ei : edges) {
        auto const* e = ei.get(uv);
        auto const group_routes = uv.pax_connection_info_.group_routes(e->pci_);
        if (std::find(begin(group_routes), end(group_routes), pgwr) ==
            end(group_routes)) {
          std::cout << "!! group route not on edge: pg=" << pg->id_
                    << ", gr=" << gr.local_group_route_index_ << " @"
                    << e->type() << "\n";
          ok = false;
        }
      }
    }
  }

  if (!ok) {
    std::cout << "check_graph_integrity failed" << std::endl;
  }
  return ok;
}

bool check_trip_times(universe const& uv, schedule const& sched,
                      trip const* trp, trip_data_index const tdi) {
  auto trip_ok = true;
  std::vector<event_node const*> nodes;
  auto const edges = uv.trip_data_.edges(tdi);
  for (auto const ei : edges) {
    auto const* e = ei.get(uv);
    nodes.emplace_back(e->from(uv));
    nodes.emplace_back(e->to(uv));
  }
  auto const sections = motis::access::sections(trp);

  auto node_idx = 0ULL;
  for (auto const& sec : sections) {
    if (node_idx + 1 > nodes.size()) {
      trip_ok = false;
      std::cout << "!! trip in paxmon graph has fewer sections\n";
      break;
    }
    auto const ev_from = sec.ev_key_from();
    auto const ev_to = sec.ev_key_to();
    auto const pm_from = nodes[node_idx];
    auto const pm_to = nodes[node_idx + 1];

    if (pm_from->type() != event_type::DEP ||
        pm_to->type() != event_type::ARR) {
      std::cout << "!! event nodes out of order @node_idx=" << node_idx << ","
                << (node_idx + 1) << "\n";
      trip_ok = false;
      break;
    }
    if (pm_from->schedule_time() != get_schedule_time(sched, ev_from)) {
      std::cout << "!! schedule time mismatch @dep "
                << sched.stations_.at(pm_from->station_idx())->name_.str()
                << "\n";
      trip_ok = false;
    }
    if (pm_to->schedule_time() != get_schedule_time(sched, ev_to)) {
      std::cout << "!! schedule time mismatch @arr "
                << sched.stations_.at(pm_to->station_idx())->name_.str()
                << "\n";
      trip_ok = false;
    }
    if (pm_from->current_time() != ev_from.get_time()) {
      std::cout << "!! current time mismatch @dep "
                << sched.stations_.at(pm_from->station_idx())->name_.str()
                << "\n";
      trip_ok = false;
    }
    if (pm_to->current_time() != ev_to.get_time()) {
      std::cout << "!! current time mismatch @arr "
                << sched.stations_.at(pm_to->station_idx())->name_.str()
                << "\n";
      trip_ok = false;
    }
    node_idx += 2;
  }
  if (node_idx != nodes.size()) {
    trip_ok = false;
    std::cout << "!! trip in paxmon graph has more sections\n";
  }
  if (!trip_ok) {
    std::cout << "trip (errors above):\n";
    print_trip(sched, trp);
    std::cout << "  sections: " << std::distance(begin(sections), end(sections))
              << ", td edges: " << edges.size()
              << ", event nodes: " << nodes.size() << std::endl;

    print_trip_sections(uv, sched, trp, tdi);
    std::cout << "\n\n";
  }
  return trip_ok;
}

bool check_graph_times(universe const& uv, schedule const& sched) {
  auto ok = true;

  for (auto const& [trp_idx, tdi] : uv.trip_data_.mapping_) {
    if (!check_trip_times(uv, sched, get_trip(sched, trp_idx), tdi)) {
      ok = false;
    }
  }

  return ok;
}

bool check_compact_journey(schedule const& sched, compact_journey const& cj,
                           bool scheduled) {
  auto ok = true;

  if (cj.legs_.empty()) {
    std::cout << "!! empty compact journey\n";
    ok = false;
  }

  for (auto const& leg : cj.legs_) {
    if (leg.enter_station_id_ == leg.exit_station_id_) {
      std::cout << "!! useless journey leg: enter == exit\n";
      ok = false;
    }
    if (scheduled && leg.exit_time_ < leg.enter_time_) {
      std::cout << "!! invalid journey leg: exit time < enter time\n";
      ok = false;
    }
  }

  for (auto const& [l1, l2] : utl::pairwise(cj.legs_)) {
    if (scheduled && l2.enter_time_ < l1.exit_time_) {
      std::cout << "!! leg enter time < previous leg exit time\n";
      ok = false;
    }
    if (!l2.enter_transfer_) {
      // TODO(pablo): support through edges
      std::cout << "!! missing leg enter transfer info\n";
      ok = false;
    }
    if (l1.exit_station_id_ != l2.enter_station_id_ &&
        (!l2.enter_transfer_ ||
         l2.enter_transfer_->type_ == transfer_info::type::SAME_STATION)) {
      std::cout << "!! leg enter station != previous leg exit station\n";
      ok = false;
    }
  }

  if (!ok) {
    std::cout << "compact journey (errors above):\n";
    for (auto const& leg : cj.legs_) {
      print_leg(sched, leg);
    }
  }

  return ok;
}

bool check_group_routes(universe const& uv) {
  auto const& pgc = uv.passenger_groups_;
  auto ok = true;
  auto disabled_but_p_gt0 = 0;
  auto disabled_but_with_edges = 0;
  auto not_disabled_but_no_edges = 0;
  auto broken_but_p_gt0 = 0;
  auto broken_reachability_mismatch = 0;
  auto prob_sum_gt1 = 0;

  for (auto const& pg : pgc) {
    auto p_sum = 0.0F;
    for (auto const& gr : pgc.routes(pg->id_)) {
      auto const edges = pgc.route_edges(gr.edges_index_);
      auto const cj = pgc.journey(gr.compact_journey_index_);
      auto const reachability = get_reachability(uv, cj);

      p_sum += gr.probability_;

      auto const print_reachability = [&]() {
        std::cout << ", reachability=" << reachability.status_;
        if (reachability.first_unreachable_transfer_) {
          auto const& bti = *reachability.first_unreachable_transfer_;
          std::cout << " (leg=" << bti.leg_index_ << ", dir="
                    << (bti.direction_ == transfer_direction_t::ENTER ? "enter"
                                                                      : "exit")
                    << ", carr=" << format_time(bti.current_arrival_time_)
                    << ", cdep=" << format_time(bti.current_departure_time_)
                    << ", tt=" << bti.required_transfer_time_
                    << ", canceled={arr=" << bti.arrival_canceled_
                    << ", dep=" << bti.departure_canceled_ << "})";
        }
      };

      if (gr.disabled_) {
        if (gr.probability_ != 0.0F) {
          ok = false;
          ++disabled_but_p_gt0;
          std::cout << "!! group " << pg->id_ << ": route "
                    << gr.local_group_route_index_
                    << " disabled, but probability " << gr.probability_ << "\n";
        }
        if (!edges.empty()) {
          ok = false;
          ++disabled_but_with_edges;
          std::cout << "!! group " << pg->id_ << ": route "
                    << gr.local_group_route_index_ << " disabled, but on "
                    << edges.size() << " graph edges\n";
        }
      } else {
        if (edges.empty()) {
          ok = false;
          ++not_disabled_but_no_edges;
          std::cout << "!! group " << pg->id_ << ": route "
                    << gr.local_group_route_index_
                    << " not disabled, probability = " << gr.probability_
                    << ", but on no graph edges\n";
        }
      }
      if (gr.broken_ && gr.probability_ != 0.0F) {
        ok = false;
        ++broken_but_p_gt0;
        std::cout << "!! group " << pg->id_ << ": route "
                  << gr.local_group_route_index_
                  << " broken, but probability = " << gr.probability_;
        print_reachability();
        std::cout << "\n";
      }
      if (gr.broken_ != !reachability.ok_) {
        ok = false;
        ++broken_reachability_mismatch;
        std::cout << "!! group " << pg->id_ << ": route "
                  << gr.local_group_route_index_ << " broken=" << gr.broken_;
        print_reachability();
        std::cout << ", probability=" << gr.probability_
                  << ", disabled=" << gr.disabled_ << ", edges=" << edges.size()
                  << "\n";
      }
    }

    if (p_sum < 0.0F || p_sum > 1.05F) {
      ok = false;
      ++prob_sum_gt1;
      std::cout << "!! group " << pg->id_ << ": probability sum=" << p_sum
                << "\n";
    }
  }

  if (!ok) {
    std::cout << "!! => group route problems:\n  " << disabled_but_p_gt0
              << "x disabled but p > 0\n  " << disabled_but_with_edges
              << "x disabled but edges not empty\n  "
              << not_disabled_but_no_edges
              << "x not disabled but edges empty\n  " << broken_but_p_gt0
              << "x broken but p > 0\n  " << broken_reachability_mismatch
              << "x broken / reachability mismatch\n  " << prob_sum_gt1
              << "x probability sum > 1" << std::endl;
  }

  return ok;
}

}  // namespace motis::paxmon
