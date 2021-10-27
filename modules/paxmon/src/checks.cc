#include "motis/paxmon/checks.h"

#include <algorithm>
#include <vector>

#include "fmt/core.h"

#include "utl/pairwise.h"
#include "utl/verify.h"

#include "motis/core/access/realtime_access.h"

#include "motis/paxmon/debug.h"

namespace motis::paxmon {

bool check_graph_integrity(universe const& uv, schedule const& sched) {
  auto ok = true;

  for (auto const& n : uv.graph_.nodes_) {
    for (auto const& e : n.outgoing_edges(uv)) {
      for (auto const pg_id : uv.pax_connection_info_.groups_[e.pci_]) {
        auto const* pg = uv.passenger_groups_.at(pg_id);
        if (pg->probability_ <= 0.0 || pg->passengers_ >= 200) {
          std::cout << "!! invalid psi @" << e.type() << ": id=" << pg->id_
                    << "\n";
          ok = false;
        }
        if (!e.is_trip()) {
          continue;
        }
        auto const& trips = e.get_trips(sched);
        for (auto const& trp : trips) {
          auto const td_edges = uv.trip_data_.edges(trp);
          if (std::find_if(begin(td_edges), end(td_edges), [&](auto const& ei) {
                return ei.get(uv) == &e;
              }) == end(td_edges)) {
            std::cout << "!! edge missing in trip_data.edges @" << e.type()
                      << "\n";
            ok = false;
          }
        }
        if (std::find_if(begin(pg->edges_), end(pg->edges_),
                         [&](auto const& ei) { return ei.get(uv) == &e; }) ==
            end(pg->edges_)) {
          std::cout << "!! edge missing in pg.edges @" << e.type() << "\n";
          ok = false;
        }
      }
    }
  }

  for (auto const& [trp, tdi] : uv.trip_data_.mapping_) {
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
    for (auto const& ei : pg->edges_) {
      auto const* e = ei.get(uv);
      auto const groups = uv.pax_connection_info_.groups_[e->pci_];
      if (std::find(begin(groups), end(groups), pg->id_) == end(groups)) {
        std::cout << "!! passenger group not on edge: id=" << pg->id_ << " @"
                  << e->type() << "\n";
        ok = false;
      }
    }
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

  for (auto const& [trp, tdi] : uv.trip_data_.mapping_) {
    if (!check_trip_times(uv, sched, trp, tdi)) {
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

}  // namespace motis::paxmon
