#include "motis/paxmon/update_load.h"

#include <algorithm>

#include "utl/erase.h"
#include "utl/verify.h"

#include "motis/paxmon/graph_access.h"

namespace motis::paxmon {

void update_load(passenger_group* pg, reachability_info const& reachability,
                 passenger_localization const& localization, universe& uv) {
  auto disabled_edges = pg->edges_;
  pg->edges_.clear();

  auto const add_to_edge = [&](edge_index const& ei, edge* e) {
    if (std::find(begin(disabled_edges), end(disabled_edges), ei) ==
        end(disabled_edges)) {
      auto guard = std::lock_guard{uv.pax_connection_info_.mutex(e->pci_)};
      add_passenger_group_to_edge(uv, e, pg);
    } else {
      utl::erase(disabled_edges, ei);
    }
    pg->edges_.emplace_back(ei);
  };

  auto const add_interchange = [&](reachable_trip const& rt,
                                   event_node* exit_node) {
    utl::verify(exit_node != nullptr,
                "paxmon::update_load: add_interchange: missing exit_node");
    auto const transfer_time = get_transfer_duration(rt.leg_->enter_transfer_);
    auto enter_node =
        uv.trip_data_.edges(rt.tdi_)[rt.enter_edge_idx_].get(uv)->from(uv);
    for (auto& e : exit_node->outgoing_edges(uv)) {
      if (e.type_ == edge_type::INTERCHANGE && e.to(uv) == enter_node &&
          e.transfer_time() == transfer_time) {
        add_to_edge(get_edge_index(uv, &e), &e);
        return;
      }
    }
    throw utl::fail("paxmon::update_load: interchange edge missing");
  };

  if (reachability.ok_) {
    utl::verify(!reachability.reachable_trips_.empty(),
                "update_load: no reachable trips but reachability ok");
    auto* exit_node = &uv.graph_.nodes_.at(uv.trip_data_.enter_exit_node(
        reachability.reachable_trips_.front().tdi_));
    for (auto const& rt : reachability.reachable_trips_) {
      utl::verify(rt.valid_exit(), "update_load: invalid exit");
      add_interchange(rt, exit_node);
      auto const td_edges = uv.trip_data_.edges(rt.tdi_);
      for (auto i = rt.enter_edge_idx_; i <= rt.exit_edge_idx_; ++i) {
        auto const& ei = td_edges[i];
        add_to_edge(ei, ei.get(uv));
      }
      exit_node = td_edges[rt.exit_edge_idx_].get(uv)->to(uv);
    }
  } else if (!reachability.reachable_trips_.empty()) {
    auto* exit_node = &uv.graph_.nodes_.at(uv.trip_data_.enter_exit_node(
        reachability.reachable_trips_.front().tdi_));
    for (auto const& rt : reachability.reachable_trips_) {
      auto const td_edges = uv.trip_data_.edges(rt.tdi_);
      auto const exit_idx =
          rt.valid_exit() ? rt.exit_edge_idx_ : td_edges.size() - 1;
      add_interchange(rt, exit_node);
      for (auto i = rt.enter_edge_idx_; i <= exit_idx; ++i) {
        auto const& ei = td_edges[i];
        auto* e = ei.get(uv);
        if (e->from(uv)->time_ > localization.current_arrival_time_) {
          break;
        }
        add_to_edge(ei, e);
        auto const to = e->to(uv);
        if (to->station_ == localization.at_station_->index_ &&
            to->time_ == localization.current_arrival_time_) {
          break;
        }
      }
      if (rt.valid_exit()) {
        exit_node = td_edges[rt.exit_edge_idx_].get(uv)->to(uv);
      } else {
        exit_node = nullptr;
      }
    }
  }

  for (auto const& ei : disabled_edges) {
    auto* e = ei.get(uv);
    auto guard = std::lock_guard{uv.pax_connection_info_.mutex(e->pci_)};
    remove_passenger_group_from_edge(uv, e, pg);
  }
}

}  // namespace motis::paxmon
