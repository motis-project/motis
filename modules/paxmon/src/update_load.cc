#include "motis/paxmon/update_load.h"

#include <algorithm>

#include "utl/erase.h"
#include "utl/verify.h"

#include "motis/paxmon/graph_access.h"

namespace motis::paxmon {

void update_load(passenger_group* pg, reachability_info const& reachability,
                 passenger_localization const& localization, graph const& g) {
  auto disabled_edges = pg->edges_;
  pg->edges_.clear();
  utl::verify(!disabled_edges.empty(), "update_load: disabled_edges empty");

  auto const add_to_edge = [&](edge* e) {
    if (std::find(begin(disabled_edges), end(disabled_edges), e) ==
        end(disabled_edges)) {
      auto guard = std::lock_guard{e->pax_connection_info_.mutex_};
      add_passenger_group_to_edge(e, pg);
    } else {
      utl::erase(disabled_edges, e);
    }
    pg->edges_.emplace_back(e);
  };

  auto const add_interchange = [&](reachable_trip const& rt,
                                   event_node* exit_node) {
    utl::verify(exit_node != nullptr,
                "update_load: add_interchange: missing exit_node");
    auto const transfer_time = get_transfer_duration(rt.leg_->enter_transfer_);
    auto enter_node = rt.td_->edges_[rt.enter_edge_idx_]->from(g);
    for (auto& e : exit_node->outgoing_edges(g)) {
      if (e->type_ == edge_type::INTERCHANGE && e->to(g) == enter_node &&
          e->transfer_time() == transfer_time) {
        add_to_edge(e.get());
        return;
      }
    }
    pg->edges_.emplace_back(add_edge(make_interchange_edge(
        exit_node, enter_node, transfer_time, pax_connection_info{pg})));
  };

  if (reachability.ok_) {
    utl::verify(!reachability.reachable_trips_.empty(),
                "update_load: no reachable trips but reachability ok");
    auto exit_node =
        &reachability.reachable_trips_.front().td_->enter_exit_node_;
    for (auto const& rt : reachability.reachable_trips_) {
      utl::verify(rt.valid_exit(), "update_load: invalid exit");
      add_interchange(rt, exit_node);
      for (auto i = rt.enter_edge_idx_; i <= rt.exit_edge_idx_; ++i) {
        auto e = rt.td_->edges_[i];
        add_to_edge(e);
      }
      exit_node = rt.td_->edges_[rt.exit_edge_idx_]->to(g);
    }
  } else if (!reachability.reachable_trips_.empty()) {
    auto exit_node =
        &reachability.reachable_trips_.front().td_->enter_exit_node_;
    for (auto const& rt : reachability.reachable_trips_) {
      auto const exit_idx =
          rt.valid_exit() ? rt.exit_edge_idx_ : rt.td_->edges_.size() - 1;
      add_interchange(rt, exit_node);
      for (auto i = rt.enter_edge_idx_; i <= exit_idx; ++i) {
        auto e = rt.td_->edges_[i];
        if (e->from(g)->time_ > localization.current_arrival_time_) {
          break;
        }
        add_to_edge(e);
        auto const to = e->to(g);
        if (to->station_ == localization.at_station_->index_ &&
            to->time_ == localization.current_arrival_time_) {
          break;
        }
      }
      if (rt.valid_exit()) {
        exit_node = rt.td_->edges_[rt.exit_edge_idx_]->to(g);
      } else {
        exit_node = nullptr;
      }
    }
  }

  for (auto e : disabled_edges) {
    auto guard = std::lock_guard{e->pax_connection_info_.mutex_};
    remove_passenger_group_from_edge(e, pg);
  }
}

}  // namespace motis::paxmon
