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

  if (reachability.ok_) {
    for (auto const& rt : reachability.reachable_trips_) {
      utl::verify(rt.valid_exit(), "update_load: invalid exit");
      for (auto i = rt.enter_edge_idx_; i <= rt.exit_edge_idx_; ++i) {
        auto e = rt.td_->edges_[i];
        if (std::find(begin(disabled_edges), end(disabled_edges), e) ==
            end(disabled_edges)) {
          auto guard = std::lock_guard{e->pax_connection_info_.mutex_};
          add_passenger_group_to_edge(e, pg);
        } else {
          utl::erase(disabled_edges, e);
        }
        pg->edges_.emplace_back(e);
      }
    }
  } else {
    for (auto const& rt : reachability.reachable_trips_) {
      auto const exit_idx =
          rt.valid_exit() ? rt.exit_edge_idx_ : rt.td_->edges_.size() - 1;
      for (auto i = rt.enter_edge_idx_; i <= exit_idx; ++i) {
        auto e = rt.td_->edges_[i];
        if (e->from(g)->time_ > localization.current_arrival_time_) {
          break;
        }
        if (std::find(begin(disabled_edges), end(disabled_edges), e) ==
            end(disabled_edges)) {
          auto guard = std::lock_guard{e->pax_connection_info_.mutex_};
          add_passenger_group_to_edge(e, pg);
        } else {
          utl::erase(disabled_edges, e);
        }
        pg->edges_.emplace_back(e);
        auto const to = e->to(g);
        if (to->station_ == localization.at_station_->index_ &&
            to->time_ == localization.current_arrival_time_) {
          break;
        }
      }
    }
  }

  for (auto e : disabled_edges) {
    auto guard = std::lock_guard{e->pax_connection_info_.mutex_};
    remove_passenger_group_from_edge(e, pg);
  }
}

}  // namespace motis::paxmon
