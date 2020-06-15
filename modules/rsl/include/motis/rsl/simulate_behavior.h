#pragma once

#include "motis/core/common/logging.h"
#include "motis/core/schedule/schedule.h"
#include "motis/hash_map.h"

#include "motis/rsl/behavior/passenger_behavior.h"
#include "motis/rsl/combined_passenger_group.h"
#include "motis/rsl/graph_access.h"
#include "motis/rsl/loader/journeys/to_compact_journey.h"
#include "motis/rsl/rsl_data.h"
#include "motis/rsl/simulation_result.h"

namespace motis::rsl {

template <typename PassengerBehavior>
void simulate_behavior(schedule const& sched, rsl_data& data,
                       combined_passenger_group const& cpg,
                       std::vector<measures::please_use> const& announcements,
                       PassengerBehavior& pb, simulation_result& sim_result) {
  if (cpg.alternatives_.empty()) {
    LOG(logging::warn) << "no alternatives found for passenger group with "
                       << cpg.passengers_ << " passengers";
    return;
  }
  auto allocation = std::vector<std::uint16_t>(cpg.alternatives_.size());
  for (auto const grp : cpg.groups_) {
    auto const grp_allocation =
        pb.pick_routes(*grp, cpg.alternatives_, announcements);
    assert(grp_allocation.size() == allocation.size());
    for (auto i = 0; i < grp_allocation.size(); ++i) {
      allocation[i] += grp_allocation[i];
    }
  }

  for (auto i = 0; i < cpg.alternatives_.size(); ++i) {
    auto const additional = allocation[i];
    if (additional == 0) {
      continue;
    }
    auto const& alternative = cpg.alternatives_[i];

    for (auto const& leg : alternative.compact_journey_.legs_) {
      auto te = get_or_add_trip(sched, data, leg.trip_);
      auto in_trip = false;
      for (auto e : te->edges_) {
        if (!in_trip) {
          auto const from = e->from(data.graph_);
          if (from->station_ == leg.enter_station_id_ &&
              from->schedule_time_ == leg.enter_time_) {
            in_trip = true;
          }
        }
        if (in_trip) {
          e->passengers_ += additional;
          sim_result.additional_passengers_[e] += additional;
          if (e->passengers_ > e->capacity_) {
            sim_result.edges_over_capacity_.insert(e);
          }
          auto const to = e->to(data.graph_);
          if (to->station_ == leg.exit_station_id_ &&
              to->schedule_time_ == leg.exit_time_) {
            break;
          }
        }
      }
    }
  }
}

void revert_simulated_behavior(simulation_result& sim_result);

}  // namespace motis::rsl
