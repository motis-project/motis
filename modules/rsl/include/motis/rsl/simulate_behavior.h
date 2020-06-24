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
std::vector<std::uint16_t> simulate_behavior(
    schedule const& sched, rsl_data& data, combined_passenger_group const& cpg,
    std::vector<measures::please_use> const& announcements,
    PassengerBehavior& pb, simulation_result& sim_result) {
  if (cpg.alternatives_.empty()) {
    LOG(logging::warn) << "no alternatives found for passenger group with "
                       << cpg.passengers_ << " passengers";
    return {};
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

    for_each_edge(sched, data, alternative.compact_journey_,
                  [&](journey_leg const&, edge* e) {
                    e->passengers_ += additional;
                    sim_result.additional_passengers_[e] += additional;
                    if (e->passengers_ > e->capacity_) {
                      sim_result.edges_over_capacity_.insert(e);
                    }
                  });
  }

  return allocation;
}

void revert_simulated_behavior(simulation_result& sim_result);

}  // namespace motis::rsl
