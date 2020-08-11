#pragma once

#include <map>

#include "utl/zip.h"

#include "motis/core/schedule/schedule.h"

#include "motis/paxmon/graph_access.h"
#include "motis/paxmon/paxmon_data.h"

#include "motis/paxforecast/behavior/passenger_behavior.h"
#include "motis/paxforecast/combined_passenger_group.h"
#include "motis/paxforecast/simulation_result.h"

namespace motis::paxforecast {

template <typename PassengerBehavior>
inline simulation_result simulate_behavior(
    schedule const& sched, motis::paxmon::paxmon_data& data,
    std::map<unsigned, std::vector<combined_passenger_group>> const&
        combined_groups,
    std::vector<measures::please_use> const& announcements,
    PassengerBehavior& pb) {
  simulation_result result;

  auto const add_group_to_alternative =
      [&](motis::paxmon::passenger_group const& grp, alternative const& alt,
          float const probability) {
        auto const total_probability = grp.probability_ * probability;
        for_each_edge(sched, data, alt.compact_journey_,
                      [&](motis::paxmon::journey_leg const&,
                          motis::paxmon::edge const* e) {
                        result.additional_groups_[e].emplace_back(
                            &grp, total_probability);
                      });
      };

  auto const simulate_group =
      [&](motis::paxmon::passenger_group const& grp,
          std::vector<alternative> const& alternatives,
          motis::paxmon::passenger_localization const& localization) {
        auto const allocation =
            pb.pick_routes(grp, alternatives, announcements);
        auto& group_result = result.group_results_[&grp];
        group_result.localization_ = &localization;
        for (auto const& [alt, probability] :
             utl::zip(alternatives, allocation)) {
          group_result.alternatives_.emplace_back(&alt, probability);
          if (probability == 0.0) {
            continue;
          }
          add_group_to_alternative(grp, alt, probability);
        }
      };

  for (auto const& cpgs : combined_groups) {
    for (auto const& cpg : cpgs.second) {
      for (auto const& grp : cpg.groups_) {
        simulate_group(*grp, cpg.alternatives_, cpg.localization_);
      }
    }
  }

  return result;
}

}  // namespace motis::paxforecast
