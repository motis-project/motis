#pragma once

#include <map>
#include <mutex>

#include "utl/zip.h"

#include "motis/core/schedule/schedule.h"

#include "motis/module/context/motis_parallel_for.h"

#include "motis/paxmon/graph_access.h"
#include "motis/paxmon/paxmon_data.h"

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
                        if (e->is_trip()) {
                          result.additional_groups_[e].emplace_back(
                              &grp, total_probability);
                        }
                      });
      };

  std::mutex result_mutex;
  motis_parallel_for(combined_groups, [&](auto const& cpgs) {
    for (auto const& cpg : cpgs.second) {
      auto const allocation = pb.pick_routes(*cpg.groups_.front(),
                                             cpg.alternatives_, announcements);
      auto guard = std::lock_guard{result_mutex};
      for (auto const& grp : cpg.groups_) {
        auto& group_result = result.group_results_[grp];
        group_result.localization_ = &cpg.localization_;
        for (auto const& [alt, probability] :
             utl::zip(cpg.alternatives_, allocation)) {
          group_result.alternatives_.emplace_back(&alt, probability);
          if (probability == 0.0) {
            continue;
          }
          add_group_to_alternative(*grp, alt, probability);
        }
      }
    }
  });

  return result;
}

}  // namespace motis::paxforecast
