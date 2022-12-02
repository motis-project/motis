#pragma once

#include <cstdint>
#include <algorithm>
#include <map>
#include <mutex>

#include "utl/zip.h"

#include "motis/core/schedule/schedule.h"

#include "motis/module/context/motis_parallel_for.h"

#include "motis/paxmon/compact_journey.h"
#include "motis/paxmon/graph_access.h"
#include "motis/paxmon/localization.h"
#include "motis/paxmon/universe.h"

#include "motis/paxforecast/behavior/util.h"
#include "motis/paxforecast/combined_passenger_group.h"
#include "motis/paxforecast/simulation_result.h"

namespace motis::paxforecast {

template <typename T>
double avg(std::vector<T> const& data) {
  double mean = 0.;
  if (data.empty()) {
    return mean;
  }
  for (auto const& v : data) {
    mean += static_cast<double>(v);
  }
  mean /= static_cast<double>(data.size());
  return mean;
}

inline void add_group_to_alternative(schedule const& sched,
                                     motis::paxmon::capacity_maps const& caps,
                                     motis::paxmon::universe& uv,
                                     simulation_result& result,
                                     motis::paxmon::additional_group const& ag,
                                     alternative const& alt) {
  for_each_edge(
      sched, caps, uv, alt.compact_journey_,
      [&](motis::paxmon::journey_leg const&, motis::paxmon::edge const* e) {
        if (e->is_trip()) {
          result.additional_groups_[e].emplace_back(ag);
        }
      });
}

struct sim_data {
  explicit sim_data(simulation_result& result) : result_{result} {}

  void finish_stats(std::uint64_t combined_group_count) {
    result_.stats_.combined_group_count_ = combined_group_count;
    result_.stats_.found_alt_count_avg_ = avg(found_alt_count_);
    result_.stats_.picked_alt_count_avg_ = avg(picked_alt_count_);
    result_.stats_.best_alt_prob_avg_ = avg(best_alt_prob_);
    result_.stats_.second_alt_prob_avg_ = avg(second_alt_prob_);
  }

  simulation_result& result_;
  std::mutex result_mutex_;
  std::vector<std::uint8_t> found_alt_count_;
  std::vector<std::uint8_t> picked_alt_count_;
  std::vector<float> best_alt_prob_;
  std::vector<float> second_alt_prob_;
};

template <typename PassengerBehavior>
inline void simulate_behavior_for_cpg(schedule const& sched,
                                      motis::paxmon::capacity_maps const& caps,
                                      motis::paxmon::universe& uv,
                                      PassengerBehavior& pb,
                                      combined_passenger_group const& cpg,
                                      sim_data& sd,
                                      float const probability_threshold) {
  if (cpg.group_routes_.empty()) {
    return;
  }
  auto const allocation = pb.pick_routes(cpg.alternatives_);
  auto guard = std::lock_guard{sd.result_mutex_};
  sd.result_.stats_.group_route_count_ += cpg.group_routes_.size();
  for (auto const& pgwrap : cpg.group_routes_) {
    auto& group_route_result = sd.result_.group_route_results_[pgwrap.pgwr_];
    group_route_result.localization_ = &cpg.localization_;
    auto const new_probs = behavior::calc_new_probabilites(
        pgwrap.probability_, allocation, probability_threshold);
    std::uint8_t picked = 0;
    for (auto const& [alt, probability] :
         utl::zip(cpg.alternatives_, new_probs)) {
      if (probability == 0.0F) {
        continue;
      }
      group_route_result.alternative_probabilities_.emplace_back(&alt,
                                                                 probability);
      add_group_to_alternative(
          sched, caps, uv, sd.result_,
          paxmon::additional_group{pgwrap.passengers_, probability}, alt);
      ++picked;
    }
    sd.found_alt_count_.emplace_back(
        static_cast<std::uint8_t>(cpg.alternatives_.size()));
    sd.picked_alt_count_.emplace_back(picked);
    if (picked == 1) {
      sd.best_alt_prob_.emplace_back(
          group_route_result.alternative_probabilities_.front().second);
    } else if (picked > 1) {
      auto top = std::vector<std::pair<alternative const*, float>>(2);
      std::partial_sort_copy(
          begin(group_route_result.alternative_probabilities_),
          end(group_route_result.alternative_probabilities_), begin(top),
          end(top), [](auto const& lhs, auto const& rhs) {
            return lhs.second > rhs.second;
          });
      sd.best_alt_prob_.emplace_back(top[0].second);
      sd.second_alt_prob_.emplace_back(top[1].second);
    }
  }
}

template <typename PassengerBehavior>
inline simulation_result simulate_behavior(
    schedule const& sched, motis::paxmon::capacity_maps const& caps,
    motis::paxmon::universe& uv,
    std::map<unsigned, std::vector<combined_passenger_group>> const&
        combined_groups,
    PassengerBehavior& pb, float const probability_threshold) {
  simulation_result result;
  sim_data sd{result};
  motis_parallel_for(combined_groups, ([&](auto const& cpgs) {
                       for (auto const& cpg : cpgs.second) {
                         simulate_behavior_for_cpg(sched, caps, uv, pb, cpg, sd,
                                                   probability_threshold);
                       }
                     }));
  sd.finish_stats(combined_groups.size());
  return result;
}

template <typename PassengerBehavior>
inline simulation_result simulate_behavior(
    schedule const& sched, motis::paxmon::capacity_maps const& caps,
    motis::paxmon::universe& uv,
    mcd::hash_map<mcd::pair<motis::paxmon::passenger_localization,
                            motis::paxmon::compact_journey>,
                  combined_passenger_group> const& combined_groups,
    PassengerBehavior& pb, float const probability_threshold) {
  simulation_result result;
  sim_data sd{result};
  motis_parallel_for(combined_groups, ([&](auto const& cpgs) {
                       simulate_behavior_for_cpg(sched, caps, uv, pb,
                                                 cpgs.second, sd,
                                                 probability_threshold);
                     }));
  sd.finish_stats(combined_groups.size());
  return result;
}

}  // namespace motis::paxforecast
