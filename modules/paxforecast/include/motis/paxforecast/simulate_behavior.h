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
                                     motis::paxmon::passenger_group const& grp,
                                     alternative const& alt,
                                     float const probability) {
  auto const total_probability = grp.probability_ * probability;
  for_each_edge(
      sched, caps, uv, alt.compact_journey_,
      [&](motis::paxmon::journey_leg const&, motis::paxmon::edge const* e) {
        if (e->is_trip()) {
          result.additional_groups_[e].emplace_back(&grp, total_probability);
        }
      });
}

template <typename PassengerBehavior>
inline simulation_result simulate_behavior(
    schedule const& sched, motis::paxmon::capacity_maps const& caps,
    motis::paxmon::universe& uv,
    std::map<unsigned, std::vector<combined_passenger_group>> const&
        combined_groups,
    PassengerBehavior& pb) {
  simulation_result result;

  std::mutex result_mutex;
  std::vector<std::uint8_t> found_alt_count;
  std::vector<std::uint8_t> picked_alt_count;
  std::vector<float> best_alt_prob;
  std::vector<float> second_alt_prob;

  motis_parallel_for(
      combined_groups, ([&](auto const& cpgs) {
        for (auto const& cpg : cpgs.second) {
          if (cpg.groups_.empty()) {
            continue;
          }
          auto const allocation =
              pb.pick_routes(*cpg.groups_.front(), cpg.alternatives_);
          auto guard = std::lock_guard{result_mutex};
          result.stats_.group_count_ += cpg.groups_.size();
          for (auto const& grp : cpg.groups_) {
            auto& group_result = result.group_results_[grp];
            group_result.localization_ = &cpg.localization_;
            std::uint8_t picked = 0;
            for (auto const& [alt, probability] :
                 utl::zip(cpg.alternatives_, allocation)) {
              if (probability < 0.01F) {
                continue;
              }
              group_result.alternatives_.emplace_back(&alt, probability);
              add_group_to_alternative(sched, caps, uv, result, *grp, alt,
                                       probability);
              ++picked;
            }
            found_alt_count.emplace_back(
                static_cast<std::uint8_t>(cpg.alternatives_.size()));
            picked_alt_count.emplace_back(picked);
            if (picked == 1) {
              best_alt_prob.emplace_back(
                  group_result.alternatives_.front().second);
            } else if (picked > 1) {
              auto top = std::vector<std::pair<alternative const*, float>>(2);
              std::partial_sort_copy(begin(group_result.alternatives_),
                                     end(group_result.alternatives_),
                                     begin(top), end(top),
                                     [](auto const& lhs, auto const& rhs) {
                                       return lhs.second > rhs.second;
                                     });
              best_alt_prob.emplace_back(top[0].second);
              second_alt_prob.emplace_back(top[1].second);
            }
          }
        }
      }));

  result.stats_.combined_group_count_ = combined_groups.size();
  result.stats_.found_alt_count_avg_ = avg(found_alt_count);
  result.stats_.picked_alt_count_avg_ = avg(picked_alt_count);
  result.stats_.best_alt_prob_avg_ = avg(best_alt_prob);
  result.stats_.second_alt_prob_avg_ = avg(second_alt_prob);

  return result;
}

// TODO(pablo): refactor, remove duplication
template <typename PassengerBehavior>
inline simulation_result simulate_behavior(
    schedule const& sched, motis::paxmon::capacity_maps const& caps,
    motis::paxmon::universe& uv,
    mcd::hash_map<mcd::pair<motis::paxmon::passenger_localization,
                            motis::paxmon::compact_journey>,
                  combined_passenger_group> const& combined_groups,
    PassengerBehavior& pb) {
  simulation_result result;

  std::mutex result_mutex;
  std::vector<std::uint8_t> found_alt_count;
  std::vector<std::uint8_t> picked_alt_count;
  std::vector<float> best_alt_prob;
  std::vector<float> second_alt_prob;

  motis_parallel_for(
      combined_groups, ([&](auto const& entry) {
        auto const& cpg = entry.second;
        if (cpg.groups_.empty()) {
          return;
        }
        auto const allocation =
            pb.pick_routes(*cpg.groups_.front(), cpg.alternatives_);
        auto guard = std::lock_guard{result_mutex};
        result.stats_.group_count_ += cpg.groups_.size();
        for (auto const& grp : cpg.groups_) {
          auto& group_result = result.group_results_[grp];
          group_result.localization_ = &cpg.localization_;
          std::uint8_t picked = 0;
          for (auto const& [alt, probability] :
               utl::zip(cpg.alternatives_, allocation)) {
            if (probability < 0.01F) {
              continue;
            }
            group_result.alternatives_.emplace_back(&alt, probability);
            add_group_to_alternative(sched, caps, uv, result, *grp, alt,
                                     probability);
            ++picked;
          }
          found_alt_count.emplace_back(
              static_cast<std::uint8_t>(cpg.alternatives_.size()));
          picked_alt_count.emplace_back(picked);
          if (picked == 1) {
            best_alt_prob.emplace_back(
                group_result.alternatives_.front().second);
          } else if (picked > 1) {
            auto top = std::vector<std::pair<alternative const*, float>>(2);
            std::partial_sort_copy(begin(group_result.alternatives_),
                                   end(group_result.alternatives_), begin(top),
                                   end(top),
                                   [](auto const& lhs, auto const& rhs) {
                                     return lhs.second > rhs.second;
                                   });
            best_alt_prob.emplace_back(top[0].second);
            second_alt_prob.emplace_back(top[1].second);
          }
        }
      }));

  result.stats_.combined_group_count_ = combined_groups.size();
  result.stats_.found_alt_count_avg_ = avg(found_alt_count);
  result.stats_.picked_alt_count_avg_ = avg(picked_alt_count);
  result.stats_.best_alt_prob_avg_ = avg(best_alt_prob);
  result.stats_.second_alt_prob_avg_ = avg(second_alt_prob);

  return result;
}

}  // namespace motis::paxforecast
