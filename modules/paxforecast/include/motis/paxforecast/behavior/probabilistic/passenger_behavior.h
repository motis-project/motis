#pragma once

#include <limits>
#include <random>
#include <vector>

#include "utl/enumerate.h"
#include "utl/to_vec.h"

#include "motis/paxmon/passenger_group.h"

#include "motis/paxforecast/alternatives.h"
#include "motis/paxforecast/behavior/util.h"

namespace motis::paxforecast::behavior::probabilistic {

template <typename Generator, typename TransferDist, typename OriginalDist,
          typename RecommendedDist, typename LoadInfoDist,
          typename LoadInfoLowDist, typename LoadInfoNoSeatsDist,
          typename LoadInfoFullDist, typename RandomDist>
struct passenger_behavior {
  passenger_behavior(Generator& gen, unsigned sample_count, bool best_only,
                     TransferDist& transfer_dist, OriginalDist& original_dist,
                     RecommendedDist& recommended_dist,
                     LoadInfoDist& load_info_dist,
                     LoadInfoLowDist& load_info_low_dist,
                     LoadInfoNoSeatsDist& load_info_no_seats_dist,
                     LoadInfoFullDist& load_info_full_dist,
                     RandomDist& random_dist)
      : gen_{gen},
        sample_count_{sample_count},
        best_only_{best_only},
        transfer_dist_{transfer_dist},
        original_dist_{original_dist},
        recommended_dist_{recommended_dist},
        load_info_dist_{load_info_dist},
        load_info_low_dist_{load_info_low_dist},
        load_info_no_seats_dist_{load_info_no_seats_dist},
        load_info_full_dist_{load_info_full_dist},
        random_dist_{random_dist} {}

  std::vector<float> pick_routes(std::vector<alternative> const& alternatives) {
    if (alternatives.empty()) {
      return {};
    }

    auto probabilities = std::vector<float>(alternatives.size());
    for (auto i = 0U; i < sample_count_; ++i) {
      sample(alternatives, probabilities);
    }
    if (best_only_) {
      only_keep_best_alternative(probabilities);
    } else {
      for (auto& p : probabilities) {
        p /= sample_count_;
      }
    }
    return probabilities;
  }

private:
  inline void sample(std::vector<alternative> const& alternatives,
                     std::vector<float>& probabilities) {
    auto best_score = std::numeric_limits<float>::max();
    auto best_alternatives = std::vector<std::size_t>{};
    auto const transfer_weight = static_cast<float>(transfer_dist_(gen_));
    auto const original_weight = static_cast<float>(original_dist_(gen_));
    auto const recommended_weight = static_cast<float>(recommended_dist_(gen_));
    auto const load_info_weight = static_cast<float>(load_info_dist_(gen_));
    auto const load_info_low_weight =
        static_cast<float>(load_info_low_dist_(gen_));
    auto const load_info_no_seats_weight =
        static_cast<float>(load_info_no_seats_dist_(gen_));
    auto const load_info_full_weight =
        static_cast<float>(load_info_full_dist_(gen_));
    for (auto const& [idx, alt] : utl::enumerate(alternatives)) {
      auto score = static_cast<float>(alt.duration_) +
                   static_cast<float>(alt.transfers_) * transfer_weight +
                   static_cast<float>(random_dist_(gen_)) +
                   static_cast<float>(alt.is_original_) * original_weight +
                   static_cast<float>(alt.is_recommended_) * recommended_weight;
      switch (alt.load_info_) {
        case measures::load_level::LOW:
          score += load_info_weight * load_info_low_weight;
          break;
        case measures::load_level::NO_SEATS:
          score += load_info_weight * load_info_no_seats_weight;
          break;
        case measures::load_level::FULL:
          score += load_info_weight * load_info_full_weight;
          break;
        default: break;
      }
      if (score == best_score) {
        best_alternatives.emplace_back(idx);
      } else if (score < best_score) {
        best_score = score;
        best_alternatives.clear();
        best_alternatives.emplace_back(idx);
      }
    }
    for (auto const idx : best_alternatives) {
      probabilities[idx] += 1.0F / static_cast<float>(best_alternatives.size());
    }
  }

  Generator& gen_;
  unsigned sample_count_{};
  bool best_only_{false};
  TransferDist& transfer_dist_;
  OriginalDist& original_dist_;
  RecommendedDist& recommended_dist_;
  LoadInfoDist& load_info_dist_;
  LoadInfoLowDist& load_info_low_dist_;
  LoadInfoNoSeatsDist& load_info_no_seats_dist_;
  LoadInfoFullDist& load_info_full_dist_;
  RandomDist& random_dist_;
};

}  // namespace motis::paxforecast::behavior::probabilistic
