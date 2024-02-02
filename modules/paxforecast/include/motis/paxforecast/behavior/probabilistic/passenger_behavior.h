#pragma once

#include <ctime>
#include <limits>
#include <random>
#include <vector>

#include "utl/enumerate.h"

#include "motis/paxforecast/alternatives.h"
#include "motis/paxforecast/behavior/util.h"

namespace motis::paxforecast::behavior::probabilistic {

struct passenger_behavior {
  passenger_behavior(
      unsigned sample_count, bool best_only,
      std::normal_distribution<float> const& transfer_dist,
      std::normal_distribution<float> const& original_dist,
      std::normal_distribution<float> const& recommended_dist,
      std::normal_distribution<float> const& load_info_dist,
      std::normal_distribution<float> const& load_info_low_dist,
      std::normal_distribution<float> const& load_info_no_seats_dist,
      std::normal_distribution<float> const& load_info_full_dist,
      std::normal_distribution<float> const& random_dist)
      : gen_{get_seed()},
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

  static std::mt19937::result_type get_seed() {
#ifdef WIN32
    return static_cast<std::mt19937::result_type>(
        std::time(nullptr) %
        std::numeric_limits<std::mt19937::result_type>::max());
#else
    auto rd = std::random_device();
    return rd();
#endif
  }

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

  std::mt19937 gen_;
  unsigned sample_count_{};
  bool best_only_{false};
  std::normal_distribution<float> transfer_dist_;
  std::normal_distribution<float> original_dist_;
  std::normal_distribution<float> recommended_dist_;
  std::normal_distribution<float> load_info_dist_;
  std::normal_distribution<float> load_info_low_dist_;
  std::normal_distribution<float> load_info_no_seats_dist_;
  std::normal_distribution<float> load_info_full_dist_;
  std::normal_distribution<float> random_dist_;
};

}  // namespace motis::paxforecast::behavior::probabilistic
