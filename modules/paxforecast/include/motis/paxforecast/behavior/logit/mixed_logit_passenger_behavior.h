#pragma once

#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "utl/enumerate.h"
#include "utl/to_vec.h"

#include "motis/paxmon/passenger_group.h"

#include "motis/paxforecast/alternatives.h"
#include "motis/paxforecast/behavior/util.h"

namespace motis::paxforecast::behavior::logit {

template <typename Generator, typename TransferDistribution>
struct mixed_logit_passenger_behavior {
  mixed_logit_passenger_behavior(Generator& gen, float const coeff_duration,
                                 TransferDistribution& transfer_dist,
                                 unsigned sample_count, bool best_only)
      : gen_{gen},
        coeff_duration_{coeff_duration},
        transfer_dist_{transfer_dist},
        sample_count_{sample_count},
        best_only_{best_only} {}

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
    auto const coeff_transfer = static_cast<float>(transfer_dist_(gen_));
    for (auto const& [idx, alt] : utl::enumerate(alternatives)) {
      auto const prob =
          utility(alt, coeff_transfer) /
          std::accumulate(begin(alternatives), end(alternatives), 0.0F,
                          [&](float const sum, alternative const& a) {
                            return sum + utility(a, coeff_transfer);
                          });
      probabilities[idx] += prob;
    }
  }

  inline float utility(alternative const& alt,
                       float const coeff_transfer) const {
    return coeff_duration_ * (-static_cast<float>(alt.duration_)) +
           coeff_transfer * (-static_cast<float>(alt.transfers_));
  }

  Generator& gen_;
  float coeff_duration_{};
  TransferDistribution& transfer_dist_;
  unsigned sample_count_{};
  bool best_only_{false};
};

}  // namespace motis::paxforecast::behavior::logit
