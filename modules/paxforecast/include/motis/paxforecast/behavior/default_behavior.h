#pragma once

#include <random>

#include "motis/paxforecast/behavior/probabilistic/passenger_behavior.h"

namespace motis::paxforecast::behavior {

struct default_behavior {
  explicit default_behavior(bool deterministic_mode,
                            unsigned sample_count = 1000)
      : rnd_gen_{get_seed()},
        pb_{rnd_gen_,
            sample_count,
            deterministic_mode,
            transfer_dist_,
            original_dist_,
            recommended_dist_,
            load_info_dist_,
            load_info_low_dist_,
            load_info_no_seats_dist_,
            load_info_full_dist_,
            random_dist_} {}

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

  std::mt19937 rnd_gen_;
  std::normal_distribution<float> transfer_dist_{30.0F, 10.0F};
  std::normal_distribution<float> original_dist_{-5.0F, 3.0F};
  // std::normal_distribution<float> recommended_dist_{-10.0F, 5.0F};
  std::normal_distribution<float> recommended_dist_{-120.0F, 5.0F};
  std::normal_distribution<float> load_info_dist_{1.0F, 0.3F};
  std::normal_distribution<float> load_info_low_dist_{-2.0F, 3.0F};
  std::normal_distribution<float> load_info_no_seats_dist_{15.0F, 5.0F};
  std::normal_distribution<float> load_info_full_dist_{60.0F, 10.0F};
  std::normal_distribution<float> random_dist_{0.0F, 5.0F};
  probabilistic::passenger_behavior<
      std::mt19937, std::normal_distribution<float>,
      std::normal_distribution<float>, std::normal_distribution<float>,
      std::normal_distribution<float>, std::normal_distribution<float>,
      std::normal_distribution<float>, std::normal_distribution<float>,
      std::normal_distribution<float>>
      pb_;
};

}  // namespace motis::paxforecast::behavior
