#pragma once

#include <cmath>
#include <cstdint>
#include <random>

namespace motis::paxmon::tools::groups {

struct group_generator {
  group_generator(double group_size_mean, double group_size_stddev,
                  double group_count_mean, double group_count_stddev)
      : rng_{std::random_device{}()},
        group_size_dist_{group_size_mean, group_size_stddev},
        group_count_dist_{group_count_mean, group_count_stddev} {}

  std::uint64_t get_group_count() {
    return static_cast<std::uint64_t>(
        std::max(1.0, std::round(group_count_dist_(rng_))));
  }

  std::uint16_t get_group_size(std::uint16_t max_size = 100) {
    return std::min(max_size, static_cast<std::uint16_t>(std::max(
                                  1.0, std::round(group_size_dist_(rng_)))));
  }

  std::mt19937 rng_;
  std::normal_distribution<> group_size_dist_;
  std::normal_distribution<> group_count_dist_;
};

}  // namespace motis::paxmon::tools::groups
