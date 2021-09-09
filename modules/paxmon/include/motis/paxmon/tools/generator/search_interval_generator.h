#pragma once

#include <ctime>
#include <random>
#include <utility>
#include <vector>

namespace motis::paxmon::tools::generator {

struct search_interval_generator {
  search_interval_generator(std::time_t start_time, std::time_t end_time)
      : start_time_(start_time),
        rng_(std::random_device{}()),
        d_(generate_distribution(start_time, end_time)) {}

  std::pair<std::time_t, std::time_t> random_interval() {
    auto start = start_time_ + d_(rng_) * 3600;
    return {start, start + 3600};
  }

private:
  static std::discrete_distribution<int> generate_distribution(
      std::time_t begin, std::time_t end) {
    auto constexpr k_two_hours = 2 * 3600;
    static const int prob[] = {
        1,  // 01: 00:00 - 01:00
        1,  // 02: 01:00 - 02:00
        1,  // 03: 02:00 - 03:00
        1,  // 04: 03:00 - 04:00
        1,  // 05: 04:00 - 05:00
        2,  // 06: 05:00 - 06:00
        3,  // 07: 06:00 - 07:00
        4,  // 08: 07:00 - 08:00
        4,  // 09: 08:00 - 09:00
        3,  // 10: 09:00 - 10:00
        2,  // 11: 10:00 - 11:00
        2,  // 12: 11:00 - 12:00
        2,  // 13: 12:00 - 13:00
        2,  // 14: 13:00 - 14:00
        3,  // 15: 14:00 - 15:00
        4,  // 16: 15:00 - 16:00
        4,  // 17: 16:00 - 17:00
        4,  // 18: 17:00 - 18:00
        4,  // 19: 18:00 - 19:00
        3,  // 20: 19:00 - 20:00
        2,  // 21: 20:00 - 21:00
        1,  // 22: 21:00 - 22:00
        1,  // 23: 22:00 - 23:00
        1  // 24: 23:00 - 24:00
    };
    std::vector<int> v;
    for (std::time_t t = begin, hour = 0; t < end - k_two_hours;
         t += 3600, ++hour) {
      v.push_back(prob[hour % 24]);  // NOLINT
    }
    return std::discrete_distribution<int>(std::begin(v), std::end(v));
  }

  std::time_t start_time_;
  std::mt19937 rng_;
  std::discrete_distribution<int> d_;
};

}  // namespace motis::paxmon::tools::generator
