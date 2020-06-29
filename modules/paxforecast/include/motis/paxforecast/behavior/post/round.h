#pragma once

#include <cstdint>
#include <algorithm>

#include "utl/enumerate.h"
#include "utl/verify.h"

#include "motis/paxforecast/behavior/util.h"

namespace motis::paxforecast::behavior::post {

struct round {
  using out_assignment_t = std::uint16_t;

  static std::vector<std::uint16_t> postprocess(
      std::vector<double> const& real_assignments,
      std::uint16_t const total_passengers) {
    auto result = std::vector<uint16_t>(real_assignments.size());

    std::uint16_t assigned = 0;
    for (auto const& [i, real] : utl::enumerate(real_assignments)) {
      auto const int_pax = static_cast<std::uint16_t>(real);
      result[i] = int_pax;
      assigned += int_pax;
    }

    if (assigned < total_passengers) {
      distribute_rest(result, real_assignments, total_passengers, assigned);
    }

    utl::verify(
        std::accumulate(begin(result), end(result),
                        static_cast<std::uint16_t>(0)) == total_passengers,
        "motis::paxforecast::behavior::post::round failed");

    return result;
  }

private:
  static void distribute_rest(std::vector<std::uint16_t> result,
                              std::vector<double> const& real_assignments,
                              std::uint16_t const total_passengers,
                              std::uint16_t const assigned) {
    auto real_rest = real_assignments;
    for (auto const& [i, real] : utl::enumerate(real_assignments)) {
      real_rest[i] = real - result[i];
    }
    auto remaining = total_passengers - assigned;
    while (remaining > 0) {
      auto const equal_best = max_indices(real_rest);
      auto const to_distribute =
          std::min(static_cast<std::uint16_t>(remaining),
                   static_cast<std::uint16_t>(equal_best.size()));
      utl::verify(
          to_distribute > 0,
          "motis::paxforecast::behavior::post::round::distribute_rest failed");
      for (auto i = 0; i < to_distribute; ++i) {
        result[equal_best[i]]++;
        remaining--;
      }
    }
  }
};

}  // namespace motis::paxforecast::behavior::post
