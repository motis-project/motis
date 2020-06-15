#pragma once

#include <cstdint>
#include <algorithm>

#include "utl/enumerate.h"

#include "motis/rsl/behavior/util.h"

namespace motis::rsl::behavior::post {

struct real {
  using out_assignment_t = double;

  static inline std::vector<double> postprocess(
      std::vector<double> const& real_assignments,
      std::uint16_t const /*total_passengers*/) {
    return real_assignments;
  }
};

}  // namespace motis::rsl::behavior::post
