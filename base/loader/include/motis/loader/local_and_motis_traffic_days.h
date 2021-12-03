#pragma once

#include "motis/core/schedule/bitfield.h"

namespace motis::loader {

struct local_and_motis_traffic_days {
  static constexpr auto const INVALID_SHIFT =
      std::numeric_limits<day_idx_t>::min();

  bitfield shifted_local_traffic_days() {
    return local_traffic_days_ >> shift_;
  }

  day_idx_t shift_{INVALID_SHIFT};
  bitfield motis_traffic_days_;
  bitfield local_traffic_days_;
};

}  // namespace motis::loader