#pragma once

#include "motis/core/schedule/bitfield.h"

namespace motis::loader {

struct local_and_motis_traffic_days {
  day_idx_t shift_;
  bitfield motis_traffic_days_;
  bitfield local_traffic_days_;
};

}  // namespace motis::loader
