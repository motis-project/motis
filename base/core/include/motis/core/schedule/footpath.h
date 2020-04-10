#pragma once

#include "motis/core/schedule/time.h"

namespace motis {

struct footpath {
  uint32_t from_station_;
  uint32_t to_station_;
  time duration_;
};

}  // namespace motis
