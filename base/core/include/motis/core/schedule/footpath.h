#pragma once

#include "motis/core/schedule/time.h"

#include "cista/reflection/comparable.h"

namespace motis {

struct footpath {
  CISTA_COMPARABLE()
  uint32_t from_station_;
  uint32_t to_station_;
  time duration_;
};

}  // namespace motis
