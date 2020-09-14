#pragma once

#include "cista/reflection/comparable.h"

#include "motis/core/schedule/time.h"

namespace motis::paxmon {

struct transfer_info {
  CISTA_COMPARABLE()

  enum class type { SAME_STATION, FOOTPATH };
  duration duration_{};
  type type_{type::SAME_STATION};
};

}  // namespace motis::paxmon
