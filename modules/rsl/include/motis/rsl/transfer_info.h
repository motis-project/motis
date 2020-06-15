#pragma once

#include "motis/core/schedule/time.h"

namespace motis::rsl {

struct transfer_info {
  enum class type { SAME_STATION, FOOTPATH };
  duration duration_{};
  type type_{type::SAME_STATION};
};

}  // namespace motis::rsl
