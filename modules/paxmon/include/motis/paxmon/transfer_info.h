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

inline duration get_transfer_duration(std::optional<transfer_info> const& ti) {
  return ti.has_value() ? ti.value().duration_ : 0;
}

}  // namespace motis::paxmon
