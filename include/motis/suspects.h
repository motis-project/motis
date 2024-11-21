#pragma once

#include <vector>

#include "nigiri/types.h"

#include "motis/fwd.h"

namespace motis {

struct suspects {
  explicit suspects(nigiri::timetable const&);
  std::vector<nigiri::route_idx_t> routes_;
};

}  // namespace motis