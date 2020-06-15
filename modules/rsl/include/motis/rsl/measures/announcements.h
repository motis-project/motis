#pragma once

#include <cstddef>

namespace motis::rsl::measures {

struct please_use {
  std::size_t alternative_id_{};
  unsigned direction_station_id_{};
};

}  // namespace motis::rsl::measures
