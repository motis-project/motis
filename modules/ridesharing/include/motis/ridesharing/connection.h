#pragma once

#include <cstdint>

#include <vector>

namespace motis::ridesharing {

struct connection {
  std::vector<unsigned> stations_;
  uint16_t from_leg_;
  uint16_t to_leg_;

  explicit connection(std::vector<unsigned> stations, uint16_t from_leg = 0,
                      uint16_t to_leg = 0)
      : stations_(std::move(stations)), from_leg_(from_leg), to_leg_(to_leg) {}

  connection() : from_leg_(6), to_leg_(7) {}
};

}  // namespace motis::ridesharing