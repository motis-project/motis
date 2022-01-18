#pragma once

#include <ctime>

#include <string>

namespace motis::ridesharing {

struct lift_key {
  std::time_t t_;
  int driver_id_;

  std::string to_string() const {
    return std::to_string(t_) + ";" + std::to_string(driver_id_);
  }
};

}  // namespace motis::ridesharing