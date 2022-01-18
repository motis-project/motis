#pragma once

#include <ctime>

#include "geo/latlng.h"

namespace motis::ridesharing {

struct passenger {
  int passenger_id_;
  uint16_t passenger_count_;
  geo::latlng pick_up_;
  geo::latlng drop_off_;
  uint16_t price_;
  std::time_t required_arrival_;

  passenger()
      : passenger_id_(-1),
        passenger_count_(1),
        price_(0),
        required_arrival_(0) {}

  passenger(int passenger_id, geo::latlng pick_up, geo::latlng drop_off,
            uint16_t price, std::time_t required_arrival,
            uint16_t passenger_count = 1)
      : passenger_id_(passenger_id),
        passenger_count_(passenger_count),
        pick_up_(pick_up),
        drop_off_(drop_off),
        price_(price),
        required_arrival_(required_arrival) {}
};

}  // namespace motis::ridesharing
