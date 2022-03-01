#pragma once

#include <string>

namespace motis::ridesharing {

struct routing_result {
  double duration_;  // s
  double distance_;  // m

  routing_result(double const& duration, double const& distance)
      : duration_(duration), distance_(distance) {}

  routing_result() = default;

  routing_result& operator+=(routing_result const& rhs) {
    duration_ += rhs.duration_;
    distance_ += rhs.distance_;
    return *this;
  }

  routing_result& operator-=(routing_result const& rhs) {
    duration_ -= rhs.duration_;
    distance_ -= rhs.distance_;
    return *this;
  }

  std::string to_string() const;
};

routing_result operator+(routing_result rhs, routing_result lhs);
routing_result operator-(routing_result rhs, routing_result lhs);

}  // namespace motis::ridesharing