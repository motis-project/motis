#include "motis/ridesharing/routing_result.h"

#include <sstream>

namespace motis::ridesharing {

routing_result operator+(routing_result rhs, routing_result lhs) {
  return {rhs.duration_ + lhs.duration_, rhs.distance_ + lhs.distance_};
}

routing_result operator-(routing_result rhs, routing_result lhs) {
  return {rhs.duration_ - lhs.duration_, rhs.distance_ - lhs.distance_};
}

std::string routing_result::to_string() const {
  std::ostringstream strs;
  strs << duration_ / 60 << "/" << distance_ / 1000;
  std::string dist = strs.str();
  return "[" + dist + "]";
}

}  // namespace motis::ridesharing