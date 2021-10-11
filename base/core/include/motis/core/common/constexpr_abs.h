#pragma once

namespace motis {

template <typename T>
constexpr T constexpr_abs(T const in) {
  return in >= 0 ? in : -in;
}

}  // namespace motis