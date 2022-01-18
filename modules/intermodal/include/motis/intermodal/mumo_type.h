#pragma once

enum class mumo_type : int { FOOT, BIKE, CAR, CAR_PARKING, RIDESHARING };

inline int to_int(mumo_type const type) {
  return static_cast<typename std::underlying_type<mumo_type>::type>(type);
}

inline std::string to_string(mumo_type const type) {
  static char const* strs[] = {"foot", "bike", "car", "car_parking",
                               "ridesharing"};
  return strs[to_int(type)];  // NOLINT
}