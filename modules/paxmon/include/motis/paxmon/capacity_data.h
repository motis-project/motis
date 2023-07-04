#pragma once

#include <cstdint>
#include <limits>
#include <utility>

namespace motis::paxmon {

enum class capacity_source : std::uint8_t {
  FORMATION_VEHICLES,
  FORMATION_VEHICLE_GROUPS,
  FORMATION_BAUREIHE,
  FORMATION_GATTUNG,
  TRIP_EXACT,
  TRIP_PRIMARY,
  TRAIN_NR_AND_STATIONS,
  TRAIN_NR,
  CATEGORY,
  CLASZ,
  OVERRIDE,
  UNLIMITED,
  UNKNOWN
};

constexpr const std::uint16_t UNKNOWN_CAPACITY = 0;

constexpr const std::uint16_t UNLIMITED_CAPACITY =
    std::numeric_limits<std::uint16_t>::max();

}  // namespace motis::paxmon
