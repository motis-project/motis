#pragma once

#include <cstdint>
#include <iosfwd>
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

inline std::ostream& operator<<(std::ostream& out, capacity_source const cs) {
  switch (cs) {
    case capacity_source::FORMATION_VEHICLES:
      return out << "FORMATION_VEHICLES";
    case capacity_source::FORMATION_VEHICLE_GROUPS:
      return out << "FORMATION_VEHICLE_GROUPS";
    case capacity_source::FORMATION_BAUREIHE:
      return out << "FORMATION_BAUREIHE";
    case capacity_source::FORMATION_GATTUNG: return out << "FORMATION_GATTUNG";
    case capacity_source::TRIP_EXACT: return out << "TRIP_EXACT";
    case capacity_source::TRIP_PRIMARY: return out << "TRIP_PRIMARY";
    case capacity_source::TRAIN_NR_AND_STATIONS:
      return out << "TRAIN_NR_AND_STATIONS";
    case capacity_source::TRAIN_NR: return out << "TRAIN_NR";
    case capacity_source::CATEGORY: return out << "CATEGORY";
    case capacity_source::CLASZ: return out << "CLASZ";
    case capacity_source::OVERRIDE: return out << "OVERRIDE";
    case capacity_source::UNLIMITED: return out << "UNLIMITED";
    case capacity_source::UNKNOWN: return out << "UNKNOWN";
  }
  return out;
}

constexpr const std::uint16_t UNKNOWN_CAPACITY = 0;

constexpr const std::uint16_t UNLIMITED_CAPACITY =
    std::numeric_limits<std::uint16_t>::max();

}  // namespace motis::paxmon
