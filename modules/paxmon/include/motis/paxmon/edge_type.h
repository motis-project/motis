#pragma once

#include <cstdint>
#include <iosfwd>

namespace motis::paxmon {

enum class edge_type : std::uint8_t {
  TRIP,
  INTERCHANGE,
  WAIT,
  THROUGH,
  DISABLED
};

inline std::ostream& operator<<(std::ostream& out, edge_type const et) {
  switch (et) {
    case edge_type::TRIP: return out << "TRIP";
    case edge_type::INTERCHANGE: return out << "INTERCHANGE";
    case edge_type::WAIT: return out << "WAIT";
    case edge_type::THROUGH: return out << "THROUGH";
    case edge_type::DISABLED: return out << "DISABLED";
  }
  return out;
}

}  // namespace motis::paxmon
