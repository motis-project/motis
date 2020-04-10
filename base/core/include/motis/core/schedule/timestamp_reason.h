#pragma once

#include <ostream>

namespace motis {

// The order is important:
// If (forecast time == schedule time), the reason should be FORECAST.
// Thus, FORECAST needs to appear after SCHEDULE (higher priority last).
enum class timestamp_reason { REPAIR, IS, SCHEDULE, PROPAGATION, FORECAST };

inline std::ostream& operator<<(std::ostream& out, timestamp_reason const& r) {
  switch (r) {
    case timestamp_reason::REPAIR: return out << "R";
    case timestamp_reason::SCHEDULE: return out << "S";
    case timestamp_reason::IS: return out << "I";
    case timestamp_reason::FORECAST: return out << "F";
    case timestamp_reason::PROPAGATION: return out << "P";
    default: return out;
  }
}

}  // namespace motis
