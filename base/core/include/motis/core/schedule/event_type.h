#pragma once

#include <cinttypes>
#include <ostream>

namespace motis {

enum class event_type : uint8_t { DEP, ARR };

inline std::ostream& operator<<(std::ostream& o, event_type const t) {
  return o << (t == event_type::DEP ? "DEP" : "ARR");
}

}  // namespace motis
