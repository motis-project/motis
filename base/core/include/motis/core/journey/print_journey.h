#pragma once

#include <iostream>
#include <ostream>

#include "motis/core/journey/journey.h"

namespace motis {

enum class realtime_format { NONE, OFFSET, TIME };

inline std::istream& operator>>(std::istream& in, realtime_format& rt_format) {
  std::string s;
  in >> s;
  if (s == "none") {
    rt_format = realtime_format::NONE;
  } else if (s == "offset") {
    rt_format = realtime_format::OFFSET;
  } else if (s == "time") {
    rt_format = realtime_format::TIME;
  } else {
    in.setstate(std::ios_base::failbit);
  }
  return in;
}

inline std::ostream& operator<<(std::ostream& out,
                                realtime_format const& rt_format) {
  switch (rt_format) {
    case realtime_format::NONE: return out << "none";
    case realtime_format::OFFSET: return out << "offset";
    case realtime_format::TIME: return out << "time";
    default: return out;
  }
}

bool print_journey(journey const& j, std::ostream& out = std::cout,
                   bool local_time = false,
                   realtime_format rt_format = realtime_format::OFFSET);

}  // namespace motis
