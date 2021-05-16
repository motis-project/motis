#include "motis/core/schedule/time.h"

#include <iomanip>
#include <sstream>

namespace motis {

std::ostream& operator<<(std::ostream& out, time const& t) {
  return !t.valid()
             ? (out << "INVALID")
             : (out << std::setfill('0') << std::setw(3) << t.day_ << "."
                    << std::setfill('0') << std::setw(2) << (t.min_ / 60) << ":"
                    << std::setfill('0') << std::setw(2) << (t.min_ % 60));
}

std::string time::to_str() const {
  std::stringstream out;
  out << *this;
  return out.str();
}

std::string format_time(time const t) {
  if (t == INVALID_TIME) {
    return "INVALID";
  }

  int day = t.day();
  int minutes = t.mam();

  std::ostringstream out;
  out << std::setw(2) << std::setfill('0') << (minutes / 60) << ":"
      << std::setw(2) << std::setfill('0') << (minutes % 60) << "." << day;

  return out.str();
}

}  // namespace motis