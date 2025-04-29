#include "motis/vdvaus/time.h"

#include "date/date.h"

namespace motis::vdvaus {

vdvaus_time now() { return std::chrono::system_clock::now(); }

std::string timestamp(const vdvaus_time t) {
  return date::format("%FT%T",
                      std::chrono::time_point_cast<std::chrono::seconds>(t));
}

vdvaus_time parse_timestamp(std::string const& str) {
  vdvaus_time parsed;
  auto ss = std::stringstream{str};
  ss >> date::parse("%FT%T", parsed);
  return parsed;
}

}  // namespace motis::vdvaus