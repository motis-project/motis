#include "motis/vdv_rt/time.h"

#include "date/date.h"

namespace motis::vdv_rt {

sys_time now() { return std::chrono::system_clock::now(); }

std::string timestamp(const sys_time t) {
  return date::format("%FT%T",
                      std::chrono::time_point_cast<std::chrono::seconds>(t));
}

sys_time parse_timestamp(std::string const& str) {
  sys_time parsed;
  auto ss = std::stringstream{str};
  ss >> date::parse("%FT%T", parsed);
  return parsed;
}

}  // namespace motis::vdv_rt