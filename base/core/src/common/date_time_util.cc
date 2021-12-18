#include "motis/core/common/date_time_util.h"

#include "boost/date_time/gregorian/gregorian_types.hpp"
#include "boost/date_time/posix_time/posix_time.hpp"

#include "date/date.h"
#include "date/tz.h"

namespace motis {

unixtime now() {
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

unixtime to_unix_time(boost::posix_time::ptime const& t) {
  boost::posix_time::ptime epoch(boost::gregorian::date(1970, 1, 1));
  return (t - epoch).total_seconds();
}

unixtime to_unix_time(boost::gregorian::date const& date) {
  return to_unix_time(boost::posix_time::ptime(date));
}

unixtime to_unix_time(int year, int month, int day) {
  return to_unix_time(boost::gregorian::date(year, month, day));
}

int hhmm_to_min(int const hhmm) {
  if (hhmm < 0) {
    return hhmm;
  } else {
    return (hhmm / 100) * 60 + (hhmm % 100);
  }
}

std::string format_unix_time(unixtime const t, char const* format) {
  return date::format(
      format, std::chrono::system_clock::time_point{std::chrono::seconds{t}});
}

unixtime parse_unix_time(std::string_view s, char const* format) {
  using namespace date;
  std::stringstream in;
  in << s;
  local_seconds ls;
  std::string tz;
  in >> parse(format, ls, tz);
  auto const time_stamp = date::make_zoned(tz, ls).get_sys_time();
  return std::chrono::duration_cast<std::chrono::seconds>(
             time_stamp.time_since_epoch())
      .count();
}

}  // namespace motis
