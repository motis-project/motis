#include "motis/core/common/date_time_util.h"

#include "boost/date_time/gregorian/gregorian_types.hpp"
#include "boost/date_time/posix_time/posix_time.hpp"

#include "date/date.h"

namespace motis {

std::time_t to_unix_time(boost::posix_time::ptime const& t) {
  boost::posix_time::ptime epoch(boost::gregorian::date(1970, 1, 1));
  return (t - epoch).total_seconds();
}

std::time_t to_unix_time(boost::gregorian::date const& date) {
  return to_unix_time(boost::posix_time::ptime(date));
}

std::time_t to_unix_time(int year, int month, int day) {
  return to_unix_time(boost::gregorian::date(year, month, day));
}

int hhmm_to_min(int const hhmm) {
  if (hhmm < 0) {
    return hhmm;
  } else {
    return (hhmm / 100) * 60 + (hhmm % 100);
  }
}

std::string format_unixtime(time_t const t, char const* format) {
  return date::format(
      format, std::chrono::system_clock::time_point{std::chrono::seconds{t}});
}

}  // namespace motis