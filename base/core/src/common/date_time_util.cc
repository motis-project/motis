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

std::string format_unix_time(time_t const t, char const* format) {
  return date::format(
      format, std::chrono::system_clock::time_point{std::chrono::seconds{t}});
}

time_t parse_unix_time(std::string_view s, char const* format) {
  date::local_time<std::chrono::system_clock::duration> tp;
  std::stringstream ss;
  ss << s;
  ss >> date::parse(format, tp);
  return std::chrono::duration_cast<std::chrono::seconds>(tp.time_since_epoch())
      .count();
}

}  // namespace motis