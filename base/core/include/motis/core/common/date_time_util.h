#pragma once

#include <ctime>
#include <string>

#include "motis/core/common/unixtime.h"

namespace boost::gregorian {
class date;
}  // namespace boost::gregorian

namespace motis {

unixtime now();
unixtime to_unix_time(int year, int month, int day);
int hhmm_to_min(int hhmm);
std::string format_unix_time(unixtime, const char* format = "%F %R");
unixtime to_unix_time(boost::gregorian::date const&);
unixtime parse_unix_time(std::string_view, char const* format = "%F %R %Z");

}  // namespace motis
