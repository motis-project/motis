#pragma once

#include <ctime>
#include <string>

namespace boost::gregorian {
class date;
}  // namespace boost::gregorian

namespace motis {

std::time_t to_unix_time(int year, int month, int day);
int hhmm_to_min(int hhmm);
std::string format_unix_time(time_t, const char* format = "%F %R");
std::time_t to_unix_time(boost::gregorian::date const&);
time_t parse_unix_time(std::string_view, char const* format);

}  // namespace motis
