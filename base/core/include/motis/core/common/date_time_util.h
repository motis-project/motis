#pragma once

#include <ctime>
#include <string>

namespace boost::gregorian {
class date;
}

namespace motis {

std::time_t to_unix_time(int year, int month, int day);
int hhmm_to_min(int hhmm);
std::string format_unixtime(time_t t);
std::time_t to_unix_time(boost::gregorian::date const&);

}  // namespace motis
