#pragma once

#include <cinttypes>
#include <climits>
#include <iomanip>
#include <sstream>
#include <string>

#define MINUTES_A_DAY 1440
#define SECONDS_A_DAY 86400  // 24 * 60 * 60

namespace motis {

using time = uint16_t;
using duration = uint16_t;

constexpr time INVALID_TIME = USHRT_MAX;
constexpr int SCHEDULE_OFFSET_DAYS = 5;
constexpr time SCHEDULE_OFFSET_MINUTES = MINUTES_A_DAY * SCHEDULE_OFFSET_DAYS;

inline time to_motis_time(int minutes) {
  // plus four days, because the maximum journey duration is 4 days
  // plus one day, because the first valid motis timestamp is MINUTES_A_DAY
  return SCHEDULE_OFFSET_MINUTES + minutes;
}

inline time to_motis_time(int day_index, int minutes) {
  return to_motis_time(day_index * MINUTES_A_DAY + minutes);
}

inline time to_motis_time(int day_index, int hours, int minutes) {
  return to_motis_time(day_index, hours * 60 + minutes);
}

inline std::string format_time(time t) {
  if (t == INVALID_TIME) {
    return "INVALID";
  }

  int day = t / MINUTES_A_DAY;
  int minutes = t % MINUTES_A_DAY;

  std::ostringstream out;
  out << std::setw(2) << std::setfill('0') << (minutes / 60) << ":"
      << std::setw(2) << std::setfill('0') << (minutes % 60) << "." << day;

  return out.str();
}

inline std::time_t motis_to_unixtime(std::time_t schedule_begin, time t) {
  return schedule_begin + t * 60;
}

inline time unix_to_motistime(std::time_t schedule_begin, std::time_t t) {
  if (t < schedule_begin) {
    return INVALID_TIME;
  }
  auto motistime = (t - schedule_begin) / 60;
  if (motistime > INVALID_TIME) {
    return INVALID_TIME;
  }
  return static_cast<time>(motistime);
}

}  // namespace motis
