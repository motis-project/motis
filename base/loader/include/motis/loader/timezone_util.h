#pragma once

#include <ctime>
#include <optional>
#include <string>

#include "cista/reflection/comparable.h"

#include "date/tz.h"

#include "motis/core/schedule/timezone.h"
#include "motis/hash_map.h"

namespace motis::loader {

constexpr auto kAdjust = 60;

using timezone_name_idx = uint8_t;

struct tz_cache_key {
  CISTA_COMPARABLE()
  timezone_name_idx tz_{0U};
  uint8_t day_idx_{0U};
  time minutes_after_midnight_{0U};
};

struct tz_cache {
  timezone_name_idx lookup_name(std::string_view timezone_name);
  std::string prev_name_;
  timezone_name_idx prev_name_idx_{0U};
  mcd::hash_map<std::string, timezone_name_idx> timezone_name_idx_;
  mcd::hash_map<tz_cache_key, time> cache_;
};

timezone create_timezone(int general_offset, int season_offset,
                         int day_idx_schedule_first_day,
                         int day_idx_schedule_last_day,
                         int day_idx_season_first_day,
                         int day_idx_season_last_day,
                         int minutes_after_midnight_season_begin,
                         int minutes_after_midnight_season_end);

time get_event_time(tz_cache&, std::time_t schedule_begin, int day_idx,
                    int local_time, timezone const* tz,
                    char const* stop_tz = nullptr,
                    char const* provider_tz = nullptr);

time get_adjusted_event_time(tz_cache&, std::time_t schedule_begin, int day_idx,
                             int local_time, timezone const* tz,
                             char const* stop_tz = nullptr,
                             char const* provider_tz = nullptr);

std::pair<time, time> get_event_times(
    tz_cache&, std::time_t schedule_begin, int day_idx,  //
    int prev_arr_motis_time, int curr_dep_local_time, int curr_arr_local_time,
    timezone const* tz_dep, char const* dep_stop_tz,
    char const* dep_provider_tz, timezone const* tz_arr,
    char const* arr_stop_tz, char const* arr_provider_tz, bool& adjusted);

}  // namespace motis::loader
