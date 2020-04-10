#pragma once

#include <ctime>

#include "motis/core/schedule/timezone.h"

#include "date/tz.h"

namespace motis::loader {

constexpr auto kAdjust = 60;

timezone create_timezone(int general_offset, int season_offset,
                         int day_idx_schedule_first_day,
                         int day_idx_schedule_last_day,
                         int day_idx_season_first_day,
                         int day_idx_season_last_day,
                         int minutes_after_midnight_season_begin,
                         int minutes_after_midnight_season_end);

time get_event_time(std::time_t schedule_begin, int day_idx, int local_time,
                    timezone const* tz, char const* stop_tz = nullptr,
                    char const* provider_tz = nullptr);

time get_adjusted_event_time(std::time_t schedule_begin, int day_idx,
                             int local_time, timezone const* tz,
                             char const* stop_tz = nullptr,
                             char const* provider_tz = nullptr);

std::pair<time, time> get_event_times(
    std::time_t schedule_begin, int day_idx,  //
    int prev_arr_motis_time, int curr_dep_local_time, int curr_arr_local_time,
    timezone const* tz_dep, char const* dep_stop_tz,
    char const* dep_provider_tz, timezone const* tz_arr,
    char const* arr_stop_tz, char const* arr_provider_tz, bool& adjusted);

}  // namespace motis::loader
