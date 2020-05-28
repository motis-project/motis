#include "motis/loader/timezone_util.h"

#include <algorithm>
#include <iostream>

#include "flatbuffers/flatbuffers.h"

#include "utl/get_or_create.h"
#include "utl/verify.h"

#include "motis/core/common/logging.h"
#include "motis/core/schedule/time.h"
#include "motis/core/schedule/timezone.h"

using namespace motis::logging;

namespace motis::loader {

timezone_name_idx tz_cache::lookup_name(std::string_view timezone_name) {
  if (prev_name_ == timezone_name) {
    return prev_name_idx_;
  } else {
    prev_name_ = timezone_name;
    return prev_name_idx_ =
               utl::get_or_create(timezone_name_idx_, timezone_name,
                                  [&]() { return timezone_name_idx_.size(); });
  }
}

int day_idx(int day_idx_schedule_first_day, int day_idx_schedule_last_day,
            int day_idx_season_event_day) {
  return std::max(
      std::min(day_idx_schedule_last_day, day_idx_season_event_day) -
          day_idx_schedule_first_day,
      0);
}

timezone create_timezone(int general_offset, int season_offset,
                         int day_idx_schedule_first_day,
                         int day_idx_schedule_last_day,
                         int day_idx_season_first_day,
                         int day_idx_season_last_day,
                         int minutes_after_midnight_season_begin,
                         int minutes_after_midnight_season_end) {
  if (day_idx_season_last_day < day_idx_schedule_first_day ||
      day_idx_schedule_last_day < day_idx_season_first_day) {
    return timezone{general_offset};
  }

  time season_begin = 0;
  if (day_idx_schedule_first_day <= day_idx_season_first_day) {
    season_begin = to_motis_time(
        day_idx(day_idx_schedule_first_day, day_idx_schedule_last_day,
                day_idx_season_first_day),
        minutes_after_midnight_season_begin - general_offset);
  }

  time season_end = INVALID_TIME - season_offset;
  if (day_idx_season_last_day <= day_idx_schedule_last_day) {
    season_end = to_motis_time(
        day_idx(day_idx_schedule_first_day, day_idx_schedule_last_day,
                day_idx_season_last_day),
        minutes_after_midnight_season_end - season_offset);
  }

  return {general_offset, {season_offset, season_begin, season_end}};
}

time get_event_time(tz_cache& cache, std::time_t const schedule_begin,
                    int day_idx, int local_time, timezone const* tz,
                    char const* stop_tz, char const* provider_tz) {
  if (tz != nullptr) {
    return tz->to_motis_time(day_idx, local_time);
  } else if (stop_tz != nullptr || provider_tz != nullptr) {
    try {
      auto const tz = stop_tz != nullptr ? stop_tz : provider_tz;
      return utl::get_or_create(
          cache.cache_,
          tz_cache_key{cache.lookup_name(tz), static_cast<uint8_t>(day_idx),
                       static_cast<time>(local_time)},
          [&]() {
            date::time_zone const* zone = date::locate_zone(tz);
            auto const zero =
                date::sys_seconds(std::chrono::seconds(schedule_begin));
            auto const abs_zoned_time = date::zoned_time<std::chrono::seconds>(
                zone,
                date::local_seconds(std::chrono::seconds(
                    schedule_begin +
                    (day_idx + SCHEDULE_OFFSET_DAYS) * MINUTES_A_DAY * 60 +
                    local_time * 60)));
            return static_cast<time>(
                std::chrono::duration_cast<std::chrono::minutes>(
                    abs_zoned_time.get_sys_time() - zero)
                    .count());
          });
    } catch (std::exception const& e) {
      LOG(error)
          << "timezone conversion error (link binary with ianatzdb-res?): "
          << e.what();
      return to_motis_time(day_idx, local_time);
    } catch (...) {
      return to_motis_time(day_idx, local_time);
    }
  } else {
    return to_motis_time(day_idx, local_time);
  }
}

time get_adjusted_event_time(tz_cache& cache, std::time_t const schedule_begin,
                             int day_idx, int local_time, timezone const* tz,
                             char const* stop_tz, char const* provider_tz) {
  auto const t = get_event_time(cache, schedule_begin, day_idx, local_time, tz,
                                stop_tz, provider_tz);
  if (t != INVALID_TIME) {
    return t;
  }

  auto const adjusted =
      get_event_time(cache, schedule_begin, day_idx, local_time + kAdjust, tz,
                     stop_tz, provider_tz);
  utl::verify(adjusted != INVALID_TIME, "adjusted needs to be valid");
  return adjusted;
}

std::pair<time, time> get_event_times(tz_cache& cache,
                                      std::time_t const schedule_begin,
                                      int day_idx,  //
                                      int prev_arr_motis_time,
                                      int curr_dep_local_time,
                                      int curr_arr_local_time,
                                      timezone const* tz_dep,  //
                                      char const* dep_stop_tz,
                                      char const* dep_provider_tz,  //
                                      timezone const* tz_arr,  //
                                      char const* arr_stop_tz,
                                      char const* arr_provider_tz,  //
                                      bool& adjusted) {
  auto const offset = adjusted ? kAdjust : 0;
  auto const total_offset = offset + kAdjust;
  auto const prev_adjusted = adjusted;

  auto dep_motis_time = get_event_time(cache, schedule_begin, day_idx,
                                       curr_dep_local_time + offset, tz_dep,
                                       dep_stop_tz, dep_provider_tz);
  auto arr_motis_time = get_event_time(cache, schedule_begin, day_idx,
                                       curr_arr_local_time + offset, tz_arr,
                                       arr_stop_tz, arr_provider_tz);

  if (prev_arr_motis_time > dep_motis_time || dep_motis_time == INVALID_TIME) {
    dep_motis_time = get_event_time(cache, schedule_begin, day_idx,
                                    curr_dep_local_time + total_offset, tz_dep,
                                    dep_stop_tz, dep_provider_tz);
    arr_motis_time = get_event_time(cache, schedule_begin, day_idx,
                                    curr_arr_local_time + total_offset, tz_arr,
                                    arr_stop_tz, arr_provider_tz);
    adjusted = true;
    utl::verify(!prev_adjusted, "double adjustment of time offset [case 1]");
  }

  if (arr_motis_time == INVALID_TIME) {
    arr_motis_time = get_event_time(cache, schedule_begin, day_idx,
                                    curr_arr_local_time + total_offset, tz_arr,
                                    arr_stop_tz, arr_provider_tz);
    adjusted = true;
    utl::verify(!prev_adjusted, "double adjustment of time offset [case 2]");
  }

  if (dep_motis_time == INVALID_TIME) {
    dep_motis_time = get_event_time(cache, schedule_begin, day_idx,
                                    curr_dep_local_time + total_offset, tz_dep,
                                    dep_stop_tz, dep_provider_tz);
    adjusted = true;
    utl::verify(!prev_adjusted, "double adjustment of time offset [case 3]");
  }

  if (arr_motis_time < dep_motis_time) {
    arr_motis_time = get_event_time(cache, schedule_begin, day_idx,
                                    curr_arr_local_time + total_offset, tz_arr,
                                    arr_stop_tz, arr_provider_tz);
    adjusted = true;
    utl::verify(!prev_adjusted, "double adjustment of time offset [case 4]");
  }

  return std::make_pair(dep_motis_time, arr_motis_time);
}

}  // namespace motis::loader
