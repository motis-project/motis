#include "motis/loader/timezone_util.h"

#include <algorithm>

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
    return timezone{.general_offset_ = general_offset, .season_ = season{}};
  }

  auto season_begin = time{0U};
  if (day_idx_schedule_first_day <= day_idx_season_first_day) {
    season_begin = to_motis_time(
        day_idx(day_idx_schedule_first_day, day_idx_schedule_last_day,
                day_idx_season_first_day),
        minutes_after_midnight_season_begin - general_offset);
  }

  auto season_end = INVALID_TIME - season_offset;
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
                             int local_time, int day_idx, timezone const* tz,
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

bool is_local_time_in_season(int const day_idx, int minutes_after_midnight,
                             timezone const* tz) {
  auto const minutes_after_schedule_begin =
      motis::to_motis_time(day_idx, minutes_after_midnight);
  return tz->season_.begin_ != INVALID_TIME &&
         tz->season_.begin_ + tz->general_offset_ <=
             minutes_after_schedule_begin &&
         minutes_after_schedule_begin <= tz->season_.end_ + tz->season_.offset_;
}

}  // namespace motis::loader
