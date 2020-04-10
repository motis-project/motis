#pragma once

#include <ctime>

#include "motis/core/schedule/time.h"

namespace motis {

struct season {
  int32_t offset_;
  time begin_;
  time end_;
};

struct timezone {
  inline time to_motis_time(int day_idx, int minutes_after_midnight) const {
    auto const minutes_after_schedule_begin =
        motis::to_motis_time(day_idx, minutes_after_midnight);

    auto const is_in_season =
        season_.begin_ != INVALID_TIME &&
        season_.begin_ + general_offset_ <= minutes_after_schedule_begin &&
        minutes_after_schedule_begin <= season_.end_ + season_.offset_;
    auto is_invalid_time =
        is_in_season &&
        minutes_after_schedule_begin < season_.begin_ + general_offset_ + 60;

    if (is_invalid_time) {
      return INVALID_TIME;
    } else {
      return minutes_after_schedule_begin -
             (is_in_season ? season_.offset_ : general_offset_);
    }
  }

  inline std::time_t to_local_time(std::time_t schedule_begin,
                                   time motis_time) const {
    auto const is_in_season = season_.begin_ != INVALID_TIME &&
                              season_.begin_ <= motis_time &&
                              motis_time <= season_.end_;
    auto const active_offset = is_in_season ? season_.offset_ : general_offset_;
    return motis_to_unixtime(schedule_begin,
                             motis_time - MINUTES_A_DAY + active_offset);
  }

  int32_t general_offset_{0};
  season season_{INVALID_TIME, INVALID_TIME, INVALID_TIME};
};

}  // namespace motis
