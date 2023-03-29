#pragma once

#include <ctime>
#include <optional>

#include "motis/core/schedule/time.h"
#include "motis/vector.h"

namespace motis {

struct season {
  int32_t offset_{INVALID_TIME};
  time begin_{INVALID_TIME};
  time end_{INVALID_TIME};
};

struct timezone {
  time to_motis_time(int day_idx, int minutes_after_midnight) const {
    auto const minutes_after_schedule_begin =
        motis::to_motis_time(day_idx, minutes_after_midnight);

    auto const is_in_season = [&](season const& s) {
      return s.begin_ != INVALID_TIME &&
             s.begin_ + general_offset_ <= minutes_after_schedule_begin &&
             minutes_after_schedule_begin <= s.end_ + s.offset_;
    };

    auto const it =
        std::find_if(begin(seasons_), end(seasons_),
                     [&](season const& s) { return is_in_season(s); });
    if (it == end(seasons_)) {
      return minutes_after_schedule_begin - general_offset_;
    }

    auto const active_season = *it;
    auto const is_invalid_time = minutes_after_schedule_begin <
                                 active_season.begin_ + general_offset_ + 60;
    if (is_invalid_time) {
      return INVALID_TIME;
    }

    return minutes_after_schedule_begin - active_season.offset_;
  }

  std::time_t to_local_time(std::time_t schedule_begin, time motis_time) const {
    auto const active_season = get_active_season(motis_time);
    auto const active_offset =
        active_season.has_value() ? active_season->offset_ : general_offset_;
    return motis_to_unixtime(schedule_begin,
                             motis_time - MINUTES_A_DAY + active_offset);
  }

  std::optional<season> get_active_season(time const motis_time) const {
    auto const it =
        std::find_if(begin(seasons_), end(seasons_), [&](season const& s) {
          return s.begin_ != INVALID_TIME && s.begin_ <= motis_time &&
                 motis_time <= s.end_;
        });
    return it == end(seasons_) ? std::nullopt : std::optional{*it};
  }

  int32_t general_offset_{0};
  mcd::vector<season> seasons_;
};

}  // namespace motis
