#include "motis/loader/interval_util.h"

#include <chrono>

#include "utl/verify.h"

#include "motis/core/schedule/schedule.h"
#include "motis/schedule-format/Schedule_generated.h"

#include "date/date.h"

namespace motis::loader {

std::string format_date(time_t const t, char const* format = "%Y-%m-%d") {
  return date::format(
      format, std::chrono::system_clock::time_point{std::chrono::seconds{t}});
}

std::pair<int, int> first_last_days(schedule const& sched,
                                    Interval const* interval) {
  auto first_day = static_cast<int>((sched.schedule_begin_ - interval->from()) /
                                    SECONDS_A_DAY);
  auto last_day = static_cast<int>(
      (sched.schedule_end_ - interval->from()) / SECONDS_A_DAY - 1);

  utl::verify(
      sched.schedule_end_ > sched.schedule_begin_ &&
          interval->from() <= static_cast<uint64_t>(sched.schedule_end_) &&
          interval->to() >= static_cast<uint64_t>(sched.schedule_begin_),
      "schedule (from={}, to={}) out of interval (from={}, to={})",
      format_date(interval->from()), format_date(interval->to()),
      format_date(sched.schedule_begin_), format_date(sched.schedule_end_));

  return {first_day, last_day};
}

}  // namespace motis::loader
