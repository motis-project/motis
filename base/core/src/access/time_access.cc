#include "motis/core/access/time_access.h"

#include "motis/core/common/date_time_util.h"
#include "motis/core/common/logging.h"
#include "motis/core/access/error.h"

namespace motis {

unixtime external_schedule_begin(schedule const& sched) {
  return sched.schedule_begin_ + SCHEDULE_OFFSET_MINUTES * 60;
}

unixtime external_schedule_end(schedule const& sched) {
  return sched.schedule_end_;
}

void verify_timestamp(schedule const& sched, unixtime const t) {
  if (t < sched.schedule_begin_ || t >= external_schedule_end(sched)) {
    auto const schedule_begin = external_schedule_begin(sched);
    auto const schedule_end = external_schedule_end(sched);
    LOG(logging::error) << "timestamp not in schedule: " << format_unix_time(t)
                        << " [" << format_unix_time(schedule_begin) << ", "
                        << format_unix_time(schedule_end) << "]";
    throw std::system_error(motis::access::error::timestamp_not_in_schedule);
  }
}

void verify_external_timestamp(schedule const& sched, unixtime const t) {
  if (t < std::min(sched.loaded_begin_, external_schedule_begin(sched)) ||
      t >= std::max(sched.loaded_end_, external_schedule_end(sched))) {
    auto const schedule_begin = external_schedule_begin(sched);
    auto const schedule_end = external_schedule_end(sched);
    LOG(logging::error) << "timestamp not in schedule: " << format_unix_time(t)
                        << " [" << format_unix_time(schedule_begin) << ", "
                        << format_unix_time(schedule_end) << "]";
    throw std::system_error(motis::access::error::timestamp_not_in_schedule);
  }
}

unixtime motis_to_unixtime(schedule const& sched, time t) {
  return motis_to_unixtime(sched.schedule_begin_, t);
}

time unix_to_motistime(schedule const& sched, unixtime t) {
  auto const mt = unix_to_motistime(sched.schedule_begin_, t);
  if (mt == INVALID_TIME) {
    throw std::system_error(motis::access::error::timestamp_not_in_schedule);
  }
  return mt;
}

time motis_time(int const hhmm, int const day_idx, int const timezone_offset) {
  return time(SCHEDULE_OFFSET_MINUTES + day_idx * MINUTES_A_DAY +
              hhmm_to_min(hhmm) - timezone_offset);
}

unixtime unix_time(schedule const& sched, int const hhmm, int const day_idx,
                   int const timezone_offset) {
  return motis_to_unixtime(sched, motis_time(hhmm, day_idx, timezone_offset));
}

}  // namespace motis
