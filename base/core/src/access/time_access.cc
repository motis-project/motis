#include "motis/core/access/time_access.h"

#include "motis/core/common/date_time_util.h"
#include "motis/core/common/logging.h"
#include "motis/core/access/error.h"

namespace motis {

std::time_t external_schedule_begin(schedule const& sched) {
  return sched.schedule_begin_ + SCHEDULE_OFFSET_MINUTES * 60;
}

std::time_t external_schedule_end(schedule const& sched) {
  return sched.schedule_end_;
}

void verify_timestamp(schedule const& sched, time_t t) {
  if (t < sched.schedule_begin_ || t >= external_schedule_end(sched)) {
    auto const schedule_begin = external_schedule_begin(sched);
    auto const schedule_end = external_schedule_end(sched);
    LOG(logging::error) << "timestamp not in schedule: " << format_unix_time(t)
                        << " [" << format_unix_time(schedule_begin) << ", "
                        << format_unix_time(schedule_end) << "]";
    throw std::system_error(motis::access::error::timestamp_not_in_schedule);
  }
}

void verify_external_timestamp(schedule const& sched, time_t t) {
  if (t < std::min(sched.first_event_schedule_time_,
                   external_schedule_begin(sched)) ||
      t >= std::max(sched.last_event_schedule_time_,
                    external_schedule_end(sched))) {
    auto const schedule_begin = external_schedule_begin(sched);
    auto const schedule_end = external_schedule_end(sched);
    LOG(logging::error) << "timestamp not in schedule: " << format_unix_time(t)
                        << " [" << format_unix_time(schedule_begin) << ", "
                        << format_unix_time(schedule_end) << "]";
    throw std::system_error(motis::access::error::timestamp_not_in_schedule);
  }
}

std::time_t motis_to_unixtime(schedule const& sched, time t) {
  return motis_to_unixtime(sched.schedule_begin_, t);
}

time unix_to_motistime(schedule const& sched, std::time_t t) {
  auto const mt = unix_to_motistime(sched.schedule_begin_, t);
  if (mt == INVALID_TIME) {
    throw std::system_error(motis::access::error::timestamp_not_in_schedule);
  }
  return mt;
}

time motis_time(int const hhmm, int const day_idx, int const timezone_offset) {
  return SCHEDULE_OFFSET_MINUTES + day_idx * MINUTES_A_DAY + hhmm_to_min(hhmm) -
         timezone_offset;
}

std::time_t unix_time(schedule const& sched, int const hhmm, int const day_idx,
                      int const timezone_offset) {
  return motis_to_unixtime(sched, motis_time(hhmm, day_idx, timezone_offset));
}

}  // namespace motis
