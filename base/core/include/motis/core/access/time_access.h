#pragma once

#include "motis/core/common/date_time_util.h"
#include "motis/core/common/logging.h"
#include "motis/core/common/unixtime.h"
#include "motis/core/schedule/schedule.h"
#include "motis/core/schedule/time.h"
#include "motis/core/access/error.h"

namespace motis {

constexpr auto const DEFAULT_TIMEZONE_OFFSET = 60;

unixtime external_schedule_begin(schedule const&);

unixtime external_schedule_end(schedule const&);

void verify_timestamp(schedule const&, time_t);

void verify_external_timestamp(schedule const&, time_t);

unixtime motis_to_unixtime(schedule const&, time);

time unix_to_motistime(schedule const&, unixtime);

time motis_time(int hhmm, int day_idx = 0,
                int timezone_offset = DEFAULT_TIMEZONE_OFFSET);

unixtime unix_time(schedule const&, int hhmm, int day_idx = 0,
                   int timezone_offset = DEFAULT_TIMEZONE_OFFSET);

}  // namespace motis
