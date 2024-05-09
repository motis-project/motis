#pragma once

#include "motis/core/common/date_time_util.h"
#include "motis/core/common/logging.h"
#include "motis/core/common/unixtime.h"
#include "motis/core/schedule/time.h"
#include "motis/core/access/error.h"

namespace motis {

constexpr auto const DEFAULT_TIMEZONE_OFFSET = 60;

time motis_time(int hhmm, int day_idx = 0,
                int timezone_offset = DEFAULT_TIMEZONE_OFFSET);

}  // namespace motis
