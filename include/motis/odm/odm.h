#pragma once

#include <vector>

#include "nigiri/types.h"

namespace nigiri {
struct timetable;
struct rt_timetable;
}  // namespace nigiri

namespace motis::odm {

std::vector<std::vector<nigiri::unixtime_t>> get_events(
    nigiri::timetable const&, nigiri::rt_timetable const&);

}