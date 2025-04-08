#pragma once

#include <chrono>

namespace motis::vdv_rt {

using sys_time = std::chrono::time_point<std::chrono::system_clock>;

sys_time now();

std::string timestamp(sys_time);

sys_time parse_timestamp(std::string const&);

}  // namespace motis::vdv_rt