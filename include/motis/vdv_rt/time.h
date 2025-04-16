#pragma once

#include <chrono>

namespace motis::vdv_rt {

using vdv_rt_time = std::chrono::time_point<std::chrono::system_clock>;

vdv_rt_time now();

std::string timestamp(vdv_rt_time);

vdv_rt_time parse_timestamp(std::string const&);

}  // namespace motis::vdv_rt