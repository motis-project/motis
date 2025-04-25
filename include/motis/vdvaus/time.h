#pragma once

#include <chrono>

namespace motis::vdvaus {

using vdvaus_time = std::chrono::time_point<std::chrono::system_clock>;

vdvaus_time now();

std::string timestamp(vdvaus_time);

vdvaus_time parse_timestamp(std::string const&);

}  // namespace motis::vdvaus