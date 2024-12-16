#pragma once

#include <chrono>
#include <vector>

namespace motis::odm {

using unixtime_t = std::chrono::sys_time<std::chrono::minutes>;

struct pos {
  double lat_;
  double lon_;
};

struct stop_times {
  pos pos_;
  std::vector<unixtime_t> times_;
};

struct capacities {
  std::uint16_t wheelchairs_;
  std::uint16_t bikes_;
  std::uint16_t passengers_;
  std::uint16_t luggage_;
};

struct prima_req {
  pos from_;
  pos to_;
  stop_times from_stops_;
  stop_times to_stops_;
  std::vector<unixtime_t> direct_;
  bool start_fixed_;
  capacities cap_;
};

}  // namespace motis::odm