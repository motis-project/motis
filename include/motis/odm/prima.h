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
  std::uint8_t wheelchairs_;
  std::uint8_t bikes_;
  std::uint8_t passengers_;
  std::uint8_t luggage_;
};

struct prima_state {
  pos from_;
  pos to_;
  std::vector<stop_times> from_stops_;
  std::vector<stop_times> to_stops_;
  std::vector<unixtime_t> direct_;
  bool start_fixed_;
  capacities cap_;
};

}  // namespace motis::odm