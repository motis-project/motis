#pragma once

#include <chrono>
#include <vector>

#include "nigiri/routing/start_times.h"

#include "motis-api/motis-api.h"

namespace motis::ep {
struct routing;
}  // namespace motis::ep

namespace motis::odm {

namespace n = nigiri;
using unixtime_t = std::chrono::sys_time<std::chrono::minutes>;

enum fixed { kArr, kDep };

struct pos {
  double lat_;
  double lon_;
};

struct pt_ride {
  n::location_idx_t loc_;
  pos pos_;
  unixtime_t dep_;
  unixtime_t arr_;
};

struct direct_ride {
  unixtime_t dep_;
  unixtime_t arr_;
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
  std::vector<pt_ride> from_rides_;
  std::vector<pt_ride> to_rides_;
  std::vector<direct_ride> direct_rides_;
  fixed fixed_;
  capacities cap_;
};

}  // namespace motis::odm