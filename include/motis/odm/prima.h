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

enum fixed { kArr, kDep };

struct pos {
  double lat_;
  double lon_;
};

struct direct_ride {
  n::unixtime_t dep_;
  n::unixtime_t arr_;
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
  std::vector<n::routing::start> from_rides_;
  std::vector<n::routing::start> to_rides_;
  std::vector<direct_ride> direct_rides_;
  fixed fixed_;
  capacities cap_;
};

}  // namespace motis::odm