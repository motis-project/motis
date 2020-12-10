#pragma once

#include <cstdint>
#include <vector>

#include "motis/core/schedule/trip.h"

#include "motis/paxmon/get_load.h"
#include "motis/paxmon/graph.h"
#include "motis/paxmon/paxmon_data.h"

namespace motis::paxmon {

struct edge_load_info {
  edge const* edge_{};
  pax_cdf forecast_cdf_;
  bool updated_{};
  bool possibly_over_capacity_{};
  std::uint16_t expected_passengers_{};
};

struct trip_load_info {
  trip const* trp_{};
  std::vector<edge_load_info> edges_;
};

trip_load_info calc_trip_load_info(paxmon_data const& data, trip const* trp);

}  // namespace motis::paxmon
