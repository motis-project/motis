#pragma once

#include <cstdint>
#include <vector>

#include "motis/core/schedule/trip.h"

#include "motis/paxmon/get_load.h"
#include "motis/paxmon/universe.h"

namespace motis::paxmon {

struct edge_load_info {
  edge const* edge_{};
  pax_pdf forecast_pdf_;
  pax_cdf forecast_cdf_;
  bool updated_{};
  bool possibly_over_capacity_{};
  float probability_over_capacity_{};
  std::uint16_t expected_passengers_{};
};

struct trip_load_info {
  trip const* trp_{};
  std::vector<edge_load_info> edges_;
};

edge_load_info make_edge_load_info(universe const& uv, edge const* e,
                                   pax_pdf&& pdf, pax_cdf&& cdf, bool updated);

edge_load_info make_edge_load_info(universe const& uv, edge const* e,
                                   pax_pdf const& pdf, pax_cdf const& cdf,
                                   bool updated);

trip_load_info calc_trip_load_info(universe const& uv, trip const* trp);

}  // namespace motis::paxmon
