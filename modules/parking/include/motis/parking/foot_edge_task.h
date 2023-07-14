#pragma once

#include <string>
#include <utility>
#include <vector>

#include "motis/core/schedule/station_lookup.h"
#include "motis/parking/parking_lot.h"

namespace motis::parking {

struct foot_edge_task {
  parking_lot const* parking_lot_{};
  std::vector<std::pair<lookup_station, double>> stations_in_radius_;
  std::string const* ppr_profile_;
};

}  // namespace motis::parking
