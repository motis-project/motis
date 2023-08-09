#pragma once

#include <string>
#include <vector>

#include "motis/core/schedule/time.h"

namespace motis::paxforecast::measures {

struct update_capacities {
  time time_{};
  std::vector<std::string> file_contents_;
  bool remove_existing_trip_capacities_{};
  bool remove_existing_category_capacities_{};
  bool remove_existing_vehicle_capacities_{};
  bool remove_existing_trip_formations_{};
  bool remove_existing_gattung_capacities_{};
  bool remove_existing_baureihe_capacities_{};
  bool remove_existing_vehicle_group_capacities_{};
  bool remove_existing_overrides_{};
  bool track_trip_updates_{};
};

}  // namespace motis::paxforecast::measures
