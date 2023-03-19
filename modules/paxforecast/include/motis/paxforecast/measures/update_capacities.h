#pragma once

#include <string>
#include <vector>

#include "motis/core/schedule/time.h"

namespace motis::paxforecast::measures {

struct update_capacities {
  time time_{};
  std::vector<std::string> file_contents;
  bool remove_existing_trip_capacities_{};
  bool remove_existing_category_capacities_{};
  bool remove_existing_vehicle_capacities_{};
  bool remove_existing_trip_formations_{};
  bool track_trip_updates_{};
};

}  // namespace motis::paxforecast::measures
