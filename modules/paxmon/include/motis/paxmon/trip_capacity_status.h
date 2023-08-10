#pragma once

#include "motis/paxmon/capacity_data.h"

namespace motis::paxmon {

struct trip_capacity_status {
  bool has_trip_formation_{};
  bool has_capacity_for_all_sections_{};
  bool has_capacity_for_some_sections_{};
  capacity_source worst_source_{capacity_source::UNKNOWN};
};

}  // namespace motis::paxmon
