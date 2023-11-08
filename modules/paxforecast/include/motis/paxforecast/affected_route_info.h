#pragma once

#include <cstdint>
#include <optional>

#include "motis/paxmon/broken_transfer_info.h"
#include "motis/paxmon/localization.h"

namespace motis::paxforecast {

// probability is stored because it may be changed before the simulation begins
// passenger count is stored for convenience
struct passenger_group_with_route_and_probability {
  motis::paxmon::passenger_group_with_route pgwr_{};
  float probability_{};
  std::uint16_t passengers_{};
};

struct affected_route_info {
  inline bool broken() const { return broken_transfer_info_.has_value(); }

  passenger_group_with_route_and_probability pgwrap_{};
  unsigned destination_station_id_{};
  motis::paxmon::passenger_localization loc_now_{};
  motis::paxmon::passenger_localization loc_broken_{};
  std::optional<motis::paxmon::broken_transfer_info> broken_transfer_info_;
  std::uint32_t alts_now_{};  // index -> alternatives set
  std::uint32_t alts_broken_{};  // index -> alternatives set
};

}  // namespace motis::paxforecast
