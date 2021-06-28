#pragma once

#include <cstdint>
#include <optional>

#include "motis/core/schedule/schedule.h"
#include "motis/core/schedule/time.h"

#include "motis/paxmon/transfer_info.h"

namespace motis::paxmon::util {

inline duration get_interchange_time(schedule const& sched,
                                     std::uint32_t station_idx) {
  return static_cast<duration>(sched.stations_.at(station_idx)->transfer_time_);
}

inline std::optional<duration> get_footpath_duration(
    schedule const& sched, std::uint32_t from_station_idx,
    std::uint32_t to_station_idx) {
  for (auto const& fp :
       sched.stations_[from_station_idx]->outgoing_footpaths_) {
    if (fp.to_station_ == to_station_idx) {
      return {static_cast<duration>(fp.duration_)};
    }
  }
  return {};
}

inline std::optional<transfer_info> get_transfer_info(
    schedule const& sched, std::uint32_t from_station_idx,
    std::uint32_t to_station_idx) {
  if (from_station_idx == to_station_idx) {
    return transfer_info{get_interchange_time(sched, from_station_idx),
                         transfer_info::type::SAME_STATION};
  } else {
    if (auto const duration =
            get_footpath_duration(sched, from_station_idx, to_station_idx);
        duration.has_value()) {
      return transfer_info{duration.value(), transfer_info::type::FOOTPATH};
    } else {
      return {};
    }
  }
}

}  // namespace motis::paxmon::util
