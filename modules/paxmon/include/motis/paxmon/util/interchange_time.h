#pragma once

#include <cstdint>
#include <optional>

#include "motis/core/schedule/schedule.h"
#include "motis/core/schedule/time.h"

#include "motis/paxmon/transfer_info.h"

namespace motis::paxmon::util {

inline duration get_interchange_time(
    schedule const& sched, std::uint32_t station_idx,
    std::optional<std::uint16_t> arrival_track,
    std::optional<std::uint16_t> departure_track) {
  auto const& st = sched.stations_.at(station_idx);
  if (arrival_track.has_value() && departure_track.has_value()) {
    return static_cast<duration>(st->get_transfer_time_between_tracks(
        arrival_track.value(), departure_track.value()));
  } else {
    return static_cast<duration>(st->transfer_time_);
  }
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
    std::optional<std::uint16_t> arrival_track, std::uint32_t to_station_idx,
    std::optional<std::uint16_t> departure_track) {
  if (from_station_idx == to_station_idx) {
    return transfer_info{get_interchange_time(sched, from_station_idx,
                                              arrival_track, departure_track),
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
