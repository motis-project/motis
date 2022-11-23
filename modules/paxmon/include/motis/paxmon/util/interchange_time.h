#pragma once

#include <cstdint>
#include <optional>

#include "motis/core/schedule/schedule.h"
#include "motis/core/schedule/time.h"
#include "motis/core/access/trip_section.h"

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

inline std::optional<transfer_info> get_transfer_info(
    schedule const& sched, access::trip_section const& arrival_section,
    access::trip_section const& departure_section) {
  // check if shared trip in arrival + departure section
  auto const& arrival_trips =
      *sched.merged_trips_.at(arrival_section.lcon().trips_);
  auto const& departure_trips =
      *sched.merged_trips_.at(departure_section.lcon().trips_);

  for (auto const& trp : arrival_trips) {
    if (std::find(begin(departure_trips), end(departure_trips), trp) !=
        end(departure_trips)) {
      return transfer_info{0, transfer_info::type::MERGE};
    }
  }

  // check if through edge from arrival to departure section
  auto const* arr_node = arrival_section.to_node();
  auto const* dep_node = departure_section.from_node();
  for (auto const& e : arr_node->edges_) {
    if (e.type() == ::motis::edge::THROUGH_EDGE && e.to_ == dep_node) {
      return transfer_info{0, transfer_info::type::THROUGH};
    }
  }

  return get_transfer_info(sched, arrival_section.to_station_id(),
                           arrival_section.lcon().full_con_->a_track_,
                           departure_section.from_station_id(),
                           departure_section.lcon().full_con_->d_track_);
}

}  // namespace motis::paxmon::util
