#pragma once

#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>

#include "motis/core/schedule/connection.h"

namespace motis::raptor {

using time = int16_t;
using time8 = int8_t;
using time32 = int32_t;

using station_id = int32_t;
using route_id = uint32_t;
using footpath_id = int32_t;

using motis_id = int32_t;

using footpath_count = uint16_t;
using route_count = uint16_t;

using trip_id = uint16_t;
using trip_count = uint16_t;
using stop_count = uint16_t;

using stop_offset = uint16_t;

using stop_times_index = uint32_t;
using route_stops_index = uint32_t;
using stop_routes_index = uint32_t;
using footpaths_index = uint32_t;

using raptor_round = uint8_t;
using transfers = uint8_t;

using earliest_arrivals = std::vector<time>;

template <typename T>
constexpr T min_value = std::numeric_limits<T>::min();
template <typename T>
constexpr T max_value = std::numeric_limits<T>::max();
template <typename T>
constexpr T invalid = max_value<T>;
// Template specializations in raptor_timetable.cc

template <typename T>
constexpr auto valid(T const& value) {
  return value != invalid<T>;
}

constexpr raptor_round max_transfers = 6;
constexpr raptor_round max_trips = max_transfers + 1;
constexpr raptor_round max_raptor_round = max_trips + 1;

// TODO(julian) implement
constexpr time max_traveltime = 1440;

struct raptor_stop {
  raptor_stop() = delete;
  raptor_stop(footpath_count const fc, route_count const rc,
              footpaths_index const it, stop_routes_index const isr)
      : footpath_count_(fc),
        route_count_(rc),
        index_to_transfers_(it),
        index_to_stop_routes_(isr) {}

  footpath_count footpath_count_;
  route_count route_count_;
  footpaths_index index_to_transfers_;
  stop_routes_index index_to_stop_routes_;
};

struct raptor_route {
  raptor_route() = delete;
  raptor_route(trip_count const tc, stop_count const sc,
               stop_times_index const sti, route_stops_index const rsi,
               time const st)
      : trip_count_(tc),
        stop_count_(sc),
        index_to_stop_times_(sti),
        index_to_route_stops_(rsi),
        stand_time_(st) {}

  trip_count trip_count_;
  stop_count stop_count_;
  stop_times_index index_to_stop_times_;
  route_stops_index index_to_route_stops_;
  time stand_time_;
};

struct stop_time {
  time arrival_{invalid<decltype(arrival_)>};
  time departure_{invalid<decltype(departure_)>};
};

struct raptor_footpath {
  raptor_footpath() = delete;
  raptor_footpath(station_id const to, time const dur)
      : to_(to), duration_(dur) {}
  station_id to_ : 24;
  time8 duration_;
};

struct raptor_incoming_footpath {
  raptor_incoming_footpath() = delete;
  raptor_incoming_footpath(station_id const from, time const duration)
      : from_(from), duration_(duration) {}
  station_id from_ : 24;
  time8 duration_;
};

struct raptor_timetable {
  raptor_timetable() = default;
  raptor_timetable(raptor_timetable const&) = default;

  raptor_timetable& operator=(raptor_timetable const&) = delete;
  raptor_timetable(raptor_timetable&&) = delete;
  raptor_timetable& operator=(raptor_timetable const&&) = delete;

  ~raptor_timetable() = default;

  std::vector<raptor_stop> stops_;
  std::vector<raptor_route> routes_;
  std::vector<raptor_footpath> footpaths_;

  std::vector<stop_time> stop_times_;
  std::vector<station_id> route_stops_;
  std::vector<route_id> stop_routes_;

  // TODO(julian) implement
  // Needed for initialization
  // duration of the footpaths INCLUDE transfer time from the departure station
  // std::vector<std::vector<raptor_footpath>> initialization_footpaths_;

  // Needed for the reconstruction
  // duration REDUCED by the transfer times from the departure station
  std::vector<std::vector<raptor_incoming_footpath>> incoming_footpaths_;

  [[nodiscard]] auto stop_count() const {  // subtract the sentinel
    return static_cast<station_id>(stops_.size() - 1);
  }

  [[nodiscard]] auto route_count() const {  // subtract the sentinel
    return static_cast<route_id>(routes_.size() - 1);
  }

  [[nodiscard]] auto footpath_count() const {
    return static_cast<footpath_id>(footpaths_.size());
  }
};

struct raptor_schedule {
  raptor_schedule() = default;
  raptor_schedule(raptor_schedule const&) = delete;

  raptor_schedule& operator=(raptor_schedule const&) = delete;
  raptor_schedule(raptor_schedule&&) = delete;
  raptor_schedule& operator=(raptor_schedule const&&) = delete;

  ~raptor_schedule() = default;

  std::unordered_map<std::string, station_id> eva_to_raptor_id_;
  std::vector<std::string> raptor_id_to_eva_;
  std::vector<unsigned> station_id_to_index_;
  std::vector<time> transfer_times_;

  std::vector<std::vector<station_id>> equivalent_stations_;

  // for every station the departure events of the station
  // and the stations reachable by foot
  std::vector<std::vector<time>> departure_events_;

  // uses the same indexing scheme as the stop times vector in the timetable,
  // but the first entry for every trip is a nullptr!
  // since #stop_times(trip) = #lcons(trip) + 1
  std::vector<light_connection const*> lcon_ptr_;

  // duration of the footpaths INCLUDE transfer time from the departure station
  std::vector<std::vector<raptor_footpath>> initialization_footpaths_;
};

}  // namespace motis::raptor
