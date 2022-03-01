#pragma once

#include <ctime>

#include "motis/module/context/motis_spawn.h"

#include "motis/ridesharing/connection.h"
#include "motis/ridesharing/leg_idx.h"
#include "motis/ridesharing/lift_key.h"
#include "motis/ridesharing/passenger.h"
#include "motis/ridesharing/routing_result.h"

#include <memory>
#include <string>
#include <unordered_map>

#include "geo/latlng.h"

namespace motis::ridesharing {

std::vector<routing_result> route(std::vector<geo::latlng> const&);

struct lift {
  lift(geo::latlng const& start, geo::latlng const& destination,
       std::time_t const& lift_start_time, int driver_id,
       uint16_t max_passengers);
  lift(std::vector<geo::latlng> const& waypoints,
       std::vector<routing_result> rrs, double max_total_duration,
       std::time_t const& lift_start_time, int driver_id,
       uint16_t max_passengers, std::vector<passenger> const& pas);
  std::vector<routing_result> route(std::vector<geo::latlng> const& waypoints);
  void initial_station_routing(
      std::vector<std::pair<geo::latlng, int>> const& station_locations);
  bool process_booking(
      passenger const& pas, geo::latlng const& pick_up, leg_idx from_leg,
      geo::latlng const& drop_off, leg_idx to_leg,
      std::vector<std::pair<geo::latlng, int>> const& station_locations);
  bool is_potential(geo::latlng const& loc, leg_idx index);
  void add_routing(
      bool to_stations,
      std::vector<std::pair<geo::latlng, int>> const& station_locations,
      leg_idx index);
  void add_routing(
      std::vector<ctx::future_ptr<motis::module::ctx_data, void>>& futures,
      std::mutex& mutex, bool to_stations,
      std::vector<std::pair<geo::latlng, int>> const& station_locations,
      leg_idx index);
  void recompute_routings(
      std::vector<std::pair<geo::latlng, int>> const& station_locations);

  bool add_waypoints(geo::latlng const& pick_up, leg_idx pick_up_on_leg_index,
                     geo::latlng const& drop_off,
                     leg_idx drop_off_on_leg_index);
  bool remove_booking(int passenger_id);

  double max_detour(double duration) {
    if (duration < 30 * 60) {
      return 0.3;
    } else if (duration > 120 * 60) {
      return 0.1;
    } else {
      double proximity = (duration / 60 - 30) / 90;
      return 0.3 * (1 - proximity) + 0.1 * proximity;
    }
  }

  std::vector<geo::latlng> waypoints_;
  std::vector<routing_result> rrs_;
  double max_total_duration_;
  routing_result current_total_routing_result_;
  std::time_t t_;
  int driver_id_;
  uint16_t max_passengers_;
  std::vector<passenger> passengers_;
  std::vector<std::unordered_map<unsigned, routing_result>>
      to_routings_;  // to_station_routings
  std::vector<std::unordered_map<unsigned, routing_result>>
      from_routings_;  // from_station_routings
  uint64_t booking_routing_time_{};
  uint64_t creation_routing_time_{};
};

bool operator<(lift const& l1, lift const& l2);
bool operator<(lift_key const& lk, lift const& l);
bool operator<(lift const& l, lift_key const& lk);
bool operator<(lift_key const& l, lift_key const& lk);
bool operator==(lift_key const& l1, lift_key const& l2);
bool operator==(lift const& l1, lift const& l2);
bool operator==(lift const& l1, lift_key const& l2);

}  // namespace motis::ridesharing