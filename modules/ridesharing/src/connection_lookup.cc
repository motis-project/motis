#include "motis/ridesharing/connection_lookup.h"

#include "motis/module/context/motis_call.h"

#include "motis/protocol/Message_generated.h"
#include "motis/ridesharing/lift.h"

#include <algorithm>
#include <numeric>

#include "utl/to_vec.h"

using namespace flatbuffers;
using namespace motis::module;
using namespace motis::osrm;

namespace motis::ridesharing {

constexpr int MINIMUM_WAGE = 15;  //.3... ct/min
constexpr int PENDLERPAUSCHALE = 30;  // ct/km
constexpr int PRICE_COMMON = 10;  // ct/min
constexpr int PRICE_DETOUR = 5;  // ct/min

bool lift_economic(connection_eval const& ce) {
  auto const detour = ce.from_leg_detour_ + ce.to_leg_detour_;
  return MINIMUM_WAGE * detour.duration_ / 60 +
             PENDLERPAUSCHALE * detour.distance_ / 1000 <
         calculate_price(detour.duration_, ce.common_.duration_);
}

uint16_t calculate_price(double detour_duration, double common_duration) {
  return std::max(200.0, PRICE_DETOUR * detour_duration / 60 +
                             PRICE_COMMON * common_duration / 60 + 0.5);
}

leg_idx waypoint_index(lift const& li, geo::latlng const& loc) {
  return std::distance(
      li.waypoints_.begin(),
      std::find(li.waypoints_.begin(), li.waypoints_.end(), loc));
}

bool valid_passenger_count(lift const& li, uint16_t const passenger_count,
                           leg_idx const from_leg, leg_idx const to_leg) {
  auto seats_per_leg =
      std::vector<uint16_t>(li.rrs_.size(), li.max_passengers_);
  for (auto const& pas : li.passengers_) {
    auto const pic = waypoint_index(li, pas.pick_up_);
    auto const dro = waypoint_index(li, pas.drop_off_);
    for (auto i = pic; i < dro; i++) {
      seats_per_leg[i] -= pas.passenger_count_;
    }
  }
  return std::all_of(
      seats_per_leg.begin() + from_leg, seats_per_leg.begin() + to_leg,
      [&](auto const& seats) { return seats > passenger_count; });
}

bool passenger_constraints(lift const& li, leg_idx from_leg, leg_idx to_leg,
                           connection_eval const& ce) {
  return std::all_of(
      li.passengers_.begin(), li.passengers_.end(), [&](auto& pas) {
        leg_idx drop_off_idx = waypoint_index(li, pas.drop_off_);
        auto leeway =
            pas.required_arrival_ - li.t_ -
            std::accumulate(
                li.rrs_.begin(), li.rrs_.begin() + drop_off_idx, 0.0,
                [](double acc, auto& rr) { return acc + rr.duration_; });
        if (from_leg < drop_off_idx) {
          leeway -= ce.from_leg_detour_.duration_;
          if (to_leg < drop_off_idx) {
            leeway -= ce.to_leg_detour_.duration_;
          }
        }
        return leeway >= 0;
      });
}

bool valid_edge(lift const& li, leg_idx const from_leg, leg_idx const to_leg,
                connection_eval const& ce, uint16_t passenger_count) {
  return lift_economic(ce) &&
         li.max_total_duration_ >= li.current_total_routing_result_.duration_ +
                                       ce.from_leg_detour_.duration_ +
                                       ce.to_leg_detour_.duration_ &&
         valid_passenger_count(li, passenger_count, from_leg, to_leg) &&
         passenger_constraints(li, from_leg, to_leg, ce);
}

void add_connections(
    lift const& li, connection_map& from_to,
    std::vector<std::vector<routing_result>> const& routing_table,
    leg_idx from_leg, leg_idx to_leg) {
  for (auto& [i, rr_from_leg] : li.to_routings_[from_leg]) {
    std::vector<unsigned> targets;
    for (auto& [j, rr_to_leg] : li.from_routings_[to_leg]) {
      if (i == j || !li.to_routings_[to_leg].count(j) ||
          !li.from_routings_[from_leg].count(i)) {
        continue;
      }
      auto const from_leg_detour =
          from_leg == to_leg ? rr_from_leg + routing_table[i][j] + rr_to_leg -
                                   li.rrs_[from_leg]
                             : rr_from_leg + li.from_routings_[from_leg].at(i) -
                                   li.rrs_[from_leg];
      auto const to_leg_detour =
          from_leg == to_leg
              ? routing_result{0, 0}
              : li.to_routings_[to_leg].at(j) + rr_to_leg - li.rrs_[to_leg];
      auto const common = routing_table[i][j];
      if (valid_edge(li, from_leg, to_leg,
                     {common, from_leg_detour, to_leg_detour}, 1)) {
        targets.push_back(j);
      }
    }
    if (targets.size() > 0) {
      std::sort(targets.begin(), targets.end());
      connection c{targets, from_leg, to_leg};
      from_to.emplace(i, c);
    }
  }
}

connection_lookup setup_acceptable_stations(
    lift const& li,
    std::vector<std::vector<routing_result>> const& routing_table) {
  auto from_to = connection_map{};
  for (leg_idx i = 0; i < li.rrs_.size(); i++) {
    for (leg_idx j = i; j < li.rrs_.size(); j++) {
      add_connections(li, from_to, routing_table, i, j);
    }
  }

  return connection_lookup{from_to, setup_to_from(from_to)};
}

// Invert Mapping
connection_map setup_to_from(connection_map const& from_to) {
  auto to_from = connection_map{};
  for (auto& entry : from_to) {
    for (auto to : entry.second.stations_) {
      bool inserted = false;
      auto found = to_from.equal_range(to);
      for (auto it = found.first; it != found.second; ++it) {
        if (it->second.from_leg_ == entry.second.from_leg_ &&
            it->second.to_leg_ == entry.second.to_leg_) {
          it->second.stations_.push_back(entry.first);
          inserted = true;
        }
      }
      if (!inserted) {
        connection from{
            {entry.first}, entry.second.from_leg_, entry.second.to_leg_};
        to_from.emplace(to, from);
      }
    }
  }
  for (auto& e : to_from) {
    std::sort(e.second.stations_.begin(), e.second.stations_.end());
  }
  return to_from;
}

}  // namespace motis::ridesharing