#include "motis/ridesharing/lift.h"

#include "motis/core/common/timing.h"

#include "motis/module/context/motis_call.h"
#include "motis/module/context/motis_spawn.h"

#include "motis/protocol/Message_generated.h"
#include "motis/ridesharing/connection_lookup.h"

#include <numeric>

#include "utl/to_vec.h"
#include "utl/zip.h"

using namespace flatbuffers;
using namespace motis::module;
using namespace motis::osrm;

namespace motis::ridesharing {

std::vector<routing_result> lift::route(
    std::vector<geo::latlng> const& waypoints) {
  std::vector<Position> pos = utl::to_vec(waypoints, [](auto& w) {
    return Position{w.lat_, w.lng_};
  });
  MOTIS_START_TIMING(routing_time);
  message_creator mc;
  mc.create_and_finish(MsgContent_OSRMRouteRequest,
                       CreateOSRMRouteRequest(mc, mc.CreateString("car"),
                                              mc.CreateVectorOfStructs(pos))
                           .Union(),
                       "/osrm/route");
  try {
    auto const osrm_msg = motis_call(make_msg(mc))->val();
    auto const osrm_resp = motis_content(OSRMRouteResponse, osrm_msg);

    auto const ret = utl::to_vec(*osrm_resp->costs(), [](auto const& c) {
      return routing_result{c->duration(), c->distance()};
    });
    MOTIS_STOP_TIMING(routing_time);
    creation_routing_time_ += MOTIS_TIMING_US(routing_time);
    return ret;
  } catch (...) {
  }
  return std::vector<routing_result>{{-1, -1}};
}

lift::lift(geo::latlng const& start, geo::latlng const& destination,
           std::time_t const& lift_start_time, int driver_id,
           uint16_t max_passengers)
    : waypoints_({start, destination}),
      rrs_(route(waypoints_)),
      max_total_duration_((max_detour(rrs_[0].duration_) + 1) *
                          rrs_[0].duration_),
      current_total_routing_result_(rrs_[0]),
      t_(lift_start_time),
      driver_id_(driver_id),
      max_passengers_(max_passengers) {}

lift::lift(std::vector<geo::latlng> const& waypoints,
           std::vector<routing_result> rrs, double max_total_duration,
           std::time_t const& lift_start_time, int driver_id,
           uint16_t max_passengers, std::vector<passenger> const& pas)
    : waypoints_(waypoints),
      rrs_(rrs),
      max_total_duration_(max_total_duration),
      current_total_routing_result_(
          std::accumulate(rrs_.begin(), rrs_.end(), routing_result{0, 0})),
      t_(lift_start_time),
      driver_id_(driver_id),
      max_passengers_(max_passengers),
      passengers_(pas) {}

msg_ptr make_osrm_request(geo::latlng const& one,
                          std::vector<geo::latlng> const& many,
                          osrm::Direction dir) {
  message_creator mc;
  Position pos = {one.lat_, one.lng_};

  mc.create_and_finish(MsgContent_OSRMOneToManyRequest,
                       osrm::CreateOSRMOneToManyRequest(
                           mc, mc.CreateString("car"), dir, &pos,
                           mc.CreateVectorOfStructs(utl::to_vec(
                               many,
                               [](auto const& location) {
                                 Position pos{location.lat_, location.lng_};
                                 return pos;
                               })))
                           .Union(),
                       "/osrm/one_to_many");
  return make_msg(mc);
}

void lift::initial_station_routing(
    std::vector<std::pair<geo::latlng, int>> const& parkings) {
  MOTIS_START_TIMING(routing_time);
  std::vector<ctx::future_ptr<ctx_data, void>> futures;
  std::mutex mutex;
  to_routings_.emplace_back();
  add_routing(futures, mutex, true, parkings, 0);
  from_routings_.emplace_back();
  add_routing(futures, mutex, false, parkings, 1);
  ctx::await_all(futures);
  MOTIS_STOP_TIMING(routing_time);
  creation_routing_time_ += MOTIS_TIMING_US(routing_time);
}

bool lift::process_booking(
    passenger const& pas, geo::latlng const& pick_up, leg_idx from_leg,
    geo::latlng const& drop_off, leg_idx to_leg,
    std::vector<std::pair<geo::latlng, int>> const& parkings) {
  waypoints_.insert(waypoints_.begin() + from_leg + 1, pick_up);
  waypoints_.insert(waypoints_.begin() + to_leg + 2, drop_off);
  auto const rrs = route(waypoints_);
  if (rrs.empty() || std::any_of(rrs.begin(), rrs.end(), [](auto const& rr) {
        return rr.duration_ < 0;
      })) {
    waypoints_.erase(waypoints_.begin() + from_leg + 1);
    waypoints_.erase(waypoints_.begin() + to_leg + 2);
    return false;
  }
  rrs_ = rrs;
  current_total_routing_result_ =
      std::accumulate(rrs_.begin(), rrs_.end(), routing_result{0, 0});
  passengers_.push_back(pas);
  to_routings_.emplace(begin(to_routings_) + from_leg + 1);
  from_routings_.emplace(begin(from_routings_) + from_leg);
  to_routings_.emplace(begin(to_routings_) + to_leg + 2);
  from_routings_.emplace(begin(from_routings_) + to_leg + 1);
  MOTIS_START_TIMING(routing_time);
  std::vector<ctx::future_ptr<ctx_data, void>> futures;
  std::mutex mutex;
  add_routing(futures, mutex, true, parkings, from_leg + 1);
  add_routing(futures, mutex, false, parkings, from_leg + 1);
  add_routing(futures, mutex, true, parkings, to_leg + 2);
  add_routing(futures, mutex, false, parkings, to_leg + 2);
  ctx::await_all(futures);
  MOTIS_STOP_TIMING(routing_time);
  booking_routing_time_ = MOTIS_TIMING_US(routing_time);
  return true;
}

std::vector<routing_result> to_rr(OSRMOneToManyResponse const* osrm_resp) {
  std::vector<routing_result> result;
  for (auto const& c : *osrm_resp->costs()) {
    result.emplace_back(c->duration(), c->distance());
  }
  return result;
}

routing_result min_rr(double const dist) {
  return {dist / 25, dist};  // 25m/s == 90km/h
}

bool lift::is_potential(geo::latlng const& loc, leg_idx index) {
  auto const from_wp = geo::distance(waypoints_[index - 1], loc);
  auto const to_wp = geo::distance(waypoints_[index], loc);
  auto const detour = min_rr(from_wp + to_wp) - rrs_[index - 1];

  auto const from_start = geo::distance(waypoints_.front(), loc);
  auto const to_start = geo::distance(waypoints_.back(), loc);
  auto const common =
      from_start > to_start ? min_rr(from_start) : min_rr(to_start);

  return max_total_duration_ >=
             current_total_routing_result_.duration_ + detour.duration_ &&
         lift_economic({common, detour, routing_result{0, 0}});
}

void lift::add_routing(std::vector<ctx::future_ptr<ctx_data, void>>& futures,
                       std::mutex& mutex, bool to_stations,
                       std::vector<std::pair<geo::latlng, int>> const& parkings,
                       leg_idx index) {
  auto filtered = std::vector<geo::latlng>{};
  auto stations = std::vector<unsigned>{};
  for (auto i = 0u; i < parkings.size(); i++) {
    if (is_potential(parkings[i].first, index + to_stations)) {
      filtered.push_back(parkings[i].first);
      stations.push_back(i);
    }
  }
  futures.emplace_back(spawn_job_void([&, filtered, stations, index,
                                       to_stations] {
    try {
      auto const msg = make_osrm_request(
          waypoints_[index], filtered,
          to_stations ? Direction_Forward : Direction_Backward);
      auto const osrm_msg = motis_call(msg)->val();
      auto const osrm_resp = motis_content(OSRMOneToManyResponse, osrm_msg);
      auto const osrm_rr = utl::to_vec(*osrm_resp->costs(), [](auto const& c) {
        return routing_result{c->duration(), c->distance()};
      });
      std::lock_guard<std::mutex> lock(mutex);
      if (to_stations) {
        for (auto const& [rr, s] : utl::zip(osrm_rr, stations)) {
          to_routings_[index].insert({s, rr});
        }
      } else {
        for (auto const& [rr, s] : utl::zip(osrm_rr, stations)) {
          from_routings_[index - 1].insert({s, rr});
        }
      }

    } catch (...) {
    }
  }));
}

void lift::recompute_routings(
    std::vector<std::pair<geo::latlng, int>> const& parkings) {
  to_routings_.clear();
  from_routings_.clear();
  std::mutex mutex;
  std::vector<ctx::future_ptr<ctx_data, void>> futures;
  for (int i = 0; i < rrs_.size() - 1; i++) {
    to_routings_.emplace_back();
    add_routing(futures, mutex, true, parkings, i);
    from_routings_.emplace_back();
    add_routing(futures, mutex, false, parkings, i + 1);
    ctx::await_all(futures);
  }
}

bool lift::remove_booking(int passenger_id) {
  auto it = std::find_if(
      passengers_.begin(), passengers_.end(),
      [passenger_id](auto& p) { return passenger_id == p.passenger_id_; });
  if (it == passengers_.end()) {
    return false;
  }
  auto waypoint_pick_up =
      std::find_if(waypoints_.begin(), waypoints_.end(), [it](auto& w) {
        return w.lat_ == it->pick_up_.lat_ && w.lng_ == it->pick_up_.lng_;
      });
  auto waypoint_drop_off =
      std::find_if(waypoints_.begin(), waypoints_.end(), [it](auto& w) {
        return w.lat_ == it->drop_off_.lat_ && w.lng_ == it->drop_off_.lng_;
      });
  if (waypoint_drop_off == waypoints_.end() ||
      waypoint_pick_up == waypoints_.end()) {
    return false;
  }
  to_routings_.erase(begin(to_routings_) +
                     std::distance(begin(waypoints_), waypoint_pick_up));
  to_routings_.erase(begin(to_routings_) +
                     std::distance(begin(waypoints_), waypoint_drop_off));
  from_routings_.erase(begin(from_routings_) +
                       std::distance(begin(waypoints_) + 1, waypoint_pick_up));
  from_routings_.erase(begin(from_routings_) +
                       std::distance(begin(waypoints_) + 1, waypoint_drop_off));
  waypoints_.erase(waypoint_pick_up);
  waypoints_.erase(waypoint_drop_off);

  rrs_ = route(waypoints_);
  current_total_routing_result_ =
      std::accumulate(rrs_.begin(), rrs_.end(), routing_result{0, 0});
  return true;
}

bool operator<(lift const& l1, lift const& l2) {
  return std::tie(l1.t_, l1.driver_id_) < std::tie(l2.t_, l2.driver_id_);
}

bool operator<(lift_key const& lk, lift const& l) {
  return std::tie(lk.t_, lk.driver_id_) < std::tie(l.t_, l.driver_id_);
}

bool operator<(lift const& l, lift_key const& lk) {
  return std::tie(l.t_, l.driver_id_) < std::tie(lk.t_, lk.driver_id_);
}

bool operator<(lift_key const& l1, lift_key const& l2) {
  return std::tie(l1.t_, l1.driver_id_) < std::tie(l2.t_, l2.driver_id_);
}

bool operator==(lift_key const& l1, lift_key const& l2) {
  return std::tie(l1.t_, l1.driver_id_) == std::tie(l2.t_, l2.driver_id_);
}

bool operator==(lift const& l1, lift const& l2) {
  return std::tie(l1.t_, l1.driver_id_) == std::tie(l2.t_, l2.driver_id_);
}

bool operator==(lift const& l1, lift_key const& l2) {
  return std::tie(l1.t_, l1.driver_id_) == std::tie(l2.t_, l2.driver_id_);
}

}  // namespace motis::ridesharing
