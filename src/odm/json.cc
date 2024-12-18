#include "motis/odm/json.h"

#include <iostream>
#include <sstream>

#include "date/date.h"

namespace motis::odm {

boost::json::value json(pos const& p) {
  return {{"lat", p.lat_}, {"lon", p.lon_}};
}

boost::json::array json(std::vector<pt_ride> v, fixed const f) {
  auto a = boost::json::array{};
  for (auto const r : v) {
    a.emplace_back(boost::json::value{
        std::format("{:%Y-%m-%dT%H:%M%z}", f == kDep ? r.dep_ : r.arr_)});
  }
  return a;
}

boost::json::value json(stop_rides const& s, fixed const f) {
  return {{"coordinates", json(s.pos_)}, {"times", json(s.rides_, f)}};
}

boost::json::array json(std::vector<stop_rides> const& v, fixed const f) {
  auto a = boost::json::array{};
  for (auto const& s : v) {
    a.emplace_back(json(s, f));
  }
  return a;
}

boost::json::value json(capacities const& c) {
  return {{"wheelchairs", c.wheelchairs_},
          {"bikes", c.bikes_},
          {"passengers", c.passengers_},
          {"luggage", c.luggage_}};
}

boost::json::value json(prima_state const& p) {
  return {{"start", json(p.from_)},
          {"target", json(p.to_)},
          {"startBusStops", json(p.from_rides_, fixed::kArr)},
          {"targetBusStops", json(p.to_rides_, fixed::kDep)},
          {"times", json(p.direct_rides_, p.fixed_)},
          {"startFixed", p.fixed_ == fixed::kDep},
          {"capacities", json(p.cap_)}};
}

std::string serialize(prima_state const& p) {
  return boost::json::serialize(json(p));
}

void update(prima_state& ps, std::string_view json) {

  auto const get_pos = [](auto const& v) {
    return pos{.lat_ = v.at("lat").as_double(),
               .lon_ = v.at("lng").as_double()};
  };

  auto const get_time = [](auto const& v) {
    auto ss = std::stringstream{v.as_string().data()};
    auto time = unixtime_t{};
    ss >> date::parse("%Y-%m-%dT%H:%M%z", time);
    return time;
  };

  auto const update_stop_times = [&](auto& stops, auto const& v) {
    for (auto const& x : v.as_array()) {
      stops.emplace_back(get_pos(x.at("coordinates")),
                         std::vector<unixtime_t>{});
      for (auto const& y : x.at("times").as_array()) {
        stops.back().times_.emplace_back(get_time(y));
      }
    }
  };

  try {
    auto const& o = boost::json::parse(json).as_object();
    if (o.contains("startBusStops")) {
      update_stop_times(ps.from_rides_, o.at("startBusStops"));
    }
    if (o.contains("targetBusStops")) {
      update_stop_times(ps.to_rides_, o.at("targetBusStops"));
    }
    if (o.contains("times")) {
      ps.direct_rides_.clear();
      for (auto const& x : o.at("times").as_array()) {
        ps.direct_rides_.emplace_back(get_time(x));
      }
    }
  } catch (std::exception const& e) {
    std::cout << e.what();
  }
}

}  // namespace motis::odm