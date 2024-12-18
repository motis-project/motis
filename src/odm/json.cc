#include "motis/odm/json.h"

#include <iostream>
#include <sstream>

#include "date/date.h"

namespace motis::odm {

boost::json::value json(pos const& p) {
  return {{"lat", p.lat_}, {"lon", p.lon_}};
}

boost::json::array json(std::vector<unixtime_t> v) {
  auto a = boost::json::array{};
  for (auto const u : v) {
    a.emplace_back(boost::json::value{std::format("{:%Y-%m-%dT%H:%M%z}", u)});
  }
  return a;
}

boost::json::value json(stop_times const& s) {
  return {{"coordinates", json(s.pos_)}, {"times", json(s.times_)}};
}

boost::json::array json(std::vector<stop_times> const& v) {
  auto a = boost::json::array{};
  for (auto const& s : v) {
    a.emplace_back(json(s));
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
          {"startBusStops", json(p.from_stops_)},
          {"targetBusStops", json(p.to_stops_)},
          {"times", json(p.direct_)},
          {"startFixed", p.start_fixed_},
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
    stops.clear();
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
      update_stop_times(ps.from_stops_, o.at("startBusStops"));
    }
    if (o.contains("targetBusStops")) {
      update_stop_times(ps.to_stops_, o.at("targetBusStops"));
    }
    if (o.contains("times")) {
      ps.direct_.clear();
      for (auto const& x : o.at("times").as_array()) {
        ps.direct_.emplace_back(get_time(x));
      }
    }
  } catch (std::exception const& e) {
    std::cout << e.what();
  }
}

}  // namespace motis::odm