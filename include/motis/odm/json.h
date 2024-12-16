#pragma once

#include <string>

#include "boost/json.hpp"

#include "motis/odm/prima.h"

namespace motis::odm {

inline boost::json::value json(pos const& p) {
  return {{"lat", p.lat_}, {"lon", p.lon_}};
}

inline boost::json::array json(std::vector<unixtime_t> v) {
  auto a = boost::json::array{};
  for (auto const u : v) {
    a.emplace_back(boost::json::value{std::format("{:%Y-%m-%dT%H:%M%z}", u)});
  }
  return a;
}

inline boost::json::value json(stop_times const& s) {
  return {{"coordinates", json(s.pos_)}, {"times", json(s.times_)}};
}

inline boost::json::value json(capacities const& c) {
  return {{"wheelchairs", c.wheelchairs_},
          {"bikes", c.bikes_},
          {"passengers", c.passengers_},
          {"luggage", c.luggage_}};
}

inline boost::json::value json(prima_req const& p) {
  return {{"start", json(p.from_)},
          {"target", json(p.to_)},
          {"startBusStops", json(p.from_stops_)},
          {"targetBusStops", json(p.to_stops_)},
          {"times", json(p.direct_)},
          {"startFixed", p.start_fixed_},
          {"capacities", json(p.cap_)}};
}

std::string string(prima_req const& p) {
  return boost::json::serialize(json(p));
}

}  // namespace motis::odm