#include "motis/odm/json.h"

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

std::string json_string(prima_state const& p) {
  return boost::json::serialize(json(p));
}

}  // namespace motis::odm