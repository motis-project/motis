#include "motis/odm/json.h"

#include <iostream>
#include <sstream>

#include "date/date.h"

#include "nigiri/common/it_range.h"
#include "nigiri/timetable.h"

namespace motis::odm {

boost::json::value json(geo::latlng const& p) {
  return {{"lat", p.lat_}, {"lon", p.lng_}};
}

boost::json::value json(n::unixtime_t const t) {
  return {std::format("{:%Y-%m-%dT%H:%M%z}", t)};
}

boost::json::value json(n::routing::start const& s) {
  return json(s.time_at_stop_);
}

boost::json::array json(std::vector<n::routing::start> const& v,
                        n::timetable const& tt) {
  auto a = boost::json::array{};
  utl::equal_ranges_linear(
      v,
      [](n::routing::start const& a, n::routing::start const& b) {
        return a.stop_ == b.stop_;
      },
      [&](auto&& from_it, auto&& to_it) {
        a.emplace_back(boost::json::value{
            {"coordinates", json(tt.locations_.coordinates_[from_it->stop_])},
            {"times", boost::json::array{}}});
        for (auto const& s : n::it_range{from_it, to_it}) {
          a.back().at("times").as_array().emplace_back(json(s));
        }
      });
  return a;
}

boost::json::array json(std::vector<direct_ride> const& v, fixed const f) {
  auto a = boost::json::array{};
  for (auto const& r : v) {
    a.emplace_back(json(f == kDep ? r.dep_ : r.arr_));
  }
  return a;
}

boost::json::value json(capacities const& c) {
  return {{"wheelchairs", c.wheelchairs_},
          {"bikes", c.bikes_},
          {"passengers", c.passengers_},
          {"luggage", c.luggage_}};
}

boost::json::value json(prima_state const& p, n::timetable const& tt) {
  return {{"start", json(p.from_)},
          {"target", json(p.to_)},
          {"startBusStops", json(p.from_rides_, tt)},
          {"targetBusStops", json(p.to_rides_, tt)},
          {"times", json(p.direct_rides_, p.fixed_)},
          {"startFixed", p.fixed_ == fixed::kDep},
          {"capacities", json(p.cap_)}};
}

std::string serialize(prima_state const& p, n::timetable const& tt) {
  return boost::json::serialize(json(p, tt));
}

void update(prima_state& ps, std::string_view json) {

  auto const get_pos = [](auto const& v) {
    return geo::latlng{.lat_ = v.at("lat").as_double(),
                       .lng_ = v.at("lng").as_double()};
  };

  auto const get_time = [](auto const& v) {
    auto ss = std::stringstream{v.as_string().data()};
    auto time = n::unixtime_t{};
    ss >> date::parse("%Y-%m-%dT%H:%M%z", time);
    return time;
  };

  auto const update_pt_rides = [&](auto& rides, auto const& v) {
    for (auto const& x : v.as_array()) {
      rides.emplace_back(get_pos(x.at("coordinates")),
                         std::vector<n::unixtime_t>{});
      for (auto const& y : x.at("times").as_array()) {
        stops.back().times_.emplace_back(get_time(y));
      }
    }
  };

  try {
    auto const& o = boost::json::parse(json).as_object();
    if (o.contains("startBusStops")) {
      update_pt_rides(ps.from_rides_, o.at("startBusStops"));
    }
    if (o.contains("targetBusStops")) {
      update_pt_rides(ps.to_rides_, o.at("targetBusStops"));
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