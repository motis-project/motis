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
        auto& times = a.back().at("times").as_array();
        for (auto const& s : n::it_range{from_it, to_it}) {
          times.emplace_back(json(s));
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

  auto const update_pt_rides = [](auto& rides, auto& prev_rides,
                                  auto const& update) {
    std::swap(rides, prev_rides);
    rides.clear();
    auto prev_it = std::begin(prev_rides);
    for (auto const& stop : update) {
      for (auto const& time : stop.as_array()) {
        if (time.as_bool()) {
          rides.emplace_back(*prev_it);
        }
        ++prev_it;
        if (prev_it == end(prev_rides)) {
          return;
        }
      }
    }
  };

  auto const update_direct_rides = [](auto& rides, auto& prev_rides,
                                      auto const& update) {
    std::swap(rides, prev_rides);
    rides.clear();
    for (auto const& [prev, time] : utl::zip(prev_rides, update)) {
      if (time.as_bool()) {
        rides.emplace_back(prev);
      }
    }
  };

  try {
    auto const& o = boost::json::parse(json).as_object();
    if (o.contains("startBusStops")) {
      update_pt_rides(ps.from_rides_, ps.prev_from_rides_,
                      o.at("startBusStops").as_array());
    }
    if (o.contains("targetBusStops")) {
      update_pt_rides(ps.to_rides_, ps.prev_to_rides_,
                      o.at("targetBusStops").as_array());
    }
    if (o.contains("times")) {
      update_direct_rides(ps.direct_rides_, ps.prev_direct_rides_,
                          o.at("times").as_array());
    }
  } catch (std::exception const& e) {
    std::cout << e.what();
  }
}

}  // namespace motis::odm