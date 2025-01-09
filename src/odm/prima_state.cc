#include "motis/odm/prima_state.h"

#include <ranges>

#include "boost/json.hpp"

#include "utl/zip.h"

#include "nigiri/timetable.h"

namespace motis::odm {

void prima_state::init(api::Place const& from,
                       api::Place const& to,
                       api::plan_params const& query) {
  from_ = geo::latlng{from.lat_, from.lon_};
  to_ = geo::latlng{to.lat_, to.lon_};
  fixed_ = query.arriveBy_ ? kArr : kDep;
  cap_ = {.wheelchairs_ = static_cast<std::uint8_t>(
              query.pedestrianProfile_ == api::PedestrianProfileEnum::WHEELCHAIR
                  ? 1U
                  : 0U),
          .bikes_ =
              static_cast<std::uint8_t>(query.requireBikeTransport_ ? 1U : 0U),
          .passengers_ = 1U,
          .luggage_ = 0U};
}

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

std::string prima_state::serialize(n::timetable const& tt) const {
  return boost::json::serialize(json(*this, tt));
}

void prima_state::blacklist_update(std::string_view json) {
  auto const update_pt_rides = [](auto& rides, auto& prev_rides,
                                  auto const& update) {
    std::swap(rides, prev_rides);
    rides.clear();
    auto prev_it = std::begin(prev_rides);
    for (auto const& stop : update) {
      for (auto const& feasible : stop.as_array()) {
        if (value_to<bool>(feasible)) {
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
      if (value_to<bool>(time)) {
        rides.emplace_back(prev);
      }
    }
  };

  auto const o = boost::json::parse(json).as_object();
  if (o.contains("startBusStops")) {
    update_pt_rides(from_rides_, prev_from_rides_,
                    o.at("startBusStops").as_array());
  }
  if (o.contains("targetBusStops")) {
    update_pt_rides(to_rides_, prev_to_rides_,
                    o.at("targetBusStops").as_array());
  }
  if (o.contains("times")) {
    update_direct_rides(direct_rides_, prev_direct_rides_,
                        o.at("times").as_array());
  }
}

void prima_state::whitelist_update(std::string_view json [[maybe_unused]]) {
  // TODO

  auto const update_pt_rides = [](auto& rides, auto& prev_rides,
                                  auto const& update) {
    std::swap(rides, prev_rides);
    rides.clear();
    auto prev_it = std::begin(prev_rides);
    for (auto const& stop : update) {
      for (auto const& time : stop.as_array()) {
        if (value_to<bool>(time)) {
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
      if (value_to<bool>(time)) {
        rides.emplace_back(prev);
      }
    }
  };

  auto const o = boost::json::parse(json).as_object();
  if (o.contains("startBusStops")) {
    update_pt_rides(from_rides_, prev_from_rides_,
                    o.at("startBusStops").as_array());
  }
  if (o.contains("targetBusStops")) {
    update_pt_rides(to_rides_, prev_to_rides_,
                    o.at("targetBusStops").as_array());
  }
  if (o.contains("times")) {
    update_direct_rides(direct_rides_, prev_direct_rides_,
                        o.at("times").as_array());
  }
}

}  // namespace motis::odm