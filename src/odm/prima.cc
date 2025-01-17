#include "motis/odm/prima.h"

#include <ranges>

#include "boost/json.hpp"

#include "utl/zip.h"

#include "nigiri/common/parse_time.h"
#include "nigiri/timetable.h"

#include "motis/odm/odm.h"

namespace motis::odm {

void prima::init(api::Place const& from,
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
  return {{"lat", p.lat_}, {"lng", p.lng_}};
}

boost::json::value json(n::unixtime_t const t) {
  return {date::format(kPrimaTimeFormat, t)};
}

boost::json::value json(n::routing::start const& s) {
  return json(s.time_at_stop_);
}

template <>
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

boost::json::value json(prima const& p, n::timetable const& tt) {
  return {{"start", json(p.from_)},
          {"target", json(p.to_)},
          {"startBusStops", json(p.from_rides_, tt)},
          {"targetBusStops", json(p.to_rides_, tt)},
          {"times", json(p.direct_rides_, p.fixed_)},
          {"startFixed", p.fixed_ == fixed::kDep},
          {"capacities", json(p.cap_)}};
}

std::string prima::get_msg_str(n::timetable const& tt) const {
  return boost::json::serialize(json(*this, tt));
}

bool prima::blacklist_update(std::string_view json) {
  std::cout << "blacklist_response: " << json << "\n";

  auto success = true;

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
    for (auto const& [prev, feasible] : utl::zip(prev_rides, update)) {
      if (value_to<bool>(feasible)) {
        rides.emplace_back(prev);
      }
    }
  };

  try {
    auto const o = boost::json::parse(json).as_object();
    update_pt_rides(from_rides_, prev_from_rides_, o.at("start").as_array());
    update_pt_rides(to_rides_, prev_to_rides_, o.at("target").as_array());
    update_direct_rides(direct_rides_, prev_direct_rides_,
                        o.at("direct").as_array());
  } catch (std::exception const& e) {
    std::cout << e.what() << "\nInvalid blacklist response: " << json << "\n";
    success = false;
  }
  return success;
}

bool prima::whitelist_update(std::string_view json) {
  auto success = true;

  auto const update_pt_rides = [](auto& rides, auto& prev_rides,
                                  auto const& update) {
    std::swap(rides, prev_rides);
    rides.clear();
    auto prev_it = std::begin(prev_rides);
    for (auto const& stop : update) {
      for (auto const& time : stop.as_array()) {
        if (time.is_null()) {

        } else {
          auto const delta =
              n::parse_time(value_to<std::string>(time), kPrimaTimeFormat) -
              prev_it->time_at_stop_;
          rides.emplace_back(prev_it->time_at_start_ + delta,
                             prev_it->time_at_stop_ + delta, prev_it->stop_);
        }
        ++prev_it;
        if (prev_it == end(prev_rides)) {
          return;
        }
      }
    }
  };

  auto const update_direct_rides = [&](auto& rides, auto& prev_rides,
                                       auto const& update) {
    std::swap(rides, prev_rides);
    rides.clear();
    for (auto const& [prev, time_str] : utl::zip(prev_rides, update)) {
      if (!time_str.is_null()) {
        auto const old_time = fixed_ == kDep ? prev.dep_ : prev.arr_;
        auto const new_time =
            n::parse_time(value_to<std::string>(time_str), kPrimaTimeFormat);
        auto const delta = new_time - old_time;
        rides.emplace_back(prev.dep_ + delta, prev.arr_ + delta);
      }
    }
  };

  try {
    auto const o = boost::json::parse(json).as_object();
    update_pt_rides(from_rides_, prev_from_rides_, o.at("start").as_array());
    update_pt_rides(to_rides_, prev_to_rides_, o.at("target").as_array());
    update_direct_rides(direct_rides_, prev_direct_rides_,
                        o.at("direct").as_array());
  } catch (std::exception const& e) {
    std::cout << e.what() << "\nInvalid whitelist response: " << json << "\n";
    success = false;
  }
  return success;
}

void prima::adjust_to_whitelisting() {
  for (auto const [from_ride, prev_from_ride] :
       utl::zip(from_rides_, prev_from_rides_)) {
    if (from_ride == prev_from_ride) {
      continue;
    }
    for (auto& j : odm_journeys_) {
      if (j.legs_.size() > 1 &&
          j.legs_.front().dep_time_ == prev_from_ride.time_at_start_ &&
          j.legs_.front().arr_time_ == prev_from_ride.time_at_stop_ &&
          j.legs_.front().to_ == prev_from_ride.stop_ &&
          is_odm_leg(j.legs_.front())) {
        j.legs_.front().dep_time_ = from_ride.time_at_start_;
        j.legs_.front().arr_time_ = from_ride.time_at_stop_;
        std::get<n::routing::offset>(j.legs_.front().uses_).duration_ =
            std::chrono::abs(from_ride.time_at_stop_ -
                             from_ride.time_at_start_);
        j.start_time_ = j.legs_.front().dep_time_;
      }
    }
  }

  for (auto const [to_ride, prev_to_ride] :
       utl::zip(to_rides_, prev_to_rides_)) {
    if (to_ride == prev_to_ride) {
      continue;
    }
    for (auto& j : odm_journeys_) {
      if (j.legs_.size() > 1 &&
          j.legs_.back().dep_time_ == prev_to_ride.time_at_stop_ &&
          j.legs_.back().arr_time_ == prev_to_ride.time_at_start_ &&
          j.legs_.back().from_ == prev_to_ride.stop_ &&
          is_odm_leg(j.legs_.back())) {
        j.legs_.back().dep_time_ = to_ride.time_at_stop_;
        j.legs_.back().arr_time_ = to_ride.time_at_start_;
        std::get<n::routing::offset>(j.legs_.back().uses_).duration_ =
            std::chrono::abs(to_ride.time_at_start_ - to_ride.time_at_stop_);
        j.dest_time_ = j.legs_.back().arr_time_;
      }
    }
  }
}

}  // namespace motis::odm