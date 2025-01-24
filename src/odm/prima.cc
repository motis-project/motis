#include "motis/odm/prima.h"

#include <ranges>

#include "boost/json.hpp"

#include "utl/zip.h"

#include "nigiri/common/parse_time.h"
#include "nigiri/timetable.h"

#include "motis/odm/odm.h"

namespace motis::odm {

namespace n = nigiri;

static constexpr auto const kInfeasible =
    std::numeric_limits<n::unixtime_t>::min();

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

template <which_mile Wm>
boost::json::value json(n::routing::start const& s) {
  if constexpr (Wm == which_mile::kFirstMile) {
    return json(s.time_at_stop_ - kODMTransferBuffer);
  } else {
    return json(s.time_at_stop_ + kODMTransferBuffer);
  }
}

template <which_mile Wm>
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
          times.emplace_back(json<Wm>(s));
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
          {"startBusStops", json<kFirstMile>(p.from_rides_, tt)},
          {"targetBusStops", json<kLastMile>(p.to_rides_, tt)},
          {"times", json(p.direct_rides_, p.fixed_)},
          {"startFixed", p.fixed_ == fixed::kDep},
          {"capacities", json(p.cap_)}};
}

std::string prima::get_msg_str(n::timetable const& tt) const {
  return boost::json::serialize(json(*this, tt));
}

size_t prima::n_events() const {
  return from_rides_.size() + to_rides_.size() + direct_rides_.size();
}

bool prima::blacklist_update(std::string_view json) {
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
          rides.emplace_back(kInfeasible, prev_it->time_at_stop_,
                             prev_it->stop_);
        } else {
          rides.emplace_back(
              n::parse_time(value_to<std::string>(time), kPrimaTimeFormat),
              prev_it->time_at_stop_, prev_it->stop_);
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
        if (fixed_ == kDep) {
          rides.emplace_back(
              prev.dep_,
              n::parse_time(value_to<std::string>(time_str), kPrimaTimeFormat));
        } else {
          rides.emplace_back(
              n::parse_time(value_to<std::string>(time_str), kPrimaTimeFormat),
              prev.arr_);
        }
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

    auto const uses_prev_from = [&](auto const& j) {
      return j.legs_.size() > 1 &&
             j.legs_.front().dep_time_ == prev_from_ride.time_at_start_ &&
             j.legs_.front().arr_time_ == prev_from_ride.time_at_stop_ &&
             j.legs_.front().to_ == prev_from_ride.stop_ &&
             is_odm_leg(j.legs_.front());
    };

    if (from_ride.time_at_start_ == kInfeasible) {
      std::erase_if(odm_journeys_, uses_prev_from);
    } else {
      for (auto& j : odm_journeys_) {
        if (uses_prev_from(j)) {
          auto const l = begin(j.legs_);
          l->dep_time_ = from_ride.time_at_start_;
          l->arr_time_ = from_ride.time_at_stop_ - kODMTransferBuffer;
          std::get<n::routing::offset>(l->uses_).duration_ =
              l->arr_time_ - l->dep_time_;
          j.start_time_ = l->dep_time_;
          j.legs_.emplace(std::next(l), n::direction::kForward, l->to_, l->to_,
                          l->arr_time_, l->arr_time_ + kODMTransferBuffer,
                          n::footpath{l->to_, kODMTransferBuffer});
        }
      }
    }
  }

  for (auto const [to_ride, prev_to_ride] :
       utl::zip(to_rides_, prev_to_rides_)) {

    auto const uses_prev_to = [&](auto const& j) {
      return j.legs_.size() > 1 &&
             j.legs_.back().dep_time_ == prev_to_ride.time_at_stop_ &&
             j.legs_.back().arr_time_ == prev_to_ride.time_at_start_ &&
             j.legs_.back().from_ == prev_to_ride.stop_ &&
             is_odm_leg(j.legs_.back());
    };

    if (to_ride.time_at_start_ == kInfeasible) {
      std::erase_if(odm_journeys_, uses_prev_to);
    } else {
      for (auto& j : odm_journeys_) {
        if (uses_prev_to(j)) {
          auto const l = std::prev(end(j.legs_));
          l->dep_time_ = to_ride.time_at_stop_ + kODMTransferBuffer;
          l->arr_time_ = to_ride.time_at_start_;
          std::get<n::routing::offset>(l->uses_).duration_ =
              l->arr_time_ - l->dep_time_;
          j.dest_time_ = l->arr_time_;
          j.legs_.emplace(l, n::direction::kForward, l->from_, l->from_,
                          l->dep_time_ - kODMTransferBuffer, l->dep_time_,
                          n::footpath{l->from_, kODMTransferBuffer});
        }
      }
    }
  }
}

}  // namespace motis::odm