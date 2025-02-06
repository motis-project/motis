#include "motis/odm/prima.h"

#include <ranges>

#include "boost/json.hpp"

#include "utl/pipes.h"
#include "utl/zip.h"

#include "nigiri/common/parse_time.h"
#include "nigiri/timetable.h"

#include "motis/odm/odm.h"

namespace motis::odm {

namespace n = nigiri;
namespace json = boost::json;

static constexpr auto const kInfeasible =
    std::numeric_limits<n::unixtime_t>::min();

void prima::init(api::Place const& from,
                 api::Place const& to,
                 api::plan_params const& query) {
  from_ = geo::latlng{from.lat_, from.lon_};
  to_ = geo::latlng{to.lat_, to.lon_};
  fixed_ = query.arriveBy_ ? kArr : kDep;
  cap_ = {
      .wheelchairs_ = static_cast<std::uint8_t>(
          query.pedestrianProfile_ == api::PedestrianProfileEnum::WHEELCHAIR
              ? 1
              : 0),
      .bikes_ = static_cast<std::uint8_t>(query.requireBikeTransport_ ? 1 : 0),
      .passengers_ = 1U,
      .luggage_ = 0U};
}

std::uint64_t to_millis(n::unixtime_t const t) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             t.time_since_epoch())
      .count();
}

template <which_mile Wm>
json::array to_json(std::vector<n::routing::start> const& v,
                    n::timetable const& tt) {
  auto a = json::array{};
  utl::equal_ranges_linear(
      v,
      [](n::routing::start const& a, n::routing::start const& b) {
        return a.stop_ == b.stop_;
      },
      [&](auto&& from_it, auto&& to_it) {
        auto const& pos = tt.locations_.coordinates_[from_it->stop_];
        a.emplace_back(json::value{
            {"lat", pos.lat_},
            {"lng", pos.lng_},
            {"times",
             utl::all(from_it, to_it) |
                 utl::transform([](n::routing::start const& s) {
                   return Wm == which_mile::kFirstMile
                              ? to_millis(s.time_at_stop_ - kODMTransferBuffer)
                              : to_millis(s.time_at_stop_ + kODMTransferBuffer);
                 }) |
                 utl::emplace_back_to<json::array>()}});
      });
  return a;
}

json::array to_json(std::vector<direct_ride> const& v, fixed const f) {
  return utl::all(v)  //
         | utl::transform([&](direct_ride const& r) {
             return to_millis(f == kDep ? r.dep_ : r.arr_);
           })  //
         | utl::emplace_back_to<json::array>();
}

json::value to_json(capacities const& c) {
  return {{"wheelchairs", c.wheelchairs_},
          {"bikes", c.bikes_},
          {"passengers", c.passengers_},
          {"luggage", c.luggage_}};
}

json::value to_json(prima const& p, n::timetable const& tt) {
  return {{"start", {{"lat", p.from_.lat_}, {"lng", p.from_.lng_}}},
          {"target", {{"lat", p.to_.lat_}, {"lng", p.to_.lng_}}},
          {"startBusStops", to_json<kFirstMile>(p.from_rides_, tt)},
          {"targetBusStops", to_json<kLastMile>(p.to_rides_, tt)},
          {"directTimes", to_json(p.direct_rides_, p.fixed_)},
          {"startFixed", p.fixed_ == fixed::kDep},
          {"capacities", to_json(p.cap_)}};
}

std::string prima::get_msg_str(n::timetable const& tt) const {
  return json::serialize(to_json(*this, tt));
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

  auto const update_direct_rides = [](std::vector<direct_ride>& rides,
                                      std::vector<direct_ride>& prev_rides,
                                      json::array const& update) {
    std::swap(rides, prev_rides);
    rides.clear();
    for (auto const& [prev, feasible] : utl::zip(prev_rides, update)) {
      if (feasible.as_bool()) {
        rides.emplace_back(prev);
      }
    }
  };

  try {
    auto const o = json::parse(json).as_object();
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

n::unixtime_t parse_time(std::string_view s) {
  std::stringstream in;
  in.exceptions(std::ios::badbit | std::ios::failbit);
  in << s;

  std::chrono::system_clock::time_point tp;
  in >> date::parse(kPrimaTimeFormat, tp);

  return std::chrono::round<n::unixtime_t::duration>(tp);
};

template <which_mile Wm>
void update_pt_rides(std::vector<nigiri::routing::start>& rides,
                     std::vector<nigiri::routing::start>& prev_rides,
                     json::array const& update) {
  std::swap(rides, prev_rides);
  rides.clear();
  auto prev_it = std::begin(prev_rides);
  for (auto const& stop : update) {
    for (auto const& event : stop.as_array()) {
      if (event.is_null()) {
        rides.emplace_back(kInfeasible, kInfeasible, prev_it->stop_);
      } else {
        auto const time_at_coord_str =
            Wm == kFirstMile
                ? value_to<std::string>(event.as_object().at("pickupTime"))
                : value_to<std::string>(event.as_object().at("dropoffTime"));
        auto const time_at_stop_str =
            Wm == kFirstMile
                ? value_to<std::string>(event.as_object().at("dropoffTime"))
                : value_to<std::string>(event.as_object().at("pickupTime"));
        rides.emplace_back(parse_time(time_at_coord_str),
                           parse_time(time_at_stop_str), prev_it->stop_);
      }
      ++prev_it;
      if (prev_it == end(prev_rides)) {
        return;
      }
    }
  }
}

void update_direct_rides(auto& rides, auto const& update) {
  rides.clear();
  for (auto const& ride : update) {
    if (ride.is_null()) {
      continue;
    }
    rides.emplace_back(
        parse_time(value_to<std::string>(ride.as_object().at("pickupTime"))),
        parse_time(value_to<std::string>(ride.as_object().at("dropoffTime"))));
  }
}

bool prima::whitelist_update(std::string_view json) {
  auto success = true;

  try {
    auto const o = json::parse(json).as_object();
    update_pt_rides<kFirstMile>(from_rides_, prev_from_rides_,
                                o.at("start").as_array());
    update_pt_rides<kLastMile>(to_rides_, prev_to_rides_,
                               o.at("target").as_array());
    update_direct_rides(direct_rides_, o.at("direct").as_array());
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
          l->arr_time_ = from_ride.time_at_stop_;
          std::get<n::routing::offset>(l->uses_).duration_ =
              l->arr_time_ - l->dep_time_;
          j.start_time_ = l->dep_time_;
          // fill gap (transfer/waiting) with footpath
          j.legs_.emplace(
              std::next(l), n::direction::kForward, l->to_, l->to_,
              l->arr_time_, std::next(l)->dep_time_,
              n::footpath{l->to_, std::next(l)->dep_time_ - l->arr_time_});
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
          l->dep_time_ = to_ride.time_at_stop_;
          l->arr_time_ = to_ride.time_at_start_;
          std::get<n::routing::offset>(l->uses_).duration_ =
              l->arr_time_ - l->dep_time_;
          j.dest_time_ = l->arr_time_;
          // fill gap (transfer/waiting) with footpath
          j.legs_.emplace(
              l, n::direction::kForward, l->from_, l->from_,
              std::prev(l)->arr_time_, l->dep_time_,
              n::footpath{l->from_, l->dep_time_ - std::prev(l)->arr_time_});
        }
      }
    }
  }
}

}  // namespace motis::odm