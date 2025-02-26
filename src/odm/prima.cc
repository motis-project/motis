#include "motis/odm/prima.h"

#include <ranges>

#include "boost/json.hpp"

#include "utl/erase_if.h"
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
  fixed_ = query.arriveBy_ ? n::event_type::kArr : n::event_type::kDep;
  cap_ = {
      .wheelchairs_ = static_cast<std::uint8_t>(
          query.pedestrianProfile_ == api::PedestrianProfileEnum::WHEELCHAIR
              ? 1U
              : 0U),
      .bikes_ = static_cast<std::uint8_t>(query.requireBikeTransport_ ? 1 : 0),
      .passengers_ = query.passengers_.value_or(1U),
      .luggage_ = query.luggage_.value_or(0U)};
}

std::int64_t to_millis(n::unixtime_t const t) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             t.time_since_epoch())
      .count();
}

n::unixtime_t to_unix(std::int64_t const t) {
  return n::unixtime_t{
      std::chrono::duration_cast<n::i32_minutes>(std::chrono::milliseconds{t})};
}

json::array to_json(std::vector<n::routing::start> const& v,
                    n::timetable const& tt,
                    which_mile const wm) {
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
                 utl::transform([&](n::routing::start const& s) {
                   return wm == which_mile::kFirstMile
                              ? to_millis(s.time_at_stop_ - kODMTransferBuffer)
                              : to_millis(s.time_at_stop_ + kODMTransferBuffer);
                 }) |
                 utl::emplace_back_to<json::array>()}});
      });
  return a;
}

json::array to_json(std::vector<direct_ride> const& v,
                    n::event_type const fixed) {
  return utl::all(v)  //
         | utl::transform([&](direct_ride const& r) {
             return to_millis(fixed == n::event_type::kDep ? r.dep_ : r.arr_);
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
          {"startBusStops", to_json(p.from_rides_, tt, kFirstMile)},
          {"targetBusStops", to_json(p.to_rides_, tt, kLastMile)},
          {"directTimes", to_json(p.direct_rides_, p.fixed_)},
          {"startFixed", p.fixed_ == n::event_type::kDep},
          {"capacities", to_json(p.cap_)}};
}

std::string prima::get_prima_request(n::timetable const& tt) const {
  return json::serialize(to_json(*this, tt));
}

std::size_t prima::n_events() const {
  return from_rides_.size() + to_rides_.size() + direct_rides_.size();
}

std::size_t n_pt_updates(json::array const& update) {
  return std::accumulate(
      update.begin(), update.end(), std::size_t{0U},
      [](auto const& a, auto const& b) { return a + b.as_array().size(); });
}

bool prima::blacklist_update(std::string_view json) {

  auto const update_pt_rides =
      [](std::vector<nigiri::routing::start>& rides,
         std::vector<nigiri::routing::start>& prev_rides,
         json::array const& update) {
        auto with_errors = false;
        std::swap(rides, prev_rides);
        rides.clear();
        auto prev_it = std::begin(prev_rides);
        for (auto const& stop : update) {
          for (auto const& ride_upd : stop.as_array()) {
            if (auto const feasible = ride_upd.try_as_bool()) {
              if (*feasible) {
                rides.emplace_back(*prev_it);
              }
            } else {
              with_errors = true;
            }
            ++prev_it;
            if (prev_it == end(prev_rides)) {
              return with_errors;
            }
          }
        }
        return with_errors;
      };

  auto const update_direct_rides = [](std::vector<direct_ride>& rides,
                                      std::vector<direct_ride>& prev_rides,
                                      json::array const& update) {
    auto with_errors = false;
    std::swap(rides, prev_rides);
    rides.clear();
    for (auto const [prev, ride_upd] : utl::zip(prev_rides, update)) {
      if (auto const feasible = ride_upd.try_as_bool()) {
        if (*feasible) {
          rides.emplace_back(prev);
        }
      } else {
        with_errors = true;
      }
    }
    return with_errors;
  };

  auto with_errors = false;
  try {
    auto const o = json::parse(json).as_object();

    auto const n_pt_updates_from = n_pt_updates(o.at("start").as_array());
    if (from_rides_.size() == n_pt_updates_from) {
      with_errors |= update_pt_rides(from_rides_, prev_from_rides_,
                                     o.at("start").as_array());
    } else {
      fmt::println(
          "[blacklisting] from_rides_.size() != n_pt_updates_from ({} != {})",
          from_rides_.size(), n_pt_updates_from);
      with_errors = true;
      from_rides_.clear();
    }

    auto const n_pt_updates_to = n_pt_updates(o.at("target").as_array());
    if (to_rides_.size() == n_pt_updates_to) {
      with_errors |=
          update_pt_rides(to_rides_, prev_to_rides_, o.at("target").as_array());
    } else {
      fmt::println(
          "[blacklisting] to_rides_.size() != n_pt_updates_to ({} != {})",
          to_rides_.size(), n_pt_updates_to);
      with_errors = true;
      to_rides_.clear();
    }

    if (direct_rides_.size() == o.at("direct").as_array().size()) {
      with_errors |= update_direct_rides(direct_rides_, prev_direct_rides_,
                                         o.at("direct").as_array());
    } else {
      fmt::println(
          "[blacklisting] direct_rides_.size() != n_direct_updates ({} != {})",
          direct_rides_.size(), o.at("direct").as_array().size());
      with_errors = true;
      direct_rides_.clear();
    }

  } catch (std::exception const&) {
    fmt::println("[blacklisting] could not parse response: {}", json);
    return false;
  }
  if (with_errors) {
    fmt::println("[blacklisting] parsed response with invalid values: {}",
                 json);
  }
  return true;
}

bool update_pt_rides(std::vector<nigiri::routing::start>& rides,
                     std::vector<nigiri::routing::start>& prev_rides,
                     json::array const& update,
                     which_mile const wm) {
  std::swap(rides, prev_rides);
  rides.clear();

  auto const n_pt_udpates = n_pt_updates(update);
  if (prev_rides.size() != n_pt_udpates) {
    fmt::println("[whitelisting] #rides != #updates ({} != {})",
                 prev_rides.size(), n_pt_udpates);
    return true;
  }

  auto prev_it = std::begin(prev_rides);
  for (auto const& stop : update) {
    for (auto const& event : stop.as_array()) {
      if (event.is_null()) {
        rides.push_back({.time_at_start_ = kInfeasible,
                         .time_at_stop_ = kInfeasible,
                         .stop_ = prev_it->stop_});
      } else {
        auto const time_at_coord_str =
            wm == kFirstMile
                ? to_unix(event.as_object().at("pickupTime").as_int64())
                : to_unix(event.as_object().at("dropoffTime").as_int64());
        auto const time_at_stop_str =
            wm == kFirstMile
                ? to_unix(event.as_object().at("dropoffTime").as_int64())
                : to_unix(event.as_object().at("pickupTime").as_int64());
        rides.push_back({.time_at_start_ = time_at_coord_str,
                         .time_at_stop_ = time_at_stop_str,
                         .stop_ = prev_it->stop_});
      }
      ++prev_it;
      if (prev_it == end(prev_rides)) {
        return false;
      }
    }
  }
  return false;
}

bool update_direct_rides(std::vector<direct_ride>& rides,
                         json::array const& update) {
  if (rides.size() != update.size()) {
    fmt::println("[whitelisting] #rides != #updates ({} != {})", rides.size(),
                 update.size());
    rides.clear();
    return true;
  }

  rides.clear();
  for (auto const& ride : update) {
    if (!ride.is_null()) {
      rides.push_back({to_unix(ride.as_object().at("pickupTime").as_int64()),
                       to_unix(ride.as_object().at("dropoffTime").as_int64())});
    }
  }

  return false;
}

bool prima::whitelist_update(std::string_view json) {
  auto with_errors = false;
  try {
    auto const o = json::parse(json).as_object();
    with_errors |= update_pt_rides(from_rides_, prev_from_rides_,
                                   o.at("start").as_array(), kFirstMile);
    with_errors |= update_pt_rides(to_rides_, prev_to_rides_,
                                   o.at("target").as_array(), kLastMile);
    with_errors |=
        update_direct_rides(direct_rides_, o.at("direct").as_array());
  } catch (std::exception const&) {
    fmt::println("[whitelisting] could not parse response: {}", json);
    return false;
  }
  if (with_errors) {
    fmt::println("[whitelisting] parsed response with errors: {}", json);
  }
  return true;
}

void prima::adjust_to_whitelisting() {
  for (auto const [from_ride, prev_from_ride] :
       utl::zip(from_rides_, prev_from_rides_)) {

    auto const uses_prev_from =
        [&, prev_from = prev_from_ride /* hack for MacOS - fixed with 16 */](
            nigiri::routing::journey const& j) {
          return j.legs_.size() > 1 &&
                 j.legs_.front().dep_time_ == prev_from.time_at_start_ &&
                 j.legs_.front().arr_time_ == prev_from.time_at_stop_ &&
                 j.legs_.front().to_ == prev_from.stop_ &&
                 is_odm_leg(j.legs_.front());
        };

    if (from_ride.time_at_start_ == kInfeasible) {
      utl::erase_if(odm_journeys_, uses_prev_from);
    } else {
      for (auto& j : odm_journeys_) {
        if (uses_prev_from(j)) {
          auto const l = begin(j.legs_);
          l->dep_time_ = from_ride.time_at_start_;
          l->arr_time_ = from_ride.time_at_stop_;
          std::get<n::routing::offset>(l->uses_).duration_ =
              l->arr_time_ - l->dep_time_;
          // fill gap (transfer/waiting) with footpath
          j.legs_.emplace(
              std::next(l),
              n::routing::journey::leg{
                  n::direction::kForward, l->to_, l->to_, l->arr_time_,
                  std::next(l)->dep_time_,
                  n::footpath{l->to_, std::next(l)->dep_time_ - l->arr_time_}});
        }
      }
    }
  }

  for (auto const [to_ride, prev_to_ride] :
       utl::zip(to_rides_, prev_to_rides_)) {

    auto const uses_prev_to = [&, prev = prev_to_ride](auto const& j) {
      return j.legs_.size() > 1 &&
             j.legs_.back().dep_time_ == prev.time_at_stop_ &&
             j.legs_.back().arr_time_ == prev.time_at_start_ &&
             j.legs_.back().from_ == prev.stop_ && is_odm_leg(j.legs_.back());
    };

    if (to_ride.time_at_start_ == kInfeasible) {
      utl::erase_if(odm_journeys_, uses_prev_to);
    } else {
      for (auto& j : odm_journeys_) {
        if (uses_prev_to(j)) {
          auto const l = std::prev(end(j.legs_));
          l->dep_time_ = to_ride.time_at_stop_;
          l->arr_time_ = to_ride.time_at_start_;
          std::get<n::routing::offset>(l->uses_).duration_ =
              l->arr_time_ - l->dep_time_;
          // fill gap (transfer/waiting) with footpath
          j.legs_.emplace(
              l, n::routing::journey::leg{
                     n::direction::kForward, l->from_, l->from_,
                     std::prev(l)->arr_time_, l->dep_time_,
                     n::footpath{l->from_,
                                 l->dep_time_ - std::prev(l)->arr_time_}});
        }
      }
    }
  }

  // adjust journey start/dest times after adjusting legs
  for (auto& j : odm_journeys_) {
    if (!j.legs_.empty()) {
      j.start_time_ = j.legs_.front().dep_time_;
      j.dest_time_ = j.legs_.back().arr_time_;
    }
  }
}

}  // namespace motis::odm