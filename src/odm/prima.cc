#include "motis/odm/prima.h"

#include <ranges>

#include "boost/json.hpp"

#include "utl/erase_if.h"
#include "utl/pipes.h"
#include "utl/zip.h"

#include "nigiri/common/parse_time.h"
#include "nigiri/logging.h"
#include "nigiri/timetable.h"

#include "motis/elevators/elevators.h"
#include "motis/endpoints/routing.h"
#include "motis/odm/bounds.h"
#include "motis/odm/odm.h"

namespace motis::odm {

namespace n = nigiri;
namespace json = boost::json;

static constexpr auto const kInfeasible =
    std::numeric_limits<n::unixtime_t>::min();

prima::prima(osr::location const& from,
             osr::location const& to,
             api::plan_params const& query)
    : from_{geo::latlng{from.lat_, from.lon_}},
      to_{geo::latlng{to.lat_, to.lon_}},
      fixed_{query.arriveBy_ ? n::event_type::kArr : n::event_type::kDep},
      cap_{
          .wheelchairs_ = static_cast<std::uint8_t>(
              query.pedestrianProfile_ == api::PedestrianProfileEnum::WHEELCHAIR
                  ? 1U
                  : 0U),
          .bikes_ =
              static_cast<std::uint8_t>(query.requireBikeTransport_ ? 1 : 0),
          .passengers_ = query.passengers_.value_or(1U),
          .luggage_ = query.luggage_.value_or(0U)} {}

n::duration_t init_direct(std::vector<direct_ride>& rides,
                          ep::routing const& r,
                          elevators const* e,
                          gbfs::gbfs_routing_data& gbfs,
                          api::Place const& from_p,
                          api::Place const& to_p,
                          n::interval<n::unixtime_t> const intvl,
                          api::plan_params const& query,
                          unsigned api_version) {
  rides.clear();

  auto [_, odm_direct_duration] = r.route_direct(
      e, gbfs, from_p, to_p, {api::ModeEnum::CAR}, std::nullopt, std::nullopt,
      std::nullopt, false, intvl.from_, false, get_osr_parameters(query),
      query.pedestrianProfile_, query.elevationCosts_, kODMMaxDuration,
      query.maxMatchingDistance_, kODMDirectFactor, api_version);

  auto const step =
      std::chrono::duration_cast<n::unixtime_t::duration>(kODMDirectPeriod);
  if (odm_direct_duration < kODMMaxDuration) {
    if (query.arriveBy_) {
      auto const base_time = intvl.to_ - odm_direct_duration;
      auto const midnight = std::chrono::floor<std::chrono::days>(base_time);
      auto const mins_since_midnight =
          std::chrono::duration_cast<std::chrono::minutes>(base_time -
                                                           midnight);
      auto const floored_5_min = (mins_since_midnight.count() / 5) * 5;
      auto const start_time = midnight + std::chrono::minutes(floored_5_min);
      for (auto arr = start_time; intvl.contains(arr); arr -= step) {
        rides.push_back({.dep_ = arr - odm_direct_duration, .arr_ = arr});
      }
    } else {
      auto const base_start = intvl.from_;
      auto const midnight_start =
          std::chrono::floor<std::chrono::days>(base_start);
      auto const mins_since_midnight_start =
          std::chrono::duration_cast<std::chrono::minutes>(base_start -
                                                           midnight_start);
      auto const ceiled_5_min_start =
          ((mins_since_midnight_start.count() + 4) / 5) * 5;
      auto const start_time_for_depart =
          midnight_start + std::chrono::minutes(ceiled_5_min_start);
      for (auto dep = start_time_for_depart; intvl.contains(dep); dep += step) {
        rides.push_back({.dep_ = dep, .arr_ = dep + odm_direct_duration});
      }
    }
  } else {
    fmt::println(
        "[init] No direct ODM connection, from: {}, to: {}: "
        "odm_direct_duration >= "
        "kODMMaxDuration ({} "
        ">= {})",
        from_p, to_p, odm_direct_duration, kODMMaxDuration);
  }

  return odm_direct_duration;
}

void init_pt(std::vector<nigiri::routing::start>& rides,
             ep::routing const& r,
             osr::location const& l,
             osr::direction dir,
             api::plan_params const& query,
             gbfs::gbfs_routing_data& gbfs_rd,
             n::timetable const& tt,
             n::rt_timetable const* rtt,
             n::interval<n::unixtime_t> const& intvl,
             n::routing::query const& start_time,
             n::routing::location_match_mode location_match_mode,
             std::chrono::seconds const max) {

  auto offsets = r.get_offsets(
      rtt, l, dir, {api::ModeEnum::CAR}, std::nullopt, std::nullopt,
      std::nullopt, false, get_osr_parameters(query), query.pedestrianProfile_,
      query.elevationCosts_, max, query.maxMatchingDistance_, gbfs_rd);

  std::erase_if(offsets, [&](n::routing::offset const& o) {
    auto const out_of_bounds =
        (r.odm_bounds_ != nullptr &&
         !r.odm_bounds_->contains(r.tt_->locations_.coordinates_[o.target_]));
    return out_of_bounds;
  });

  for (auto& o : offsets) {
    o.duration_ += kODMTransferBuffer;
  }

  rides.clear();
  rides.reserve(offsets.size() * 2);

  n::routing::get_starts(
      dir == osr::direction::kForward ? n::direction::kForward
                                      : n::direction::kBackward,
      tt, rtt, intvl, offsets, {}, n::routing::kMaxTravelTime,
      location_match_mode, false, rides, true, start_time.prf_idx_,
      start_time.transfer_time_settings_);
}

void prima::init(n::interval<n::unixtime_t> const& search_intvl,
                 n::interval<n::unixtime_t> const& odm_intvl,
                 bool odm_pre_transit,
                 bool odm_post_transit,
                 bool odm_direct,
                 bool ride_sharing_pre_transit,
                 bool ride_sharing_post_transit,
                 bool ride_sharing_direct,
                 nigiri::timetable const& tt,
                 nigiri::rt_timetable const* rtt,
                 ep::routing const& r,
                 elevators const* e,
                 gbfs::gbfs_routing_data& gbfs,
                 api::Place const& from_p,
                 api::Place const& to_p,
                 api::plan_params const& query,
                 nigiri::routing::query const& n_query,
                 unsigned api_version) {
  auto direct_duration = std::optional<std::chrono::seconds>{};
  if ((ride_sharing_direct || odm_direct) && r.w_ && r.l_) {
    direct_duration = init_direct(direct_ride_sharing_, r, e, gbfs, from_p,
                                  to_p, search_intvl, query, api_version);
  }

  auto const max_offset_duration =
      direct_duration
          ? std::min(std::max(*direct_duration, kODMOffsetMinImprovement) -
                         kODMOffsetMinImprovement,
                     kODMMaxDuration)
          : kODMMaxDuration;

  if (ride_sharing_pre_transit || odm_pre_transit) {
    init_pt(
        first_mile_ride_sharing_, r, from_, osr::direction::kForward, query,
        gbfs, tt, rtt_, odm_intvl, n_query,
        query.arriveBy_ ? n_query.dest_match_mode_ : n_query.start_match_mode_,
        max_offset_duration);
  }

  if (ride_sharing_post_transit || odm_post_transit_) {
    init_pt(
        last_mile_ride_sharing_, r, to_, osr::direction::kBackward, query, gbfs,
        tt, rtt_, odm_intvl, n_query,
        query.arriveBy_ ? n_query.start_match_mode_ : n_query.dest_match_mode_,
        max_offset_duration);
  }
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
      n::log(
          n::log_lvl::debug, "motis.prima",
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
      n::log(n::log_lvl::debug, "motis.prima",
             "[blacklisting] to_rides_.size() != n_pt_updates_to ({} != {})",
             to_rides_.size(), n_pt_updates_to);
      with_errors = true;
      to_rides_.clear();
    }

    if (direct_rides_.size() == o.at("direct").as_array().size()) {
      with_errors |= update_direct_rides(direct_rides_, prev_direct_rides_,
                                         o.at("direct").as_array());
    } else {
      n::log(
          n::log_lvl::debug, "motis.prima",
          "[blacklisting] direct_rides_.size() != n_direct_updates ({} != {})",
          direct_rides_.size(), o.at("direct").as_array().size());
      with_errors = true;
      direct_rides_.clear();
    }

  } catch (std::exception const&) {
    n::log(n::log_lvl::debug, "motis.prima",
           "[blacklisting] could not parse response: {}", json);
    return false;
  }
  if (with_errors) {
    n::log(n::log_lvl::debug, "motis.prima",
           "[blacklisting] parsed response with invalid values: {}", json);
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
    n::log(n::log_lvl::debug, "motis.prima",
           "[whitelisting] #rides != #updates ({} != {})", prev_rides.size(),
           n_pt_udpates);
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
    n::log(n::log_lvl::debug, "motis.prima",
           "[whitelisting] #rides != #updates ({} != {})", rides.size(),
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
    n::log(n::log_lvl::debug, "motis.prima",
           "[whitelisting] could not parse response: {}", json);
    return false;
  }
  if (with_errors) {
    n::log(n::log_lvl::debug, "motis.prima",
           "[whitelisting] parsed response with errors: {}", json);
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