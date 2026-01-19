#include "motis/odm/prima.h"

#include <variant>

#include "boost/asio/io_context.hpp"
#include "boost/json.hpp"

#include "utl/erase_if.h"
#include "utl/pipes.h"
#include "utl/zip.h"

#include "nigiri/common/parse_time.h"
#include "nigiri/logging.h"
#include "nigiri/timetable.h"

#include "motis/elevators/elevators.h"
#include "motis/endpoints/routing.h"
#include "motis/http_req.h"
#include "motis/odm/bounds.h"
#include "motis/odm/odm.h"
#include "motis/transport_mode_ids.h"

namespace n = nigiri;
namespace nr = nigiri::routing;
namespace json = boost::json;

namespace motis::odm {

prima::prima(std::string const& prima_url,
             osr::location const& from,
             osr::location const& to,
             api::plan_params const& query)
    : query_{query},
      taxi_blacklist_{prima_url + kBlacklistPath},
      taxi_whitelist_{prima_url + kWhitelistPath},
      ride_sharing_whitelist_{prima_url + kRidesharingPath},
      from_{from},
      to_{to},
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
  auto [_, direct_duration] = r.route_direct(
      e, gbfs, {}, from_p, to_p, {api::ModeEnum::CAR}, std::nullopt,
      std::nullopt, std::nullopt, std::nullopt, false, intvl.from_, false,
      get_osr_parameters(query), query.pedestrianProfile_,
      query.elevationCosts_, kODMMaxDuration, query.maxMatchingDistance_,
      kODMDirectFactor, api_version);

  auto const step =
      std::chrono::duration_cast<n::unixtime_t::duration>(kODMDirectPeriod);
  if (direct_duration < kODMMaxDuration) {
    if (query.arriveBy_) {
      auto const base_time = intvl.to_ - direct_duration;
      auto const midnight = std::chrono::floor<std::chrono::days>(base_time);
      auto const mins_since_midnight =
          std::chrono::duration_cast<std::chrono::minutes>(base_time -
                                                           midnight);
      auto const floored_5_min = (mins_since_midnight.count() / 5) * 5;
      auto const start_time = midnight + std::chrono::minutes(floored_5_min);
      for (auto arr = start_time; intvl.contains(arr); arr -= step) {
        rides.push_back({.dep_ = arr - direct_duration, .arr_ = arr});
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
        rides.push_back({.dep_ = dep, .arr_ = dep + direct_duration});
      }
    }
  }

  return direct_duration;
}

void init_pt(std::vector<n::routing::offset>& offsets,
             std::vector<n::routing::start>& rides,
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
  auto stats = std::map<std::string, std::uint64_t>{};
  offsets = r.get_offsets(rtt, l, dir, {api::ModeEnum::CAR}, std::nullopt,
                          std::nullopt, std::nullopt, std::nullopt, false,
                          get_osr_parameters(query), query.pedestrianProfile_,
                          query.elevationCosts_, max,
                          query.maxMatchingDistance_, gbfs_rd, stats);

  std::erase_if(offsets, [&](n::routing::offset const& o) {
    return r.ride_sharing_bounds_ != nullptr &&
           !r.ride_sharing_bounds_->contains(
               r.tt_->locations_.coordinates_[o.target_]);
  });

  for (auto& o : offsets) {
    o.duration_ += kODMTransferBuffer;
  }

  rides.reserve(offsets.size() * 2);

  n::routing::get_starts(
      dir == osr::direction::kForward ? n::direction::kForward
                                      : n::direction::kBackward,
      tt, rtt, intvl, offsets, {}, {}, n::routing::kMaxTravelTime,
      location_match_mode, false, rides, true, start_time.prf_idx_,
      start_time.transfer_time_settings_);
}

void prima::init(n::interval<n::unixtime_t> const& search_intvl,
                 n::interval<n::unixtime_t> const& taxi_intvl,
                 bool use_first_mile_taxi,
                 bool use_last_mile_taxi,
                 bool use_direct_taxi,
                 bool use_first_mile_ride_sharing,
                 bool use_last_mile_ride_sharing,
                 bool use_direct_ride_sharing,
                 n::timetable const& tt,
                 n::rt_timetable const* rtt,
                 ep::routing const& r,
                 elevators const* e,
                 gbfs::gbfs_routing_data& gbfs,
                 api::Place const& from,
                 api::Place const& to,
                 api::plan_params const& query,
                 n::routing::query const& n_query,
                 unsigned api_version) {
  direct_duration_ = std::optional<std::chrono::minutes>{};
  if ((use_direct_ride_sharing || use_direct_taxi) && r.w_ && r.l_ &&
      (r.ride_sharing_bounds_ == nullptr ||
       (r.ride_sharing_bounds_->contains(from_.pos_) &&
        r.ride_sharing_bounds_->contains(to_.pos_)))) {
    direct_duration_ = init_direct(direct_ride_sharing_, r, e, gbfs, from, to,
                                   search_intvl, query, api_version);

    if (use_direct_taxi && r.odm_bounds_ != nullptr &&
        r.odm_bounds_->contains(from_.pos_) &&
        r.odm_bounds_->contains(to_.pos_)) {
      direct_taxi_ = direct_ride_sharing_;
    }

    if (!use_direct_ride_sharing) {
      direct_ride_sharing_.clear();
    }
  }

  auto const max_offset_duration =
      direct_duration_
          ? std::min(std::max(*direct_duration_, kODMOffsetMinImprovement) -
                         kODMOffsetMinImprovement,
                     kODMMaxDuration)
          : kODMMaxDuration;

  if (use_first_mile_ride_sharing || use_first_mile_taxi) {
    init_pt(
        first_mile_taxi_, first_mile_ride_sharing_, r, from_,
        osr::direction::kForward, query, gbfs, tt, rtt, taxi_intvl, n_query,
        query.arriveBy_ ? n_query.dest_match_mode_ : n_query.start_match_mode_,
        max_offset_duration);

    if (!use_first_mile_taxi || r.odm_bounds_ == nullptr ||
        !r.odm_bounds_->contains(from_.pos_)) {
      first_mile_taxi_.clear();
    } else {
      std::erase_if(first_mile_taxi_, [&](n::routing::offset const& o) {
        return !r.odm_bounds_->contains(
            r.tt_->locations_.coordinates_[o.target_]);
      });
    }

    if (!use_first_mile_ride_sharing) {
      first_mile_ride_sharing_.clear();
    }
  }

  if (use_last_mile_ride_sharing || use_last_mile_taxi) {
    init_pt(
        last_mile_taxi_, last_mile_ride_sharing_, r, to_,
        osr::direction::kBackward, query, gbfs, tt, rtt, taxi_intvl, n_query,
        query.arriveBy_ ? n_query.start_match_mode_ : n_query.dest_match_mode_,
        max_offset_duration);

    if (!use_last_mile_taxi || r.odm_bounds_ == nullptr ||
        !r.odm_bounds_->contains(to_.pos_)) {
      last_mile_taxi_.clear();
    } else {
      std::erase_if(last_mile_taxi_, [&](n::routing::offset const& o) {
        return !r.odm_bounds_->contains(
            r.tt_->locations_.coordinates_[o.target_]);
      });
    }

    if (!use_last_mile_ride_sharing) {
      last_mile_ride_sharing_.clear();
    }
  }

  auto const by_duration = [](auto const& a, auto const& b) {
    return a.duration_ < b.duration_;
  };
  utl::sort(first_mile_taxi_, by_duration);
  utl::sort(last_mile_taxi_, by_duration);
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

json::array to_json(std::vector<n::routing::start> const& rides,
                    n::timetable const& tt,
                    which_mile const wm) {
  auto a = json::array{};
  utl::equal_ranges_linear(
      rides,
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
                   return wm == kFirstMile
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

void tag_invoke(json::value_from_tag const&,
                json::value& jv,
                capacities const& c) {
  jv = {{"wheelchairs", c.wheelchairs_},
        {"bikes", c.bikes_},
        {"passengers", c.passengers_},
        {"luggage", c.luggage_}};
}

std::string make_whitelist_request(
    osr::location const& from,
    osr::location const& to,
    std::vector<n::routing::start> const& first_mile,
    std::vector<n::routing::start> const& last_mile,
    std::vector<direct_ride> const& direct,
    n::event_type const fixed,
    capacities const& cap,
    n::timetable const& tt) {
  return json::serialize(
      json::value{{"start", {{"lat", from.pos_.lat_}, {"lng", from.pos_.lng_}}},
                  {"target", {{"lat", to.pos_.lat_}, {"lng", to.pos_.lng_}}},
                  {"startBusStops", to_json(first_mile, tt, kFirstMile)},
                  {"targetBusStops", to_json(last_mile, tt, kLastMile)},
                  {"directTimes", to_json(direct, fixed)},
                  {"startFixed", fixed == n::event_type::kDep},
                  {"capacities", json::value_from(cap)}});
}

std::size_t prima::n_ride_sharing_events() const {
  return first_mile_ride_sharing_.size() + last_mile_ride_sharing_.size() +
         direct_ride_sharing_.size();
}

std::size_t n_rides_in_response(json::array const& ja) {
  return std::accumulate(
      ja.begin(), ja.end(), std::size_t{0U},
      [](auto const& a, auto const& b) { return a + b.as_array().size(); });
}

void fix_first_mile_duration(std::vector<nr::journey>& journeys,
                             std::vector<nr::start> const& first_mile,
                             std::vector<nr::start> const& prev_first_mile,
                             n::transport_mode_id_t const mode) {
  for (auto const [curr, prev] : utl::zip(first_mile, prev_first_mile)) {

    auto const uses_prev = [&,
                            prev2 = prev /* hack for MacOS - fixed with 16 */](
                               n::routing::journey const& j) {
      return j.legs_.size() > 1 &&
             j.legs_.front().dep_time_ == prev2.time_at_start_ &&
             j.legs_.front().arr_time_ >= prev2.time_at_stop_ &&
             (j.legs_.front().arr_time_ == prev2.time_at_stop_ ||
              mode == kRideSharingTransportModeId) &&
             j.legs_.front().to_ == prev2.stop_ &&
             is_odm_leg(j.legs_.front(), mode);
    };

    if (curr.time_at_start_ == kInfeasible) {
      utl::erase_if(journeys, uses_prev);
    } else {
      for (auto& j : journeys) {
        if (uses_prev(j)) {
          auto const l = begin(j.legs_);
          if (std::holds_alternative<n::footpath>(std::next(l)->uses_)) {
            continue;  // odm leg fixed already before with a different
                       // time_at_stop (rideshare)
          }
          l->dep_time_ = curr.time_at_start_;
          l->arr_time_ =
              curr.time_at_stop_ - (mode == kRideSharingTransportModeId
                                        ? kODMTransferBuffer
                                        : n::duration_t{0});
          std::get<n::routing::offset>(l->uses_).duration_ =
              l->arr_time_ - l->dep_time_;
          // fill gap (transfer/waiting) with footpath
          j.legs_.emplace(
              std::next(l), n::direction::kForward, l->to_, l->to_,
              l->arr_time_, std::next(l)->dep_time_,
              n::footpath{l->to_, std::next(l)->dep_time_ - l->arr_time_});
        }
      }
    }
  }
};

void fix_last_mile_duration(std::vector<nr::journey>& journeys,
                            std::vector<nr::start> const& last_mile,
                            std::vector<nr::start> const& prev_last_mile,
                            n::transport_mode_id_t const mode) {
  for (auto const [curr, prev] : utl::zip(last_mile, prev_last_mile)) {
    auto const uses_prev =
        [&, prev2 = prev /* hack for MacOS - fixed with 16 */](auto const& j) {
          return j.legs_.size() > 1 &&
                 j.legs_.back().dep_time_ <= prev2.time_at_stop_ &&
                 (j.legs_.back().dep_time_ == prev2.time_at_stop_ ||
                  mode == kRideSharingTransportModeId) &&
                 j.legs_.back().arr_time_ == prev2.time_at_start_ &&
                 j.legs_.back().from_ == prev2.stop_ &&
                 is_odm_leg(j.legs_.back(), mode);
        };

    if (curr.time_at_start_ == kInfeasible) {
      utl::erase_if(journeys, uses_prev);
    } else {
      for (auto& j : journeys) {
        if (uses_prev(j)) {
          auto const l = std::prev(end(j.legs_));
          if (std::holds_alternative<n::footpath>(std::prev(l)->uses_)) {
            continue;  // odm leg fixed already before with a different
                       // time_at_stop (rideshare)
          }
          l->dep_time_ =
              curr.time_at_stop_ + (mode == kRideSharingTransportModeId
                                        ? kODMTransferBuffer
                                        : n::duration_t{0});
          l->arr_time_ = curr.time_at_start_;
          std::get<n::routing::offset>(l->uses_).duration_ =
              l->arr_time_ - l->dep_time_;
          // fill gap (transfer/waiting) with footpath
          j.legs_.emplace(
              l, n::direction::kForward, l->from_, l->from_,
              std::prev(l)->arr_time_, l->dep_time_,
              n::footpath{l->from_, l->dep_time_ - std::prev(l)->arr_time_});
        }
      }
    }
  }
};

void add_direct_odm(std::vector<direct_ride> const& direct,
                    std::vector<nr::journey>& odm_journeys,
                    place_t const& from,
                    place_t const& to,
                    bool arrive_by,
                    n::transport_mode_id_t const mode) {
  auto from_l = std::visit(
      utl::overloaded{[](osr::location const&) {
                        return get_special_station(n::special_station::kStart);
                      },
                      [](tt_location const& tt_l) { return tt_l.l_; }},
      from);
  auto to_l = std::visit(
      utl::overloaded{[](osr::location const&) {
                        return get_special_station(n::special_station::kEnd);
                      },
                      [](tt_location const& tt_l) { return tt_l.l_; }},
      to);

  if (arrive_by) {
    std::swap(from_l, to_l);
  }

  for (auto const& d : direct) {
    odm_journeys.push_back(n::routing::journey{
        .legs_ = {{n::direction::kForward, from_l, to_l, d.dep_, d.arr_,
                   n::routing::offset{to_l, std::chrono::abs(d.arr_ - d.dep_),
                                      mode}}},
        .start_time_ = d.dep_,
        .dest_time_ = d.arr_,
        .dest_ = to_l,
        .transfers_ = 0U});
  }
  n::log(n::log_lvl::debug, "motis.prima",
         "[whitelist] added {} direct rides for mode {}", direct.size(), mode);
}

}  // namespace motis::odm