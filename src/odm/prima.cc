#include "motis/odm/prima.h"

#include <ranges>
#include <variant>

#include "boost/asio/co_spawn.hpp"
#include "boost/asio/detached.hpp"
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

namespace motis::odm {

namespace n = nigiri;
namespace json = boost::json;
using namespace std::chrono_literals;

constexpr auto const kODMDirectPeriod = 300s;
constexpr auto const kODMDirectFactor = 1.0;
constexpr auto const kODMOffsetMinImprovement = 60s;
constexpr auto const kODMMaxDuration = 3600s;
constexpr auto const kBlacklistPath = "/api/blacklist";
constexpr auto const kWhitelistPath = "/api/whitelist";
constexpr auto const kRidesharingPath = "/api/whitelistRideShare";
static auto const kReqHeaders = std::map<std::string, std::string>{
    {"Content-Type", "application/json"}, {"Accept", "application/json"}};

static constexpr auto const kInfeasible =
    std::numeric_limits<n::unixtime_t>::min();

prima::prima(std::string const& prima_url,
             osr::location const& from,
             osr::location const& to,
             api::plan_params const& query)
    : taxi_blacklist_{prima_url + kBlacklistPath},
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
  rides.clear();

  auto [_, direct_duration] = r.route_direct(
      e, gbfs, from_p, to_p, {api::ModeEnum::CAR}, std::nullopt, std::nullopt,
      std::nullopt, std::nullopt, false, intvl.from_, false,
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

void init_pt(std::vector<n::routing::start>& rides,
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

  auto offsets = r.get_offsets(rtt, l, dir, {api::ModeEnum::CAR}, std::nullopt,
                               std::nullopt, std::nullopt, std::nullopt, false,
                               get_osr_parameters(query),
                               query.pedestrianProfile_, query.elevationCosts_,
                               max, query.maxMatchingDistance_, gbfs_rd);

  for (auto& o : offsets) {
    o.duration_ += kODMTransferBuffer;
  }

  rides.reserve(offsets.size() * 2);

  n::routing::get_starts(
      dir == osr::direction::kForward ? n::direction::kForward
                                      : n::direction::kBackward,
      tt, rtt, intvl, offsets, {}, n::routing::kMaxTravelTime,
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
  auto direct_duration = std::optional<std::chrono::seconds>{};
  if ((use_direct_ride_sharing || use_direct_taxi) && r.w_ && r.l_) {
    direct_duration = init_direct(direct_ride_sharing_, r, e, gbfs, from, to,
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
      direct_duration
          ? std::min(std::max(*direct_duration, kODMOffsetMinImprovement) -
                         kODMOffsetMinImprovement,
                     kODMMaxDuration)
          : kODMMaxDuration;

  if (use_first_mile_ride_sharing || use_first_mile_taxi) {
    init_pt(
        first_mile_ride_sharing_, r, from_, osr::direction::kForward, query,
        gbfs, tt, rtt, taxi_intvl, n_query,
        query.arriveBy_ ? n_query.dest_match_mode_ : n_query.start_match_mode_,
        max_offset_duration);

    if (use_first_mile_taxi && r.odm_bounds_ != nullptr &&
        r.odm_bounds_->contains(from_.pos_)) {
      for (auto const& i : first_mile_ride_sharing_) {
        if (r.odm_bounds_->contains(tt.locations_.coordinates_[i.stop_])) {
          first_mile_taxi_.emplace_back(i);
        }
      }
    }

    if (!use_first_mile_ride_sharing) {
      first_mile_ride_sharing_.clear();
    }
  }

  if (use_last_mile_ride_sharing || use_last_mile_taxi) {
    init_pt(
        last_mile_ride_sharing_, r, to_, osr::direction::kBackward, query, gbfs,
        tt, rtt, taxi_intvl, n_query,
        query.arriveBy_ ? n_query.start_match_mode_ : n_query.dest_match_mode_,
        max_offset_duration);

    if (use_last_mile_taxi && r.odm_bounds_ != nullptr &&
        r.odm_bounds_->contains(to_.pos_)) {
      for (auto const& i : last_mile_ride_sharing_) {
        if (r.odm_bounds_->contains(tt.locations_.coordinates_[i.stop_])) {
          last_mile_taxi_.emplace_back(i);
        }
      }
    }

    if (!use_last_mile_ride_sharing) {
      last_mile_ride_sharing_.clear();
    }
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

std::string make_request(osr::location const& from,
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
                  {"capacities", to_json(cap)}});
}

std::string prima::make_taxi_request(n::timetable const& tt) const {
  return make_request(from_, to_, first_mile_taxi_, last_mile_taxi_,
                      direct_taxi_, fixed_, cap_, tt);
}

std::string prima::make_ride_sharing_request(
    nigiri::timetable const& tt) const {
  return make_request(from_, to_, first_mile_ride_sharing_,
                      last_mile_ride_sharing_, direct_ride_sharing_, fixed_,
                      cap_, tt);
}

std::size_t prima::n_taxi_events() const {
  return first_mile_taxi_.size() + last_mile_taxi_.size() + direct_taxi_.size();
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

bool prima::consume_blacklist_taxis_response(std::string_view json) {
  auto const update_pt_rides = [](std::vector<n::routing::start>& rides,
                                  json::array const& update) {
    auto with_errors = false;
    auto prev_rides = std::exchange(rides, std::vector<n::routing::start>{});
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
      }
    }
    return with_errors;
  };

  auto const update_direct_rides = [](std::vector<direct_ride>& rides,
                                      json::array const& update) {
    auto with_errors = false;
    auto prev_rides = std::exchange(rides, std::vector<direct_ride>{});
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

    auto const n_updates_first_mile =
        n_rides_in_response(o.at("start").as_array());
    if (first_mile_taxi_.size() == n_updates_first_mile) {
      with_errors |=
          update_pt_rides(first_mile_taxi_, o.at("start").as_array());
    } else {
      n::log(
          n::log_lvl::debug, "motis.prima",
          "[blacklist taxi] from_rides_.size() != n_updates_first_mile ({} != "
          "{})",
          first_mile_taxi_.size(), n_updates_first_mile);
      with_errors = true;
      first_mile_taxi_.clear();
    }

    auto const n_update_last_mile =
        n_rides_in_response(o.at("target").as_array());
    if (last_mile_taxi_.size() == n_update_last_mile) {
      with_errors |=
          update_pt_rides(last_mile_taxi_, o.at("target").as_array());
    } else {
      n::log(
          n::log_lvl::debug, "motis.prima",
          "[blacklist taxi] to_rides_.size() != n_update_last_mile ({} != {})",
          last_mile_taxi_.size(), n_update_last_mile);
      with_errors = true;
      last_mile_taxi_.clear();
    }

    if (direct_taxi_.size() == o.at("direct").as_array().size()) {
      with_errors |=
          update_direct_rides(direct_taxi_, o.at("direct").as_array());
    } else {
      n::log(n::log_lvl::debug, "motis.prima",
             "[blacklist taxi] direct_rides_.size() != n_direct_updates ({} != "
             "{})",
             direct_taxi_.size(), o.at("direct").as_array().size());
      with_errors = true;
      direct_taxi_.clear();
    }

  } catch (std::exception const&) {
    n::log(n::log_lvl::debug, "motis.prima",
           "[blacklist taxi] could not parse response: {}", json);
    return false;
  }
  if (with_errors) {
    n::log(n::log_lvl::debug, "motis.prima",
           "[blacklist taxi] parsed response with invalid values: {}", json);
    return false;
  }
  return true;
}

bool prima::blacklist_taxis(nigiri::timetable const& tt) {
  auto blacklist_response = std::optional<std::string>{};
  auto ioc = boost::asio::io_context{};
  try {
    n::log(n::log_lvl::debug, "motis.prima",
           "[blacklist taxi] request for {} events", n_taxi_events());
    boost::asio::co_spawn(
        ioc,
        [&]() -> boost::asio::awaitable<void> {
          auto const prima_msg = co_await http_POST(
              taxi_blacklist_, kReqHeaders, make_taxi_request(tt), 10s);
          blacklist_response = get_http_body(prima_msg);
        },
        boost::asio::detached);
    ioc.run();
  } catch (std::exception const& e) {
    n::log(n::log_lvl::debug, "motis.prima",
           "[blacklist taxi] networking failed: {}", e.what());
    blacklist_response = std::nullopt;
  }
  if (!blacklist_response) {
    return false;
  }

  return consume_blacklist_taxis_response(*blacklist_response);
}

void prima::extract_taxis(
    std::vector<nigiri::routing::journey> const& journeys) {
  first_mile_taxi_.clear();
  last_mile_taxi_.clear();
  for (auto const& j : journeys) {
    if (!j.legs_.empty()) {
      if (is_odm_leg(j.legs_.front(), kOdmTransportModeId)) {
        first_mile_taxi_.push_back({.time_at_start_ = j.legs_.front().dep_time_,
                                    .time_at_stop_ = j.legs_.front().arr_time_,
                                    .stop_ = j.legs_.front().to_});
      }
    }
    if (j.legs_.size() > 1) {
      if (is_odm_leg(j.legs_.back(), kOdmTransportModeId)) {
        last_mile_taxi_.push_back({.time_at_start_ = j.legs_.back().arr_time_,
                                   .time_at_stop_ = j.legs_.back().dep_time_,
                                   .stop_ = j.legs_.back().from_});
      }
    }
  }
  utl::erase_duplicates(first_mile_taxi_, by_stop, std::equal_to<>{});
  utl::erase_duplicates(last_mile_taxi_, by_stop, std::equal_to<>{});
}

void prima::fix_first_mile_duration(
    std::vector<nigiri::routing::journey>& journeys,
    std::vector<nigiri::routing::start> const& first_mile,
    std::vector<nigiri::routing::start> const& prev_first_mile,
    nigiri::transport_mode_id_t const mode) {
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
                                        : nigiri::duration_t{0});
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

void prima::fix_last_mile_duration(
    std::vector<nigiri::routing::journey>& journeys,
    std::vector<nigiri::routing::start> const& last_mile,
    std::vector<nigiri::routing::start> const& prev_last_mile,
    nigiri::transport_mode_id_t const mode) {
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
                                        : nigiri::duration_t{0});
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

bool prima::consume_whitelist_taxis_response(
    std::string_view json, std::vector<nigiri::routing::journey>& journeys) {

  auto const update_first_mile = [&](json::array const& update) {
    auto const n_pt_udpates = n_rides_in_response(update);
    if (first_mile_taxi_.size() != n_pt_udpates) {
      n::log(n::log_lvl::debug, "motis.prima",
             "[whitelist taxi] first mile taxi #rides != #updates ({} != {})",
             first_mile_taxi_.size(), n_pt_udpates);
      return true;
    }

    auto const prev_first_mile =
        std::exchange(first_mile_taxi_, std::vector<n::routing::start>{});

    auto prev_it = std::begin(prev_first_mile);
    for (auto const& stop : update) {
      for (auto const& event : stop.as_array()) {
        if (event.is_null()) {
          first_mile_taxi_.push_back({.time_at_start_ = kInfeasible,
                                      .time_at_stop_ = kInfeasible,
                                      .stop_ = prev_it->stop_});
        } else {
          first_mile_taxi_.push_back(
              {.time_at_start_ =
                   to_unix(event.as_object().at("pickupTime").as_int64()),
               .time_at_stop_ =
                   to_unix(event.as_object().at("dropoffTime").as_int64()),
               .stop_ = prev_it->stop_});
        }
        ++prev_it;
      }
    }
    fix_first_mile_duration(journeys, first_mile_taxi_, prev_first_mile,
                            kOdmTransportModeId);
    return false;
  };

  auto const update_last_mile = [&](json::array const& update) {
    auto const n_pt_udpates = n_rides_in_response(update);
    if (last_mile_taxi_.size() != n_pt_udpates) {
      n::log(n::log_lvl::debug, "motis.prima",
             "[whitelist taxi] last mile taxi #rides != #updates ({} != {})",
             last_mile_taxi_.size(), n_pt_udpates);
      return true;
    }

    auto const prev_last_mile =
        std::exchange(last_mile_taxi_, std::vector<n::routing::start>{});

    auto prev_it = std::begin(prev_last_mile);
    for (auto const& stop : update) {
      for (auto const& event : stop.as_array()) {
        if (event.is_null()) {
          last_mile_taxi_.push_back({.time_at_start_ = kInfeasible,
                                     .time_at_stop_ = kInfeasible,
                                     .stop_ = prev_it->stop_});
        } else {
          last_mile_taxi_.push_back(
              {.time_at_start_ =
                   to_unix(event.as_object().at("dropoffTime").as_int64()),
               .time_at_stop_ =
                   to_unix(event.as_object().at("pickupTime").as_int64()),
               .stop_ = prev_it->stop_});
        }
        ++prev_it;
      }
    }

    fix_last_mile_duration(journeys, last_mile_taxi_, prev_last_mile,
                           kOdmTransportModeId);
    return false;
  };

  auto const update_direct_rides = [&](json::array const& update) {
    if (direct_taxi_.size() != update.size()) {
      n::log(n::log_lvl::debug, "motis.prima",
             "[whitelist taxi] direct taxi #rides != #updates ({} != {})",
             direct_taxi_.size(), update.size());
      direct_taxi_.clear();
      return true;
    }

    direct_taxi_.clear();
    for (auto const& ride : update) {
      if (!ride.is_null()) {
        direct_taxi_.push_back(
            {to_unix(ride.as_object().at("pickupTime").as_int64()),
             to_unix(ride.as_object().at("dropoffTime").as_int64())});
      }
    }

    return false;
  };

  auto with_errors = false;
  try {
    auto const o = json::parse(json).as_object();
    with_errors |= update_first_mile(o.at("start").as_array());
    with_errors |= update_last_mile(o.at("target").as_array());
    with_errors |= update_direct_rides(o.at("direct").as_array());
  } catch (std::exception const&) {
    n::log(n::log_lvl::debug, "motis.prima",
           "[whitelist taxi] could not parse response: {}", json);
    return false;
  }
  if (with_errors) {
    n::log(n::log_lvl::debug, "motis.prima",
           "[whitelist taxi] parsed response with errors: {}", json);
    return false;
  }

  // adjust journey start/dest times after adjusting legs
  for (auto& j : journeys) {
    if (!j.legs_.empty()) {
      j.start_time_ = j.legs_.front().dep_time_;
      j.dest_time_ = j.legs_.back().arr_time_;
    }
  }

  return true;
}

bool prima::whitelist_taxis(
    std::vector<nigiri::routing::journey>& taxi_journeys,
    nigiri::timetable const& tt) {
  extract_taxis(taxi_journeys);

  auto whitelist_response = std::optional<std::string>{};
  auto ioc = boost::asio::io_context{};
  try {
    n::log(n::log_lvl::debug, "motis.prima",
           "[whitelist taxi] request for {} events", n_taxi_events());
    boost::asio::co_spawn(
        ioc,
        [&]() -> boost::asio::awaitable<void> {
          auto const prima_msg = co_await http_POST(
              taxi_whitelist_, kReqHeaders, make_taxi_request(tt), 10s);
          whitelist_response = get_http_body(prima_msg);
        },
        boost::asio::detached);
    ioc.run();
  } catch (std::exception const& e) {
    n::log(n::log_lvl::debug, "motis.prima",
           "[whitelist taxi] networking failed: {}", e.what());
    whitelist_response = std::nullopt;
  }
  if (!whitelist_response) {
    n::log(n::log_lvl::debug, "motis.prima",
           "[whitelist taxi] failed, discarding taxi journeys");
    return false;
  }

  return consume_whitelist_taxis_response(*whitelist_response, taxi_journeys);
}

void prima::add_direct_odm(std::vector<direct_ride> const& direct,
                           std::vector<nigiri::routing::journey>& odm_journeys,
                           place_t const& from,
                           place_t const& to,
                           bool arrive_by,
                           nigiri::transport_mode_id_t const mode) const {
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

bool prima::consume_whitelist_ride_sharing_response(std::string_view json) {
  auto const update_first_mile = [&](json::array const& update) {
    auto const n = n_rides_in_response(update);
    if (first_mile_ride_sharing_.size() != n) {
      n::log(n::log_lvl::debug, "motis.prima",
             "[whitelist taxi] first mile ride-sharing #rides != #updates ({} "
             "!= {})",
             first_mile_ride_sharing_.size(), n);
      first_mile_ride_sharing_.clear();
      return true;
    }

    auto prev_first_mile = std::exchange(first_mile_ride_sharing_,
                                         std::vector<n::routing::start>{});
    auto prev_it = std::begin(prev_first_mile);
    for (auto const& stop : update) {
      for (auto const& time : stop.as_array()) {
        if (!time.is_null() && time.is_array()) {
          for (auto const& event : time.as_array()) {
            first_mile_ride_sharing_.push_back(
                {.time_at_start_ =
                     to_unix(event.as_object().at("pickupTime").as_int64()),
                 .time_at_stop_ =
                     to_unix(event.as_object().at("dropoffTime").as_int64()) +
                     kODMTransferBuffer,
                 .stop_ = prev_it->stop_});
            first_mile_ride_sharing_tour_ids_.push_back(
                static_cast<std::uint32_t>(
                    event.as_object().at("tour").as_int64()));
          }
        }
        ++prev_it;
      }
    }
    return false;
  };

  auto const update_last_mile = [&](json::array const& update) {
    auto const n = n_rides_in_response(update);
    if (last_mile_ride_sharing_.size() != n) {
      n::log(n::log_lvl::debug, "motis.prima",
             "[whitelist taxi] last mile ride-sharing #rides != #updates ({} "
             "!= {})",
             last_mile_ride_sharing_.size(), n);
      last_mile_ride_sharing_.clear();
      return true;
    }

    auto prev_last_mile = std::exchange(last_mile_ride_sharing_,
                                        std::vector<n::routing::start>{});
    auto prev_it = std::begin(prev_last_mile);
    for (auto const& stop : update) {
      for (auto const& time : stop.as_array()) {
        if (!time.is_null() && time.is_array()) {
          for (auto const& event : time.as_array()) {
            last_mile_ride_sharing_.push_back(
                {.time_at_start_ =
                     to_unix(event.as_object().at("dropoffTime").as_int64()),
                 .time_at_stop_ =
                     to_unix(event.as_object().at("pickupTime").as_int64()) -
                     kODMTransferBuffer,
                 .stop_ = prev_it->stop_});
            last_mile_ride_sharing_tour_ids_.push_back(
                static_cast<std::uint32_t>(
                    event.as_object().at("tour").as_int64()));
          }
          ++prev_it;
        }
      }
    }
    return false;
  };

  auto const update_direct = [&](json::array const& update) {
    if (direct_ride_sharing_.size() != update.size()) {
      n::log(n::log_lvl::debug, "motis.prima",
             "[whitelist ride-sharing] direct ride-sharing #rides != "
             "#updates "
             "({} != {})",
             direct_ride_sharing_.size(), update.size());
      direct_ride_sharing_.clear();
      return true;
    }

    direct_ride_sharing_.clear();
    for (auto const& time : update) {
      if (time.is_array()) {
        for (auto const& ride : time.as_array()) {
          if (!ride.is_null()) {
            direct_ride_sharing_.push_back(
                {to_unix(ride.as_object().at("pickupTime").as_int64()),
                 to_unix(ride.as_object().at("dropoffTime").as_int64())});
            direct_ride_sharing_tour_ids_.push_back(static_cast<std::uint32_t>(
                ride.as_object().at("tour").as_int64()));
          }
        }
      }
    }

    return false;
  };

  auto with_errors = false;
  try {
    auto const o = json::parse(json).as_object();
    with_errors |= update_first_mile(o.at("start").as_array());
    with_errors |= update_last_mile(o.at("target").as_array());
    with_errors |= update_direct(o.at("direct").as_array());
  } catch (std::exception const&) {
    n::log(n::log_lvl::debug, "motis.prima",
           "[whitelist ride-sharing] could not parse response: {}", json);
    return false;
  }
  if (with_errors) {
    n::log(n::log_lvl::debug, "motis.prima",
           "[whitelist ride-sharing] parsed response with errors: {}", json);
    return false;
  }

  return true;
}

bool prima::whitelist_ride_sharing(nigiri::timetable const& tt) {
  auto response = std::optional<std::string>{};
  auto ioc = boost::asio::io_context{};
  try {
    n::log(n::log_lvl::debug, "motis.prima",
           "[whitelist ride-sharing] request for {} events",
           n_ride_sharing_events());
    boost::asio::co_spawn(
        ioc,
        [&]() -> boost::asio::awaitable<void> {
          auto const prima_msg =
              co_await http_POST(ride_sharing_whitelist_, kReqHeaders,
                                 make_ride_sharing_request(tt), 30s);
          response = get_http_body(prima_msg);
        },
        boost::asio::detached);
    ioc.run();
  } catch (std::exception const& e) {
    n::log(n::log_lvl::debug, "motis.prima",
           "[whitelist ride-sharing] networking failed: {}", e.what());
    response = std::nullopt;
  }
  if (!response) {
    n::log(n::log_lvl::debug, "motis.prima",
           "[whitelist ride share] failed, discarding ride share journeys");
    return false;
  }

  return consume_whitelist_ride_sharing_response(*response);
}
}  // namespace motis::odm