#if defined(_MSC_VER)
// needs to be the first to include WinSock.h
#include "boost/asio.hpp"
#endif

#include "motis/odm/meta_router.h"

#include <vector>

#include "fmt/std.h"

#include "boost/asio/co_spawn.hpp"
#include "boost/asio/detached.hpp"
#include "boost/asio/io_context.hpp"
#include "boost/fiber/future.hpp"
#include "boost/fiber/future/packaged_task.hpp"
#include "boost/thread/tss.hpp"

#include "prometheus/histogram.h"

#include "utl/erase_duplicates.h"

#include "nigiri/logging.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor_search.h"
#include "nigiri/routing/start_times.h"
#include "nigiri/types.h"

#include "osr/routing/route.h"

#include "motis-api/motis-api.h"
#include "motis/constants.h"
#include "motis/elevators/elevators.h"
#include "motis/endpoints/routing.h"
#include "motis/gbfs/routing_data.h"
#include "motis/http_req.h"
#include "motis/journey_to_response.h"
#include "motis/metrics_registry.h"
#include "motis/odm/bounds.h"
#include "motis/odm/mixer.h"
#include "motis/odm/odm.h"
#include "motis/odm/prima.h"
#include "motis/odm/shorten.h"
#include "motis/place.h"
#include "motis/street_routing.h"
#include "motis/timetable/modes_to_clasz_mask.h"
#include "motis/timetable/time_conv.h"
#include "motis/transport_mode_ids.h"

namespace motis::odm {

namespace n = nigiri;
using namespace std::chrono_literals;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static boost::thread_specific_ptr<prima> p;

constexpr auto const kODMLookAhead = 48h;
constexpr auto const kSearchIntervalSize = 24h;
constexpr auto const kODMDirectPeriod = 1h;
constexpr auto const kODMDirectFactor = 1.0;
constexpr auto const kODMOffsetMinImprovement = 60s;
constexpr auto const kODMMaxDuration = 3600s;
constexpr auto const kBlacklistPath = "/api/blacklist";
constexpr auto const kWhitelistPath = "/api/whitelist";
static auto const kReqHeaders = std::map<std::string, std::string>{
    {"Content-Type", "application/json"}, {"Accept", "application/json"}};
static auto const kMixer = get_default_mixer();

using td_offsets_t =
    n::hash_map<n::location_idx_t, std::vector<n::routing::td_offset>>;

void print_time(auto const& start,
                std::string_view name,
                prometheus::Histogram& metric) {
  auto const millis = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start);
  n::log(n::log_lvl::debug, "motis.odm", "{} {}", name, millis);
  metric.Observe(static_cast<double>(millis.count()) / 1000.0);
}

meta_router::meta_router(ep::routing const& r,
                         api::plan_params const& query,
                         std::vector<api::ModeEnum> const& pre_transit_modes,
                         std::vector<api::ModeEnum> const& post_transit_modes,
                         std::vector<api::ModeEnum> const& direct_modes,
                         std::variant<osr::location, tt_location> const& from,
                         std::variant<osr::location, tt_location> const& to,
                         api::Place const& from_p,
                         api::Place const& to_p,
                         nigiri::routing::query const& start_time,
                         std::vector<api::Itinerary>& direct,
                         nigiri::duration_t const fastest_direct,
                         bool const odm_pre_transit,
                         bool const odm_post_transit,
                         bool const odm_direct,
                         unsigned const api_version)
    : r_{r},
      query_{query},
      pre_transit_modes_{pre_transit_modes},
      post_transit_modes_{post_transit_modes},
      direct_modes_{direct_modes},
      from_{from},
      to_{to},
      from_place_{from_p},
      to_place_{to_p},
      start_time_{start_time},
      direct_{direct},
      fastest_direct_{fastest_direct},
      odm_pre_transit_{odm_pre_transit},
      odm_post_transit_{odm_post_transit},
      odm_direct_{odm_direct},
      api_version_{api_version},
      tt_{r_.tt_},
      rt_{r.rt_},
      rtt_{rt_->rtt_.get()},
      e_{rt_->e_.get()},
      gbfs_rd_{r.w_, r.l_, r.gbfs_},
      start_{query_.arriveBy_ ? to_ : from_},
      dest_{query_.arriveBy_ ? from_ : to_},
      start_modes_{query_.arriveBy_ ? post_transit_modes_ : pre_transit_modes_},
      dest_modes_{query_.arriveBy_ ? pre_transit_modes_ : post_transit_modes_},
      start_form_factors_{query_.arriveBy_
                              ? query_.postTransitRentalFormFactors_
                              : query_.preTransitRentalFormFactors_},
      dest_form_factors_{query_.arriveBy_
                             ? query_.preTransitRentalFormFactors_
                             : query_.postTransitRentalFormFactors_},
      start_propulsion_types_{query_.arriveBy_
                                  ? query_.postTransitRentalPropulsionTypes_
                                  : query_.preTransitRentalPropulsionTypes_},
      dest_propulsion_types_{query_.arriveBy_
                                 ? query_.preTransitRentalPropulsionTypes_
                                 : query_.postTransitRentalPropulsionTypes_},
      start_rental_providers_{query_.arriveBy_
                                  ? query_.postTransitRentalProviders_
                                  : query_.preTransitRentalProviders_},
      dest_rental_providers_{query_.arriveBy_
                                 ? query_.preTransitRentalProviders_
                                 : query_.postTransitRentalProviders_},
      start_ignore_rental_return_constraints_{
          query.arriveBy_ ? query_.ignorePreTransitRentalReturnConstraints_
                          : query_.ignorePostTransitRentalReturnConstraints_},
      dest_ignore_rental_return_constraints_{
          query.arriveBy_ ? query_.ignorePostTransitRentalReturnConstraints_
                          : query_.ignorePreTransitRentalReturnConstraints_} {
  if (ep::blocked.get() == nullptr && r.w_ != nullptr) {
    ep::blocked.reset(new osr::bitvec<osr::node_idx_t>{r.w_->n_nodes()});
  }
}

n::duration_t init_direct(std::vector<direct_ride>& direct_rides,
                          ep::routing const& r,
                          elevators const* e,
                          gbfs::gbfs_routing_data& gbfs,
                          api::Place const& from_p,
                          api::Place const& to_p,
                          n::interval<n::unixtime_t> const intvl,
                          api::plan_params const& query,
                          unsigned api_version) {
  direct_rides.clear();

  auto const from_pos = geo::latlng{from_p.lat_, from_p.lon_};
  auto const to_pos = geo::latlng{to_p.lat_, to_p.lon_};
  if (r.odm_bounds_ != nullptr && (!r.odm_bounds_->contains(from_pos) ||
                                   !r.odm_bounds_->contains(to_pos))) {
    n::log(n::log_lvl::debug, "motis.odm",
           "No direct connection, from: {}, to: {}", from_pos, to_pos);
    return ep::kInfinityDuration;
  }

  auto [_, odm_direct_duration] = r.route_direct(
      e, gbfs, from_p, to_p, {api::ModeEnum::CAR}, std::nullopt, std::nullopt,
      std::nullopt, false, intvl.from_, false, query.pedestrianProfile_,
      query.elevationCosts_, kODMMaxDuration, query.maxMatchingDistance_,
      kODMDirectFactor, api_version);

  if (odm_direct_duration < kODMMaxDuration) {
    if (query.arriveBy_) {
      for (auto arr = std::chrono::floor<std::chrono::hours>(
                          intvl.to_ - odm_direct_duration) +
                      odm_direct_duration;
           intvl.contains(arr); arr -= kODMDirectPeriod) {
        direct_rides.push_back(
            {.dep_ = arr - odm_direct_duration, .arr_ = arr});
      }
    } else {
      for (auto dep = std::chrono::ceil<std::chrono::hours>(intvl.from_);
           intvl.contains(dep); dep += kODMDirectPeriod) {
        direct_rides.push_back(
            {.dep_ = dep, .arr_ = dep + odm_direct_duration});
      }
    }
  } else {
    fmt::println(
        "[init] No direct ODM connection, from: {}, to: {}: "
        "odm_direct_duration >= "
        "kODMMaxDuration ({} "
        ">= {})",
        from_pos, to_pos, odm_direct_duration, kODMMaxDuration);
  }

  return odm_direct_duration;
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
  if (r.odm_bounds_ != nullptr && !r.odm_bounds_->contains(l.pos_)) {
    n::log(n::log_lvl::debug, "motis.odm",
           "no ODM-PT connection at {}: terminal out of bounds", l.pos_);
    return;
  }

  auto offsets = r.get_offsets(rtt, l, dir, {api::ModeEnum::ODM}, std::nullopt,
                               std::nullopt, std::nullopt, false,
                               query.pedestrianProfile_, query.elevationCosts_,
                               max, query.maxMatchingDistance_, gbfs_rd);

  std::erase_if(offsets, [&](n::routing::offset const& o) {
    auto const out_of_bounds =
        (r.odm_bounds_ != nullptr &&
         !r.odm_bounds_->contains(r.tt_->locations_.coordinates_[o.target_]));
    /*if (out_of_bounds) {
      n::log(n::log_lvl::debug, "motis.odm", "Bounds filtered: {}",
             n::location{*r.tt_, o.target_});
    }*/
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

void meta_router::init_prima(n::interval<n::unixtime_t> const& search_intvl,
                             n::interval<n::unixtime_t> const& odm_intvl) {
  if (p.get() == nullptr) {
    p.reset(new prima{});
  }

  p->init(from_place_, to_place_, query_);

  auto direct_duration = std::optional<std::chrono::seconds>{};
  if (odm_direct_ && r_.w_ && r_.l_) {
    direct_duration =
        init_direct(p->direct_rides_, r_, e_, gbfs_rd_, from_place_, to_place_,
                    search_intvl, query_, api_version_);
  }

  auto const max_offset_duration =
      direct_duration
          ? std::min(std::max(*direct_duration, kODMOffsetMinImprovement) -
                         kODMOffsetMinImprovement,
                     kODMMaxDuration)
          : kODMMaxDuration;

  if (odm_pre_transit_ && holds_alternative<osr::location>(from_)) {
    init_pt(p->from_rides_, r_, std::get<osr::location>(from_),
            osr::direction::kForward, query_, gbfs_rd_, *tt_, rtt_, odm_intvl,
            start_time_,
            query_.arriveBy_ ? start_time_.dest_match_mode_
                             : start_time_.start_match_mode_,
            max_offset_duration);
  }

  if (odm_post_transit_ && holds_alternative<osr::location>(to_)) {
    init_pt(p->to_rides_, r_, std::get<osr::location>(to_),
            osr::direction::kBackward, query_, gbfs_rd_, *tt_, rtt_, odm_intvl,
            start_time_,
            query_.arriveBy_ ? start_time_.start_match_mode_
                             : start_time_.dest_match_mode_,
            max_offset_duration);
  }

  std::erase(start_modes_, api::ModeEnum::ODM);
  std::erase(dest_modes_, api::ModeEnum::ODM);
}

bool ride_comp(n::routing::start const& a, n::routing::start const& b) {
  return std::tie(a.stop_, a.time_at_start_, a.time_at_stop_) <
         std::tie(b.stop_, b.time_at_start_, b.time_at_stop_);
}

auto ride_time_halves(std::vector<n::routing::start>& rides) {
  auto const by_duration = [&](auto const& a, auto const& b) {
    return duration(a) < duration(b);
  };

  utl::sort(rides, by_duration);
  auto const split =
      rides.empty() ? 0
                    : std::distance(begin(rides),
                                    std::upper_bound(begin(rides), end(rides),
                                                     rides[rides.size() / 2],
                                                     by_duration));

  auto lo = rides | std::views::take(split);
  auto hi = rides | std::views::drop(split);
  std::ranges::sort(lo, ride_comp);
  std::ranges::sort(hi, ride_comp);
  return std::pair{lo, hi};
}

auto get_td_offsets(auto const& rides) {
  auto td_offsets = td_offsets_t{};
  utl::equal_ranges_linear(
      rides, [](auto const& a, auto const& b) { return a.stop_ == b.stop_; },
      [&](auto&& from_it, auto&& to_it) {
        td_offsets.emplace(from_it->stop_,
                           std::vector<n::routing::td_offset>{});
        for (auto const& r : n::it_range{from_it, to_it}) {
          auto const dep = std::min(r.time_at_stop_, r.time_at_start_);
          auto const dur = std::chrono::abs(r.time_at_stop_ - r.time_at_start_);
          if (td_offsets.at(from_it->stop_).size() > 1) {
            auto last = rbegin(td_offsets.at(from_it->stop_));
            auto const second_last = std::next(last);
            if (dep ==
                std::clamp(dep, second_last->valid_from_, last->valid_from_)) {
              // increase validity interval of last offset
              last->valid_from_ = dep + dur;
              continue;
            }
          }
          // add new offset
          td_offsets.at(from_it->stop_)
              .push_back({.valid_from_ = dep,
                          .duration_ = dur,
                          .transport_mode_id_ = kOdmTransportModeId});
          td_offsets.at(from_it->stop_)
              .push_back({.valid_from_ = dep + dur,
                          .duration_ = n::footpath::kMaxDuration,
                          .transport_mode_id_ = kOdmTransportModeId});
        }
      });
  return td_offsets;
}

n::routing::query meta_router::get_base_query(
    n::interval<n::unixtime_t> const& intvl) const {
  return {
      .start_time_ = intvl,
      .start_match_mode_ = motis::ep::get_match_mode(start_),
      .dest_match_mode_ = motis::ep::get_match_mode(dest_),
      .use_start_footpaths_ = !motis::ep::is_intermodal(start_),
      .max_transfers_ = static_cast<std::uint8_t>(
          query_.maxTransfers_.has_value() ? *query_.maxTransfers_
                                           : n::routing::kMaxTransfers),
      .max_travel_time_ = query_.maxTravelTime_
                              .and_then([](std::int64_t const dur) {
                                return std::optional{n::duration_t{dur}};
                              })
                              .value_or(ep::kInfinityDuration),
      .min_connection_count_ = 0U,
      .extend_interval_earlier_ = false,
      .extend_interval_later_ = false,
      .max_interval_ = std::nullopt,
      .prf_idx_ = static_cast<n::profile_idx_t>(
          query_.useRoutedTransfers_
              ? (query_.pedestrianProfile_ ==
                         api::PedestrianProfileEnum::WHEELCHAIR
                     ? 2U
                     : 1U)
              : 0U),
      .allowed_claszes_ = to_clasz_mask(query_.transitModes_),
      .require_bike_transport_ = query_.requireBikeTransport_,
      .require_car_transport_ = query_.requireCarTransport_,
      .transfer_time_settings_ =
          n::routing::transfer_time_settings{
              .default_ = (query_.minTransferTime_ == 0 &&
                           query_.additionalTransferTime_ == 0 &&
                           query_.transferTimeFactor_ == 1.0),
              .min_transfer_time_ = n::duration_t{query_.minTransferTime_},
              .additional_time_ = n::duration_t{query_.additionalTransferTime_},
              .factor_ = static_cast<float>(query_.transferTimeFactor_)},
      .via_stops_ = motis::ep::get_via_stops(*tt_, *r_.tags_, query_.via_,
                                             query_.viaMinimumStay_),
      .fastest_direct_ = fastest_direct_ == ep::kInfinityDuration
                             ? std::nullopt
                             : std::optional{fastest_direct_}};
}

std::vector<meta_router::routing_result> meta_router::search_interval(
    std::vector<n::routing::query> const& sub_queries) const {

  auto const make_task = [&](n::routing::query q) {
    return boost::fibers::packaged_task<routing_result()>{
        [&, q = std::move(q)]() mutable {
          if (ep::search_state.get() == nullptr) {
            ep::search_state.reset(new n::routing::search_state{});
          }
          if (ep::raptor_state.get() == nullptr) {
            ep::raptor_state.reset(new n::routing::raptor_state{});
          }

          auto const timeout = std::chrono::seconds{query_.timeout_.value_or(
              r_.config_.limits_.value().routing_max_timeout_seconds_)};

          return routing_result{raptor_search(
              *tt_, rtt_, *ep::search_state, *ep::raptor_state, std::move(q),
              query_.arriveBy_ ? n::direction::kBackward
                               : n::direction::kForward,
              timeout)};
        }};
  };

  auto tasks = std::vector<boost::fibers::packaged_task<routing_result()>>{};
  for (auto& q : sub_queries) {
    tasks.push_back(make_task(q));
  }

  auto futures = utl::to_vec(tasks, [](auto& t) { return t.get_future(); });

  for (auto& task : tasks) {
    boost::fibers::fiber(std::move(task)).detach();
  }

  for (auto const& f : futures) {
    f.wait();
  }

  auto results = std::vector<meta_router::routing_result>{};
  for (auto& f : futures) {
    results.push_back(f.get());
  }

  return results;
}

void collect_odm_journeys(
    std::vector<meta_router::routing_result> const& results) {
  p->odm_journeys_.clear();
  for (auto const& r : results | std::views::drop(1)) {
    for (auto const& j : r.journeys_) {
      if (uses_odm(j)) {
        p->odm_journeys_.push_back(j);
      }
    }
  }
  n::log(n::log_lvl::debug, "motis.odm",
         "[routing] collected {} ODM-PT journeys", p->odm_journeys_.size());
}

void extract_rides() {
  p->from_rides_.clear();
  p->to_rides_.clear();
  for (auto const& j : p->odm_journeys_) {
    if (!j.legs_.empty()) {
      if (is_odm_leg(j.legs_.front())) {
        p->from_rides_.push_back({.time_at_start_ = j.legs_.front().dep_time_,
                                  .time_at_stop_ = j.legs_.front().arr_time_,
                                  .stop_ = j.legs_.front().to_});
      }
    }
    if (j.legs_.size() > 1) {
      if (is_odm_leg(j.legs_.back())) {
        p->to_rides_.push_back({.time_at_start_ = j.legs_.back().arr_time_,
                                .time_at_stop_ = j.legs_.back().dep_time_,
                                .stop_ = j.legs_.back().from_});
      }
    }
  }

  utl::erase_duplicates(p->from_rides_, ride_comp, std::equal_to<>{});
  utl::erase_duplicates(p->to_rides_, ride_comp, std::equal_to<>{});
}

void meta_router::add_direct() const {
  auto from_l = std::visit(
      utl::overloaded{[](osr::location const&) {
                        return get_special_station(n::special_station::kStart);
                      },
                      [](tt_location const& tt_l) { return tt_l.l_; }},
      from_);
  auto to_l = std::visit(
      utl::overloaded{[](osr::location const&) {
                        return get_special_station(n::special_station::kEnd);
                      },
                      [](tt_location const& tt_l) { return tt_l.l_; }},
      to_);

  if (query_.arriveBy_) {
    std::swap(from_l, to_l);
  }

  for (auto const& d : p->direct_rides_) {
    p->odm_journeys_.push_back(n::routing::journey{
        .legs_ = {{n::direction::kForward, from_l, to_l, d.dep_, d.arr_,
                   n::routing::offset{to_l, std::chrono::abs(d.arr_ - d.dep_),
                                      kOdmTransportModeId}}},
        .start_time_ = d.dep_,
        .dest_time_ = d.arr_,
        .dest_ = to_l,
        .transfers_ = 0U});
  }
  n::log(n::log_lvl::debug, "motis.odm", "[whitelisting] added {} direct rides",
         p->direct_rides_.size());
}

api::plan_response meta_router::run() {
  // init
  auto const init_start = std::chrono::steady_clock::now();
  utl::verify(r_.tt_ != nullptr && r_.tags_ != nullptr,
              "mode=TRANSIT requires timetable to be loaded");
  auto stats = motis::ep::stats_map_t{};
  auto const start_intvl = std::visit(
      utl::overloaded{[](n::interval<n::unixtime_t> const i) { return i; },
                      [](n::unixtime_t const t) {
                        return n::interval<n::unixtime_t>{t, t};
                      }},
      start_time_.start_time_);
  auto const odm_intvl =
      !start_time_.extend_interval_later_
          ? n::interval<n::unixtime_t>{start_intvl.to_ - kODMLookAhead,
                                       start_intvl.to_ +
                                           std::chrono::minutes{
                                               kMixer.max_distance_}}
          : n::interval<n::unixtime_t>{
                start_intvl.from_ - std::chrono::minutes{kMixer.max_distance_},
                start_intvl.from_ + kODMLookAhead};
  auto const search_intvl =
      !start_time_.extend_interval_later_
          ? n::interval<n::unixtime_t>{start_intvl.to_ - kSearchIntervalSize,
                                       start_intvl.to_}
          : n::interval<n::unixtime_t>{start_intvl.from_,
                                       start_intvl.from_ + kSearchIntervalSize};
  auto const context_intvl =
      !start_time_.extend_interval_later_
          ? n::interval<n::unixtime_t>{search_intvl.from_,
                                       search_intvl.to_ +
                                           std::chrono::minutes{
                                               kMixer.max_distance_}}
          : n::interval<n::unixtime_t>{
                search_intvl.from_ - std::chrono::minutes{kMixer.max_distance_},
                search_intvl.to_};

  init_prima(context_intvl, odm_intvl);
  print_time(init_start,
             fmt::format("[init] (first_mile: {}, last_mile: {}, direct: {})",
                         p->from_rides_.size(), p->to_rides_.size(),
                         p->direct_rides_.size()),
             r_.metrics_->routing_execution_duration_seconds_init_);

  // blacklisting
  auto blacklist_response = std::optional<std::string>{};
  auto ioc = boost::asio::io_context{};
  auto const bl_start = std::chrono::steady_clock::now();
  try {
    n::log(n::log_lvl::debug, "motis.odm",
           "[blacklisting] request for {} events", p->n_events());
    boost::asio::co_spawn(
        ioc,
        [&]() -> boost::asio::awaitable<void> {
          auto const prima_msg = co_await http_POST(
              boost::urls::url{r_.config_.odm_->url_ + kBlacklistPath},
              kReqHeaders, p->get_prima_request(*tt_), 10s);
          blacklist_response = get_http_body(prima_msg);
        },
        boost::asio::detached);
    ioc.run();
  } catch (std::exception const& e) {
    n::log(n::log_lvl::debug, "motis.odm",
           "[blacklisting] networking failed: {}", e.what());
    blacklist_response = std::nullopt;
  }
  auto const blacklisted =
      blacklist_response && p->blacklist_update(*blacklist_response);
  n::log(n::log_lvl::debug, "motis.odm",
         "[blacklisting] ODM events after blacklisting: {}", p->n_events());
  print_time(
      bl_start,
      fmt::format("[blacklisting] (first_mile: {}, last_mile: {}, direct: {})",
                  p->from_rides_.size(), p->to_rides_.size(),
                  p->direct_rides_.size()),
      r_.metrics_->routing_execution_duration_seconds_blacklisting_);
  r_.metrics_->routing_odm_journeys_found_blacklist_.Observe(
      static_cast<double>(p->n_events()));

  // prepare queries
  auto const prep_queries_start = std::chrono::steady_clock::now();

  auto const [from_rides_short, from_rides_long] =
      ride_time_halves(p->from_rides_);
  auto const [to_rides_short, to_rides_long] = ride_time_halves(p->to_rides_);

  auto const qf = query_factory{
      .base_query_ = get_base_query(context_intvl),
      .start_walk_ = r_.get_offsets(
          rtt_, start_,
          query_.arriveBy_ ? osr::direction::kBackward
                           : osr::direction::kForward,
          start_modes_, start_form_factors_, start_propulsion_types_,
          start_rental_providers_, start_ignore_rental_return_constraints_,
          query_.pedestrianProfile_, query_.elevationCosts_,
          std::chrono::seconds{query_.maxPreTransitTime_},
          query_.maxMatchingDistance_, gbfs_rd_),
      .dest_walk_ = r_.get_offsets(
          rtt_, dest_,
          query_.arriveBy_ ? osr::direction::kForward
                           : osr::direction::kBackward,
          dest_modes_, dest_form_factors_, dest_propulsion_types_,
          dest_rental_providers_, dest_ignore_rental_return_constraints_,
          query_.pedestrianProfile_, query_.elevationCosts_,
          std::chrono::seconds{query_.maxPostTransitTime_},
          query_.maxMatchingDistance_, gbfs_rd_),
      .td_start_walk_ = r_.get_td_offsets(
          rtt_, e_, start_,
          query_.arriveBy_ ? osr::direction::kBackward
                           : osr::direction::kForward,
          start_modes_, query_.pedestrianProfile_, query_.elevationCosts_,
          query_.maxMatchingDistance_,
          std::chrono::seconds{query_.maxPreTransitTime_}, context_intvl),
      .td_dest_walk_ = r_.get_td_offsets(
          rtt_, e_, dest_,
          query_.arriveBy_ ? osr::direction::kForward
                           : osr::direction::kBackward,
          dest_modes_, query_.pedestrianProfile_, query_.elevationCosts_,
          query_.maxMatchingDistance_,
          std::chrono::seconds{query_.maxPostTransitTime_}, context_intvl),
      .odm_start_short_ = query_.arriveBy_ ? get_td_offsets(to_rides_short)
                                           : get_td_offsets(from_rides_short),
      .odm_start_long_ = query_.arriveBy_ ? get_td_offsets(to_rides_long)
                                          : get_td_offsets(from_rides_long),
      .odm_dest_short_ = query_.arriveBy_ ? get_td_offsets(from_rides_short)
                                          : get_td_offsets(to_rides_short),
      .odm_dest_long_ = query_.arriveBy_ ? get_td_offsets(from_rides_long)
                                         : get_td_offsets(to_rides_long)};
  print_time(prep_queries_start, "[prepare queries]",
             r_.metrics_->routing_execution_duration_seconds_preparing_);

  auto const routing_start = std::chrono::steady_clock::now();
  auto sub_queries = qf.make_queries(blacklisted);
  auto const results = search_interval(sub_queries);
  auto const& pt_result = results.front();
  collect_odm_journeys(results);
  shorten(p->odm_journeys_, p->from_rides_, p->to_rides_, *tt_, rtt_, query_);
  utl::erase_duplicates(
      p->odm_journeys_,
      [](auto const& a, auto const& b) {
        return a.start_time_ < b.start_time_;
      },
      [](auto const& a, auto const& b) {
        return a == b &&
               odm_time(a.legs_.front()) == odm_time(b.legs_.front()) &&
               odm_time(a.legs_.back()) == odm_time(b.legs_.back());
      });
  n::log(n::log_lvl::debug, "motis.odm", "[routing] interval searched: {}",
         pt_result.interval_);
  print_time(routing_start, "[routing]",
             r_.metrics_->routing_execution_duration_seconds_routing_);

  // whitelisting
  auto const wl_start = std::chrono::steady_clock::now();
  auto whitelist_response = std::optional<std::string>{};
  auto ioc2 = boost::asio::io_context{};
  if (blacklisted) {
    extract_rides();
    try {
      n::log(n::log_lvl::debug, "motis.odm",
             "[whitelisting] request for {} events", p->n_events());
      boost::asio::co_spawn(
          ioc2,
          [&]() -> boost::asio::awaitable<void> {
            auto const prima_msg = co_await http_POST(
                boost::urls::url{r_.config_.odm_->url_ + kWhitelistPath},
                kReqHeaders, p->get_prima_request(*tt_), 10s);
            whitelist_response = get_http_body(prima_msg);
          },
          boost::asio::detached);
      ioc2.run();
    } catch (std::exception const& e) {
      n::log(n::log_lvl::debug, "motis.odm",
             "[whitelisting] networking failed: {}", e.what());
      whitelist_response = std::nullopt;
    }
  }

  // mixing
  auto const mixing_start = std::chrono::steady_clock::now();
  auto const whitelisted =
      whitelist_response && p->whitelist_update(*whitelist_response);
  if (whitelisted) {
    p->adjust_to_whitelisting();
    add_direct();
  } else {
    p->odm_journeys_.clear();
    n::log(n::log_lvl::debug, "motis.odm",
           "[whitelisting] failed, discarding ODM journeys");
  }
  print_time(
      wl_start,
      fmt::format("[whitelisting] (first_mile: {}, last_mile: {}, direct: {})",
                  p->from_rides_.size(), p->to_rides_.size(),
                  p->direct_rides_.size()),
      r_.metrics_->routing_execution_duration_seconds_whitelisting_);
  r_.metrics_->routing_odm_journeys_found_whitelist_.Observe(
      static_cast<double>(p->odm_journeys_.size()));
  n::log(n::log_lvl::debug, "motis.odm",
         "[mixing] {} PT journeys and {} ODM journeys",
         pt_result.journeys_.size(), p->odm_journeys_.size());

  kMixer.mix(pt_result.journeys_, p->odm_journeys_, r_.metrics_);

  r_.metrics_->routing_odm_journeys_found_non_dominated_.Observe(
      static_cast<double>(p->odm_journeys_.size() -
                          pt_result.journeys_.size()));

  print_time(mixing_start, "[mixing]",
             r_.metrics_->routing_execution_duration_seconds_mixing_);

  // remove journeys added for mixing context
  std::erase_if(p->odm_journeys_, [&](auto const& j) {
    return !search_intvl.contains(j.start_time_);
  });

  r_.metrics_->routing_journeys_found_.Increment(
      static_cast<double>(p->odm_journeys_.size()));
  r_.metrics_->routing_execution_duration_seconds_total_.Observe(
      static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now() - init_start)
                              .count()) /
      1000.0);

  return {
      .from_ = from_place_,
      .to_ = to_place_,
      .direct_ = std::move(direct_),
      .itineraries_ = utl::to_vec(
          p->odm_journeys_,
          [&, cache = street_routing_cache_t{}](auto&& j) mutable {
            r_.metrics_->routing_journey_duration_seconds_.Observe(
                static_cast<double>(
                    to_seconds(j.arrival_time() - j.departure_time())));
            return journey_to_response(
                r_.w_, r_.l_, r_.pl_, *tt_, *r_.tags_, r_.fa_, e_, rtt_,
                r_.matches_, r_.elevations_, r_.shapes_, gbfs_rd_, j, start_,
                dest_, cache, ep::blocked.get(),
                query_.requireCarTransport_ && query_.useRoutedTransfers_,
                query_.pedestrianProfile_, query_.elevationCosts_,
                query_.joinInterlinedLegs_, query_.detailedTransfers_,
                query_.withFares_, query_.withScheduledSkippedStops_,
                r_.config_.timetable_.value().max_matching_distance_,
                query_.maxMatchingDistance_, api_version_,
                query_.ignorePreTransitRentalReturnConstraints_,
                query_.ignorePostTransitRentalReturnConstraints_,
                query_.language_);
          }),
      .previousPageCursor_ =
          fmt::format("EARLIER|{}", to_seconds(search_intvl.from_)),
      .nextPageCursor_ = fmt::format("LATER|{}", to_seconds(search_intvl.to_))};
}

}  // namespace motis::odm
