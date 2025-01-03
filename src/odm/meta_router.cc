#include "motis/odm/meta_router.h"

#include <vector>

#include "boost/asio/co_spawn.hpp"
#include "boost/asio/detached.hpp"
#include "boost/asio/io_context.hpp"
#include "boost/fiber/future.hpp"
#include "boost/fiber/future/packaged_task.hpp"
#include "boost/thread/tss.hpp"

#include "nigiri/routing/journey.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/start_times.h"
#include "nigiri/types.h"

#include "osr/routing/route.h"

#include "motis-api/motis-api.h"
#include "motis/elevators/elevators.h"
#include "motis/endpoints/routing.h"
#include "motis/gbfs/routing_data.h"
#include "motis/http_req.h"
#include "motis/journey_to_response.h"
#include "motis/odm/json.h"
#include "motis/odm/mix.h"
#include "motis/odm/prima_state.h"
#include "motis/odm/query_factory.h"
#include "motis/odm/raptor_wrapper.h"
#include "motis/place.h"
#include "motis/street_routing.h"
#include "motis/timetable/modes_to_clasz_mask.h"
#include "motis/timetable/time_conv.h"

namespace motis::odm {

namespace n = nigiri;
using namespace std::chrono_literals;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static boost::thread_specific_ptr<prima_state> p_state;

static auto const kBlacklistingUrl = boost::urls::url{""};
static auto const kWhitelistingUrl = boost::urls::url{""};
static auto const kPrimaHeaders = std::map<std::string, std::string>{
    {"Content-Type", "application/json"}, {"Accept", "application/json"}};

constexpr auto const kInfinityDuration =
    n::duration_t{std::numeric_limits<n::duration_t::rep>::max()};

using td_offsets_t =
    n::hash_map<n::location_idx_t, std::vector<n::routing::td_offset>>;

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
                         bool const odm_direct)
    : r_{r},
      query_{query},
      pre_transit_modes_{pre_transit_modes},
      post_transit_modes_{post_transit_modes},
      direct_modes_{direct_modes},
      from_{from},
      to_{to},
      from_p_{from_p},
      to_p_{to_p},
      start_time_{start_time},
      direct_{direct},
      fastest_direct_{fastest_direct},
      odm_pre_transit_{odm_pre_transit},
      odm_post_transit_{odm_post_transit},
      odm_direct_{odm_direct},
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
                                 ? query_.postTransitRentalPropulsionTypes_
                                 : query_.preTransitRentalPropulsionTypes_},
      start_rental_providers_{query_.arriveBy_
                                  ? query_.postTransitRentalProviders_
                                  : query_.preTransitRentalProviders_},
      dest_rental_providers_{query_.arriveBy_
                                 ? query_.preTransitRentalProviders_
                                 : query_.postTransitRentalProviders_} {
  if (ep::blocked.get() == nullptr && r.w_ != nullptr) {
    ep::blocked.reset(new osr::bitvec<osr::node_idx_t>{r.w_->n_nodes()});
  }
}

n::interval<n::unixtime_t> get_dest_intvl(
    n::direction dir, n::interval<n::unixtime_t> const& start_intvl) {
  return dir == n::direction::kForward
             ? n::interval<n::unixtime_t>{start_intvl.from_,
                                          start_intvl.to_ + 24h}
             : n::interval<n::unixtime_t>{start_intvl.from_ - 24h,
                                          start_intvl.to_};
}

void init(prima_state& ps,
          api::Place const& from,
          api::Place const& to,
          api::plan_params const& query) {
  ps.from_ = geo::latlng{from.lat_, from.lon_};
  ps.to_ = geo::latlng{to.lat_, to.lon_};
  ps.fixed_ = query.arriveBy_ ? kArr : kDep;
  ps.cap_ = {
      .wheelchairs_ = static_cast<std::uint8_t>(
          query.pedestrianProfile_ == api::PedestrianProfileEnum::WHEELCHAIR
              ? 1U
              : 0U),
      .bikes_ =
          static_cast<std::uint8_t>(query.requireBikeTransport_ ? 1U : 0U),
      .passengers_ = 1U,
      .luggage_ = 0U};
}

void sort(std::vector<n::routing::start>& rides) {
  utl::sort(rides, [](auto&& a, auto&& b) {
    return std::tie(a.stop_, a.time_at_start_, a.time_at_stop_) <
           std::tie(b.stop_, b.time_at_start_, b.time_at_stop_);
  });
}

std::vector<n::routing::offset> get_offsets(
    ep::routing const& r,
    osr::location const& pos,
    osr::direction const dir,
    std::vector<api::ModeEnum> const& modes,
    bool const wheelchair,
    std::chrono::seconds const max,
    unsigned const max_matching_distance,
    gbfs::gbfs_routing_data& gbfs_rd,
    ep::stats_map_t& stats) {
  return r.get_offsets(pos, dir, modes, std::nullopt, std::nullopt,
                       std::nullopt, wheelchair, max, max_matching_distance,
                       gbfs_rd, stats);
}

void init_pt(std::vector<n::routing::start>& rides,
             ep::routing const& r,
             osr::location const& l,
             osr::direction dir,
             api::plan_params const& query,
             gbfs::gbfs_routing_data& gbfs_rd,
             motis::ep::stats_map_t& odm_stats,
             n::timetable const& tt,
             n::rt_timetable const* rtt,
             n::interval<n::unixtime_t> const& intvl,
             n::routing::query const& start_time,
             n::routing::location_match_mode location_match_mode) {
  static auto const kNoTdOffsets =
      hash_map<n::location_idx_t, std::vector<n::routing::td_offset>>{};

  auto const offsets = get_offsets(
      r, l, dir, {api::ModeEnum::CAR},
      query.pedestrianProfile_ == api::PedestrianProfileEnum::WHEELCHAIR,
      std::chrono::seconds{query.maxPreTransitTime_},
      query.maxMatchingDistance_, gbfs_rd, odm_stats);

  rides.clear();
  rides.reserve(offsets.size() * 2);

  n::routing::get_starts(
      dir == osr::direction::kForward ? n::direction::kForward
                                      : n::direction::kBackward,
      tt, rtt, intvl, offsets, kNoTdOffsets, n::routing::kMaxTravelTime,
      location_match_mode, false, rides, true, start_time.prf_idx_,
      start_time.transfer_time_settings_);

  sort(rides);
}

void init_direct(std::vector<direct_ride>& direct_rides,
                 ep::routing const& r,
                 elevators const* e,
                 gbfs::gbfs_routing_data& gbfs,
                 api::Place const& from_p,
                 api::Place const& to_p,
                 n::interval<n::unixtime_t> const intvl,
                 api::plan_params const& query) {
  auto [direct, duration] = r.route_direct(
      e, gbfs, from_p, to_p, {api::ModeEnum::CAR}, std::nullopt, std::nullopt,
      std::nullopt, intvl.from_,
      query.pedestrianProfile_ == api::PedestrianProfileEnum::WHEELCHAIR,
      std::chrono::seconds{query.maxDirectTime_});
  direct_rides.clear();
  if (query.arriveBy_) {
    for (auto arr =
             std::chrono::floor<std::chrono::hours>(intvl.to_ - duration) +
             duration;
         intvl.contains(arr); arr -= 1h) {
      direct_rides.emplace_back(arr - duration, arr);
    }
  } else {
    for (auto dep = std::chrono::ceil<std::chrono::hours>(intvl.from_);
         intvl.contains(dep); dep += 1h) {
      direct_rides.emplace_back(dep, dep + duration);
    }
  }
}

auto ride_time_halves(std::vector<n::routing::start>& rides) {
  utl::sort(rides, [](auto const& a, auto const& b) {
    auto const ride_time = [](auto const& ride) {
      return std::chrono::abs(ride.time_at_stop_ - ride.time_at_start_);
    };
    return ride_time(a) < ride_time(b);
  });
  auto const half = rides.size() / 2;
  auto lo = rides | std::views::take(half);
  auto hi = rides | std::views::drop(half);
  auto const stop_comp = [](auto const& a, auto const& b) {
    return a.stop_ < b.stop_;
  };
  std::ranges::sort(lo, stop_comp);
  std::ranges::sort(hi, stop_comp);
  return std::pair{lo, hi};
}

enum offset_event_type { kTimeAtStart, kTimeAtStop };
auto get_td_offsets(auto const& rides, offset_event_type const oet) {
  auto td_offsets = td_offsets_t{};
  utl::equal_ranges_linear(
      rides, [](auto const& a, auto const& b) { return a.stop_ == b.stop_; },
      [&](auto&& from_it, auto&& to_it) {
        td_offsets.emplace(from_it->stop_,
                           std::vector<n::routing::td_offset>{});
        for (auto const& r : n::it_range{from_it, to_it}) {
          td_offsets.at(from_it->stop_)
              .emplace_back(
                  oet == kTimeAtStart ? r.time_at_start_ : r.time_at_stop_,
                  std::chrono::abs(r.time_at_start_ - r.time_at_stop_), kODM);
          td_offsets.at(from_it->stop_)
              .emplace_back(oet == kTimeAtStart ? r.time_at_start_ + 1min
                                                : r.time_at_stop_ + 1min,
                            n::kInfeasible, kODM);
        }
      });
  return td_offsets;
}

auto collect_odm_journeys(auto& futures) {
  p_state->odm_journeys_.clear();
  for (auto& f : futures | std::views::drop(1)) {
    p_state->odm_journeys_.append_range(*f.get().journeys_);
  }
}

auto extract_rides() {
  p_state->from_rides_.clear();
  p_state->to_rides_.clear();
  for (auto const& j : p_state->odm_journeys_) {
    if (j.legs_.size() > 0) {
      if (std::holds_alternative<n::routing::offset>(j.legs_.front().uses_) &&
          std::get<n::routing::offset>(j.legs_.front().uses_)
                  .transport_mode_id_ == kODM) {
        p_state->from_rides_.emplace_back(j.legs_.front().dep_time_,
                                          j.legs_.front().arr_time_,
                                          j.legs_.front().to_);
      }
    }
    if (j.legs_.size() > 1) {
      if (std::holds_alternative<n::routing::offset>(j.legs_.back().uses_) &&
          std::get<n::routing::offset>(j.legs_.back().uses_)
                  .transport_mode_id_ == kODM) {
        p_state->to_rides_.emplace_back(j.legs_.back().arr_time_,
                                        j.legs_.back().dep_time_,
                                        j.legs_.back().from_);
      }
    }
  }

  auto const remove_dupes = [&](auto& rides) {
    sort(rides);
    rides.erase(std::unique(begin(rides), end(rides),
                            [](auto&& a, auto&& b) {
                              return std::tie(a.stop_, a.time_at_start_,
                                              a.time_at_stop_) ==
                                     std::tie(b.stop_, b.time_at_start_,
                                              b.time_at_stop_);
                            }),
                end(rides));
  };

  remove_dupes(p_state->from_rides_);
  remove_dupes(p_state->to_rides_);
}

void remove_not_whitelisted() {
  std::erase_if(p_state->odm_journeys_, [&](auto const& j) {
    return (j.legs_.size() > 0 &&
            std::holds_alternative<n::routing::offset>(j.legs_.front().uses_) &&
            std::get<n::routing::offset>(j.legs_.front().uses_)
                    .transport_mode_id_ == kODM &&
            std::find_if(begin(p_state->from_rides_), end(p_state->from_rides_),
                         [&](auto const& r) {
                           return std::tie(r.time_at_start_, r.time_at_stop_,
                                           r.stop_) ==
                                  std::tie(j.legs_.front().dep_time_,
                                           j.legs_.front().arr_time_,
                                           j.legs_.front().to_);
                         }) == end(p_state->from_rides_)) ||
           (j.legs_.size() > 1 &&
            std::holds_alternative<n::routing::offset>(j.legs_.back().uses_) &&
            std::get<n::routing::offset>(j.legs_.back().uses_)
                    .transport_mode_id_ == kODM &&
            std::find_if(begin(p_state->to_rides_), end(p_state->to_rides_),
                         [&](auto const& r) {
                           return std::tie(r.time_at_start_, r.time_at_stop_,
                                           r.stop_) ==
                                  std::tie(j.legs_.back().arr_time_,
                                           j.legs_.back().dep_time_,
                                           j.legs_.back().from_);
                         }) == end(p_state->to_rides_));
  });
}

void add_direct() {
  for (auto const& d : p_state->direct_rides_) {
    p_state->odm_journeys_.push_back(n::routing::journey{
        .legs_ = {n::routing::journey::leg{
            n::direction::kForward,
            get_special_station(n::special_station::kStart),
            get_special_station(n::special_station::kEnd), d.dep_, d.arr_,
            n::routing::offset{get_special_station(n::special_station::kEnd),
                               std::chrono::abs(d.arr_ - d.dep_), kODM}}},
        .start_time_ = d.dep_,
        .dest_time_ = d.arr_,
        .dest_ = get_special_station(n::special_station::kEnd),
        .transfers_ = 0U});
  }
}

void meta_router::extract_direct() {
  std::erase_if(p_state->odm_journeys_, [&](auto const& j) {
    if (j.legs_.size() == 1 &&
        std::holds_alternative<n::routing::offset>(j.legs_.front().uses_) &&
        std::get<n::routing::offset>(j.legs_.front().uses_)
                .transport_mode_id_ == kODM) {
      direct_.push_back(dummy_itinerary(from_p_, to_p_, api::ModeEnum::ODM,
                                        j.start_time_, j.dest_time_));
      return true;
    }
    return false;
  });
}

api::plan_response meta_router::run() {
  utl::verify(r_.tt_ != nullptr && r_.tags_ != nullptr,
              "mode=TRANSIT requires timetable to be loaded");

  auto stats = motis::ep::stats_map_t{};

  auto const start_intvl = std::visit(
      utl::overloaded{[](n::interval<n::unixtime_t> const i) { return i; },
                      [](n::unixtime_t const t) {
                        return n::interval<n::unixtime_t>{t, t};
                      }},
      start_time_.start_time_);
  auto const dest_intvl = get_dest_intvl(
      query_.arriveBy_ ? n::direction::kBackward : n::direction::kForward,
      start_intvl);
  auto const& from_intvl = query_.arriveBy_ ? dest_intvl : start_intvl;
  auto const& to_intvl = query_.arriveBy_ ? start_intvl : dest_intvl;

  if (p_state.get() == nullptr) {
    p_state.reset(new prima_state{});
  }

  init(*p_state, from_p_, to_p_, query_);

  if (odm_pre_transit_ && holds_alternative<osr::location>(from_)) {
    init_pt(p_state->from_rides_, r_, std::get<osr::location>(from_),
            osr::direction::kForward, query_, gbfs_rd_, stats, *tt_, rtt_,
            from_intvl, start_time_,
            query_.arriveBy_ ? start_time_.dest_match_mode_
                             : start_time_.start_match_mode_);
  }

  if (odm_post_transit_ && holds_alternative<osr::location>(to_)) {
    init_pt(p_state->to_rides_, r_, std::get<osr::location>(to_),
            osr::direction::kBackward, query_, gbfs_rd_, stats, *tt_, rtt_,
            to_intvl, start_time_,
            query_.arriveBy_ ? start_time_.start_match_mode_
                             : start_time_.dest_match_mode_);
  }

  if (odm_direct_ && r_.w_ && r_.l_) {
    init_direct(p_state->direct_rides_, r_, e_, gbfs_rd_, from_p_, to_p_,
                to_intvl, query_);
  }

  auto odm_networking = true;
  auto ioc = boost::asio::io_context{};
  try {
    // TODO the fiber/thread should yield until network response arrives?
    boost::asio::co_spawn(
        ioc,
        [&]() -> boost::asio::awaitable<void> {
          auto const blacklisting_response = co_await http_POST(
              kBlacklistingUrl, kPrimaHeaders, serialize(*p_state, *tt_), 10s);
          update(*p_state, get_http_body(blacklisting_response));
        },
        boost::asio::detached);
    ioc.run();
  } catch (std::exception const& e) {
    std::cout << "blacklisting failed: " << e.what();
    odm_networking = false;
  }

  auto const [from_rides_short, from_rides_long] =
      ride_time_halves(p_state->from_rides_);
  auto const [to_rides_short, to_rides_long] =
      ride_time_halves(p_state->to_rides_);

  auto const qf = query_factory{
      .start_time_ = start_time_.start_time_,
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
                              .value_or(kInfinityDuration),
      .min_connection_count_ = static_cast<unsigned>(query_.numItineraries_),
      .extend_interval_earlier_ = start_time_.extend_interval_earlier_,
      .extend_interval_later_ = start_time_.extend_interval_later_,
      .prf_idx_ = static_cast<n::profile_idx_t>(
          query_.useRoutedTransfers_
              ? (query_.pedestrianProfile_ ==
                         api::PedestrianProfileEnum::WHEELCHAIR
                     ? 2U
                     : 1U)
              : 0U),
      .allowed_claszes_ = to_clasz_mask(query_.transitModes_),
      .require_bike_transport_ = query_.requireBikeTransport_,
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
      .fastest_direct_ = fastest_direct_ == kInfinityDuration
                             ? std::nullopt
                             : std::optional{fastest_direct_},
      .start_walk_ = std::visit(
          utl::overloaded{[&](tt_location const l) {
                            return motis::ep::station_start(l.l_);
                          },
                          [&](osr::location const& pos) {
                            auto const dir = query_.arriveBy_
                                                 ? osr::direction::kBackward
                                                 : osr::direction::kForward;
                            return r_.get_offsets(
                                pos, dir, start_modes_, start_form_factors_,
                                start_propulsion_types_,
                                start_rental_providers_,
                                query_.pedestrianProfile_ ==
                                    api::PedestrianProfileEnum::WHEELCHAIR,
                                std::chrono::seconds{query_.maxPreTransitTime_},
                                query_.maxMatchingDistance_, gbfs_rd_, stats);
                          }},
          start_),
      .dest_walk_ = std::visit(
          utl::overloaded{
              [&](tt_location const l) {
                return motis::ep::station_start(l.l_);
              },
              [&](osr::location const& pos) {
                auto const dir = query_.arriveBy_ ? osr::direction::kForward
                                                  : osr::direction::kBackward;
                return r_.get_offsets(
                    pos, dir, dest_modes_, dest_form_factors_,
                    dest_propulsion_types_, dest_rental_providers_,
                    query_.pedestrianProfile_ ==
                        api::PedestrianProfileEnum::WHEELCHAIR,
                    std::chrono::seconds{query_.maxPostTransitTime_},
                    query_.maxMatchingDistance_, gbfs_rd_, stats);
              }},
          dest_),
      .td_start_walk_ =
          e_ != nullptr
              ? std::visit(
                    utl::overloaded{
                        [&](tt_location) { return td_offsets_t{}; },
                        [&](osr::location const& pos) {
                          auto const dir = query_.arriveBy_
                                               ? osr::direction::kBackward
                                               : osr::direction::kForward;
                          return r_.get_td_offsets(
                              *e_, pos, dir, start_modes_,
                              query_.pedestrianProfile_ ==
                                  api::PedestrianProfileEnum::WHEELCHAIR,
                              std::chrono::seconds{query_.maxPreTransitTime_});
                        }},
                    start_)
              : td_offsets_t{},
      .td_dest_walk_ =
          e_ != nullptr
              ? std::visit(
                    utl::overloaded{
                        [&](tt_location) { return td_offsets_t{}; },
                        [&](osr::location const& pos) {
                          auto const dir = query_.arriveBy_
                                               ? osr::direction::kForward
                                               : osr::direction::kBackward;
                          return r_.get_td_offsets(
                              *e_, pos, dir, dest_modes_,
                              query_.pedestrianProfile_ ==
                                  api::PedestrianProfileEnum::WHEELCHAIR,
                              std::chrono::seconds{query_.maxPostTransitTime_});
                        }},
                    dest_)
              : td_offsets_t{},
      .odm_start_short_ = query_.arriveBy_
                              ? get_td_offsets(to_rides_short, kTimeAtStart)
                              : get_td_offsets(from_rides_short, kTimeAtStart),
      .odm_start_long_ = query_.arriveBy_
                             ? get_td_offsets(to_rides_long, kTimeAtStart)
                             : get_td_offsets(from_rides_long, kTimeAtStart),
      .odm_dest_short_ = query_.arriveBy_
                             ? get_td_offsets(from_rides_short, kTimeAtStop)
                             : get_td_offsets(to_rides_short, kTimeAtStop),
      .odm_dest_long_ = query_.arriveBy_
                            ? get_td_offsets(from_rides_long, kTimeAtStop)
                            : get_td_offsets(to_rides_long, kTimeAtStop)};

  auto const make_task = [&](n::routing::query&& q) {
    return boost::fibers::packaged_task<
        n::routing::routing_result<n::routing::raptor_stats>()>{[&]() {
      return route(
          *tt_, rtt_, q,
          query_.arriveBy_ ? n::direction::kBackward : n::direction::kForward,
          query_.timeout_.has_value()
              ? std::optional<std::chrono::seconds>{*query_.timeout_}
              : std::nullopt);
    }};
  };

  auto tasks = std::vector<boost::fibers::packaged_task<
      n::routing::routing_result<n::routing::raptor_stats>()>>{};
  tasks.emplace_back(make_task(qf.walk_walk()));
  if (odm_networking) {
    tasks.emplace_back(make_task(qf.walk_short()));
    tasks.emplace_back(make_task(qf.walk_long()));
    tasks.emplace_back(make_task(qf.short_walk()));
    tasks.emplace_back(make_task(qf.long_walk()));
    tasks.emplace_back(make_task(qf.short_short()));
    tasks.emplace_back(make_task(qf.short_long()));
    tasks.emplace_back(make_task(qf.long_short()));
    tasks.emplace_back(make_task(qf.long_long()));
  }

  auto futures = utl::to_vec(tasks, [](auto& t) { return t.get_future(); });

  for (auto& t : tasks) {
    boost::fibers::fiber(std::move(t)).detach();
  }

  for (auto const& f : futures) {
    f.wait();
  }

  auto const pt_result = futures.front().get();
  collect_odm_journeys(futures);
  extract_rides();

  try {
    // TODO the fiber/thread should yield until network response arrives?
    boost::asio::co_spawn(
        ioc,
        [&]() -> boost::asio::awaitable<void> {
          auto const whitelisting_response = co_await http_POST(
              kWhitelistingUrl, kPrimaHeaders, serialize(*p_state, *tt_), 10s);
          update(*p_state, get_http_body(whitelisting_response));
        },
        boost::asio::detached);
    ioc.run();
  } catch (std::exception const& e) {
    std::cout << "whitelisting failed: " << e.what();
    odm_networking = false;
  }

  if (odm_networking) {
    remove_not_whitelisted();
    add_direct();
  } else {
    p_state->odm_journeys_.clear();
  }

  mix(*pt_result.journeys_, p_state->odm_journeys_);

  extract_direct();

  return {.from_ = from_p_,
          .to_ = to_p_,
          .direct_ = std::move(direct_),
          .itineraries_ = utl::to_vec(
              p_state->odm_journeys_,
              [&, cache = street_routing_cache_t{}](auto&& j) mutable {
                return journey_to_response(
                    r_.w_, r_.l_, r_.pl_, *tt_, *r_.tags_, e_, rtt_,
                    r_.matches_, r_.shapes_, gbfs_rd_,
                    query_.pedestrianProfile_ ==
                        api::PedestrianProfileEnum::WHEELCHAIR,
                    j, start_, dest_, cache, ep::blocked.get());
              }),
          .previousPageCursor_ =
              fmt::format("EARLIER|{}", to_seconds(pt_result.interval_.from_)),
          .nextPageCursor_ =
              fmt::format("LATER|{}", to_seconds(pt_result.interval_.to_))};
}

}  // namespace motis::odm