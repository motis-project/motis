#include "motis/odm/meta_routing.h"

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
#include "motis/odm/json.h"
#include "motis/odm/prima_state.h"
#include "motis/odm/routing_fiber.h"
#include "motis/place.h"

namespace motis::odm {

namespace n = nigiri;
using namespace std::chrono_literals;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static boost::thread_specific_ptr<prima_state> p_state;

static auto const kPrimaUrl = boost::urls::url{""};
static auto const kPrimaHeaders = std::map<std::string, std::string>{
    {"Content-Type", "application/json"}, {"Accept", "application/json"}};

static auto const kTransportModeIdODM = n::transport_mode_id_t{23};

using td_offsets_t =
    n::hash_map<n::location_idx_t, std::vector<n::routing::td_offset>>;

n::interval<n::unixtime_t> get_dest_intvl(
    n::direction dir, n::interval<n::unixtime_t> const& start_intvl) {
  return dir == n::direction::kForward
             ? n::interval<n::unixtime_t>{start_intvl.from_,
                                          start_intvl.to_ + 24h}
             : n::interval<n::unixtime_t>{start_intvl.from_ - 24h,
                                          start_intvl.to_};
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
  utl::sort(rides, [](auto&& a, auto&& b) {
    return std::tie(a.stop_, a.time_at_start_, a.time_at_stop_) <
           std::tie(b.stop_, b.time_at_start_, b.time_at_stop_);
  });
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
                  std::chrono::abs(r.time_at_start_ - r.time_at_stop_),
                  kTransportModeIdODM);
          td_offsets.at(from_it->stop_)
              .emplace_back(oet == kTimeAtStart ? r.time_at_start_ + 1min
                                                : r.time_at_stop_ + 1min,
                            n::kInfeasible, kTransportModeIdODM);
        }
      });
  return td_offsets;
}

bool is_intermodal(place_t const& p) {
  return std::holds_alternative<osr::location>(p);
}

n::routing::location_match_mode get_match_mode(place_t const& p) {
  return is_intermodal(p) ? n::routing::location_match_mode::kIntermodal
                          : n::routing::location_match_mode::kEquivalent;
}

std::vector<n::routing::offset> station_start(n::location_idx_t const l) {
  return {{l, n::duration_t{0U}, 0U}};
}

std::vector<n::routing::query> meta_router::get_queries() {
  auto queries = std::vector<n::routing::query>{};
  queries.push_back(
      n::routing::query{.start_time_ = start_time_.start_time_,
                        .start_match_mode_ = get_match_mode(start_),
                        .dest_match_mode_ = get_match_mode(dest_),
                        .use_start_footpaths_ = !is_intermodal(start_)});

  return queries;
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
      odm_pre_transit_{odm_pre_transit},
      odm_post_transit_{odm_post_transit},
      odm_direct_{odm_direct},
      tt_{r_.tt_},
      rt_{r_.rt_},
      rtt_{rt_->rtt_.get()},
      e_{rt_->e_.get()},
      gbfs_rd_{r_.w_, r_.l_, r_.gbfs_},
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
  if (ep::blocked.get() == nullptr && r_.w_ != nullptr) {
    ep::blocked.reset(new osr::bitvec<osr::node_idx_t>{r_.w_->n_nodes()});
  }
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
  try {
    // TODO the fiber/thread should yield until network response arrives?
    auto ioc = boost::asio::io_context{};
    boost::asio::co_spawn(
        ioc,
        [&]() -> boost::asio::awaitable<void> {
          auto const bl_response = co_await http_POST(
              kPrimaUrl, kPrimaHeaders, serialize(*p_state, *tt_), 10s);
          update(*p_state, get_http_body(bl_response));
        },
        boost::asio::detached);
    ioc.run();
  } catch (std::exception const& e) {
    std::cout << "blacklisting failed: " << e.what();
    odm_networking = false;
  }

  auto const start_walk = std::visit(
      utl::overloaded{[&](tt_location const l) { return station_start(l.l_); },
                      [&](osr::location const& pos) {
                        auto const dir = query_.arriveBy_
                                             ? osr::direction::kBackward
                                             : osr::direction::kForward;
                        return r_.get_offsets(
                            pos, dir, start_modes_, start_form_factors_,
                            start_propulsion_types_, start_rental_providers_,
                            query_.pedestrianProfile_ ==
                                api::PedestrianProfileEnum::WHEELCHAIR,
                            std::chrono::seconds{query_.maxPreTransitTime_},
                            query_.maxMatchingDistance_, gbfs_rd_, stats);
                      }},
      start_);

  auto const dest_walk = std::visit(
      utl::overloaded{[&](tt_location const l) { return station_start(l.l_); },
                      [&](osr::location const& pos) {
                        auto const dir = query_.arriveBy_
                                             ? osr::direction::kForward
                                             : osr::direction::kBackward;
                        return r_.get_offsets(
                            pos, dir, dest_modes_, dest_form_factors_,
                            dest_propulsion_types_, dest_rental_providers_,
                            query_.pedestrianProfile_ ==
                                api::PedestrianProfileEnum::WHEELCHAIR,
                            std::chrono::seconds{query_.maxPostTransitTime_},
                            query_.maxMatchingDistance_, gbfs_rd_, stats);
                      }},
      dest_);

  auto const td_start_walk =
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
          : td_offsets_t{};

  auto const td_dest_walk =
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
          : td_offsets_t{};

  auto const [from_rides_short, from_rides_long] =
      ride_time_halves(p_state->from_rides_);
  auto const [to_rides_short, to_rides_long] =
      ride_time_halves(p_state->to_rides_);

  auto const td_offsets_from_short = get_td_offsets(
      from_rides_short, query_.arriveBy_ ? offset_event_type::kTimeAtStop
                                         : offset_event_type::kTimeAtStart);
  auto const td_offsets_from_long = get_td_offsets(
      from_rides_long, query_.arriveBy_ ? offset_event_type::kTimeAtStop
                                        : offset_event_type::kTimeAtStart);
  auto const td_offsets_to_short = get_td_offsets(
      to_rides_short, query_.arriveBy_ ? offset_event_type::kTimeAtStart
                                       : offset_event_type::kTimeAtStop);
  auto const td_offsets_to_long = get_td_offsets(
      to_rides_long, query_.arriveBy_ ? offset_event_type::kTimeAtStart
                                      : offset_event_type::kTimeAtStop);

  auto const queries = get_queries();

  auto tasks = std::vector<boost::fibers::packaged_task<
      n::routing::routing_result<n::routing::raptor_stats>()>>{};

  // TODO start fibers to do the ODM routing

  // TODO whitelist request for ODM rides used in journeys

  // TODO remove journeys with non-whitelisted ODM rides

  return {.from_ = to_place(tt_, r_.tags_, r_.w_, r_.pl_, r_.matches_, from_),
          .to_ = to_place(tt_, r_.tags_, r_.w_, r_.pl_, r_.matches_, to_),
          .direct_ = std::move(direct),
          .itineraries_ = {}};
}

}  // namespace motis::odm