#include "motis/odm/odm.h"

#include "boost/asio/co_spawn.hpp"
#include "boost/thread/tss.hpp"

#include "nigiri/routing/journey.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor/raptor_state.h"
#include "nigiri/routing/raptor_search.h"
#include "nigiri/routing/start_times.h"
#include "nigiri/types.h"

#include "osr/routing/route.h"

#include "motis-api/motis-api.h"
#include "motis/endpoints/routing.h"
#include "motis/gbfs/routing_data.h"
#include "motis/http_req.h"
#include "motis/odm/json.h"
#include "motis/odm/prima.h"
#include "motis/place.h"

namespace motis::odm {

namespace n = nigiri;
using namespace std::chrono_literals;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static boost::thread_specific_ptr<prima_state> odm_state;

static auto const kPrimaUrl = boost::urls::url{""};
static auto const kPrimaHeaders = std::map<std::string, std::string>{
    {"Content-Type", "application/json"}, {"Accept", "application/json"}};

n::interval<n::unixtime_t> get_dest_intvl(
    n::direction dir, n::interval<n::unixtime_t> const& start_intvl) {
  return dir == n::direction::kForward
             ? n::interval<n::unixtime_t>{start_intvl.from_ + 30min,
                                          start_intvl.to_ + 24h}
             : n::interval<n::unixtime_t>{start_intvl.from_ - 24h,
                                          start_intvl.to_ - 30min};
}

void inflate(n::duration_t& d) {
  static constexpr auto const kInflationFactor = 1.5;
  static constexpr auto const kMinInflation = n::duration_t{10};

  d = std::max(std::chrono::duration_cast<n::duration_t>(d * kInflationFactor),
               d + kMinInflation);
}

std::vector<n::routing::offset> get_offsets(
    ep::routing const& r,
    osr::location const& pos,
    osr::direction const dir,
    std::vector<api::ModeEnum> const& modes,
    bool const wheelchair,
    std::chrono::seconds const max,
    unsigned const max_matching_distance,
    gbfs::gbfs_routing_data& gbfs,
    ep::stats_map_t& stats) {
  auto offsets =
      r.get_offsets(pos, dir, modes, std::nullopt, std::nullopt, std::nullopt,
                    wheelchair, max, max_matching_distance, gbfs, stats);
  for (auto& o : offsets) {
    inflate(o.duration_);
  }
  return offsets;
}

std::vector<n::routing::start> get_events(
    n::direction const search_dir,
    n::timetable const& tt,
    n::rt_timetable const* rtt,
    n::routing::start_time_t const& time,
    std::vector<n::routing::offset> const& offsets,
    n::duration_t const max_start_offset,
    n::routing::location_match_mode const mode,
    n::profile_idx_t const prf_idx,
    n::routing::transfer_time_settings const& tts) {
  static auto const kNoTdOffsets =
      hash_map<n::location_idx_t, std::vector<n::routing::td_offset>>{};
  auto events = std::vector<n::routing::start>{};
  events.reserve(offsets.size() * 2);
  n::routing::get_starts(search_dir, tt, rtt, time, offsets, kNoTdOffsets,
                         max_start_offset, mode, false, events, true, prf_idx,
                         tts);
  utl::sort(events, [](auto&& a, auto&& b) {
    return std::tie(a.stop_, a.time_at_start_, a.time_at_stop_) <
           std::tie(b.stop_, b.time_at_start_, b.time_at_stop_);
  });
  return events;
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
             gbfs::gbfs_routing_data& gbfs,
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
      query.maxMatchingDistance_, gbfs, odm_stats);

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
  inflate(duration);
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

std::optional<std::vector<n::routing::journey>> odm_routing(
    ep::routing const& r,
    api::plan_params const& query,
    std::vector<api::ModeEnum> const& pre_transit_modes,
    std::vector<api::ModeEnum> const& post_transit_modes,
    std::vector<api::ModeEnum> const& direct_modes,
    std::variant<osr::location, tt_location> const& from,
    std::variant<osr::location, tt_location> const& to,
    api::Place const& from_p,
    api::Place const& to_p,
    n::routing::query const& start_time) {
  utl::verify(r.tt_ != nullptr && r.tags_ != nullptr,
              "mode=TRANSIT requires timetable to be loaded");

  auto const tt = r.tt_;
  auto const rt = r.rt_;
  auto const rtt = rt->rtt_.get();
  auto const e = r.rt_->e_.get();
  auto gbfs_rd = gbfs::gbfs_routing_data{r.w_, r.l_, r.gbfs_};
  if (ep::blocked.get() == nullptr && r.w_ != nullptr) {
    ep::blocked.reset(new osr::bitvec<osr::node_idx_t>{r.w_->n_nodes()});
  }

  auto const odm_pre_transit =
      std::find(begin(pre_transit_modes), end(pre_transit_modes),
                api::ModeEnum::ODM) != end(pre_transit_modes);
  auto const odm_post_transit =
      std::find(begin(post_transit_modes), end(post_transit_modes),
                api::ModeEnum::ODM) != end(post_transit_modes);
  auto const odm_direct = std::find(begin(direct_modes), end(direct_modes),
                                    api::ModeEnum::ODM) != end(direct_modes);
  auto const odm_any = odm_pre_transit || odm_post_transit || odm_direct;
  if (!odm_any) {
    return std::nullopt;
  }

  auto odm_stats = motis::ep::stats_map_t{};

  auto const start_intvl = std::visit(
      utl::overloaded{[](n::interval<n::unixtime_t> const i) { return i; },
                      [](n::unixtime_t const t) {
                        return n::interval<n::unixtime_t>{t, t};
                      }},
      start_time.start_time_);
  auto const dest_intvl = get_dest_intvl(
      query.arriveBy_ ? n::direction::kBackward : n::direction::kForward,
      start_intvl);
  auto const& from_intvl = query.arriveBy_ ? dest_intvl : start_intvl;
  auto const& to_intvl = query.arriveBy_ ? start_intvl : dest_intvl;

  if (odm_state.get() == nullptr) {
    odm_state.reset(new prima_state{});
  }

  init(*odm_state, from_p, to_p, query);

  if (odm_pre_transit && holds_alternative<osr::location>(from)) {
    init_pt(odm_state->from_rides_, r, std::get<osr::location>(from),
            osr::direction::kForward, query, gbfs_rd, odm_stats, *tt, rtt,
            from_intvl, start_time,
            query.arriveBy_ ? start_time.dest_match_mode_
                            : start_time.start_match_mode_);
  }

  if (odm_post_transit && holds_alternative<osr::location>(to)) {
    init_pt(odm_state->to_rides_, r, std::get<osr::location>(to),
            osr::direction::kBackward, query, gbfs_rd, odm_stats, *tt, rtt,
            to_intvl, start_time,
            query.arriveBy_ ? start_time.start_match_mode_
                            : start_time.dest_match_mode_);
  }

  if (odm_direct && r.w_ && r.l_) {
    init_direct(odm_state->direct_rides_, r, e, gbfs_rd, from_p, to_p, to_intvl,
                query);
  }

  try {
    // TODO the fiber/thread should yield until network response arrives?
    boost::asio::co_spawn(
        ioc,
        [&]() -> boost::asio::awaitable<void> {
          auto const bl_response = co_await http_POST(
              kPrimaUrl, kPrimaHeaders, serialize(*odm_state, *tt), 10s);
          update(*odm_state, get_http_body(bl_response));
        },
        boost::asio::deferred);
  } catch (std::exception const& e) {
    std::cout << "prima blacklisting failed: " << e.what();
    return std::nullopt;
  }

  // TODO start fibers to do the ODM routing

  // TODO whitelist request for ODM rides used in journeys

  // TODO remove journeys with non-whitelisted ODM rides

  return std::vector<nigiri::routing::journey>{};
}

}  // namespace motis::odm