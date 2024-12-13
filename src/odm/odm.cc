#include "motis/odm/odm.h"

#include "boost/fiber/fss.hpp"

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
#include "motis/place.h"

namespace motis::ep {

namespace n = nigiri;
using namespace std::chrono_literals;

// TODO or should each thread rather use its thread-specifics while executing
// the fiber?

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static boost::fibers::fiber_specific_ptr<n::routing::search_state> search_state;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static boost::fibers::fiber_specific_ptr<n::routing::raptor_state> raptor_state;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static boost::fibers::fiber_specific_ptr<osr::bitvec<osr::node_idx_t>> blocked;

static auto const kNoTdOffsets =
    hash_map<n::location_idx_t, std::vector<n::routing::td_offset>>{};

n::interval<n::unixtime_t> get_dest_intvl(
    n::direction dir, n::interval<n::unixtime_t> const& start_intvl) {
  return dir == n::direction::kForward
             ? n::interval<n::unixtime_t>{start_intvl.from_ + 30min,
                                          start_intvl.to_ + 24h}
             : n::interval<n::unixtime_t>{start_intvl.from_ - 24h,
                                          start_intvl.to_ - 30min};
}

void populate_direct(
    n::interval<n::unixtime_t> itvl,
    n::duration_t dur,
    std::vector<std::pair<n::unixtime_t, n::unixtime_t>>& direct_events) {
  for (auto dep = std::chrono::ceil<std::chrono::hours>(itvl.from_);
       itvl.contains(dep); dep += 1h) {
    direct_events.emplace_back(dep, dep + dur);
  }
}

bool comp_starts(n::routing::start const& a, n::routing::start const& b) {
  return std::tie(a.stop_, a.time_at_start_, a.time_at_stop_) <
         std::tie(b.stop_, b.time_at_start_, b.time_at_stop_);
}

std::optional<std::vector<n::routing::journey>> odm_routing(
    routing const& r,
    api::plan_params const& query,
    std::vector<api::ModeEnum> const& pre_transit_modes,
    std::vector<api::ModeEnum> const& post_transit_modes,
    std::vector<api::ModeEnum> const& direct_modes,
    std::variant<osr::location, tt_location> const& from,
    std::variant<osr::location, tt_location> const& to,
    api::Place const& from_p,
    api::Place const& to_p,
    std::variant<osr::location, tt_location> const& start,
    std::variant<osr::location, tt_location> const& dest,
    std::vector<api::ModeEnum> const& start_modes,
    std::vector<api::ModeEnum> const& dest_modes,
    n::routing::query const& start_time,
    std::optional<n::unixtime_t> const& t) {
  auto const tt = r.tt_;
  auto const rt = r.rt_;
  auto const rtt = rt->rtt_.get();
  auto const e = r.rt_->e_.get();
  auto const gbfs = r.gbfs_;
  if (blocked.get() == nullptr && r.w_ != nullptr) {
    blocked.reset(new osr::bitvec<osr::node_idx_t>{r.w_->n_nodes()});
  }

  auto const odm_pre_transit =
      std::find(begin(pre_transit_modes), end(pre_transit_modes),
                api::ModeEnum::ODM) != end(pre_transit_modes);
  auto const odm_post_transit =
      std::find(begin(post_transit_modes), end(post_transit_modes),
                api::ModeEnum::ODM) != end(post_transit_modes);
  auto const odm_start = query.arriveBy_ ? odm_post_transit : odm_pre_transit;
  auto const odm_dest = query.arriveBy_ ? odm_pre_transit : odm_post_transit;
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

  auto const odm_offsets_from =
      odm_pre_transit && holds_alternative<osr::location>(from)
          ? r.get_offsets(std::get<osr::location>(from),
                          osr::direction::kForward, {api::ModeEnum::CAR},
                          query.wheelchair_,
                          std::chrono::seconds{query.maxPreTransitTime_},
                          query.maxMatchingDistance_, gbfs.get(), odm_stats)
          : std::vector<n::routing::offset>{};

  auto const odm_offsets_to =
      odm_post_transit && holds_alternative<osr::location>(to)
          ? r.get_offsets(std::get<osr::location>(to),
                          osr::direction::kBackward, {api::ModeEnum::CAR},
                          query.wheelchair_,
                          std::chrono::seconds{query.maxPostTransitTime_},
                          query.maxMatchingDistance_, gbfs.get(), odm_stats)
          : std::vector<n::routing::offset>{};

  auto from_events = std::vector<n::routing::start>{};
  if (odm_pre_transit) {
    from_events.reserve(odm_offsets_from.size() * 2);
    n::routing::get_starts(n::direction::kForward, *tt, rtt, from_intvl,
                           odm_offsets_from, kNoTdOffsets,
                           n::routing::kMaxTravelTime,
                           query.arriveBy_ ? start_time.dest_match_mode_
                                           : start_time.start_match_mode_,
                           false, from_events, true, start_time.prf_idx_,
                           start_time.transfer_time_settings_);
    utl::sort(from_events, comp_starts);
  }

  auto to_events = std::vector<n::routing::start>{};
  if (odm_post_transit) {
    to_events.reserve(odm_offsets_to.size() * 2);
    n::routing::get_starts(n::direction::kBackward, *tt, rtt, to_intvl,
                           odm_offsets_to, kNoTdOffsets,
                           n::routing::kMaxTravelTime,
                           query.arriveBy_ ? start_time.start_match_mode_
                                           : start_time.dest_match_mode_,
                           false, to_events, true, start_time.prf_idx_,
                           start_time.transfer_time_settings_);
    utl::sort(to_events, comp_starts);
  }

  auto direct_events = std::vector<std::pair<n::unixtime_t, n::unixtime_t>>{};
  if (odm_direct && r.w_ && r.l_) {
    auto const [direct, duration] = r.route_direct(
        e, gbfs.get(), from_p, to_p, {api::ModeEnum::CAR}, start_intvl.from_,
        query.wheelchair_, std::chrono::seconds{query.maxDirectTime_});
    populate_direct(start_intvl, duration, direct_events);
  }

  // TODO blacklist request

  // TODO remove blacklisted offsets

  // TODO start fibers to do the ODM routing

  // TODO whitelist request for ODM rides used in journeys

  // TODO remove journeys with non-whitelisted ODM rides

  return std::vector<nigiri::routing::journey>{};
}

}  // namespace motis::ep