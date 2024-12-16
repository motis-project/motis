#include "motis/odm/odm.h"

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
#include "motis/odm/prima.h"
#include "motis/place.h"

namespace motis::odm {

namespace n = nigiri;
using namespace std::chrono_literals;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static boost::thread_specific_ptr<prima_state> ps;

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
    gbfs::gbfs_data const* gbfs,
    ep::stats_map_t& stats) {
  auto offsets = r.get_offsets(pos, dir, modes, wheelchair, max,
                               max_matching_distance, gbfs, stats);
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

std::vector<std::pair<n::unixtime_t, n::unixtime_t>> populate_direct(
    n::interval<n::unixtime_t> itvl, n::duration_t dur) {
  auto direct_events = std::vector<std::pair<n::unixtime_t, n::unixtime_t>>{};
  for (auto dep = std::chrono::ceil<std::chrono::hours>(itvl.from_);
       itvl.contains(dep); dep += 1h) {
    direct_events.emplace_back(dep, dep + dur);
  }
  return direct_events;
}

std::vector<std::pair<n::unixtime_t, n::unixtime_t>> get_direct_events(
    ep::routing const& r,
    elevators const* e,
    gbfs::gbfs_data const* gbfs,
    api::Place const& from,
    api::Place const& to,
    n::interval<n::unixtime_t> const start_intvl,
    bool wheelchair,
    std::chrono::seconds max) {
  auto [direct, duration] =
      r.route_direct(e, gbfs, from, to, {api::ModeEnum::CAR}, start_intvl.from_,
                     wheelchair, max);
  inflate(duration);
  return populate_direct(start_intvl, duration);
}

void prima_init(prima_state& ps,
                api::Place const& from,
                api::Place const& to,
                std::vector<n::routing::start> const& from_events,
                std::vector<n::routing::start> const& to_events, )

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
  auto const tt = r.tt_;
  auto const rt = r.rt_;
  auto const rtt = rt->rtt_.get();
  auto const e = r.rt_->e_.get();
  auto const gbfs = r.gbfs_;
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

  auto const odm_offsets_from =
      odm_pre_transit && holds_alternative<osr::location>(from)
          ? get_offsets(r, std::get<osr::location>(from),
                        osr::direction::kForward, {api::ModeEnum::CAR},
                        query.wheelchair_,
                        std::chrono::seconds{query.maxPreTransitTime_},
                        query.maxMatchingDistance_, gbfs.get(), odm_stats)
          : std::vector<n::routing::offset>{};

  auto const odm_offsets_to =
      odm_post_transit && holds_alternative<osr::location>(to)
          ? get_offsets(r, std::get<osr::location>(to),
                        osr::direction::kBackward, {api::ModeEnum::CAR},
                        query.wheelchair_,
                        std::chrono::seconds{query.maxPostTransitTime_},
                        query.maxMatchingDistance_, gbfs.get(), odm_stats)
          : std::vector<n::routing::offset>{};

  auto const from_events =
      odm_pre_transit
          ? get_events(n::direction::kForward, *tt, rtt, from_intvl,
                       odm_offsets_from, n::routing::kMaxTravelTime,
                       query.arriveBy_ ? start_time.dest_match_mode_
                                       : start_time.start_match_mode_,
                       start_time.prf_idx_, start_time.transfer_time_settings_)
          : std::vector<n::routing::start>{};

  auto const to_events =
      odm_post_transit
          ? get_events(n::direction::kBackward, *tt, rtt, to_intvl,
                       odm_offsets_to, n::routing::kMaxTravelTime,
                       query.arriveBy_ ? start_time.start_match_mode_
                                       : start_time.dest_match_mode_,
                       start_time.prf_idx_, start_time.transfer_time_settings_)
          : std::vector<n::routing::start>{};

  auto const direct_events =
      (odm_direct && r.w_ && r.l_)
          ? get_direct_events(r, e, gbfs.get(), from_p, to_p, start_intvl,
                              query.wheelchair_,
                              std::chrono::seconds{query.maxDirectTime_})
          : std::vector<std::pair<n::unixtime_t, n::unixtime_t>>{};

  if (ps.get() == nullptr) {
    ps.reset(new prima_state{});
  }

  // TODO blacklist request

  // TODO remove blacklisted offsets

  // TODO start fibers to do the ODM routing

  // TODO whitelist request for ODM rides used in journeys

  // TODO remove journeys with non-whitelisted ODM rides

  return std::vector<nigiri::routing::journey>{};
}

}  // namespace motis::odm