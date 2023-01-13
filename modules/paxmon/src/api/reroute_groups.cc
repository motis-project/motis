#include "motis/paxmon/api/reroute_groups.h"

#include <algorithm>

#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/core/common/date_time_util.h"

#include "motis/paxmon/access/groups.h"
#include "motis/paxmon/debug.h"
#include "motis/paxmon/get_universe.h"
#include "motis/paxmon/messages.h"

using namespace motis::module;
using namespace motis::paxmon;

namespace motis::paxmon::api {

namespace {

template <typename CompactJourney>
inline void before_journey_load_updated(universe& uv,
                                        CompactJourney const& cj) {
  for (auto const& leg : cj.legs()) {
    uv.update_tracker_.before_trip_load_updated(leg.trip_idx_);
  }
}

struct log_entry_info {
  reroute_log_entry& entry_;
  typename dynamic_fws_multimap<reroute_log_route_info>::mutable_bucket
      new_routes_;
  bool extended_entry_{};
};

inline log_entry_info append_or_extend_log_entry(
    universe& uv, schedule const& sched, passenger_group_index const pgi,
    local_group_route_index const old_route_idx,
    float const old_route_probability, reroute_reason_t const reason,
    std::optional<broken_transfer_info> const& bti, bool const has_new_routes) {
  auto& pgc = uv.passenger_groups_;
  auto const system_time = sched.system_time_;
  auto log_entries = pgc.reroute_log_entries(pgi);
  if (has_new_routes && reason == reroute_reason_t::MAJOR_DELAY_EXPECTED) {
    if (auto it = std::find_if(
            log_entries.rbegin(), log_entries.rend(),
            [&](reroute_log_entry const& entry) {
              return entry.reason_ == reroute_reason_t::MAJOR_DELAY_EXPECTED &&
                     entry.old_route_.route_ == old_route_idx &&
                     entry.system_time_ == system_time &&
                     pgc.log_entry_new_routes_.at(entry.index_).empty();
            });
        it != log_entries.rend()) {
      return {*it, pgc.log_entry_new_routes_.at(it->index_), true};
    }
  }
  auto log_new_routes = pgc.log_entry_new_routes_.emplace_back();
  log_entries.emplace_back(reroute_log_entry{
      static_cast<reroute_log_entry_index>(log_new_routes.index()),
      reroute_log_route_info{old_route_idx, old_route_probability, 0},
      system_time, now(), reason, bti});
  return {log_entries.back(), log_new_routes, false};
}

}  // namespace

msg_ptr reroute_groups(paxmon_data& data, msg_ptr const& msg) {
  auto const req = motis_content(PaxMonRerouteGroupsRequest, msg);
  auto const uv_access =
      get_universe_and_schedule(data, req->universe(), ctx::access_t::WRITE);
  auto const& sched = uv_access.sched_;
  auto& uv = uv_access.uv_;
  auto const tracking_updates = uv.update_tracker_.is_tracking();

  message_creator mc;

  auto const reroutes = utl::to_vec(*req->reroutes(), [&](auto const& rr) {
    auto const pgi = static_cast<passenger_group_index>(rr->group());
    auto const old_route_idx =
        static_cast<local_group_route_index>(rr->old_route_index());
    auto const reason = static_cast<reroute_reason_t>(rr->reason());
    auto const bti = from_fbs(sched, rr->broken_transfer());
    auto const override_probabilities = rr->override_probabilities();
    auto routes = uv.passenger_groups_.routes(pgi);

    auto routes_backup = utl::to_vec(
        routes, [](group_route const& gr) -> group_route { return gr; });

    auto& old_route = routes.at(old_route_idx);
    auto const old_route_probability = old_route.probability_;
    auto lei = append_or_extend_log_entry(uv, sched, pgi, old_route_idx,
                                          old_route_probability, reason, bti,
                                          rr->new_routes()->size() != 0);

    if (reason != reroute_reason_t::DESTINATION_UNREACHABLE &&
        reason != reroute_reason_t::DESTINATION_REACHABLE) {
      if (tracking_updates) {
        before_journey_load_updated(
            uv, uv.passenger_groups_.journey(old_route.compact_journey_index_));
      }
      old_route.probability_ = 0;
      uv.update_tracker_.after_group_route_updated(
          passenger_group_with_route{pgi, old_route_idx}, old_route_probability,
          0, false);
    } else {
      utl::verify(
          rr->new_routes()->size() == 0,
          "reroute_groups: destination (un)reachable, but new groups provided");
    }

    auto new_routes = utl::to_vec(*rr->new_routes(), [&](auto const& nr) {
      auto const tgr = from_fbs(sched, nr);
      if (tracking_updates) {
        before_journey_load_updated(uv, tgr.journey_);
      }
      auto const result =
          add_group_route(uv, sched, data.capacity_maps_, pgi, tgr,
                          override_probabilities, true, pci_log_reason_t::API);
      auto const previous_probability =
          lei.extended_entry_ && result.pgwr_.route_ == old_route_idx &&
                  result.previous_probability_ == 0.0F
              ? lei.entry_.old_route_.previous_probability_
              : (result.pgwr_.route_ == old_route_idx
                     ? old_route_probability
                     : result.previous_probability_);
      lei.new_routes_.emplace_back(reroute_log_route_info{
          result.pgwr_.route_, previous_probability, result.new_probability_});
      uv.update_tracker_.after_group_route_updated(
          result.pgwr_, result.previous_probability_, result.new_probability_,
          result.new_route_);
      if (result.pgwr_.route_ == old_route_idx) {
        lei.entry_.old_route_.new_probability_ = result.new_probability_;
      }
      return PaxMonRerouteRouteInfo{result.pgwr_.route_,
                                    result.previous_probability_,
                                    result.new_probability_};
    });

    return CreatePaxMonRerouteGroupResult(mc, pgi, old_route_idx,
                                          mc.CreateVectorOfStructs(new_routes));
  });

  mc.create_and_finish(
      MsgContent_PaxMonRerouteGroupsResponse,
      CreatePaxMonRerouteGroupsResponse(mc, mc.CreateVector(reroutes)).Union());
  return make_msg(mc);
}

}  // namespace motis::paxmon::api
