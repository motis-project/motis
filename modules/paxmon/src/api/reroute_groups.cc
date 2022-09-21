#include "motis/paxmon/api/reroute_groups.h"

#include "utl/to_vec.h"

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
void before_journey_load_updated(universe& uv, CompactJourney const& cj) {
  for (auto const& leg : cj.legs()) {
    uv.update_tracker_.before_trip_load_updated(leg.trip_idx_);
  }
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
    auto routes = uv.passenger_groups_.routes(pgi);

    auto routes_backup = utl::to_vec(
        routes, [](group_route const& gr) -> group_route { return gr; });

    auto& old_route = routes.at(old_route_idx);
    if (tracking_updates) {
      before_journey_load_updated(
          uv, uv.passenger_groups_.journey(old_route.compact_journey_index_));
    }
    auto const old_route_probability = old_route.probability_;
    old_route.probability_ = 0;
    uv.update_tracker_.after_group_route_updated(
        passenger_group_with_route{pgi, old_route_idx}, old_route_probability,
        0, false);

    auto log_new_routes =
        uv.passenger_groups_.log_entry_new_routes_.emplace_back();
    uv.passenger_groups_.reroute_log_entries(pgi).emplace_back(
        reroute_log_entry{
            static_cast<reroute_log_entry_index>(log_new_routes.index()),
            old_route_idx, old_route_probability, sched.system_time_, now(),
            reason});

    auto new_routes = utl::to_vec(*rr->new_routes(), [&](auto const& nr) {
      auto const tgr = from_fbs(sched, nr);
      if (tracking_updates) {
        before_journey_load_updated(uv, tgr.journey_);
      }
      auto const result =
          add_group_route(uv, sched, data.capacity_maps_, pgi, tgr);
      log_new_routes.emplace_back(reroute_log_new_route{
          result.pgwr_.route_, result.previous_probability_,
          result.new_probability_});
      uv.update_tracker_.after_group_route_updated(
          result.pgwr_, result.previous_probability_, result.new_probability_,
          result.new_route_);
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
