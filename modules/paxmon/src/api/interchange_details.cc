#include "motis/paxmon/api/interchange_details.h"

#include <limits>
#include <vector>

#include "boost/range/join.hpp"

#include "utl/verify.h"

#include "motis/core/access/time_access.h"

#include "motis/paxmon/get_universe.h"
#include "motis/paxmon/messages.h"
#include "motis/paxmon/util/interchange_info.h"

using namespace motis::module;
using namespace motis::paxmon;
using namespace motis::paxmon::util;
using namespace flatbuffers;

namespace motis::paxmon::api {

msg_ptr interchange_details(paxmon_data& data, msg_ptr const& msg) {
  auto const req = motis_content(PaxMonInterchangeDetailsRequest, msg);
  auto const uv_access = get_universe_and_schedule(data, req->universe());
  auto const& sched = uv_access.sched_;
  auto const& uv = uv_access.uv_;

  auto const ei =
      edge_index{static_cast<event_node_index>(req->id()->n()), req->id()->e()};

  auto const include_disabled_group_routes =
      req->include_disabled_group_routes();
  auto const include_full_groups = req->include_full_groups();
  auto const include_reroute_log = req->include_reroute_log();

  auto const* e = ei.get(uv);
  utl::verify(e->is_interchange(), "interchange_details: invalid id");

  message_creator mc;

  auto const normal_routes = uv.pax_connection_info_.group_routes(e->pci_);
  auto const broken_routes =
      uv.pax_connection_info_.broken_group_routes(e->pci_);

  auto const info = get_interchange_info(
      uv, sched, ei, mc,
      get_interchange_info_options{
          .include_group_infos_ = true,
          .include_disabled_group_routes_ = include_disabled_group_routes,
          .include_delay_info_ = true});

  auto const get_fbs_groups = [&](auto const& groups) {
    auto result = std::vector<Offset<PaxMonGroup>>{};
    auto last_pg = std::numeric_limits<passenger_group_index>::max();
    for (auto const& pgwr : groups) {
      if (last_pg == pgwr.pg_) {
        continue;
      }
      last_pg = pgwr.pg_;
      auto const& pg = uv.passenger_groups_.at(pgwr.pg_);
      result.emplace_back(
          to_fbs(sched, uv.passenger_groups_, mc, *pg, include_reroute_log));
    }
    return result;
  };

  auto fbs_groups = std::vector<Offset<PaxMonGroup>>{};
  if (include_full_groups) {
    if (include_disabled_group_routes) {
      fbs_groups = get_fbs_groups(boost::join(normal_routes, broken_routes));
    } else {
      fbs_groups = get_fbs_groups(normal_routes);
    }
  }

  mc.create_and_finish(
      MsgContent_PaxMonInterchangeDetailsResponse,
      CreatePaxMonInterchangeDetailsResponse(
          mc, info.to_fbs_interchange_info(mc, uv, sched, true),
          info.normal_routes_, info.broken_routes_, mc.CreateVector(fbs_groups))
          .Union());
  return make_msg(mc);
}

}  // namespace motis::paxmon::api
