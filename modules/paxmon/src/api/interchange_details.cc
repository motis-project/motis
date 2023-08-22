#include "motis/paxmon/api/interchange_details.h"

#include <iostream>

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

  auto const* e = ei.get(uv);
  utl::verify(e->is_interchange(), "interchange_details: invalid id");

  message_creator mc;

  auto const info = get_interchange_info(
      uv, sched, ei, mc,
      get_interchange_info_options{
          .include_group_infos_ = true,
          .include_disabled_group_routes_ = include_disabled_group_routes,
          .include_delay_info_ = true});

  mc.create_and_finish(
      MsgContent_PaxMonInterchangeDetailsResponse,
      CreatePaxMonInterchangeDetailsResponse(
          mc, info.to_fbs_interchange_info(mc, uv, sched, true),
          info.normal_routes_, info.broken_routes_)
          .Union());
  return make_msg(mc);
}

}  // namespace motis::paxmon::api
