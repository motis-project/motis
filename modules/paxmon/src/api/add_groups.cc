#include "motis/paxmon/api/add_groups.h"

#include <algorithm>

#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/paxmon/access/groups.h"
#include "motis/paxmon/get_universe.h"
#include "motis/paxmon/messages.h"
#include "motis/paxmon/print_stats.h"

using namespace motis::module;
using namespace motis::paxmon;

namespace motis::paxmon::api {

msg_ptr add_groups(paxmon_data& data, msg_ptr const& msg) {
  auto const req = motis_content(PaxMonAddGroupsRequest, msg);
  auto const uv_access =
      get_universe_and_schedule(data, req->universe(), ctx::access_t::WRITE);
  auto const& sched = uv_access.sched_;
  auto& uv = uv_access.uv_;

  auto const added_groups = utl::to_vec(*req->groups(), [&](PaxMonGroup const*
                                                                pg_fbs) {
    utl::verify(
        pg_fbs->routes()->size() != 0,
        "paxmon::add_groups: trying to add a passenger group with no routes");
    auto input_pg = from_fbs(sched, pg_fbs);
    utl::verify(std::all_of(begin(input_pg.routes_), end(input_pg.routes_),
                            [](auto const& route) {
                              return !route.journey_.legs().empty();
                            }),
                "paxmon::add_groups: trying to add a passenger group with an "
                "empty route");
    return add_passenger_group(uv, sched, data.capacity_maps_, input_pg, true,
                               pci_log_reason_t::API);
  });

  print_allocator_stats(uv);

  message_creator mc;
  mc.create_and_finish(
      MsgContent_PaxMonAddGroupsResponse,
      CreatePaxMonAddGroupsResponse(
          mc, mc.CreateVector(utl::to_vec(
                  added_groups, [](auto const pg) { return pg->id_; })))
          .Union());
  return make_msg(mc);
}

}  // namespace motis::paxmon::api
