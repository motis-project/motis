#include "motis/paxmon/api/remove_groups.h"

#include "utl/verify.h"

#include "motis/core/common/logging.h"
#include "motis/core/access/trip_access.h"

#include "motis/paxmon/build_graph.h"
#include "motis/paxmon/checks.h"
#include "motis/paxmon/get_universe.h"
#include "motis/paxmon/messages.h"
#include "motis/paxmon/print_stats.h"

using namespace motis::module;
using namespace motis::paxmon;
using namespace motis::logging;

namespace motis::paxmon::api {

msg_ptr remove_groups(paxmon_data& data, bool const keep_group_history,
                      bool const check_graph_integrity_end,
                      msg_ptr const& msg) {
  auto const req = motis_content(PaxMonRemoveGroupsRequest, msg);
  auto const uv_access =
      get_universe_and_schedule(data, req->universe(), ctx::access_t::WRITE);
  auto const& sched = uv_access.sched_;
  auto& uv = uv_access.uv_;
  auto removed_groups = 0ULL;

  for (auto const id : *req->ids()) {
    auto pg = uv.passenger_groups_.at(id);
    if (pg == nullptr) {
      continue;
    }
    ++removed_groups;
    uv.update_tracker_.before_group_removed(pg);
    remove_passenger_group_from_graph(uv, pg);
    if (!keep_group_history) {
      uv.passenger_groups_.release(pg->id_);
    }
  }

  LOG(info) << "remove_groups: " << removed_groups << " removed (universe "
            << uv.id_ << ")";

  print_allocator_stats(uv);

  if (check_graph_integrity_end) {
    utl::verify(check_graph_integrity(uv, sched), "remove_groups (end)");
  }

  return {};
}

}  // namespace motis::paxmon::api
