#include "motis/paxmon/api/get_status.h"

#include <vector>

#include "utl/verify.h"

#include "motis/paxmon/get_universe.h"
#include "motis/paxmon/localization.h"
#include "motis/paxmon/messages.h"
#include "motis/paxmon/reachability.h"

using namespace motis::module;
using namespace motis::paxmon;

namespace motis::paxmon::api {

msg_ptr get_groups(paxmon_data& data, msg_ptr const& msg) {
  auto const req = motis_content(PaxMonGetGroupsRequest, msg);
  auto const uv_access = get_universe_and_schedule(data, req->universe());
  auto const& sched = uv_access.sched_;
  auto& uv = uv_access.uv_;

  message_creator mc;
  std::vector<flatbuffers::Offset<PaxMonGroup>> groups;

  auto const add_by_data_source = [&](data_source const& ds) {
    if (auto const it = uv.passenger_groups_.groups_by_source_.find(ds);
        it != end(uv.passenger_groups_.groups_by_source_)) {
      for (auto const pgid : it->second) {
        if (auto const pg = uv.passenger_groups_.at(pgid); pg != nullptr) {
          groups.emplace_back(to_fbs(sched, uv.passenger_groups_, mc, *pg));
        }
      }
    }
  };

  for (auto const pgid : *req->ids()) {
    if (auto const pg = uv.passenger_groups_.at(pgid); pg != nullptr) {
      groups.emplace_back(to_fbs(sched, uv.passenger_groups_, mc, *pg));
    }
  }

  for (auto const ds : *req->sources()) {
    add_by_data_source(from_fbs(ds));
  }

  mc.create_and_finish(
      MsgContent_PaxMonGetGroupsResponse,
      CreatePaxMonGetGroupsResponse(mc, mc.CreateVector(groups)).Union());
  return make_msg(mc);
}

}  // namespace motis::paxmon::api
