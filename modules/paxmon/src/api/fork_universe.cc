#include "motis/paxmon/api/fork_universe.h"

#include "motis/core/common/logging.h"
#include "motis/module/context/motis_publish.h"

#include "motis/paxmon/get_universe.h"

using namespace motis::module;
using namespace motis::paxmon;
using namespace motis::logging;

namespace motis::paxmon::api {

msg_ptr fork_universe(paxmon_data& data, msg_ptr const& msg) {
  auto const req = motis_content(PaxMonForkUniverseRequest, msg);
  auto const uv_access = get_universe_and_schedule(data, req->universe());
  auto const& base_uv = uv_access.uv_;
  auto const& base_sched = uv_access.sched_;
  auto const fork_schedule = req->fork_schedule();
  scoped_timer timer{"paxmon: fork universe"};

  auto ttl = std::chrono::seconds{req->ttl()};
  if ((ttl.count() == 0 && !data.allow_infinite_universe_ttl_) ||
      (ttl > data.max_universe_ttl_)) {
    ttl = data.max_universe_ttl_;
  }

  auto* new_uv =
      data.multiverse_->fork(base_uv, base_sched, fork_schedule, ttl);

  message_creator mc;
  mc.create_and_finish(
      MsgContent_PaxMonForkUniverseResponse,
      CreatePaxMonForkUniverseResponse(mc, new_uv->id_,
                                       new_uv->schedule_res_id_, ttl.count())
          .Union());
  return make_msg(mc);
}

}  // namespace motis::paxmon::api
