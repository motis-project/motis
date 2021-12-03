#include "motis/paxmon/api/get_status.h"

#include <cstdint>

using namespace motis::module;
using namespace motis::paxmon;

namespace motis::paxmon::api {

msg_ptr get_status(schedule const& sched,
                   tick_statistics const& last_tick_stats) {
  message_creator mc;
  mc.create_and_finish(
      MsgContent_PaxMonStatusResponse,
      CreatePaxMonStatusResponse(
          mc, static_cast<std::uint64_t>(sched.system_time_),
          last_tick_stats.tracked_ok_groups_ +
              last_tick_stats.tracked_broken_groups_,
          last_tick_stats.affected_groups_,
          last_tick_stats.affected_passengers_, last_tick_stats.broken_groups_,
          last_tick_stats.broken_passengers_)
          .Union());
  return make_msg(mc);
}

}  // namespace motis::paxmon::api
