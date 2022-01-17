#include "motis/paxmon/api/get_status.h"

#include <cstdint>

#include "motis/paxmon/get_universe.h"

using namespace motis::module;
using namespace motis::paxmon;

namespace motis::paxmon::api {

msg_ptr get_status(paxmon_data& data, motis::module::msg_ptr const& msg) {
  auto const req = motis_content(PaxMonStatusRequest, msg);
  auto const uv_access = get_universe_and_schedule(data, req->universe());
  auto const& sched = uv_access.sched_;
  auto const& uv = uv_access.uv_;

  message_creator mc;
  mc.create_and_finish(
      MsgContent_PaxMonStatusResponse,
      CreatePaxMonStatusResponse(
          mc, static_cast<std::uint64_t>(sched.system_time_),
          uv.passenger_groups_.active_groups(), uv.trip_data_.size())
          .Union());
  return make_msg(mc);
}

}  // namespace motis::paxmon::api
