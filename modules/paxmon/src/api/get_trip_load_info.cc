#include "motis/paxmon/api/get_trip_load_info.h"

#include "utl/to_vec.h"

#include "motis/core/conv/trip_conv.h"

#include "motis/paxmon/get_universe.h"
#include "motis/paxmon/load_info.h"
#include "motis/paxmon/messages.h"

using namespace motis::module;
using namespace motis::paxmon;

namespace motis::paxmon::api {

msg_ptr get_trip_load_info(paxmon_data& data, msg_ptr const& msg) {
  auto const req = motis_content(PaxMonGetTripLoadInfosRequest, msg);
  auto const uv_access = get_universe_and_schedule(data, req->universe());
  auto const& sched = uv_access.sched_;
  auto& uv = uv_access.uv_;

  message_creator mc;

  auto const to_fbs_load_info_for_universe = [&](universe const& uv) {
    return [&](TripId const* fbs_tid) {
      auto const trp = from_fbs(sched, fbs_tid);
      auto const tli = calc_trip_load_info(uv, trp);
      return to_fbs(mc, sched, uv, tli);
    };
  };

  mc.create_and_finish(
      MsgContent_PaxMonGetTripLoadInfosResponse,
      CreatePaxMonGetTripLoadInfosResponse(
          mc, mc.CreateVector(utl::to_vec(*req->trips(),
                                          to_fbs_load_info_for_universe(uv))))

          .Union());
  return make_msg(mc);
}

}  // namespace motis::paxmon::api
