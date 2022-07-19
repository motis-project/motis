#include "motis/paxmon/api/keep_alive.h"

#include "utl/to_vec.h"

#include "motis/paxmon/get_universe.h"

using namespace motis::module;
using namespace motis::paxmon;

namespace motis::paxmon::api {

msg_ptr keep_alive(paxmon_data& data, msg_ptr const& msg) {
  auto const req = motis_content(PaxMonKeepAliveRequest, msg);

  auto const res = data.multiverse_.keep_alive(utl::to_vec(*req->universes()));

  message_creator mc;
  mc.create_and_finish(
      MsgContent_PaxMonKeepAliveResponse,
      CreatePaxMonKeepAliveResponse(
          mc,
          mc.CreateVector(utl::to_vec(
              res.found_,
              [&](auto const& kaui) {
                return CreatePaxMonUniverseKeepAliveInfo(
                    mc, kaui.id_, kaui.schedule_res_id_,
                    kaui.expires_in_ ? kaui.expires_in_.value().count() : 0);
              })),
          mc.CreateVector(res.not_found_))
          .Union());
  return make_msg(mc);
}

}  // namespace motis::paxmon::api
