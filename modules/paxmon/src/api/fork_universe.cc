#include "motis/paxmon/api/fork_universe.h"

#include "motis/core/common/logging.h"
#include "motis/module/context/motis_publish.h"

#include "motis/paxmon/error.h"
#include "motis/paxmon/get_universe.h"

using namespace motis::module;
using namespace motis::paxmon;
using namespace motis::logging;

namespace motis::paxmon::api {

msg_ptr fork_universe(paxmon_data& data, msg_ptr const& msg) {
  auto const broadcast = [&](universe const& base, universe const& fork) {
    message_creator mc;
    mc.create_and_finish(
        MsgContent_PaxMonUniverseForked,
        CreatePaxMonUniverseForked(mc, base.id_, fork.id_).Union(),
        "/paxmon/universe_forked");
    auto const msg = make_msg(mc);
    motis_publish(msg);
  };

  auto const fork = [&](universe const& base) -> msg_ptr {
    scoped_timer timer{"paxmon: fork universe"};
    auto new_uv = data.multiverse_.fork(base.id_);
    broadcast(base, *new_uv);
    message_creator mc;
    mc.create_and_finish(
        MsgContent_PaxMonForkUniverseResponse,
        CreatePaxMonForkUniverseResponse(mc, new_uv->id_).Union());
    return make_msg(mc);
  };

  switch (msg->get()->content_type()) {
    case MsgContent_PaxMonForkUniverseRequest: {
      auto const req = motis_content(PaxMonForkUniverseRequest, msg);
      return fork(get_universe(data, req->universe()));
    }
    case MsgContent_MotisNoMessage: return fork(get_primary_universe(data));
    default:
      throw std::system_error{motis::module::error::unexpected_message_type};
  }
}

}  // namespace motis::paxmon::api
