#include "motis/paxmon/api/destroy_universe.h"

#include "motis/module/context/motis_publish.h"

#include "motis/paxmon/error.h"
#include "motis/paxmon/get_universe.h"

using namespace motis::module;
using namespace motis::paxmon;

namespace motis::paxmon::api {

msg_ptr destroy_universe(schedule const& sched, paxmon_data& data,
                         msg_ptr const& msg) {
  auto const req = motis_content(PaxMonDestroyUniverseRequest, msg);
  auto& uv = get_universe(data, req->universe());
  if (data.multiverse_.destroy(uv.id_)) {
    message_creator mc;
    mc.create_and_finish(
        MsgContent_PaxMonUniverseDestroyed,
        CreatePaxMonUniverseDestroyed(mc, req->universe()).Union(),
        "/paxmon/universe_destroyed");
    auto const msg = make_msg(mc);
    motis_publish(msg);
    return make_success_msg();
  } else {
    throw std::system_error{error::universe_destruction_failed};
  }
}

}  // namespace motis::paxmon::api
