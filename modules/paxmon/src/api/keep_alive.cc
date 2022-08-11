#include "motis/paxmon/api/keep_alive.h"

#include "utl/to_vec.h"

#include "motis/paxmon/get_universe.h"

using namespace motis::module;
using namespace motis::paxmon;

namespace motis::paxmon::api {

msg_ptr keep_alive(paxmon_data& data, msg_ptr const& msg) {
  auto const req = motis_content(PaxMonKeepAliveRequest, msg);
  auto const multiverse_id = req->multiverse_id();

  if (multiverse_id == 0 || multiverse_id == data.multiverse_->id()) {
    auto const res =
        data.multiverse_->keep_alive(utl::to_vec(*req->universes()));

    message_creator mc;
    mc.create_and_finish(
        MsgContent_PaxMonKeepAliveResponse,
        CreatePaxMonKeepAliveResponse(
            mc, data.multiverse_->id(),
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
  } else {
    message_creator mc;
    mc.create_and_finish(
        MsgContent_PaxMonKeepAliveResponse,
        CreatePaxMonKeepAliveResponse(
            mc, data.multiverse_->id(),
            mc.CreateVector(
                std::vector<
                    flatbuffers::Offset<PaxMonUniverseKeepAliveInfo>>{}),
            mc.CreateVector(utl::to_vec(*req->universes())))
            .Union());
    return make_msg(mc);
  }
}

}  // namespace motis::paxmon::api
