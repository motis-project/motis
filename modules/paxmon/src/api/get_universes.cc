#include "motis/paxmon/api/get_universes.h"

#include "utl/to_vec.h"

#include "motis/paxmon/get_universe.h"

using namespace motis::module;
using namespace motis::paxmon;

namespace motis::paxmon::api {

msg_ptr get_universes(paxmon_data& data,
                      motis::module::msg_ptr const& /*msg*/) {
  auto const uv_infos = data.multiverse_->get_current_universe_infos();

  message_creator mc;
  mc.create_and_finish(
      MsgContent_PaxMonGetUniversesResponse,
      CreatePaxMonGetUniversesResponse(
          mc, data.multiverse_->id(),
          mc.CreateVector(utl::to_vec(
              uv_infos,
              [&](current_universe_info const& uvi) {
                auto const ttl = uvi.ttl_ ? uvi.ttl_->count() : 0;
                auto const expires_in =
                    uvi.expires_in_ ? uvi.expires_in_->count() : 0;
                return CreatePaxMonUniverseInfo(
                    mc, uvi.uv_id_, uvi.schedule_res_, ttl, expires_in);
              })))
          .Union());
  return make_msg(mc);
}

}  // namespace motis::paxmon::api
