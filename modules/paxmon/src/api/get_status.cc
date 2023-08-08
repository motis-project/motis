#include "motis/paxmon/api/get_status.h"

#include "motis/core/common/date_time_util.h"
#include "motis/module/context/motis_call.h"

#include "motis/paxmon/get_universe.h"

using namespace motis::module;
using namespace motis::paxmon;
using namespace motis::ris;

namespace motis::paxmon::api {

msg_ptr get_status(paxmon_data& data, motis::module::msg_ptr const& msg) {
  auto const req = motis_content(PaxMonStatusRequest, msg);
  auto const uv_access = get_universe_and_schedule(data, req->universe());
  auto const& sched = uv_access.sched_;
  auto const& uv = uv_access.uv_;

  auto const ris_status_msg = motis_call(make_no_msg("/ris/status"))->val();
  auto const ris_status = motis_content(RISStatusResponse, ris_status_msg);
  auto const current_time = now();

  message_creator mc;

  auto const to_feed_status = [&](RISSourceStatus const* status) {
    auto const receiving = status->last_update_time() != 0 &&
                           (current_time - status->last_update_time()) <
                               (status->update_interval() * 2);
    auto const up_to_date = status->last_message_time() != 0 &&
                            (current_time - status->last_message_time()) <
                                (status->update_interval() * 2);
    return CreatePaxMonFeedStatus(mc, status->enabled(), receiving, up_to_date,
                                  status->last_update_time(),
                                  status->last_message_time());
  };

  mc.create_and_finish(
      MsgContent_PaxMonStatusResponse,
      CreatePaxMonStatusResponse(
          mc, static_cast<std::uint64_t>(sched.system_time_),
          data.multiverse_->id(), uv.passenger_groups_.active_groups(),
          uv.trip_data_.size(), ris_status->system_time(), current_time,
          to_feed_status(ris_status->ribasis_fahrt_status()),
          to_feed_status(ris_status->ribasis_formation_status()))
          .Union());
  return make_msg(mc);
}

}  // namespace motis::paxmon::api
