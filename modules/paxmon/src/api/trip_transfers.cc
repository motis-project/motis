#include "motis/paxmon/api/trip_transfers.h"

#include <vector>

#include "utl/pipes.h"

#include "motis/core/conv/station_conv.h"
#include "motis/core/conv/trip_conv.h"

#include "motis/paxmon/get_universe.h"
#include "motis/paxmon/messages.h"
#include "motis/paxmon/util/detailed_transfer_info.h"

using namespace motis::module;
using namespace motis::paxmon;
using namespace motis::paxmon::util;
using namespace flatbuffers;

namespace motis::paxmon::api {

msg_ptr trip_transfers(paxmon_data& data, msg_ptr const& msg) {
  auto const req = motis_content(PaxMonTripTransfersRequest, msg);
  auto const uv_access = get_universe_and_schedule(data, req->universe());
  auto const& sched = uv_access.sched_;
  auto const& uv = uv_access.uv_;

  auto const trp = from_fbs(sched, req->trip());
  auto const tdi = uv.trip_data_.get_index(trp->trip_idx_);

  auto const gdti_options = get_detailed_transfer_info_options{
      .include_group_infos_ = true,
      .include_disabled_group_routes_ = true,
      .include_delay_info_ = req->include_delay_info()};

  message_creator mc;

  auto incoming_transfers = std::vector<Offset<PaxMonTripTransfersAtStop>>{};
  auto outgoing_transfers = std::vector<Offset<PaxMonTripTransfersAtStop>>{};

  for (auto const& ei : uv.trip_data_.edges(tdi)) {
    auto const* e = ei.get(uv);
    auto const* from = e->from(uv);
    auto const* to = e->to(uv);

    incoming_transfers.emplace_back(CreatePaxMonTripTransfersAtStop(
        mc, to_fbs(mc, from->get_station(sched)),
        mc.CreateVector(utl::all(from->incoming_edges(uv))  //
                        | utl::remove_if([&](auto const& ie) {
                            return !ie.is_interchange();
                          })  //
                        | utl::transform([&](auto const& ie) {
                            return get_detailed_transfer_info(uv, sched, &ie,
                                                              mc, gdti_options)
                                .to_fbs_transfer_info(
                                    mc, uv, sched,
                                    gdti_options.include_delay_info_);
                          })  //
                        | utl::vec())));

    outgoing_transfers.emplace_back(CreatePaxMonTripTransfersAtStop(
        mc, to_fbs(mc, to->get_station(sched)),
        mc.CreateVector(utl::all(to->outgoing_edges(uv))  //
                        | utl::remove_if([&](auto const& oe) {
                            return !oe.is_interchange();
                          })  //
                        | utl::transform([&](auto const& oe) {
                            return get_detailed_transfer_info(uv, sched, &oe,
                                                              mc, gdti_options)
                                .to_fbs_transfer_info(
                                    mc, uv, sched,
                                    gdti_options.include_delay_info_);
                          })  //
                        | utl::vec())));
  }

  mc.create_and_finish(
      MsgContent_PaxMonTripTransfersResponse,
      CreatePaxMonTripTransfersResponse(mc, mc.CreateVector(incoming_transfers),
                                        mc.CreateVector(outgoing_transfers))
          .Union());
  return make_msg(mc);
}

}  // namespace motis::paxmon::api
