#include "motis/rt/shifted_nodes_msg_builder.h"

#include "motis/core/access/edge_access.h"
#include "motis/core/access/time_access.h"
#include "motis/core/conv/event_type_conv.h"
#include "motis/core/conv/timestamp_reason_conv.h"
#include "motis/core/conv/trip_conv.h"

using namespace motis::module;
using namespace flatbuffers;

namespace motis::rt {

shifted_nodes_msg_builder::shifted_nodes_msg_builder(message_creator& fbb,
                                                     schedule const& sched)
    : fbb_(fbb), sched_(sched) {}

void shifted_nodes_msg_builder::add(delay_info const* di) {
  delays_.insert(di);
}

Offset<RtUpdate> shifted_nodes_msg_builder::build_shifted_node(
    delay_info const* di) {
  auto const& k = di->get_ev_key();
  auto const trp =
      sched_.merged_trips_[get_lcon(k.route_edge_, k.lcon_idx_).trips_]->at(0);
  auto const event_info = CreateRtEventInfo(
      fbb_, to_fbs(sched_, fbb_, trp),
      fbb_.CreateString(sched_.stations_.at(k.get_station_idx())->eva_nr_),
      motis_to_unixtime(sched_, di->get_schedule_time()), to_fbs(k.ev_type_));

  if (k.lcon()->valid_ != 0U) {
    return CreateRtUpdate(
        fbb_, Content_RtDelayUpdate,
        CreateRtDelayUpdate(fbb_, event_info,
                            motis_to_unixtime(sched_, di->get_current_time()),
                            to_fbs(di->get_reason()))
            .Union());
  } else {
    return CreateRtUpdate(fbb_, Content_RtCancelUpdate,
                          CreateRtCancelUpdate(fbb_, event_info, true).Union());
  }
}

void shifted_nodes_msg_builder::finish(
    std::vector<flatbuffers::Offset<RtUpdate>>& updates) {
  for (auto const& di : delays_) {
    updates.push_back(build_shifted_node(di));
  }
}

bool shifted_nodes_msg_builder::empty() const { return delays_.empty(); }

std::size_t shifted_nodes_msg_builder::size() const { return delays_.size(); }

}  // namespace motis::rt
