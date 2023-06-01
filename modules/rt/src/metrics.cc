#include "motis/rt/metrics.h"

namespace motis::rt {

void count_message(rt_metrics& metrics, motis::ris::RISMessage const* msg,
                   unixtime const processing_time) {
  auto const count = [&](auto const& fn) {
    metrics.update(msg->timestamp(), processing_time, [&](metrics_entry* m) {
      ++m->messages_;
      fn(m);
    });
  };

  switch (msg->content_type()) {
    case ris::RISMessageUnion_DelayMessage:
      count([](metrics_entry* m) { ++m->delay_messages_; });
      break;
    case ris::RISMessageUnion_CancelMessage:
      count([](metrics_entry* m) { ++m->cancel_messages_; });
      break;
    case ris::RISMessageUnion_AdditionMessage:
      count([](metrics_entry* m) { ++m->additional_messages_; });
      break;
    case ris::RISMessageUnion_RerouteMessage:
      count([](metrics_entry* m) { ++m->reroute_messages_; });
      break;
    case ris::RISMessageUnion_TrackMessage:
      count([](metrics_entry* m) { ++m->track_messages_; });
      break;
    case ris::RISMessageUnion_FullTripMessage:
      count([](metrics_entry* m) { ++m->full_trip_messages_; });
      break;
    case ris::RISMessageUnion_TripFormationMessage:
      count([](metrics_entry* m) { ++m->trip_formation_messages_; });
      break;
    default: break;
  }
}

}  // namespace motis::rt
