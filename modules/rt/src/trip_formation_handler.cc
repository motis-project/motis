#include "motis/rt/trip_formation_handler.h"

#include "motis/rt/util.h"

namespace motis::rt {

struct find_trip_result {
  trip* trip_{};
  int matching_trips_{};
};

find_trip_result find_trip_by_primary_trip_id(schedule const& sched,
                                              primary_trip_id const& ptid) {
  auto result = find_trip_result{};
  for (auto it =
           std::lower_bound(begin(sched.trips_), end(sched.trips_),
                            std::make_pair(ptid, static_cast<trip*>(nullptr)));
       it != end(sched.trips_) && it->first == ptid; ++it) {
    result.trip_ = it->second;
    ++result.matching_trips_;
  }
  return result;
}

void handle_trip_formation_msg(statistics& stats, schedule& sched,
                               update_msg_builder& update_builder,
                               ris::TripFormationMessage const* msg,
                               rt_metrics& metrics, unixtime msg_timestamp,
                               unixtime processing_time) {
  ++stats.trip_formation_msgs_;
  update_builder.trip_formation_message(msg);

  auto const update_metrics = [&](auto const& fn) {
    metrics.update(msg_timestamp, processing_time, fn);
  };

  switch (msg->message_type()) {
    case ris::TripFormationMessageType_Schedule:
      update_metrics(
          [](metrics_entry* m) { ++m->formation_schedule_messages_; });
      break;
    case ris::TripFormationMessageType_Preview:
      update_metrics(
          [](metrics_entry* m) { ++m->formation_preview_messages_; });
      break;
    case ris::TripFormationMessageType_Is:
      update_metrics([](metrics_entry* m) { ++m->formation_is_messages_; });
      break;
  }
  primary_trip_id ptid;
  if (get_primary_trip_id(sched, msg->trip_id(), ptid)) {
    auto const find_result = find_trip_by_primary_trip_id(sched, ptid);
    if (find_result.matching_trips_ == 0) {
      update_metrics(
          [](metrics_entry* m) { ++m->formation_primary_trip_id_not_found_; });
    } else if (find_result.matching_trips_ > 1) {
      update_metrics(
          [](metrics_entry* m) { ++m->formation_primary_trip_id_ambiguous_; });
    }
  } else {
    update_metrics(
        [](metrics_entry* m) { ++m->formation_invalid_primary_trip_id_; });
  }
}

}  // namespace motis::rt
