#include "motis/rt/trip_formation_handler.h"

#include "motis/rt/util.h"

namespace motis::rt {

trip* find_trip_by_primary_trip_id(schedule const& sched,
                                   primary_trip_id const& ptid) {
  trip* result = nullptr;
  auto matching_trips = 0;
  for (auto it =
           std::lower_bound(begin(sched.trips_), end(sched.trips_),
                            std::make_pair(ptid, static_cast<trip*>(nullptr)));
       it != end(sched.trips_) && it->first == ptid; ++it) {
    result = it->second;
    ++matching_trips;
  }
  // only return trip if there is an unambiguous match
  return matching_trips == 1 ? result : nullptr;
}

void handle_trip_formation_msg(statistics& stats, schedule& sched,
                               update_msg_builder& update_builder,
                               ris::TripFormationMessage const* msg) {
  ++stats.trip_formation_msgs_;
  update_builder.trip_formation_message(msg);

  // store trip uuid mapping
  primary_trip_id ptid;
  if (get_primary_trip_id(sched, msg->trip_id(), ptid)) {
    auto const trip_uuid = parse_uuid(view(msg->trip_id()->uuid()));
    if (sched.uuid_to_trip_.find(trip_uuid) != end(sched.uuid_to_trip_)) {
      return;
    }
    if (auto* trp = find_trip_by_primary_trip_id(sched, ptid); trp != nullptr) {
      sched.uuid_to_trip_[trip_uuid] = trp;
      if (trp->uuid_.is_nil()) {
        trp->uuid_ = trip_uuid;
      }
    }
  }
}

}  // namespace motis::rt
